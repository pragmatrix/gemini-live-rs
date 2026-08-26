use std::collections::VecDeque;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use tokio::sync::mpsc;

use crate::error::AudioIoError;

use super::CapturedAudio;
use super::aec::{AEC_FRAME_SIZE, AEC_SAMPLE_RATE, AecHandle};
use super::pcm::{decode_i16_to_f32_into, drain_chunks, encode_f32_to_pcm_i16le};
use super::resample::linear_resample_into;

/// Output shaping for [`MicCapture`].
///
/// AEC always runs at [`AEC_SAMPLE_RATE`] internally (echo cancellation needs
/// the same clock as the speaker render path). When `output_sample_rate`
/// differs, the cleaned audio is resampled after AEC and accumulated into
/// `chunk_samples`-sized chunks before emission.
#[derive(Debug, Clone, Copy)]
pub struct MicCaptureConfig {
    /// Sample rate of the emitted [`CapturedAudio`] chunks.
    pub output_sample_rate: u32,
    /// Samples per emitted chunk at `output_sample_rate`.
    pub chunk_samples: usize,
}

impl Default for MicCaptureConfig {
    /// Passthrough: one 10 ms frame per chunk at the AEC rate (48 kHz).
    fn default() -> Self {
        Self {
            output_sample_rate: AEC_SAMPLE_RATE,
            chunk_samples: AEC_FRAME_SIZE,
        }
    }
}

impl MicCaptureConfig {
    /// 16 kHz mono in 100 ms chunks — the shape required by Live Transcribe
    /// input audio, and identical to `SystemAudioCapture` output.
    pub const fn transcribe() -> Self {
        Self {
            output_sample_rate: 16_000,
            chunk_samples: 1_600,
        }
    }

    fn is_aec_passthrough(&self) -> bool {
        self.output_sample_rate == AEC_SAMPLE_RATE && self.chunk_samples == AEC_FRAME_SIZE
    }
}

/// Active capture stream for the default input device.
pub struct MicCapture {
    _stream: cpal::Stream,
    /// Native sample rate reported by the capture device.
    pub input_sample_rate: u32,
    /// Sample rate of emitted chunks after AEC processing (and optional
    /// output resampling).
    pub output_sample_rate: u32,
}

impl std::fmt::Debug for MicCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicCapture")
            .field("input_sample_rate", &self.input_sample_rate)
            .field("output_sample_rate", &self.output_sample_rate)
            .finish()
    }
}

/// Reusable scratch buffers for the capture callback hot path.
#[derive(Default)]
struct MicBuffers {
    mono: Vec<f32>,
    resampled: Vec<f32>,
    aec_frame: Vec<f32>,
    processed: Vec<f32>,
    output_resampled: Vec<f32>,
    pending: VecDeque<f32>,
    chunk: Vec<f32>,
}

#[derive(Default)]
struct MicCallbackState {
    input_f32: Vec<f32>,
    buffers: MicBuffers,
}

struct MicCallbackContext<'a> {
    channels: usize,
    input_sample_rate: u32,
    config: MicCaptureConfig,
    aec: &'a AecHandle,
    tx: &'a mpsc::Sender<CapturedAudio>,
}

impl MicCallbackState {
    fn process_f32(&mut self, data: &[f32], ctx: MicCallbackContext<'_>) {
        process_mic_samples(data, ctx, &mut self.buffers);
    }

    fn process_i16(&mut self, data: &[i16], ctx: MicCallbackContext<'_>) {
        decode_i16_to_f32_into(&mut self.input_f32, data);
        process_mic_samples(&self.input_f32, ctx, &mut self.buffers);
    }
}

fn process_mic_samples(data: &[f32], ctx: MicCallbackContext<'_>, buffers: &mut MicBuffers) {
    let MicCallbackContext {
        channels,
        input_sample_rate,
        config,
        aec,
        tx,
    } = ctx;
    let MicBuffers {
        mono,
        resampled,
        aec_frame,
        processed,
        output_resampled,
        pending,
        chunk,
    } = buffers;

    mono.resize(data.len() / channels, 0.0);
    for (slot, frame) in mono.iter_mut().zip(data.chunks_exact(channels)) {
        let mut sum = 0.0;
        for &sample in frame {
            sum += sample;
        }
        *slot = sum / channels as f32;
    }

    let aec_input = if input_sample_rate == AEC_SAMPLE_RATE {
        mono.as_slice()
    } else {
        linear_resample_into(resampled, mono, input_sample_rate, AEC_SAMPLE_RATE);
        resampled.as_slice()
    };

    let passthrough = config.is_aec_passthrough();
    aec_frame.resize(AEC_FRAME_SIZE, 0.0);
    processed.clear();
    for frame in aec_input.chunks(AEC_FRAME_SIZE) {
        if frame.len() < AEC_FRAME_SIZE {
            break;
        }

        aec_frame.copy_from_slice(frame);
        if aec
            .processor()
            .process_capture_frame(&mut [&mut aec_frame[..]])
            .is_err()
        {
            continue;
        }

        if passthrough {
            tx.try_send(CapturedAudio {
                pcm_i16_le: encode_f32_to_pcm_i16le(aec_frame),
                sample_rate: AEC_SAMPLE_RATE,
            })
            .ok();
        } else {
            processed.extend_from_slice(aec_frame);
        }
    }

    if passthrough || processed.is_empty() {
        return;
    }

    let output = if config.output_sample_rate == AEC_SAMPLE_RATE {
        processed.as_slice()
    } else {
        linear_resample_into(
            output_resampled,
            processed,
            AEC_SAMPLE_RATE,
            config.output_sample_rate,
        );
        output_resampled.as_slice()
    };

    pending.extend(output.iter().copied());
    drain_chunks(pending, chunk, config.chunk_samples, |samples| {
        tx.try_send(CapturedAudio {
            pcm_i16_le: encode_f32_to_pcm_i16le(samples),
            sample_rate: config.output_sample_rate,
        })
        .ok();
    });
}

impl MicCapture {
    /// Start capturing from the default input device and forward cleaned PCM
    /// through `tx` in the default shape (48 kHz, one 10 ms frame per chunk).
    pub fn start(tx: mpsc::Sender<CapturedAudio>, aec: AecHandle) -> Result<Self, AudioIoError> {
        Self::start_with_config(tx, aec, MicCaptureConfig::default())
    }

    /// Start capturing with an explicit output shape.
    pub fn start_with_config(
        tx: mpsc::Sender<CapturedAudio>,
        aec: AecHandle,
        config: MicCaptureConfig,
    ) -> Result<Self, AudioIoError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioIoError::NoInputDevice)?;
        let supported = device
            .default_input_config()
            .map_err(|e| AudioIoError::DefaultInputConfig(e.to_string()))?;
        let input_sample_rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let stream_config: StreamConfig = supported.config();

        let stream = match supported.sample_format() {
            SampleFormat::F32 => {
                let aec = aec.clone();
                let tx = tx.clone();
                let mut callback_state = MicCallbackState::default();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _| {
                        callback_state.process_f32(
                            data,
                            MicCallbackContext {
                                channels,
                                input_sample_rate,
                                config,
                                aec: &aec,
                                tx: &tx,
                            },
                        );
                    },
                    |e| tracing::warn!("mic: {e}"),
                    None,
                )
            }
            SampleFormat::I16 => {
                let aec = aec.clone();
                let tx = tx.clone();
                let mut callback_state = MicCallbackState::default();
                device.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _| {
                        callback_state.process_i16(
                            data,
                            MicCallbackContext {
                                channels,
                                input_sample_rate,
                                config,
                                aec: &aec,
                                tx: &tx,
                            },
                        );
                    },
                    |e| tracing::warn!("mic: {e}"),
                    None,
                )
            }
            format => {
                return Err(AudioIoError::UnsupportedInputFormat(format!("{format:?}")));
            }
        }
        .map_err(|e| AudioIoError::BuildInputStream(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioIoError::StartStream(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            input_sample_rate,
            output_sample_rate: config.output_sample_rate,
        })
    }
}
