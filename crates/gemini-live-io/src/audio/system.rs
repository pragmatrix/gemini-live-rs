use std::collections::VecDeque;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use tokio::sync::mpsc;

use crate::error::AudioIoError;

use super::CapturedAudio;
use super::pcm::{decode_i16_to_f32_into, drain_chunks, encode_f32_to_pcm_i16le};
use super::resample::linear_resample_into;

const SYSTEM_AUDIO_SAMPLE_RATE: u32 = 16_000;
const SYSTEM_AUDIO_CHUNK_SAMPLES: usize = (SYSTEM_AUDIO_SAMPLE_RATE / 10) as usize;

/// Active loopback capture stream for the default output device.
pub struct SystemAudioCapture {
    _stream: cpal::Stream,
    /// Human-readable output device name used for loopback capture.
    pub device_name: String,
    /// Native sample rate reported by the output device.
    pub input_sample_rate: u32,
    /// Output sample rate after downmix + resample.
    pub output_sample_rate: u32,
}

impl std::fmt::Debug for SystemAudioCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemAudioCapture")
            .field("device_name", &self.device_name)
            .field("input_sample_rate", &self.input_sample_rate)
            .field("output_sample_rate", &self.output_sample_rate)
            .finish()
    }
}

/// Reusable scratch buffers for the loopback callback hot path.
#[derive(Default)]
struct SystemAudioBuffers {
    mono: Vec<f32>,
    resampled: Vec<f32>,
    pending: VecDeque<f32>,
    chunk: Vec<f32>,
}

#[derive(Default)]
struct SystemAudioCallbackState {
    input_f32: Vec<f32>,
    buffers: SystemAudioBuffers,
}

impl SystemAudioCallbackState {
    fn process_f32(
        &mut self,
        data: &[f32],
        channels: usize,
        input_sample_rate: u32,
        tx: &mpsc::Sender<CapturedAudio>,
    ) {
        process_system_audio_samples(data, channels, input_sample_rate, tx, &mut self.buffers);
    }

    fn process_i16(
        &mut self,
        data: &[i16],
        channels: usize,
        input_sample_rate: u32,
        tx: &mpsc::Sender<CapturedAudio>,
    ) {
        decode_i16_to_f32_into(&mut self.input_f32, data);
        process_system_audio_samples(
            &self.input_f32,
            channels,
            input_sample_rate,
            tx,
            &mut self.buffers,
        );
    }
}

fn process_system_audio_samples(
    data: &[f32],
    channels: usize,
    input_sample_rate: u32,
    tx: &mpsc::Sender<CapturedAudio>,
    buffers: &mut SystemAudioBuffers,
) {
    let SystemAudioBuffers {
        mono,
        resampled,
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

    let output = if input_sample_rate == SYSTEM_AUDIO_SAMPLE_RATE {
        mono.as_slice()
    } else {
        linear_resample_into(
            resampled,
            mono.as_slice(),
            input_sample_rate,
            SYSTEM_AUDIO_SAMPLE_RATE,
        );
        resampled.as_slice()
    };

    pending.extend(output.iter().copied());
    drain_chunks(pending, chunk, SYSTEM_AUDIO_CHUNK_SAMPLES, |samples| {
        tx.try_send(CapturedAudio {
            pcm_i16_le: encode_f32_to_pcm_i16le(samples),
            sample_rate: SYSTEM_AUDIO_SAMPLE_RATE,
        })
        .ok();
    });
}

impl SystemAudioCapture {
    /// Start loopback capture from the current default output device.
    ///
    /// On backends that support output-device loopback, this records the audio
    /// currently being rendered through the system output path. On unsupported
    /// backends, stream creation fails and the caller receives the backend
    /// error rather than a fake fallback capture path.
    pub fn start(tx: mpsc::Sender<CapturedAudio>) -> Result<Self, AudioIoError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioIoError::NoOutputDevice)?;
        let device_name = device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| "default output".into());
        let supported = device
            .default_output_config()
            .map_err(|e| AudioIoError::DefaultOutputConfig(e.to_string()))?;
        let input_sample_rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        let config: StreamConfig = supported.config();

        let stream = match supported.sample_format() {
            SampleFormat::F32 => {
                let tx = tx.clone();
                let mut callback_state = SystemAudioCallbackState::default();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _| {
                        callback_state.process_f32(data, channels, input_sample_rate, &tx);
                    },
                    |e| tracing::warn!("system audio: {e}"),
                    None,
                )
            }
            SampleFormat::I16 => {
                let tx = tx.clone();
                let mut callback_state = SystemAudioCallbackState::default();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        callback_state.process_i16(data, channels, input_sample_rate, &tx);
                    },
                    |e| tracing::warn!("system audio: {e}"),
                    None,
                )
            }
            format => {
                return Err(AudioIoError::UnsupportedOutputFormat(format!("{format:?}")));
            }
        }
        .map_err(|e| AudioIoError::BuildInputStream(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioIoError::StartStream(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            device_name,
            input_sample_rate,
            output_sample_rate: SYSTEM_AUDIO_SAMPLE_RATE,
        })
    }
}
