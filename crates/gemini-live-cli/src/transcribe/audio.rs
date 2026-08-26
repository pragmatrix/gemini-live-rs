//! Audio-source management and transcript file output for transcribe mode.
//!
//! Both sources emit the same shape — 16 kHz mono i16le `CapturedAudio` in
//! 100 ms chunks — so the event loop treats them uniformly:
//!
//! - **mic**: default input device through the shared AEC pipeline
//!   ([`MicCaptureConfig::transcribe`]).
//! - **system**: default output device loopback ([`SystemAudioCapture`]).
//!   macOS CoreAudio has no native loopback, so this usually requires a
//!   virtual output device (e.g. BlackHole); the backend error surfaces as a
//!   TUI notice and the previous source stays active.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::str::FromStr;

use gemini_live_io::audio::{
    AecHandle, CapturedAudio, MicCapture, MicCaptureConfig, SystemAudioCapture,
};
use tokio::sync::mpsc;

/// Which capture adapter feeds the transcription session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioSourceKind {
    Mic,
    System,
}

impl AudioSourceKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Mic => "mic",
            Self::System => "system",
        }
    }
}

impl fmt::Display for AudioSourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AudioSourceKind {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "mic" => Ok(Self::Mic),
            "system" => Ok(Self::System),
            other => Err(format!(
                "unsupported audio source {other:?}; expected `mic` or `system`"
            )),
        }
    }
}

enum ActiveSource {
    Mic(#[allow(dead_code)] MicCapture),
    System(#[allow(dead_code)] SystemAudioCapture),
}

/// Owns the active capture stream and the channel it feeds.
///
/// Switching sources drops the old `cpal` stream (stopping capture) and
/// starts the new one on the same channel, so in-flight chunks from the old
/// source drain naturally.
pub(crate) struct TranscribeAudio {
    tx: mpsc::Sender<CapturedAudio>,
    rx: mpsc::Receiver<CapturedAudio>,
    aec: AecHandle,
    active: ActiveSource,
    pub(crate) source: AudioSourceKind,
}

impl TranscribeAudio {
    pub(crate) fn start(source: AudioSourceKind) -> Result<Self, Box<dyn std::error::Error>> {
        let (tx, rx) = mpsc::channel::<CapturedAudio>(32);
        let aec = AecHandle::new()?;
        let active = start_source(source, &tx, &aec)?;
        Ok(Self {
            tx,
            rx,
            aec,
            active,
            source,
        })
    }

    /// Switch to `source`, keeping the current one on failure.
    pub(crate) fn switch(&mut self, source: AudioSourceKind) -> Result<(), String> {
        if source == self.source {
            return Ok(());
        }
        let next = start_source(source, &self.tx, &self.aec).map_err(|e| e.to_string())?;
        self.active = next;
        self.source = source;
        Ok(())
    }

    pub(crate) fn description(&self) -> String {
        match &self.active {
            ActiveSource::Mic(mic) => format!(
                "mic ({} Hz → {} Hz)",
                mic.input_sample_rate, mic.output_sample_rate
            ),
            ActiveSource::System(system) => format!(
                "system loopback of {} ({} Hz → {} Hz)",
                system.device_name, system.input_sample_rate, system.output_sample_rate
            ),
        }
    }

    pub(crate) async fn next_captured(&mut self) -> Option<CapturedAudio> {
        self.rx.recv().await
    }
}

fn start_source(
    source: AudioSourceKind,
    tx: &mpsc::Sender<CapturedAudio>,
    aec: &AecHandle,
) -> Result<ActiveSource, Box<dyn std::error::Error>> {
    match source {
        AudioSourceKind::Mic => Ok(ActiveSource::Mic(MicCapture::start_with_config(
            tx.clone(),
            aec.clone(),
            MicCaptureConfig::transcribe(),
        )?)),
        AudioSourceKind::System => Ok(ActiveSource::System(SystemAudioCapture::start(tx.clone())?)),
    }
}

/// Line-oriented appender for `--output`.
///
/// Opens in append mode so repeated runs accumulate, and flushes after every
/// line so `tail -f` observes finalized segments as they land.
pub(crate) struct TranscriptWriter {
    file: File,
}

impl TranscriptWriter {
    pub(crate) fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    pub(crate) fn append_line(&mut self, line: &str) -> std::io::Result<()> {
        self.file.write_all(line.as_bytes())?;
        self.file.write_all(b"\n")?;
        self.file.flush()
    }
}
