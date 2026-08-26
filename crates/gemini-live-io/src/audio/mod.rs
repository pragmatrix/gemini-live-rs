//! Desktop audio adapters backed by `cpal` and WebRTC AEC.
//!
//! The microphone and speaker adapters share an [`AecHandle`] so the echo
//! canceller can subtract speaker output from microphone capture before hosts
//! forward audio to the Live API. The system-audio adapter captures the
//! default output mix through backend loopback support when the platform makes
//! it available.

#[cfg(feature = "aec")]
mod aec;
mod captured;
#[cfg(feature = "mic")]
mod mic;
#[cfg(any(feature = "mic", feature = "system-audio"))]
mod pcm;
mod resample;
#[cfg(feature = "speaker")]
mod speaker;
#[cfg(feature = "system-audio")]
mod system;

#[cfg(feature = "aec")]
pub use aec::{AEC_FRAME_SIZE, AEC_SAMPLE_RATE, AecHandle};
#[cfg(any(feature = "mic", feature = "system-audio"))]
pub use captured::CapturedAudio;
#[cfg(feature = "mic")]
pub use mic::{MicCapture, MicCaptureConfig};
#[cfg(feature = "speaker")]
pub use speaker::{MODEL_AUDIO_SAMPLE_RATE, SpeakerPlayback};
#[cfg(feature = "system-audio")]
pub use system::SystemAudioCapture;
