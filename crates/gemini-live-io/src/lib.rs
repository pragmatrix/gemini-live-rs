//! Desktop media adapters for Gemini Live hosts.
//!
//! This crate is the canonical home for reusable desktop-side microphone,
//! speaker, system-audio loopback, and screen-capture adapters. It
//! intentionally does not know about TUI concerns, slash commands, profile
//! persistence, or Gemini session orchestration. Host applications wire these
//! adapters into their own runtime and product surface.

pub mod error;

#[cfg(any(
    feature = "aec",
    feature = "mic",
    feature = "speaker",
    feature = "system-audio"
))]
pub mod audio;

#[cfg(feature = "screen")]
pub mod screen;
