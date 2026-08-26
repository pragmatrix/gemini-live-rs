/// Mono PCM chunk captured from a desktop audio source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedAudio {
    /// i16 little-endian PCM payload suitable for `send_audio_at_rate`.
    pub pcm_i16_le: Vec<u8>,
    /// Sample rate of `pcm_i16_le`.
    pub sample_rate: u32,
}
