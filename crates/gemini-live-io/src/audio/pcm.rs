//! Shared PCM sample-format helpers for the capture adapters.
//!
//! Both the microphone and system-audio paths convert between interleaved
//! i16 wire samples and normalized f32 processing buffers, and both emit
//! fixed-size chunks from a rolling pending queue. Keeping these helpers in
//! one place keeps the two adapters byte-identical in output shape.

use std::collections::VecDeque;

/// Decode i16 samples into normalized f32, reusing `output`.
pub(crate) fn decode_i16_to_f32_into(output: &mut Vec<f32>, data: &[i16]) {
    output.resize(data.len(), 0.0);
    for (slot, &sample) in output.iter_mut().zip(data) {
        *slot = sample as f32 / 32768.0;
    }
}

/// Encode f32 samples into little-endian 16-bit PCM bytes.
pub(crate) fn encode_f32_to_pcm_i16le(samples: &[f32]) -> Vec<u8> {
    let mut pcm_i16_le = Vec::with_capacity(samples.len() * std::mem::size_of::<i16>());
    for &sample in samples {
        let normalized = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        pcm_i16_le.extend_from_slice(&normalized.to_le_bytes());
    }
    pcm_i16_le
}

/// Drain `pending` through `emit` in fixed `chunk_samples`-sized chunks,
/// reusing `chunk` as scratch. Samples short of a full chunk stay queued for
/// the next call, so chunk boundaries are stable across callback buffers.
pub(crate) fn drain_chunks(
    pending: &mut VecDeque<f32>,
    chunk: &mut Vec<f32>,
    chunk_samples: usize,
    mut emit: impl FnMut(&[f32]),
) {
    chunk.resize(chunk_samples, 0.0);
    while pending.len() >= chunk_samples {
        for sample in chunk.iter_mut() {
            *sample = pending
                .pop_front()
                .expect("pending length already validated");
        }
        emit(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_chunks_emits_fixed_chunks_and_keeps_remainder() {
        let mut pending: VecDeque<f32> = (0..10).map(|i| i as f32).collect();
        let mut chunk = Vec::new();
        let mut emitted = Vec::new();

        drain_chunks(&mut pending, &mut chunk, 4, |c| emitted.push(c.to_vec()));

        assert_eq!(emitted.len(), 2);
        assert_eq!(emitted[0], vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(emitted[1], vec![4.0, 5.0, 6.0, 7.0]);
        // The two leftover samples wait for the next callback.
        assert_eq!(pending.len(), 2);

        pending.extend([10.0, 11.0]);
        drain_chunks(&mut pending, &mut chunk, 4, |c| emitted.push(c.to_vec()));
        assert_eq!(emitted.len(), 3);
        assert_eq!(emitted[2], vec![8.0, 9.0, 10.0, 11.0]);
        assert!(pending.is_empty());
    }

    #[test]
    fn drain_chunks_emits_nothing_below_one_chunk() {
        let mut pending: VecDeque<f32> = VecDeque::from(vec![1.0, 2.0]);
        let mut chunk = Vec::new();
        let mut calls = 0;

        drain_chunks(&mut pending, &mut chunk, 3, |_| calls += 1);

        assert_eq!(calls, 0);
        assert_eq!(pending.len(), 2);
    }

    #[test]
    fn encode_f32_clamps_and_round_trips() {
        let bytes = encode_f32_to_pcm_i16le(&[0.0, 1.0, -1.0, 2.0]);
        assert_eq!(bytes.len(), 8);
        let samples: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(samples[0], 0);
        assert_eq!(samples[1], 32767);
        // Out-of-range input clamps instead of wrapping.
        assert_eq!(samples[3], 32767);

        let mut decoded = Vec::new();
        decode_i16_to_f32_into(&mut decoded, &samples);
        assert!((decoded[1] - 1.0).abs() < 1e-3);
    }
}
