//! Prompt text for the recorder host.
//!
//! Keep the observation contract in code so changes to the tool schema and the
//! host-side window lifecycle happen in the same patch.

use chrono::{DateTime, Utc};

use crate::tool::SUBMIT_MINUTE_SUMMARY_TOOL;

pub(crate) fn system_instruction(extra_instruction: Option<&str>) -> String {
    let mut text = String::from(
        "You are a desktop activity observer. The host streams periodic \
         screenshots from one or more desktop monitors and may also stream \
         system-output audio from the same desktop session.\n\
         \n\
         When the host asks for a completed observation-window summary, call \
         `",
    );
    text.push_str(SUBMIT_MINUTE_SUMMARY_TOOL);
    text.push_str(
        "` exactly once.\n\
         Do not answer a summary request in plain text.\n\
         The host may stream screenshots from multiple monitors in close \
         succession; treat them as simultaneous views of the same desktop \
         state.\n\
         Be factual and conservative. If the visible evidence is weak, set \
         `uncertain=true` and lower `confidence`.\n\
         Never invent app names, websites, or activities that are not visible \
         in the observed screenshots.",
    );

    if let Some(extra_instruction) = extra_instruction
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        text.push_str("\n\nAdditional host instruction:\n");
        text.push_str(extra_instruction);
    }

    text
}

pub(crate) fn summary_prompt(
    window_id: u64,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    observed_frame_count: u32,
    observed_audio_chunk_count: u32,
) -> String {
    format!(
        "Observation window {window_id} is complete.\n\
         Start: {}\n\
         End: {}\n\
         Observed frames sent during this window: {observed_frame_count}\n\
         \n\
         Observed audio chunks sent during this window: {observed_audio_chunk_count}\n\
         \n\
         Based only on screenshots and system audio observed during this \
         completed window, call \
         `{}` exactly once.\n\
         Do not answer in plain text.\n\
         If evidence is insufficient, still call the tool with `uncertain=true` \
         and explain the uncertainty in `summary`.",
        started_at.to_rfc3339(),
        ended_at.to_rfc3339(),
        SUBMIT_MINUTE_SUMMARY_TOOL,
    )
}
