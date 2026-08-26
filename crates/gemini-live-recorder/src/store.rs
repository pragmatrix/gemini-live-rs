//! Durable append-only storage for recorder output.

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum WindowRecordStatus {
    Ok,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WindowRecord {
    pub schema_version: u8,
    pub status: WindowRecordStatus,
    pub window_id: u64,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_sent_at: Option<DateTime<Utc>>,
    pub recorded_at: DateTime<Utc>,
    pub observed_frame_count: u32,
    pub observed_audio_chunk_count: u32,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub apps: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncertain: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl WindowRecord {
    // Record constructors mirror the JSONL schema field-for-field; a params
    // struct would just duplicate `WindowRecord` itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn success(
        model: &str,
        tool_call_id: String,
        recorded_at: DateTime<Utc>,
        window_id: u64,
        started_at: DateTime<Utc>,
        ended_at: DateTime<Utc>,
        prompt_sent_at: DateTime<Utc>,
        observed_frame_count: u32,
        observed_audio_chunk_count: u32,
        activity: String,
        summary: String,
        apps: Vec<String>,
        confidence: f32,
        uncertain: bool,
    ) -> Self {
        Self {
            schema_version: 1,
            status: WindowRecordStatus::Ok,
            window_id,
            started_at,
            ended_at,
            prompt_sent_at: Some(prompt_sent_at),
            recorded_at,
            observed_frame_count,
            observed_audio_chunk_count,
            model: model.to_string(),
            tool_call_id: Some(tool_call_id),
            activity: Some(activity),
            summary: Some(summary),
            apps: Some(apps),
            confidence: Some(confidence),
            uncertain: Some(uncertain),
            error: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn error(
        model: &str,
        recorded_at: DateTime<Utc>,
        window_id: u64,
        started_at: DateTime<Utc>,
        ended_at: DateTime<Utc>,
        prompt_sent_at: Option<DateTime<Utc>>,
        observed_frame_count: u32,
        observed_audio_chunk_count: u32,
        error: String,
    ) -> Self {
        Self {
            schema_version: 1,
            status: WindowRecordStatus::Error,
            window_id,
            started_at,
            ended_at,
            prompt_sent_at,
            recorded_at,
            observed_frame_count,
            observed_audio_chunk_count,
            model: model.to_string(),
            tool_call_id: None,
            activity: None,
            summary: None,
            apps: None,
            confidence: None,
            uncertain: None,
            error: Some(error),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct JsonlStore {
    path: PathBuf,
}

impl JsonlStore {
    pub(crate) fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub(crate) fn append(&self, record: &WindowRecord) -> io::Result<()> {
        if let Some(parent) = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            create_dir_all(parent)?;
        }

        let mut file = open_append_file(&self.path)?;
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        Ok(())
    }
}

fn open_append_file(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

#[cfg(test)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    #[test]
    fn appends_one_json_object_per_line() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let path = temp_dir.path().join("records.jsonl");
        let store = JsonlStore::new(&path);
        let record = WindowRecord::error(
            "models/test",
            Utc.timestamp_opt(10, 0).single().expect("timestamp"),
            7,
            Utc.timestamp_opt(0, 0).single().expect("timestamp"),
            Utc.timestamp_opt(5, 0).single().expect("timestamp"),
            None,
            0,
            0,
            "missing observation".into(),
        );

        store.append(&record).expect("append record");

        let written = std::fs::read_to_string(path).expect("read record");
        let lines = written.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 1);
        let round_trip: WindowRecord = serde_json::from_str(lines[0]).expect("json line");
        assert_eq!(round_trip.window_id, 7);
        assert_eq!(round_trip.status, WindowRecordStatus::Error);
    }
}
