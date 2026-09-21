//! Tool schema and validation for minute summaries.

use gemini_live::types::{FunctionDeclaration, FunctionResponse};
use serde::Deserialize;
use serde_json::json;

pub(crate) const SUBMIT_MINUTE_SUMMARY_TOOL: &str = "submit_minute_summary";

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MinuteSummarySubmission {
    pub activity: String,
    pub summary: String,
    pub apps: Vec<String>,
    pub confidence: f32,
    pub uncertain: bool,
}

impl MinuteSummarySubmission {
    pub(crate) fn parse(args: serde_json::Value) -> Result<Self, String> {
        let submission: Self = serde_json::from_value(args)
            .map_err(|error| format!("invalid tool arguments: {error}"))?;
        submission.validate()?;
        Ok(submission.normalize())
    }

    fn validate(&self) -> Result<(), String> {
        if self.activity.trim().is_empty() {
            return Err("`activity` must not be empty".into());
        }
        if self.summary.trim().is_empty() {
            return Err("`summary` must not be empty".into());
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err("`confidence` must be within 0.0..=1.0".into());
        }
        if self.apps.iter().any(|value| value.trim().is_empty()) {
            return Err("`apps` must not contain empty strings".into());
        }
        Ok(())
    }

    fn normalize(mut self) -> Self {
        self.activity = self.activity.trim().to_string();
        self.summary = self.summary.trim().to_string();
        self.apps = self
            .apps
            .into_iter()
            .map(|value| value.trim().to_string())
            .collect();
        self
    }
}

pub(crate) fn declaration() -> FunctionDeclaration {
    FunctionDeclaration {
        name: SUBMIT_MINUTE_SUMMARY_TOOL.into(),
        description: "Commit the just-finished observation window as one structured desktop-activity summary. Call exactly once when the host asks for a completed window summary.".into(),
        parameters: json!({
            "type": "object",
            "properties": {
                "activity": {
                    "type": "string",
                    "description": "Short activity label, for example 'coding rust', 'reading docs', or 'chatting'."
                },
                "summary": {
                    "type": "string",
                    "description": "One concise factual sentence about what happened during the completed window."
                },
                "apps": {
                    "type": "array",
                    "description": "Visible applications, websites, or surfaces. Use an empty array if unknown.",
                    "items": {
                        "type": "string"
                    }
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score from 0.0 to 1.0."
                },
                "uncertain": {
                    "type": "boolean",
                    "description": "True when the visible evidence was weak or ambiguous."
                }
            },
            "required": ["activity", "summary", "apps", "confidence", "uncertain"]
        }),
        scheduling: None,
        behavior: None,
    }
}

pub(crate) fn ok_response(call_id: Option<String>) -> FunctionResponse {
    FunctionResponse {
        id: call_id,
        name: SUBMIT_MINUTE_SUMMARY_TOOL.into(),
        response: json!({
            "ok": true
        }),
        scheduling: None,
    }
}

pub(crate) fn error_response(
    call_id: Option<String>,
    call_name: String,
    message: impl Into<String>,
) -> FunctionResponse {
    FunctionResponse {
        id: call_id,
        name: call_name,
        response: json!({
            "ok": false,
            "error": {
                "message": message.into()
            }
        }),
        scheduling: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_out_of_range_confidence() {
        let result = MinuteSummarySubmission::parse(json!({
            "activity": "coding",
            "summary": "Editing files",
            "apps": ["Terminal"],
            "confidence": 1.5,
            "uncertain": false
        }));

        assert!(result.is_err());
    }

    #[test]
    fn trims_and_accepts_valid_submission() {
        let result = MinuteSummarySubmission::parse(json!({
            "activity": " coding ",
            "summary": " Editing files ",
            "apps": [" Terminal ", "Browser"],
            "confidence": 0.8,
            "uncertain": false
        }))
        .expect("valid submission");

        assert_eq!(result.activity, "coding");
        assert_eq!(result.summary, "Editing files");
        assert_eq!(result.apps, vec!["Terminal", "Browser"]);
    }
}
