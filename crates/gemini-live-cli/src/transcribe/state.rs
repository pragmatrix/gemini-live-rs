//! Transcribe-TUI state and pure reducer methods.
//!
//! `TranscribeApp` holds everything the transcribe view renders. All
//! transitions are synchronous and IO-free so line assembly — the part whose
//! upstream semantics (fragments + `finished` markers vs whole segments) we
//! may need to adjust — stays unit-testable.
//!
//! Line-assembly model:
//!
//! - `InterimInputTranscription` **replaces** [`TranscribeApp::interim`].
//! - `InputTranscription` fragments **append** to
//!   [`TranscribeApp::current_line`] and clear the interim text they
//!   supersede.
//! - `InputTranscriptionFinished` **flushes** `current_line` into a
//!   timestamped [`TranscriptEntry::Line`]; the flushed text is what the
//!   `--output` writer appends to the file.

use std::time::{Duration, Instant};

use gemini_live::session::SessionStatus;

use crate::input::InputEditor;

use super::audio::AudioSourceKind;
use super::slash::{self, CompletionItem};
use super::startup::TranscribeSettings;

/// Documented Live Transcribe cap on continuous streaming per session.
pub(crate) const SESSION_STREAM_CAP: Duration = Duration::from_secs(600);

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TranscriptEntry {
    /// A finalized transcript line, stamped with elapsed time at flush.
    Line { at: Duration, text: String },
    /// A system notice (connection changes, errors, setting hints).
    Notice(String),
}

pub(crate) struct TranscribeApp {
    pub(crate) entries: Vec<TranscriptEntry>,
    /// Finalized fragments of the line currently being spoken.
    pub(crate) current_line: String,
    /// Latest speculative hypothesis; superseded by finalized text.
    pub(crate) interim: String,
    /// Settings the current session connected with.
    pub(crate) active: TranscribeSettings,
    /// Settings edited by slash commands, applied on `/apply`.
    pub(crate) staged: TranscribeSettings,
    pub(crate) source: AudioSourceKind,
    pub(crate) model: String,
    pub(crate) connection: SessionStatus,
    /// Set when the session (re)connects; drives the mm:ss/10:00 timer.
    pub(crate) session_started_at: Option<Instant>,
    /// Wall-clock zero for transcript line timestamps.
    pub(crate) started_at: Instant,
    /// `true` between `/start` and `/end` in manual-VAD mode.
    pub(crate) manual_activity_open: bool,
    pub(crate) lagged_events: u64,
    pub(crate) output_label: Option<String>,
    pub(crate) input: InputEditor,
    completions: Vec<CompletionItem>,
    completion_index: usize,
    pub(crate) quit: bool,
}

impl TranscribeApp {
    pub(crate) fn new(
        model: String,
        settings: TranscribeSettings,
        source: AudioSourceKind,
        output_label: Option<String>,
    ) -> Self {
        let mut app = Self {
            entries: Vec::new(),
            current_line: String::new(),
            interim: String::new(),
            active: settings.clone(),
            staged: settings,
            source,
            model,
            connection: SessionStatus::Connecting,
            session_started_at: None,
            started_at: Instant::now(),
            manual_activity_open: false,
            lagged_events: 0,
            output_label,
            input: InputEditor::new(),
            completions: Vec::new(),
            completion_index: 0,
            quit: false,
        };
        app.notice(
            "ready — speak to transcribe; /mode /lang /vocab /vad stage settings, /apply reconnects"
                .into(),
        );
        app
    }

    pub(crate) fn notice(&mut self, text: String) {
        self.entries.push(TranscriptEntry::Notice(text));
    }

    // ── Transcript assembly ──────────────────────────────────────────────

    pub(crate) fn set_interim(&mut self, text: String) {
        self.interim = text;
    }

    pub(crate) fn append_final_fragment(&mut self, text: String) {
        self.current_line.push_str(&text);
        // Finalized text supersedes the hypothesis that produced it.
        self.interim.clear();
    }

    /// Flush the in-progress line into the transcript. Returns the flushed
    /// text so the caller can append it to the `--output` file.
    pub(crate) fn flush_line(&mut self) -> Option<String> {
        let text = self.current_line.trim().to_string();
        self.current_line.clear();
        self.interim.clear();
        if text.is_empty() {
            return None;
        }
        self.entries.push(TranscriptEntry::Line {
            at: self.started_at.elapsed(),
            text: text.clone(),
        });
        Some(text)
    }

    pub(crate) fn has_pending_line(&self) -> bool {
        !self.current_line.trim().is_empty()
    }

    // ── Session lifecycle ────────────────────────────────────────────────

    pub(crate) fn note_connected(&mut self) {
        self.connection = SessionStatus::Connected;
        self.session_started_at = Some(Instant::now());
    }

    /// Called on each tick with the freshly polled status. Detects the
    /// `Reconnecting → Connected` transition (there is no reconnect event)
    /// and resets the stream timer.
    pub(crate) fn observe_status(&mut self, status: SessionStatus) -> Option<String> {
        let previous = self.connection;
        self.connection = status;
        match (previous, status) {
            (SessionStatus::Reconnecting, SessionStatus::Connected) => {
                self.session_started_at = Some(Instant::now());
                self.manual_activity_open = false;
                Some("session rotated — streaming continues on a fresh connection".into())
            }
            (SessionStatus::Connected, SessionStatus::Reconnecting) => {
                Some("connection lost — reconnecting".into())
            }
            (previous, SessionStatus::Closed) if previous != SessionStatus::Closed => {
                Some("session closed — /apply reconnects".into())
            }
            _ => None,
        }
    }

    pub(crate) fn session_elapsed(&self) -> Option<Duration> {
        self.session_started_at.map(|at| at.elapsed())
    }

    pub(crate) fn has_staged_changes(&self) -> bool {
        self.active != self.staged
    }

    pub(crate) fn mark_settings_applied(&mut self) {
        self.active = self.staged.clone();
    }

    // ── Completions (same interaction model as chat's `App`) ─────────────

    pub(crate) fn refresh_completions(&mut self) {
        let selected = self
            .completions
            .get(self.completion_index)
            .map(|item| item.label.clone());
        self.completions = slash::completions(&self.input.text());
        self.completion_index = selected
            .and_then(|label| self.completions.iter().position(|item| item.label == label))
            .unwrap_or(0);
    }

    pub(crate) fn completion_count(&self) -> usize {
        self.completions.len().min(5)
    }

    pub(crate) fn completion_items(&self) -> &[CompletionItem] {
        &self.completions
    }

    pub(crate) fn selected_completion_index(&self) -> usize {
        self.completion_index
    }

    pub(crate) fn has_completions(&self) -> bool {
        !self.completions.is_empty()
    }

    pub(crate) fn select_next_completion(&mut self) {
        if self.completions.is_empty() {
            return;
        }
        self.completion_index = (self.completion_index + 1) % self.completions.len();
    }

    pub(crate) fn select_prev_completion(&mut self) {
        if self.completions.is_empty() {
            return;
        }
        self.completion_index = if self.completion_index == 0 {
            self.completions.len() - 1
        } else {
            self.completion_index - 1
        };
    }

    pub(crate) fn apply_selected_completion(&mut self) -> bool {
        let Some(item) = self.completions.get(self.completion_index).cloned() else {
            return false;
        };
        self.input
            .replace_range(item.replace_range, &item.replacement);
        self.refresh_completions();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> TranscribeApp {
        TranscribeApp::new(
            "models/test".into(),
            TranscribeSettings::default(),
            AudioSourceKind::Mic,
            None,
        )
    }

    #[test]
    fn interim_replaces_and_final_appends() {
        let mut app = app();
        app.set_interim("hel".into());
        app.set_interim("hello wor".into());
        assert_eq!(app.interim, "hello wor");

        app.append_final_fragment("Hello ".into());
        assert_eq!(app.current_line, "Hello ");
        assert!(app.interim.is_empty(), "finalized text clears interim");

        app.set_interim("wor".into());
        app.append_final_fragment("world.".into());
        assert_eq!(app.current_line, "Hello world.");
    }

    #[test]
    fn flush_line_moves_current_line_into_entries() {
        let mut app = app();
        app.append_final_fragment("Hello world.".into());
        app.set_interim("stale".into());

        let flushed = app.flush_line();
        assert_eq!(flushed.as_deref(), Some("Hello world."));
        assert!(app.current_line.is_empty());
        assert!(app.interim.is_empty());
        assert!(matches!(
            app.entries.last(),
            Some(TranscriptEntry::Line { text, .. }) if text == "Hello world."
        ));

        // A finished marker with nothing accumulated is a no-op.
        assert_eq!(app.flush_line(), None);
    }

    #[test]
    fn observe_status_reports_rotation_and_resets_timer() {
        let mut app = app();
        app.note_connected();
        let first_start = app.session_started_at;

        assert!(app.observe_status(SessionStatus::Connected).is_none());
        let lost = app.observe_status(SessionStatus::Reconnecting);
        assert!(lost.is_some());
        let rotated = app.observe_status(SessionStatus::Connected);
        assert!(rotated.unwrap().contains("rotated"));
        assert!(app.session_started_at.is_some());
        assert_ne!(app.session_started_at, first_start);

        let closed = app.observe_status(SessionStatus::Closed);
        assert!(closed.unwrap().contains("/apply"));
        assert!(app.observe_status(SessionStatus::Closed).is_none());
    }

    #[test]
    fn staged_changes_tracked_against_active() {
        let mut app = app();
        assert!(!app.has_staged_changes());
        app.staged.vocabulary.push("Kubernetes".into());
        assert!(app.has_staged_changes());
        app.mark_settings_applied();
        assert!(!app.has_staged_changes());
        assert_eq!(app.active.vocabulary, vec!["Kubernetes".to_string()]);
    }
}
