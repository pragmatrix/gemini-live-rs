//! `gemini-live transcribe` — real-time speech-to-text TUI.
//!
//! This module is the canonical home for transcribe-mode product behavior
//! (the counterpart of `startup.rs` + `main.rs` for chat mode):
//!
//! - Streams one audio source (mic through AEC, or system-output loopback)
//!   as 16 kHz mono PCM in 100 ms chunks to a Live Transcribe model over the
//!   plain core [`Session`] — no runtime/session-manager layer, because
//!   transcription needs no tools, memory, or dormancy.
//! - Renders speculative interim text (gray italics) that finalized
//!   transcript lines replace; `--output` appends each finalized line.
//! - Slash commands stage settings (`/mode`, `/lang`, `/vocab`, `/vad`);
//!   `/apply` reconnects with the staged setup because setup is only sent at
//!   handshake. `/source` switches capture immediately; `/start`/`/end` mark
//!   segments when automatic activity detection is off.
//! - The upstream 10-minute streaming cap ends sessions server-side; the
//!   default [`ReconnectPolicy`](gemini_live::session::ReconnectPolicy)
//!   re-handshakes automatically and the status-line timer restarts. The
//!   line in flight at rotation is flushed as a partial line.
//!
//! Applied settings persist into the `transcribe` section of the active
//! profile, so the next `gemini-live transcribe` starts where you left off.

mod audio;
mod render;
mod slash;
mod startup;
mod state;

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use clap::Args;
use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use gemini_live::ServerEvent;
use gemini_live::session::{Session, SessionObservation, SessionStatus};

use crate::profile;
use crate::render as terminal_render;
use crate::startup::resolve_startup_config;

use audio::{TranscribeAudio, TranscriptWriter};
use slash::{LangCommand, TranscribeSlashCommand, VocabCommand};
use startup::{
    MAX_VOCABULARY_TERMS, TranscribeStartup, build_transcribe_setup, persisted_transcribe_profile,
    resolve_transcribe_startup,
};
use state::TranscribeApp;

/// Real-time speech-to-text TUI backed by the Live Transcribe API.
#[derive(Debug, Args)]
pub(crate) struct TranscribeArgs {
    /// Append finalized transcript lines to this file.
    #[arg(long, short = 'o')]
    pub(crate) output: Option<PathBuf>,
    /// BCP-47 language codes (comma-separated or repeated). Empty = auto-detect.
    #[arg(long = "lang", value_delimiter = ',')]
    pub(crate) languages: Vec<String>,
    /// Transcript style: verbatim (literal) or smart (cleaned up).
    #[arg(long)]
    pub(crate) mode: Option<String>,
    /// Custom vocabulary bias terms (comma-separated or repeated).
    #[arg(long = "vocab", value_delimiter = ',')]
    pub(crate) vocabulary: Vec<String>,
    /// Disable automatic activity detection; mark segments with /start and /end.
    #[arg(long)]
    pub(crate) manual_vad: bool,
    /// Audio source: mic (default) or system (output loopback).
    #[arg(long)]
    pub(crate) source: Option<String>,
    /// Model override (default: models/gemini-3.5-transcribe-live).
    #[arg(long)]
    pub(crate) model: Option<String>,
}

/// Entry point called from `main` after tracing/profile setup.
///
/// Everything that can fail fast (config resolution, output file, first
/// connect, capture start) happens before the terminal enters the alternate
/// screen, so errors print as plain lines.
pub(crate) async fn run_command(
    args: &TranscribeArgs,
    mut profile_store: profile::ProfileStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let chat_startup = resolve_startup_config(
        |key| std::env::var(key).ok(),
        profile_store.active_profile(),
        profile_store.active_profile_name(),
    )?;
    let startup = resolve_transcribe_startup(
        args,
        |key| std::env::var(key).ok(),
        profile_store.active_profile(),
        chat_startup.transport.clone(),
    )?;

    let mut writer = args
        .output
        .as_deref()
        .map(TranscriptWriter::open)
        .transpose()
        .map_err(|e| format!("cannot open --output file: {e}"))?;

    let audio = TranscribeAudio::start(startup.source)?;
    let session = Session::connect(startup.session_config()).await?;

    let output_label = args
        .output
        .as_deref()
        .map(|path| path.display().to_string());
    let mut app = TranscribeApp::new(
        startup.model.clone(),
        startup.settings.clone(),
        startup.source,
        output_label,
    );
    app.notice(format!("audio source: {}", audio.description()));
    app.note_connected();

    terminal_render::install_panic_hook();
    let mut terminal = terminal_render::init_terminal()?;
    let result = run_loop(
        &mut terminal,
        &startup,
        &mut app,
        session,
        audio,
        &mut writer,
        &mut profile_store,
    )
    .await;
    terminal_render::restore_terminal(&mut terminal)?;
    result
}

async fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    startup: &TranscribeStartup,
    app: &mut TranscribeApp,
    mut session: Session,
    mut audio: TranscribeAudio,
    writer: &mut Option<TranscriptWriter>,
    profile_store: &mut profile::ProfileStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut term_events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    // The receive branch must pause once the session runner is gone, or an
    // immediately-ready `None` would spin the select loop.
    let mut session_dead = false;

    loop {
        terminal.draw(|frame| render::draw(frame, app))?;
        if app.quit {
            break;
        }

        tokio::select! {
            maybe_event = term_events.next() => {
                let Some(event) = maybe_event.transpose()? else { break };
                if let Event::Key(key) = event
                    && key.kind != KeyEventKind::Release
                    && let Some(command) = handle_key(app, key)
                {
                    apply_slash_command(
                        command,
                        startup,
                        app,
                        &mut session,
                        &mut session_dead,
                        &mut audio,
                        writer,
                        profile_store,
                    )
                    .await;
                }
            }
            Some(captured) = audio.next_captured() => {
                if !session_dead && session.status() == SessionStatus::Connected {
                    let _ = session
                        .send_audio_at_rate(&captured.pcm_i16_le, captured.sample_rate)
                        .await;
                }
                // While reconnecting, chunks are dropped: replaying stale
                // audio into a fresh transcription session has no value.
            }
            observed = session.next_observed_event(), if !session_dead => {
                match observed {
                    Some(SessionObservation::Event(event)) => {
                        handle_server_event(event, app, writer);
                    }
                    Some(SessionObservation::Lagged { count }) => {
                        app.lagged_events += count;
                    }
                    None => {
                        session_dead = true;
                        if let Some(notice) = app.observe_status(SessionStatus::Closed) {
                            app.notice(notice);
                        }
                    }
                }
            }
            _ = tick.tick() => {
                if !session_dead
                    && let Some(notice) = app.observe_status(session.status())
                {
                    app.notice(notice);
                }
            }
        }
    }

    // Flush the line in flight so it reaches the transcript and the file.
    if app.has_pending_line()
        && let Some(line) = app.flush_line()
    {
        append_output_line(writer, &line, app);
    }
    if !session_dead {
        let _ = session.audio_stream_end().await;
        let _ = session.close().await;
    }
    Ok(())
}

fn handle_server_event(
    event: ServerEvent,
    app: &mut TranscribeApp,
    writer: &mut Option<TranscriptWriter>,
) {
    match event {
        ServerEvent::SetupComplete => app.note_connected(),
        ServerEvent::InterimInputTranscription(text) => app.set_interim(text),
        ServerEvent::InputTranscription(text) => app.append_final_fragment(text),
        ServerEvent::InputTranscriptionFinished => {
            if let Some(line) = app.flush_line() {
                append_output_line(writer, &line, app);
            }
        }
        ServerEvent::GoAway { time_left } => {
            let hint = time_left
                .map(|left| format!(" in {}s", left.as_secs()))
                .unwrap_or_default();
            app.notice(format!("server rotating the session{hint}"));
        }
        ServerEvent::Closed { reason } => {
            if let Some(notice) = app.observe_status(SessionStatus::Closed) {
                app.notice(notice);
            }
            app.notice(format!("connection closed: {reason}"));
        }
        ServerEvent::Error(error) => app.notice(format!("api error: {}", error.message)),
        _ => {}
    }
}

fn append_output_line(writer: &mut Option<TranscriptWriter>, line: &str, app: &mut TranscribeApp) {
    if let Some(writer) = writer
        && let Err(error) = writer.append_line(line)
    {
        app.notice(format!("output write failed: {error}"));
    }
}

fn handle_key(app: &mut TranscribeApp, key: KeyEvent) -> Option<TranscribeSlashCommand> {
    match key.code {
        KeyCode::Enter => {
            let raw = app.input.take_text();
            app.refresh_completions();
            let trimmed = raw.trim().to_string();
            if trimmed.is_empty() {
                return None;
            }
            match slash::parse(&trimmed) {
                Some(Ok(command)) => return Some(command),
                Some(Err(error)) => app.notice(format!("[slash] {error}")),
                None => app.notice(
                    "transcribe mode has no text input — use /commands (Tab lists them)".into(),
                ),
            }
        }
        KeyCode::Tab => {
            if !app.apply_selected_completion() {
                app.input.handle_key(key);
                app.refresh_completions();
            }
        }
        KeyCode::BackTab => app.select_prev_completion(),
        KeyCode::Up if app.has_completions() => app.select_prev_completion(),
        KeyCode::Down if app.has_completions() => app.select_next_completion(),
        KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.quit = true;
        }
        KeyCode::Esc => app.quit = true,
        _ => {
            app.input.handle_key(key);
            app.refresh_completions();
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
async fn apply_slash_command(
    command: TranscribeSlashCommand,
    startup: &TranscribeStartup,
    app: &mut TranscribeApp,
    session: &mut Session,
    session_dead: &mut bool,
    audio: &mut TranscribeAudio,
    writer: &mut Option<TranscriptWriter>,
    profile_store: &mut profile::ProfileStore,
) {
    match command {
        TranscribeSlashCommand::Mode(None) => {
            let active = app.active.mode.map(|m| m.as_str()).unwrap_or("verbatim");
            let staged = app.staged.mode.map(|m| m.as_str()).unwrap_or("verbatim");
            app.notice(format!("mode: active {active}, staged {staged}"));
        }
        TranscribeSlashCommand::Mode(Some(mode)) => {
            app.staged.mode = Some(mode);
            note_staged(app, format!("mode staged: {mode}"));
        }
        TranscribeSlashCommand::Lang(LangCommand::Show) => {
            app.notice(format!(
                "languages: active {}, staged {}",
                describe_languages(&app.active.languages),
                describe_languages(&app.staged.languages)
            ));
        }
        TranscribeSlashCommand::Lang(LangCommand::Set(codes)) => {
            app.staged.languages = codes;
            let described = describe_languages(&app.staged.languages);
            note_staged(app, format!("languages staged: {described}"));
        }
        TranscribeSlashCommand::Lang(LangCommand::Clear) => {
            app.staged.languages.clear();
            note_staged(app, "languages staged: auto-detect".into());
        }
        TranscribeSlashCommand::Vocab(VocabCommand::List) => {
            app.notice(format!(
                "vocabulary: active {} terms, staged {} terms{}",
                app.active.vocabulary.len(),
                app.staged.vocabulary.len(),
                if app.staged.vocabulary.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", app.staged.vocabulary.join(", "))
                }
            ));
        }
        TranscribeSlashCommand::Vocab(VocabCommand::Add(terms)) => {
            if terms.is_empty() {
                app.notice("usage: /vocab add <term>…".into());
                return;
            }
            for term in terms {
                if !app.staged.vocabulary.contains(&term) {
                    app.staged.vocabulary.push(term);
                }
            }
            if app.staged.vocabulary.len() > MAX_VOCABULARY_TERMS {
                app.staged.vocabulary.truncate(MAX_VOCABULARY_TERMS);
                app.notice(format!("vocabulary capped at {MAX_VOCABULARY_TERMS} terms"));
            }
            let count = app.staged.vocabulary.len();
            note_staged(app, format!("vocabulary staged: {count} terms"));
        }
        TranscribeSlashCommand::Vocab(VocabCommand::Remove(term)) => {
            let before = app.staged.vocabulary.len();
            app.staged.vocabulary.retain(|existing| existing != &term);
            if app.staged.vocabulary.len() == before {
                app.notice(format!("vocabulary: {term:?} not staged"));
            } else {
                let count = app.staged.vocabulary.len();
                note_staged(app, format!("vocabulary staged: {count} terms"));
            }
        }
        TranscribeSlashCommand::Vocab(VocabCommand::Clear) => {
            app.staged.vocabulary.clear();
            note_staged(app, "vocabulary staged: empty".into());
        }
        TranscribeSlashCommand::Vad(None) => {
            app.notice(format!(
                "vad: active {}, staged {}",
                describe_vad(app.active.manual_vad),
                describe_vad(app.staged.manual_vad)
            ));
        }
        TranscribeSlashCommand::Vad(Some(automatic)) => {
            app.staged.manual_vad = !automatic;
            let described = describe_vad(app.staged.manual_vad);
            note_staged(app, format!("vad staged: {described}"));
        }
        TranscribeSlashCommand::Apply => {
            apply_staged_settings(
                startup,
                app,
                session,
                session_dead,
                audio,
                writer,
                profile_store,
            )
            .await;
        }
        TranscribeSlashCommand::Source(None) => {
            app.notice(format!("source: {}", audio.description()));
        }
        TranscribeSlashCommand::Source(Some(source)) => {
            if source == audio.source {
                app.notice(format!("source already {source}"));
                return;
            }
            match audio.switch(source) {
                Ok(()) => {
                    app.source = source;
                    if !*session_dead {
                        // Tell the server the previous audio stream ended.
                        let _ = session.audio_stream_end().await;
                    }
                    app.notice(format!("audio source: {}", audio.description()));
                    persist(app, audio, profile_store, &startup.model);
                }
                Err(error) => {
                    app.notice(format!(
                        "source switch failed, staying on {}: {error}",
                        audio.source
                    ));
                }
            }
        }
        TranscribeSlashCommand::Start => {
            manual_activity(app, session, *session_dead, true).await;
        }
        TranscribeSlashCommand::End => {
            manual_activity(app, session, *session_dead, false).await;
        }
        TranscribeSlashCommand::Clear => {
            app.entries.clear();
            app.notice("transcript cleared (output file untouched)".into());
        }
    }
}

async fn apply_staged_settings(
    startup: &TranscribeStartup,
    app: &mut TranscribeApp,
    session: &mut Session,
    session_dead: &mut bool,
    audio: &mut TranscribeAudio,
    writer: &mut Option<TranscriptWriter>,
    profile_store: &mut profile::ProfileStore,
) {
    if let Err(error) = app.staged.validate() {
        app.notice(format!("cannot apply: {error}"));
        return;
    }

    // The line in flight belongs to the old session; flush it first.
    if app.has_pending_line()
        && let Some(line) = app.flush_line()
    {
        append_output_line(writer, &line, app);
    }

    if !*session_dead {
        let _ = session.audio_stream_end().await;
        let _ = session.clone().close().await;
    }

    let mut config = startup.session_config();
    config.setup = build_transcribe_setup(&app.staged, &startup.model);
    match Session::connect(config).await {
        Ok(new_session) => {
            *session = new_session;
            *session_dead = false;
            app.mark_settings_applied();
            app.note_connected();
            app.manual_activity_open = false;
            app.notice("reconnected with the staged settings".into());
            persist(app, audio, profile_store, &startup.model);
        }
        Err(error) => {
            *session_dead = true;
            if let Some(notice) = app.observe_status(SessionStatus::Closed) {
                app.notice(notice);
            }
            app.notice(format!("reconnect failed: {error} — /apply retries"));
        }
    }
}

async fn manual_activity(
    app: &mut TranscribeApp,
    session: &mut Session,
    session_dead: bool,
    start: bool,
) {
    if !app.active.manual_vad {
        app.notice("automatic activity detection is on — stage /vad off and /apply first".into());
        return;
    }
    if session_dead || session.status() != SessionStatus::Connected {
        app.notice("not connected — /apply reconnects".into());
        return;
    }
    let result = if start {
        session.activity_start().await
    } else {
        session.activity_end().await
    };
    match result {
        Ok(()) => {
            app.manual_activity_open = start;
            app.notice(if start {
                "activity started — /end closes the segment".into()
            } else {
                "activity ended".into()
            });
        }
        Err(error) => app.notice(format!("activity signal failed: {error}")),
    }
}

fn persist(
    app: &TranscribeApp,
    audio: &TranscribeAudio,
    profile_store: &mut profile::ProfileStore,
    model: &str,
) {
    let snapshot = persisted_transcribe_profile(model, &app.active, audio.source);
    if let Err(error) = profile_store.set_transcribe_profile(snapshot) {
        tracing::warn!("failed to persist transcribe profile: {error}");
    }
}

fn note_staged(app: &mut TranscribeApp, message: String) {
    app.notice(format!(
        "{message} — /apply reconnects with staged settings"
    ));
}

fn describe_languages(languages: &[String]) -> String {
    if languages.is_empty() {
        "auto-detect".into()
    } else {
        languages.join(",")
    }
}

fn describe_vad(manual: bool) -> &'static str {
    if manual {
        "manual (/start /end)"
    } else {
        "automatic"
    }
}
