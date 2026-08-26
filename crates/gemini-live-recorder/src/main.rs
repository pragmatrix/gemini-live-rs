//! Experimental continuous desktop activity recorder.
//!
//! This host keeps one Gemini Live runtime hot, streams screenshots from all
//! selected monitors plus system-output audio, and closes fixed observation
//! windows on a timer. Each completed window is summarized by one required
//! tool call and appended to a JSONL log on disk.
//!
//! This binary intentionally does not depend on `gemini-live-harness`. Its
//! durable responsibility is narrow and local:
//!
//! - capture all selected desktop monitors
//! - capture system-output audio
//! - ask Gemini Live for one structured summary per completed window
//! - write one append-only record per window, including failure records

mod prompt;
mod store;
mod tool;

use std::collections::VecDeque;
use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use clap::{Args, Parser};
use gemini_live::ReconnectPolicy;
use gemini_live::audio::INPUT_SAMPLE_RATE;
use gemini_live::transport::{Auth, TransportConfig};
use gemini_live::types::{
    ActivityHandling, AudioTranscriptionConfig, AutomaticActivityDetection, ClientMessage, Content,
    EmptyObject, FunctionCallRequest, GenerationConfig, MediaResolution, Modality, Part,
    RealtimeInput, RealtimeInputConfig, SessionResumptionConfig, SetupConfig, ThinkingConfig,
    ThinkingLevel, Tool,
};
use gemini_live_io::audio::{CapturedAudio, SystemAudioCapture};
use gemini_live_io::error::{AudioIoError, ScreenCaptureError};
use gemini_live_io::screen::{
    CaptureTarget, EncodedFrame, ScreenCapture, ScreenCaptureConfig, list_monitor_targets,
    list_targets,
};
use gemini_live_runtime::{
    GeminiSessionDriver, ManagedRuntime, RuntimeConfig, RuntimeEvent, RuntimeEventReceiver,
    RuntimeLifecycleEvent,
};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::{Instant, MissedTickBehavior};
use tracing::{debug, info, warn};

use crate::prompt::{summary_prompt, system_instruction};
use crate::store::{JsonlStore, WindowRecord};
use crate::tool::{
    MinuteSummarySubmission, SUBMIT_MINUTE_SUMMARY_TOOL, declaration, error_response, ok_response,
};

const DEFAULT_MODEL: &str = "models/gemini-3.1-flash-live-preview";
const DEFAULT_OUTPUT_PATH: &str = "activity-log.jsonl";
const RUNTIME_RESTART_ATTEMPTS: usize = 3;
const RUNTIME_RESTART_BACKOFF: Duration = Duration::from_secs(2);

#[derive(Debug, Parser)]
#[command(name = env!("CARGO_PKG_NAME"))]
struct Cli {
    /// Print capture targets and exit.
    #[arg(long)]
    list_targets: bool,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, Clone, Args)]
struct RunArgs {
    /// Gemini API key. Falls back to the `GEMINI_API_KEY` environment variable.
    #[arg(long)]
    api_key: Option<String>,
    /// Live model resource name.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,
    /// Monitor id returned by `--list-targets`. Repeat to capture only a subset.
    ///
    /// When omitted, the recorder captures all available monitors.
    #[arg(long = "monitor-id")]
    monitor_ids: Vec<usize>,
    /// Seconds between captured screenshots.
    #[arg(long, default_value_t = 5.0)]
    frame_interval_secs: f64,
    /// Seconds per observation window.
    #[arg(long, default_value_t = 60.0)]
    summary_interval_secs: f64,
    /// Maximum seconds to wait for one summary turn before recording a failure.
    #[arg(long, default_value_t = 20.0)]
    summary_timeout_secs: f64,
    /// Maximum screenshot dimension before JPEG encoding.
    #[arg(long, default_value_t = 1280)]
    max_dimension: u32,
    /// JPEG quality passed to the screen adapter.
    #[arg(long, default_value_t = 75)]
    jpeg_quality: u8,
    /// JSONL output path.
    #[arg(long, default_value = DEFAULT_OUTPUT_PATH)]
    output: PathBuf,
    /// Extra system-instruction text appended to the built-in recorder contract.
    #[arg(long)]
    extra_instruction: Option<String>,
}

#[derive(Debug, Error)]
enum RecorderError {
    #[error("missing Gemini API key; set `GEMINI_API_KEY` or pass `--api-key`")]
    MissingApiKey,
    #[error("no monitor targets are available")]
    NoMonitorsAvailable,
    #[error("monitor id {0} is not available")]
    MonitorIdNotFound(usize),
    #[error("`frame-interval-secs` must be greater than 0")]
    InvalidFrameInterval,
    #[error("`summary-interval-secs` must be greater than 0")]
    InvalidSummaryInterval,
    #[error("`summary-timeout-secs` must be greater than 0")]
    InvalidSummaryTimeout,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Runtime(#[from] gemini_live_runtime::RuntimeError),
    #[error(transparent)]
    Audio(#[from] AudioIoError),
    #[error(transparent)]
    Screen(#[from] ScreenCaptureError),
}

#[derive(Debug, Clone)]
struct ActiveWindow {
    id: u64,
    started_at: DateTime<Utc>,
    observed_frame_count: u32,
    observed_audio_chunk_count: u32,
}

impl ActiveWindow {
    fn new(id: u64, started_at: DateTime<Utc>) -> Self {
        Self {
            id,
            started_at,
            observed_frame_count: 0,
            observed_audio_chunk_count: 0,
        }
    }

    fn observe_frame(&mut self) {
        self.observed_frame_count = self.observed_frame_count.saturating_add(1);
    }

    fn observe_audio_chunk(&mut self) {
        self.observed_audio_chunk_count = self.observed_audio_chunk_count.saturating_add(1);
    }

    fn close(self, ended_at: DateTime<Utc>) -> ClosedWindow {
        ClosedWindow {
            id: self.id,
            started_at: self.started_at,
            ended_at,
            observed_frame_count: self.observed_frame_count,
            observed_audio_chunk_count: self.observed_audio_chunk_count,
        }
    }
}

#[derive(Debug, Clone)]
struct ClosedWindow {
    id: u64,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    observed_frame_count: u32,
    observed_audio_chunk_count: u32,
}

#[derive(Debug)]
enum PendingMedia {
    Frame(EncodedFrame),
    Audio(CapturedAudio),
}

#[derive(Debug)]
struct SummaryTurnState {
    window: ClosedWindow,
    prompt_sent_at: DateTime<Utc>,
    deadline: Instant,
    accepted_submission: bool,
    issues: Vec<String>,
    model_text_fragments: Vec<String>,
}

impl SummaryTurnState {
    fn new(window: ClosedWindow, prompt_sent_at: DateTime<Utc>, deadline: Instant) -> Self {
        Self {
            window,
            prompt_sent_at,
            deadline,
            accepted_submission: false,
            issues: Vec::new(),
            model_text_fragments: Vec::new(),
        }
    }

    fn note_issue(&mut self, issue: impl Into<String>) {
        self.issues.push(issue.into());
    }

    fn note_model_text(&mut self, text: &str) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        if self.model_text_fragments.len() >= 6 {
            return;
        }
        self.model_text_fragments.push(trimmed.to_string());
    }

    fn missing_submission_reason(&self) -> String {
        let mut parts = Vec::new();
        if !self.issues.is_empty() {
            parts.push(self.issues.join("; "));
        }
        if !self.model_text_fragments.is_empty() {
            parts.push(format!(
                "model text: {}",
                self.model_text_fragments.join(" | ")
            ));
        }
        if parts.is_empty() {
            "summary turn finished without a valid submit_minute_summary tool call".into()
        } else {
            parts.join("; ")
        }
    }
}

struct RecorderApp {
    model: String,
    store: JsonlStore,
    runtime: ManagedRuntime<GeminiSessionDriver>,
    runtime_events: RuntimeEventReceiver,
    active_window: ActiveWindow,
    next_window_id: u64,
    pending_summaries: VecDeque<ClosedWindow>,
    current_summary: Option<SummaryTurnState>,
    pending_media: VecDeque<PendingMedia>,
    observation_activity_open: bool,
    summary_timeout: Duration,
}

impl RecorderApp {
    async fn new(args: &RunArgs) -> Result<Self, RecorderError> {
        let api_key = args
            .api_key
            .clone()
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or(RecorderError::MissingApiKey)?;
        let (runtime, runtime_events) = ManagedRuntime::new(
            RuntimeConfig {
                session: gemini_live::SessionConfig {
                    transport: TransportConfig {
                        auth: Auth::ApiKey(api_key),
                        ..Default::default()
                    },
                    setup: build_setup(&args.model, args.extra_instruction.as_deref()),
                    reconnect: ReconnectPolicy::default(),
                },
            },
            GeminiSessionDriver,
        );
        let mut app = Self {
            model: args.model.clone(),
            store: JsonlStore::new(&args.output),
            runtime,
            runtime_events,
            active_window: ActiveWindow::new(0, Utc::now()),
            next_window_id: 1,
            pending_summaries: VecDeque::new(),
            current_summary: None,
            pending_media: VecDeque::new(),
            observation_activity_open: false,
            summary_timeout: Duration::from_secs_f64(args.summary_timeout_secs),
        };
        app.runtime.connect().await?;
        let now = Utc::now();
        app.active_window = ActiveWindow::new(app.allocate_window_id(), now);
        info!(
            active_window_id = app.active_window.id,
            active_window_started_at = %app.active_window.started_at.to_rfc3339(),
            "recorder runtime established and initial observation window opened"
        );
        Ok(app)
    }

    async fn run(
        mut self,
        screen_captures: Vec<ScreenCapture>,
        system_audio_capture: SystemAudioCapture,
        mut frame_rx: mpsc::Receiver<EncodedFrame>,
        mut audio_rx: mpsc::Receiver<CapturedAudio>,
        summary_interval: Duration,
    ) -> Result<(), RecorderError> {
        let mut screen_captures = Some(screen_captures);
        let mut system_audio_capture = Some(system_audio_capture);
        let mut summary_tick = tokio::time::interval(summary_interval);
        summary_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        summary_tick.tick().await;
        let mut shutting_down = false;

        loop {
            if shutting_down && self.current_summary.is_none() && self.pending_summaries.is_empty()
            {
                break;
            }

            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    if shutting_down {
                        continue;
                    }
                    info!("received interrupt; finalizing pending windows");
                    shutting_down = true;
                    let _ = screen_captures.take();
                    let _ = system_audio_capture.take();
                    let now = Utc::now();
                    let closed = self.rotate_window(now);
                    if closed.observed_frame_count > 0 || closed.observed_audio_chunk_count > 0 {
                        self.pending_summaries.push_back(closed);
                    }
                    self.dispatch_next_summary_if_idle().await?;
                }
                _ = async {
                    if shutting_down {
                        std::future::pending::<()>().await;
                    } else {
                        summary_tick.tick().await;
                    }
                } => {
                    let now = Utc::now();
                    let closed = self.rotate_window(now);
                    if closed.observed_frame_count == 0 && closed.observed_audio_chunk_count == 0 {
                        self.write_error_record(
                            &closed,
                            None,
                            "window closed without any observed frames or audio".into(),
                        )?;
                        self.restart_runtime("observation window closed without media")
                            .await?;
                    } else {
                        self.pending_summaries.push_back(closed);
                    }
                    self.dispatch_next_summary_if_idle().await?;
                }
                Some(frame) = frame_rx.recv() => {
                    self.handle_frame(frame).await?;
                }
                Some(audio) = audio_rx.recv() => {
                    self.handle_audio(audio).await?;
                }
                Some(event) = self.runtime_events.recv() => {
                    self.handle_runtime_event(event).await?;
                }
                _ = async {
                    if let Some(deadline) = self.current_summary.as_ref().map(|state| state.deadline) {
                        tokio::time::sleep_until(deadline).await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    self.handle_summary_timeout().await?;
                }
                else => break,
            }
        }

        self.runtime.close().await?;
        Ok(())
    }

    fn allocate_window_id(&mut self) -> u64 {
        let id = self.next_window_id;
        self.next_window_id = self.next_window_id.saturating_add(1);
        id
    }

    fn rotate_window(&mut self, ended_at: DateTime<Utc>) -> ClosedWindow {
        let next_id = self.allocate_window_id();
        let active = std::mem::replace(
            &mut self.active_window,
            ActiveWindow::new(next_id, ended_at),
        );
        let closed = active.close(ended_at);
        info!(
            closed_window_id = closed.id,
            next_window_id = self.active_window.id,
            started_at = %closed.started_at.to_rfc3339(),
            ended_at = %closed.ended_at.to_rfc3339(),
            observed_frame_count = closed.observed_frame_count,
            observed_audio_chunk_count = closed.observed_audio_chunk_count,
            "rotated observation window"
        );
        closed
    }

    async fn handle_frame(&mut self, frame: EncodedFrame) -> Result<(), RecorderError> {
        let is_first_frame = self.active_window.observed_frame_count == 0;
        self.active_window.observe_frame();
        if self.current_summary.is_some() || !self.pending_summaries.is_empty() {
            if self.pending_media.is_empty() {
                info!(
                    active_window_id = self.active_window.id,
                    summary_in_flight = self.current_summary.is_some(),
                    pending_summary_windows = self.pending_summaries.len(),
                    "buffering live media while summary pipeline is busy"
                );
            }
            self.pending_media.push_back(PendingMedia::Frame(frame));
            return Ok(());
        }
        self.send_frame(frame, is_first_frame).await?;
        Ok(())
    }

    async fn handle_audio(&mut self, audio: CapturedAudio) -> Result<(), RecorderError> {
        let is_first_audio_chunk = self.active_window.observed_audio_chunk_count == 0;
        self.active_window.observe_audio_chunk();
        if self.current_summary.is_some() || !self.pending_summaries.is_empty() {
            if self.pending_media.is_empty() {
                info!(
                    active_window_id = self.active_window.id,
                    summary_in_flight = self.current_summary.is_some(),
                    pending_summary_windows = self.pending_summaries.len(),
                    "buffering live media while summary pipeline is busy"
                );
            }
            self.pending_media.push_back(PendingMedia::Audio(audio));
            return Ok(());
        }
        self.send_audio(audio, is_first_audio_chunk).await?;
        Ok(())
    }

    async fn handle_runtime_event(&mut self, event: RuntimeEvent) -> Result<(), RecorderError> {
        match event {
            RuntimeEvent::Lifecycle(RuntimeLifecycleEvent::Connected) => {
                info!("runtime connected");
            }
            RuntimeEvent::Lifecycle(RuntimeLifecycleEvent::Closed { reason }) => {
                warn!("runtime closed: {reason}");
                if let Some(state) = self.current_summary.take() {
                    self.write_error_record(
                        &state.window,
                        Some(state.prompt_sent_at),
                        format!("runtime closed before summary completed: {reason}"),
                    )?;
                }
                self.restart_runtime(format!("runtime closed: {reason}"))
                    .await?;
            }
            RuntimeEvent::ToolCallRequested { call } => {
                self.handle_tool_call(call).await?;
            }
            RuntimeEvent::ToolCallCancellationRequested { ids } => {
                warn!(tool_call_ids = %ids.join(","), "model cancelled pending tool call");
                if let Some(state) = &mut self.current_summary {
                    state.note_issue(format!("tool call cancelled: {}", ids.join(",")));
                }
            }
            RuntimeEvent::Server(gemini_live::ServerEvent::ModelText(text)) => {
                if let Some(state) = &mut self.current_summary {
                    state.note_model_text(&text);
                }
                debug!("model text: {text}");
            }
            RuntimeEvent::Server(gemini_live::ServerEvent::TurnComplete) => {
                self.finish_summary_turn(None).await?;
            }
            RuntimeEvent::Server(gemini_live::ServerEvent::Interrupted) => {
                warn!("summary turn was interrupted");
                if let Some(state) = &mut self.current_summary {
                    state.note_issue("summary turn was interrupted");
                }
            }
            RuntimeEvent::Server(gemini_live::ServerEvent::Error(error)) => {
                warn!("api error: {}", error.message);
            }
            RuntimeEvent::SendFailed(failure) => {
                warn!(
                    "runtime send failure ({:?}): {}",
                    failure.operation, failure.reason
                );
            }
            RuntimeEvent::Lagged { count } => {
                warn!("runtime event receiver lagged by {count} events");
            }
            RuntimeEvent::Lifecycle(_) | RuntimeEvent::Server(_) => {}
        }
        Ok(())
    }

    async fn handle_tool_call(&mut self, call: FunctionCallRequest) -> Result<(), RecorderError> {
        let call_id = call.id.clone();
        let call_name = call.name.clone();
        info!(
            tool_call_id = %call_id,
            tool_name = %call_name,
            summary_window_id = self.current_summary.as_ref().map(|state| state.window.id),
            "received tool call from model"
        );

        let Some(state) = &mut self.current_summary else {
            self.runtime
                .send_tool_response(vec![error_response(
                    call.id,
                    call.name,
                    "no summary window is currently awaiting submission",
                )])
                .await?;
            return Ok(());
        };

        if call.name != SUBMIT_MINUTE_SUMMARY_TOOL {
            state.note_issue(format!("unexpected tool `{}`", call.name));
            warn!(
                tool_call_id = %call_id,
                tool_name = %call_name,
                expected_tool_name = SUBMIT_MINUTE_SUMMARY_TOOL,
                summary_window_id = state.window.id,
                "rejecting unexpected tool call"
            );
            self.runtime
                .send_tool_response(vec![error_response(
                    call.id,
                    call.name,
                    "unexpected tool name for recorder host",
                )])
                .await?;
            return Ok(());
        }

        if state.accepted_submission {
            state.note_issue("received duplicate submit_minute_summary call");
            warn!(
                tool_call_id = %call_id,
                tool_name = %call_name,
                summary_window_id = state.window.id,
                "rejecting duplicate minute summary submission"
            );
            self.runtime
                .send_tool_response(vec![error_response(
                    call.id,
                    call.name,
                    "submit_minute_summary was already accepted for this window",
                )])
                .await?;
            return Ok(());
        }

        match MinuteSummarySubmission::parse(call.args) {
            Ok(submission) => {
                let window = state.window.clone();
                let MinuteSummarySubmission {
                    activity,
                    summary,
                    apps,
                    confidence,
                    uncertain,
                } = submission;
                info!(
                    tool_call_id = %call_id,
                    summary_window_id = window.id,
                    activity = %activity,
                    confidence,
                    uncertain,
                    app_count = apps.len(),
                    "accepted minute summary submission"
                );
                let record = WindowRecord::success(
                    &self.model,
                    call.id.clone(),
                    Utc::now(),
                    window.id,
                    window.started_at,
                    window.ended_at,
                    state.prompt_sent_at,
                    window.observed_frame_count,
                    window.observed_audio_chunk_count,
                    activity,
                    summary,
                    apps,
                    confidence,
                    uncertain,
                );
                self.store.append(&record)?;
                state.accepted_submission = true;
                self.runtime
                    .send_tool_response(vec![ok_response(call.id)])
                    .await?;
            }
            Err(error) => {
                state.note_issue(error.clone());
                warn!(
                    tool_call_id = %call_id,
                    tool_name = %call_name,
                    summary_window_id = state.window.id,
                    error = %error,
                    "rejecting malformed minute summary submission"
                );
                self.runtime
                    .send_tool_response(vec![error_response(call.id, call.name, error)])
                    .await?;
            }
        }
        Ok(())
    }

    async fn handle_summary_timeout(&mut self) -> Result<(), RecorderError> {
        let Some(state) = self.current_summary.take() else {
            return Ok(());
        };

        if !state.accepted_submission {
            self.write_error_record(
                &state.window,
                Some(state.prompt_sent_at),
                format!(
                    "summary turn timed out after {:.1}s; {}",
                    self.summary_timeout.as_secs_f64(),
                    state.missing_submission_reason()
                ),
            )?;
        }
        self.restart_runtime(format!(
            "summary turn timed out for window {}",
            state.window.id
        ))
        .await
    }

    async fn finish_summary_turn(
        &mut self,
        terminal_issue: Option<String>,
    ) -> Result<(), RecorderError> {
        let Some(mut state) = self.current_summary.take() else {
            return Ok(());
        };
        if let Some(issue) = terminal_issue {
            state.note_issue(issue);
        }
        if state.accepted_submission {
            info!(
                summary_window_id = state.window.id,
                issues = state.issues.len(),
                "summary turn completed with an accepted submission"
            );
        } else {
            warn!(
                summary_window_id = state.window.id,
                issues = %state.missing_submission_reason(),
                "summary turn completed without a valid submission"
            );
        }
        if !state.accepted_submission {
            self.write_error_record(
                &state.window,
                Some(state.prompt_sent_at),
                state.missing_submission_reason(),
            )?;
        }
        self.flush_or_continue().await
    }

    async fn flush_or_continue(&mut self) -> Result<(), RecorderError> {
        self.dispatch_next_summary_if_idle().await?;
        if self.current_summary.is_none() && self.pending_summaries.is_empty() {
            if !self.pending_media.is_empty() {
                info!(
                    active_window_id = self.active_window.id,
                    buffered_media_count = self.pending_media.len(),
                    "flushing buffered live media into runtime"
                );
            }
            while let Some(media) = self.pending_media.pop_front() {
                match media {
                    PendingMedia::Frame(frame) => self.send_frame(frame, false).await?,
                    PendingMedia::Audio(audio) => self.send_audio(audio, false).await?,
                }
            }
        }
        Ok(())
    }

    async fn begin_observation_activity_if_needed(&mut self) -> Result<(), RecorderError> {
        if self.observation_activity_open {
            return Ok(());
        }
        self.runtime
            .send_raw(ClientMessage::RealtimeInput(RealtimeInput {
                activity_start: Some(EmptyObject {}),
                ..Default::default()
            }))
            .await?;
        self.observation_activity_open = true;
        info!(
            active_window_id = self.active_window.id,
            "opened manual observation activity"
        );
        Ok(())
    }

    async fn end_observation_activity_if_open(&mut self) -> Result<(), RecorderError> {
        if !self.observation_activity_open {
            return Ok(());
        }
        self.runtime
            .send_raw(ClientMessage::RealtimeInput(RealtimeInput {
                activity_end: Some(EmptyObject {}),
                ..Default::default()
            }))
            .await?;
        self.observation_activity_open = false;
        info!(
            active_window_id = self.active_window.id,
            "closed manual observation activity"
        );
        Ok(())
    }

    async fn send_frame(
        &mut self,
        frame: EncodedFrame,
        is_first_frame: bool,
    ) -> Result<(), RecorderError> {
        self.begin_observation_activity_if_needed().await?;
        self.runtime
            .send_video(&frame.bytes, frame.mime_type)
            .await?;
        if is_first_frame {
            debug!(
                active_window_id = self.active_window.id,
                mime_type = frame.mime_type,
                bytes = frame.bytes.len(),
                "started streaming frames into active observation window"
            );
        }
        Ok(())
    }

    async fn send_audio(
        &mut self,
        audio: CapturedAudio,
        is_first_audio_chunk: bool,
    ) -> Result<(), RecorderError> {
        self.begin_observation_activity_if_needed().await?;
        self.runtime
            .send_audio_at_rate(&audio.pcm_i16_le, audio.sample_rate)
            .await?;
        if is_first_audio_chunk {
            debug!(
                active_window_id = self.active_window.id,
                sample_rate = audio.sample_rate,
                bytes = audio.pcm_i16_le.len(),
                "started streaming system audio into active observation window"
            );
        }
        Ok(())
    }

    async fn dispatch_next_summary_if_idle(&mut self) -> Result<(), RecorderError> {
        if self.current_summary.is_some() {
            return Ok(());
        }

        while let Some(window) = self.pending_summaries.pop_front() {
            let prompt_sent_at = Utc::now();
            let prompt = summary_prompt(
                window.id,
                window.started_at,
                window.ended_at,
                window.observed_frame_count,
                window.observed_audio_chunk_count,
            );
            if let Err(error) = self.end_observation_activity_if_open().await {
                self.write_error_record(
                    &window,
                    None,
                    format!("failed to close observation activity before summary: {error}"),
                )?;
                warn!(
                    "failed to close observation activity for window {}: {error}",
                    window.id
                );
                continue;
            }
            info!(
                summary_window_id = window.id,
                started_at = %window.started_at.to_rfc3339(),
                ended_at = %window.ended_at.to_rfc3339(),
                observed_frame_count = window.observed_frame_count,
                observed_audio_chunk_count = window.observed_audio_chunk_count,
                timeout_secs = self.summary_timeout.as_secs_f64(),
                "sending summary prompt to model"
            );
            debug!(
                summary_window_id = window.id,
                prompt = %prompt,
                "summary prompt body"
            );
            if let Err(error) = self.runtime.send_text(&prompt).await {
                self.write_error_record(
                    &window,
                    Some(prompt_sent_at),
                    format!("failed to send summary prompt: {error}"),
                )?;
                warn!(
                    "failed to send summary prompt for window {}: {error}",
                    window.id
                );
                continue;
            }
            self.current_summary = Some(SummaryTurnState::new(
                window,
                prompt_sent_at,
                Instant::now() + self.summary_timeout,
            ));
            info!(
                summary_window_id = self.current_summary.as_ref().expect("summary state just set").window.id,
                prompt_sent_at = %prompt_sent_at.to_rfc3339(),
                deadline_secs = self.summary_timeout.as_secs_f64(),
                "summary turn started"
            );
            break;
        }
        Ok(())
    }

    fn write_error_record(
        &self,
        window: &ClosedWindow,
        prompt_sent_at: Option<DateTime<Utc>>,
        error: String,
    ) -> Result<(), RecorderError> {
        warn!(
            summary_window_id = window.id,
            started_at = %window.started_at.to_rfc3339(),
            ended_at = %window.ended_at.to_rfc3339(),
            prompt_sent_at = prompt_sent_at.map(|ts| ts.to_rfc3339()),
            observed_frame_count = window.observed_frame_count,
            observed_audio_chunk_count = window.observed_audio_chunk_count,
            error = %error,
            "writing summary error record"
        );
        self.store.append(&WindowRecord::error(
            &self.model,
            Utc::now(),
            window.id,
            window.started_at,
            window.ended_at,
            prompt_sent_at,
            window.observed_frame_count,
            window.observed_audio_chunk_count,
            error,
        ))?;
        Ok(())
    }

    async fn restart_runtime(&mut self, reason: impl AsRef<str>) -> Result<(), RecorderError> {
        let reason = reason.as_ref();
        warn!(
            pending_media = self.pending_media.len(),
            pending_summaries = self.pending_summaries.len(),
            "{reason}; restarting recorder runtime"
        );

        let mut last_error = None;
        for attempt in 1..=RUNTIME_RESTART_ATTEMPTS {
            match self.runtime.connect().await {
                Ok(()) => {
                    self.observation_activity_open = false;
                    info!(attempt, "{reason}; runtime restarted");
                    self.flush_or_continue().await?;
                    return Ok(());
                }
                Err(error) => {
                    warn!(
                        attempt,
                        max_attempts = RUNTIME_RESTART_ATTEMPTS,
                        "failed to restart runtime after {reason}: {error}"
                    );
                    last_error = Some(error);
                    if attempt < RUNTIME_RESTART_ATTEMPTS {
                        tokio::time::sleep(RUNTIME_RESTART_BACKOFF).await;
                    }
                }
            }
        }

        Err(last_error
            .expect("restart attempts should capture the last error")
            .into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    if cli.list_targets {
        print_targets()?;
        return Ok(());
    }

    let args = validate_run_args(cli.run)?;
    let selected_monitors = resolve_monitor_targets(&args)?;
    let (screen_tx, screen_rx) = mpsc::channel::<EncodedFrame>(32);
    let (audio_tx, audio_rx) = mpsc::channel::<CapturedAudio>(64);
    let mut captures = Vec::with_capacity(selected_monitors.len());
    for monitor in &selected_monitors {
        captures.push(ScreenCapture::start(
            monitor.id,
            ScreenCaptureConfig {
                interval: Duration::from_secs_f64(args.frame_interval_secs),
                max_dimension: args.max_dimension,
                jpeg_quality: args.jpeg_quality,
            },
            screen_tx.clone(),
        )?);
    }
    drop(screen_tx);
    let system_audio_capture = SystemAudioCapture::start(audio_tx)?;

    info!(
        monitor_count = selected_monitors.len(),
        output = %args.output.display(),
        system_audio_sample_rate = INPUT_SAMPLE_RATE,
        frame_interval_secs = args.frame_interval_secs,
        summary_interval_secs = args.summary_interval_secs,
        "starting recorder"
    );
    for monitor in &selected_monitors {
        info!(
            monitor_id = monitor.id,
            monitor_name = %monitor.name,
            width = monitor.width,
            height = monitor.height,
            "capturing monitor"
        );
    }
    info!(
        audio_device = %system_audio_capture.device_name,
        input_sample_rate = system_audio_capture.input_sample_rate,
        output_sample_rate = system_audio_capture.output_sample_rate,
        "capturing system audio"
    );

    let app = RecorderApp::new(&args).await?;
    app.run(
        captures,
        system_audio_capture,
        screen_rx,
        audio_rx,
        Duration::from_secs_f64(args.summary_interval_secs),
    )
    .await?;
    Ok(())
}

fn validate_run_args(mut args: RunArgs) -> Result<RunArgs, RecorderError> {
    if args.frame_interval_secs <= 0.0 {
        return Err(RecorderError::InvalidFrameInterval);
    }
    if args.summary_interval_secs <= 0.0 {
        return Err(RecorderError::InvalidSummaryInterval);
    }
    if args.summary_timeout_secs <= 0.0 {
        return Err(RecorderError::InvalidSummaryTimeout);
    }
    if args.api_key.is_none() {
        args.api_key = std::env::var("GEMINI_API_KEY").ok();
    }
    Ok(args)
}

fn resolve_monitor_targets(args: &RunArgs) -> Result<Vec<CaptureTarget>, RecorderError> {
    let monitors = list_monitor_targets()?;
    if monitors.is_empty() {
        return Err(RecorderError::NoMonitorsAvailable);
    }
    if args.monitor_ids.is_empty() {
        return Ok(monitors);
    }

    let mut selected = Vec::with_capacity(args.monitor_ids.len());
    for requested_id in &args.monitor_ids {
        let Some(target) = monitors.iter().find(|target| target.id == *requested_id) else {
            return Err(RecorderError::MonitorIdNotFound(*requested_id));
        };
        selected.push(target.clone());
    }
    Ok(selected)
}

fn build_setup(model: &str, extra_instruction: Option<&str>) -> SetupConfig {
    SetupConfig {
        model: model.to_string(),
        generation_config: Some(GenerationConfig {
            response_modalities: Some(vec![Modality::Audio]),
            thinking_config: Some(ThinkingConfig {
                thinking_level: Some(ThinkingLevel::High),
                ..Default::default()
            }),
            media_resolution: Some(MediaResolution::MediaResolutionHigh),
            ..Default::default()
        }),
        system_instruction: Some(system_instruction_content(&system_instruction(
            extra_instruction,
        ))),
        realtime_input_config: Some(RealtimeInputConfig {
            automatic_activity_detection: Some(AutomaticActivityDetection {
                disabled: Some(true),
                ..Default::default()
            }),
            activity_handling: Some(ActivityHandling::NoInterruption),
            ..Default::default()
        }),
        input_audio_transcription: Some(AudioTranscriptionConfig::default()),
        output_audio_transcription: Some(AudioTranscriptionConfig::default()),
        session_resumption: Some(SessionResumptionConfig::default()),
        tools: Some(vec![Tool::FunctionDeclarations(vec![declaration()])]),
        ..Default::default()
    }
}

fn system_instruction_content(text: &str) -> Content {
    Content {
        role: None,
        parts: vec![Part {
            text: Some(text.to_string()),
            inline_data: None,
        }],
    }
}

fn print_targets() -> Result<(), RecorderError> {
    let targets = list_targets()?;
    if targets.is_empty() {
        println!("No capture targets found.");
        return Ok(());
    }
    for target in targets {
        println!(
            "{}\t[{}]\t{}\t{}x{}",
            target.id, target.kind, target.name, target.width, target.height
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_positive_intervals() {
        let result = validate_run_args(RunArgs {
            api_key: Some("key".into()),
            model: DEFAULT_MODEL.into(),
            monitor_ids: Vec::new(),
            frame_interval_secs: 0.0,
            summary_interval_secs: 60.0,
            summary_timeout_secs: 20.0,
            max_dimension: 1280,
            jpeg_quality: 75,
            output: PathBuf::from(DEFAULT_OUTPUT_PATH),
            extra_instruction: None,
        });

        assert!(matches!(result, Err(RecorderError::InvalidFrameInterval)));
    }

    #[test]
    fn build_setup_exposes_one_summary_tool() {
        let setup = build_setup(DEFAULT_MODEL, None);
        let tools = setup.tools.expect("tools");
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            Tool::FunctionDeclarations(functions) => {
                assert_eq!(functions.len(), 1);
                assert_eq!(functions[0].name, SUBMIT_MINUTE_SUMMARY_TOOL);
            }
            other => panic!("unexpected tool variant: {other:?}"),
        }
    }

    #[test]
    fn build_setup_uses_manual_activity_boundaries() {
        let setup = build_setup(DEFAULT_MODEL, None);
        let realtime_input = setup.realtime_input_config.expect("realtime input config");
        let activity_detection = realtime_input
            .automatic_activity_detection
            .expect("automatic activity detection config");
        assert_eq!(activity_detection.disabled, Some(true));
        assert!(matches!(
            realtime_input.activity_handling,
            Some(ActivityHandling::NoInterruption)
        ));
    }
}
