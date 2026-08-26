//! TUI rendering for transcribe mode.
//!
//! Same three-band layout as the chat view (transcript / completion popup /
//! input): terminal lifecycle stays in `crate::render`, this module only maps
//! `TranscribeApp` state to widgets.
//!
//! Visual language:
//!
//! - finalized lines: dim `[mm:ss]` prefix + default text
//! - the in-progress line: `›` prefix, finalized fragments in default style,
//!   the interim hypothesis appended in dark-gray italics (the same "pending"
//!   gray the chat view uses for streaming model output)
//! - notices: dim, matching chat's `Role::System`

use std::time::Duration;

use gemini_live::session::SessionStatus;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::state::{SESSION_STREAM_CAP, TranscribeApp, TranscriptEntry};

pub(crate) fn draw(frame: &mut ratatui::Frame, app: &mut TranscribeApp) {
    let completion_height = if app.has_completions() {
        app.completion_count() as u16 + 2
    } else {
        0
    };
    let [transcript_area, completion_area, input_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(completion_height),
            Constraint::Length(3),
        ])
        .areas(frame.area());

    let lines = transcript_lines(app);
    let wrapped_lines = total_wrapped_lines(&lines, transcript_area.width);
    let visible = transcript_area.height.saturating_sub(2) as usize;
    let scroll = wrapped_lines.saturating_sub(visible) as u16;

    let transcript = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" transcribe — {} ", app.model)),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(transcript, transcript_area);

    if app.has_completions() {
        let completion = Paragraph::new(Text::from(completion_lines(app)))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" completions: Tab accept, Up/Down select "),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(completion, completion_area);
    }

    let status = status_line(app);
    let input_widget = app.input.render_widget(status);
    frame.render_widget(input_widget, input_area);
}

fn transcript_lines(app: &TranscribeApp) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for entry in &app.entries {
        match entry {
            TranscriptEntry::Line { at, text } => lines.push(Line::from(vec![
                Span::styled(
                    format!("[{}] ", format_mm_ss(*at)),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::raw(text.clone()),
            ])),
            TranscriptEntry::Notice(text) => lines.push(Line::from(Span::styled(
                format!("  {text}"),
                Style::default().fg(Color::DarkGray),
            ))),
        }
    }

    if !app.current_line.is_empty() || !app.interim.is_empty() {
        let mut spans = vec![Span::styled(
            "› ".to_string(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        if !app.current_line.is_empty() {
            spans.push(Span::raw(app.current_line.clone()));
        }
        if !app.interim.is_empty() {
            spans.push(Span::styled(
                app.interim.clone(),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ));
        }
        lines.push(Line::from(spans));
    }

    lines
}

fn total_wrapped_lines(lines: &[Line<'_>], area_width: u16) -> usize {
    let content_width = area_width.saturating_sub(2) as usize;
    lines
        .iter()
        .map(|line| {
            let line_width: usize = line.spans.iter().map(|span| span.content.len()).sum();
            line_width
                .checked_div(content_width)
                .map_or(1, |wraps| wraps + 1)
        })
        .sum()
}

fn completion_lines(app: &TranscribeApp) -> Vec<Line<'static>> {
    app.completion_items()
        .iter()
        .take(app.completion_count())
        .enumerate()
        .map(|(idx, item)| {
            let selected = idx == app.selected_completion_index();
            let marker = if selected { "› " } else { "  " };
            let marker_style = if selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            let label_style = if selected {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::Cyan)
            };
            Line::from(vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(item.label.clone(), label_style),
                Span::raw(" "),
                Span::styled(item.detail.clone(), Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect()
}

pub(crate) fn status_line(app: &TranscribeApp) -> String {
    let connection = match app.connection {
        SessionStatus::Connecting => "connecting",
        SessionStatus::Connected => "connected",
        SessionStatus::Reconnecting => "reconnecting",
        SessionStatus::Closed => "closed",
    };
    let mode = app
        .active
        .mode
        .map(|mode| mode.as_str())
        .unwrap_or("verbatim");
    let vad = if app.active.manual_vad {
        if app.manual_activity_open {
            "manual (SPEAKING)"
        } else {
            "manual (idle)"
        }
    } else {
        "auto"
    };
    let lang = if app.active.languages.is_empty() {
        "auto".to_string()
    } else {
        app.active.languages.join(",")
    };

    let mut status = format!(
        " {connection} | src: {} | mode: {mode} | vad: {vad} | lang: {lang}",
        app.source
    );
    if let Some(elapsed) = app.session_elapsed() {
        status.push_str(&format!(
            " | {}/{}",
            format_mm_ss(elapsed),
            format_mm_ss(SESSION_STREAM_CAP)
        ));
    }
    if let Some(output) = app.output_label.as_deref() {
        status.push_str(&format!(" | out: {output}"));
    }
    if app.has_staged_changes() {
        status.push_str(" | staged* (/apply)");
    }
    if app.lagged_events > 0 {
        status.push_str(&format!(" | lagged: {}", app.lagged_events));
    }
    status.push(' ');
    status
}

fn format_mm_ss(duration: Duration) -> String {
    let total = duration.as_secs();
    format!("{:02}:{:02}", total / 60, total % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::audio::AudioSourceKind;
    use crate::transcribe::startup::TranscribeSettings;

    #[test]
    fn status_line_reflects_state() {
        let mut app = TranscribeApp::new(
            "models/test".into(),
            TranscribeSettings {
                manual_vad: true,
                ..Default::default()
            },
            AudioSourceKind::System,
            Some("notes.txt".into()),
        );
        app.staged.vocabulary.push("term".into());
        app.lagged_events = 2;

        let status = status_line(&app);
        assert!(status.contains("src: system"));
        assert!(status.contains("vad: manual (idle)"));
        assert!(status.contains("lang: auto"));
        assert!(status.contains("out: notes.txt"));
        assert!(status.contains("staged*"));
        assert!(status.contains("lagged: 2"));
    }

    #[test]
    fn mm_ss_formats_minutes_and_seconds() {
        assert_eq!(format_mm_ss(Duration::from_secs(0)), "00:00");
        assert_eq!(format_mm_ss(Duration::from_secs(192)), "03:12");
        assert_eq!(format_mm_ss(SESSION_STREAM_CAP), "10:00");
    }
}
