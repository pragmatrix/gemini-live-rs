//! Slash-command grammar and completion model for transcribe mode.
//!
//! Same dual-structure discipline as the chat grammar in `crate::slash`:
//! `clap` owns the accepted command grammar while `command_specs()` feeds the
//! completion popup. The `specs_stay_in_sync_with_grammar` test keeps the two
//! catalogs aligned.

use std::ops::Range;

use clap::{Parser, Subcommand, ValueEnum};
use gemini_live::types::TranscriptionMode;

use super::audio::AudioSourceKind;

/// Reuse the chat CLI's completion item shape so the render path is shared.
pub(crate) use crate::slash::CompletionItem;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TranscribeSlashCommand {
    /// `/mode` shows, `/mode verbatim|smart` stages.
    Mode(Option<TranscriptionMode>),
    /// `/lang` shows, `/lang clear` stages auto-detect, `/lang codes…` stages.
    Lang(LangCommand),
    /// `/vocab [list]` shows, `add`/`remove`/`clear` stage edits.
    Vocab(VocabCommand),
    /// `/vad` shows, `/vad on|off` stages automatic activity detection.
    Vad(Option<bool>),
    /// Reconnect with the staged settings (or just reconnect when unchanged).
    Apply,
    /// `/source` shows, `/source mic|system` switches immediately.
    Source(Option<AudioSourceKind>),
    /// Manual activity start (manual VAD only).
    Start,
    /// Manual activity end (manual VAD only).
    End,
    /// Clear the transcript pane (the `--output` file is untouched).
    Clear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LangCommand {
    Show,
    Set(Vec<String>),
    Clear,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VocabCommand {
    List,
    Add(Vec<String>),
    Remove(String),
    Clear,
}

pub(crate) fn parse(input: &str) -> Option<Result<TranscribeSlashCommand, String>> {
    let trimmed = input.trim();
    if !trimmed.starts_with('/') {
        return None;
    }

    let Some(mut argv) = shlex::split(trimmed) else {
        return Some(Err("invalid quoting in slash command".into()));
    };
    if argv.is_empty() {
        return Some(Err("empty slash command".into()));
    }
    argv[0] = argv[0].trim_start_matches('/').to_string();

    Some(
        CliTranscribeSlash::try_parse_from(std::iter::once("slash".to_string()).chain(argv))
            .map(Into::into)
            .map_err(|e| e.to_string().trim().to_string()),
    )
}

#[derive(Debug, Parser)]
#[command(
    name = "slash",
    disable_help_flag = true,
    disable_help_subcommand = true
)]
struct CliTranscribeSlash {
    #[command(subcommand)]
    command: CliTranscribeCommand,
}

#[derive(Debug, Subcommand)]
enum CliTranscribeCommand {
    #[command(name = "mode")]
    Mode { mode: Option<CliModeArg> },
    #[command(name = "lang")]
    Lang { codes: Vec<String> },
    #[command(name = "vocab")]
    Vocab {
        #[command(subcommand)]
        action: Option<CliVocabCommand>,
    },
    #[command(name = "vad")]
    Vad { state: Option<CliOnOffArg> },
    #[command(name = "apply")]
    Apply,
    #[command(name = "source")]
    Source { source: Option<CliSourceArg> },
    #[command(name = "start")]
    Start,
    #[command(name = "end")]
    End,
    #[command(name = "clear")]
    Clear,
}

#[derive(Debug, Subcommand)]
enum CliVocabCommand {
    List,
    Add { terms: Vec<String> },
    Remove { term: String },
    Clear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliModeArg {
    Verbatim,
    Smart,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliOnOffArg {
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliSourceArg {
    Mic,
    System,
}

impl From<CliTranscribeSlash> for TranscribeSlashCommand {
    fn from(value: CliTranscribeSlash) -> Self {
        match value.command {
            CliTranscribeCommand::Mode { mode } => Self::Mode(mode.map(|mode| match mode {
                CliModeArg::Verbatim => TranscriptionMode::Verbatim,
                CliModeArg::Smart => TranscriptionMode::Smart,
            })),
            CliTranscribeCommand::Lang { codes } => Self::Lang(if codes.is_empty() {
                LangCommand::Show
            } else if codes.len() == 1 && codes[0].eq_ignore_ascii_case("clear") {
                LangCommand::Clear
            } else {
                LangCommand::Set(codes)
            }),
            CliTranscribeCommand::Vocab { action } => Self::Vocab(match action {
                None | Some(CliVocabCommand::List) => VocabCommand::List,
                Some(CliVocabCommand::Add { terms }) => VocabCommand::Add(terms),
                Some(CliVocabCommand::Remove { term }) => VocabCommand::Remove(term),
                Some(CliVocabCommand::Clear) => VocabCommand::Clear,
            }),
            CliTranscribeCommand::Vad { state } => Self::Vad(state.map(|state| match state {
                CliOnOffArg::On => true,
                CliOnOffArg::Off => false,
            })),
            CliTranscribeCommand::Apply => Self::Apply,
            CliTranscribeCommand::Source { source } => {
                Self::Source(source.map(|source| match source {
                    CliSourceArg::Mic => AudioSourceKind::Mic,
                    CliSourceArg::System => AudioSourceKind::System,
                }))
            }
            CliTranscribeCommand::Start => Self::Start,
            CliTranscribeCommand::End => Self::End,
            CliTranscribeCommand::Clear => Self::Clear,
        }
    }
}

// ── Completions ──────────────────────────────────────────────────────────────

struct CommandSpec {
    name: &'static str,
    detail: &'static str,
}

fn command_specs() -> [CommandSpec; 9] {
    [
        CommandSpec {
            name: "/mode",
            detail: "show or stage the transcript style (verbatim | smart)",
        },
        CommandSpec {
            name: "/lang",
            detail: "show or stage language codes (clear = auto-detect)",
        },
        CommandSpec {
            name: "/vocab",
            detail: "list or stage custom vocabulary (add/remove/clear)",
        },
        CommandSpec {
            name: "/vad",
            detail: "show or stage automatic activity detection (on | off)",
        },
        CommandSpec {
            name: "/apply",
            detail: "reconnect with the staged settings",
        },
        CommandSpec {
            name: "/source",
            detail: "show or switch the audio source (mic | system)",
        },
        CommandSpec {
            name: "/start",
            detail: "mark speech start (manual VAD)",
        },
        CommandSpec {
            name: "/end",
            detail: "mark speech end (manual VAD)",
        },
        CommandSpec {
            name: "/clear",
            detail: "clear the transcript pane (output file untouched)",
        },
    ]
}

/// Argument completions per command; `(value, detail)` pairs offered after
/// the command name.
fn argument_specs(command: &str) -> &'static [(&'static str, &'static str)] {
    match command {
        "mode" => &[
            ("verbatim", "literal transcript (server default)"),
            (
                "smart",
                "cleaned transcript: disfluency removal, formatting",
            ),
        ],
        "vad" => &[
            ("on", "server voice-activity detection"),
            ("off", "manual segments via /start and /end"),
        ],
        "source" => &[
            ("mic", "default input device through AEC"),
            ("system", "system output loopback"),
        ],
        "vocab" => &[
            ("list", "show active and staged vocabulary"),
            ("add", "stage new bias terms"),
            ("remove", "stage removal of one term"),
            ("clear", "stage an empty vocabulary"),
        ],
        "lang" => &[("clear", "stage automatic language detection")],
        _ => &[],
    }
}

pub(crate) fn completions(input: &str) -> Vec<CompletionItem> {
    let left_trimmed = input.trim_start();
    let leading_offset = input.len() - left_trimmed.len();
    if !left_trimmed.starts_with('/') {
        return Vec::new();
    }

    let trailing_space = input.chars().last().is_some_and(char::is_whitespace);
    let current_range = current_token_range(input);
    let current_fragment = if trailing_space {
        ""
    } else {
        &input[current_range.clone()]
    };
    let parts = left_trimmed.split_whitespace().collect::<Vec<_>>();
    if parts.is_empty() {
        return command_completions("/", leading_offset..input.len(), true);
    }

    let command = parts[0].trim_start_matches('/');
    let recognized = command_specs()
        .iter()
        .any(|spec| spec.name.trim_start_matches('/') == command);

    if parts.len() == 1 && !recognized {
        return command_completions(current_fragment, current_range, false);
    }

    let arguments = argument_specs(command);
    if arguments.is_empty() {
        return Vec::new();
    }

    if parts.len() == 1 {
        let insert_at = input.len()..input.len();
        return arguments
            .iter()
            .map(|(value, detail)| CompletionItem {
                label: (*value).to_string(),
                replacement: format!(" {value}"),
                replace_range: insert_at.clone(),
                detail: (*detail).to_string(),
            })
            .collect();
    }

    if parts.len() == 2 && !trailing_space {
        return arguments
            .iter()
            .filter(|(value, _)| value.starts_with(current_fragment))
            .map(|(value, detail)| CompletionItem {
                label: (*value).to_string(),
                replacement: (*value).to_string(),
                replace_range: current_range.clone(),
                detail: (*detail).to_string(),
            })
            .collect();
    }

    Vec::new()
}

fn command_completions(
    prefix: &str,
    replace_range: Range<usize>,
    append_space: bool,
) -> Vec<CompletionItem> {
    command_specs()
        .into_iter()
        .filter(|spec| spec.name.starts_with(prefix))
        .map(|spec| CompletionItem {
            label: spec.name.to_string(),
            replacement: if append_space {
                format!("{} ", spec.name)
            } else {
                spec.name.to_string()
            },
            replace_range: replace_range.clone(),
            detail: spec.detail.to_string(),
        })
        .collect()
}

fn current_token_range(input: &str) -> Range<usize> {
    let start = input
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .unwrap_or(0);
    start..input.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_stay_in_sync_with_grammar() {
        // Every completion spec must parse as a valid command on its own.
        for spec in command_specs() {
            let parsed = parse(spec.name).expect("slash command");
            assert!(
                parsed.is_ok(),
                "completion spec {} does not parse: {:?}",
                spec.name,
                parsed
            );
        }
    }

    #[test]
    fn parse_mode_defaults_to_show() {
        let command = parse("/mode").expect("slash").expect("valid");
        assert_eq!(command, TranscribeSlashCommand::Mode(None));
    }

    #[test]
    fn parse_mode_smart() {
        let command = parse("/mode smart").expect("slash").expect("valid");
        assert_eq!(
            command,
            TranscribeSlashCommand::Mode(Some(TranscriptionMode::Smart))
        );
    }

    #[test]
    fn parse_lang_variants() {
        assert_eq!(
            parse("/lang").expect("slash").expect("valid"),
            TranscribeSlashCommand::Lang(LangCommand::Show)
        );
        assert_eq!(
            parse("/lang clear").expect("slash").expect("valid"),
            TranscribeSlashCommand::Lang(LangCommand::Clear)
        );
        assert_eq!(
            parse("/lang en-US zh-TW").expect("slash").expect("valid"),
            TranscribeSlashCommand::Lang(LangCommand::Set(vec!["en-US".into(), "zh-TW".into()]))
        );
    }

    #[test]
    fn parse_vocab_add_with_quoted_term() {
        let command = parse("/vocab add \"Gemini Live\" serde")
            .expect("slash")
            .expect("valid");
        assert_eq!(
            command,
            TranscribeSlashCommand::Vocab(VocabCommand::Add(vec![
                "Gemini Live".into(),
                "serde".into()
            ]))
        );
    }

    #[test]
    fn parse_vad_off() {
        let command = parse("/vad off").expect("slash").expect("valid");
        assert_eq!(command, TranscribeSlashCommand::Vad(Some(false)));
    }

    #[test]
    fn parse_source_system() {
        let command = parse("/source system").expect("slash").expect("valid");
        assert_eq!(
            command,
            TranscribeSlashCommand::Source(Some(AudioSourceKind::System))
        );
    }

    #[test]
    fn parse_rejects_unknown_command() {
        let result = parse("/definitely-not-a-command").expect("slash");
        assert!(result.is_err());
    }

    #[test]
    fn complete_partial_command_name() {
        let items = completions("/vo");
        assert!(items.iter().any(|item| item.label == "/vocab"));
    }

    #[test]
    fn complete_mode_arguments() {
        let items = completions("/mode ");
        assert!(items.iter().any(|item| item.label == "smart"));
        let items = completions("/mode ver");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "verbatim");
    }

    #[test]
    fn complete_vocab_subcommands() {
        let items = completions("/vocab a");
        assert!(items.iter().any(|item| item.label == "add"));
    }
}
