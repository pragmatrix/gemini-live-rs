//! Transcribe-mode startup resolution and session-template construction.
//!
//! This module is the canonical home for how `gemini-live transcribe` turns:
//!
//! - CLI flags
//! - environment variables
//! - the persisted `transcribe` profile section
//! - built-in defaults
//!
//! into a Live Transcribe `SetupConfig`. Precedence is
//! `CLI flag > environment > profile > built-in default` — one step stronger
//! than chat startup because transcribe exposes its settings as flags.
//!
//! The transcribe session template is deliberately minimal (only fields in
//! the documented Live Transcribe setup):
//!
//! - `responseModalities = ["TEXT"]`
//! - `inputAudioTranscription = { languageCodes?, customVocabulary?, mode? }`
//! - `realtimeInputConfig.automaticActivityDetection.disabled = true` only
//!   when manual VAD is selected
//!
//! No tools, no system instruction, and no session resumption: the 10-minute
//! streaming cap means reconnects are always fresh handshakes.

use gemini_live::session::{ReconnectPolicy, SessionConfig};
use gemini_live::transport::TransportConfig;
use gemini_live::types::{
    AudioTranscriptionConfig, AutomaticActivityDetection, GenerationConfig, Modality,
    RealtimeInputConfig, SetupConfig, TranscriptionMode,
};

use crate::profile;
use crate::startup::CliConfigError;

use super::TranscribeArgs;
use super::audio::AudioSourceKind;

pub(crate) const DEFAULT_TRANSCRIBE_MODEL: &str = "models/gemini-3.5-transcribe-live";

/// Upstream limit on `customVocabulary` terms.
pub(crate) const MAX_VOCABULARY_TERMS: usize = 1000;

/// User-adjustable transcription settings.
///
/// The TUI keeps two copies: the `active` settings the current session was
/// connected with, and the `staged` settings edited by slash commands and
/// applied on the next reconnect (`/apply`).
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct TranscribeSettings {
    /// BCP-47 language codes; empty = automatic language detection.
    pub(crate) languages: Vec<String>,
    /// Custom vocabulary bias terms (≤ [`MAX_VOCABULARY_TERMS`]).
    pub(crate) vocabulary: Vec<String>,
    /// Transcript style; `None` lets the server default to `VERBATIM`.
    pub(crate) mode: Option<TranscriptionMode>,
    /// `true` disables server VAD — the user marks segments with
    /// `/start` and `/end`.
    pub(crate) manual_vad: bool,
}

impl TranscribeSettings {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.vocabulary.len() > MAX_VOCABULARY_TERMS {
            return Err(format!(
                "custom vocabulary holds {} terms; the API caps it at {MAX_VOCABULARY_TERMS}",
                self.vocabulary.len()
            ));
        }
        Ok(())
    }
}

/// Fully resolved transcribe-mode startup state.
#[derive(Debug, Clone)]
pub(crate) struct TranscribeStartup {
    pub(crate) model: String,
    pub(crate) transport: TransportConfig,
    pub(crate) settings: TranscribeSettings,
    pub(crate) source: AudioSourceKind,
}

impl TranscribeStartup {
    pub(crate) fn session_config(&self) -> SessionConfig {
        SessionConfig {
            transport: self.transport.clone(),
            setup: build_transcribe_setup(&self.settings, &self.model),
            reconnect: ReconnectPolicy::default(),
        }
    }
}

/// Resolve transcribe startup from flags, env, and the stored profile.
///
/// The transport (backend + credentials) reuses the chat resolution via
/// [`crate::startup::resolve_startup_config`] — callers pass the resolved
/// `transport` in, so `GEMINI_API_KEY` handling stays in one place.
pub(crate) fn resolve_transcribe_startup(
    args: &TranscribeArgs,
    env: impl Fn(&str) -> Option<String>,
    stored_profile: &profile::ProfileConfig,
    transport: TransportConfig,
) -> Result<TranscribeStartup, CliConfigError> {
    let stored = stored_profile.transcribe.clone().unwrap_or_default();

    let model = args
        .model
        .clone()
        .or_else(|| env("GEMINI_TRANSCRIBE_MODEL"))
        .filter(|value| !value.trim().is_empty())
        .or(stored.model)
        .unwrap_or_else(|| DEFAULT_TRANSCRIBE_MODEL.into());

    let mode = match &args.mode {
        Some(raw) => Some(
            raw.parse::<TranscriptionMode>()
                .map_err(|error| CliConfigError::new(format!("--mode {error}")))?,
        ),
        None => stored.mode,
    };

    let languages = if args.languages.is_empty() {
        stored.language_codes.unwrap_or_default()
    } else {
        args.languages.clone()
    };

    let vocabulary = if args.vocabulary.is_empty() {
        stored.custom_vocabulary.unwrap_or_default()
    } else {
        args.vocabulary.clone()
    };

    let manual_vad = if args.manual_vad {
        true
    } else {
        // The profile stores the positive setting (automatic detection on?).
        !stored.automatic_activity_detection.unwrap_or(true)
    };

    let source = match &args.source {
        Some(raw) => raw
            .parse::<AudioSourceKind>()
            .map_err(|error| CliConfigError::new(format!("--source {error}")))?,
        None => match stored.source {
            Some(profile::TranscribeSource::System) => AudioSourceKind::System,
            Some(profile::TranscribeSource::Mic) | None => AudioSourceKind::Mic,
        },
    };

    let settings = TranscribeSettings {
        languages,
        vocabulary,
        mode,
        manual_vad,
    };
    settings.validate().map_err(CliConfigError::new)?;

    Ok(TranscribeStartup {
        model,
        transport,
        settings,
        source,
    })
}

/// Build the minimal Live Transcribe setup message.
pub(crate) fn build_transcribe_setup(settings: &TranscribeSettings, model: &str) -> SetupConfig {
    SetupConfig {
        model: model.to_string(),
        generation_config: Some(GenerationConfig {
            response_modalities: Some(vec![Modality::Text]),
            ..Default::default()
        }),
        input_audio_transcription: Some(AudioTranscriptionConfig {
            language_codes: non_empty(&settings.languages),
            custom_vocabulary: non_empty(&settings.vocabulary),
            mode: settings.mode,
        }),
        realtime_input_config: settings.manual_vad.then(|| RealtimeInputConfig {
            automatic_activity_detection: Some(AutomaticActivityDetection {
                disabled: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Snapshot the given state into the persisted `transcribe` profile section.
pub(crate) fn persisted_transcribe_profile(
    model: &str,
    settings: &TranscribeSettings,
    source: AudioSourceKind,
) -> profile::TranscribeProfile {
    profile::TranscribeProfile {
        model: Some(model.to_string()),
        mode: settings.mode,
        language_codes: Some(settings.languages.clone()),
        custom_vocabulary: Some(settings.vocabulary.clone()),
        automatic_activity_detection: Some(!settings.manual_vad),
        source: Some(match source {
            AudioSourceKind::Mic => profile::TranscribeSource::Mic,
            AudioSourceKind::System => profile::TranscribeSource::System,
        }),
    }
}

fn non_empty(values: &[String]) -> Option<Vec<String>> {
    if values.is_empty() {
        None
    } else {
        Some(values.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemini_live::transport::{Auth, Endpoint};

    fn test_transport() -> TransportConfig {
        TransportConfig {
            endpoint: Endpoint::GeminiApi,
            auth: Auth::ApiKey("test-key".into()),
            ..Default::default()
        }
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn empty_args() -> TranscribeArgs {
        TranscribeArgs {
            output: None,
            languages: Vec::new(),
            mode: None,
            vocabulary: Vec::new(),
            manual_vad: false,
            source: None,
            model: None,
        }
    }

    #[test]
    fn resolves_built_in_defaults() {
        let startup = resolve_transcribe_startup(
            &empty_args(),
            no_env,
            &profile::ProfileConfig::default(),
            test_transport(),
        )
        .expect("startup");
        assert_eq!(startup.model, DEFAULT_TRANSCRIBE_MODEL);
        assert_eq!(startup.settings, TranscribeSettings::default());
        assert_eq!(startup.source, AudioSourceKind::Mic);
    }

    #[test]
    fn flags_override_env_and_profile() {
        let mut args = empty_args();
        args.model = Some("models/custom".into());
        args.mode = Some("smart".into());
        args.languages = vec!["zh-TW".into()];
        args.manual_vad = true;
        args.source = Some("system".into());

        let stored = profile::ProfileConfig {
            transcribe: Some(profile::TranscribeProfile {
                model: Some("models/from-profile".into()),
                mode: Some(TranscriptionMode::Verbatim),
                language_codes: Some(vec!["en-US".into()]),
                custom_vocabulary: Some(vec!["serde".into()]),
                automatic_activity_detection: Some(true),
                source: Some(profile::TranscribeSource::Mic),
            }),
            ..Default::default()
        };

        let env =
            |key: &str| (key == "GEMINI_TRANSCRIBE_MODEL").then(|| "models/from-env".to_string());

        let startup =
            resolve_transcribe_startup(&args, env, &stored, test_transport()).expect("startup");
        assert_eq!(startup.model, "models/custom");
        assert_eq!(startup.settings.mode, Some(TranscriptionMode::Smart));
        assert_eq!(startup.settings.languages, vec!["zh-TW".to_string()]);
        // Vocabulary was not passed as a flag, so the profile value holds.
        assert_eq!(startup.settings.vocabulary, vec!["serde".to_string()]);
        assert!(startup.settings.manual_vad);
        assert_eq!(startup.source, AudioSourceKind::System);
    }

    #[test]
    fn env_model_beats_profile_model() {
        let stored = profile::ProfileConfig {
            transcribe: Some(profile::TranscribeProfile {
                model: Some("models/from-profile".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let env =
            |key: &str| (key == "GEMINI_TRANSCRIBE_MODEL").then(|| "models/from-env".to_string());
        let startup = resolve_transcribe_startup(&empty_args(), env, &stored, test_transport())
            .expect("startup");
        assert_eq!(startup.model, "models/from-env");
    }

    #[test]
    fn profile_manual_vad_round_trips() {
        let stored = profile::ProfileConfig {
            transcribe: Some(profile::TranscribeProfile {
                automatic_activity_detection: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        let startup = resolve_transcribe_startup(&empty_args(), no_env, &stored, test_transport())
            .expect("startup");
        assert!(startup.settings.manual_vad);

        let persisted =
            persisted_transcribe_profile(&startup.model, &startup.settings, startup.source);
        assert_eq!(persisted.automatic_activity_detection, Some(false));
    }

    #[test]
    fn rejects_oversized_vocabulary() {
        let mut args = empty_args();
        args.vocabulary = (0..=MAX_VOCABULARY_TERMS)
            .map(|i| format!("term-{i}"))
            .collect();
        let error = resolve_transcribe_startup(
            &args,
            no_env,
            &profile::ProfileConfig::default(),
            test_transport(),
        )
        .expect_err("should reject oversized vocabulary");
        assert!(error.to_string().contains("caps it at"));
    }

    #[test]
    fn setup_uses_text_modality_and_omits_empty_fields() {
        let setup = build_transcribe_setup(&TranscribeSettings::default(), "models/m");
        let generation = setup.generation_config.expect("generation config");
        assert_eq!(generation.response_modalities, Some(vec![Modality::Text]));
        let transcription = setup
            .input_audio_transcription
            .expect("input transcription");
        assert_eq!(transcription, AudioTranscriptionConfig::default());
        assert!(setup.realtime_input_config.is_none());
        assert!(setup.tools.is_none());
        assert!(setup.session_resumption.is_none());
        assert!(setup.output_audio_transcription.is_none());
    }

    #[test]
    fn setup_disables_vad_only_in_manual_mode() {
        let settings = TranscribeSettings {
            manual_vad: true,
            ..Default::default()
        };
        let setup = build_transcribe_setup(&settings, "models/m");
        let vad = setup
            .realtime_input_config
            .expect("realtime input config")
            .automatic_activity_detection
            .expect("automatic activity detection");
        assert_eq!(vad.disabled, Some(true));
    }
}
