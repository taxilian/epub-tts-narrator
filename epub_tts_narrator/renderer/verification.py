"""Audio-only recognition with optional second recognition and JEV confirmation."""

from .quality import MlxWhisperVerifier


class ReferenceCorroboratingVerifier:
    """Keep timed audiobook verification separate from reference rechecking."""

    def __init__(self, primary, reference):
        self.primary = primary
        self.reference = reference
        self.corroboration_identity = reference.identity
        self.identity = {
            **primary.identity,
            "reference_corroborator": reference.identity,
        }

    def transcribe(self, path):
        return self.primary.transcribe(path)

    def transcribe_reference(self, path):
        return self.primary.transcribe_reference(path)

    def corroborate_reference(self, path):
        return self.reference.transcribe_reference(path)

    def close(self):
        for verifier in (self.primary, self.reference):
            close = getattr(verifier, "close", None)
            if callable(close):
                close()


class ReferenceTextJudgingVerifier:
    """Add a text-only semantic gate without changing either recognizer's input."""

    def __init__(self, verifier, judge, phonetic_reviewer=None):
        self.verifier = verifier
        self.judge = judge
        self.identity = dict(verifier.identity)
        self.reference_judge_identity = getattr(judge, "identity", None)
        if self.reference_judge_identity is not None:
            self.identity["text_judge"] = self.reference_judge_identity
        self.phonetic_reviewer = phonetic_reviewer
        self.phonetic_review_identity = getattr(phonetic_reviewer, "identity", None) if judge is not None else None
        self.reference_phonetic_review_identity = getattr(phonetic_reviewer, "reference_identity", None)
        if self.phonetic_review_identity is not None:
            self.identity["phonetic_review"] = self.phonetic_review_identity

    def __getattr__(self, name):
        return getattr(self.verifier, name)

    def judge_reference(self, expected_text, primary_transcript, secondary_transcript=None):
        if self.judge is None:
            return None
        return self.judge.judge(expected_text, primary_transcript, secondary_transcript)

    def judge_skipped_word(self, expected_text, transcript, word, context):
        judge = getattr(self.judge, "judge_skipped_word", None)
        if not callable(judge):
            return None
        return judge(expected_text, transcript, word, context)

    def review_phonetics(self, expected_text, transcript):
        if self.phonetic_reviewer is None:
            return None
        return self.phonetic_reviewer.review(expected_text, transcript)

    def review_reference_phonetics(self, expected_text, transcript):
        if self.phonetic_reviewer is None:
            return None
        return self.phonetic_reviewer.review_reference(expected_text, transcript)

    def close(self):
        for component in (self.verifier, self.judge, self.phonetic_reviewer):
            close = getattr(component, "close", None)
            if callable(close):
                close()


class RecurringSpeechVerifier:
    """Expose bounded recovery while preserving all underlying verification APIs."""

    def __init__(self, verifier, resolver):
        self.verifier = verifier
        self.speech_resolver = resolver
        self.speech_resolution_identity = resolver.identity
        self.identity = {**verifier.identity, "speech_resolution": resolver.identity}

    def __getattr__(self, name):
        return getattr(self.verifier, name)

    def close(self):
        try:
            close = getattr(self.verifier, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(self.speech_resolver, "close", None)
            if callable(close):
                close()


def create_speech_resolver(settings, provider_settings):
    """Use the configured analysis endpoint only after a recurring speech failure."""
    if (settings.get("recurring_resolution", "analysis") == "off"
            or not provider_settings or not provider_settings.get("model")):
        return None
    from .recovery_client import RecoveryModelClient
    from .speech_resolver import SpeechResolver
    options = dict(provider_settings)
    options["max_retries"] = min(options.get("max_retries", 0), 2)
    return SpeechResolver(RecoveryModelClient(options), secrets=(options.get("api_key"),))


def create_phonetic_reviewer(settings, provider_settings):
    """Reuse the explicitly configured analysis endpoint for bounded tie-breaking."""
    if (settings.get("phonetic_fallback", "analysis") == "off"
            or not provider_settings or not provider_settings.get("model")):
        return None
    from .recovery_client import RecoveryModelClient
    from .phonetic_review import PronunciationReviewer
    options = dict(provider_settings)
    options["max_retries"] = min(options.get("max_retries", 0), 2)
    return PronunciationReviewer(RecoveryModelClient(options))


def assemble_verifier(primary, settings, jev_settings, provider_settings, reference_builder):
    """Wrap a primary recognizer with the configured reference corroboration,
    JEV text judging, phonetic review, and recurring speech recovery."""
    verifier = primary
    if settings.get("reference_model"):
        verifier = ReferenceCorroboratingVerifier(primary, reference_builder())
    reviewer = create_phonetic_reviewer(settings, provider_settings)
    judge = None
    if jev_settings and jev_settings.get("enabled"):
        from epub_tts_narrator.jev import JevTranscriptJudge
        judge = JevTranscriptJudge(jev_settings)
    if judge is not None or reviewer is not None:
        verifier = ReferenceTextJudgingVerifier(verifier, judge, reviewer)
    resolver = create_speech_resolver(settings, provider_settings)
    if resolver is not None:
        verifier = RecurringSpeechVerifier(verifier, resolver)
    return verifier


def create_verifier(settings, jev_settings=None, provider_settings=None):
    from .phonetic_match import require_phonetic_matcher
    require_phonetic_matcher()
    kind = settings.get("type", "mlx_whisper")
    if kind == "faster_whisper":
        # Casting, imports, and ordinary local rendering use this same factory.
        # Keep the configured backend when they share the server's global file.
        from .chapter_verification import create_chapter_verifier
        return create_chapter_verifier(settings, jev_settings, provider_settings)
    if kind != "mlx_whisper":
        raise ValueError("Unsupported verifier type")

    def reference_builder():
        from .reference_asr import MlxParakeetVerifier
        return MlxParakeetVerifier(settings["reference_model"])

    return assemble_verifier(
        MlxWhisperVerifier(settings["model"], language=settings["language"]),
        settings, jev_settings, provider_settings, reference_builder)
