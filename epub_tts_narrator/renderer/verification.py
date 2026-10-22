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
        # Include the policy in chapter checkpoints as well as per-take keys.
        self.identity = {**verifier.identity, "text_judge": judge.identity}
        self.reference_judge_identity = judge.identity
        self.phonetic_reviewer = phonetic_reviewer
        self.phonetic_review_identity = getattr(phonetic_reviewer, "identity", None)
        if self.phonetic_review_identity is not None:
            self.identity["phonetic_review"] = self.phonetic_review_identity

    def __getattr__(self, name):
        return getattr(self.verifier, name)

    def judge_reference(self, expected_text, primary_transcript, secondary_transcript=None):
        return self.judge.judge(expected_text, primary_transcript, secondary_transcript)

    def review_phonetics(self, expected_text, transcript):
        if self.phonetic_reviewer is None:
            return None
        return self.phonetic_reviewer.review(expected_text, transcript)

    def close(self):
        for component in (self.verifier, self.judge, self.phonetic_reviewer):
            close = getattr(component, "close", None)
            if callable(close):
                close()


def create_phonetic_reviewer(settings, provider_settings):
    """Reuse the explicitly configured analysis endpoint for bounded tie-breaking."""
    if (settings.get("phonetic_fallback", "analysis") == "off"
            or not provider_settings or not provider_settings.get("model")):
        return None
    from epub_tts_narrator.provider import OpenAICompatibleClient
    from .phonetic_review import PronunciationReviewer
    options = dict(provider_settings)
    # A handful of word-pair decisions needs a bounded response, not the book
    # analyzer's potentially very large output budget.
    options["max_output_tokens"] = min(options.get("max_output_tokens") or 2048, 2048)
    options["max_retries"] = min(options.get("max_retries", 0), 2)
    return PronunciationReviewer(OpenAICompatibleClient(**options))


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
    primary = MlxWhisperVerifier(settings["model"], language=settings["language"])
    verifier = primary
    if settings.get("reference_model"):
        from .reference_asr import MlxParakeetVerifier
        verifier = ReferenceCorroboratingVerifier(
            primary, MlxParakeetVerifier(settings["reference_model"]))
    if jev_settings and jev_settings.get("enabled"):
        from epub_tts_narrator.jev import JevTranscriptJudge
        verifier = ReferenceTextJudgingVerifier(
            verifier, JevTranscriptJudge(jev_settings), create_phonetic_reviewer(settings, provider_settings))
    return verifier
