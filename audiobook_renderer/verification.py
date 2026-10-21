"""Configured speech verifier with optional independent reference rechecking."""

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

    def __init__(self, verifier, judge):
        self.verifier = verifier
        self.judge = judge
        # Ordinary speech verification keeps its existing identity. The quality
        # layer binds this extra policy only into reference validation keys.
        self.identity = verifier.identity
        self.reference_judge_identity = judge.identity

    def __getattr__(self, name):
        return getattr(self.verifier, name)

    def judge_reference(self, expected_text, primary_transcript, secondary_transcript=None):
        return self.judge.judge(expected_text, primary_transcript, secondary_transcript)

    def close(self):
        for component in (self.verifier, self.judge):
            close = getattr(component, "close", None)
            if callable(close):
                close()


def create_verifier(settings, jev_settings=None):
    primary = MlxWhisperVerifier(settings["model"], language=settings["language"])
    verifier = primary
    if settings.get("reference_model"):
        from .reference_asr import MlxParakeetVerifier
        verifier = ReferenceCorroboratingVerifier(
            primary, MlxParakeetVerifier(settings["reference_model"]))
    if jev_settings and jev_settings.get("enabled"):
        from epub_audiobook.jev import JevTranscriptJudge
        verifier = ReferenceTextJudgingVerifier(verifier, JevTranscriptJudge(jev_settings))
    return verifier
