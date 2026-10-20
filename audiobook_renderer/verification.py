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


def create_verifier(settings):
    primary = MlxWhisperVerifier(settings["model"], language=settings["language"])
    if not settings.get("reference_model"):
        return primary
    from .reference_asr import MlxParakeetVerifier
    return ReferenceCorroboratingVerifier(
        primary, MlxParakeetVerifier(settings["reference_model"]))
