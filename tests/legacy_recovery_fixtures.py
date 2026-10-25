from epub_tts_narrator.renderer.speech_resolver import SpeechResolver


class LegacySpeechResolver(SpeechResolver):
    """Exercise the original recovery contract without the new exhaustion capability."""

    @property
    def clip_recovery_client(self):
        return None
