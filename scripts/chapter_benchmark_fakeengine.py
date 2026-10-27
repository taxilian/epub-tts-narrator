"""Deterministic offline engine and verifier stand-ins for benchmark QA.

These replace exactly the two external model leaves the production pool
resolves: ``pool.create_engine`` (the local worker's lazy engine factory) and
``chapter_verification.create_chapter_verifier``. Everything else on the run
path — ``create_render_engine``/RenderPool construction, scheduler selection,
resource lanes, admission, ``wrap_verifier``, ChapterJobs, ChapterExecutor,
the HTTP server, ChapterClient session/ACK/delivery, verification gates,
assembly and publication — is the real production code, so fake runs still
exercise the actual RenderPool scheduling under both legacy and pipeline
modes. This module exists for offline runs and manual QA only; a real
measured benchmark never passes --fake-engine, and every fake run is
permanently marked ineligible in its report.
"""

from __future__ import annotations

import itertools
import math
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.renderer.common import digest, pcm_sha256

TRANSCRIPTS = {}
LEAF_ENGINES = []
_RATE = 44100


def _pcm_key(path):
    return pcm_sha256(path, format_error=ValueError("Fixture audio must be mono 16-bit PCM"))


def _transcript(text):
    words = text.split()
    return {"text": text, "segments": [{
        "text": text, "start": 0.0, "end": 1.0,
        "avg_logprob": -0.1, "no_speech_prob": 0.001,
        "words": [{"word": word, "start": index / len(words),
                   "end": (index + 1) / len(words), "probability": 0.99}
                  for index, word in enumerate(words)],
    }]}


class FakeBenchmarkEngine:
    """Writes deterministic tone fixtures and registers their exact text."""

    identity = {"engine": "chapter-benchmark-fake", "model": "offline-tone-fixture", "fixture": True}
    capabilities = {"reference_voice": True, "multi_speaker": True}
    calls = []
    fail_texts = set()
    _engine_ids = itertools.count(1)

    def __init__(self):
        self.engine_id = next(self._engine_ids)

    def _record(self, kind, text):
        FakeBenchmarkEngine.calls.append({"engine": self.engine_id, "kind": kind, "text": text})
        if text in FakeBenchmarkEngine.fail_texts:
            raise ValueError("Injected offline fixture failure for this passage")

    def _write(self, text, path, seed):
        frequency = 150 + int(digest({"text": text, "seed": seed})[:8], 16) % 3000
        frames = round(0.6 * _RATE)
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, _RATE, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack(
                "<" + "h" * frames,
                *(round(5000 * math.sin(2 * math.pi * frequency * index / _RATE))
                  for index in range(frames))))
        TRANSCRIPTS[_pcm_key(path)] = text
        return {"fixture": True, "seed_requested": seed, "seed_applied": False}

    def synthesize(self, text, voice, path, *, seed, direction=""):
        self._record("speech", text)
        return self._write(text, path, seed)

    def synthesize_dialogue(self, turns, path, *, seed):
        text = "".join(turn["text"] for turn in turns)
        self._record("dialogue", text)
        return self._write(text, path, seed)

    def close(self):
        pass


class FakeBenchmarkVerifier:
    """Reads back the engine's registered text through the normal ASR contract."""

    identity = {"backend": "chapter-benchmark-fake-asr", "fixture": True}

    def transcribe(self, path):
        text = TRANSCRIPTS.get(_pcm_key(path))
        if text is None:
            raise ValueError("Fixture audio is not registered with the offline verifier")
        return _transcript(text)

    def close(self):
        pass


LEAF_ENGINES = []


def install_fake_models():
    """Swap only the external model leaves; the real RenderPool stays production.

    ``pool.create_engine`` is the leaf factory behind ``_Worker.engine()`` for
    local workers, so one fake instance serves the declared slot while the
    pool itself, its admission, its lanes, and ``wrap_verifier`` scheduling
    all run real code. The chapter verifier factory is the other boundary.
    """
    import epub_tts_narrator.renderer.chapter_verification as chapter_verification
    import epub_tts_narrator.renderer.pool as pool

    leaf_engine_factory, verifier_factory = pool.create_engine, chapter_verification.create_chapter_verifier

    def fake_leaf_engine(engine_settings):
        engine = FakeBenchmarkEngine()
        LEAF_ENGINES.append(engine)
        return engine

    def fake_verifier(verification, jev, provider):
        return FakeBenchmarkVerifier()

    pool.create_engine = fake_leaf_engine
    chapter_verification.create_chapter_verifier = fake_verifier

    def restore():
        pool.create_engine = leaf_engine_factory
        chapter_verification.create_chapter_verifier = verifier_factory

    return restore


FAKE_ENGINE = FakeBenchmarkEngine
