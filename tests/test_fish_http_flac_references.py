"""Chapter-service voices are stored as FLAC; `serve` accepts WAV references only.

Regression: every Runpod chapter request failed with HTTP 400 "Invalid
reference 0 ... expected ... WAV" because the stored FLAC was sent as is.
Uses the real HTTP server with a fake engine on loopback; no models.
"""

import hashlib
import io
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import soundfile as sf

from test_fish_http import FakeEngine

from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.fish_http import FishHTTPEngine, wire_reference_audio
from epub_tts_narrator.renderer.server import make_server


def flac_bytes(*, channels=1, subtype="PCM_16", seconds=1.0, rate=44100):
    frames = int(rate * seconds)
    tone = (0.25 * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, frames))).astype(np.float32)
    data = np.repeat(tone[:, None], channels, axis=1)
    output = io.BytesIO()
    sf.write(output, data, rate, format="FLAC", subtype=subtype)
    return output.getvalue()


class WireReferenceAudioTests(unittest.TestCase):
    def test_lossless_mono_flac_becomes_wav_with_identical_pcm(self):
        blob = flac_bytes()
        wav = wire_reference_audio(blob)
        self.assertTrue(wav.startswith(b"RIFF"))
        decoded, rate = sf.read(io.BytesIO(wav), dtype="int16", always_2d=True)
        original, original_rate = sf.read(io.BytesIO(blob), dtype="int16", always_2d=True)
        self.assertEqual((rate, sf.info(io.BytesIO(wav)).subtype), (original_rate, "PCM_16"))
        np.testing.assert_array_equal(decoded, original)
        self.assertEqual(wire_reference_audio(blob), wav)  # deterministic, so reuse ids stay stable

    def test_wav_and_other_bytes_pass_through_unchanged(self):
        output = io.BytesIO()
        sf.write(output, np.zeros((100, 1), dtype=np.int16), 24000, format="WAV", subtype="PCM_16")
        self.assertEqual(wire_reference_audio(output.getvalue()), output.getvalue())

    def test_flac_that_is_not_lossless_mono_16_bit_is_refused_not_converted(self):
        for blob in (flac_bytes(channels=2), flac_bytes(subtype="PCM_24")):
            with self.assertRaisesRegex(EngineError, "lossless mono 16-bit"):
                wire_reference_audio(blob)
        with self.assertRaisesRegex(EngineError, "could not be decoded"):
            wire_reference_audio(b"fLaC" + b"\0" * 64)


class FlacReferenceTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engine = FakeEngine()
        config = {"server": {"host": "127.0.0.1", "port": 0, "api_key": "", "max_request_bytes": 104857600,
                             "max_pending_requests": 4}, "engine": {"type": "drama3_mlx"}}
        try:
            self.server = make_server(config, self.engine, notify=lambda message: None)
        except PermissionError:
            self.skipTest("localhost sockets are disabled by the sandbox")
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(lambda: (self.server.shutdown(), thread.join(), self.server.server_close()))
        self.client = FishHTTPEngine({"url": f"http://127.0.0.1:{self.server.server_port}/v1/tts",
                                      "name": "runpod-copy", "api_key": ""})
        self.addCleanup(self.client.close)

    def voice(self, blob):
        path = self.root / "reference.flac"  # the chapter service's stored voice
        path.write_bytes(blob)
        return {"kind": "reference", "reference_audio": str(path), "reference_text": "The reference words.",
                "voice_id": "character-1", "reference_sha256": hashlib.sha256(blob).hexdigest()}

    def test_registered_flac_voice_renders_through_serve(self):
        blob = flac_bytes()
        self.client.synthesize("A passage for the narrator.", self.voice(blob), self.root / "take.wav", seed=7)
        (received, transcript, _speaker), = self.engine.calls[0]["references"]
        self.assertTrue(received.startswith(b"RIFF"))
        self.assertEqual(transcript, "The reference words.")
        np.testing.assert_array_equal(sf.read(io.BytesIO(received), dtype="int16", always_2d=True)[0],
                                      sf.read(io.BytesIO(blob), dtype="int16", always_2d=True)[0])

    def test_unsupported_flac_never_reaches_the_server(self):
        with self.assertRaisesRegex(EngineError, "lossless mono 16-bit"):
            self.client.synthesize("A passage.", self.voice(flac_bytes(channels=2)), self.root / "take.wav", seed=7)
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
