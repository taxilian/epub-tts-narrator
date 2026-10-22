"""Offline checks for honest benchmark accounting and immutable references."""

from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from scripts.benchmark_drama_quantized import reference, summarize, verify


def take(*, elapsed=2.0, duration=4.0, warmup=False, **fields):
    row = {"elapsed_seconds": elapsed, "warmup": warmup, "peak_mlx_gb": 8.0}
    if duration is not None:
        row["duration_seconds"] = duration
    row.update(fields)
    return row


class BenchmarkSummaryTests(unittest.TestCase):
    def test_failed_generations_and_rejected_audio_cost_time_without_earning_accepted_audio(self):
        rows = [
            take(elapsed=2, duration=8, sanity_passed=True),
            take(elapsed=3, duration=None, error="generation failed", sanity_passed=False),
            take(elapsed=5, duration=12, sanity_passed=False),
        ]

        result = summarize(rows)

        self.assertEqual(result["measured_attempts"], 3)
        self.assertEqual(result["generation_errors"], 1)
        self.assertEqual(result["generation_seconds"], 10)
        self.assertEqual(result["audio_seconds"], 20)
        self.assertEqual(result["sanity_passed"], 1)
        self.assertEqual(result["audio_seconds_per_wall_second"], 2)
        self.assertEqual(result["wall_seconds_per_audio_second"], 0.5)
        self.assertEqual(result["accepted_audio_seconds_per_wall_second"], 0.8)
        self.assertTrue(result["verification_complete"])

    def test_warmups_do_not_change_throughput_or_verification_but_count_toward_peak_memory(self):
        measured = take(elapsed=3, duration=9, sanity_passed=True, peak_mlx_gb=12)
        warmups = [
            take(elapsed=1000, duration=100, warmup=True, peak_mlx_gb=22),
            take(elapsed=1000, duration=None, warmup=True,
                 error="warmup failed", sanity_passed=False, peak_mlx_gb=14),
        ]

        result = summarize([*warmups, measured])
        without_warmups = summarize([measured])

        self.assertEqual(result["peak_generation_gb"], 22)
        self.assertEqual(
            {key: value for key, value in result.items() if key != "peak_generation_gb"},
            {key: value for key, value in without_warmups.items() if key != "peak_generation_gb"},
        )

    def test_partial_verification_cannot_claim_accepted_throughput(self):
        rows = [take(sanity_passed=True), take()]

        result = summarize(rows)

        self.assertFalse(result["verification_complete"])
        self.assertIsNone(result["sanity_passed"])
        self.assertIsNone(result["accepted_audio_seconds_per_wall_second"])
        self.assertEqual(result["measured_attempts"], 2)
        self.assertEqual(result["audio_seconds_per_wall_second"], 2)

    def test_all_failed_generations_report_zero_accepted_throughput(self):
        result = summarize([
            take(duration=None, error="no audio", sanity_passed=False),
            take(elapsed=3, duration=None, error="out of memory", sanity_passed=False),
        ])

        self.assertEqual(result["generation_errors"], 2)
        self.assertEqual(result["generation_seconds"], 5)
        self.assertEqual(result["sanity_passed"], 0)
        self.assertEqual(result["accepted_audio_seconds_per_wall_second"], 0)
        self.assertIsNone(result["wall_seconds_per_audio_second"])

    def test_empty_or_warmup_only_runs_do_not_look_verified(self):
        for rows in ([], [take(warmup=True, sanity_passed=True)]):
            with self.subTest(rows=rows):
                result = summarize(rows)
                self.assertEqual(result["measured_attempts"], 0)
                self.assertFalse(result["verification_complete"])
                self.assertIsNone(result["audio_seconds_per_wall_second"])
                self.assertIsNone(result["wall_seconds_per_audio_second"])
                self.assertIsNone(result["accepted_audio_seconds_per_wall_second"])


class BenchmarkReferenceTests(unittest.TestCase):
    def test_reference_preserves_transcript_and_rejects_changed_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio = directory / "reference.wav"
            audio.write_bytes(b"immutable reference recording")
            transcript = "The lantern is still warm."
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            metadata = directory / "voice.json"
            metadata.write_text(json.dumps({
                "reference_text": transcript, "reference_sha256": digest,
            }))
            original_metadata = metadata.read_bytes()

            actual = reference(directory)

            self.assertEqual(actual["reference_text"], transcript)
            self.assertEqual(actual["reference_audio"], str(audio.resolve()))
            self.assertEqual(actual["reference_sha256"], digest)
            self.assertEqual(metadata.read_bytes(), original_metadata)
            audio.write_bytes(b"changed recording")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                reference(directory)


class FixedTranscriptVerifier:
    """The only substituted component is the external speech recognizer."""

    identity = {"backend": "offline-test"}

    def transcribe(self, path):
        return {
            "text": "Hello world.",
            "segments": [{
                "start": 0, "end": 1, "text": "Hello world.",
                "avg_logprob": -0.1, "no_speech_prob": 0.001,
                "compression_ratio": 1.1,
                "words": [
                    {"word": "Hello", "start": 0, "end": 0.5, "probability": 0.9},
                    {"word": "world.", "start": 0.5, "end": 1, "probability": 0.9},
                ],
            }],
        }


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for real audio normalization")
class BenchmarkVerificationTests(unittest.TestCase):
    def test_float_output_is_verified_as_pcm_without_changing_original_or_accepting_capped_takes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            audio = directory / "generated.wav"
            rate = 24000
            samples = [0.1 * math.sin(2 * math.pi * 220 * n / rate) for n in range(rate)]
            payload = struct.pack(f"<{rate}f", *samples)
            # IEEE float WAV, the output encoding of both inference backends.
            audio.write_bytes(
                b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
                + b"data" + struct.pack("<I", len(payload)) + payload
            )
            original = audio.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            rows = [
                take(name="eos", duration=1, expected_text="Hello world.",
                     audio=audio.name, audio_sha256=digest, hit_frame_cap=False),
                take(name="capped", duration=1, expected_text="Hello world.",
                     audio=audio.name, audio_sha256=digest, hit_frame_cap=True),
            ]
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"results": rows}))

            with patch("epub_tts_narrator.renderer.quality.MlxWhisperVerifier",
                       return_value=FixedTranscriptVerifier()), redirect_stdout(io.StringIO()):
                verify(SimpleNamespace(whisper="unused-offline-model", manifests=[manifest]))

            report = json.loads(manifest.read_text())
            accepted, capped = report["results"]
            self.assertTrue(accepted["sanity_passed"], accepted["verification"])
            self.assertEqual(accepted["verification"]["metrics"]["wer"], 0)
            self.assertTrue(capped["verification"]["passed"])
            self.assertFalse(capped["sanity_passed"])
            self.assertEqual(report["summary"]["sanity_passed"], 1)
            self.assertEqual(report["summary"]["generation_seconds"], 4)
            self.assertEqual(audio.read_bytes(), original)
            normalized = directory / accepted["verification_audio"]
            with wave.open(str(normalized)) as checked:
                self.assertEqual(checked.getsampwidth(), 2)
                self.assertEqual(checked.getframerate(), 44100)
                self.assertEqual(checked.getnchannels(), 1)
            self.assertEqual(hashlib.sha256(normalized.read_bytes()).hexdigest(),
                             accepted["verification_audio_sha256"])


if __name__ == "__main__":
    unittest.main()
