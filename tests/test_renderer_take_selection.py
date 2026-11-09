from copy import deepcopy
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from typing import Any

from epub_tts_narrator.renderer.audio_naming import speech_take_filename
from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.speech_cache import SpeechCache, ordinarily_verified
from epub_tts_narrator.renderer.take_selection import (
    list_takes, select_take, validate_manual_selection)


class TakeSelectionTests(unittest.TestCase):
    root: Path
    request: dict[str, Any]
    request_hash: str

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.request = {
            "kind": "speech",
            "chunk": {"chunk_id": "chunk-042", "spoken_text": "Do not leave.",
                      "segment": {"voice_id": "voice-mara"}},
        }
        self.request_hash = digest(self.request)

    def clip(self, label: str) -> tuple[Path, Path]:
        clips = self.root / label / "render-work" / "clips"
        directory = clips / self.request_hash
        directory.mkdir(parents=True)
        return clips, directory

    @staticmethod
    def write_audio(path: Path, kind: str = "valid", sample: int = 1000):
        if kind == "corrupt":
            path.write_bytes(b"not a RIFF wave file")
            return
        frames = 400
        values = [0] * frames if kind == "silent" else [sample + index % 50 for index in range(frames)]
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack("<" + "h" * frames, *values))

    def attempt(self, directory: Path, number: int, *, passed: bool = False,
                audio: str = "valid", echo: bool = False) -> dict[str, Any]:
        request = deepcopy(self.request)
        name = speech_take_filename(request, number)
        path = directory / name
        self.write_audio(path, audio, sample=1000 + number * 100)
        report: dict[str, Any] = {
            "passed": passed,
            "reasons": [] if passed else ["text_mismatch"],
            "expected_text": "Do not leave.",
            "transcript": "Do not leave." if passed else "Do leave.",
            "metrics": {"wer": 0.0 if passed else 1 / 3},
            "text_confirmation_method": "verbatim" if passed else None,
        }
        if echo:
            report["history_echo"] = {"version": 1, "heard_request_hash": "a" * 64,
                                      "heard_share": 1.0, "expected_share": 0.0}
        record = {
            "request_hash": self.request_hash,
            "request": request,
            "attempt": number,
            "seed": 17,
            "validation_key": digest({"fixture_attempt": number}),
            "engine_metadata": {},
            "audio_file": name,
            "audio_sha256": file_hash(path),
            "report": report,
        }
        write_json(directory / f"attempt-{number:03d}.json", record)
        return record

    @staticmethod
    def accept(directory: Path, record: dict[str, Any]):
        write_json(directory / "accepted.json", record)

    @staticmethod
    def snapshot(directory: Path) -> dict[str, bytes | None]:
        return {str(path.relative_to(directory)): None if path.is_dir() else path.read_bytes()
                for path in directory.rglob("*")}

    def assert_rejected_without_changes(self, clips: Path, directory: Path, attempt: int,
                                        pattern: str, *, note: str = ""):
        before = self.snapshot(directory)
        with self.assertRaisesRegex(ValueError, pattern):
            select_take(clips, self.request_hash, attempt, note=note)
        self.assertEqual(self.snapshot(directory), before)

    def test_list_takes_returns_all_attempts_with_acceptance_and_eligibility(self):
        clips, directory = self.clip("listing")
        first = self.attempt(directory, 1, passed=True)
        second = self.attempt(directory, 2)
        self.attempt(directory, 3, audio="silent")
        self.attempt(directory, 4, echo=True)
        self.accept(directory, first)

        rows = list_takes(clips, self.request_hash)

        self.assertEqual([row["attempt"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(set(rows[0]), {
            "attempt", "record", "audio_path", "passed", "reasons", "text_confirmation_method",
            "duration_seconds", "transcript", "echo", "eligible", "ineligible_reason", "accepted"})
        self.assertEqual(rows[0]["record"], first)
        self.assertEqual(rows[0]["audio_path"], directory / first["audio_file"])
        self.assertTrue(rows[0]["passed"])
        self.assertTrue(rows[0]["eligible"])
        self.assertTrue(rows[0]["accepted"])
        self.assertEqual(rows[0]["text_confirmation_method"], "verbatim")
        self.assertEqual(rows[0]["duration_seconds"], 400 / 8000)
        self.assertEqual(rows[1]["record"], second)
        self.assertFalse(rows[1]["passed"])
        self.assertTrue(rows[1]["eligible"])
        self.assertFalse(rows[1]["accepted"])
        self.assertEqual(rows[1]["reasons"], ["text_mismatch"])
        self.assertEqual(rows[1]["transcript"], "Do leave.")
        self.assertFalse(rows[2]["eligible"])
        self.assertIn("physical_audio", rows[2]["ineligible_reason"])
        self.assertIsNone(rows[2]["duration_seconds"])
        self.assertTrue(rows[3]["echo"])
        self.assertFalse(rows[3]["eligible"])
        self.assertEqual(rows[3]["ineligible_reason"], "history echo")

    def test_list_takes_rejects_unknown_or_invalid_clip_directories(self):
        clips = self.root / "absent" / "clips"
        for request_hash in ("0" * 64, "not-a-request-hash"):
            with self.subTest(request_hash=request_hash):
                with self.assertRaisesRegex(ValueError, "^clip directory missing/invalid$"):
                    list_takes(clips, request_hash)

    def test_select_passed_attempt_promotes_verbatim_archives_and_is_idempotent(self):
        clips, directory = self.clip("ordinary")
        first = self.attempt(directory, 1, passed=True)
        second = self.attempt(directory, 2, passed=True)
        self.accept(directory, first)

        result = select_take(clips, self.request_hash, 2, note="Second performance is cleaner")

        self.assertEqual(result, {"previous_audio_sha256": first["audio_sha256"],
                                  "accepted_audio_sha256": second["audio_sha256"],
                                  "attempt": 2, "basis": "ordinary",
                                  "audio_file": second["audio_file"]})
        accepted = read_json(directory / "accepted.json")
        self.assertEqual(accepted, second)
        self.assertNotIn("manual_selection", accepted["engine_metadata"])
        self.assertTrue(ordinarily_verified(accepted))
        archived = list(directory.glob("accepted.superseded-*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(read_json(archived[0]), first)
        evidence_paths = list(directory.glob("manual-selection-*.json"))
        self.assertEqual(len(evidence_paths), 1)
        evidence = read_json(evidence_paths[0])
        self.assertEqual(evidence["previous_accepted"], first)
        self.assertEqual(evidence["new_choice"], {
            "attempt": 2, "note": "Second performance is cleaner",
            "audio_sha256": second["audio_sha256"]})
        self.assertIsInstance(evidence["at"], str)

        before = self.snapshot(directory)
        repeated = select_take(clips, self.request_hash, 2, note="Second performance is cleaner")
        self.assertEqual(repeated, result)
        self.assertEqual(self.snapshot(directory), before)

    def test_select_failed_attempt_writes_a_fully_audited_manual_acceptance(self):
        clips, directory = self.clip("manual")
        first = self.attempt(directory, 1, passed=True)
        failed = self.attempt(directory, 2)
        self.accept(directory, first)

        result = select_take(clips, self.request_hash, 2, note="Keep the warmer read")

        accepted = read_json(directory / "accepted.json")
        identity = {"name": "manual_take_selection", "version": 1,
                    "eligibility": "bound_full_span_physical_pcm16_v1",
                    "selection": "user_decisive_audited", "note": "Keep the warmer read"}
        self.assertEqual(result, {"previous_audio_sha256": first["audio_sha256"],
                                  "accepted_audio_sha256": failed["audio_sha256"],
                                  "attempt": 2, "basis": "manual_take_selection",
                                  "audio_file": accepted["audio_file"]})
        self.assertEqual(accepted["request"], self.request)
        self.assertEqual(accepted["request_hash"], self.request_hash)
        self.assertEqual(accepted["attempt"], 2)
        self.assertEqual(accepted["audio_sha256"], failed["audio_sha256"])
        self.assertIn("chunk-042", accepted["audio_file"])
        self.assertIn("voice-mara", accepted["audio_file"])
        self.assertTrue(accepted["audio_file"].endswith("-manual-selection.wav"))
        self.assertNotEqual(accepted["audio_file"], failed["audio_file"])
        report = accepted["report"]
        self.assertTrue(report["passed"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["expected_text"], "Do not leave.")
        self.assertEqual(report["text_confirmation_method"], "manual_take_selection")
        self.assertEqual(report["acceptance_policy"], identity)
        self.assertIs(report["ordinary_verification_passed"], False)
        self.assertEqual(report["original_failed_report"], failed["report"])
        self.assertEqual(report["transcript"], failed["report"]["transcript"])
        self.assertEqual(report["metrics"], failed["report"]["metrics"])
        marker = accepted["engine_metadata"]["manual_selection"]
        self.assertEqual(set(marker), {"request", "audit", "selected_record"})
        self.assertEqual(marker["request"], self.request)
        self.assertEqual(marker["selected_record"], failed)
        self.assertEqual(marker["audit"]["identity"], identity)
        self.assertEqual(marker["audit"]["choice"], {
            "attempt": 2, "note": "Keep the warmer read", "audio_sha256": failed["audio_sha256"]})
        self.assertEqual(accepted["validation_key"], digest({
            "selection": marker, "audio": failed["audio_sha256"]}))
        self.assertEqual(validate_manual_selection(accepted), failed)
        self.assertFalse(ordinarily_verified(accepted))
        copied = directory / accepted["audio_file"]
        original = directory / failed["audio_file"]
        self.assertEqual(copied.read_bytes(), original.read_bytes())
        self.assertTrue(original.is_file())
        archived = list(directory.glob("accepted.superseded-*.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(read_json(archived[0]), first)
        evidence = read_json(next(directory.glob("manual-selection-*.json")))
        self.assertEqual(evidence["previous_accepted"], first)
        self.assertEqual(evidence["new_choice"], {
            "attempt": 2, "note": "Keep the warmer read", "audio_sha256": failed["audio_sha256"]})

        before = self.snapshot(directory)
        repeated = select_take(clips, self.request_hash, 2, note="Keep the warmer read")
        self.assertEqual(repeated, result)
        self.assertEqual(self.snapshot(directory), before)

    def test_ineligible_or_invalid_selections_leave_every_file_unchanged(self):
        for case in ("silent", "corrupt", "echo", "missing", "changed", "long_note", "request_binding"):
            with self.subTest(case=case):
                clips, directory = self.clip("reject-" + case)
                first = self.attempt(directory, 1, passed=True)
                self.accept(directory, first)
                attempt = 2
                note = ""
                failed = self.attempt(directory, 2, audio=case if case in {"silent", "corrupt"} else "valid",
                                      echo=case == "echo")
                if case == "changed":
                    audio = directory / failed["audio_file"]
                    audio.write_bytes(audio.read_bytes() + b"changed")
                if case == "long_note":
                    note = "x" * 241
                if case == "request_binding":
                    failed["request"]["chunk"]["spoken_text"] = "Changed request."
                    write_json(directory / "attempt-002.json", failed)
                if case == "missing":
                    attempt = 99
                patterns = {"silent": "physical_audio", "corrupt": "physical_audio", "echo": "history echo",
                            "missing": "attempt 99 does not exist", "changed": "audio_sha256",
                            "long_note": "note.*240", "request_binding": "request binding"}
                self.assert_rejected_without_changes(clips, directory, attempt, patterns[case], note=note)

    def test_invalid_manual_marker_fails_closed_when_speech_cache_reads_it(self):
        clips, directory = self.clip("invalid-marker")
        first = self.attempt(directory, 1, passed=True)
        self.attempt(directory, 2)
        self.accept(directory, first)
        select_take(clips, self.request_hash, 2, note="Audited choice")
        changed = read_json(directory / "accepted.json")
        changed["engine_metadata"]["manual_selection"]["selected_record"]["report"]["transcript"] = "Changed."
        write_json(directory / "accepted.json", changed)
        cache = SpeechCache(clips.parent, {})

        with self.assertRaisesRegex(ValueError, "Manual acceptance|manual_selection"):
            cache._accepted(self.request_hash)


if __name__ == "__main__":
    unittest.main()
