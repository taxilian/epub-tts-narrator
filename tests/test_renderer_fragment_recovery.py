"""Cutoff-tail fragment recovery keeps exact source, voice, and full verification."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.common import digest, file_hash, read_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import _directed_text
from epub_tts_narrator.renderer.fragment_recovery import (
    fragment_plan, fragment_part_take, fragment_requests, fragment_started,
)
from epub_tts_narrator.renderer.speech_cache import SpeechCache
from epub_tts_narrator.renderer.speech_identity import (
    generation_identity, single_speech_recovery_part,
)
from epub_tts_narrator.renderer.speech_recovery import recover_speech, split_recovery_text
from epub_tts_narrator.renderer.speech_resolver import SpeechResolver
from epub_tts_narrator.renderer.verification import RecurringSpeechVerifier
from epub_tts_narrator.renderer.workflow import SpeechRetryBudget, SpeechVerificationFailed
from test_renderer_speech_identity import reference, speech_request
from test_renderer_speech_recovery import AudioVerifier
from test_renderer_speech_resolver import Client
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine


TEXT = "Insang Mortuary Services! This is—ack!"
PART = "This is—ack!"


class FragmentPlanTests(unittest.TestCase):
    def test_final_emdash_tail_plan_keeps_exact_contiguous_offsets(self):
        parts = fragment_plan("Ready or not—here I come, right now—ack!")
        self.assertEqual([part["text"] for part in parts],
                         ["Ready or not—here I come, right now—", "ack!"])
        self.assertEqual("".join(part["text"] for part in parts),
                         "Ready or not—here I come, right now—ack!")
        self.assertEqual(parts[0]["start"], 0)
        self.assertEqual(parts[0]["end"], parts[1]["start"])
        self.assertEqual(parts[-1]["end"], len("Ready or not—here I come, right now—ack!"))

    def test_long_short_or_unsafe_tails_are_never_planned(self):
        for text in ("He said it—", "This is—so entirely much later than expected!",
                     "It—was here.", "Nothing special here at all.",
                     "See https://example.org—now", "She said—", ""):
            with self.subTest(text=text):
                self.assertIsNone(fragment_plan(text))

    def test_fragment_requests_require_a_bound_single_speech_part(self):
        voice = reference()
        parent = speech_request(voice, TEXT)
        parent["chunk"]["segment"]["delivery"]["direction"] = "A tired receptionist."
        part = split_recovery_text(TEXT)[1]
        bound = {"kind": "speech_recovery_part", "version": 1,
                 "parent_request_hash": digest(parent), "recovery_request_hash": digest(parent),
                 "part": part, "synthesis_request": single_speech_recovery_part(parent, part)}
        planned = fragment_requests(PART, bound, deepcopy(DEFAULTS))
        self.assertEqual([piece["text"] for piece in planned[0]["recovery"]["parts"]],
                         ["This is—", "ack!"])
        self.assertEqual(planned[1]["record_sha256"], digest(
            {key: value for key, value in planned[1].items() if key != "record_sha256"}))
        legacy = {key: value for key, value in bound.items() if key != "synthesis_request"}
        self.assertIsNone(fragment_requests(PART, legacy, deepcopy(DEFAULTS)))
        self.assertIsNone(fragment_requests("Different words.", bound, deepcopy(DEFAULTS)))
        whole = speech_request(voice, PART)
        self.assertIsNone(fragment_requests(PART, whole, deepcopy(DEFAULTS)))


class FragmentRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.voice = reference()
        self.voices = {self.voice["asset_hash"]: self.voice}
        self.request = speech_request(self.voice, TEXT)
        self.direction = "A tired receptionist."
        self.request["chunk"]["segment"]["delivery"]["direction"] = self.direction
        self.request["synthesis_identity"] = generation_identity(self.request, self.voices)
        self.parent = self.work / "clips" / digest(self.request)
        self.config = deepcopy(DEFAULTS)
        self.config["render"].update(max_attempts=2, ffmpeg=FFMPEG, ffprobe=FFPROBE,
                                     keep_intermediate_audio=True)
        self.config["quality"]["max_wer"] = 0.0
        self.config["verification"]["max_resolution_attempts"] = 5
        self.fixtures = {}
        self.engine = FixtureEngine(self.fixtures)
        self.asr = AudioVerifier(self.fixtures)
        self.client = Client(action="accept")
        self.verifier = RecurringSpeechVerifier(self.asr, SpeechResolver(self.client))
        self.calls, self.messages = [], []
        self.interrupt_tail = False

    def generate(self, text, path, seed):
        spoken = "This is—" if text == PART else text
        metadata = self.engine._write(spoken, path, seed)
        if self.interrupt_tail and text == "ack!":
            raise KeyboardInterrupt("test interruption")
        self.calls.append(text)
        directed, delivery = _directed_text(text.strip(), self.direction)
        return {**metadata, "synthesis_text": directed, **delivery}

    def plain(self, text, path, seed):
        self.calls.append(text)
        metadata = self.engine._write(text, path, seed)
        directed, delivery = _directed_text(text.strip(), "")
        return {**metadata, "synthesis_text": directed, **delivery}

    def recover(self):
        return recover_speech(TEXT, self.generate, self.parent, self.request, self.verifier,
                              self.config, notify=self.messages.append,
                              retry_budget=SpeechRetryBudget(), voices_by_asset=self.voices,
                              synthesize_plain=self.plain)

    def fragment_audit(self, accepted):
        evidence = accepted["engine_metadata"]["recovery"]["parts"][1]
        return evidence, evidence["acceptance"]["engine_metadata"]["fragment_recovery"]

    def part_directory(self):
        marker = read_json(self.parent / "speech-recovery.json")
        final = self.parent.parent / marker["recovery_request_hash"]
        for path in sorted(final.glob("parts/*")):
            for name in ("accepted.json", "attempt-001.json"):
                record_path = path / name
                if record_path.exists():
                    if read_json(record_path)["request"]["part"]["text"] == PART:
                        return path
                    break
        raise AssertionError("fragment part directory not found")

    def test_cutoff_tail_pieces_and_full_join_are_verified_with_unchanged_identity(self):
        wav, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(accepted["report"]["transcript"], TEXT)
        self.assertEqual(self.calls.count("Insang Mortuary Services! "), 1)
        self.assertEqual(self.calls.count(PART), 2)
        self.assertEqual(self.calls.count("This is—"), 1)
        self.assertEqual(self.calls.count("ack!"), 1)
        evidence, audit = self.fragment_audit(accepted)
        self.assertEqual(audit["strategy"], "cutoff_tail")
        self.assertEqual(audit["child_acceptance_version"], 1)
        self.assertEqual([piece["text"] for piece in audit["parts"]], ["This is—", "ack!"])
        self.assertEqual(evidence["acceptance"]["request"]["part"]["text"], PART)
        piece_paths = [row["audio_file"] for row in audit["evidence"]]
        self.assertTrue(all(path.startswith("fragments/") for path in piece_paths))
        self.assertEqual(generation_identity(accepted["request"], self.voices,
                                             engine_metadata=accepted["engine_metadata"]),
                         generation_identity(self.request, self.voices))
        cache = SpeechCache(self.work, {"voice": self.voice})
        identity = cache.identity(self.request)
        cache.remember(identity, accepted)
        self.assertEqual(cache.find(identity, request=self.request), (wav, accepted))
        calls = list(self.calls)
        self.assertEqual(self.recover(), (wav, accepted))
        self.assertEqual(self.calls, calls)

    def test_interrupted_tail_piece_resumes_whole_part_takes_stay_retired(self):
        self.interrupt_tail = True
        with self.assertRaises(KeyboardInterrupt):
            self.recover()
        self.assertEqual(self.calls.count(PART), 2)
        self.assertEqual(self.calls.count("This is—"), 1)
        part_dir = self.part_directory()
        self.assertTrue((part_dir / "speech-fragment-recovery.json").is_file())
        self.interrupt_tail = False
        wav, accepted = self.recover()
        self.assertTrue(accepted["report"]["passed"])
        self.assertEqual(self.calls.count(PART), 2)
        self.assertEqual(self.calls.count("This is—"), 1)
        self.assertEqual(self.calls.count("ack!"), 1)
        self.assertTrue(wav.is_file())

    def test_exhausted_fragment_join_is_terminal_and_never_rebuilt(self):
        self.asr.final_transcript = "Insang Mortuary Services! This is—what"
        for _ in range(2):
            with self.assertRaisesRegex(SpeechVerificationFailed, "Cutoff-tail recovery is exhausted"):
                self.recover()
        self.assertEqual(self.calls.count(PART), 2)
        self.assertEqual(self.calls.count("This is—"), 1)
        self.assertEqual(self.calls.count("ack!"), 1)
        part_dir = self.part_directory()
        self.assertFalse((part_dir / "accepted.json").exists())
        join = read_json(part_dir / "attempt-003.json")
        self.assertEqual(join["report"]["reasons"], ["text_mismatch"])
        piece_hashes = {file_hash(path) for path in part_dir.glob("fragments/*/accepted.json")}
        with self.assertRaisesRegex(SpeechVerificationFailed, "Cutoff-tail recovery is exhausted"):
            self.recover()
        self.assertEqual({file_hash(path) for path in part_dir.glob("fragments/*/accepted.json")},
                         piece_hashes)

    def test_fragment_audit_tampering_cannot_change_identity(self):
        _, accepted = self.recover()
        evidence = accepted["engine_metadata"]["recovery"]["parts"][1]
        for change in ("voice", "text", "marker", "waveform", "nesting", "binding"):
            metadata = deepcopy(accepted["engine_metadata"])
            row = metadata["recovery"]["parts"][1]
            child = row["acceptance"]
            if change == "voice":
                piece = child["request"]["synthesis_request"]
                piece["voice_asset_hash"] = digest("another voice")
            elif change == "text":
                piece = child["request"]["synthesis_request"]
                piece["chunk"]["spoken_text"] = "This was—"
            elif change == "marker":
                del child["engine_metadata"]["fragment_recovery"]["evidence"][1]["acceptance"]
            elif change == "waveform":
                child["audio_sha256"] = digest("different audio")
            elif change == "nesting":
                child["engine_metadata"]["fragment_recovery"] = {"version": 1}
            else:
                child["request"]["recovery_request_hash"] = digest("another plan")
            row["acceptance"] = child
            row["accepted_sha256"] = hashlib.sha256(
                (json.dumps(child, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                generation_identity(accepted["request"], self.voices, engine_metadata=metadata)

    def test_without_a_resolver_a_failing_part_keeps_its_plain_terminal_failure(self):
        part = split_recovery_text(TEXT)[1]
        bound = {"kind": "speech_recovery_part", "version": 1,
                 "parent_request_hash": digest(self.request),
                 "recovery_request_hash": digest(self.request),
                 "part": part, "synthesis_request": single_speech_recovery_part(self.request, part)}
        directory = self.work / "parts" / digest(bound)
        with self.assertRaises(SpeechVerificationFailed) as caught:
            fragment_part_take(PART, lambda path, seed: self.generate(PART, path, seed),
                               directory, bound, self.asr, self.config,
                               notify=self.messages.append, retry_budget=SpeechRetryBudget(),
                               synthesize_part=self.generate, generate_plain=self.plain,
                               allow_fragments=True)
        self.assertEqual(caught.exception.request, bound)
        self.assertEqual(self.calls.count(PART), 2)
        self.assertFalse((directory / "speech-fragment-recovery.json").exists())
        self.assertFalse(fragment_started(directory, bound, PART, self.config))


if __name__ == "__main__":
    unittest.main()
