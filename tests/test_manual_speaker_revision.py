"""Transactional publication of terminal speaker choices and prompt-defined voices."""

from copy import deepcopy
from contextlib import contextmanager
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (_completed_directory, continuation_registry,
    fresh_registry, publish_directory)
from epub_tts_narrator.analysis_revision import _install, revise_directories
from epub_tts_narrator.audition_scripts import FILENAME, prepare_document_samples
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.series import link_analysis
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from shared_fixtures import Terminal
from test_series import fixture, propose


AUDITION_SCRIPT = {
    "text": ("The tide is turning already, and we should reach the far light before it fails. "
             "Keep the lantern low and tell me what you see along the water. "
             "We can still choose the safer channel if you are unsure."),
    "basis": "casting", "reason": "A source-independent character audition.",
}


class ManualSpeakerRevisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.paths, self.documents = [], []
        for number in range(2):
            path = self.root / f"book-{number + 1}"
            document = fixture(sequel=number > 0)
            document["source"]["book"]["id"] = f"manual-revision-{number + 1}"
            if number == 0:
                document["segments"][0].update(role_id=None, voice_id=None)
                document["segments"][0]["attribution"].update(
                    basis="unresolved", review_required=True, reason="Uncertain speaker")
                document["issues"].append({"id": "speaker-choice", "kind": "attribution",
                    "target": document["segments"][0]["block_id"], "reason": "Uncertain speaker",
                    "resolved": False})
                registry = fresh_registry(path)
            else:
                registry = continuation_registry(self.paths[-1], document["source"])
                propose(document, self.documents[0]["cast"]["characters"][0]["id"])
            linked = link_analysis(document, registry)
            if number == 0:
                self.sample_voice = linked["cast"]["characters"][0]["voice_id"]
                linked["provenance"]["voice_samples"] = {
                    "version": 1, "source_fingerprint": fingerprint(linked["source"]),
                    "cast_fingerprint": fingerprint(linked["cast"]),
                    "scripts": {self.sample_voice: deepcopy(AUDITION_SCRIPT)}, "samples": {}}
            publish_directory(linked, registry, path, export_bundle,
                              continue_from=self.paths[-1] if self.paths else None)
            self.paths.append(path)
            self.documents.append(linked)
        (self.paths[0] / FILENAME).write_text(json.dumps({
            "version": 1, "source_fingerprint": fingerprint(self.documents[0]["source"]),
            "cast_fingerprint": fingerprint(self.documents[0]["cast"]),
            "scripts": {self.sample_voice: deepcopy(AUDITION_SCRIPT)}, "samples": {}}), encoding="utf-8")
        for path in self.paths:
            (path / "voices").mkdir()
            (path / "voices/reference.wav").write_bytes(b"untouched voice")
            (path / "render-work").mkdir()
            (path / "render-work/accepted.json").write_text("untouched cache")
            (path / "voice-aliases.json").write_text("untouched aliases")

    def candidate(self, answers="n\nLater Revealed Speaker\nA low, precise adult voice.\n", *, prepare=True):
        speaker_input = deepcopy(self.documents[0])
        result = prompt_speaker_decisions(speaker_input, input_stream=Terminal(answers),
                                          output_stream=StringIO())
        if prepare:
            result = prepare_document_samples(result, None, allow_incomplete=True)
        return speaker_input, result

    def revise(self, speaker_input, candidate):
        return revise_directories({self.paths[0]: candidate}, directories=self.paths,
                                  scope="speaker-decisions",
                                  speaker_decision_inputs={self.paths[0]: speaker_input},
                                  progress=lambda _: None)

    def managed(self):
        names = ["analysis.json", "series-cast.json", "manifest.json", "review.json", FILENAME,
                 "voices/reference.wav", "render-work/accepted.json", "voice-aliases.json"]
        return {path: {name: (path / name).read_bytes() for name in names if (path / name).exists()}
                for path in self.paths}

    def test_new_voice_publishes_exact_append_and_rebinds_auditions_without_descendant_cast_changes(self):
        speaker_input, candidate = self.candidate()
        old_cast = deepcopy(self.documents[0]["cast"])
        self.revise(speaker_input, candidate)
        revised, registry, _ = _completed_directory(self.paths[0])
        for collection in ("characters", "voices", "roles"):
            self.assertEqual(revised["cast"][collection][:-1], old_cast[collection])
        created = revised["cast"]["characters"][-1]
        self.assertEqual(created["name"], "Later Revealed Speaker")
        self.assertEqual(created["evidence"], [])
        self.assertEqual(revised["segments"][0]["voice_id"], created["voice_id"])
        self.assertEqual(revised["provenance"]["speaker_decisions"],
                         candidate["provenance"]["speaker_decisions"])
        samples = revised["provenance"]["voice_samples"]
        self.assertEqual(samples["cast_fingerprint"], fingerprint(revised["cast"]))
        self.assertEqual(samples["scripts"][self.sample_voice], AUDITION_SCRIPT)
        self.assertIn(created["voice_id"], samples["pending"])
        sidecar = json.loads((self.paths[0] / FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(sidecar["cast_fingerprint"], fingerprint(revised["cast"]))
        self.assertEqual(sidecar["scripts"], {self.sample_voice: AUDITION_SCRIPT})
        self.assertEqual(registry["books"][0]["cast"], revised["cast"])
        descendant, descendant_registry, _ = _completed_directory(self.paths[1])
        self.assertEqual(descendant["cast"], self.documents[1]["cast"])
        self.assertEqual(descendant_registry["books"][0]["cast"], revised["cast"])
        for path in self.paths:
            self.assertEqual((path / "voices/reference.wav").read_bytes(), b"untouched voice")
            self.assertEqual((path / "render-work/accepted.json").read_text(), "untouched cache")
            self.assertEqual((path / "voice-aliases.json").read_text(), "untouched aliases")

    def test_existing_role_and_automatic_only_calls_keep_prior_receipts_and_sidecar_bytes(self):
        speaker_input, candidate = self.candidate("1\n", prepare=False)
        sidecar_before = (self.paths[0] / FILENAME).read_bytes()
        self.revise(speaker_input, candidate)
        first, _, _ = _completed_directory(self.paths[0])
        receipt = deepcopy(first["provenance"]["speaker_decisions"])
        automatic = deepcopy(first)
        automatic["segments"][2]["attribution"]["reason"] = "Automatic completion retained this narrator."
        automatic["provenance"]["speaker_completion"] = {"automatic": True}
        automatic = prepare_document_samples(automatic, None, allow_incomplete=True)
        automatic["readiness"] = readiness(automatic)
        self.revise(automatic, automatic)
        second, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(second["provenance"]["speaker_decisions"], receipt)
        self.assertNotIn("speaker_decision_history", second["provenance"])
        self.assertEqual((self.paths[0] / FILENAME).read_bytes(), sidecar_before)

    def test_partial_manual_completion_is_publishable_with_remaining_unassigned_speech(self):
        speaker_input = deepcopy(self.documents[0])
        speaker_input["segments"][2].update(role_id=None, voice_id=None)
        speaker_input["segments"][2]["attribution"].update(basis="unresolved", review_required=True)
        candidate = prompt_speaker_decisions(speaker_input, input_stream=Terminal("1\nq\n"),
                                             output_stream=StringIO())
        self.revise(speaker_input, candidate)
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertIsNone(revised["segments"][2]["voice_id"])
        self.assertEqual(revised["provenance"]["speaker_decisions"]["remaining_unassigned_segments"], 1)
        self.assertFalse(revised["readiness"]["ready"])

    def test_default_scope_and_invalid_manual_deltas_leave_every_file_unchanged(self):
        speaker_input, candidate = self.candidate()
        with self.assertRaisesRegex(ValueError, "preserve source, cast"):
            revise_directories({self.paths[0]: candidate}, progress=lambda _: None)
        mutations = [
            lambda doc: doc["provenance"]["speaker_decisions"].update(source_fingerprint="0" * 64),
            lambda doc: doc["provenance"]["speaker_decisions"].update(cast_fingerprint="0" * 64),
            lambda doc: doc["provenance"]["speaker_decisions"].update(input_script_fingerprint="0" * 64),
            lambda doc: doc["provenance"]["speaker_decisions"].update(output_script_fingerprint="0" * 64),
            lambda doc: doc["provenance"]["speaker_decisions"].update(remaining_unassigned_segments=1),
            lambda doc: doc["provenance"]["speaker_decisions"]["decisions"][0].update(segment_id="s3"),
            lambda doc: doc["provenance"]["speaker_decisions"]["decisions"][0].update(previous_kind="dialogue"),
            lambda doc: doc["provenance"]["speaker_decisions"]["decisions"][0].update(selected_roles=["mara"]),
            lambda doc: doc["provenance"]["speaker_decisions"]["decisions"][0]["defined_voice"]["created"]["voice"].update(description="Forged"),
            lambda doc: doc["provenance"]["speaker_decisions"]["decisions"][0]["defined_voice"].update(character_id="forged"),
            lambda doc: doc["cast"]["voices"][-1].update(description="Forged"),
            lambda doc: doc["cast"]["characters"].append(deepcopy(doc["cast"]["characters"][0])),
            lambda doc: doc["segments"][1]["attribution"].update(reason="Unrelated edit"),
            lambda doc: doc["issues"].append({"id": "other", "kind": "casting", "target": "b1",
                                               "reason": "Keep pending", "resolved": True}),
            lambda doc: doc["provenance"].update(unrelated=True),
            lambda doc: doc.update(readiness={"ready": False, "errors": [], "review_count": 999}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                before = self.managed()
                forged = deepcopy(candidate)
                mutate(forged)
                with self.assertRaises(ValueError):
                    self.revise(speaker_input, forged)
                self.assertEqual(self.managed(), before)
        with self.assertRaises(ValueError):
            revise_directories({self.paths[0]: candidate}, scope="speaker-decisions")
        forged_input = deepcopy(speaker_input)
        forged_input["cast"]["voices"][0]["description"] = "Not the published cast"
        with self.assertRaises(ValueError):
            self.revise(forged_input, candidate)

    def test_publication_rollback_restores_analysis_history_and_rebound_sidecar(self):
        speaker_input, candidate = self.candidate()
        before = self.managed()
        calls = 0
        def fail_once(source, destination):
            nonlocal calls
            calls += 1
            if calls == 8:
                raise OSError("interrupted manual publication")
            return _install(source, destination)
        with patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "interrupted manual publication"):
                self.revise(speaker_input, candidate)
        self.assertEqual(self.managed(), before)

    def test_pending_profile_refresh_blocks_publication_without_removing_its_lock(self):
        speaker_input, candidate = self.candidate()
        lock = self.paths[0] / ".voice-profile-refresh.lock"
        lock.mkdir()
        before = self.managed()
        with self.assertRaisesRegex(ValueError, "Voice profiles are invalid"):
            self.revise(speaker_input, candidate)
        self.assertTrue(lock.is_dir())
        self.assertEqual(self.managed(), before)

    def test_pending_refresh_created_between_precheck_and_lease_acquisition_is_rejected(self):
        from epub_tts_narrator.voice_profile_state import FILENAME as PROFILE_STATE
        speaker_input, candidate = self.candidate()
        before = self.managed()

        @contextmanager
        def refresh_became_pending(directory):
            (self.paths[0] / PROFILE_STATE).write_text("{}", encoding="utf-8")
            yield

        with patch("epub_tts_narrator.voice_profile_guard.refresh_locks", refresh_became_pending):
            with self.assertRaisesRegex(ValueError, "Voice profiles are invalid"):
                self.revise(speaker_input, candidate)
        self.assertEqual(self.managed(), before)


if __name__ == "__main__":
    unittest.main()
