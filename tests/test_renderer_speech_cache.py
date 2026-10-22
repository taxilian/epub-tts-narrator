"""Only verified performances can be reused independently of source position."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import wave

from epub_tts_narrator.renderer.common import digest, file_hash, read_json, write_json
from epub_tts_narrator.renderer.speech_cache import SpeechCache
from test_renderer_speech_identity import reference, speech_request


class SpeechCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        reference_path = self.root / "reference.wav"
        self.write_wave(reference_path)
        self.voice = reference(reference_audio=str(reference_path),
                               reference_sha256=file_hash(reference_path))
        self.request = speech_request(self.voice)
        self.cache = self.new_cache()
        self.identity = self.cache.identity(self.request)

    @staticmethod
    def write_wave(path, sample=b"\x01\x00"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
            output.writeframes(sample * 100)

    def new_cache(self, voice=None):
        voice = voice or self.voice
        return SpeechCache(self.root, {voice["voice_id"]: voice})

    def accepted(self, *, passed=True, request=None):
        request = deepcopy(request or self.request)
        key = digest(request)
        directory = self.root / "clips" / key
        audio = directory / "attempt-001.wav"
        self.write_wave(audio)
        record = {
            "request_hash": key, "request": request, "attempt": 1, "seed": 17,
            "audio_file": audio.name, "audio_sha256": file_hash(audio),
            "validation_key": digest("fixture-verification"),
            "engine_metadata": {"synthesis_text": request["chunk"]["spoken_text"]},
            "report": {"passed": passed, "reasons": [] if passed is True else ["text_mismatch"]},
        }
        write_json(directory / "accepted.json", record)
        return audio, record

    def index_path(self):
        return self.root / "speech-cache" / (digest(self.identity) + ".json")

    def legacy_runtime_take(self):
        original = deepcopy(self.request)
        runtime = {"runtime": "mlx-audio", "installed_version": "0.1.1",
                   "installed_revision": "old-runtime-revision", "adapter_version": 1}
        original["engine"].update(runtime)
        audio, accepted = self.accepted(request=original)
        # Persist the pre-upgrade format explicitly: earlier indexes included
        # runtime metadata even when model weights and all speech inputs matched.
        identity = self.cache.identity(original)
        identity["engine"].update(runtime)
        entry = {"format": "audiobook-speech-cache", "version": 1,
                 "identity": identity, "request_hash": accepted["request_hash"],
                 "accepted_sha256": file_hash(audio.parent / "accepted.json"),
                 "audio_sha256": accepted["audio_sha256"]}
        write_json(self.root / "speech-cache" / (digest(identity) + ".json"),
                   {**entry, "record_sha256": digest(entry)})
        current = deepcopy(original)
        current["engine"].update(runtime="vendored-mlx-audio", installed_version="0.1.1+epub.1",
                                 installed_revision=None, adapter_version=2)
        return audio, accepted, current

    def test_runtime_upgrade_reuses_persisted_index_without_rewriting_provenance(self):
        audio, accepted, current_request = self.legacy_runtime_take()
        original_acceptance = (audio.parent / "accepted.json").read_bytes()
        original_audio = audio.read_bytes()
        # The durable index must also work when later voice notes prevent
        # resolving the original candidate's metadata from the current cast.
        current_voice = {**self.voice, "asset_hash": digest("updated voice notes")}
        current_request["voice_asset_hash"] = current_voice["asset_hash"]
        current = self.new_cache(current_voice)

        self.assertEqual(current.find(current.identity(current_request), request=current_request),
                         (audio, accepted))
        self.assertTrue(current.matches(current_request, accepted))
        self.assertEqual((audio.parent / "accepted.json").read_bytes(), original_acceptance)
        self.assertEqual(audio.read_bytes(), original_audio)
        self.assertEqual(list((self.root / "clips").rglob("*.wav")), [audio])

    def test_runtime_upgrade_does_not_reimport_rejected_or_altered_acceptance(self):
        for state in ("rejected", "altered"):
            with self.subTest(state=state):
                audio, accepted, current_request = self.legacy_runtime_take()
                if state == "rejected":
                    accepted["report"]["passed"] = False
                else:
                    accepted["engine_metadata"]["extra"] = "Changed after publication"
                write_json(audio.parent / "accepted.json", accepted)
                current = self.new_cache()

                self.assertIsNone(current.find(current.identity(current_request),
                                               request=current_request))

    def test_runtime_upgrade_still_checks_cached_audio_bytes(self):
        audio, _, current_request = self.legacy_runtime_take()
        self.write_wave(audio, b"\x02\x00")
        current = self.new_cache()

        with self.assertRaisesRegex(ValueError, "audio changed"):
            current.find(current.identity(current_request), request=current_request)

    def test_rejected_legacy_audio_is_not_imported_or_published(self):
        audio, rejected = self.accepted(passed=False)
        with self.assertRaisesRegex(ValueError, "successful"):
            self.cache.remember(self.identity, rejected)
        self.assertIsNone(self.cache.find(self.identity))
        self.assertFalse(self.index_path().exists())
        self.assertTrue(audio.exists())

    def test_index_cannot_turn_a_rejected_report_into_success(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        for passed in (False, 1, "true"):
            with self.subTest(passed=passed):
                rejected = deepcopy(accepted)
                rejected["report"]["passed"] = passed
                acceptance_path = audio.parent / "accepted.json"
                write_json(acceptance_path, rejected)
                entry = read_json(self.index_path())
                entry.pop("record_sha256")
                entry["accepted_sha256"] = file_hash(acceptance_path)
                write_json(self.index_path(), {**entry, "record_sha256": digest(entry)})
                self.assertIsNone(self.new_cache().find(self.identity))

    def test_missing_waveform_is_a_cache_miss_after_an_index_was_published(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        self.assertIsNotNone(self.cache.find(self.identity))
        audio.unlink()
        self.assertIsNone(self.cache.find(self.identity))
        self.assertIsNone(self.new_cache().find(self.identity))

    def test_missing_acceptance_is_a_cache_miss(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        (audio.parent / "accepted.json").unlink()
        self.assertIsNone(self.cache.find(self.identity))
        self.assertIsNone(self.new_cache().find(self.identity))

    def test_altered_waveform_is_rejected_even_after_an_in_memory_hit(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        self.assertIsNotNone(self.cache.find(self.identity))
        self.write_wave(audio, b"\x02\x00")
        for cache in (self.cache, self.new_cache()):
            with self.subTest(cache=cache), self.assertRaisesRegex(ValueError, "audio changed"):
                cache.find(self.identity)

    def test_changed_acceptance_does_not_silently_reimport_an_invalidated_index(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        self.assertIsNotNone(self.cache.find(self.identity))
        changed = deepcopy(accepted)
        changed["engine_metadata"]["extra"] = "Changed after cache publication"
        write_json(audio.parent / "accepted.json", changed)
        self.assertIsNone(self.cache.find(self.identity))
        self.assertIsNone(self.new_cache().find(self.identity))

    def test_published_content_identity_survives_voice_candidate_metadata_changes(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        changed_voice = {**self.voice, "asset_hash": digest("revised candidate metadata"),
                         "voice_id": "renamed-voice", "profile": {"description": "Updated notes"},
                         "reference_audio": str(self.root / "moved-reference.wav")}
        Path(changed_voice["reference_audio"]).write_bytes(Path(self.voice["reference_audio"]).read_bytes())
        current = self.new_cache(changed_voice)
        identity = current.identity(speech_request(changed_voice))
        self.assertEqual(current.find(identity), (audio, accepted))

    def test_successful_legacy_take_is_imported_without_copying_audio_or_changing_provenance(self):
        audio, accepted = self.accepted()
        acceptance_path = audio.parent / "accepted.json"
        original = acceptance_path.read_bytes()
        original_audio = audio.read_bytes()
        self.assertEqual(self.cache.find(self.identity), (audio, accepted))
        self.assertEqual(acceptance_path.read_bytes(), original)
        self.assertEqual(audio.read_bytes(), original_audio)
        self.assertEqual(list((self.root / "clips").rglob("*.wav")), [audio])
        self.assertTrue(self.index_path().is_file())

    def test_unresolved_legacy_voice_is_not_guessed_from_a_current_equivalent_voice(self):
        audio, _ = self.accepted()
        changed_voice = {**self.voice, "asset_hash": digest("unresolvable old candidate")}
        current = self.new_cache(changed_voice)
        self.assertIsNone(current.find(current.identity(speech_request(changed_voice))))
        self.assertTrue(audio.exists())
        self.assertFalse(self.index_path().exists())

    def test_duplicate_position_requests_share_audio_but_retain_original_request(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        moved = deepcopy(self.request)
        moved["chunk"].update(chunk_id="moved-chunk", source_ranges=[{"start": 200, "end": 206}])
        moved["chunk"]["segment"].update(sequence=90, block_id="new-block", chapter_id="later-chapter")
        self.assertNotEqual(digest(moved), accepted["request_hash"])
        self.assertEqual(self.cache.find(self.cache.identity(moved)), (audio, accepted))
        self.assertFalse((self.root / "clips" / digest(moved)).exists())
        self.assertEqual(accepted["request"], self.request)

    def test_publication_rejects_a_successful_record_for_different_speech(self):
        _, accepted = self.accepted()
        other_identity = self.cache.identity(speech_request(self.voice, "Goodbye."))
        with self.assertRaises(ValueError):
            self.cache.remember(other_identity, accepted)
        self.assertIsNone(self.cache.find(other_identity))

    def test_incompatible_engine_audio_is_not_reused_or_inspected_as_a_candidate(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        self.write_wave(audio, b"\x02\x00")
        different = deepcopy(self.request)
        different["engine"]["model"] = "another-model"
        self.assertIsNone(self.cache.find(self.cache.identity(different), request=different))
        # Explicitly opting into that prior model makes its bytes relevant,
        # and a changed waveform must still fail the normal integrity check.
        different["engine"]["cache_model_alias"] = self.request["engine"]["model"]
        with self.assertRaisesRegex(ValueError, "audio changed"):
            self.cache.find(self.cache.identity(different), request=different)

    def test_another_cache_publication_is_visible_after_an_earlier_miss(self):
        second = self.new_cache()
        self.assertIsNone(self.cache.find(self.identity))
        self.assertIsNone(second.find(self.identity))
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)

        self.assertEqual(second.find(self.identity), (audio, accepted))

    def test_switching_back_finds_an_older_model_when_alias_points_to_another_take(self):
        original = deepcopy(self.request)
        original["engine"]["cache_model_alias"] = "shared-model-family"
        changed = deepcopy(original)
        changed["engine"]["model"] = "another-model"
        changed["engine"]["checkpoint_files_sha256"] = {"weights.safetensors": "other-weights"}
        old_audio, old_accepted = self.accepted(request=original)
        _, changed_accepted = self.accepted(request=changed)
        # Existing caches can contain multiple successful concrete models but
        # only the last one published under their shared logical model alias.
        self.cache.remember(self.cache.identity(changed), changed_accepted)
        original_metadata = (old_audio.parent / "accepted.json").read_bytes()
        original_audio = old_audio.read_bytes()

        for cache in (self.cache, self.new_cache()):
            with self.subTest(cache=cache):
                self.assertEqual(cache.find(self.identity, request=self.request),
                                 (old_audio, old_accepted))
        self.assertEqual((old_audio.parent / "accepted.json").read_bytes(), original_metadata)
        self.assertEqual(old_audio.read_bytes(), original_audio)

    def test_alias_collision_does_not_recover_a_rejected_or_missing_old_take(self):
        original = deepcopy(self.request)
        original["engine"]["cache_model_alias"] = "shared-model-family"
        changed = deepcopy(original)
        changed["engine"]["model"] = "another-model"
        old_audio, old_accepted = self.accepted(request=original)
        _, changed_accepted = self.accepted(request=changed)
        self.cache.remember(self.cache.identity(changed), changed_accepted)

        for state in ("rejected", "missing"):
            with self.subTest(state=state):
                rejected = deepcopy(old_accepted)
                rejected["report"]["passed"] = state != "rejected"
                write_json(old_audio.parent / "accepted.json", rejected)
                if state == "missing":
                    old_audio.unlink()
                self.assertIsNone(self.new_cache().find(self.identity, request=self.request))

    def test_new_aliased_take_can_be_found_by_concrete_model_after_prior_cache_miss(self):
        second = self.new_cache()
        self.assertIsNone(second.find(self.identity, request=self.request))
        aliased = deepcopy(self.request)
        aliased["engine"]["cache_model_alias"] = "shared-model-family"
        audio, accepted = self.accepted(request=aliased)
        self.cache.remember(self.cache.identity(aliased), accepted)

        self.assertEqual(second.find(self.identity, request=self.request), (audio, accepted))

    def test_old_alias_index_preserves_reference_identity_when_returning_to_concrete_model(self):
        aliased = deepcopy(self.request)
        aliased["engine"]["cache_model_alias"] = "shared-model-family"
        audio, accepted = self.accepted(request=aliased)
        self.cache.remember(self.cache.identity(aliased), accepted)
        # Simulate an older cache with only its logical alias lookup persisted.
        self.index_path().unlink(missing_ok=True)
        changed_voice = {**self.voice, "asset_hash": digest("new-candidate-notes"),
                         "profile": {"description": "Updated notes"}}
        current = self.new_cache(changed_voice)
        request = speech_request(changed_voice)

        self.assertEqual(current.find(current.identity(request), request=request), (audio, accepted))

    def test_alias_cannot_recover_acceptance_altered_after_publication(self):
        aliased = deepcopy(self.request)
        aliased["engine"]["cache_model_alias"] = "shared-model-family"
        audio, accepted = self.accepted(request=aliased)
        self.cache.remember(self.cache.identity(aliased), accepted)
        changed = deepcopy(accepted)
        changed["engine_metadata"]["extra"] = "Changed after publication"
        write_json(audio.parent / "accepted.json", changed)

        self.assertIsNone(self.new_cache().find(self.identity, request=self.request))

    def test_another_cache_replacement_wins_over_a_previously_loaded_entry(self):
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)
        second = self.new_cache()
        self.assertEqual(second.find(self.identity), (audio, accepted))
        moved = deepcopy(self.request)
        moved["chunk"]["chunk_id"] = "replacement-position"
        replacement_audio, replacement = self.accepted(request=moved)
        self.cache.remember(self.identity, replacement)
        published_index = self.index_path().read_bytes()

        self.assertEqual(second.find(self.identity), (replacement_audio, replacement))
        self.assertEqual(self.index_path().read_bytes(), published_index)


if __name__ == "__main__":
    unittest.main()
