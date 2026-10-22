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

    def test_another_cache_publication_is_visible_after_an_earlier_miss(self):
        second = self.new_cache()
        self.assertIsNone(self.cache.find(self.identity))
        self.assertIsNone(second.find(self.identity))
        audio, accepted = self.accepted()
        self.cache.remember(self.identity, accepted)

        self.assertEqual(second.find(self.identity), (audio, accepted))

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
