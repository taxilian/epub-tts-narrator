from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import analyze
from epub_tts_narrator.series import link_analysis, load_registry, save_registry, series_context
from epub_tts_narrator.speech_traits import speech_traits, validate_speech_traits
from test_contracts import valid_document
from test_pipeline import FakeClient
from test_series import fixture, propose


class SpeechTraitClient(FakeClient):
    def discover(self, payload):
        response = super().discover(payload)
        for voice in response["voices"]:
            voice["speechTraits"] = "neutral General American accent; quick, clipped cadence"
        return response


class SpeechTraitsTests(unittest.TestCase):
    def test_validator_returns_valid_text_unchanged_and_absence_is_empty(self):
        value = "neutral General American accent; quick, clipped cadence"

        self.assertEqual(validate_speech_traits(value), value)
        self.assertEqual(validate_speech_traits("x" * 400), "x" * 400)
        self.assertEqual(speech_traits({"description": "Legacy profile."}), "")
        self.assertEqual(speech_traits({"speechTraits": value}), value)

    def test_validator_rejects_invalid_present_values(self):
        invalid = [None, "", "   ", "x" * 401, "first\nsecond", "[fast]", "<prosody>",
                   "quiet\tpace", "zero\u200bwidth", "accent\u2028", "accent\u2029"]

        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "speechTraits"):
                validate_speech_traits(value)

    def test_document_validation_rejects_an_invalid_present_field(self):
        document = valid_document()
        document["cast"]["voices"][0]["speechTraits"] = "<fast>"

        errors = validate_document(document)

        self.assertTrue(any("cast.voices[0].speechTraits" in error for error in errors), errors)

    def test_generated_field_reaches_analysis_and_export(self):
        source = deepcopy(valid_document()["source"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = analyze(source, SpeechTraitClient(), root / "cache", identity_consensus=False)
            target = root / "bundle"
            export_bundle(document, target)
            exported = json.loads((target / "cast.json").read_text(encoding="utf-8"))

        analyzed = next(voice for voice in document["cast"]["voices"]
                        if voice["id"] != "voice_narrator_main")
        saved = next(voice for voice in exported["voices"] if voice["id"] == analyzed["id"])
        self.assertEqual(saved["speechTraits"], analyzed["speechTraits"])

    def test_old_generated_profiles_without_field_remain_valid_and_omitted(self):
        source = deepcopy(valid_document()["source"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            document = analyze(source, FakeClient(), root / "cache", identity_consensus=False)
            replay_client = FakeClient()
            replayed = analyze(source, replay_client, root / "cache", identity_consensus=False)
            export_bundle(document, root / "bundle")
            exported = json.loads((root / "bundle" / "cast.json").read_text(encoding="utf-8"))

        self.assertEqual(validate_document(document), [])
        self.assertEqual(replayed, document)
        self.assertEqual(replay_client.calls, [])
        legacy_response_voices = [voice for voice in document["cast"]["voices"]
                                  if voice["id"] != "voice_narrator_main"]
        self.assertTrue(all("speechTraits" not in voice for voice in legacy_response_voices))
        exported_legacy = [voice for voice in exported["voices"]
                           if voice["id"] != "voice_narrator_main"]
        self.assertTrue(all("speechTraits" not in voice for voice in exported_legacy))

    def test_series_continuation_preserves_field_in_context_and_reused_voice(self):
        first = fixture()
        first["cast"]["voices"][1]["speechTraits"] = "specific rural Maine accent; deliberate rhythm"
        current = fixture(sequel=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "series.json"
            registry = load_registry(path)
            linked_first = link_analysis(first, registry)
            save_registry(path, registry, linked_first)
            registry = load_registry(path)
            old_character = linked_first["cast"]["characters"][0]
            propose(current, old_character["id"])
            linked_current = link_analysis(current, registry)

        context_voice = next(voice for voice in series_context(registry, current["source"])["voices"]
                             if voice["id"] == old_character["voice_id"])
        reused = next(voice for voice in linked_current["cast"]["voices"]
                      if voice["id"] == old_character["voice_id"])
        self.assertEqual(context_voice["speechTraits"], first["cast"]["voices"][1]["speechTraits"])
        self.assertEqual(reused["speechTraits"], first["cast"]["voices"][1]["speechTraits"])

    def test_legacy_series_context_keeps_bounded_voice_shape(self):
        first = fixture()
        first["cast"]["voices"][1]["sample_script"] = "Metadata not sent to the model."
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "series.json"
            registry = load_registry(path)
            linked = link_analysis(first, registry)
            save_registry(path, registry, linked)
            registry = load_registry(path)

        voice_id = linked["cast"]["characters"][0]["voice_id"]
        context_voice = next(voice for voice in series_context(registry, first["source"])["voices"]
                             if voice["id"] == voice_id)
        self.assertEqual(set(context_voice), {"id", "description", "traits", "source_fingerprint"})

    def test_first_traits_cannot_be_added_only_to_a_descendant_voice(self):
        for voice_index in (0, 1):
            with self.subTest(voice_index=voice_index), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "series.json"
                registry = load_registry(path)
                first = link_analysis(fixture(), registry)
                save_registry(path, registry, first)
                registry = load_registry(path)
                original = deepcopy(registry)
                current = fixture(sequel=True)
                propose(current, first["cast"]["characters"][0]["id"])
                current["cast"]["voices"][voice_index]["speechTraits"] = "Glaswegian Scottish accent"
                owner = first["cast"]["voices"][voice_index]["id"]

                with self.assertRaisesRegex(ValueError, "originating book") as raised:
                    link_analysis(current, registry)

                self.assertIn(owner, str(raised.exception))
                self.assertIn(first["source"]["book"]["title"], str(raised.exception))
                self.assertEqual(registry, original)

    def test_later_book_refinements_are_retained_and_exposed_to_future_books(self):
        initial_traits = "Neutral General American accent; measured pace"
        refined_traits = "Rural Maine accent; clipped, measured phrasing"
        original = fixture()
        original["cast"]["voices"][1]["speechTraits"] = initial_traits
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "series.json"
            registry = load_registry(path)
            first = link_analysis(original, registry)
            save_registry(path, registry, first)
            registry = load_registry(path)
            current = fixture(sequel=True)
            old_character = first["cast"]["characters"][0]
            propose(current, old_character["id"])
            current["cast"]["voices"][1]["speechTraits"] = refined_traits

            later = link_analysis(current, registry)
            save_registry(path, registry, later)
            registry = load_registry(path)
            context = series_context(registry, current["source"])
            following = fixture(sequel=True)
            following["source"]["book"].update(id="book3", title="A Third Walk")
            propose(following, old_character["id"])
            continued = link_analysis(following, registry)

        later_voice = next(voice for voice in later["cast"]["voices"]
                           if voice["id"] == old_character["voice_id"])
        context_voice = next(voice for voice in context["voices"]
                             if voice["id"] == old_character["voice_id"])
        self.assertEqual(later_voice["speechTraits"], refined_traits)
        self.assertEqual(context_voice["speechTraits"], refined_traits)
        continued_voice = next(voice for voice in continued["cast"]["voices"]
                               if voice["id"] == old_character["voice_id"])
        self.assertEqual(continued_voice["speechTraits"], refined_traits)
        self.assertEqual(first["cast"]["voices"][1]["speechTraits"], initial_traits)
        self.assertEqual(registry["books"][0]["cast"]["voices"][1]["speechTraits"], initial_traits)


if __name__ == "__main__":
    unittest.main()
