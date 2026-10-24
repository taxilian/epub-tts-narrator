from copy import deepcopy
import unittest

from test_renderer_speech_identity import paragraph_request, reference, speech_request


TRAITS = "Specific northern English accent; measured, clipped rhythm"


class RendererSpeechTraitsTests(unittest.TestCase):
    def test_logical_voice_traits_remain_distinct_when_reference_audio_is_aliased(self):
        from epub_tts_narrator.renderer.speech_traits import bind_plan_speech_traits

        first = reference("canonical")
        second = reference("second")
        single = speech_request(first)["chunk"]
        single["segment"]["voice_id"] = "alias"
        simultaneous = deepcopy(single)
        simultaneous["chunk_id"] = "simultaneous"
        simultaneous["segment"].update(
            role_id=None,
            voice_id=None,
            simultaneous_speakers=[
                {"role_id": "one", "voice_id": "alias"},
                {"role_id": "two", "voice_id": second["voice_id"]},
            ],
        )
        paragraph = paragraph_request([
            ("First. ", first, "softly"),
            ("Second.", second, "quickly"),
        ])["chunk"]
        plan = {"chapters": [{"chunks": [single, simultaneous, paragraph]}]}
        original = deepcopy(plan)
        cast = {"voices": [
            {"id": "canonical", "speechTraits": TRAITS},
            {"id": "alias", "speechTraits": "Mexican Spanish-influenced English accent; clipped phrasing"},
            {"id": "second"},
        ]}

        conditioned = bind_plan_speech_traits(plan, cast)

        self.assertEqual(plan, original)
        chunks = conditioned["chapters"][0]["chunks"]
        expected = {"alias": cast["voices"][1]["speechTraits"]}
        self.assertEqual(chunks[0]["speech_traits"], expected)
        self.assertEqual(chunks[1]["speech_traits"], expected)
        self.assertEqual(chunks[2]["paragraph_turns"][0]["speech_traits"], {"canonical": TRAITS})
        self.assertNotIn("speech_traits", chunks[2]["paragraph_turns"][1])

    def test_traits_prepend_scene_direction_and_absent_traits_preserve_legacy_direction(self):
        from epub_tts_narrator.renderer.speech_traits import performance_direction

        voice = reference()
        chunk = speech_request(voice)["chunk"]
        chunk["segment"]["delivery"]["direction"] = "with restrained concern"

        self.assertEqual(performance_direction(chunk, voice["voice_id"]), "with restrained concern")
        chunk["speech_traits"] = {voice["voice_id"]: TRAITS}
        self.assertEqual(
            performance_direction(chunk, voice["voice_id"]),
            TRAITS + "; with restrained concern",
        )

    def test_traits_survive_plain_delivery_and_bind_generation_identity(self):
        from epub_tts_narrator.renderer.delivery_recovery import plain_delivery_request
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        from epub_tts_narrator.renderer.speech_traits import performance_direction

        voice = reference()
        request = speech_request(voice)
        request["chunk"]["segment"]["delivery"]["direction"] = "urgently"
        baseline = generation_identity(request, {voice["asset_hash"]: voice})
        request["chunk"]["speech_traits"] = {voice["voice_id"]: TRAITS}

        conditioned = generation_identity(request, {voice["asset_hash"]: voice})
        plain = plain_delivery_request(request)

        self.assertNotEqual(conditioned, baseline)
        self.assertEqual(performance_direction(plain["chunk"], voice["voice_id"]), TRAITS)
        self.assertEqual(conditioned["voice"], baseline["voice"])

    def test_paragraph_traits_condition_each_turn_and_change_dialogue_identity(self):
        from epub_tts_narrator.renderer.engines import format_dialogue
        from epub_tts_narrator.renderer.speech_identity import generation_identity, paragraph_rewrite_turn
        from epub_tts_narrator.renderer.speech_traits import performance_direction

        voice = reference()
        request = paragraph_request([("Hello, ", voice, "softly"), ("again.", voice, "softly")])
        voices = {voice["asset_hash"]: voice}
        baseline = generation_identity(request, voices)
        for turn in request["chunk"]["paragraph_turns"]:
            turn["speech_traits"] = {voice["voice_id"]: TRAITS}

        conditioned = generation_identity(request, voices)
        collapsed = paragraph_rewrite_turn(request, voices)
        rendered, _, _ = format_dialogue([collapsed])

        self.assertNotEqual(conditioned, baseline)
        self.assertIn("[" + TRAITS + "; softly]", rendered)
        self.assertEqual(
            collapsed["direction"],
            performance_direction(request["chunk"]["paragraph_turns"][0], voice["voice_id"]),
        )

    def test_simultaneous_performer_selects_its_own_traits_for_cache_identity(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity

        first = reference("first")
        second = reference("second")
        chunk = speech_request(first)["chunk"]
        chunk["segment"].update(role_id=None, voice_id=None, simultaneous_speakers=[
            {"role_id": "first-role", "voice_id": "first"},
            {"role_id": "second-role", "voice_id": "second"},
        ])
        chunk["speech_traits"] = {"first": "slow cadence", "second": "rapid cadence"}
        voices = {first["asset_hash"]: first, second["asset_hash"]: second}

        first_request = speech_request(first, chunk=chunk,
                                       performer={"role_id": "first-role", "voice_id": "first"})
        second_request = speech_request(second, chunk=chunk,
                                        performer={"role_id": "second-role", "voice_id": "second"})

        first_identity = generation_identity(first_request, voices)
        second_identity = generation_identity(second_request, voices)
        self.assertTrue(first_identity["text"].startswith("[slow cadence]"))
        self.assertTrue(second_identity["text"].startswith("[rapid cadence]"))

    def test_overlong_combined_traits_and_scene_direction_fail_closed(self):
        from epub_tts_narrator.renderer.speech_traits import performance_direction

        voice = reference()
        chunk = speech_request(voice)["chunk"]
        chunk["speech_traits"] = {voice["voice_id"]: "a" * 400}
        chunk["segment"]["delivery"]["direction"] = "b" * 800

        with self.assertRaisesRegex(ValueError, "exceeds 1200"):
            performance_direction(chunk, voice["voice_id"])

    def test_trait_changes_invalidate_plan_and_remote_selection_while_absence_is_legacy_identical(self):
        from epub_tts_narrator.renderer.chapter_client import _selection
        from epub_tts_narrator.renderer.common import digest
        from epub_tts_narrator.renderer.speech_traits import bind_plan_speech_traits

        voice = reference()
        chunk = speech_request(voice)["chunk"]
        plan = {"chapters": [{"number": 1, "id": "chapter-0001", "title": "One",
                              "chunks": [chunk]}]}
        cast = {"voices": [{"id": voice["voice_id"]}]}
        legacy = bind_plan_speech_traits(plan, cast)
        self.assertEqual(legacy, plan)

        cast["voices"][0]["speechTraits"] = TRAITS
        first = bind_plan_speech_traits(plan, cast)
        cast["voices"][0]["speechTraits"] = "Precise Parisian French accent; flowing pace"
        second = bind_plan_speech_traits(plan, cast)
        bundle = {"source": {"book": {"title": "Book"}},
                  "manifest": {"source_fingerprint": "a" * 64}, "chapters": [{}]}
        config = {"render": {}}
        bindings = {voice["voice_id"]: voice["asset_hash"]}

        self.assertNotEqual(digest(first), digest(second))
        self.assertNotEqual(
            digest(_selection(bundle, first["chapters"][0], config, bindings)),
            digest(_selection(bundle, second["chapters"][0], config, bindings)),
        )

    def test_remote_protocol_accepts_bound_traits_and_rejects_malformed_trait_provenance(self):
        from epub_tts_narrator.renderer.chapter_protocol import validate_spec
        from epub_tts_narrator.spoken_text import spoken_source_ranges

        voice = reference()
        chunk = speech_request(voice)["chunk"]
        text = chunk["spoken_text"]
        chunk["chunk_id"] = "1" * 32
        chunk["segment"].update(id="segment-one", text=text, start=0, end=len(text))
        chunk["source_ranges"] = spoken_source_ranges(chunk["segment"], 0, len(text))
        chunk["speech_traits"] = {voice["voice_id"]: TRAITS}
        spec = {
            "version": 1, "book": {"title": "Book"},
            "source_fingerprint": "a" * 64, "renderer_fingerprint": "b" * 64,
            "chapter": {"number": 1, "id": "chapter-0001", "title": "One", "chunks": [chunk]},
            "total_chapters": 1, "voice_bindings": {voice["voice_id"]: voice["asset_hash"]},
            "render": {"chapter_format": "m4a"},
        }

        self.assertEqual(validate_spec(spec), spec)
        for malformed in ({"other": TRAITS}, {voice["voice_id"]: "[whisper]"}, {}):
            changed = deepcopy(spec)
            changed["chapter"]["chunks"][0]["speech_traits"] = malformed
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                validate_spec(changed)

    def test_sentence_and_fragment_recovery_children_retain_traits(self):
        from epub_tts_narrator.renderer.config import DEFAULTS
        from epub_tts_narrator.renderer.fragment_recovery import _piece_request, fragment_requests
        from epub_tts_narrator.renderer.speech_identity import single_speech_recovery_part
        from epub_tts_narrator.renderer.speech_recovery import split_recovery_text
        from epub_tts_narrator.renderer.speech_traits import performance_direction

        voice = reference()
        text = "Please stay here—now. We can leave later."
        parent = speech_request(voice, text)
        parent["chunk"]["speech_traits"] = {voice["voice_id"]: TRAITS}
        sentence = split_recovery_text(text)[0]
        actual = single_speech_recovery_part(parent, sentence)
        child = {
            "kind": "speech_recovery_part", "version": 1,
            "parent_request_hash": "a" * 64, "recovery_request_hash": "b" * 64,
            "part": sentence, "synthesis_request": actual,
        }

        planned = fragment_requests(sentence["text"], child, DEFAULTS)
        fragment_request = planned[0]
        fragment = fragment_request["recovery"]["parts"][0]
        fragment_actual, _ = _piece_request(child, fragment_request, fragment)

        self.assertEqual(performance_direction(actual["chunk"], voice["voice_id"]), TRAITS)
        self.assertEqual(fragment_actual["chunk"]["speech_traits"], {voice["voice_id"]: TRAITS})
