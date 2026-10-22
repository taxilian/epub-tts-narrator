"""Speech reuse follows audible inputs, not a passage's editorial location."""

from copy import deepcopy
from pathlib import Path
import unittest

from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.engines import format_dialogue


def reference(name="first", **changes):
    return {
        "kind": "reference", "asset_hash": digest(name), "voice_id": name,
        "reference_sha256": digest({"audio": name}),
        "reference_text": f"Reference speech for {name}.",
        "reference_audio": str(Path("/voices") / name / "reference.wav"),
        "profile": {"description": "A fictional character."},
        **changes,
    }


def speech_request(voice, text="Hello.", **changes):
    return {
        "kind": "speech",
        "engine": {"engine": "drama3_mlx_audio", "model": "example/drama",
                   "checkpoint_files_sha256": {"weights.safetensors": "weights"},
                   "runtime": "mlx-audio", "installed_version": "1.0",
                   "temperature": 1.0, "top_p": .9, "top_k": 30},
        "base_seed": 42, "sample_rate": 44100,
        "voice_asset_hash": voice["asset_hash"],
        "chunk": {"chunk_id": "chunk-one", "spoken_text": text,
                  "spoken_start": 0, "spoken_end": len(text), "silent_control": False,
                  "source_ranges": [{"start": 0, "end": len(text)}],
                  "segment": {"sequence": 1, "chapter_id": "chapter-one",
                              "block_id": "block-one", "role_id": "role-one",
                              "voice_id": voice["voice_id"], "kind": "dialogue",
                              "delivery": {"direction": "", "evidence": []}}},
        **changes,
    }


def paragraph_request(turns):
    result = speech_request(turns[0][1])
    result.pop("voice_asset_hash")
    result["kind"] = "paragraph_speech"
    result["version"] = 1
    result["voice_assets"] = {voice["voice_id"]: voice["asset_hash"] for _, voice, _ in turns}
    result["chunk"]["paragraph_turns"] = []
    for index, (text, voice, direction) in enumerate(turns):
        chunk = speech_request(voice, text)["chunk"]
        chunk["chunk_id"] = f"turn-{index}"
        chunk["segment"]["delivery"]["direction"] = direction
        result["chunk"]["paragraph_turns"].append(chunk)
    result["chunk"]["spoken_text"] = "".join(text for text, _, _ in turns)
    return result


class SpeechIdentityTests(unittest.TestCase):
    def key(self, request, *voices, **kwargs):
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        return digest(generation_identity(request, {voice["asset_hash"]: voice for voice in voices}, **kwargs))

    def test_voice_metadata_changes_do_not_split_dialogue_speaker(self):
        first = reference()
        alias = {**first, "asset_hash": digest("new-profile"), "voice_id": "alias",
                 "profile": {"description": "An updated description."},
                 "reference_audio": "/moved/reference.wav"}
        text, voices, _ = format_dialogue([
            {"text": "Hello, ", "voice": first},
            {"text": "again.", "voice": alias},
        ])
        self.assertEqual(text, "<|speaker:0|> Hello, again.")
        self.assertEqual(len(voices), 1)

    def test_position_and_attribution_changes_reuse_speech(self):
        voice = reference()
        original = speech_request(voice)
        moved = deepcopy(original)
        moved["chunk"].update(chunk_id="new-chunk", spoken_start=10, spoken_end=16,
                              source_ranges=[{"start": 100, "end": 106}])
        moved["chunk"]["segment"].update(sequence=20, chapter_id="chapter-nine",
            block_id="new-block", role_id="renamed-role", reasoning="New attribution evidence",
            kind="thought", evidence=[{"block_id": "evidence", "quote": "He said"}])
        moved["performer"] = {"role_id": "someone", "voice_id": "alias"}
        self.assertEqual(self.key(original, voice), self.key(moved, voice))

    def test_equivalent_voice_content_survives_profile_and_location_changes(self):
        voice = reference()
        revised = {**voice, "asset_hash": digest("revised"), "voice_id": "renamed",
                   "reference_audio": "/new/path.wav", "profile": {"description": "Edited"},
                   "creation": {"designer": "different"}, "verification": {"passed": True}}
        self.assertEqual(self.key(speech_request(voice), voice),
                         self.key(speech_request(revised), revised))

    def test_wrapping_formatting_is_not_a_new_performance(self):
        voice = reference()
        self.assertEqual(self.key(speech_request(voice, "Lie,"), voice),
                         self.key(speech_request(voice, "  ‘(Lie,  "), voice))

    def test_only_applied_delivery_changes_the_performance(self):
        voice = reference()
        plain = speech_request(voice)
        emotional = deepcopy(plain)
        emotional["chunk"]["segment"]["delivery"]["direction"] = "with resolve"
        self.assertEqual(self.key(plain, voice), self.key(emotional, voice))
        whisper = deepcopy(plain)
        whisper["chunk"]["segment"]["delivery"]["direction"] = "whisper"
        synonym = deepcopy(whisper)
        synonym["chunk"]["segment"]["delivery"]["direction"] = "whispered"
        self.assertNotEqual(self.key(plain, voice), self.key(whisper, voice))
        self.assertEqual(self.key(whisper, voice), self.key(synonym, voice))

    def test_changed_generation_inputs_do_not_reuse_speech(self):
        voice = reference()
        original = speech_request(voice)
        variants = []
        for field, value in (("spoken_text", "Goodbye."),):
            variant = deepcopy(original)
            variant["chunk"][field] = value
            variants.append(variant)
        for field, value in (("base_seed", 99), ("sample_rate", 48000)):
            variants.append({**original, field: value})
        for field, value in (("temperature", .8), ("top_p", .8), ("top_k", 20),
                             ("installed_version", "2.0"),
                             ("checkpoint_files_sha256", {"weights.safetensors": "different"})):
            variant = deepcopy(original)
            variant["engine"][field] = value
            variants.append(variant)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(self.key(original, voice), self.key(variant, voice))
        for field, value in (("reference_text", "Different reference speech."),
                             ("reference_sha256", digest("different-audio"))):
            changed = {**voice, field: value}
            self.assertNotEqual(self.key(original, voice), self.key(original, changed))

    def test_model_location_and_requirements_do_not_invalidate_identical_weights(self):
        voice = reference()
        original = speech_request(voice)
        changed = deepcopy(original)
        changed["engine"].update(checkpoint="/new/models/drama", model_revision="different-metadata",
            recommended_model_revision="updated-recommendation", required_runtime_version=">=2",
            packages={"numpy": "different"}, min_audio_seconds=.2,
            completion_check="new guard")
        self.assertEqual(self.key(original, voice), self.key(changed, voice))

    def test_remote_worker_capability_is_not_a_single_speech_input(self):
        voice = reference()
        original = speech_request(voice, engine={"engine": "fish_http", "model": "drama-3",
            "protocol": "fish-inline-reference-wav-v1", "sampling": {"temperature": 1.0},
            "multi_speaker": True})
        changed = deepcopy(original)
        changed["engine"]["multi_speaker"] = False
        self.assertEqual(self.key(original, voice), self.key(changed, voice))

    def test_paragraph_reuses_equivalent_turns_after_role_and_voice_metadata_edits(self):
        first, second = reference(), reference("second")
        revised = {**first, "asset_hash": digest("revised"), "voice_id": "alias"}
        original = paragraph_request([("Hello. ", first, ""), ("Stay here.", second, "whisper")])
        changed = paragraph_request([("Hello. ", revised, "warmly"), ("Stay here.", second, "whispering")])
        changed["chunk"]["paragraph_turns"][0]["segment"]["block_id"] = "new-location"
        self.assertEqual(self.key(original, first, second), self.key(changed, revised, second))

    def test_paragraph_keeps_word_boundaries_and_turn_order_in_identity(self):
        first, second = reference(), reference("second")
        spaced = paragraph_request([("look ", first, ""), ("ed", first, "")])
        joined = paragraph_request([("look", first, ""), ("ed", first, "")])
        self.assertNotEqual(self.key(spaced, first), self.key(joined, first))
        original = paragraph_request([("Hello. ", first, ""), ("Goodbye.", second, "")])
        reversed_turns = paragraph_request([("Goodbye. ", second, ""), ("Hello.", first, "")])
        swapped_voices = paragraph_request([("Hello. ", second, ""), ("Goodbye.", first, "")])
        self.assertNotEqual(self.key(original, first, second), self.key(reversed_turns, first, second))
        self.assertNotEqual(self.key(original, first, second), self.key(swapped_voices, first, second))

    def test_paragraph_alias_uses_one_reference_and_same_engine_text(self):
        voice = reference()
        alias = {**voice, "asset_hash": digest("alias"), "voice_id": "alias"}
        original = paragraph_request([("Hello, ", voice, ""), ("again.", voice, "")])
        renamed = paragraph_request([("Hello, ", voice, ""), ("again.", alias, "")])
        self.assertEqual(self.key(original, voice), self.key(renamed, voice, alias))

    def test_unknown_engines_keep_custom_synthesis_inputs(self):
        voice = reference()
        original = speech_request(voice, engine={"engine": "plugin", "custom_setting": "first"})
        direction = deepcopy(original)
        direction["chunk"]["segment"]["delivery"]["direction"] = "warmly"
        custom_voice = {**voice, "plugin_prompt": "Use a bright delivery"}
        custom_engine = {**original, "engine": {"engine": "plugin", "custom_setting": "second"}}
        self.assertNotEqual(self.key(original, voice), self.key(direction, voice))
        self.assertNotEqual(self.key(original, voice), self.key(original, custom_voice))
        self.assertNotEqual(self.key(original, voice), self.key(custom_engine, voice))

    def test_missing_or_mismatched_voice_binding_cannot_be_canonicalized(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        voice = reference()
        request = speech_request(voice)
        with self.assertRaises(ValueError):
            generation_identity(request)
        with self.assertRaises(ValueError):
            generation_identity(request, {voice["asset_hash"]: reference("wrong")})

    def test_accepted_legacy_edge_formatting_can_reuse_the_current_input(self):
        voice = reference()
        request = speech_request(voice, "(Lie,")
        current = self.key(request, voice)
        for recorded in ("Lie,", "(Lie,", "  ‘Lie, ’  "):
            self.assertEqual(current, self.key(request, voice,
                             engine_metadata={"synthesis_text": recorded}))
        for recorded in ("Unrelated speech", "Lie.", "<whisper>Lie,</whisper>"):
            with self.subTest(recorded=recorded), self.assertRaises(ValueError):
                self.key(request, voice, engine_metadata={"synthesis_text": recorded})

    def test_local_and_http_recorded_delivery_text_are_accepted(self):
        voice = reference()
        request = speech_request(voice)
        request["chunk"]["segment"]["delivery"]["direction"] = "whisper"
        key = self.key(request, voice)
        for actual in ("Hello.", "<whisper>Hello.</whisper>"):
            self.assertEqual(key, self.key(request, voice, engine_metadata={"synthesis_text": actual}))

    def test_saved_take_with_different_actual_delivery_cannot_be_relabelled(self):
        voice = reference()
        request = speech_request(voice)
        request["chunk"]["segment"]["delivery"]["direction"] = "whisper"
        with self.assertRaises(ValueError):
            self.key(request, voice, engine_metadata={"synthesis_text": "Hello.",
                                                      "applied_control": None})

    def test_legacy_paragraph_with_different_actual_speaker_grouping_is_not_reused(self):
        voice = reference()
        alias = {**voice, "asset_hash": digest("alias"), "voice_id": "alias"}
        request = paragraph_request([("Hello, ", voice, ""), ("again.", alias, "")])
        with self.assertRaises(ValueError):
            self.key(request, voice, alias, engine_metadata={"speaker_count": 2,
                                                            "rendered_turn_count": 2})

    def test_identity_does_not_mutate_provenance(self):
        voice = reference()
        request = paragraph_request([("Hello. ", voice, ""), ("Again.", voice, "")])
        before = deepcopy((request, voice))
        self.key(request, voice)
        self.assertEqual((request, voice), before)


if __name__ == "__main__":
    unittest.main()
