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


class PoolSpeechCompatibilityTests(unittest.TestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.drama3_models import get_drama3_model
        self.voice = reference()
        self.local = {
            "engine": "drama3_mlx", "model": "fishaudio/drama-3",
            "model_revision": get_drama3_model("fishaudio/drama-3").revision,
            "checkpoint_files_sha256": {"weights.safetensors": digest("original-weights")},
            "runtime": "Sogni-AI/drama3-mlx", "installed_version": "1.0",
            "adapter_version": 1, "experts": "q8", "temperature": 1.0,
            "top_p": .9, "top_k": 30, "max_new_tokens": 2048,
        }
        self.pool = {
            "engine": "fish_tts_pool", "model": self.local["model"], "contract_version": 1,
            "model_revision": self.local["model_revision"],
            "sampling": {field: self.local[field] for field in (
                "experts", "temperature", "top_p", "top_k", "max_new_tokens")},
        }

    def compatible(self, saved, current, *, metadata=None, saved_request=None, current_request=None):
        from epub_tts_narrator.renderer.speech_identity import generation_compatible, generation_identity
        saved_request = saved_request or speech_request(self.voice, engine=saved)
        current_request = current_request or speech_request(self.voice, engine=current)
        voices = {self.voice["asset_hash"]: self.voice}
        return generation_compatible(
            generation_identity(saved_request, voices), generation_identity(current_request, voices),
            saved_engine=saved, current_engine=current, engine_metadata=metadata)

    def test_switching_to_pool_and_back_reuses_same_model_contract(self):
        self.assertTrue(self.compatible(self.local, self.pool))
        self.assertTrue(self.compatible(self.pool, self.local))

    def test_pool_model_revision_and_sampling_changes_require_new_speech(self):
        variants = [dict(self.pool, model="mlx-community/fish-audio-drama-3-4bit"),
                    dict(self.pool, model_revision="another-revision"),
                    dict(self.pool, contract_version=2)]
        for field, value in (("temperature", .5), ("top_p", .7), ("top_k", 20),
                             ("max_new_tokens", 512), ("experts", "q4")):
            variants.append({**self.pool, "sampling": {**self.pool["sampling"], field: value}})
        for changed in variants:
            with self.subTest(changed=changed):
                self.assertFalse(self.compatible(self.local, changed))
                self.assertFalse(self.compatible(changed, self.local))

    def test_pool_cannot_hide_changed_local_weights_or_runtime_when_worker_is_known(self):
        metadata = {"worker_engine": self.local}
        self.assertTrue(self.compatible(self.pool, self.local, metadata=metadata))
        for field, value in (("checkpoint_files_sha256", {"weights.safetensors": digest("new")}),
                             ("installed_version", "2.0"), ("adapter_version", 2)):
            with self.subTest(field=field):
                changed = dict(self.local, **{field: value})
                self.assertFalse(self.compatible(self.pool, changed, metadata=metadata))
                self.assertFalse(self.compatible(self.local, changed))

    def test_unknown_local_revision_requires_exact_recorded_worker(self):
        unknown = dict(self.local, model_revision=None,
                       recommended_model_revision=self.local["model_revision"])
        self.assertFalse(self.compatible(unknown, self.pool))
        self.assertFalse(self.compatible(self.pool, unknown))
        self.assertTrue(self.compatible(self.pool, unknown, metadata={"worker_engine": unknown}))
        changed = dict(unknown, installed_version="2.0")
        self.assertFalse(self.compatible(self.pool, changed, metadata={"worker_engine": unknown}))

    def test_exact_worker_does_not_override_known_revision_mismatch(self):
        local = dict(self.local, model_revision="new-revision")
        self.assertFalse(self.compatible(self.pool, local, metadata={"worker_engine": local}))
        unknown = dict(local, model_revision=None)
        self.assertFalse(self.compatible(self.pool, unknown, metadata={"worker_engine": local}))

    def test_http_worker_uses_declared_pool_contract(self):
        metadata = {"worker_engine": {"engine": "fish_http", "model": "drama-3"}}
        self.assertTrue(self.compatible(self.pool, self.local, metadata=metadata))
        changed = dict(self.pool, model_revision="new-revision")
        self.assertFalse(self.compatible(changed, self.local, metadata=metadata))

    def test_mlx_audio_ignores_experts_but_preserves_real_sampling(self):
        from epub_tts_narrator.renderer.drama3_models import get_drama3_model
        model = get_drama3_model("mlx-community/fish-audio-drama-3-4bit")
        local = dict(self.local, engine="drama3_mlx_audio", model=model.repo,
                     runtime="mlx-audio", model_revision=model.revision)
        local.pop("experts")
        pool = dict(self.pool, model=model.repo, model_revision=model.revision,
                    sampling={**self.pool["sampling"], "experts": "bf16", "min_audio_seconds": .2})
        self.assertTrue(self.compatible(local, pool))
        pool["sampling"]["temperature"] = .5
        self.assertFalse(self.compatible(local, pool))

    def test_pool_does_not_allow_changed_text_voice_seed_or_sample_rate(self):
        for field, value in (("base_seed", 123), ("sample_rate", 48000)):
            request = speech_request(self.voice, engine=self.pool, **{field: value})
            with self.subTest(field=field):
                self.assertFalse(self.compatible(self.local, self.pool, current_request=request))
        request = speech_request(self.voice, text="Something else.", engine=self.pool)
        self.assertFalse(self.compatible(self.local, self.pool, current_request=request))
        from epub_tts_narrator.renderer.speech_identity import generation_compatible, generation_identity
        voices = {self.voice["asset_hash"]: self.voice}
        saved = generation_identity(speech_request(self.voice, engine=self.local), voices)
        current = generation_identity(speech_request(self.voice, engine=self.pool), voices)
        current["voice"]["reference_sha256"] = digest("another-voice")
        self.assertFalse(generation_compatible(saved, current, saved_engine=self.local,
                                              current_engine=self.pool))

    def test_unknown_engines_are_not_assumed_compatible_with_pool(self):
        plugin = dict(self.local, engine="some_plugin")
        self.assertFalse(self.compatible(plugin, self.pool))

    def test_current_canonical_keys_stay_strict_even_when_pool_reuse_is_compatible(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        voices = {self.voice["asset_hash"]: self.voice}
        original = generation_identity(speech_request(self.voice, engine=self.local), voices)
        switched = generation_identity(speech_request(self.voice, engine=self.pool), voices)
        self.assertNotEqual(original, switched)
        self.assertTrue(self.compatible(self.local, self.pool))

    def quantized(self, alias=None):
        from epub_tts_narrator.renderer.drama3_models import get_drama3_model
        model = get_drama3_model("mlx-community/fish-audio-drama-3-4bit")
        engine = dict(self.local, engine="drama3_mlx_audio", model=model.repo,
                      runtime="mlx-audio", model_revision=model.revision,
                      checkpoint_files_sha256={"weights.safetensors": digest("quantized")})
        engine.pop("experts")
        if alias is not None:
            engine["cache_model_alias"] = alias
        return engine

    def test_explicit_original_model_alias_reuses_old_full_model_with_quantized_model(self):
        self.assertFalse(self.compatible(self.local, self.quantized()))
        self.assertTrue(self.compatible(self.local, self.quantized(self.local["model"])))

    def test_arbitrary_alias_does_not_claim_unlabelled_old_recordings(self):
        self.assertFalse(self.compatible(self.local, self.quantized("my-audiobook-model")))

    def test_same_explicit_alias_uses_same_key_across_ports_weights_and_revision(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity
        saved = dict(self.local, cache_model_alias="my-audiobook-model")
        current = self.quantized("my-audiobook-model")
        voices = {self.voice["asset_hash"]: self.voice}
        original = generation_identity(speech_request(self.voice, engine=saved), voices)
        switched = generation_identity(speech_request(self.voice, engine=current), voices)
        self.assertEqual(original, switched)
        self.assertTrue(self.compatible(saved, current))

    def test_saved_alias_alone_does_not_relax_unaliased_current_model(self):
        saved = dict(self.local, cache_model_alias=self.quantized()["model"])
        self.assertFalse(self.compatible(saved, self.quantized()))
        self.assertTrue(self.compatible(saved, self.local))
        self.assertFalse(self.compatible(saved, dict(self.local, installed_version="2.0")))

    def test_alias_does_not_transitively_relabel_another_logical_model(self):
        saved = self.quantized(self.local["model"])
        current = dict(self.local, cache_model_alias=saved["model"])
        self.assertFalse(self.compatible(saved, current))

    def test_alias_keeps_common_sampling_as_cache_boundary(self):
        current = self.quantized(self.local["model"])
        self.assertTrue(self.compatible(self.local, current))
        for field, value in (("temperature", .5), ("top_p", .7), ("top_k", 20),
                             ("max_new_tokens", 512)):
            with self.subTest(field=field):
                self.assertFalse(self.compatible(self.local, dict(current, **{field: value})))

    def test_alias_keeps_text_reference_seed_and_sample_rate_as_cache_boundaries(self):
        from epub_tts_narrator.renderer.speech_identity import generation_compatible, generation_identity
        engine = self.quantized(self.local["model"])
        voices = {self.voice["asset_hash"]: self.voice}
        saved = generation_identity(speech_request(self.voice, engine=self.local), voices)
        original = generation_identity(speech_request(self.voice, engine=engine), voices)
        variants = [dict(original, text="Different."), dict(original, base_seed=101),
                    dict(original, sample_rate=48000),
                    dict(original, voice={**original["voice"], "reference_sha256": digest("new-voice")}),
                    dict(original, voice={**original["voice"], "reference_text": "Different reference."})]
        for changed in variants:
            with self.subTest(changed=changed):
                self.assertFalse(generation_compatible(saved, changed, saved_engine=self.local,
                                                      current_engine=engine))

    def test_unknown_plugin_alias_preserves_plugin_generation_identity(self):
        from epub_tts_narrator.renderer.speech_identity import synthesis_engine_identity
        saved = dict(self.local, engine="plugin", cache_model_alias="custom")
        current = dict(saved, model="different", installed_version="2.0")
        self.assertNotEqual(synthesis_engine_identity(saved), synthesis_engine_identity(current))
        self.assertFalse(self.compatible(saved, current))


if __name__ == "__main__":
    unittest.main()
