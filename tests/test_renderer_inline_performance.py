"""Inline attribution controls drive planning, compilation, and remote validation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.performance_planning import prepare_performance
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.chapter_protocol import validate_spec
from epub_tts_narrator.renderer.common import digest
from epub_tts_narrator.renderer.engines import prepare_synthesis
from epub_tts_narrator.renderer.performance_compiler import (
    compile_performance, performance_kwargs,
)
from epub_tts_narrator.renderer.performance_runtime import (
    bind_chunk_performance, row_performance, verification_options,
)
from epub_tts_narrator.renderer.pronunciation_runtime import merge_row_pronunciations
from epub_tts_narrator.renderer.workflow import make_render_plan
from epub_tts_narrator.spoken_text import build_spoken
from shared_fixtures import renderer_config
from test_contracts import valid_document
from test_performance_planning import FakeClient
from test_renderer_speech_identity import reference, speech_request

NARRATION = ("They left the hall quietly, and no one spoke of the letter "
             "again until the following morning came.")


def spoken_record(row, controls):
    spoken = build_spoken(row)[0]
    entries = []
    for control in controls:
        item = dict(control)
        if "start" in item:
            item["text"] = spoken[item["start"]:item["end"]]
        entries.append(item)
    return {"version": 1, "text_sha256": text_sha256(spoken), "controls": entries}


def inline_document(mode):
    document = valid_document()
    block = document["source"]["blocks"][1]
    block["text"] = NARRATION
    narration = document["segments"][2]
    narration.update(start=0, end=len(NARRATION), text=NARRATION)
    if mode == "plain":
        return document
    if mode == "empty":
        narration["performance_controls"] = spoken_record(narration, [])
        return document
    quiet = NARRATION.index("quietly")
    hushed = NARRATION.index("no one spoke")
    narration["performance_controls"] = spoken_record(narration, [
        {"kind": "pronunciation", "start": quiet, "end": quiet + len("quietly"),
         "alphabet": "respelling", "replacement": "KWY-et-lee"},
        {"kind": "delivery", "start": hushed, "end": hushed + len("no one spoke"),
         "tag": "whisper", "direction": ""},
        {"kind": "vocalization", "offset": NARRATION.index("letter"), "sound": "sigh"},
    ])
    dialogue = document["segments"][1]
    if mode == "conflict":
        dialogue["performance_controls"] = spoken_record(dialogue, [
            {"kind": "delivery", "start": 0, "end": 4, "tag": "whisper", "direction": ""}])
        return document
    dialogue["performance_controls"] = spoken_record(dialogue, [
        {"kind": "delivery", "start": 1, "end": 6, "tag": "soft", "direction": ""},
    ])
    return document


class InlinePerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def bundle_for(self, document, name):
        directory = self.root / name
        export_bundle(document, directory, allow_review=True)
        return load_bundle(directory)

    def settings(self, **render):
        config = renderer_config(self.root, series_id="fixture")
        config["render"].update(max_chunk_chars=40, **render)
        return config

    @staticmethod
    def delivery_union(chunks, tag):
        intervals = sorted((chunk["spoken_start"] + control["start"],
                            chunk["spoken_start"] + control["end"])
                           for chunk in chunks
                           for control in chunk.get("expressive_performance", {}).get("controls", [])
                           if control["kind"] == "delivery" and control["tag"] == tag)
        merged = []
        for start, end in intervals:
            if merged and merged[-1][1] == start:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return merged

    def test_legacy_mode_binds_inline_cues_exactly_once_without_a_prepared_plan(self):
        plain = self.bundle_for(inline_document("plain"), "plain-book")
        book = self.bundle_for(inline_document("inline"), "inline-book")
        settings = self.settings()
        baseline = make_plan(plain, settings)
        plan = make_plan(book, settings)
        for before, after in zip(baseline["chapters"][0]["chunks"], plan["chapters"][0]["chunks"]):
            for key in ("chunk_id", "spoken_text", "source_ranges"):
                self.assertEqual(before[key], after[key])
        chunks = plan["chapters"][0]["chunks"]
        self.assertNotIn("expressive_performance", chunks[0])
        self.assertTrue(any("expressive_performance" in chunk for chunk in chunks[1:]))
        quiet = (NARRATION.index("quietly"), NARRATION.index("quietly") + len("quietly"))
        sighs = [(chunk["spoken_start"] + control["offset"], control["sound"])
                 for chunk in chunks
                 for control in chunk.get("expressive_performance", {}).get("controls", [])
                 if control["kind"] == "vocalization"]
        self.assertEqual(sighs, [(NARRATION.index("letter"), "sigh")])
        self.assertEqual(self.delivery_union(chunks, "whisper"),
                         [[NARRATION.index("no one spoke"),
                           NARRATION.index("no one spoke") + len("no one spoke")]])
        self.assertEqual(self.delivery_union(chunks, "soft"), [[1, 6]])
        for chunk in chunks:
            for control in chunk.get("expressive_performance", {}).get("controls", []):
                if control["kind"] == "pronunciation":
                    span = (control["start"] + chunk["spoken_start"], control["end"] + chunk["spoken_start"])
                    self.assertEqual(span, quiet)

    def test_inline_cues_compile_through_the_engine_boundary_for_each_speaker(self):
        book = self.bundle_for(inline_document("inline"), "compiled-book")
        plan = make_plan(book, self.settings())
        chunks = plan["chapters"][0]["chunks"]
        dialogue = chunks[1]
        prepared, directed, metadata = prepare_synthesis(
            dialogue["spoken_text"], "", **performance_kwargs(dialogue, dialogue["segment"]["voice_id"]))
        self.assertEqual(metadata["applied_control"], "structured_performance")
        compiled = metadata["expressive_performance"]
        self.assertEqual(compiled["expected_text"], prepared)
        self.assertIn("<soft>Hello", directed)
        self.assertIn("Hello", compiled["expected_text"])
        self.assertNotIn(dialogue["segment"]["role_id"], directed)
        self.assertNotIn(dialogue["segment"]["voice_id"], directed)
        hushed = (NARRATION.index("no one spoke"), NARRATION.index("no one spoke") + len("no one spoke"))
        for chunk in chunks[2:]:
            controls = chunk.get("expressive_performance", {}).get("controls", [])
            prepared, directed, metadata = prepare_synthesis(
                chunk["spoken_text"], "", **performance_kwargs(chunk, "vn"))
            if not controls:
                self.assertNotIn("expressive_performance", metadata)
                continue
            compiled = metadata["expressive_performance"]
            self.assertEqual(compiled["expected_text"], prepared)
            if any(control["kind"] == "delivery" for control in controls):
                self.assertIn("<whisper>", directed)
                clipped = max(chunk["spoken_start"], hushed[0]), min(chunk["spoken_end"], hushed[1])
                self.assertGreater(clipped[1], clipped[0])
            if any(control["kind"] == "vocalization" for control in controls):
                self.assertEqual(directed.count(" [sigh] "), 1)
            if any(control["kind"] == "pronunciation" for control in controls):
                self.assertIn("KWY-et-lee", directed)
                self.assertNotIn("KWY-et-lee", compiled["expected_text"])
                self.assertIn("quietly", compiled["expected_text"])
        self.assertEqual(sum(any(control["kind"] == "vocalization"
                                 for control in chunk.get("expressive_performance", {}).get("controls", []))
                             for chunk in chunks[2:]), 1)

    def test_cache_identity_follows_conditioning_but_not_mere_field_presence(self):
        from epub_tts_narrator.renderer.speech_identity import generation_identity

        voice = reference()
        plain = speech_request(voice)
        voices = {voice["asset_hash"]: voice}
        text = plain["chunk"]["spoken_text"]
        conditioned = deepcopy(plain)
        conditioned["chunk"].update(
            expressive_performance={"version": 1, "text_sha256": text_sha256(text),
                                    "controls": [{"kind": "delivery", "start": 0, "end": len(text),
                                                  "text": text, "tag": "soft", "direction": ""}]},
            expressive_direction="", expressive_traits={})
        self.assertNotEqual(digest(generation_identity(conditioned, voices)),
                            digest(generation_identity(plain, voices)))
        empty = deepcopy(plain)
        empty["chunk"]["segment"]["performance_controls"] = {
            "version": 1, "text_sha256": text_sha256(text), "controls": []}
        self.assertEqual(digest(generation_identity(empty, voices)),
                         digest(generation_identity(plain, voices)))

    def test_paragraph_coalescing_and_remote_validation_keep_controls_exact_once(self):
        book = self.bundle_for(inline_document("inline"), "grouped-book")
        plan = make_render_plan(book, self.settings())
        chapter = plan["chapters"][0]
        groups = [chunk for chunk in chapter["chunks"] if "paragraph_turns" in chunk]
        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertNotIn("performance_controls", group["segment"])
        turns = group["paragraph_turns"]
        quiet = (NARRATION.index("quietly"), NARRATION.index("quietly") + len("quietly"))
        sighs = [(turn["spoken_start"] + control["offset"], control["sound"])
                 for turn in turns
                 for control in turn.get("expressive_performance", {}).get("controls", [])
                 if control["kind"] == "vocalization"]
        self.assertEqual(sighs, [(NARRATION.index("letter"), "sigh")])
        self.assertEqual(self.delivery_union(turns, "whisper"),
                         [[NARRATION.index("no one spoke"),
                           NARRATION.index("no one spoke") + len("no one spoke")]])
        for turn in turns:
            for control in turn.get("expressive_performance", {}).get("controls", []):
                if control["kind"] == "pronunciation":
                    span = (control["start"] + turn["spoken_start"], control["end"] + turn["spoken_start"])
                    self.assertEqual(span, quiet)
        combined = verification_options({"chunk": group})["performance"]
        self.assertEqual(sum(control["kind"] == "vocalization" for control in combined["controls"]), 1)
        joined = []
        for start, end in sorted((control["start"], control["end"])
                                 for control in combined["controls"]
                                 if control["kind"] == "delivery"):
            if joined and joined[-1][1] == start:
                joined[-1][1] = end
            else:
                joined.append([start, end])
        self.assertEqual(joined, [[NARRATION.index("no one spoke"),
                                   NARRATION.index("no one spoke") + len("no one spoke")]])
        bindings = {voice_id: "a" * 64 for voice_id in plan["voice_ids"]}
        spec = {"version": 1, "book": {"title": "Book"},
                "source_fingerprint": "a" * 64, "renderer_fingerprint": "b" * 64,
                "chapter": {"number": 1, "id": chapter["id"], "title": chapter["title"],
                            "chunks": [group]},
                "total_chapters": 1, "voice_bindings": bindings,
                "render": {"chapter_format": "m4a"}}
        self.assertEqual(validate_spec(spec), spec)
        stale = deepcopy(spec)
        stale["chapter"]["chunks"][0]["paragraph_turns"][0]["segment"]["performance_controls"][
            "text_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_spec(stale)

    def test_inline_and_plan_controls_dedup_duplicates_and_fail_conflicts_by_row(self):
        book = self.bundle_for(inline_document("inline"), "merge-book")
        dialogue = book["chapters"][0]["rows"][1]
        record = row_performance(dialogue, None, normalization=False)
        self.assertEqual(len(record["controls"]), 1)
        self.assertEqual(row_performance(dialogue, {"s2": deepcopy(record)},
                                         normalization=False), record)
        spoken = build_spoken(dialogue)[0]
        external = {"version": 1, "text_sha256": text_sha256(spoken), "controls": [
            {"kind": "delivery", "start": 0, "end": 4, "text": spoken[0:4],
             "tag": "whisper", "direction": ""}]}
        with self.assertRaisesRegex(ValueError, "s2 has conflicting delivery controls"):
            row_performance(dialogue, {"s2": external}, normalization=False)
        narration = book["chapters"][0]["rows"][2]
        spoken = build_spoken(narration)[0]
        plan_record = {"version": 1, "text_sha256": text_sha256(spoken), "controls": [
            {"kind": "pause", "offset": 0, "duration_ms": 700}]}
        narration["performance_controls"] = spoken_record(narration, [
            {"kind": "pause", "offset": 0, "duration_ms": 300}])
        with self.assertRaisesRegex(ValueError, "s3 has conflicting pause events"):
            row_performance(narration, {"s3": plan_record}, normalization=False)
        narration["performance_controls"] = spoken_record(narration, [
            {"kind": "pause", "offset": 0, "duration_ms": 700},
            {"kind": "vocalization", "offset": 0, "sound": "sigh"}])
        merged = row_performance(narration, {"s3": plan_record}, normalization=False)
        self.assertEqual(sorted((control["kind"], json.dumps(control, sort_keys=True))
                                for control in merged["controls"]),
                         sorted((control["kind"], json.dumps(control, sort_keys=True))
                                for control in narration["performance_controls"]["controls"]))

    def test_confirmed_pronunciation_merges_last_with_existing_guards(self):
        book = self.bundle_for(inline_document("inline"), "pronunciation-book")
        narration = book["chapters"][0]["rows"][2]
        entry = {"status": "confirmed", "word": "quietly", "variants": ["quietly"],
                 "pronunciation": {"alphabet": "respelling", "value": "KWY-et-lee"}}
        merged = merge_row_pronunciations(narration, row_performance(narration, None,
                                                                     normalization=False),
                                          {"quietly": entry})
        self.assertEqual([control["replacement"] for control in merged["controls"]
                          if control["kind"] == "pronunciation"], ["KWY-et-lee"])
        entry["pronunciation"]["value"] = "KWEE-lee"
        with self.assertRaisesRegex(ValueError, "performance plan"):
            merge_row_pronunciations(narration, row_performance(narration, None,
                                                                normalization=False),
                                     {"quietly": entry})

    def test_stale_inline_records_fail_with_the_row_id(self):
        book = self.bundle_for(inline_document("inline"), "stale-book")
        plan = make_render_plan(book, self.settings())
        chunk = next(item for item in plan["chapters"][0]["chunks"]
                     if "paragraph_turns" not in item and item["segment"]["id"] == "s2")
        chunk["segment"]["performance_controls"]["text_sha256"] = "0" * 64
        from epub_tts_narrator.renderer.chapter_protocol import _leaf
        with self.assertRaisesRegex(ValueError, "s2 carries invalid inline performance"):
            _leaf(chunk, {chunk["segment"]["voice_id"]: "a" * 64})
        narration = book["chapters"][0]["rows"][2]
        narration["performance_controls"]["text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "s3"):
            row_performance(narration, None, normalization=False)

    def test_scoped_style_resets_to_baseline_and_literal_words_are_not_doubled(self):
        text = "She laughed softly, then composed herself."
        record = {"version": 1, "text_sha256": text_sha256(text), "controls": [
            {"kind": "delivery", "start": 0, "end": 20, "text": text[:20],
             "tag": "style", "direction": "with a low chuckle"},
            {"kind": "vocalization", "offset": 0, "sound": "laugh"}]}
        compiled = compile_performance(text, record, traits="measured pace", direction="gently")
        self.assertEqual(compiled["expected_text"], text)
        self.assertEqual(compiled["text"].count(" [laugh] "), 1)
        self.assertIn("laughed", compiled["text"])
        self.assertIn("[with a low chuckle] ", compiled["text"])
        self.assertIn(" [measured pace; gently] ", compiled["text"])
        self.assertGreater(compiled["text"].index(" [measured pace; gently] "),
                           compiled["text"].index("with a low chuckle"))
        self.assertIn("scoped_style_reset_is_approximate", compiled["warnings"])
        applied = next(control for control in compiled["applied_controls"]
                       if control["kind"] == "delivery" and control["tag"] == "style")
        self.assertEqual(applied["scope_accuracy"], "approximate_reset")

    def test_empty_or_absent_inline_records_keep_legacy_plans_control_free(self):
        empty = self.bundle_for(inline_document("empty"), "empty-book")
        plain = self.bundle_for(inline_document("plain"), "plain-book")
        settings = self.settings()
        before = make_plan(plain, settings)
        after = make_plan(empty, settings)
        self.assertEqual(
            [[chunk["chunk_id"], chunk["spoken_text"], chunk["source_ranges"]]
             for chunk in before["chapters"][0]["chunks"]],
            [[chunk["chunk_id"], chunk["spoken_text"], chunk["source_ranges"]]
             for chunk in after["chapters"][0]["chunks"]])
        self.assertTrue(all("expressive_performance" not in chunk
                            for chunk in after["chapters"][0]["chunks"]))
        row = empty["chapters"][0]["rows"][2]
        spoken = build_spoken(row)[0]
        chunk = {"segment": row, "spoken_text": spoken,
                 "spoken_start": 0, "spoken_end": len(spoken)}
        bind_chunk_performance(chunk, row_performance(row, None, normalization=False), sparse=True)
        self.assertNotIn("expressive_performance", chunk)
        self.assertEqual(bind_chunk_performance(chunk, None, sparse=True), None)


    def test_expressive_mode_merges_inline_and_plan_duplicates_and_rejects_conflicts(self):
        def prepared_plan(name, mode="inline"):
            book = self.bundle_for(inline_document(mode), name)
            sidecar = self.root / f"{name}-plan.json"
            prepare_performance(book, deepcopy(DEFAULTS), sidecar,
                                client=FakeClient(), notify=lambda _: None)
            return book, sidecar

        book, sidecar = prepared_plan("expressive-book")
        merged = make_plan(book, self.settings(performance_mode="expressive",
                                               performance_plan=str(sidecar)))
        inline_only = make_plan(book, self.settings())
        dialogue_chunks = [chunk for chunk in merged["chapters"][0]["chunks"]
                           if chunk["segment"]["id"] == "s2"]
        self.assertEqual(len(dialogue_chunks), 1)
        record = dialogue_chunks[0]["expressive_performance"]
        self.assertEqual(len([control for control in record["controls"]
                              if control["kind"] == "delivery"]), 1)
        self.assertEqual(record, inline_only["chapters"][0]["chunks"][1]["expressive_performance"])
        conflict, conflict_sidecar = prepared_plan("conflict-book", mode="conflict")
        settings = self.settings(performance_mode="expressive",
                                 performance_plan=str(conflict_sidecar))
        with self.assertRaisesRegex(ValueError, "s2 has conflicting delivery controls"):
            make_plan(conflict, settings)


if __name__ == "__main__":
    unittest.main()
