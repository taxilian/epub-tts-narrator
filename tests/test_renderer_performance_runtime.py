"""Performance controls follow exact source chunks without altering legacy plans."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.performance_planning import prepare_performance
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.common import write_json
from epub_tts_narrator.renderer.chapter_protocol import _leaf, _chunk_shape, validate_render_settings
from epub_tts_narrator.renderer.performance_runtime import (
    bind_chunk_performance, protected_spans, require_engine_support,
    row_performance, verification_options,
)
from epub_tts_narrator.renderer.workflow import make_render_plan
from epub_tts_narrator.spoken_text import build_spoken, spoken_source_ranges
from shared_fixtures import renderer_config
from test_contracts import valid_document
from test_performance_planning import FakeClient


def record(text, controls=()):
    return {"version": 1, "text_sha256": text_sha256(text), "controls": list(controls)}


class PerformanceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        export_bundle(valid_document(), self.root / "book", allow_review=True)
        self.bundle = load_bundle(self.root / "book")
        self.settings = renderer_config(self.root, series_id="fixture")
        self.sidecar = self.root / "plan.json"

    def prepare(self):
        return prepare_performance(self.bundle, deepcopy(DEFAULTS), self.sidecar,
                                   client=FakeClient(), notify=lambda _: None)

    def test_legacy_mode_ignores_optional_sidecar_and_keeps_chunk_identity(self):
        before = make_plan(self.bundle, self.settings)
        self.settings["render"]["performance_plan"] = str(self.sidecar)
        self.sidecar.write_text("not JSON")
        self.assertEqual(make_plan(self.bundle, self.settings), before)

    def test_expressive_mode_binds_every_chunk_without_changing_words_or_source_ranges(self):
        self.prepare()
        old = make_plan(self.bundle, self.settings)
        self.settings["render"].update(performance_mode="expressive", performance_plan=str(self.sidecar))
        new = make_render_plan(self.bundle, self.settings, multi_speaker=False)
        for before, after in zip(old["chapters"][0]["chunks"], new["chapters"][0]["chunks"]):
            for key in ("chunk_id", "segment", "spoken_text", "source_ranges"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(after["expressive_traits"], after.get("speech_traits", {}))
            self.assertEqual(after["expressive_performance"]["text_sha256"], text_sha256(after["spoken_text"]))
        self.assertTrue(any(c["expressive_performance"]["controls"] for c in new["chapters"][0]["chunks"]))

    def test_missing_incomplete_or_stale_plan_fails_before_rendering(self):
        self.settings["render"]["performance_mode"] = "expressive"
        with self.assertRaisesRegex(ValueError, "requires"):
            make_plan(self.bundle, self.settings)
        plan = self.prepare()
        self.settings["render"]["performance_plan"] = str(self.sidecar)
        plan["segments"].pop("s2")
        write_json(self.sidecar, plan)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            make_plan(self.bundle, self.settings)

    def test_normalization_hint_applies_only_to_its_replacement(self):
        row = {"id": "s", "start": 30, "end": 47, "text": "He sssaid hello.",
               "delivery": {"direction": "quietly", "evidence": []}}
        row["end"] = row["start"] + len(row["text"])
        row["speech_normalization"] = {"version": 1, "edits": [{
            "start": 3, "end": 9, "original": "sssaid", "replacement": "said",
            "direction": "lengthen the initial s", "reason": "Preserve the hiss."}]}
        spoken = build_spoken(row)[0]
        canonical = row_performance(row, {"s": record(spoken)})
        self.assertEqual(canonical["controls"][0]["text"], "said")
        chunk = {"segment": row, "spoken_text": spoken, "spoken_start": 0, "spoken_end": len(spoken)}
        bind_chunk_performance(chunk, canonical)
        self.assertEqual(chunk["expressive_direction"], "quietly")
        self.assertEqual(spoken_source_ranges(row, 3, 7), [{"start": 33, "end": 39}])

    def test_pronunciation_remains_atomic_across_model_chunk_boundaries(self):
        text = "Hello Jasper again."
        plan = record(text, [{"kind": "pronunciation", "start": 6, "end": 12, "text": "Jasper",
                              "alphabet": "cmu_arpabet", "phonemes": "JH AE1 S P ER0"}])
        spans = protected_spans(text, [(0, 9), (9, len(text))], plan, 20)
        self.assertEqual("".join(text[a:b] for a, b in spans), text)
        self.assertFalse(any(6 < b < 12 for a, b in spans))

    def test_empty_silent_segment_can_keep_a_bound_empty_record(self):
        row = {"id": "s", "start": 0, "end": 0, "text": ""}
        chunk = {"segment": row, "spoken_text": "", "spoken_start": 0, "spoken_end": 0}
        bind_chunk_performance(chunk, record(""))
        self.assertEqual(chunk["expressive_performance"], record(""))

    def test_paragraph_verification_rebases_pause_offsets(self):
        first, second = "Hello. ", "Wait here."
        controls = [{"kind": "pause", "offset": 5, "duration_ms": 700}]
        turns = [{"spoken_text": first, "expressive_performance": record(first)},
                 {"spoken_text": second, "expressive_performance": record(second, controls)}]
        options = verification_options({"chunk": {"paragraph_turns": turns}})
        self.assertEqual(options["performance"]["text_sha256"], text_sha256(first + second))
        self.assertEqual(options["performance"]["controls"][0]["offset"], len(first) + 5)
        self.assertEqual(verification_options({"chunk": {"spoken_text": first}}), {})

    def test_unsupported_engines_fail_explicitly_without_dropping_controls(self):
        engine = type("LegacyEngine", (), {"capabilities": {}})()
        plan = {"chapters": [{"chunks": [{"expressive_performance": record("Hello")}]}]}
        with self.assertRaisesRegex(ValueError, "does not support"):
            require_engine_support(plan, engine)
        require_engine_support({"chapters": [{"chunks": [{}]}]}, engine)

    def test_remote_boundary_validates_the_same_bound_plan_and_traits(self):
        self.prepare()
        self.settings["render"].update(performance_mode="expressive", performance_plan=str(self.sidecar))
        plan = make_render_plan(self.bundle, self.settings, multi_speaker=False)
        bindings = {key: "a" * 64 for key in plan["voice_ids"]}
        for chunk in plan["chapters"][0]["chunks"]:
            _chunk_shape(chunk, set())
            _leaf(chunk, bindings)
        chunk = deepcopy(plan["chapters"][0]["chunks"][0])
        chunk["expressive_traits"] = {chunk["segment"]["voice_id"]: "unbound accent"}
        with self.assertRaisesRegex(ValueError, "traits differ"):
            _leaf(chunk, bindings)
        chunk = deepcopy(plan["chapters"][0]["chunks"][0])
        chunk["expressive_performance"]["text_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            _leaf(chunk, bindings)
        with self.assertRaisesRegex(ValueError, "performance_mode"):
            validate_render_settings({"performance_mode": "unknown"})
