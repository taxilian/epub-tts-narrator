"""Performance proposals preserve text and survive interrupted model work."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.performance_planning import (
    _BoundedRequests, _payload, _validate_proposal, _validate_review,
    load_prepared_performance, prepare_performance,
)
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.renderer.bundle import load_bundle
from test_contracts import valid_document


class FakeClient:
    def __init__(self):
        self.identity = {"model": "offline-test", "base_url": "https://example.invalid/v1"}
        self.calls = []
        self.fail_mode = None
        self.remove_controls = False

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if payload["mode"] == self.fail_mode:
            raise ProviderError("Offline interrupted request")
        if payload["mode"] == "performance_review":
            return {"segments": [{"segment_id": row["segment_id"],
                                   "keep": [] if self.remove_controls else list(range(len(payload["proposals"][row["segment_id"]]["controls"]))),
                                   "reason": "Checked scope against the source cue."}
                                  for row in payload["target_segments"]]}
        result = []
        for row in payload["target_segments"]:
            controls = []
            if row["segment_id"] == "s2":
                controls = [{"kind": "delivery", "quote": "Hello", "occurrence": 1,
                             "tag": "soft", "direction": "",
                             "evidence": [{"block_id": "b1", "quote": "said softly"}],
                             "reason": "The speaker says this utterance softly."}]
            result.append({"segment_id": row["segment_id"], "controls": controls})
        return {"segments": result}


class PerformancePlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.book = self.root / "book"
        export_bundle(valid_document(), self.book, allow_review=True)
        self.bundle = load_bundle(self.book)
        self.config = deepcopy(DEFAULTS)
        self.config["analysis"].update(attribution_workers=1, repair_attempts=0)
        self.output = self.root / "comparison" / "performance.json"
        self.client = FakeClient()

    def prepare(self, **kwargs):
        return prepare_performance(self.bundle, self.config, self.output,
                                   client=self.client, notify=lambda _: None, **kwargs)

    def payload(self):
        return _payload(self.bundle, self.bundle["chapters"][0]["rows"],
                        {block["id"]: index for index, block in enumerate(self.bundle["source"]["blocks"])})

    def test_spans_are_exact_source_grounded_and_book_files_remain_unchanged(self):
        before = {str(path.relative_to(self.book)): path.read_bytes() for path in self.book.rglob("*") if path.is_file()}
        bundle_before = deepcopy(self.bundle)
        result = self.prepare()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["segments"]), {"s1", "s2", "s3"})
        self.assertEqual(result["segments"]["s1"]["controls"], [])
        control = result["segments"]["s2"]["controls"][0]
        self.assertEqual((control["start"], control["end"], control["text"], control["tag"]), (1, 6, "Hello", "soft"))
        self.assertEqual(control["evidence"], [{"block_id": "b1", "quote": "said softly"}])
        self.assertEqual([call["mode"] for call in self.client.calls], ["performance_planning", "performance_review"])
        self.assertEqual(bundle_before, self.bundle)
        self.assertEqual(before, {str(path.relative_to(self.book)): path.read_bytes() for path in self.book.rglob("*") if path.is_file()})
        self.assertEqual(load_prepared_performance(self.bundle, self.output), result)

    def test_completed_plan_reuses_every_decision_without_a_model_call(self):
        first = self.prepare()
        self.client.calls.clear()
        self.client.fail_mode = "performance_planning"
        self.assertEqual(self.prepare(), first)
        self.assertEqual(self.client.calls, [])

    def test_independent_review_can_remove_but_cannot_add_or_rewrite_controls(self):
        self.client.remove_controls = True
        result = self.prepare()
        self.assertEqual(result["segments"]["s2"]["controls"], [])
        payload = self.payload()
        proposal = _validate_proposal(FakeClient().complete("", payload), payload)
        review = self.client.complete("", {**payload, "mode": "performance_review", "proposals": proposal})
        review["segments"][1]["keep"] = [1]
        with self.assertRaisesRegex(ValueError, "existing control indexes"):
            _validate_review(review, proposal, payload["target_segments"])

    def test_review_failure_leaves_pending_and_retry_reuses_validated_proposal(self):
        self.client.fail_mode = "performance_review"
        with self.assertRaises(ProviderError):
            self.prepare()
        state = json.loads(self.output.read_text())
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["segments"], {})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_prepared_performance(self.bundle, self.output)
        self.client.fail_mode = None
        self.client.calls.clear()
        self.prepare()
        self.assertEqual([call["mode"] for call in self.client.calls], ["performance_review"])

    def test_failed_later_batch_keeps_completed_rows_and_resumes_only_missing_work(self):
        self.client.fail_mode = "performance_review"
        with patch("epub_tts_narrator.performance_planning.BATCH_ROWS", 1):
            with self.assertRaises(ProviderError):
                self.prepare()
            state = json.loads(self.output.read_text())
            self.assertEqual(set(state["segments"]), {"s1"})
            self.client.fail_mode = None
            self.client.calls.clear()
            result = self.prepare()
        self.assertEqual(result["status"], "complete")
        self.assertEqual([call["target_segments"][0]["segment_id"] for call in self.client.calls], ["s2", "s3"])
        self.assertEqual([call["mode"] for call in self.client.calls], ["performance_review", "performance_planning"])

    def test_changed_model_invalidates_completed_plan_before_call_failure(self):
        self.prepare()
        self.client.identity["model"] = "different-model"
        self.client.fail_mode = "performance_planning"
        with self.assertRaises(ProviderError):
            self.prepare()
        state = json.loads(self.output.read_text())
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["segments"], {})
        self.assertEqual(state["provenance"]["provider"]["model"], "different-model")

    def test_changed_scripts_and_tampered_text_bindings_do_not_reuse_a_completed_plan(self):
        self.prepare()
        cast_modified = deepcopy(self.bundle)
        cast_modified["cast"]["voices"][0]["description"] = "A changed habitual voice direction."
        with self.assertRaisesRegex(ValueError, "current book"):
            load_prepared_performance(cast_modified, self.output)
        modified = deepcopy(self.bundle)
        modified["chapters"][0]["rows"][0]["voice_id"] = "another-voice"
        with self.assertRaisesRegex(ValueError, "scripts"):
            load_prepared_performance(modified, self.output)
        state = json.loads(self.output.read_text())
        state["segments"]["s2"]["text_sha256"] = "0" * 64
        self.output.write_text(json.dumps(state))
        self.client.calls.clear()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.client.calls, [])

    def test_schema_and_grounding_errors_are_rejected_without_echoing_source(self):
        payload = self.payload()
        original = self.client.complete("", payload)
        mutations = [
            lambda value: value["segments"].pop(),
            lambda value: value["segments"][1]["controls"][0].update(quote="DO NOT ECHO"),
            lambda value: value["segments"][1]["controls"][0].update(occurrence=2),
            lambda value: value["segments"][1]["controls"][0]["evidence"][0].update(quote="DO NOT ECHO"),
            lambda value: value["segments"][1]["controls"][0].update(kind="vocalization"),
            lambda value: value["segments"][1]["controls"][0].update(tag="style", direction="[DO NOT ECHO]"),
            lambda value: value["segments"][1]["controls"].append(deepcopy(value["segments"][1]["controls"][0])),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = deepcopy(original)
                mutation(value)
                with self.assertRaises(ValueError) as failure:
                    _validate_proposal(value, payload)
                self.assertNotIn("DO NOT ECHO", str(failure.exception))

    def test_saved_evidence_is_revalidated_against_current_source(self):
        state = self.prepare()
        state["segments"]["s2"]["controls"][0]["evidence"][0]["quote"] = "An invented supporting cue."
        self.output.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "exact visible source"):
            load_prepared_performance(self.bundle, self.output)

    def test_repeat_occurrences_and_internal_pauses_resolve_without_rewriting_text(self):
        payload = {"target_segments": [{"segment_id": "s", "spoken_text": "Go. Go, then wait."}],
                   "source_blocks": [{"block_id": "b", "text": "Go. Go, then wait. She paused before the last instruction."}]}
        proposed = {"segments": [{"segment_id": "s", "controls": [
            {"kind": "delivery", "quote": "Go", "occurrence": 2, "tag": "emphasis", "direction": "",
             "evidence": [{"block_id": "b", "quote": "Go. Go"}], "reason": "The instruction repeats."},
            {"kind": "pause", "after": "Go,", "occurrence": 1, "duration_ms": 500,
             "evidence": [{"block_id": "b", "quote": "She paused"}], "reason": "Explicit pause before continuing."},
        ]}]}
        controls = _validate_proposal(proposed, payload)["s"]["controls"]
        self.assertEqual((controls[0]["start"], controls[0]["end"]), (4, 6))
        self.assertEqual(controls[1]["offset"], 7)
        proposed["segments"][0]["controls"][1]["after"] = "wait."
        with self.assertRaisesRegex(ValueError, "internal offset"):
            _validate_proposal(proposed, payload)

    def test_source_instructions_remain_payload_data_and_secrets_do_not_enter_state(self):
        self.client.identity["api_key"] = "DO NOT ECHO"
        self.config["provider"]["api_key"] = "DO NOT ECHO"
        result = self.prepare()
        self.assertNotIn("DO NOT ECHO", json.dumps(result))
        for call in self.client.calls:
            self.assertNotIn("api_key", call)
        from epub_tts_narrator.performance_planning_prompts import SYSTEM, REVIEW_SYSTEM
        self.assertIn("untrusted data", SYSTEM)
        self.assertIn("untrusted data", REVIEW_SYSTEM)

    def test_symlink_outputs_are_refused_before_model_calls(self):
        self.output.parent.mkdir()
        other = self.root / "other.json"
        other.write_text("protected")
        self.output.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.prepare()
        self.assertEqual(other.read_text(), "protected")
        self.assertEqual(self.client.calls, [])

    def test_cache_replay_is_bounded_and_revalidated(self):
        path = self.root / "cache.json"
        path.write_text("[" + " " * 100)
        requests = _BoundedRequests(self.client, self.root)
        with patch("epub_tts_narrator.performance_planning.MAX_RESPONSE_BYTES", 32):
            self.assertEqual(requests._cached_response(path, lambda value: value), (False, None))

    def test_configured_concurrency_is_bounded_at_five(self):
        # More rows let all slots run; no renderer or real transport is involved.
        rows = self.bundle["chapters"][0]["rows"]
        self.bundle["chapters"][0]["rows"] = [dict(rows[0], id=f"row-{number}") for number in range(12)]
        self.config["analysis"]["attribution_workers"] = 8
        client = self.client
        original = client.complete
        mutex = threading.Lock()
        active, peak = 0, 0
        def delayed(system, payload):
            nonlocal active, peak
            with mutex:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.02)
                return original(system, payload)
            finally:
                with mutex:
                    active -= 1
        client.complete = delayed
        with patch("epub_tts_narrator.performance_planning.BATCH_ROWS", 1):
            self.prepare()
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 5)


if __name__ == "__main__":
    unittest.main()
