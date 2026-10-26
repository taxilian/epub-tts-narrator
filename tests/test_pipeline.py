"""Exercise the full analysis/correction flow without making API requests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Event, Lock
import unittest

from epub_tts_narrator.contracts import readiness, validate_document
from epub_tts_narrator.pipeline import (
    CachedRequests, DEFAULT_POLICY, align_units, analyze, apply_edits, fingerprint, make_units, merge_patch, seed_cast,
)
from epub_tts_narrator.provider import ProviderError
from test_contracts import valid_document
from audition_fixtures import script_response


class FakeClient:
    """A deterministic fixture model using only IDs supplied in each request."""

    identity = {"base_url": "http://offline.invalid/v1", "model": "fixture-model"}

    def __init__(self, *, unresolved=False, invalid_text=False, duplicate=False, mutation=None):
        self.unresolved = unresolved
        self.invalid_text = invalid_text
        self.duplicate = duplicate
        self.mutation = mutation
        self.calls = []

    def complete(self, system, payload):
        self.calls.append((system, deepcopy(payload)))
        if payload.get("mode") == "speech_spelling_scan":
            return {"segments": [{"segment_id": row["segment_id"], "edits": []}
                                  for row in payload["target_segments"]]}
        if payload.get("mode") == "audition_script":
            return script_response()
        if payload.get("mode") == "voice_ensemble":
            return {"voices": [], "reasons": {}}
        if payload.get("mode") == "discover":
            return self.discover(payload)
        if payload.get("mode") == "reconcile":
            return self.reconcile(payload)
        if payload.get("mode") == "speaker_completion":
            # Keep this fixture's deliberate uncertainty through the mandatory
            # jury, rather than accidentally treating a new response as ordinary
            # unit attribution. Dedicated integration tests exercise resolution.
            fields = ("text", "kind", "role_id", "basis", "reason", "evidence",
                      "candidates", "review_required", "delivery")
            unit = payload["target_units"][0]
            evidence = [{"block_id": unit["block_id"], "quote": unit["text"]}]
            pieces = [{key: deepcopy(item[key]) for key in fields}
                      for item in payload["original_segments"]]
            for item in pieces:
                if item["basis"] != "unresolved" and not item["evidence"]:
                    item["evidence"] = deepcopy(evidence)
            return {"block_id": unit["block_id"], "reason": "Fixture preserves uncertain speaker.",
                    "evidence": evidence, "segments": pieces}
        response = self.attribute(payload)
        if self.mutation:
            self.mutation(response["units"][0]["segments"][0])
        if self.invalid_text:
            response["units"][0]["segments"][0]["text"] = "Rewritten source text."
        return response

    @staticmethod
    def patch(**values):
        return dict({"characters": [], "roles": [], "voices": [], "merges": [],
                     "issues": [], "summary": "Mara speaks softly, then the group leaves."}, **values)

    def discover(self, payload):
        if payload["known_cast"]["characters"]:
            return self.patch()
        available = payload["available_ids"]
        source_unit = next(unit for unit in payload["target_units"] if "Mara" in unit["text"])
        evidence = [{"block_id": source_unit["block_id"], "quote": "Mara"}]
        characters, roles, voices = [], [], []
        count = 2 if self.duplicate else 1
        for index in range(count):
            character_id = available["characters"][index]
            voice_id = available["voices"][index]
            name = "Mara" if index == 0 else "the speaker"
            characters.append({"id": character_id, "name": name, "aliases": [],
                               "description": "A character who speaks softly.",
                               "evidence": deepcopy(evidence), "voice_id": voice_id})
            roles.append({"id": available["roles"][index], "kind": "dialogue",
                          "character_id": character_id, "voice_id": voice_id,
                          "description": f"Dialogue spoken by {name}."})
            voices.append({"id": voice_id, "description": f"Proposed distinct voice for {name}.",
                           "traits": [{"name": "timbre", "value": "light and clear",
                                       "basis": "casting", "evidence": []}]})
        return self.patch(characters=characters, roles=roles, voices=voices)

    def reconcile(self, payload):
        characters = payload["known_cast"]["characters"]
        mara = deepcopy(next(character for character in characters if character["name"] == "Mara"))
        mara["description"] = "Mara, reconciled across the complete book."
        mara["aliases"] = ["Mara"]
        merges = []
        if self.duplicate:
            duplicate = next(character for character in characters if character["name"] != "Mara")
            merges = [{"from_character_id": duplicate["id"], "into_character_id": mara["id"],
                       "evidence": deepcopy(mara["evidence"])}]
        return self.patch(characters=[mara], merges=merges)

    def attribute(self, payload):
        cast = payload["final_cast"]
        mara = next(character for character in cast["characters"] if character["name"] == "Mara")
        dialogue = next(role for role in cast["roles"]
                        if role["kind"] == "dialogue" and role["character_id"] == mara["id"])
        units = []
        for unit in payload["target_units"]:
            text = unit["text"]
            start, finish = text.find("“"), text.find("”")
            spans = [(text, "narration")]
            if start >= 0 and finish > start:
                spans = [(text[:start], "narration"), (text[start:finish + 1], "dialogue"),
                         (text[finish + 1:], "narration")]
            pieces = []
            for exact_text, kind in spans:
                if not exact_text:
                    continue
                unresolved = self.unresolved and kind == "dialogue"
                pieces.append({
                    "text": exact_text, "kind": kind,
                    "role_id": None if unresolved else dialogue["id"] if kind == "dialogue" else "narrator_main",
                    "basis": "unresolved" if unresolved else "explicit",
                    "reason": "Speaker needs review." if unresolved else "The narration names Mara as speaker.",
                    "evidence": deepcopy(mara["evidence"]) if kind == "dialogue" else [],
                    "candidates": [dialogue["id"]] if unresolved else [],
                    "review_required": unresolved,
                    "delivery": {"direction": "softly" if kind == "dialogue" else "",
                                 "evidence": [{"block_id": unit["block_id"], "quote": "softly"}] if kind == "dialogue" else []},
                })
            units.append({"unit_id": unit["unit_id"], "segments": pieces})
        return {"units": units}


class ConcurrentFakeClient(FakeClient):
    """Force the second batch to reply first without relying on sleep timing."""

    def __init__(self, *, fail_second=False):
        super().__init__()
        self.fail_second = fail_second
        self.first_started = Event()
        self.second_finished = Event()
        self.lock = Lock()
        self.active = self.max_active = 0
        self.started = []
        self.finished = []

    def attribute(self, payload):
        unit_id = payload["target_units"][0]["unit_id"]
        with self.lock:
            self.started.append(unit_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if unit_id == "u0000001":
                self.first_started.set()
                if not self.second_finished.wait(5):
                    raise AssertionError("Second attribution batch never ran concurrently")
            if unit_id == "u0000002":
                if not self.first_started.wait(5):
                    raise AssertionError("First attribution batch never started")
                if self.fail_second:
                    raise RuntimeError("Fixture transport failure")
            result = super().attribute(payload)
            with self.lock:
                self.finished.append(unit_id)
            return result
        finally:
            with self.lock:
                self.active -= 1
            if unit_id == "u0000002":
                self.second_finished.set()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"
        self.source = deepcopy(valid_document()["source"])

    def run_analysis(self, client=None, **kwargs):
        # These tests isolate the original discovery/attribution mechanics.
        # Consensus integration has dedicated cases below and in test_identity.
        kwargs.setdefault("identity_consensus", False)
        return analyze(self.source, client or FakeClient(), self.cache, **kwargs)

    def test_full_analysis_discovers_upserts_reconciles_and_attributes(self):
        client = FakeClient()
        progress = []
        original = deepcopy(self.source)
        document = self.run_analysis(client, progress=progress.append)
        self.assertEqual([payload.get("mode", "attribute") for _, payload in client.calls],
                         ["discover", "reconcile", "voice_ensemble", "attribute"])
        self.assertEqual(validate_document(document), [])
        self.assertTrue(document["readiness"]["ready"])
        self.assertEqual(len(document["cast"]["characters"]), 1)
        character = document["cast"]["characters"][0]
        allocated_id = client.calls[0][1]["available_ids"]["characters"][0]
        self.assertEqual(character["id"], allocated_id)
        self.assertEqual(character["description"], "Mara, reconciled across the complete book.")
        self.assertEqual(character["aliases"], ["Mara"])
        self.assertEqual(document["provenance"]["source_fingerprint"], fingerprint(self.source))
        self.assertEqual(self.source, original)
        self.assertTrue(any("Reconciling" in message for message in progress))
        self.assertEqual(len(client.calls[1][1]["scene_summaries"]), 1)
        final_cast_sent = client.calls[3][1]["final_cast"]
        self.assertEqual(final_cast_sent, document["cast"])

    def test_mixed_narration_and_dialogue_preserve_every_character(self):
        document = self.run_analysis()
        self.assertEqual([item["kind"] for item in document["segments"]],
                         ["narration", "dialogue", "narration"])
        self.assertEqual(document["segments"][0]["role_id"], "narrator_main")
        self.assertNotEqual(document["segments"][1]["voice_id"], document["segments"][0]["voice_id"])
        self.assertEqual(document["segments"][1]["text"], "“Hello.”")
        for block in self.source["blocks"]:
            spans = [item for item in document["segments"] if item["block_id"] == block["id"]]
            self.assertEqual("".join(item["text"] for item in spans), block["text"])
            self.assertEqual(spans[0]["start"], 0)
            self.assertEqual(spans[-1]["end"], len(block["text"]))
            for left, right in zip(spans, spans[1:]):
                self.assertEqual(left["end"], right["start"])

    def test_rerun_uses_only_validated_cache_and_makes_zero_calls(self):
        first_client = FakeClient()
        original = self.run_analysis(first_client)
        self.assertEqual(len(first_client.calls), 4)
        second_client = FakeClient()
        repeated = self.run_analysis(second_client)
        self.assertEqual(second_client.calls, [])
        self.assertEqual(repeated, original)
        self.assertEqual(len(list(self.cache.glob("*.json"))), 4)

    def test_invalid_cached_attribution_is_recomputed(self):
        original = self.run_analysis()
        for path in self.cache.glob("*.json"):
            cached = json.loads(path.read_text())
            if "units" in cached:
                cached["units"][0]["segments"][0]["text"] = "Fabricated text"
                path.write_text(json.dumps(cached))
                break
        else:
            self.fail("Expected a cached attribution response")
        client = FakeClient()
        regenerated = self.run_analysis(client)
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("mode", client.calls[0][1])
        self.assertEqual(regenerated, original)

    def test_changed_model_identity_does_not_reuse_other_provider_cache(self):
        self.run_analysis()
        client = FakeClient()
        client.identity = dict(client.identity, model="another-fixture-model")
        self.run_analysis(client)
        self.assertEqual(len(client.calls), 4)

    def test_changed_model_text_is_retried_once_then_rejected_and_not_cached(self):
        client = FakeClient(invalid_text=True)
        with self.assertRaisesRegex(ValueError, "failed validation after 2 attempt.*Text changed"):
            self.run_analysis(client)
        attribute_calls = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertEqual(len(attribute_calls), 2)
        self.assertNotIn("correction", attribute_calls[0])
        self.assertIn("Text changed", attribute_calls[1]["correction"]["validator_error"])
        self.assertEqual(attribute_calls[1]["correction"]["previous_response"]["units"][0]["segments"][0]["text"],
                         "Rewritten source text.")
        self.assertEqual(attribute_calls[0]["target_units"], attribute_calls[1]["target_units"])
        cached = [json.loads(path.read_text()) for path in self.cache.glob("*.json")]
        self.assertEqual(len(cached), 3)
        self.assertTrue(all("units" not in response for response in cached))
        failures = sorted((self.cache / "failed").glob("*.json"))
        self.assertEqual(len(failures), 2)
        self.assertTrue(failures[0].name.endswith("-attempt1.json"))
        self.assertTrue(failures[1].name.endswith("-attempt2.json"))
        self.assertTrue(all(json.loads(path.read_text())["units"][0]["segments"][0]["text"] == "Rewritten source text."
                            for path in failures))
        repaired_client = FakeClient()
        document = self.run_analysis(repaired_client)
        self.assertEqual(len(repaired_client.calls), 1)
        self.assertTrue(readiness(document)["ready"])

    def test_validation_repairs_use_previous_response_when_it_fits_and_report_progress(self):
        class RepairClient:
            identity = {"model": "offline-repair-fixture"}

            def __init__(self):
                self.calls = []

            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                return {"answer": "x" * 500} if len(self.calls) == 1 else {"answer": "valid"}

        def validate(response):
            if response.get("answer") != "valid":
                raise ValueError("Expected valid answer")
            return response

        for budget, expect_previous in ((2000, True), (300, False)):
            with self.subTest(budget=budget):
                client = RepairClient()
                progress = []
                cache = self.cache / str(budget)
                requests = CachedRequests(client, cache, max_request_chars=budget, progress=progress.append)
                result = requests.run("Fixture system", {"task": "Return a valid answer"}, validate)
                self.assertEqual(result, {"answer": "valid"})
                self.assertEqual(len(client.calls), 2)
                correction = client.calls[1]["correction"]
                self.assertEqual("previous_response" in correction, expect_previous)
                if expect_previous:
                    self.assertEqual(correction["previous_response"], {"answer": "x" * 500})
                self.assertEqual(correction["validator_error"], "Expected valid answer")
                self.assertEqual(client.calls[1]["task"], client.calls[0]["task"])
                self.assertLessEqual(len("Fixture system") + len(json.dumps(client.calls[1], ensure_ascii=False)), budget)
                self.assertEqual(progress, ["Repairing validation 1/1: Expected valid answer"])
                failed_paths = list((cache / "failed").glob("*.json"))
                self.assertEqual(len(failed_paths), 1)
                self.assertEqual(json.loads(failed_paths[0].read_text()), {"answer": "x" * 500})
                self.assertEqual(requests.run("Fixture system", {"task": "Return a valid answer"}, validate), result)
                self.assertEqual(len(client.calls), 2)

    def test_validation_repair_progress_preserves_diagnostics_longer_than_240_characters(self):
        class RepairClient(FakeClient):
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                return {"answer": len(self.calls)}

        diagnostic = ("Speech spelling scan batch 166/245: segments[19].edits[5]: "
                      + "fixture spelling mismatch; " * 20
                      + "Copy original exactly and use offsets relative to the segment text.")

        def validate(response):
            if response["answer"] == 1:
                raise ValueError(diagnostic)
            return response

        client, progress = RepairClient(), []
        requests = CachedRequests(client, self.cache, progress=progress.append)
        self.assertEqual(requests.run("Fixture system", {"task": "repair"}, validate), {"answer": 2})
        self.assertEqual(progress, ["Repairing validation 1/1: " + diagnostic])
        self.assertEqual(client.calls[1]["correction"]["validator_error"], diagnostic)

    def test_spelling_repair_guidance_for_control_syntax_in_direction(self):
        class RepairClient(FakeClient):
            def complete(self, system, payload):
                self.calls.append(deepcopy(payload))
                if len(self.calls) == 1:
                    return {"segments": [{"segment_id": "s1", "edits": [
                        {"start": 0, "end": 2, "original": "No", "replacement": "No",
                         "direction": "Shout, then <Slam> lands.", "reason": "beat"}]}]}
                return {"segments": [{"segment_id": "s1", "edits": []}]}

        payload = {"mode": "speech_spelling_scan", "target_segments": [{"segment_id": "s1"}]}

        def validate(response):
            if response["segments"][0]["edits"]:
                raise ValueError("Speech delivery direction must be nonempty plain text without control syntax")
            return response

        client = RepairClient()
        requests = CachedRequests(client, self.cache)
        result = requests.run("Fixture system", payload, validate)
        self.assertEqual(result["segments"][0]["edits"], [])
        guidance = client.calls[1]["correction"]["guidance"]
        self.assertTrue(any("plain words" in item and "<Slam>" in item for item in guidance))
        spelling_guidance = [item for item in guidance if "plain words" in item]
        self.assertEqual(len(spelling_guidance), 1)

    def test_analysis_forwards_validation_repair_progress(self):
        progress = []
        with self.assertRaisesRegex(ValueError, "Text changed"):
            self.run_analysis(FakeClient(invalid_text=True), progress=progress.append)
        repair_messages = [message for message in progress if message.startswith("Repairing validation")]
        self.assertEqual(len(repair_messages), 1)
        self.assertIn("Text changed", repair_messages[0])

    def test_missing_unit_feedback_identifies_counts_and_exact_id_for_repair(self):
        class MissingUnitClient(FakeClient):
            def attribute(self, payload):
                response = super().attribute(payload)
                if "correction" not in payload:
                    response["units"].pop(1)
                return response
        client = MissingUnitClient()
        result = self.run_analysis(client)
        self.assertEqual(validate_document(result), [])
        calls = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertEqual(len(calls), 2)
        feedback = calls[1]["correction"]["validator_error"]
        self.assertIn("Expected 2 unit(s), received 1", feedback)
        self.assertIn("Missing IDs (1): 'u0000002'", feedback)
        self.assertIn("Duplicate IDs (0): none", feedback)
        self.assertIn("units[1].unit_id", feedback)
        self.assertIn("<missing unit>", feedback)

    def test_unit_coverage_reports_duplicates_unexpected_ids_and_reordering_boundedly(self):
        units = make_units(self.source)[0]
        for ids, expected in [
            ([units[0]["unit_id"]] * 2, ("Missing IDs (1)", "Duplicate IDs (1): 'u0000001' (2 occurrences)")),
            (["not-a-target", units[1]["unit_id"]], ("Missing IDs (1)", "Unexpected IDs (1): 'not-a-target'")),
            ([units[1]["unit_id"], units[0]["unit_id"]], ("Missing IDs (0)", "First mismatch at units[0].unit_id")),
        ]:
            with self.subTest(ids=ids), self.assertRaises(ValueError) as caught:
                align_units(self.source, seed_cast(), units, {"units": [{"unit_id": value} for value in ids]})
            for value in expected:
                self.assertIn(value, str(caught.exception))
        many = [dict(units[0], unit_id=f"u{index:07}") for index in range(500)]
        returned = [{"unit_id": "x\x00" * 10000 + str(index)} for index in range(500)]
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), many, {"units": returned})
        message = str(caught.exception)
        self.assertLessEqual(len(message), 1800)
        self.assertIn("Expected 500 unit(s), received 500", message)
        self.assertIn("Missing IDs (500)", message)
        self.assertIn("Unexpected IDs (500)", message)

    def test_empty_segment_feedback_identifies_unit_block_and_response_path(self):
        class EmptySegmentClient(FakeClient):
            def attribute(self, payload):
                response = super().attribute(payload)
                if "correction" not in payload:
                    response["units"][1]["segments"][0]["text"] = ""
                return response
        client = EmptySegmentClient()
        self.assertEqual(validate_document(self.run_analysis(client)), [])
        calls = [payload for _, payload in client.calls if "mode" not in payload]
        feedback = calls[1]["correction"]["validator_error"]
        self.assertIn("units[1].segments[0].text", feedback)
        self.assertIn("unit 'u0000002'", feedback)
        self.assertIn("source block 'b2'", feedback)
        self.assertIn("text is an empty string", feedback)
        unit = make_units(self.source)[0][1]
        for piece in (None, [], {}, {"text": 42}):
            with self.subTest(piece=piece), self.assertRaises(ValueError) as caught:
                align_units(self.source, seed_cast(), [unit],
                            {"units": [{"unit_id": unit["unit_id"], "segments": [piece]}]})
            self.assertIn("units[0].segments[0].text", str(caught.exception))
            self.assertIn("source block 'b2'", str(caught.exception))

    def test_unknown_role_feedback_identifies_path_and_cast_roles_for_repair(self):
        class UnknownRoleClient(FakeClient):
            def attribute(self, payload):
                response = super().attribute(payload)
                if "correction" not in payload:
                    response["units"][0]["segments"][1]["role_id"] = "character-instead-of-role"
                return response
        client = UnknownRoleClient()
        result = self.run_analysis(client)
        self.assertEqual(validate_document(result), [])
        calls = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertEqual(len(calls), 2)
        feedback = calls[1]["correction"]["validator_error"]
        self.assertIn("Segment references an unknown role", feedback)
        self.assertIn("units[0].segments[1].role_id", feedback)
        self.assertIn('unit "u0000001"', feedback)
        self.assertIn('source block "b1"', feedback)
        self.assertIn('"character-instead-of-role"', feedback)
        self.assertIn("cast.roles[].id", feedback)
        for role in result["cast"]["roles"]:
            self.assertIn(json.dumps(role["id"]), feedback)
        self.assertIn("not a character or voice ID", feedback)

    def test_unknown_role_feedback_bounds_ids_and_hides_malformed_values(self):
        unit = make_units(self.source)[0][1]
        cast = seed_cast()
        # The early role check need not format the rest of these synthetic
        # records; test both a large cast and escaped, oversized identifier data.
        cast["roles"].extend({"id": f"r{index}" + "\x00" * 10000} for index in range(20))
        piece = {
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narration.", "evidence": [],
            "candidates": [], "review_required": False,
            "delivery": {"direction": "", "evidence": []},
        }
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [piece]}]}
        for rejected, visible in [
            ("missing\x00" * 10000, '"missing\\u0000'),
            ({"private-value": "must-not-appear"}, "<dict>"),
            (["must-not-appear"], "<list>"),
            (42, "<int>"),
            (False, "<bool>"),
        ]:
            piece["role_id"] = rejected
            before = deepcopy(response)
            with self.subTest(value_type=type(rejected).__name__), self.assertRaises(ValueError) as caught:
                align_units(self.source, cast, [unit], response)
            feedback = str(caught.exception)
            self.assertIn("units[0].segments[0].role_id", feedback)
            self.assertIn('unit "u0000002"', feedback)
            self.assertIn('source block "b2"', feedback)
            self.assertIn(visible, feedback)
            self.assertIn("21 total", feedback)
            self.assertIn("15 additional role ID(s) not shown", feedback)
            self.assertNotIn("must-not-appear", feedback)
            self.assertNotIn("private-value", feedback)
            self.assertNotIn("\x00", feedback)
            self.assertLess(len(feedback), 1800)
            self.assertEqual(response, before)

    def test_attribution_provider_failure_includes_safe_batch_and_unit_range(self):
        for failure, expected in [
            (ProviderError("private-url private-key private-book"), "Provider request failed"),
            (ProviderError("The model output was truncated. Increase the output token limit or reduce the input."), "model output was truncated"),
            (ProviderError("private-server-details", status_code=429, code="tpm_uncached_exceeded", retryable=True), "HTTP 429"),
        ]:
            class FailingClient(FakeClient):
                def attribute(self, payload):
                    raise failure
            with self.subTest(expected=expected), self.assertRaises(ProviderError) as caught:
                self.run_analysis(FailingClient())
            message = str(caught.exception)
            self.assertIn("Attribution batch 1/1", message)
            self.assertIn("units u0000001..u0000002", message)
            self.assertIn("source blocks b1..b2", message)
            self.assertIn(expected, message)
            self.assertNotIn("private", message)
            self.assertIsNone(caught.exception.__cause__)
            self.assertEqual(caught.exception.retryable, failure.retryable)
            self.assertEqual(caught.exception.status_code, failure.status_code)
            self.assertEqual(caught.exception.code, failure.code)

    def test_request_budget_prevents_api_call(self):
        client = FakeClient()
        with self.assertRaisesRegex(ValueError, "max_request_chars"):
            self.run_analysis(client, max_request_chars=1)
        self.assertEqual(client.calls, [])

    def test_model_cannot_claim_user_correction_or_silently_default_malformed_fields(self):
        for mutation, message in [
            (lambda piece: piece.update(basis="user"), "models cannot claim a user correction"),
            (lambda piece: piece.update(evidence=None), "Evidence and candidates must be arrays"),
            (lambda piece: piece.update(candidates=None), "Evidence and candidates must be arrays"),
            (lambda piece: piece.update(delivery=None), "Delivery must have direction"),
            (lambda piece: piece.update(delivery={"direction": "Urgently", "evidence": []}), "exact source quotations"),
        ]:
            with self.subTest(message=message):
                client = FakeClient(mutation=mutation)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_analysis(client)
                self.assertEqual(len([payload for _, payload in client.calls if "mode" not in payload]), 2)

    def test_evidence_error_uses_response_path_and_both_source_block_ids(self):
        # The first source block becomes validator filler, so a flattened
        # segments[1] error would not identify this response's segments[0].
        unit = make_units(self.source)[0][1]
        piece = {
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narration.",
            "evidence": [{"block_id": "b1", "quote": "They left."}],
            "candidates": [], "review_required": False,
            "delivery": {"direction": "", "evidence": []},
        }
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [piece]}]}
        before = deepcopy(response)
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), [unit], response)
        error = str(caught.exception)
        self.assertIn("units[0].segments[0].evidence[0].quote", error)
        self.assertIn("unit 'u0000002'", error)
        self.assertIn("source block 'b2'", error)
        self.assertIn("referenced block 'b1'", error)
        self.assertIn("exact substring", error)
        self.assertIn("correct source block", error)
        self.assertEqual(response, before)
        self.assertEqual(self.source, valid_document()["source"])

        # Correcting the reference, without changing the quotation or source,
        # still returns the same ordinary source-preserving annotation shape.
        piece["evidence"][0]["block_id"] = "b2"
        accepted = align_units(self.source, seed_cast(), [unit], response)
        self.assertEqual(accepted[0]["text"], unit["text"])
        self.assertEqual(accepted[0]["attribution"]["evidence"], piece["evidence"])
        self.assertNotIn("location", accepted[0])

    def test_empty_delivery_evidence_error_names_field_and_safe_repair(self):
        unit = make_units(self.source)[0][1]
        piece = {
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narration.", "evidence": [],
            "candidates": [], "review_required": False,
            "delivery": {"direction": "urgently", "evidence": []},
        }
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [piece]}]}
        before = deepcopy(response)
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), [unit], response)
        error = str(caught.exception)
        self.assertIn("units[0].segments[0].delivery.evidence", error)
        self.assertIn("unit 'u0000002'", error)
        self.assertIn("source block 'b2'", error)
        self.assertIn("exact source quotations", error)
        self.assertIn("leave delivery.direction empty", error)
        self.assertEqual(response, before)

        # Explicit user delivery directions retain their existing exception.
        piece["basis"] = "user"
        accepted = align_units(self.source, seed_cast(), [unit], response, allow_user=True)
        self.assertEqual(accepted[0]["delivery"], piece["delivery"])

    def test_delivery_evidence_context_preserves_strict_spelling_validation(self):
        unit = make_units(self.source)[0][0]
        piece = {
            "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narration.", "evidence": [],
            "candidates": [], "review_required": False,
            "delivery": {"direction": "softly", "evidence": [{"block_id": "b1", "quote": "Softly"}]},
        }
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [piece]}]}
        before = deepcopy(response)
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), [unit], response)
        error = str(caught.exception)
        self.assertIn("units[0].segments[0].delivery.evidence[0].quote", error)
        self.assertIn("referenced block 'b1'", error)
        self.assertIn("exact substring", error)
        self.assertEqual(response, before)

        piece["delivery"]["evidence"][0]["quote"] = "softly"
        accepted = align_units(self.source, seed_cast(), [unit], response)
        self.assertEqual(accepted[0]["delivery"], piece["delivery"])
        self.assertEqual(accepted[0]["text"], self.source["blocks"][0]["text"])

    def test_all_empty_delivery_evidence_paths_are_reported_together(self):
        units = make_units(self.source)[0]
        response = {"units": []}
        for unit in units:
            piece = {
                "text": unit["text"], "kind": "narration", "role_id": "narrator_main",
                "basis": "explicit", "reason": "Narration.", "evidence": [],
                "candidates": [], "review_required": False,
                "delivery": {"direction": "softly", "evidence": []},
            }
            response["units"].append({"unit_id": unit["unit_id"], "segments": [piece]})
        before = deepcopy(response)
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), units, response)
        error = str(caught.exception)
        self.assertIn("2 segment(s)", error)
        self.assertIn("units[0].segments[0].delivery.evidence", error)
        self.assertIn("units[1].segments[0].delivery.evidence", error)
        self.assertIn("source block 'b1'", error)
        self.assertIn("source block 'b2'", error)
        self.assertIn("exact source quotations", error)
        self.assertLessEqual(len(error), 2000)
        self.assertEqual(response, before)

        # Fixing one location must still reject the other. Supporting the first
        # and clearing the unsupported second allows the unchanged source text.
        response["units"][0]["segments"][0]["delivery"]["evidence"] = [{"block_id": "b1", "quote": "softly"}]
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), units, response)
        self.assertIn("1 segment(s)", str(caught.exception))
        self.assertIn("units[1].segments[0].delivery.evidence", str(caught.exception))
        response["units"][1]["segments"][0]["delivery"]["direction"] = ""
        accepted = align_units(self.source, seed_cast(), units, response)
        self.assertEqual([item["text"] for item in accepted], [unit["text"] for unit in units])

    def test_many_empty_delivery_evidence_paths_fit_repair_feedback_budget(self):
        self.source["blocks"][0]["text"] = "x" * 60
        unit = make_units(self.source)[0][0]
        piece = {
            "text": "x", "kind": "narration", "role_id": "narrator_main",
            "basis": "explicit", "reason": "Narration.", "evidence": [],
            "candidates": [], "review_required": False,
            "delivery": {"direction": "urgently", "evidence": []},
        }
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [deepcopy(piece) for _ in range(60)]}]}
        with self.assertRaises(ValueError) as caught:
            align_units(self.source, seed_cast(), [unit], response)
        error = str(caught.exception)
        self.assertIn("60 segment(s)", error)
        self.assertIn("units[0].segments[0].delivery.evidence", error)
        self.assertIn("additional location(s) omitted", error)
        self.assertIn("Check every nonempty delivery direction", error)
        self.assertLessEqual(len(error), 2000)

    def test_chunking_preserves_unicode_long_blocks_and_chapter_boundaries(self):
        self.source["blocks"][0]["text"] = ("😀é漢e\u0301 words " * 70) + "last"
        self.source["blocks"][1]["text"] = "尾" * 601
        self.source["blocks"][1]["chapter_id"] = "ch2"
        self.source["chapters"][0]["block_ids"] = ["b1"]
        self.source["chapters"].append({"id": "ch2", "title": "Two", "source_href": "text/ch2.xhtml", "block_ids": ["b2"]})
        batches = make_units(self.source, chunk_chars=256)
        flat = [unit for batch in batches for unit in batch]
        self.assertEqual(len({unit["unit_id"] for unit in flat}), len(flat))
        self.assertGreater(len(flat), 4)
        for batch in batches:
            self.assertLessEqual(sum(len(unit["text"]) for unit in batch), 256)
            self.assertEqual(len({unit["chapter_id"] for unit in batch}), 1)
        for block in self.source["blocks"]:
            units = [unit for unit in flat if unit["block_id"] == block["id"]]
            self.assertEqual("".join(unit["text"] for unit in units), block["text"])
            cursor = 0
            for unit in units:
                self.assertEqual(unit["start"], cursor)
                self.assertGreater(len(unit["text"]), 0)
                self.assertLessEqual(len(unit["text"]), 256)
                self.assertEqual(block["text"][cursor:cursor + len(unit["text"])], unit["text"])
                cursor += len(unit["text"])

    def test_invalid_chunk_sizes_are_rejected(self):
        for size in (0, 255, True, 256.0, "256"):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "at least 256"):
                make_units(self.source, size)

    def test_multibatch_analysis_restores_absolute_offsets_without_emitting_context(self):
        self.source["blocks"][0]["text"] += " 😀 αβγδεζ " * 90
        client = FakeClient()
        document = self.run_analysis(client, chunk_chars=256)
        batches = make_units(self.source, chunk_chars=256)
        self.assertGreater(len(batches), 3)
        self.assertEqual(len(client.calls), 2 * len(batches) + 2)
        self.assertEqual(len(document["provenance"]["scene_summaries"]), len(batches))
        self.assertTrue(readiness(document)["ready"])
        self.assertTrue(any(payload["context_units"] for _, payload in client.calls if "target_units" in payload))
        for block in self.source["blocks"]:
            spans = [item for item in document["segments"] if item["block_id"] == block["id"]]
            self.assertEqual("".join(item["text"] for item in spans), block["text"])
            cursor = 0
            for item in spans:
                self.assertEqual(item["start"], cursor)
                self.assertEqual(item["text"], block["text"][item["start"]:item["end"]])
                cursor = item["end"]
            self.assertEqual(cursor, len(block["text"]))

    def test_whole_book_discovery_keeps_original_units_and_one_global_summary(self):
        self.source["blocks"][0]["text"] += " 😀 αβγδεζ " * 90
        self.source["blocks"][1]["chapter_id"] = "ch2"
        self.source["chapters"][0]["block_ids"] = ["b1"]
        self.source["chapters"].append({"id": "ch2", "title": "Two", "source_href": "text/ch2.xhtml", "block_ids": ["b2"]})
        batches = make_units(self.source, chunk_chars=256)
        client = FakeClient()
        document = self.run_analysis(client, chunk_chars=256, whole_book_discovery=True)
        discovery = [payload for _, payload in client.calls if payload.get("mode") == "discover"]
        reconciliation = [payload for _, payload in client.calls if payload.get("mode") == "reconcile"]
        attribution = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertEqual(len(discovery), 1)
        self.assertEqual(len(reconciliation), 1)
        self.assertEqual(len(attribution), len(batches))
        self.assertEqual(discovery[0]["target_units"], [unit for batch in batches for unit in batch])
        self.assertEqual(discovery[0]["context_units"], [])
        self.assertEqual(discovery[0]["scene_summaries"], [])
        self.assertEqual({unit["chapter_id"] for unit in discovery[0]["target_units"]}, {"ch1", "ch2"})
        self.assertTrue(all(len(values) == 256 for values in discovery[0]["available_ids"].values()))
        self.assertIn("book-wide", client.calls[0][0])
        self.assertIn("2400", client.calls[0][0])
        summaries = document["provenance"]["scene_summaries"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["scope"], "book")
        self.assertIsNone(summaries[0]["chapter_id"])
        self.assertEqual(reconciliation[0]["scene_summaries"], summaries)
        self.assertEqual(reconciliation[0]["target_units"], discovery[0]["target_units"])
        self.assertEqual(reconciliation[0]["context_units"], [])
        reconciliation_prompt = next(system for system, payload in client.calls if payload.get("mode") == "reconcile")
        self.assertIn("cast completeness", reconciliation_prompt)
        self.assertIn("chorus or unison", reconciliation_prompt)
        self.assertIn("their solo turns", reconciliation_prompt)
        self.assertIn("role's character_id", reconciliation_prompt)
        self.assertTrue(all(payload["scene_summary"] == summaries[0] for payload in attribution))
        self.assertEqual([payload["target_units"] for payload in attribution], batches)
        self.assertTrue(readiness(document)["ready"])
        repeat = FakeClient()
        self.assertEqual(self.run_analysis(repeat, chunk_chars=256, whole_book_discovery=True), document)
        self.assertEqual(repeat.calls, [])

    def test_concurrent_attribution_is_bounded_ordered_and_resumes_from_cache(self):
        self.source["blocks"][0]["text"] += " 😀 αβγδεζ " * 90
        batches = make_units(self.source, chunk_chars=256)
        client = ConcurrentFakeClient()
        progress = []
        document = self.run_analysis(client, chunk_chars=256, whole_book_discovery=True,
                                     attribution_workers=2, progress=progress.append)
        self.assertEqual(client.max_active, 2)
        self.assertEqual(client.finished[0], batches[1][0]["unit_id"])
        self.assertGreater(client.finished.index(batches[0][0]["unit_id"]), 0)
        self.assertEqual(client.active, 0)
        attribution = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertEqual(len(attribution), len(batches))
        self.assertTrue(all(payload["prior_segments"] == [] for payload in attribution))
        self.assertTrue(all(payload["final_cast"] == document["cast"] for payload in attribution))
        by_id = {payload["target_units"][0]["unit_id"]: payload for payload in attribution}
        all_units = [unit for batch in batches for unit in batch]
        for index, batch in enumerate(batches):
            payload = by_id[batch[0]["unit_id"]]
            before = "".join(unit["text"] for unit in all_units if unit["unit_id"] < batch[0]["unit_id"])[-1500:]
            after = "".join(unit["text"] for unit in all_units if unit["unit_id"] > batch[-1]["unit_id"])[:1500]
            self.assertEqual("".join(unit["text"] for unit in payload["context_units"]), before + after)
        self.assertEqual([message for message in progress if message.startswith("Attributing")],
                         [f"Attributing voices {done}/{len(batches)} complete" for done in range(len(batches) + 1)])
        self.assertEqual(validate_document(document), [])
        for block in self.source["blocks"]:
            spans = [item for item in document["segments"] if item["block_id"] == block["id"]]
            self.assertEqual("".join(item["text"] for item in spans), block["text"])
        repeat = FakeClient()
        expected = deepcopy(document)
        expected["provenance"]["attribution_workers"] = 4
        self.assertEqual(self.run_analysis(repeat, chunk_chars=256, whole_book_discovery=True,
                                          attribution_workers=4), expected)
        self.assertEqual(repeat.calls, [])
        sequential = FakeClient()
        expected["provenance"]["attribution_workers"] = 1
        self.assertEqual(self.run_analysis(sequential, chunk_chars=256, whole_book_discovery=True), expected)
        # The first batch has the same actual prompt; later sequential requests
        # include prior segments and must not reuse the independent prompts.
        self.assertEqual(len(sequential.calls), len(batches) - 1)
        self.assertTrue(all(payload["prior_segments"] for _, payload in sequential.calls))

    def test_whole_book_review_notes_only_change_reconciliation_cache(self):
        client = FakeClient()
        self.run_analysis(client, whole_book_discovery=True)
        original_system, original_payload = client.calls[0]
        # Whole-book mode swaps the batched discovery prompt for one book-wide
        # pass: distinctive markers, not a verbatim copy of the prompt source.
        self.assertIn("brief book-wide identity/narrator summary, at most 2400 characters", original_system)
        self.assertIn("target_units contain the complete book in source order", original_system)
        self.assertNotIn("brief scene/identity/narrator summary", original_system)
        self.assertEqual(original_payload["mode"], "discover")
        self.assertEqual(original_payload["book"], self.source["book"])
        self.assertEqual(original_payload["known_cast"], seed_cast())
        self.assertEqual(original_payload["target_units"],
                         [unit for batch in make_units(self.source) for unit in batch])
        self.assertEqual(original_payload["context_units"], [])
        self.assertEqual(original_payload["scene_summaries"], [])
        self.assertEqual(original_payload["performance_policy"], DEFAULT_POLICY)
        self.assertEqual(set(original_payload["available_ids"]), {"characters", "roles", "voices"})
        self.assertTrue(all(len(ids) == 256 for ids in original_payload["available_ids"].values()))
        notes = "Check whether an unnamed voice has a solo turn."
        repeat = FakeClient()
        document = self.run_analysis(repeat, whole_book_discovery=True, cast_review_notes=notes)
        self.assertEqual(len(repeat.calls), 1)
        system, payload = repeat.calls[0]
        self.assertEqual(payload["mode"], "reconcile")
        self.assertEqual(payload["review_notes"], notes)
        self.assertEqual(payload["target_units"], original_payload["target_units"])
        self.assertIn("untrusted observations", system)
        self.assertIn("Verify them", system)
        self.assertEqual(document["provenance"]["cast_review_notes"], notes)
        self.assertTrue(document["provenance"]["whole_book_discovery"])
        self.assertEqual(document["provenance"]["attribution_workers"], 1)

    def test_performance_policy_reaches_all_discovery_and_attribution_calls_and_cache_keys(self):
        for whole_book in (False, True):
            with self.subTest(whole_book=whole_book):
                self.cache = Path(self.directory.name) / str(whole_book)
                # Multiple incremental batches must each receive the same policy.
                self.source["blocks"][0]["text"] += " More narrative context." * 30
                policy = dict(DEFAULT_POLICY, thoughts="Use the applicable narration role for direct thoughts and mental messages.")
                before = deepcopy(policy)
                client = FakeClient()
                document = self.run_analysis(client, chunk_chars=256, whole_book_discovery=whole_book, policy=policy)
                self.assertEqual(policy, before)
                self.assertEqual(document["provenance"]["performance_policy"], policy)
                self.assertEqual({payload.get("mode", "attribute") for _, payload in client.calls},
                                 {"discover", "reconcile", "voice_ensemble", "attribute"})
                self.assertTrue(all(payload["performance_policy"] == policy for _, payload in client.calls))
                replay = FakeClient()
                self.assertEqual(self.run_analysis(replay, chunk_chars=256, whole_book_discovery=whole_book, policy=policy),
                                 document)
                self.assertEqual(replay.calls, [])
                changed = FakeClient()
                self.run_analysis(changed, chunk_chars=256, whole_book_discovery=whole_book, policy=DEFAULT_POLICY)
                self.assertEqual(len(changed.calls), len(client.calls))
                self.assertTrue(all(payload["performance_policy"] == DEFAULT_POLICY for _, payload in changed.calls))

    def test_neighbor_context_collects_multiple_units_without_crossing_chapters(self):
        template = deepcopy(self.source["blocks"][0])
        self.source["blocks"] = []
        for number in range(1, 21):
            block = deepcopy(template)
            block.update(id=f"b{number}", chapter_id="ch1" if number <= 10 else "ch2",
                         text=("Mara " if number == 1 else f"Paragraph {number}. ") + "x" * 190,
                         inline_spans=[{"start": 10, "end": 190, "kind": "emphasis", "origin": "tag"}])
            self.source["blocks"].append(block)
        self.source["chapters"][0]["block_ids"] = [f"b{number}" for number in range(1, 11)]
        self.source["chapters"].append({"id": "ch2", "title": "Two", "source_href": "text/ch2.xhtml",
                                        "block_ids": [f"b{number}" for number in range(11, 21)]})
        client = FakeClient()
        self.run_analysis(client, chunk_chars=256, whole_book_discovery=True)
        attribution = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertTrue(any(len(payload["context_units"]) > 2 for payload in attribution))
        for payload in attribution:
            target = payload["target_units"][0]
            neighbors = payload["context_units"]
            self.assertTrue(all(unit["chapter_id"] == target["chapter_id"] for unit in neighbors))
            before = [unit for unit in neighbors if unit["unit_id"] < target["unit_id"]]
            after = [unit for unit in neighbors if unit["unit_id"] > target["unit_id"]]
            self.assertLessEqual(sum(len(unit["text"]) for unit in before), 1500)
            self.assertLessEqual(sum(len(unit["text"]) for unit in after), 1500)
            for unit in neighbors:
                block = next(block for block in self.source["blocks"] if block["id"] == unit["block_id"])
                self.assertEqual(unit["text"], block["text"][unit["start"]:unit["start"] + len(unit["text"])])
                for span in unit["inline_spans"]:
                    self.assertGreaterEqual(span["start"], 0)
                    self.assertLessEqual(span["end"], len(unit["text"]))
        chapter_two_start = attribution[10]
        self.assertTrue(all(unit["unit_id"] > chapter_two_start["target_units"][0]["unit_id"]
                            for unit in chapter_two_start["context_units"]))

    def test_concurrent_failures_stop_new_batches_and_never_return_partial_document(self):
        self.source["blocks"][0]["text"] += " more source words " * 120
        client = ConcurrentFakeClient(fail_second=True)
        with self.assertRaisesRegex(RuntimeError, "Fixture transport failure"):
            self.run_analysis(client, chunk_chars=256, whole_book_discovery=True, attribution_workers=2)
        self.assertEqual(len(client.started), 2)
        self.assertEqual(client.active, 0)
        repaired = self.run_analysis(FakeClient(), chunk_chars=256, whole_book_discovery=True,
                                     attribution_workers=2)
        self.assertTrue(readiness(repaired)["ready"])
        for block in self.source["blocks"]:
            self.assertEqual("".join(item["text"] for item in repaired["segments"] if item["block_id"] == block["id"]),
                             block["text"])

    def test_concurrent_attribution_uses_matching_summaries_with_incremental_discovery(self):
        self.source["blocks"][0]["text"] += " more source words " * 80
        client = FakeClient()
        document = self.run_analysis(client, chunk_chars=256, attribution_workers=3)
        batches = make_units(self.source, chunk_chars=256)
        summaries = document["provenance"]["scene_summaries"]
        self.assertEqual(len(summaries), len(batches))
        attribution = {payload["target_units"][0]["unit_id"]: payload
                       for _, payload in client.calls if "mode" not in payload}
        for index, batch in enumerate(batches):
            self.assertEqual(attribution[batch[0]["unit_id"]]["scene_summary"], summaries[index])
        self.assertEqual(len(client.calls), len(batches) * 2 + 2)
        self.assertTrue(readiness(document)["ready"])

    def test_concurrent_invalid_text_is_rejected_without_caching_bad_attribution(self):
        self.source["blocks"][0]["text"] += " more source words " * 120
        client = FakeClient(invalid_text=True)
        with self.assertRaisesRegex(ValueError, "Text changed"):
            self.run_analysis(client, chunk_chars=256, whole_book_discovery=True, attribution_workers=2)
        cached = [json.loads(path.read_text()) for path in self.cache.glob("*.json")]
        self.assertEqual(len(cached), 3)
        self.assertTrue(all("units" not in response for response in cached))

    def test_analysis_options_reject_invalid_types_and_worker_limits_before_requests(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(whole_book_discovery=value), self.assertRaisesRegex(ValueError, "whole_book_discovery"):
                client = FakeClient()
                self.run_analysis(client, whole_book_discovery=value)
            self.assertEqual(client.calls, [])
        for value in (None, 0, True, [], {}):
            with self.subTest(cast_review_notes=value), self.assertRaisesRegex(ValueError, "cast_review_notes"):
                client = FakeClient()
                self.run_analysis(client, cast_review_notes=value)
            self.assertEqual(client.calls, [])
        for value in (None, 0, -1, 9, True, False, 2.0, "2", [], {}):
            with self.subTest(attribution_workers=value), self.assertRaisesRegex(ValueError, "attribution_workers"):
                client = FakeClient()
                self.run_analysis(client, attribution_workers=value)
            self.assertEqual(client.calls, [])

    def test_boundary_whitespace_recovery_emits_only_exact_source_text(self):
        def aligned(source_text, texts):
            source = deepcopy(self.source)
            source["blocks"] = [dict(source["blocks"][0], text=source_text)]
            source["chapters"][0]["block_ids"] = ["b1"]
            unit = make_units(source)[0][0]
            response = {"units": [{"unit_id": unit["unit_id"], "segments": [
                {"text": text, "kind": "narration", "role_id": "narrator_main", "basis": "explicit",
                 "reason": "Fixture narrator.", "evidence": [], "candidates": [], "review_required": False,
                 "delivery": {"direction": "", "evidence": []}}
                for text in texts]}]}
            original = deepcopy(response)
            result = align_units(source, seed_cast(), [unit], response)
            self.assertEqual(response, original)
            self.assertEqual("".join(item["text"] for item in result), source_text)
            cursor = 0
            for item in result:
                self.assertEqual(item["start"], cursor)
                self.assertEqual(item["text"], source_text[item["start"]:item["end"]])
                self.assertEqual(item["id"], f"b1:{item['start']}:{item['end']}")
                cursor = item["end"]
            self.assertEqual(cursor, len(source_text))
            return result

        source_text = " \t“Café”—漢字\u00a0said。\n \t"
        for texts in (["“Café”—漢字", "said。"], [" \n“Café”—漢字  ", "\tsaid。 "]):
            result = aligned(source_text, texts)
            self.assertEqual([item["text"] for item in result], [" \t“Café”—漢字", "\u00a0said。\n \t"])
        exact = ["Alpha \t", "Beta \n"]
        self.assertEqual([item["text"] for item in aligned("".join(exact), exact)], exact)
        for source_text, texts in (
            ("Alpha  Beta", ["Alpha Beta"]),
            ("Alpha, Beta", ["Alpha", "Beta"]),
            ("Alpha Beta", ["Alpha"]),
            ("Alpha Beta", ["Beta", "Alpha"]),
            ("Alpha", ["alpha"]),
            ("Café", ["Cafe\u0301"]),
            ("Alpha", [" \t"]),
            ("Alpha", [""]),
        ):
            with self.subTest(source=source_text, texts=texts), self.assertRaises(ValueError):
                aligned(source_text, texts)

    def test_long_attribution_mismatch_reports_actual_divergence_not_segment_start(self):
        prefix = "A matching sentence. " * 16
        expected_text = "Intro. " + prefix + "The fire may not have even been lit."
        expected_offset = expected_text.index("even ")
        mutations = (
            (expected_text[7:].replace("even ", "", 1), expected_offset, "even been"),
            (expected_text[7:].replace("even ", "truly even ", 1), expected_offset, "truly even"),
            (expected_text[7:].replace("even ", "ever ", 1), expected_offset + 3, "ever been"),
        )
        for changed, offset, excerpt in mutations:
            with self.subTest(offset=offset, excerpt=excerpt):
                source = deepcopy(self.source)
                source["blocks"] = [dict(source["blocks"][0], text=expected_text)]
                source["chapters"][0]["block_ids"] = ["b1"]
                unit = make_units(source)[0][0]
                def piece(text):
                    return {"text": text, "kind": "narration", "role_id": "narrator_main", "basis": "explicit",
                            "reason": "Fixture narrator.", "evidence": [], "candidates": [], "review_required": False,
                            "delivery": {"direction": "", "evidence": []}}
                response = {"units": [{"unit_id": unit["unit_id"], "segments": [piece("Intro. "), piece(changed)]}]}
                original = deepcopy((source, unit, response))
                with self.assertRaises(ValueError) as raised:
                    align_units(source, seed_cast(), [unit], response)
                error = str(raised.exception)
                self.assertIn(f"unit {unit['unit_id']} at offset {offset}", error)
                self.assertIn(f"source block 'b1' offset {offset}", error)
                self.assertIn("units[0].segments[1].text", error)
                self.assertIn(excerpt, error)
                self.assertIn("even been", error)
                self.assertLess(len(error), 1000)
                self.assertEqual((source, unit, response), original)
                response["units"][0]["segments"][1]["text"] = expected_text[7:]
                accepted = align_units(source, seed_cast(), [unit], response)
                self.assertEqual("".join(item["text"] for item in accepted), expected_text)

    def test_mismatch_offsets_account_for_boundary_whitespace_and_source_unit_start(self):
        source = deepcopy(self.source)
        leading = "Earlier block text. "
        unit_text = " \t  " + "Matching words. " * 20 + "even been lit."
        source["blocks"] = [dict(source["blocks"][0], text=leading + unit_text)]
        source["chapters"][0]["block_ids"] = ["b1"]
        unit = {"unit_id": "late_unit", "block_id": "b1", "chapter_id": "ch1", "start": len(leading), "text": unit_text}
        text = "\n " + unit_text.strip().replace("even ", "", 1) + " \t"
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [{"text": text}]}]}
        offset = unit_text.index("even ")
        with self.assertRaises(ValueError) as raised:
            align_units(source, seed_cast(), [unit], response)
        error = str(raised.exception)
        self.assertIn(f"unit late_unit at offset {offset}", error)
        self.assertIn(f"source block 'b1' offset {len(leading) + offset}", error)
        self.assertIn(f"received segment offset {offset - 2}", error)
        self.assertIn("even been lit.", error)

    def test_too_long_attribution_reports_unit_end_and_bounds_escaped_context(self):
        source = deepcopy(self.source)
        text = "Mara " + "\n\t\\" * 1000 + " finished."
        source["blocks"] = [dict(source["blocks"][0], text=text)]
        source["chapters"][0]["block_ids"] = ["b1"]
        unit = {"unit_id": "full_unit", "block_id": "b1", "chapter_id": "ch1", "start": 0, "text": text}
        response = {"units": [{"unit_id": unit["unit_id"], "segments": [{"text": text + " EXTRA"}]}]}
        with self.assertRaises(ValueError) as raised:
            align_units(source, seed_cast(), [unit], response)
        error = str(raised.exception)
        self.assertIn(f"unit full_unit at offset {len(text)}", error)
        self.assertIn("Expected text ends at this unit boundary", error)
        self.assertIn("EXTRA", error)
        self.assertLess(len(error), 1800)
        self.assertNotIn("\n", error)

    def test_repeated_character_mismatch_names_code_points_and_both_run_lengths(self):
        for letter, following in (("s", "a"), ("é", "x")):
            for expected_count, received_count in ((4, 3), (3, 4)):
                with self.subTest(letter=letter, expected=expected_count, received=received_count):
                    source = deepcopy(self.source)
                    text = "“I " + letter * expected_count + following + "w it.”"
                    received = "“I " + letter * received_count + following + "w it.”"
                    source["blocks"] = [dict(source["blocks"][0], text=text)]
                    source["chapters"][0]["block_ids"] = ["b1"]
                    unit = make_units(source)[0][0]
                    response = {"units": [{"unit_id": unit["unit_id"], "segments": [{"text": received}]}]}
                    original = deepcopy((source, unit, response))
                    with self.assertRaises(ValueError) as raised:
                        align_units(source, seed_cast(), [unit], response)
                    error = str(raised.exception)
                    expected_char = letter if expected_count > received_count else following
                    received_char = following if expected_count > received_count else letter
                    self.assertIn("at offset 6", error)
                    self.assertIn(f"Expected character: {expected_char!r} (U+{ord(expected_char):04X})", error)
                    self.assertIn(f"received character: {received_char!r} (U+{ord(received_char):04X})", error)
                    self.assertIn(f"Consecutive {letter!r} (U+{ord(letter):04X}) run", error)
                    self.assertIn(f"expected {expected_count}, received {received_count}", error)
                    self.assertLess(len(error), 1000)
                    self.assertEqual((source, unit, response), original)
    def test_units_clip_inline_spans_and_retain_break_only_at_block_start(self):
        self.source["blocks"][0].update(text="a" * 650, break_before="blank_paragraph", inline_spans=[
            {"start": 250, "end": 270, "kind": "emphasis", "origin": "tag"},
            {"start": 500, "end": 600, "kind": "strong", "origin": "style"},
        ])
        original = deepcopy(self.source)
        units = [unit for batch in make_units(self.source, 256) for unit in batch if unit["block_id"] == "b1"]
        self.assertEqual([unit["start"] for unit in units], [0, 256, 512])
        self.assertEqual(units[0]["break_before"], "blank_paragraph")
        self.assertNotIn("break_before", units[1])
        self.assertNotIn("break_before", units[2])
        self.assertEqual(units[0]["inline_spans"], [{"start": 250, "end": 256, "kind": "emphasis", "origin": "tag"}])
        self.assertEqual(units[1]["inline_spans"], [
            {"start": 0, "end": 14, "kind": "emphasis", "origin": "tag"},
            {"start": 244, "end": 256, "kind": "strong", "origin": "style"},
        ])
        self.assertEqual(units[2]["inline_spans"], [{"start": 0, "end": 88, "kind": "strong", "origin": "style"}])
        self.assertEqual("".join(unit["text"] for unit in units), self.source["blocks"][0]["text"])
        self.assertEqual(self.source, original)

    def test_clipped_context_preserves_correct_typography_offsets(self):
        text = self.source["blocks"][0]["text"] + "x" * 3200
        self.source["blocks"][0].update(text=text, break_before="blank_paragraph", inline_spans=[
            {"start": 10, "end": 20, "kind": "emphasis", "origin": "tag"},
            {"start": len(text) - 1600, "end": len(text) - 1400, "kind": "strong", "origin": "style"},
        ])
        self.source["blocks"][1].update(text="y" * 3200, break_before="blank_paragraph", inline_spans=[
            {"start": 1400, "end": 1600, "kind": "emphasis", "origin": "style"},
        ])
        client = FakeClient()
        self.run_analysis(client, chunk_chars=4096, whole_book_discovery=True)
        unit_stages = [system for system, payload in client.calls if "target_units" in payload]
        self.assertTrue(unit_stages)
        self.assertTrue(all("zero-based, end-exclusive character offsets" in system for system in unit_stages))
        attribution = [payload for _, payload in client.calls if "mode" not in payload]
        self.assertEqual(len(attribution), 2)
        before = attribution[1]["context_units"][0]
        self.assertEqual(before["start"], len(text) - 1500)
        self.assertNotIn("break_before", before)
        self.assertEqual(before["inline_spans"], [{"start": 0, "end": 100, "kind": "strong", "origin": "style"}])
        after = attribution[0]["context_units"][0]
        self.assertEqual(after["start"], 0)
        self.assertEqual(after["break_before"], "blank_paragraph")
        self.assertEqual(after["inline_spans"], [{"start": 1400, "end": 1500, "kind": "emphasis", "origin": "style"}])

    def test_unresolved_dialogue_remains_unassigned_and_requires_review(self):
        document = self.run_analysis(FakeClient(unresolved=True))
        dialogue = document["segments"][1]
        self.assertIsNone(dialogue["role_id"])
        self.assertIsNone(dialogue["voice_id"])
        self.assertEqual(dialogue["attribution"]["basis"], "unresolved")
        self.assertTrue(dialogue["attribution"]["review_required"])
        self.assertEqual(len(dialogue["attribution"]["candidates"]), 1)
        self.assertEqual(document["readiness"], {"ready": False, "errors": [], "review_count": 1})

    def test_corrections_reject_wrong_or_stale_source_fingerprint(self):
        document = self.run_analysis()
        with self.assertRaisesRegex(ValueError, "source_fingerprint"):
            apply_edits(document, {"source_fingerprint": "wrong"})
        edits = {"source_fingerprint": fingerprint(document["source"]),
                 "cast_fingerprint": fingerprint(document["cast"])}
        modified = deepcopy(document)
        modified["source"]["blocks"][0]["text"] += " Added later."
        with self.assertRaisesRegex(ValueError, "source_fingerprint"):
            apply_edits(modified, edits)

    def test_corrections_reject_missing_or_stale_cast_fingerprint(self):
        document = self.run_analysis()
        edits = {"source_fingerprint": fingerprint(document["source"])}
        with self.assertRaisesRegex(ValueError, "cast_fingerprint"):
            apply_edits(document, edits)
        edits["cast_fingerprint"] = fingerprint(document["cast"])
        reanalyzed = deepcopy(document)
        reanalyzed["cast"]["characters"][0]["description"] = "Character details from another analysis."
        with self.assertRaisesRegex(ValueError, "cast_fingerprint"):
            apply_edits(reanalyzed, edits)

    def test_local_reassignment_resolves_only_requested_segment_and_keeps_original(self):
        original = self.run_analysis(FakeClient(unresolved=True))
        snapshot = deepcopy(original)
        pending = original["segments"][1]
        role_id = pending["attribution"]["candidates"][0]
        edits = {"source_fingerprint": fingerprint(original["source"]),
                 "cast_fingerprint": fingerprint(original["cast"]),
                 "assignments": [{"segment_id": pending["id"], "role_id": role_id}]}
        corrected = apply_edits(original, edits)
        self.assertEqual(original, snapshot)
        self.assertEqual(corrected["segments"][0], original["segments"][0])
        self.assertEqual(corrected["segments"][2], original["segments"][2])
        fixed = corrected["segments"][1]
        role = next(role for role in corrected["cast"]["roles"] if role["id"] == role_id)
        self.assertEqual(fixed["role_id"], role_id)
        self.assertEqual(fixed["voice_id"], role["voice_id"])
        self.assertEqual(fixed["attribution"]["basis"], "user")
        self.assertEqual(fixed["text"], pending["text"])
        self.assertTrue(corrected["readiness"]["ready"])
        self.assertEqual(corrected["provenance"]["corrections"], [edits])

    def test_block_replacement_can_change_boundaries_without_changing_text(self):
        original = self.run_analysis()
        dialogue_role = original["segments"][1]["role_id"]
        edits = {"source_fingerprint": fingerprint(original["source"]),
                 "cast_fingerprint": fingerprint(original["cast"]), "blocks": [{
            "block_id": "b1", "segments": [
                {"text": "Mara ", "role_id": "narrator_main", "kind": "narration"},
                {"text": "said softly, ", "role_id": "narrator_main", "kind": "narration"},
                {"text": "“Hello.”", "role_id": dialogue_role, "kind": "dialogue"},
            ],
        }]}
        corrected = apply_edits(original, edits)
        self.assertEqual(len(corrected["segments"]), 4)
        self.assertEqual(len(original["segments"]), 3)
        self.assertEqual([item["start"] for item in corrected["segments"][:3]], [0, 5, 18])
        self.assertEqual(corrected["segments"][-1], original["segments"][-1])
        self.assertTrue(all(item["attribution"]["basis"] == "user" for item in corrected["segments"][:3]))
        self.assertEqual(corrected["source"], original["source"])
        self.assertTrue(corrected["readiness"]["ready"])
        edits["blocks"][0]["segments"][0]["text"] = "Maria "
        with self.assertRaisesRegex(ValueError, "Text changed"):
            apply_edits(original, edits)

    def test_cast_merge_redirects_dialogue_roles_and_leaves_review_issue(self):
        document = self.run_analysis(FakeClient(duplicate=True))
        self.assertEqual(len(document["cast"]["characters"]), 1)
        mara = document["cast"]["characters"][0]
        self.assertIn("the speaker", mara["aliases"])
        dialogue_roles = [role for role in document["cast"]["roles"] if role["kind"] == "dialogue"]
        self.assertEqual(len(dialogue_roles), 2)
        self.assertTrue(all(role["character_id"] == mara["id"] for role in dialogue_roles))
        self.assertTrue(all(role["voice_id"] == mara["voice_id"] for role in dialogue_roles))
        self.assertEqual(document["issues"][0]["kind"], "identity_merge")
        self.assertEqual(document["issues"][0]["target"], mara["id"])
        self.assertFalse(document["issues"][0]["resolved"])
        self.assertEqual(document["readiness"], {"ready": True, "errors": [], "review_count": 1})
        corrected = apply_edits(document, {
            "source_fingerprint": fingerprint(document["source"]),
            "cast_fingerprint": fingerprint(document["cast"]),
            "resolved_issues": [document["issues"][0]["id"]],
        })
        self.assertTrue(corrected["readiness"]["ready"])

    def test_merge_with_target_upsert_preserves_raw_cache_and_replays_identically(self):
        original = self.run_analysis(FakeClient(duplicate=True))
        cached_merges = [json.loads(path.read_text()) for path in self.cache.glob("*.json")
                         if json.loads(path.read_text()).get("merges")]
        self.assertEqual(len(cached_merges), 1)
        cached_target = cached_merges[0]["characters"][0]
        self.assertEqual(len(cached_target["evidence"]), 1)
        self.assertNotIn("the speaker", cached_target["aliases"])
        self.assertEqual(len(original["cast"]["characters"][0]["evidence"]), 3)
        for _ in range(2):
            client = FakeClient(duplicate=True)
            repeated = self.run_analysis(client)
            self.assertEqual(repeated, original)
            self.assertEqual(client.calls, [])

    def test_merge_preserves_separate_narration_voice_but_redirects_shared_voice(self):
        client = FakeClient(duplicate=True)
        available = {"characters": ["c1", "c2"], "roles": ["r1", "r2"], "voices": ["v1", "v2"]}
        payload = {"known_cast": seed_cast(), "available_ids": available,
                   "target_units": make_units(self.source)[0]}
        cast, _, _ = merge_patch(self.source, seed_cast(), client.discover(payload), available)
        cast["roles"].extend([
            {"id": "shared_narration", "kind": "narration", "character_id": "c2",
             "voice_id": "v2", "description": "Narrating in their dialogue voice."},
            {"id": "separate_narration", "kind": "narration", "character_id": "c2",
             "voice_id": "voice_narrator_main", "description": "Narrating in a distinct voice."},
        ])
        response = client.patch(merges=[{"from_character_id": "c2", "into_character_id": "c1",
                                        "evidence": [{"block_id": "b1", "quote": "Mara"}]}])
        updated, issues, _ = merge_patch(self.source, cast, response, available)
        roles = {role["id"]: role for role in updated["roles"]}
        self.assertEqual(roles["shared_narration"]["voice_id"], "v1")
        self.assertEqual(roles["separate_narration"]["voice_id"], "voice_narrator_main")
        self.assertEqual(roles["shared_narration"]["character_id"], "c1")
        self.assertEqual(roles["separate_narration"]["character_id"], "c1")
        self.assertEqual(len(issues), 1)
        self.assertEqual(len(cast["characters"]), 2)


if __name__ == "__main__":
    unittest.main()
