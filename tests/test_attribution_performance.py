"""Inline proposal review: mechanical passes, remove-only review, and publication."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.attribution_compact import parse_proposals
from epub_tts_narrator.attribution_performance import review_inline_performance
from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.pipeline import CachedRequests
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.spoken_text import build_spoken


def segment(block_id, text, start=0, *, proposals=None, performance=None):
    row = {"id": f"{block_id}:{start}:{start + len(text)}", "block_id": block_id, "start": start,
           "end": start + len(text), "text": text, "kind": "dialogue", "role_id": "mara",
           "voice_id": "v1", "delivery": {"direction": "", "evidence": []},
           "attribution": {"basis": "explicit", "reason": "Fixture.", "evidence": [],
                           "candidates": [], "review_required": False}}
    if performance is not None:
        row["performance"] = performance
    if proposals is not None:
        row["inline_proposals"] = parse_proposals(deepcopy(proposals), text, f"segment {block_id!r}")
    return row


def document(rows, *, block_text=None, characters=None):
    return {"format": "epub-audiobook-analysis", "version": 1,
            "source": {"blocks": [{"id": "b1", "chapter_id": "ch1",
                                   "text": block_text if block_text is not None else "", "source": {}}]},
            "cast": {"characters": characters or []}, "segments": rows, "issues": [], "provenance": {}}


class ReviewClient:
    identity = {"base_url": "https://offline.invalid/v1", "model": "fixture"}

    def __init__(self, verdict="keep", outcomes=None, failures=0):
        self.verdict = verdict
        self.outcomes = outcomes or {}
        self.failures = failures
        self.calls = []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if self.failures:
            self.failures -= 1
            raise ProviderError("Fixture transport failure.", retryable=True)
        decisions = []
        for item in payload["proposals"]:
            verdict = self.outcomes.get(item["id"], self.verdict)
            decisions.append({"id": item["id"], "verdict": verdict})
        return {"decisions": decisions}


class MalformedClient(ReviewClient):
    def __init__(self, mutate):
        super().__init__()
        self.mutate = mutate
        self.attempts = 0

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        response = {"decisions": [{"id": item["id"], "verdict": "keep"}
                                  for item in payload["proposals"]]}
        if self.attempts == 0:
            self.attempts += 1
            self.mutate(response, payload["proposals"])
            return response
        return response


class CueClient(ReviewClient):
    """Keeps a proposal only when its cue word is visible in the review payload."""

    def __init__(self, cue):
        super().__init__()
        self.cue = cue

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        visible = " ".join(item["text"] for item in payload.get("segments", [])
                           + payload.get("context", []))
        verdict = "keep" if self.cue in visible else "remove"
        return {"decisions": [{"id": item["id"], "verdict": verdict}
                              for item in payload["proposals"]]}


class InlinePerformanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "cache"

    def request(self, client, *, repair_attempts=0):
        return CachedRequests(client, self.cache, repair_attempts=repair_attempts)

    def test_mechanical_stretches_publish_without_any_review_request(self):
        row = segment("b1", "“Sooo,” she said.", proposals={
            "edits": [{"original": "Sooo", "replacement": "So", "direction": "stretched sigh",
                       "reason": "Bounded run-length stretch."}],
            "controls": []})
        client = ReviewClient()
        result = review_inline_performance(document([row]), self.request(client))
        self.assertEqual(client.calls, [])
        published = result["segments"][0]
        self.assertEqual(published["speech_normalization"], {"version": 1, "edits": [
            {"start": 1, "end": 5, "original": "Sooo", "replacement": "So",
             "direction": "stretched sigh", "reason": "Bounded run-length stretch."}]})
        self.assertEqual(published["spoken_text"], "“So,” she said.")
        self.assertEqual(published["performance_controls"]["controls"], [])
        self.assertNotIn("inline_proposals", published)
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"mechanical": 1})
        self.assertEqual(result["provenance"]["inline_performance"]["batches"], [])

    def test_rejected_and_uncertain_proposals_retain_source_and_drop_dependent_controls(self):
        proposals = {
            "edits": [{"original": "gasped", "replacement": "gasp'd", "direction": "softer voice",
                       "reason": "Dialect softening."}],
            "controls": [{"kind": "delivery", "tag": "whisper", "original": "gasped",
                          "reason": "Whispered gasp."},
                         {"kind": "pause", "duration_ms": 250, "position": "before", "original": "she",
                          "reason": "A beat before the tag."}]}
        row = segment("b1", "“Nooo,” she gasped.", proposals=proposals)
        for verdict, outcome in (("remove", "removed"), ("uncertain", "uncertain")):
            with self.subTest(verdict=verdict):
                cache = Path(self.directory.name) / f"cache-{verdict}"
                client = ReviewClient(outcomes={"r001": verdict, "r002": "keep", "r003": "keep"})
                result = review_inline_performance(document(deepcopy([row])),
                                                   CachedRequests(client, cache))
                published = result["segments"][0]
                self.assertEqual(published["text"], "“Nooo,” she gasped.")
                self.assertNotIn("speech_normalization", published)
                controls = published["performance_controls"]["controls"]
                self.assertEqual([control["kind"] for control in controls], ["pause"])
                self.assertNotIn("inline_proposals", published)
                audit = result["provenance"]["inline_performance"]
                self.assertEqual(audit["counts"], {"dependent_drop": 1, "kept": 1, outcome: 1})
                self.assertEqual([item["outcome"] for item in audit["outcomes"]],
                                 [outcome, "dependent_drop", "kept"])

    def test_review_covers_exact_ids_once_and_repairs_malformed_responses(self):
        row = segment("b1", "“Nooo,” she said quietly.", proposals={
            "edits": [{"original": "quietly", "replacement": "quietlike", "direction": "plain tone",
                       "reason": "Dialect respelling."}],
            "controls": [{"kind": "pause", "duration_ms": 200, "position": "after", "original": "Nooo",
                          "reason": "Trailing beat."}]})
        cases = [
            ("missing decision", lambda response, proposals: response["decisions"].pop()),
            ("unknown decision", lambda response, proposals: response["decisions"].append(
                {"id": "r999", "verdict": "keep"})),
            ("duplicate decision", lambda response, proposals: response["decisions"].append(
                dict(response["decisions"][0]))),
            ("bad verdict", lambda response, proposals: response["decisions"][0].update(verdict="maybe")),
            ("nonobject decision", lambda response, proposals: response["decisions"].append("keep")),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                cache = Path(self.directory.name) / f"cache-{label.replace(' ', '-')}"
                client = MalformedClient(mutate)
                result = review_inline_performance(document(deepcopy([row])),
                                                   CachedRequests(client, cache, repair_attempts=1))
                self.assertEqual(client.attempts, 1)
                self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"kept": 2})
                published = result["segments"][0]
                self.assertIn("speech_normalization", published)
                self.assertIn("performance_controls", published)

    def test_transport_failure_propagates_and_a_rerun_resumes_from_the_cached_review(self):
        row = segment("b1", "“Nooo,” she said.", proposals={
            "edits": [{"original": "said", "replacement": "says", "direction": "present tense",
                       "reason": "Dialect tense."}],
            "controls": []})
        failing = ReviewClient(failures=1)
        with self.assertRaises(ProviderError):
            review_inline_performance(document(deepcopy([row])), self.request(failing))
        resumed = ReviewClient()
        result = review_inline_performance(document(deepcopy([row])), self.request(resumed))
        self.assertEqual(len(resumed.calls), 1)
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"kept": 1})
        replayed = ReviewClient()
        replayed_request = CachedRequests(replayed, self.cache, repair_attempts=0)
        review_inline_performance(document(deepcopy([row])), replayed_request)
        self.assertEqual(replayed.calls, [])
        self.assertEqual(replayed_request.cache_hits, 1)

    def test_large_sparse_sets_split_into_bounded_batches(self):
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
        rows = []
        for index in range(3):
            text = f"“{index} {' '.join(words)}.”"
            edits = [{"original": word, "replacement": word[:-1], "direction": "clipped ending",
                      "reason": f"Dialect clipping {word}."} for word in words]
            rows.append(segment("b1", text, proposals={"edits": edits, "controls": []}))
        client = ReviewClient()
        result = review_inline_performance(document(rows), self.request(client))
        sizes = [len(payload["proposals"]) for payload in client.calls]
        self.assertEqual(sizes, [24, 3])
        batch_ids = [item["id"] for payload in client.calls for item in payload["proposals"]]
        self.assertEqual(batch_ids, [f"r{number:03d}" for number in range(1, 28)])
        audit = result["provenance"]["inline_performance"]
        self.assertEqual(len(audit["batches"]), 2)
        self.assertTrue(all("request_cache_fingerprint" in batch for batch in audit["batches"]))
        self.assertEqual(audit["counts"], {"kept": 27})

    def test_controls_bind_to_spoken_text_when_source_brackets_are_omitted(self):
        text = "“[Hm],” she said."
        row = segment("b1", text, proposals={
            "edits": [],
            "controls": [{"kind": "delivery", "tag": "soft", "original": "Hm",
                          "direction": "a soft considering sound", "reason": "Bracketed aside."}]},
            performance={"kind": "bracketed", "group_id": "bracket:b1:1", "direction": "Soften.",
                         "delimiters": "omit", "enter": True, "exit": True})
        client = ReviewClient()
        result = review_inline_performance(document([row]), self.request(client))
        published = result["segments"][0]
        spoken, _ = build_spoken(published)
        self.assertEqual(spoken, "“Hm,” she said.")
        controls = published["performance_controls"]
        self.assertEqual(validate_performance(controls, spoken), controls)
        self.assertEqual(controls["text_sha256"], text_sha256(spoken))
        style = controls["controls"][0]
        self.assertEqual((style["kind"], spoken[style["start"]:style["end"]]),
                         ("delivery", "Hm"))

    def test_rows_without_proposals_are_returned_untouched_without_requests(self):
        rows = [segment("b1", "Plain narration."), segment("b2", "More narration.", 0)]
        client = ReviewClient()
        payload = document(deepcopy(rows))
        result = review_inline_performance(payload, self.request(client))
        self.assertEqual(client.calls, [])
        self.assertIs(result, payload)
        self.assertNotIn("inline_performance", payload["provenance"])

    def test_bracket_split_siblings_only_publish_where_the_locator_matches(self):
        proposals = {"edits": [], "controls": [{"kind": "pause", "duration_ms": 300,
                                                "position": "after", "original": "Nooo",
                                                "reason": "Beat after the word."}]}
        cleaned = parse_proposals(deepcopy(proposals), "“Nooo,”", "segment b1!r")
        first = segment("b1", "“Nooo,”")
        first["inline_proposals"] = cleaned
        second = segment("b1", " she said.", 7)
        second["inline_proposals"] = deepcopy(cleaned)
        client = ReviewClient()
        result = review_inline_performance(document([first, second]), self.request(client))
        self.assertEqual([control["offset"] for row in result["segments"]
                          for control in row.get("performance_controls", {}).get("controls", [])],
                         [5])
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"kept": 1})

    def test_mechanical_approval_stays_limited_to_shrunk_elongated_runs(self):
        suspicious = [("of", "off"), ("god", "good"), ("off", "of"), ("run", "runn")]
        for original, replacement in suspicious:
            with self.subTest(pair=f"{original}->{replacement}"):
                row = segment("b1", f"“{original},” she said.", proposals={
                    "edits": [{"original": original, "replacement": replacement,
                               "direction": "spoken plainly", "reason": "Suspected elongation."}],
                    "controls": []})
                client = ReviewClient()
                result = review_inline_performance(document(deepcopy([row])), self.request(client))
                self.assertEqual(len(client.calls), 1,
                                 "a non-elongating pair must reach the reviewer, not auto-approve")
                self.assertEqual(result["segments"][0]["speech_normalization"]["edits"],
                                 [{"start": 1, "end": 1 + len(original), "original": original,
                                   "replacement": replacement, "direction": "spoken plainly",
                                   "reason": "Suspected elongation."}])

    def test_mixed_mechanical_and_suspicious_edits_review_only_the_suspicious_one(self):
        row = segment("b1", "“Nooo of god,” she said.", proposals={
            "edits": [{"original": "Nooo", "replacement": "No", "direction": "drawn out",
                       "reason": "Clear elongation."},
                      {"original": "of", "replacement": "off", "direction": "spoken plainly",
                       "reason": "Suspected elongation."}],
            "controls": []})
        client = ReviewClient()
        result = review_inline_performance(document(deepcopy([row])), self.request(client))
        reviewed = [item for payload in client.calls for item in payload["proposals"]]
        self.assertEqual([(item["locator"]["original"], item["proposal"]["replacement"])
                          for item in reviewed], [("of", "off")])
        audit = result["provenance"]["inline_performance"]
        self.assertEqual(audit["counts"], {"kept": 1, "mechanical": 1})

    def test_protected_lexical_edits_are_dropped_before_any_review(self):
        characters = [{"id": "c1", "name": "Mara", "aliases": [], "description": "A speaker.",
                       "voice_id": "v1"}]
        cases = [
            ("digit", "3 spoons", "three spoons"),
            ("negation", "never there", "ever there"),
            ("name", "Mara said", "Mera said"),
            ("number word", "one spoon", "ono spoon"),
        ]
        for label, original, replacement in cases:
            with self.subTest(guard=label):
                row = segment("b1", f"“{original}.”", proposals={
                    "edits": [{"original": original, "replacement": replacement,
                               "direction": "spoken plainly", "reason": "Suspected dialect."}],
                    "controls": []})
                client = ReviewClient()
                result = review_inline_performance(document(deepcopy([row]), characters=characters),
                                                   self.request(client))
                self.assertEqual(client.calls, [], "protected edits must not pay for review")
                self.assertNotIn("speech_normalization", result["segments"][0])
                self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"protected": 1})

    def test_broken_word_coverage_is_dropped_before_any_review(self):
        row = segment("b1", "“alpha beta.”", proposals={
            "edits": [{"original": "alpha beta", "replacement": "alpha", "direction": "spoken plainly",
                       "reason": "Suspected dialect."}],
            "controls": []})
        client = ReviewClient()
        result = review_inline_performance(document(deepcopy([row])), self.request(client))
        self.assertEqual(client.calls, [])
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"uncovered": 1})

    def test_elongated_negation_still_auto_approves(self):
        row = segment("b1", "“nooot now,” she said.", proposals={
            "edits": [{"original": "nooot", "replacement": "not", "direction": "drawn out refusal",
                       "reason": "Stretched negation."}],
            "controls": []})
        client = ReviewClient()
        result = review_inline_performance(document(deepcopy([row])), self.request(client))
        self.assertEqual(client.calls, [])
        published = result["segments"][0]
        self.assertEqual(published["speech_normalization"]["edits"],
                         [{"start": 1, "end": 6, "original": "nooot", "replacement": "not",
                           "direction": "drawn out refusal", "reason": "Stretched negation."}])
        self.assertEqual(published["spoken_text"], "“not now,” she said.")

    def test_reviewer_sees_the_adjacent_speech_tag_cue_outside_the_quoted_row(self):
        quote = segment("b1", "“Nooo,”", proposals={
            "edits": [],
            "controls": [{"kind": "delivery", "tag": "whisper", "original": "Nooo",
                          "reason": "The narration states a whisper."}]})
        tag = segment("b1", " Mara whispered the word.", 7)
        client = CueClient("whispered")
        result = review_inline_performance(document([quote, tag]), self.request(client))
        self.assertEqual(len(client.calls), 1)
        context_ids = [item["segment_id"] for item in client.calls[0].get("context", [])]
        self.assertIn(tag["id"], context_ids)
        controls = result["segments"][0].get("performance_controls", {}).get("controls", [])
        self.assertEqual([control["tag"] for control in controls], ["whisper"])
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"kept": 1})

    def test_reviewer_without_the_neighbor_cue_rejects_the_control(self):
        quote = segment("b1", "“Nooo,”", proposals={
            "edits": [],
            "controls": [{"kind": "delivery", "tag": "whisper", "original": "Nooo",
                          "reason": "Unsupported drama."}]})
        client = CueClient("whispered")
        result = review_inline_performance(document([quote]), self.request(client))
        self.assertEqual(result["segments"][0].get("performance_controls", {}).get("controls", []), [])
        self.assertEqual(result["provenance"]["inline_performance"]["counts"], {"removed": 1})

    def test_oversized_single_row_payload_stays_bounded_around_the_proposals(self):
        filler = "x" * 6800
        text = f"“Nooo {filler} zebra,” she said."
        row = segment("b1", text, proposals={
            "edits": [{"original": "zebra", "replacement": "zeebra", "direction": "drawn opening",
                       "reason": "Suspected elongation."}],
            "controls": []})
        client = ReviewClient()
        result = review_inline_performance(document(deepcopy([row])), self.request(client))
        self.assertEqual(len(client.calls), 1)
        visible = client.calls[0]["segments"][0]["text"]
        self.assertLessEqual(len(visible), 6000)
        self.assertIn("zebra", visible)
        self.assertEqual(result["segments"][0]["speech_normalization"]["edits"],
                         [{"start": 6807, "end": 6812, "original": "zebra", "replacement": "zeebra",
                           "direction": "drawn opening", "reason": "Suspected elongation."}])

    def test_control_boundary_inside_an_approved_replacement_is_dropped_not_widened(self):
        row = segment("b1", "“Sooo,” she said.", proposals={
            "edits": [{"original": "Sooo", "replacement": "So", "direction": "stretched sigh",
                       "reason": "Clear elongation."}],
            "controls": [{"kind": "vocalization", "sound": "sob", "position": "before",
                          "original": "ooo", "reason": "Crying onset."},
                         {"kind": "pause", "duration_ms": 300, "position": "after",
                          "original": "ooo", "reason": "Beat after the word."}]})
        client = ReviewClient()
        result = review_inline_performance(document(deepcopy([row])), self.request(client))
        published = result["segments"][0]
        spoken = published["spoken_text"]
        controls = published["performance_controls"]["controls"]
        self.assertEqual([(control["kind"], control.get("offset")) for control in controls],
                         [("pause", 3)])
        self.assertEqual(published["performance_controls"]["text_sha256"], text_sha256(spoken))
        self.assertEqual(result["provenance"]["inline_performance"]["counts"],
                         {"kept": 1, "mechanical": 1, "unmapped": 1})

    def test_cross_boundary_locator_context_resolves_from_the_source_block(self):
        block_text = "“Nooo,” she said."
        proposals = {"edits": [{"original": "said", "replacement": "saide",
                                "direction": "drawled ending", "reason": "Dialect drawl.",
                                "left": "“Nooo,” she "}],
                     "controls": []}
        cleaned = parse_proposals(deepcopy(proposals), block_text, "segment b1!r")
        first = segment("b1", "“Nooo,”")
        second = segment("b1", " she said.", 7)
        second["inline_proposals"] = deepcopy(cleaned)
        client = ReviewClient()
        result = review_inline_performance(document([first, second], block_text=block_text),
                                           self.request(client))
        self.assertEqual(len(client.calls), 1)
        published = result["segments"][1]
        self.assertEqual(published["speech_normalization"]["edits"],
                         [{"start": 5, "end": 9, "original": "said", "replacement": "saide",
                           "direction": "drawled ending", "reason": "Dialect drawl."}])
        self.assertEqual(published["spoken_text"], " she saide.")

    def test_lost_proposals_leave_no_private_fields_behind(self):
        block_text = "“Nooo,” she said."
        proposals = {"edits": [{"original": "“Nooo,” she", "replacement": "“Nooo,” shee",
                                "direction": "spoken plainly", "reason": "Crossing locator."}],
                     "controls": []}
        cleaned = parse_proposals(deepcopy(proposals), block_text, "segment b1!r")
        first = segment("b1", "“Nooo,”")
        first["inline_proposals"] = deepcopy(cleaned)
        second = segment("b1", " she said.", 7)
        second["inline_proposals"] = deepcopy(cleaned)
        client = ReviewClient()
        result = review_inline_performance(document([first, second], block_text=block_text),
                                           self.request(client))
        self.assertEqual(client.calls, [])
        for row in result["segments"]:
            self.assertNotIn("inline_proposals", row)
            self.assertNotIn("speech_normalization", row)
        self.assertNotIn("inline_performance", result["provenance"])


if __name__ == "__main__":
    unittest.main()
