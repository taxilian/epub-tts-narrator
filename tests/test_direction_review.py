"""Rerunning analyze reviews every spoken line's direction and replaces only the wrong ones."""

from copy import deepcopy
import contextlib
import io
import json
from threading import Lock
import unittest

from epub_tts_narrator import direction_review
from epub_tts_narrator.config import DEFAULTS
from test_attribution_verification_cli import AttributionCLIHarness


class Reviewer:
    """Corrects the direction of the lines in ``fixes`` (spoken text → new direction)."""

    def __init__(self, fixes=None):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.fixes, self.calls, self.lock = dict(fixes or {}), [], Lock()

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        if payload["mode"] != "direction_review":
            raise AssertionError(f"Unexpected request mode: {payload['mode']}")
        rows = []
        for line in payload["text"].splitlines():
            for text, direction in self.fixes.items():
                if text in line:
                    qid = line.split("‹", 1)[1].split(" ", 1)[0]
                    rows.append([qid, direction])
        return {"q": rows}


class DirectionReviewTests(AttributionCLIHarness):
    def review(self, client):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            published = direction_review.review_book_directions(self.book, config, lambda settings: client,
                                                                lambda message: None)
        return published, output.getvalue()

    def directions(self):
        analysis = json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))
        return {row["text"]: row["delivery"]["direction"] for row in analysis["segments"] if row["kind"] == "dialogue"}

    def directed(self, document):
        for row in document["segments"]:
            if row["kind"] == "dialogue":
                row["delivery"] = {"direction": "furious, shouting", "evidence": []}

    def test_only_a_wrong_direction_is_replaced_and_the_book_is_not_reviewed_twice(self):
        self.make_book(risk=False, prepare=self.directed)
        client = Reviewer({"Hello": "soft, friendly"})
        published, output = self.review(client)
        self.assertTrue(published, output)
        self.assertEqual(self.directions(), {"“Hello.”": "soft, friendly", "They left.": "furious, shouting"})
        self.assertIn("1 corrected", output)
        # The reviewer saw each line's current direction.
        self.assertEqual(sorted(client.calls[0]["d"].values()), ["furious, shouting", "furious, shouting"])

        again = Reviewer({"Hello": "different again"})
        published, output = self.review(again)
        self.assertFalse(published)
        self.assertEqual(again.calls, [])
        self.assertIn("already reviewed", output)

    def test_a_typed_direction_is_never_sent_or_changed(self):
        def typed(document):
            self.directed(document)
            for row in document["segments"]:
                if row["kind"] == "dialogue":
                    row["delivery"]["basis"] = "user"
        self.make_book(risk=False, prepare=typed)
        client = Reviewer({"Hello": "soft, friendly", "They left": "flat"})
        self.review(client)
        self.assertEqual(client.calls, [])
        self.assertEqual(set(self.directions().values()), {"furious, shouting"})

    def test_a_changed_paragraph_is_reviewed_again(self):
        self.make_book(risk=False, prepare=self.directed)
        self.review(Reviewer())
        document = json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(direction_review.unreviewed_blocks(document), [])
        row = next(row for row in document["segments"] if row["text"] == "They left.")
        row["delivery"]["direction"] = "something new"
        self.assertEqual(direction_review.unreviewed_blocks(document), [row["block_id"]])


if __name__ == "__main__":
    unittest.main()
