"""Transport failures retain validated work through analysis and publication."""

from collections import Counter
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from threading import Event, Lock
import unittest
from unittest.mock import patch

from epub_audiobook import pipeline
from epub_audiobook.analysis_directory import continuation_directories
from epub_audiobook.cli import main
from epub_audiobook.contracts import validate_document
from epub_audiobook.prompts import PROMPT_VERSION
from epub_audiobook.provider import ProviderError
from test_contracts import valid_document
from test_identity_pipeline import JuryClient
from test_pipeline import FakeClient


def source_fixture(number=1):
    source = deepcopy(valid_document()["source"])
    source["book"].update(id=f"book{number}", title=f"A Short Walk {number}")
    original = source["blocks"][0]
    source["blocks"] = [dict(deepcopy(original), id=f"b{index + 1}",
        text="Mara said softly, “Hello.” " + "The path led toward the lighthouse. " * 5)
        for index in range(12)]
    source["chapters"][0]["block_ids"] = [block["id"] for block in source["blocks"]]
    return source


class DurableReplies:
    """Release the failing request only after four peers are saved on disk."""

    def __init__(self, stage):
        self.stage = stage
        self.ready = Event()
        self.lock = Lock()
        self.keys = set()
        self.write_json = pipeline.write_json

    def write(self, path, response, **kwargs):
        self.write_json(path, response, **kwargs)
        field = "units" if self.stage == "attribute" else "votes"
        if field in response:
            with self.lock:
                self.keys.add(Path(path).stem)
                if len(self.keys) >= 4:
                    self.ready.set()


class TimeoutClient(JuryClient):
    model = "fixture-model"
    base_url = "http://offline.invalid/v1"

    def __init__(self, *, fail_stage=None, durable=None, semantic=False, duplicate=True):
        super().__init__(duplicate=duplicate)
        self.fail_stage, self.durable, self.semantic = fail_stage, durable, semantic
        self.attempted, self.stage_counts = {}, Counter()
        self.failed_key = None
        self.lock = Lock()

    def complete(self, system, payload):
        stage = payload.get("mode", "attribute")
        key = pipeline.fingerprint({"provider": self.identity, "prompt_version": PROMPT_VERSION,
                                    "system": system, "payload": payload})
        with self.lock:
            self.attempted[key] = stage
            self.stage_counts[stage] += 1
            fail = stage == self.fail_stage and self.stage_counts[stage] == 5
        if fail:
            if not self.durable.ready.wait(5):
                raise AssertionError("The completed peer responses were not saved before timeout")
            self.failed_key = key
            raise ProviderError("API request timed out.", retryable=True)
        if stage == "attribution_review_scan":
            # Initial narration is deliberately wrong; this independent scan
            # proposes the same exact source spans the ordinary fixture uses.
            expanded = deepcopy(payload)
            for character in expanded["final_cast"]["characters"]:
                character["evidence"] = [{"block_id": payload["target_units"][0]["block_id"],
                                          "quote": "Mara said softly"}]
            proposed = FakeClient.attribute(self, expanded)["units"]
            return {"corrections": [
                {"block_id": unit["block_id"],
                 "reason": "The speech tag names Mara as the speaker.",
                 "evidence": [{"block_id": unit["block_id"], "quote": "Mara said softly"}],
                 "segments": row["segments"]}
                for unit, row in zip(payload["target_units"], proposed)]}
        if stage == "attribution_review_vote":
            return {"votes": [
                {"block_id": proposal["block_id"], "verdict": "approve",
                 "reason": "Mara's explicit speech tag supports the proposed assignment.",
                 "evidence": [{"block_id": proposal["block_id"], "quote": "Mara said softly"}]}
                for proposal in payload["proposals"]]}
        return super().complete(system, payload)

    def attribute(self, payload):
        result = super().attribute(payload)
        if self.semantic:
            for unit in result["units"]:
                for segment in unit["segments"]:
                    segment.update(kind="narration", role_id="narrator_main",
                                   delivery={"direction": "", "evidence": []})
        return result


class TimeoutResumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "cache"
        self.source = source_fixture()

    def analyze(self, client, *, workers=5, semantic=False, cache=None):
        return pipeline.analyze(
            self.source, client, cache or self.cache, chunk_chars=256,
            whole_book_discovery=True, attribution_workers=workers,
            repair_attempts=0, attribution_review=semantic,
            attribution_review_chunk_chars=256,
        )

    def cached(self, cache=None):
        return {path.stem: path.read_bytes() for path in (cache or self.cache).glob("*.json")}

    def assert_source_preserved(self, result):
        self.assertEqual(validate_document(result), [])
        self.assertEqual(result["source"], self.source)
        for block in self.source["blocks"]:
            self.assertEqual("".join(item["text"] for item in result["segments"]
                                     if item["block_id"] == block["id"]), block["text"])

    def test_late_attribution_timeout_resumes_serial_and_five_worker_runs(self):
        for workers in (1, 5):
            with self.subTest(workers=workers):
                cache = self.cache / str(workers)
                durable = DurableReplies("attribute")
                first = TimeoutClient(fail_stage="attribute", durable=durable)
                with patch("epub_audiobook.pipeline.write_json", side_effect=durable.write):
                    with self.assertRaises(ProviderError) as raised:
                        self.analyze(first, workers=workers, cache=cache)
                self.assertTrue(raised.exception.retryable)
                saved = self.cached(cache)
                self.assertGreaterEqual(len(durable.keys), 4)
                self.assertNotIn(first.failed_key, saved)
                self.assertEqual(first.stage_counts["identity_vote"], 3)

                resumed = TimeoutClient()
                result = self.analyze(resumed, workers=workers, cache=cache)
                self.assertIn(first.failed_key, resumed.attempted)
                self.assertTrue(saved.keys().isdisjoint(resumed.attempted))
                self.assertEqual(set(resumed.attempted.values()), {"attribute"})
                self.assertEqual({key: self.cached(cache)[key] for key in saved}, saved)
                self.assert_source_preserved(result)

                offline = TimeoutClient()
                self.assertEqual(self.analyze(offline, workers=workers, cache=cache), result)
                self.assertEqual(offline.attempted, {})

    def test_five_worker_semantic_jury_timeout_keeps_attribution_and_completed_votes(self):
        durable = DurableReplies("attribution_review_vote")
        first = TimeoutClient(fail_stage="attribution_review_vote", durable=durable, semantic=True)
        with patch("epub_audiobook.pipeline.write_json", side_effect=durable.write):
            with self.assertRaises(ProviderError) as raised:
                self.analyze(first, semantic=True)
        self.assertTrue(raised.exception.retryable)
        saved = self.cached()
        self.assertGreaterEqual(len(durable.keys), 4)
        self.assertGreater(first.stage_counts["attribute"], 5)
        self.assertEqual(first.stage_counts["attribution_review_scan"], 12)
        self.assertNotIn(first.failed_key, saved)

        resumed = TimeoutClient(semantic=True)
        result = self.analyze(resumed, semantic=True)
        self.assertIn(first.failed_key, resumed.attempted)
        self.assertTrue(saved.keys().isdisjoint(resumed.attempted))
        self.assertEqual(set(resumed.attempted.values()), {"attribution_review_vote"})
        self.assertEqual({key: self.cached()[key] for key in saved}, saved)
        self.assert_source_preserved(result)
        audit = result["provenance"]["attribution_review"]
        self.assertEqual(audit["applied_blocks"], 12)
        for change in audit["changes"]:
            self.assertEqual(len(change["votes"]), 3)
            self.assertEqual({vote["verdict"] for vote in change["votes"]}, {"approve"})
            self.assertEqual(len({vote["reviewer_id"] for vote in change["votes"]}), 3)

        offline = TimeoutClient(semantic=True)
        self.assertEqual(self.analyze(offline, semantic=True), result)
        self.assertEqual(offline.attempted, {})

    def test_identical_continued_cli_command_resumes_without_partial_publication(self):
        config = self.root / "settings.toml"
        config.write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nchunk_chars = 256\nwhole_book_discovery = true\n'
            'identity_consensus = false\nattribution_review = false\n'
            'repair_attempts = 0\nattribution_workers = 5\n', encoding="utf-8")
        sources = []
        for number in (1, 2):
            path = self.root / f"source-{number}.json"
            path.write_text(json.dumps({"format": "epub-audiobook-source", "version": 1,
                                        "source": source_fixture(number)}), encoding="utf-8")
            sources.append(path)

        def command(client, arguments):
            stderr, stdout = io.StringIO(), io.StringIO()
            with patch("epub_audiobook.cli.OpenAICompatibleClient", return_value=client), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = main([str(value) for value in arguments])
            return status, stderr.getvalue()

        first_book, second_book = self.root / "book-one", self.root / "book-two"
        status, error = command(TimeoutClient(duplicate=False),
                               ["analyze", sources[0], "-o", first_book, "--config", config])
        self.assertEqual(status, 0, error)
        original_parent = {str(path.relative_to(first_book)): path.read_bytes()
                           for path in first_book.rglob("*") if path.is_file()}
        arguments = ["analyze", sources[1], "-o", second_book, "--continue-from", first_book,
                     "--config", config]
        durable = DurableReplies("attribute")
        first = TimeoutClient(fail_stage="attribute", durable=durable, duplicate=False)
        with patch("epub_audiobook.pipeline.write_json", side_effect=durable.write):
            status, error = command(first, arguments)
        self.assertNotEqual(status, 0, error)
        self.assertFalse(second_book.exists())
        saved = self.cached(self.root / ".epub-audiobook-cache")
        self.assertGreaterEqual(len(durable.keys), 4)

        resumed = TimeoutClient(duplicate=False)
        status, error = command(resumed, arguments)
        self.assertEqual(status, 0, error)
        self.assertIn(first.failed_key, resumed.attempted)
        self.assertTrue(saved.keys().isdisjoint(resumed.attempted))
        self.assertEqual(set(resumed.attempted.values()), {"attribute"})
        self.assertEqual(continuation_directories(second_book), [first_book.resolve()])
        self.assertTrue((second_book / "manifest.json").is_file())
        self.assertEqual(validate_document(json.loads((second_book / "analysis.json").read_text())), [])
        self.assertEqual({str(path.relative_to(first_book)): path.read_bytes()
                          for path in first_book.rglob("*") if path.is_file()}, original_parent)


if __name__ == "__main__":
    unittest.main()
