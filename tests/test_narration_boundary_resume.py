"""Narrative speech tags are repaired in new analyses and saved book reruns."""

import contextlib
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (
    _completed_directory, continuation_directories, continuation_registry,
    fresh_registry, publish_directory,
)
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.contracts import readiness, validate_document
from epub_tts_narrator.pipeline import analyze, fingerprint, write_json
from epub_tts_narrator.series import link_analysis
from shared_fixtures import snapshot
from test_contracts import valid_document
from test_pipeline import FakeClient


TEXT = '"Hello," Mara said, "Again."'


def incorrectly_grouped_document(number=1):
    """A valid older bundle can still have an incorrect narrative boundary."""
    document = valid_document()
    source = document["source"]
    source["book"]["id"] = f"narration-boundary-{number}"
    source["book"]["title"] = f"Narration boundary {number}"
    before = deepcopy(source["blocks"][1])
    before.update(id="b0", text="I waited for her answer.")
    source["blocks"].insert(0, before)
    source["blocks"][1]["text"] = TEXT
    source["chapters"][0]["block_ids"] = ["b0", "b1", "b2"]
    document["cast"]["voices"][1]["traits"] = []
    narration = deepcopy(document["segments"][2])
    narration.update(id="s0", block_id="b0", start=0, end=len(before["text"]), text=before["text"])
    dialogue = deepcopy(document["segments"][1])
    dialogue.update(id="s1", start=0, end=len(TEXT), text=TEXT)
    document["segments"] = [narration, dialogue, document["segments"][2]]
    document["provenance"]["source_fingerprint"] = fingerprint(source)
    document["readiness"] = readiness(document)
    return document


class WholeParagraphClient(FakeClient):
    """The model incorrectly gives the speech tag its character's voice."""

    def attribute(self, payload):
        roles = {role["kind"]: role for role in payload["final_cast"]["roles"]}
        units = []
        for unit in payload["target_units"]:
            kind = "dialogue" if unit["text"] == TEXT else "narration"
            units.append({"unit_id": unit["unit_id"], "segments": [{
                "text": unit["text"], "kind": kind, "role_id": roles[kind]["id"],
                "basis": "explicit", "reason": "Fixture model grouped the entire paragraph.",
                "evidence": [{"block_id": unit["block_id"], "quote": unit["text"]}],
                "candidates": [], "review_required": False,
                "delivery": {"direction": "", "evidence": []},
            }]})
        return {"units": units}


class NarrationBoundaryResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.config = self.root / "global.toml"
        self.config.write_text('[provider]\nmodel="fixture"\nbase_url="http://offline.invalid/v1"\n'
                               '[analysis]\nattribution_workers=1\nrepair_attempts=0\n')

    def seed(self, number=1, *, parent=None):
        document = incorrectly_grouped_document(number)
        path = self.root / f"book-{number:02d}"
        registry = continuation_registry(parent, document["source"]) if parent else fresh_registry(path)
        document = link_analysis(document, registry)
        publish_directory(document, registry, path, export_bundle, continue_from=parent)
        (path / "voices").mkdir()
        (path / "voices/reference.wav").write_bytes(b"Keep this existing cloning reference")
        (path / "render-work").mkdir()
        (path / "render-work/accepted.wav").write_bytes(b"Keep verified render audio for cache reuse")
        source_path = self.root / f"book-{number:02d}-source.json"
        write_json(source_path, {"format": "epub-audiobook-source", "version": 1,
                                 "source": document["source"]})
        return path, source_path, document

    def run_cli(self, *args, command="analyze"):
        output, error = StringIO(), StringIO()
        client = FakeClient()
        client.model, client.base_url = "fixture", "http://offline.invalid/v1"
        with (patch("sys.stdin", StringIO()), contextlib.redirect_stdout(output),
              contextlib.redirect_stderr(error),
              patch("epub_tts_narrator.cli.analyze", side_effect=AssertionError("Must reuse saved analysis")),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client) as provider,
              patch("epub_tts_narrator.cli.JevClient") as judge):
            status = main([command, *map(str, args), "--config", str(self.config)])
        # Old bundles need the new spelling pass, but known speaker/cast work
        # must not be repeated. A completed second run makes no model calls.
        self.assertLessEqual(provider.call_count, 1)
        self.assertTrue(all(payload.get("mode") in {"speech_spelling_scan", "pronunciation_discovery", "pronunciation_proposal"}
                            for _, payload in client.calls))
        if command == "resolve-speakers":
            provider.assert_not_called()
        judge.assert_not_called()
        return status, output.getvalue(), error.getvalue()

    def assert_repaired(self, document):
        self.assertEqual(validate_document(document), [])
        rows = [row for row in document["segments"] if row["block_id"] == "b1"]
        roles = {role["kind"]: role for role in document["cast"]["roles"]}
        self.assertEqual(len(rows), 3)
        self.assertEqual("".join(row["text"] for row in rows), TEXT)
        self.assertEqual(rows[0]["start"], 0)
        self.assertEqual(rows[-1]["end"], len(TEXT))
        for left, right in zip(rows, rows[1:]):
            self.assertEqual(left["end"], right["start"])
        self.assertEqual(rows[1]["text"].strip(), "Mara said,")
        self.assertEqual([row["kind"] for row in rows], ["dialogue", "narration", "dialogue"])
        self.assertEqual([row["role_id"] for row in rows],
                         [roles[kind]["id"] for kind in ("dialogue", "narration", "dialogue")])
        self.assertEqual([row["voice_id"] for row in rows],
                         [roles[kind]["voice_id"] for kind in ("dialogue", "narration", "dialogue")])
        self.assertTrue(document["provenance"]["narration_boundaries"]["repairs"])

    def test_new_analysis_repairs_middle_speech_tag_even_when_every_voice_is_assigned(self):
        source = incorrectly_grouped_document()["source"]
        before = deepcopy(source)
        client = WholeParagraphClient()
        document = analyze(source, client, self.root / "cache", identity_consensus=False,
                           attribution_review=False, attribution_workers=1)
        self.assert_repaired(document)
        self.assertEqual(source, before)
        self.assertEqual(document["source"], before)
        self.assertFalse(any(payload.get("mode") == "speaker_completion" for _, payload in client.calls))

    def test_final_analysis_boundary_check_catches_a_model_stage_regrouping_the_tag(self):
        def regroup(document, *args, **kwargs):
            result = deepcopy(document)
            role = next(role for role in result["cast"]["roles"] if role["kind"] == "dialogue")
            rows = [row for row in result["segments"] if row["block_id"] == "b1"]
            grouped = deepcopy(rows[0])
            grouped.update(start=0, end=len(TEXT), text=TEXT, kind="dialogue",
                           role_id=role["id"], voice_id=role["voice_id"])
            result["segments"] = ([row for row in result["segments"] if row["block_id"] == "b0"]
                                  + [grouped]
                                  + [row for row in result["segments"] if row["block_id"] == "b2"])
            return result

        with patch("epub_tts_narrator.speaker_completion.complete_speakers", side_effect=regroup):
            document = analyze(incorrectly_grouped_document()["source"], WholeParagraphClient(),
                               self.root / "cache", identity_consensus=False,
                               attribution_review=False, attribution_workers=1)
        self.assert_repaired(document)

    def test_rerun_repairs_fully_assigned_saved_book_and_preserves_artifacts(self):
        for number, use_source in enumerate((False, True), 1):
            with self.subTest(source_input=use_source):
                path, source, original = self.seed(number)
                self.assertTrue(original["readiness"]["ready"])
                before = snapshot(path)
                args = [source, "-o", path] if use_source else [path]
                status, _, error = self.run_cli(*args)
                self.assertEqual(status, 0, error)
                updated = _completed_directory(path)[0]
                self.assert_repaired(updated)
                self.assertEqual(updated["source"], original["source"])
                self.assertEqual(updated["cast"], original["cast"])
                self.assertEqual(updated["provenance"]["series"], original["provenance"]["series"])
                for filename in ("voices/reference.wav", "render-work/accepted.wav"):
                    self.assertEqual((path / filename).read_bytes(), before[filename])
                journals = list((path / ".analysis-revisions").glob("*/revision.json"))
                self.assertEqual(len(journals), 1)
                journal = json.loads(journals[0].read_text())
                self.assertEqual(journal["status"], "complete")
                backup = Path(journal["books"][0]["backup"])
                self.assertEqual((backup / "analysis.json").read_bytes(), before["analysis.json"])
                repaired_snapshot = snapshot(path)
                status, _, error = self.run_cli(*args)
                self.assertEqual(status, 0, error)
                self.assertEqual(snapshot(path), repaired_snapshot)

    def test_repair_keeps_descendant_continuation_valid_without_recasting(self):
        parent, _, original = self.seed()
        child, _, original_child = self.seed(2, parent=parent)
        child_assets = {name: (child / name).read_bytes()
                        for name in ("voices/reference.wav", "render-work/accepted.wav")}
        status, _, error = self.run_cli(parent)
        self.assertEqual(status, 0, error)
        repaired = _completed_directory(parent)[0]
        self.assert_repaired(repaired)
        descendant = _completed_directory(child)[0]
        self.assertEqual(continuation_directories(child), [parent])
        self.assertEqual(descendant["segments"], original_child["segments"])
        self.assertEqual(descendant["cast"], original_child["cast"])
        self.assertEqual(repaired["cast"], original["cast"])
        for name, content in child_assets.items():
            self.assertEqual((child / name).read_bytes(), content)

    def test_resolve_speakers_also_repairs_assigned_narrative_tags_without_model_calls(self):
        path, _, original = self.seed()
        status, _, error = self.run_cli(path, command="resolve-speakers")
        self.assertEqual(status, 0, error)
        updated = _completed_directory(path)[0]
        self.assert_repaired(updated)
        self.assertEqual(updated["source"], original["source"])
        self.assertEqual(updated["cast"], original["cast"])


if __name__ == "__main__":
    unittest.main()
