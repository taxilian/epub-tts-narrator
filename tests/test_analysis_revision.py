"""Attribution repairs update a continuation chain without touching voice assets."""
from copy import deepcopy
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_audiobook.analysis_directory import (_completed_directory, continuation_directories,
    continuation_registry, fresh_registry, publish_directory)
from epub_audiobook.analysis_revision import related_directories, revise_directories, _install
from epub_audiobook.cli import export_bundle, main
from epub_audiobook.config import DEFAULTS
from epub_audiobook.contracts import readiness
from epub_audiobook.pipeline import write_json
from epub_audiobook.provider import ProviderError
from epub_audiobook.series import link_analysis
from test_series import fixture, propose


class AnalysisRevisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.paths, self.documents = [], []
        for number in range(1, 4):
            path = self.root / f"book-{number}"
            document = fixture(sequel=number > 1)
            document["source"]["book"]["id"] = f"revision-book-{number}"
            # Speaker completion targets missing performers, not advisory notes.
            document["segments"][0].update(role_id=None, voice_id=None)
            document["segments"][0]["attribution"].update(
                basis="unresolved", review_required=True, reason="Uncertain speaker")
            if number == 1:
                registry = fresh_registry(path)
                document["issues"].append({"id": "attribution-test", "kind": "attribution",
                    "target": document["segments"][0]["block_id"], "reason": "Uncertain speaker", "resolved": False})
            else:
                registry = continuation_registry(self.paths[-1], document["source"])
                propose(document, self.documents[0]["cast"]["characters"][0]["id"])
            linked = link_analysis(document, registry)
            publish_directory(linked, registry, path, export_bundle,
                              continue_from=self.paths[-1] if self.paths else None)
            self.paths.append(path)
            self.documents.append(linked)
        for path in self.paths:
            (path / "voices").mkdir()
            (path / "voices" / "reference.wav").write_bytes(b"untouched voice")
            (path / "render-work").mkdir()
            (path / "render-work" / "accepted.json").write_text("untouched cache")
            (path / "voice-aliases.json").write_text("untouched aliases")

    def revision(self):
        document = deepcopy(self.documents[0])
        role = next(role for role in document["cast"]["roles"] if role["kind"] == "dialogue")
        document["segments"][0].update(role_id=role["id"], voice_id=role["voice_id"])
        document["segments"][0]["attribution"].update(
            basis="explicit", review_required=False, reason="The source identifies this speaker")
        document["issues"][0]["resolved"] = True
        document["provenance"]["speaker_completion"] = {"fixture": True}
        document["readiness"] = readiness(document)
        return document

    def managed(self):
        return {path: {name: (path / name).read_bytes() for name in
                      ["analysis.json", "series-cast.json", "manifest.json", "review.json"]}
                for path in self.paths}

    def test_parent_repair_refreshes_descendants_and_preserves_voices(self):
        before = self.managed()
        self.assertEqual(related_directories([self.paths[0]]), self.paths)
        result = revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        for index, path in enumerate(self.paths):
            document, registry, _ = _completed_directory(path)
            self.assertEqual(continuation_directories(path), list(reversed(self.paths[:index])))
            self.assertEqual(document["cast"], self.documents[index]["cast"])
            self.assertEqual(document["source"], self.documents[index]["source"])
            self.assertTrue(registry["books"][0]["issues"][0]["resolved"])
            self.assertEqual((path / "voices/reference.wav").read_bytes(), b"untouched voice")
            self.assertEqual((path / "render-work/accepted.json").read_text(), "untouched cache")
            self.assertEqual((path / "voice-aliases.json").read_text(), "untouched aliases")
            backup = Path(result["backup_directory"]) / f"book-{index + 1:04d}" / "before"
            self.assertEqual((backup / "analysis.json").read_bytes(), before[path]["analysis.json"])

    def test_changed_cast_or_source_is_rejected_before_publication(self):
        before = self.managed()
        revision = self.revision()
        revision["cast"]["voices"][0]["description"] = "changed"
        with self.assertRaisesRegex(ValueError, "preserve source, cast"):
            revise_directories({self.paths[0]: revision}, progress=lambda _: None)
        self.assertEqual(self.managed(), before)

    def test_concurrent_manifest_change_and_active_render_block_update(self):
        hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                  for path in self.paths}
        hashes[str(self.paths[0])] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed while speaker"):
            revise_directories({self.paths[0]: self.revision()}, expected_manifest_hashes=hashes)
        (self.paths[1] / "render-work/.render.lock").mkdir()
        with self.assertRaisesRegex(ValueError, "Rendering is active"):
            revise_directories({self.paths[0]: self.revision()})
        (self.paths[1] / "render-work/.render.lock").rmdir()
        # The renderer's work directory can live outside the book directory.
        (self.paths[1] / ".audio-render.lock").mkdir()
        with self.assertRaisesRegex(ValueError, "Rendering is active"):
            revise_directories({self.paths[0]: self.revision()})

    def test_partial_publication_failure_restores_all_originals(self):
        before = self.managed()
        calls = 0
        def fail_once(source, destination):
            nonlocal calls
            calls += 1
            if calls == 10:
                raise OSError("interrupted publication")
            return _install(source, destination)
        with patch("epub_audiobook.analysis_revision._install", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "interrupted publication"):
                revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        self.assertEqual(self.managed(), before)
        for path in self.paths:
            _completed_directory(path)
            continuation_directories(path)
            self.assertFalse((path / ".analysis-update.lock").exists())

    def test_interrupt_after_file_replace_also_restores_that_file(self):
        before = self.managed()
        calls = 0
        def replace_then_interrupt(source, destination):
            nonlocal calls
            calls += 1
            _install(source, destination)
            if calls == 1:
                raise KeyboardInterrupt()
        with patch("epub_audiobook.analysis_revision._install", side_effect=replace_then_interrupt):
            with self.assertRaises(KeyboardInterrupt):
                revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        self.assertEqual(self.managed(), before)
        for path in self.paths:
            _completed_directory(path)
            continuation_directories(path)

    def invoke(self, selected, complete):
        config = deepcopy(DEFAULTS)
        config["provider"].update(model="test-model", api_key="test-key")
        config["analysis"]["attribution_workers"] = 8
        output, errors = io.StringIO(), io.StringIO()
        with (patch("epub_audiobook.cli.load_config", return_value=config),
              patch("epub_audiobook.speaker_completion.complete_speakers", side_effect=complete) as completion,
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["resolve-speakers", *map(str, selected)])
        return code, completion, output.getvalue(), errors.getvalue()

    def test_cli_repairs_selected_book_and_refreshes_descendants_with_five_worker_cap(self):
        code, completion, output, _ = self.invoke([self.paths[0]], lambda document, requests, **kwargs: self.revision())
        self.assertEqual(code, 0)
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(completion.call_args.kwargs["workers"], 5)
        self.assertIn("0 unassigned spoken passage(s) remain", output)
        self.assertIn("Original analysis files and revision journal", output)
        for index, path in enumerate(self.paths):
            _completed_directory(path)
            self.assertEqual(continuation_directories(path), list(reversed(self.paths[:index])))
            self.assertEqual((path / "voices/reference.wav").read_bytes(), b"untouched voice")

    def test_cli_transport_failure_does_not_publish_part_of_the_chain(self):
        before = self.managed()
        calls = 0
        def complete(document, requests, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ProviderError("test transport failure")
            return self.revision()
        code, _, _, errors = self.invoke(self.paths[:2], complete)
        self.assertEqual(code, 1)
        self.assertIn("Completed, validated responses remain cached", errors)
        self.assertEqual(self.managed(), before)
        self.assertFalse((self.paths[0] / ".analysis-revisions").exists())

    def test_cli_retains_disagreement_and_returns_blocked_status(self):
        def complete(document, requests, **kwargs):
            result = self.revision()
            segment = result["segments"][0]
            segment["role_id"] = segment["voice_id"] = None
            segment["attribution"].update(basis="unresolved", review_required=True,
                                           reason="The independent reviewers disagree.")
            result["readiness"] = readiness(result)
            return result
        code, _, output, _ = self.invoke([self.paths[0]], complete)
        self.assertEqual(code, 2)
        self.assertIn("1 unassigned spoken passage(s) remain", output)
        document, _, _ = _completed_directory(self.paths[0])
        self.assertIsNone(document["segments"][0]["voice_id"])
        self.assertFalse(readiness(document)["ready"])


if __name__ == "__main__":
    unittest.main()
