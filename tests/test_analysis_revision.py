"""Attribution repairs update a continuation chain without touching voice assets."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (_completed_directory, continuation_directories,
    continuation_registry, fresh_registry, publish_directory)
from epub_tts_narrator.analysis_revision import related_directories, revise_directories, _install
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.provider import ProviderError
from epub_tts_narrator.series import link_analysis
from shared_fixtures import Terminal
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

    def test_concurrent_manifest_change_still_blocks_the_revision(self):
        hashes = {str(path): hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                  for path in self.paths}
        hashes[str(self.paths[0])] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed while speaker"):
            revise_directories({self.paths[0]: self.revision()}, expected_manifest_hashes=hashes)

    def test_placeholder_render_lock_directories_do_not_block_an_attribution_revision(self):
        # Legacy/placeholder lock directories once refused saves; attribution
        # publication never probes render activity.
        (self.paths[1] / "render-work/.render.lock").mkdir()
        (self.paths[1] / ".audio-render.lock").mkdir()
        try:
            revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        finally:
            (self.paths[1] / "render-work/.render.lock").rmdir()
            (self.paths[1] / ".audio-render.lock").rmdir()
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "explicit")

    def test_abandoned_analysis_update_lock_is_reclaimed_after_a_crash(self):
        lock = self.paths[0] / ".analysis-update.lock"
        crashed = subprocess.run([
            sys.executable, "-c",
            "import os, sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock; "
            "acquire_directory_lock(sys.argv[1]); os._exit(130)",
            str(lock),
        ])
        self.assertEqual(crashed.returncode, 130)
        self.assertTrue((lock / "owner.json").is_file())
        revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        self.assertFalse(lock.exists())

    def crashed_render_lock(self, path):
        lock = path / ".audio-render.lock"
        crashed = subprocess.run([
            sys.executable, "-c",
            "import os, sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock; "
            "acquire_directory_lock(sys.argv[1]); os._exit(130)",
            str(lock),
        ])
        self.assertEqual(crashed.returncode, 130)
        self.assertTrue((lock / "owner.json").is_file())
        return lock

    def test_abandoned_marked_render_lock_publishes_without_deleting_ownership_records(self):
        lock = self.crashed_render_lock(self.paths[1])
        lease = lock.with_name(lock.name + ".lease")
        inode = lease.stat().st_ino
        owner = (lock / "owner.json").read_bytes()
        revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "explicit")
        self.assertTrue(lock.is_dir())
        self.assertEqual((lock / "owner.json").read_bytes(), owner)
        self.assertEqual(lease.stat().st_ino, inode)

    def live_render_lock(self, book):
        """Hold the book's audio lock from a live external process until released."""
        lock = book / ".audio-render.lock"
        held, release = self.root / f"held-{lock.parent.name}-{book.name}", self.root / "release"
        release.unlink(missing_ok=True)
        child = subprocess.Popen([
            sys.executable, "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from epub_tts_narrator.renderer.run_locks import acquire_directory_lock\n"
            "lock = acquire_directory_lock(sys.argv[1])\n"
            "Path(sys.argv[2]).write_text('held')\n"
            "while not Path(sys.argv[3]).exists():\n"
            "    time.sleep(0.01)\n",
            str(lock), str(held), str(release)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return lock, held, release, child

    def test_live_marked_render_lock_publishes_attribution_and_keeps_ownership_artifacts(self):
        for book in (self.paths[0], self.paths[1]):
            # Same-book and descendant renders alike must never refuse a review
            # save; the active renderer keeps finishing its loaded snapshot.
            lock, held, release, child = self.live_render_lock(book)
            try:
                deadline = time.monotonic() + 15
                while not held.exists():
                    self.assertIsNone(child.poll())
                    self.assertGreater(deadline, time.monotonic(), "lock holder never signaled")
                owner = (lock / "owner.json").read_bytes()
                lease = lock.with_name(lock.name + ".lease")
                inode = lease.stat().st_ino
                revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
                revised, _, _ = _completed_directory(self.paths[0])
                self.assertEqual(revised["segments"][0]["attribution"]["basis"], "explicit")
                for path in self.paths:
                    continuation_directories(path)
                self.assertTrue(lock.is_dir())
                self.assertEqual((lock / "owner.json").read_bytes(), owner)
                self.assertEqual(lease.stat().st_ino, inode)
            finally:
                release.write_text("go")
                self.assertEqual(child.wait(timeout=15), 0)
                held.unlink(missing_ok=True)
        revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "explicit")

    def test_reserved_scope_holds_publication_ownership_for_the_whole_chain(self):
        from epub_tts_narrator.analysis_revision import reserve_revision_scope
        with reserve_revision_scope([self.paths[0]]) as reserved:
            self.assertEqual(sorted(reserved), sorted(self.paths))
            for path in self.paths:
                self.assertTrue((path / ".analysis-update.lock").is_dir())
            with self.assertRaisesRegex(ValueError, "already active"):
                revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None)
            revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                               reserved_update_locks=reserved)
        for path in self.paths:
            self.assertFalse((path / ".analysis-update.lock").exists())
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "explicit")

    def test_reserved_scope_reserves_and_publishes_while_a_renderer_is_live(self):
        from epub_tts_narrator.analysis_revision import reserve_revision_scope
        lock, held, release, child = self.live_render_lock(self.paths[0])
        try:
            deadline = time.monotonic() + 15
            while not held.exists():
                self.assertIsNone(child.poll())
                self.assertGreater(deadline, time.monotonic(), "lock holder never signaled")
            owner = (lock / "owner.json").read_bytes()
            lease = lock.with_name(lock.name + ".lease")
            inode = lease.stat().st_ino
            with reserve_revision_scope([self.paths[0]]) as reserved:
                self.assertEqual(sorted(reserved), sorted(self.paths))
                for path in self.paths:
                    self.assertTrue((path / ".analysis-update.lock").is_dir())
                revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                                   reserved_update_locks=reserved)
            revised, _, _ = _completed_directory(self.paths[0])
            self.assertEqual(revised["segments"][0]["attribution"]["basis"], "explicit")
            for path in self.paths:
                self.assertFalse((path / ".analysis-update.lock").exists())
            self.assertTrue(lock.is_dir())
            self.assertEqual((lock / "owner.json").read_bytes(), owner)
            self.assertEqual(lease.stat().st_ino, inode)
        finally:
            release.write_text("go")
            self.assertEqual(child.wait(timeout=15), 0)

    def test_unowned_reservation_mapping_is_rejected_without_publication(self):
        from epub_tts_narrator.analysis_revision import reserve_revision_scope
        before = self.managed()
        for forged in ({path: object() for path in self.paths},
                       {path: None for path in self.paths}):
            with self.subTest(value=forged[self.paths[0]]):
                with self.assertRaisesRegex(ValueError, "not currently owned"):
                    revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                                       reserved_update_locks=forged)
                self.assertEqual(self.managed(), before)
        # A handle for the wrong lock path is equally unowned, even when real.
        with reserve_revision_scope([self.paths[0]]) as reserved:
            mismatched = {path: (reserved[path] if path == self.paths[0] else reserved[self.paths[0]])
                          for path in self.paths}
            with self.assertRaisesRegex(ValueError, "not currently owned"):
                revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                                   reserved_update_locks=mismatched)
            self.assertEqual(self.managed(), before)

    def test_reservation_mapping_cannot_be_reused_after_release(self):
        from epub_tts_narrator.analysis_revision import reserve_revision_scope
        with reserve_revision_scope([self.paths[0]]) as reserved:
            pass
        for path in self.paths:
            self.assertFalse((path / ".analysis-update.lock").exists())
        with self.assertRaisesRegex(ValueError, "not currently owned"):
            revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                               reserved_update_locks=reserved)

    def test_reservation_with_altered_ownership_is_rejected(self):
        from epub_tts_narrator.analysis_revision import reserve_revision_scope
        with reserve_revision_scope([self.paths[0]]) as reserved:
            marker = self.paths[0] / ".analysis-update.lock" / "owner.json"
            lease = self.paths[0] / ".analysis-update.lock.lease"
            original_marker = marker.read_bytes()
            original_lease = lease.read_bytes()
            with self.subTest(tamper="owner record"):
                record = json.loads(marker.read_text(encoding="utf-8"))
                record["token"] = "0" * 32
                marker.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "not currently owned"):
                    revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                                       reserved_update_locks=reserved)
                marker.write_bytes(original_marker)
            # The restored record owns the lock again, so publication proceeds.
            revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                               reserved_update_locks=reserved)
            with self.subTest(tamper="replaced lease inode"):
                lease.unlink()
                lease.write_bytes(original_lease)
                with self.assertRaisesRegex(ValueError, "not currently owned"):
                    revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                                       reserved_update_locks=reserved)

    def test_reservation_destroyed_during_staging_cannot_write(self):
        from epub_tts_narrator.analysis_revision import reserve_revision_scope
        before = self.managed()

        def tamper_after_export(document, after, **kwargs):
            result = export_bundle(document, after, **kwargs)
            marker = self.paths[0] / ".analysis-update.lock" / "owner.json"
            marker.write_text(json.dumps({**json.loads(marker.read_text(encoding="utf-8")),
                                          "token": "1" * 32}))
            return result

        with reserve_revision_scope([self.paths[0]]) as reserved:
            marker = self.paths[0] / ".analysis-update.lock" / "owner.json"
            original_marker = marker.read_bytes()
            with patch("epub_tts_narrator.cli.export_bundle", side_effect=tamper_after_export):
                with self.assertRaisesRegex(ValueError, "not currently owned"):
                    revise_directories({self.paths[0]: self.revision()}, progress=lambda _: None,
                                       reserved_update_locks=reserved)
            self.assertEqual(self.managed(), before)
            marker.write_bytes(original_marker)
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "unresolved")

    def test_partial_publication_failure_restores_all_originals(self):
        before = self.managed()
        calls = 0
        def fail_once(source, destination):
            nonlocal calls
            calls += 1
            if calls == 10:
                raise OSError("interrupted publication")
            return _install(source, destination)
        with patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_once):
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
        with patch("epub_tts_narrator.analysis_revision._install", side_effect=replace_then_interrupt):
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
        # Without this, a run inside a real terminal would wait at /dev/tty for a person.
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.speaker_completion.complete_speakers", side_effect=complete) as completion,
              patch("epub_tts_narrator.speaker_decisions.open", side_effect=OSError("No controlling terminal")),
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

    def test_cli_saves_explicit_new_voice_in_published_parent_without_changing_descendant_casts(self):
        before = deepcopy(self.documents[0]["cast"])
        answers = Terminal("N\nLater Revealed Speaker\nA low, lightly textured adult voice with precise conversational articulation.\n")
        with patch("sys.stdin", answers):
            code, _, _, errors = self.invoke([self.paths[0]], lambda document, requests, **kwargs: document)
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.paths[0])
        character = next(row for row in revised["cast"]["characters"] if row["name"] == "Later Revealed Speaker")
        self.assertEqual(revised["segments"][0]["voice_id"], character["voice_id"])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "user")
        for collection in ("characters", "voices", "roles"):
            self.assertEqual(revised["cast"][collection][:len(before[collection])], before[collection])
            self.assertEqual(len(revised["cast"][collection]), len(before[collection]) + 1)
        for index, path in enumerate(self.paths):
            document, registry, _ = _completed_directory(path)
            self.assertEqual(registry["books"][0]["cast"], revised["cast"])
            self.assertEqual(continuation_directories(path), list(reversed(self.paths[:index])))
            if index:
                self.assertEqual(document["cast"], self.documents[index]["cast"])
            self.assertEqual((path / "voices/reference.wav").read_bytes(), b"untouched voice")
            self.assertEqual((path / "render-work/accepted.json").read_text(), "untouched cache")
            self.assertEqual((path / "voice-aliases.json").read_text(), "untouched aliases")

    def test_analyze_existing_directory_saves_new_voice_without_reanalyzing_source(self):
        config = deepcopy(DEFAULTS)
        config["provider"].update(model="test-model", api_key="test-key")
        answers = Terminal("N\nLater Revealed Speaker\nA low, lightly textured adult voice with precise conversational articulation.\n")
        output, errors = io.StringIO(), io.StringIO()
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.speaker_completion.complete_speakers", side_effect=lambda document, requests, **kwargs: document),
              patch("epub_tts_narrator.cli.analyze") as fresh_analysis,
              patch("sys.stdin", answers),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["analyze", str(self.paths[0])])
        self.assertEqual(code, 0, errors.getvalue())
        fresh_analysis.assert_not_called()
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(revised["source"], self.documents[0]["source"])
        self.assertEqual(revised["segments"][0]["attribution"]["basis"], "user")
        self.assertEqual(revised["cast"]["characters"][-1]["name"], "Later Revealed Speaker")

    def test_failed_new_voice_publication_replays_the_human_journal_without_defining_twice(self):
        before = self.managed()
        calls = 0

        def fail_once(source, destination):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("Fixture interrupted manual publication")
            return _install(source, destination)

        with (patch("sys.stdin", Terminal("N\nLater Revealed Speaker\nA low, lightly textured adult voice.\n")),
              patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_once)):
            code, _, _, _ = self.invoke([self.paths[0]], lambda document, requests, **kwargs: document)
        self.assertEqual(code, 1)
        self.assertEqual(self.managed(), before)
        with patch("sys.stdin", Terminal("")):
            code, _, output, errors = self.invoke([self.paths[0]], lambda document, requests, **kwargs: document)
        self.assertEqual(code, 0, errors)
        self.assertIn("Replaying 1 saved speaker decision", errors)
        revised, _, _ = _completed_directory(self.paths[0])
        self.assertEqual(sum(row["name"] == "Later Revealed Speaker" for row in revised["cast"]["characters"]), 1)

    def test_new_voice_is_saved_when_the_user_quits_with_another_passage_pending(self):
        path = self.root / "partial-manual"
        document = fixture()
        for row in document["segments"][:2]:
            row.update(role_id=None, voice_id=None)
            row["attribution"].update(basis="unresolved", review_required=True, candidates=[])
        registry = fresh_registry(path)
        document = link_analysis(document, registry)
        publish_directory(document, registry, path, export_bundle)
        with patch("sys.stdin", Terminal("N\nLater Revealed Speaker\nA low, lightly textured adult voice.\nQ\n")):
            code, _, _, errors = self.invoke([path], lambda document, requests, **kwargs: document)
        self.assertEqual(code, 2, errors)
        revised, _, _ = _completed_directory(path)
        self.assertEqual(revised["cast"]["characters"][-1]["name"], "Later Revealed Speaker")
        self.assertIsNotNone(revised["segments"][0]["voice_id"])
        self.assertIsNone(revised["segments"][1]["voice_id"])
        with patch("sys.stdin", Terminal("O\n3\n")):
            code, _, _, errors = self.invoke([path], lambda document, requests, **kwargs: document)
        self.assertEqual(code, 0, errors)
        resumed, _, _ = _completed_directory(path)
        self.assertEqual(resumed["cast"], revised["cast"])
        self.assertEqual(resumed["segments"][1]["voice_id"], revised["segments"][0]["voice_id"])

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
