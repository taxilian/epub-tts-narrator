"""CLI regressions for fresh attribution audits and post-audit decisions."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import (
    _completed_directory, continuation_registry, fresh_registry, publish_directory,
)
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.pipeline import write_json
from epub_tts_narrator.series import link_analysis
from test_attribution_verification import AuditClient, risk_document


class TerminalIO(io.StringIO):
    def isatty(self):
        return True


class NoFindingAuditClient(AuditClient):
    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        if payload["mode"] == "attribution_review_scan":
            return {"corrections": []}
        raise AssertionError(f"Unexpected request mode: {payload['mode']}")


class AttributionCLIHarness(unittest.TestCase):
    root: Path
    book: Path

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.book = self.root / "book"

    def make_book(self, risk=True, prepare=None):
        document = risk_document(risk)
        if prepare is not None:
            prepare(document)
        registry = fresh_registry(self.book)
        linked = link_analysis(document, registry)
        publish_directory(linked, registry, self.book, export_bundle)
        return linked

    def make_invalid_descendant(self):
        document = risk_document()
        document["source"]["book"].update(id="book-two", title="The Return")
        descendant = self.root / f"{self.book.name}-descendant"
        registry = continuation_registry(self.book, document["source"])
        linked = link_analysis(document, registry)
        publish_directory(linked, registry, descendant, export_bundle, continue_from=self.book)

        analysis_path = descendant / "analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        source = json.loads((descendant / "source.json").read_text(encoding="utf-8"))
        segment = analysis["segments"][2]
        start, end = segment["start"], segment["end"]
        replacement = segment["text"] + " Drift."
        for record in (analysis["source"], source):
            block = next(row for row in record["blocks"] if row["id"] == segment["block_id"])
            block["text"] = block["text"][:start] + replacement + block["text"][end:]
        segment.update(text=replacement, end=start + len(replacement))
        write_json(analysis_path, analysis, overwrite=True)
        write_json(descendant / "source.json", source, overwrite=True)

        manifest_path = descendant / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        script_name = manifest["files"]["scripts"][0]["path"]
        script_path = descendant / script_name
        rows = [json.loads(line) for line in script_path.read_text(encoding="utf-8").splitlines()]
        row = next(row for row in rows if row["id"] == segment["id"])
        row["text"] = replacement
        if "end" in row:
            row["end"] = segment["end"]
        script_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for name in ("source.json", script_name):
            manifest["sha256"][name] = hashlib.sha256((descendant / name).read_bytes()).hexdigest()
        write_json(manifest_path, manifest, overwrite=True)
        with self.assertRaisesRegex(ValueError, "checksum mismatch: analysis.json"):
            _completed_directory(descendant)
        return descendant

    def published_snapshot(self, *directories):
        return {path: path.read_bytes() for directory in directories for path in directory.rglob("*")
                if path.is_file() and ".analysis-cache" not in path.relative_to(directory).parts}

    def make_descendant(self):
        document = risk_document()
        document["source"]["book"].update(id="book-two", title="The Return")
        descendant = self.root / f"{self.book.name}-descendant"
        registry = continuation_registry(self.book, document["source"])
        linked = link_analysis(document, registry)
        publish_directory(linked, registry, descendant, export_bundle, continue_from=self.book)
        return linked, descendant

    def hold_render_lock(self, directory):
        """Run a live renderer stand-in that owns the book's audio lock until released."""
        held, release = self.root / "render-held", self.root / "render-release"
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import sys, time\n"
             "from pathlib import Path\n"
             "from epub_tts_narrator.renderer.run_locks import acquire_directory_lock\n"
             "lock = acquire_directory_lock(sys.argv[1])\n"
             "Path(sys.argv[2]).write_text('held')\n"
             "while not Path(sys.argv[3]).exists():\n"
             "    time.sleep(0.01)\n",
             str(directory / ".audio-render.lock"), str(held), str(release)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        while not held.exists():
            self.assertIsNone(child.poll())
            self.assertGreater(deadline, time.monotonic(), "renderer stand-in never signaled")
        return child, release

    def invoke(self, *arguments, input_text="", tty=False, client=None):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        client = client or AuditClient()
        output = TerminalIO() if tty else io.StringIO()
        errors = io.StringIO()
        stdin = TerminalIO(input_text) if tty else io.StringIO(input_text)
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client),
              patch("sys.stdin", stdin),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["verify-attribution", str(self.book), *map(str, arguments)])
        return code, output.getvalue(), errors.getvalue(), client


class AttributionVerificationCLITests(AttributionCLIHarness):
    def test_report_mode_prints_findings_without_revising_the_book(self):
        self.make_book()
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, _ = self.invoke(input_text="a\n")
        self.assertEqual(code, 0, errors)
        self.assertIn("from Mara to Tom", output)
        self.assertIn("2 attribution finding(s).", output)
        self.assertIn("instructions=no", output)
        self.assertIn("No changes applied; 2 finding(s) left unchanged.", output)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())

    def test_report_only_allows_an_invalid_descendant_without_changing_published_files(self):
        self.book = self.root / "report"
        self.make_book()
        descendant = self.make_invalid_descendant()
        before = self.published_snapshot(self.book, descendant)

        code, output, errors, client = self.invoke()

        self.assertEqual(code, 0, errors)
        self.assertTrue(client.calls)
        self.assertIn("Report-only invocation: no publication ownership is taken", errors)
        self.assertIn("2 attribution finding(s).", output)
        self.assertIn("No changes applied; 2 finding(s) left unchanged.", output)
        self.assertEqual(self.published_snapshot(self.book, descendant), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())

    def test_write_capable_modes_refuse_an_invalid_descendant_before_model_requests(self):
        self.make_book()
        descendant = self.make_invalid_descendant()
        before = self.published_snapshot(self.book, descendant)
        for arguments, input_text, tty in [(("--apply",), "", False),
                                           (("--interactive",), "y\nn\n", True),
                                           ((), "l\n", True)]:
            with self.subTest(arguments=arguments or "tty"):
                client = AuditClient()
                code, _, errors, client = self.invoke(*arguments, input_text=input_text, tty=tty, client=client)
                self.assertEqual(code, 1)
                self.assertEqual(client.calls, [])
                self.assertIn("checksum mismatch: analysis.json", errors)
                self.assertIn(str(descendant), errors)
                self.assertEqual(self.published_snapshot(self.book, descendant), before)
                self.assertFalse((self.book / ".analysis-revisions").exists())
                self.assertFalse((self.book / ".analysis-update.lock").exists())

    def test_apply_rejects_a_target_manifest_changed_during_the_audit(self):
        self.make_book()
        analysis_before = (self.book / "analysis.json").read_bytes()
        client = AuditClient()
        original_complete = client.complete
        changed_manifest = []

        def complete(system, payload):
            response = original_complete(system, payload)
            if payload["mode"] == "attribution_review_scan":
                with client.lock:
                    if not changed_manifest:
                        path = self.book / "manifest.json"
                        path.write_bytes(path.read_bytes() + b" \n")
                        changed_manifest.append(path.read_bytes())
            return response

        client.complete = complete
        code, _, errors, _ = self.invoke("--apply", client=client)

        self.assertEqual(code, 1)
        self.assertTrue(changed_manifest)
        self.assertIn("Attribution publication did not succeed", errors)
        self.assertIn(f"Analysis changed while attribution was being reviewed: {self.book}", errors)
        self.assertEqual((self.book / "manifest.json").read_bytes(), changed_manifest[0])
        self.assertEqual((self.book / "analysis.json").read_bytes(), analysis_before)
        self.assertFalse((self.book / ".analysis-revisions").exists())

    def test_apply_mode_publishes_the_whole_audit_with_a_revision_journal(self):
        original = self.make_book()
        code, output, errors, _ = self.invoke("--apply")
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        roles = {row["id"]: row["role_id"] for row in revised["segments"]}
        self.assertEqual(roles[original["segments"][1]["id"]], "tom")
        self.assertEqual(roles[original["segments"][2]["id"]], "tom")
        self.assertIn("attribution_review", revised["provenance"])
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))
        self.assertIn("Original analysis files and revision journal:", output)
        self.assertIn("2 applied / 0 left unchanged.", output)

    def test_interactive_wins_over_apply_and_publishes_only_accepted_findings(self):
        original = self.make_book()
        code, output, errors, _ = self.invoke("--apply", "--interactive", input_text="y\nn\n")
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        roles = {row["id"]: row["role_id"] for row in revised["segments"]}
        self.assertEqual(roles[original["segments"][1]["id"]], "tom")
        self.assertEqual(roles[original["segments"][2]["id"]], "mara")
        self.assertNotIn("attribution_review", revised["provenance"])
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))
        self.assertIn("1 applied / 1 left unchanged.", output)

    def test_chapter_scoped_instructions_reach_the_scan_as_review_notes(self):
        self.make_book()
        code, output, errors, client = self.invoke("--chapter", 1, "--instructions", "Check the visitor carefully.")
        self.assertEqual(code, 0, errors)
        scans = [row for row in client.calls if row["mode"] == "attribution_review_scan"]
        self.assertTrue(scans)
        self.assertTrue(all(row["review_notes"] == "Check the visitor carefully." for row in scans))
        self.assertIn("instructions=yes", output)

    def test_sample_mode_audits_the_deterministic_risk_subset(self):
        self.make_book()
        code, _, errors, client = self.invoke("--sample", 1)
        self.assertEqual(code, 0, errors)
        scans = [row for row in client.calls if row["mode"] == "attribution_review_scan"]
        self.assertEqual(len(scans), 1)
        self.assertEqual([row["block_id"] for row in scans[0]["target_units"]], ["b2"])

    def test_empty_risk_set_exits_before_building_model_requests(self):
        self.make_book(risk=False)
        code, output, errors, client = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertIn("No context-basis or flagged dialogue lines to verify.", output)
        self.assertEqual(client.calls, [])
        self.assertFalse((self.book / ".analysis-cache").exists())

    def test_invalid_instruction_lengths_and_worker_counts_return_errors_without_model_calls(self):
        self.make_book()
        for arguments, message in [(("--instructions", ""), "--instructions"),
                                   (("--instructions", "x" * 2001), "--instructions"),
                                   (("--workers", 0), "workers"), (("--workers", 6), "workers")]:
            with self.subTest(arguments=arguments):
                code, _, errors, client = self.invoke(*arguments)
                self.assertEqual(code, 1)
                self.assertIn(message, errors)
                self.assertEqual(client.calls, [])

    def test_tty_accept_all_publishes_every_finding_and_reports_saved_counts(self):
        original = self.make_book()
        code, output, errors, _ = self.invoke(input_text="a\n", tty=True)
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        roles = {row["id"]: row["role_id"] for row in revised["segments"]}
        self.assertEqual(roles[original["segments"][1]["id"]], "tom")
        self.assertEqual(roles[original["segments"][2]["id"]], "tom")
        self.assertIn("2 applied / 0 left unchanged.", output)
        self.assertIn("Original analysis files and revision journal:", output)

    def test_tty_edit_review_publishes_only_the_selected_finding(self):
        original = self.make_book()
        code, output, errors, _ = self.invoke(input_text="e\ny\nn\n", tty=True)
        self.assertEqual(code, 0, errors)
        revised, _, _ = _completed_directory(self.book)
        roles = {row["id"]: row["role_id"] for row in revised["segments"]}
        self.assertEqual(roles[original["segments"][1]["id"]], "tom")
        self.assertEqual(roles[original["segments"][2]["id"]], "mara")
        self.assertIn("1 applied / 1 left unchanged.", output)
        self.assertIn("1 accepted proposal(s); 0 manual edit(s).", output)
        for segment in original["segments"][1:]:
            self.assertEqual(output.count(f"segment {segment['id']}"), 2)
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))

    def test_tty_leave_unchanged_preserves_the_book_without_a_revision(self):
        self.make_book()
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, _ = self.invoke(input_text="l\n", tty=True)
        self.assertEqual(code, 0, errors)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())
        self.assertIn("No changes applied; 2 finding(s) left unchanged.", output)

    def test_tty_eof_at_the_action_prompt_preserves_the_book(self):
        self.make_book()
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, _ = self.invoke(tty=True)
        self.assertEqual(code, 0, errors)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())
        self.assertIn("No changes applied; 2 finding(s) left unchanged.", output)

    def test_zero_findings_do_not_offer_an_action_or_write_a_revision(self):
        self.make_book()
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, _ = self.invoke(tty=True, input_text="a\n", client=NoFindingAuditClient())
        self.assertEqual(code, 0, errors)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertFalse((self.book / ".analysis-revisions").exists())
        self.assertIn("No attribution corrections or new review flags. No changes applied.", output)

    def test_repeat_normal_invocations_make_fresh_model_calls(self):
        self.make_book()
        first_code, _, first_errors, first_client = self.invoke()
        second_code, _, second_errors, second_client = self.invoke()
        self.assertEqual(first_code, 0, first_errors)
        self.assertEqual(second_code, 0, second_errors)
        self.assertGreater(len(first_client.calls), 0)
        self.assertEqual(len(second_client.calls), len(first_client.calls))
        self.assertIn("Fresh verification run:", first_errors)
        self.assertIn("Fresh verification run:", second_errors)

    def test_explicit_resume_reuses_the_named_run_without_new_model_calls(self):
        self.make_book()
        code, _, errors, first_client = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertGreater(len(first_client.calls), 0)
        runs = list((self.book / ".analysis-cache" / "attribution-verification").iterdir())
        self.assertEqual(len(runs), 1)
        code, _, errors, resumed_client = self.invoke("--resume", runs[0].name)
        self.assertEqual(code, 0, errors)
        self.assertEqual(resumed_client.calls, [])
        self.assertIn(f"Resuming verification run: {runs[0].name}", errors)


class AttributionVerificationLockTests(AttributionCLIHarness):
    # A real renderer entry: load the actual bundle and run the production
    # book-audio handshake. Exit 17 means the real .analysis-update.lock guard
    # refused the render (expected while the audit holds its reservation);
    # 98 means a different block; 99 means the renderer acquired the lock,
    # i.e. the production protocol stopped honoring update locks.
    RENDERER_HANDSHAKE = (
        "import sys\n"
        "from epub_tts_narrator.renderer.bundle import load_bundle\n"
        "from epub_tts_narrator.renderer.workflow import RenderBlocked, _lock_book_audio\n"
        "bundle = load_bundle(sys.argv[1])\n"
        "try:\n"
        "    lock = _lock_book_audio(bundle)\n"
        "except RenderBlocked as exc:\n"
        "    sys.exit(17 if 'analysis is being updated' in str(exc) else 98)\n"
        "lock.rmdir()\n"
        "sys.exit(99)\n")

    def test_write_mode_reserves_publication_before_model_requests(self):
        self.make_book()
        code, output, errors, _ = self.invoke("--apply")
        self.assertEqual(code, 0, errors)
        self.assertIn("Publication ownership reserved before verification: 1 book(s)", errors)
        self.assertFalse((self.book / ".analysis-update.lock").exists())
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))

    def test_live_render_lock_fails_write_modes_before_model_requests(self):
        self.make_book()
        child, release = self.hold_render_lock(self.book)
        try:
            for arguments, input_text, tty in [(("--apply",), "", False),
                                               (("--interactive",), "y\n", True)]:
                with self.subTest(arguments=arguments):
                    code, _, errors, client = self.invoke(*arguments, input_text=input_text, tty=tty)
                    self.assertEqual(code, 1)
                    self.assertEqual(client.calls, [])
                    self.assertIn("Rendering is active", errors)
                    self.assertIn(".audio-render.lock", errors)
                    self.assertIn("live process", errors)
                    self.assertFalse((self.book / ".analysis-update.lock").exists())
                    self.assertFalse((self.book / ".analysis-cache").exists())
        finally:
            release.write_text("go")
            self.assertEqual(child.wait(timeout=15), 0)
        code, output, errors, _ = self.invoke("--apply")
        self.assertEqual(code, 0, errors)
        self.assertIn("2 applied / 0 left unchanged.", output)

    def test_abandoned_marked_render_lock_is_proven_idle_and_the_audit_publishes(self):
        self.make_book()
        lock = self.book / ".audio-render.lock"
        lease = lock.with_name(lock.name + ".lease")
        crashed = subprocess.run(
            [sys.executable, "-c",
             "import os, sys; from epub_tts_narrator.renderer.run_locks import acquire_directory_lock; "
             "acquire_directory_lock(sys.argv[1]); os._exit(130)", str(lock)])
        self.assertEqual(crashed.returncode, 130)
        inode = lease.stat().st_ino
        owner = (lock / "owner.json").read_bytes()
        code, output, errors, _ = self.invoke("--apply")
        self.assertEqual(code, 0, errors)
        self.assertIn("Publication ownership reserved before verification", errors)
        self.assertIn("2 applied / 0 left unchanged.", output)
        self.assertTrue(lock.is_dir())
        self.assertEqual((lock / "owner.json").read_bytes(), owner)
        self.assertEqual(lease.stat().st_ino, inode)

    def test_renderer_starting_during_the_audit_is_refused_by_the_reservation(self):
        self.make_book()
        owner = self
        probes = []

        class RendererProbingClient(AuditClient):
            def complete(self, system, payload):
                if payload["mode"] == "attribution_review_scan":
                    probes.append(subprocess.run(
                        [sys.executable, "-c", owner.RENDERER_HANDSHAKE, str(owner.book)],
                        timeout=15, capture_output=True, text=True))
                return super().complete(system, payload)

        code, output, errors, _ = self.invoke("--apply", client=RendererProbingClient())
        self.assertEqual(code, 0, errors)
        self.assertTrue(probes)
        self.assertTrue(all(probe.returncode == 17 for probe in probes),
                        [(probe.returncode, probe.stderr[-300:]) for probe in probes])
        self.assertIn("2 applied / 0 left unchanged.", output)
        self.assertFalse((self.book / ".analysis-update.lock").exists())
        self.assertFalse((self.book / ".audio-render.lock").exists())

    def test_report_only_needs_no_ownership_and_audits_during_a_live_render(self):
        self.make_book()
        child, release = self.hold_render_lock(self.book)
        try:
            before = self.published_snapshot(self.book)
            code, output, errors, client = self.invoke()
            self.assertEqual(code, 0, errors)
            self.assertTrue(client.calls)
            self.assertIn("Report-only invocation: no publication ownership is taken", errors)
            self.assertIn("2 attribution finding(s).", output)
            self.assertIn("No changes applied; 2 finding(s) left unchanged.", output)
            self.assertEqual(self.published_snapshot(self.book), before)
            self.assertTrue((self.book / ".audio-render.lock" / "owner.json").is_file())
        finally:
            release.write_text("go")
            self.assertEqual(child.wait(timeout=15), 0)

    def test_late_publication_failure_keeps_the_audit_resumable_without_new_model_calls(self):
        from epub_tts_narrator.analysis_revision import _install
        self.make_book()
        calls = 0

        def fail_first(source, destination):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("disk failure during publication")
            return _install(source, destination)

        with patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_first):
            code, _, errors, _ = self.invoke("--apply")
        self.assertEqual(code, 1)
        self.assertIn("Attribution publication did not succeed", errors)
        match = re.search(r"Fresh verification run: ([0-9a-f]{32})", errors)
        self.assertIsNotNone(match, errors)
        run_id = match.group(1)
        self.assertIn(f"--resume {run_id}", errors)
        runs = sorted(path.name for path in (self.book / ".analysis-cache" / "attribution-verification").iterdir())
        self.assertEqual(runs, [run_id])
        self.assertFalse((self.book / ".analysis-update.lock").exists())

        code, output, errors, client = self.invoke("--resume", run_id, "--apply")
        self.assertEqual(code, 0, errors)
        self.assertEqual(client.calls, [])
        self.assertIn("2 applied / 0 left unchanged.", output)
        revised, _, _ = _completed_directory(self.book)
        self.assertEqual(revised["segments"][1]["role_id"], "tom")

    def test_earlier_book_render_does_not_block_editing_a_later_book(self):
        self.make_book()
        parent = self.book
        parent_analysis = (parent / "analysis.json").read_bytes()
        linked, descendant = self.make_descendant()
        child, release = self.hold_render_lock(parent)
        try:
            self.book = descendant
            code, output, errors, _ = self.invoke("--apply")
            self.assertEqual(code, 0, errors)
            self.assertIn("Publication ownership reserved before verification: 1 book(s)", errors)
            self.assertIn("2 applied / 0 left unchanged.", output)
            revised, _, _ = _completed_directory(descendant)
            self.assertEqual(revised["segments"][1]["role_id"], "tom")
            self.assertEqual((parent / "analysis.json").read_bytes(), parent_analysis)
            self.assertTrue((parent / ".audio-render.lock" / "owner.json").is_file())
        finally:
            release.write_text("go")
            self.assertEqual(child.wait(timeout=15), 0)


if __name__ == "__main__":
    unittest.main()
