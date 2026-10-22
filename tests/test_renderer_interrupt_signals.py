"""Ctrl-C drains active cache work; another Ctrl-C bypasses blocked executors."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.interrupts import (
    RenderInterrupted, RenderInterrupts, defer_force_stop, run_subprocess,
    stop_if_requested, stop_requested,
)


class RenderInterruptControllerTests(unittest.TestCase):
    def test_first_signal_requests_stop_visible_to_workers_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        notices, forced = [], Mock()
        self.assertFalse(stop_requested())
        stop_if_requested()
        with RenderInterrupts(notices.append, force_exit=forced) as controller:
            controller._handle(signal.SIGINT, None)
            self.assertTrue(controller.requested)
            self.assertFalse(controller.forced)
            with ThreadPoolExecutor(max_workers=1) as executor:
                self.assertTrue(executor.submit(stop_requested).result())
            with self.assertRaises(RenderInterrupted):
                stop_if_requested()
            forced.assert_not_called()
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)
        self.assertFalse(stop_requested())
        self.assertIn("save to cache", notices[0])
        self.assertIn("Ctrl-C again", notices[0])

    def test_second_signal_exits_130_without_waiting_for_workers(self):
        forced = Mock()
        with RenderInterrupts(lambda _: None, force_exit=forced) as controller:
            controller._handle(signal.SIGINT, None)
            controller._handle(signal.SIGINT, None)
            self.assertTrue(controller.forced)
            forced.assert_called_once_with(130)

    def test_force_waits_only_for_ownership_publication(self):
        forced = Mock()
        with RenderInterrupts(lambda _: None, force_exit=forced) as controller:
            with defer_force_stop():
                controller._handle(signal.SIGINT, None)
                controller._handle(signal.SIGINT, None)
                forced.assert_not_called()
            forced.assert_called_once_with(130)

    def test_force_waits_for_a_worker_ownership_transaction(self):
        entered, release = threading.Event(), threading.Event()
        forced = Mock()

        def publish_owner():
            with defer_force_stop():
                entered.set()
                release.wait(3)

        with RenderInterrupts(lambda _: None, force_exit=forced) as controller:
            worker = threading.Thread(target=publish_owner)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                controller._handle(signal.SIGINT, None)
                controller._handle(signal.SIGINT, None)
                forced.assert_not_called()
            finally:
                release.set()
                worker.join(3)
            forced.assert_called_once_with(130)

    def test_background_context_does_not_install_a_process_signal_handler(self):
        def background():
            with RenderInterrupts() as controller:
                return controller._installed, stop_requested()

        previous = signal.getsignal(signal.SIGINT)
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.assertEqual(executor.submit(background).result(), (False, False))
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_nested_controller_restores_outer_state(self):
        with RenderInterrupts(lambda _: None) as outer:
            outer._handle(signal.SIGINT, None)
            with RenderInterrupts(lambda _: None):
                self.assertFalse(stop_requested())
            self.assertTrue(stop_requested())

    def test_notice_failure_cannot_prevent_interrupt(self):
        with RenderInterrupts(Mock(side_effect=OSError("closed stream"))) as controller:
            controller._handle(signal.SIGINT, None)
            self.assertTrue(stop_requested())

    def test_child_registration_is_complete_before_deferred_force_exit(self):
        forced = Mock()
        process = Mock(pid=4321, args=["test-child"])
        process.communicate.return_value = (b"ok", b"")
        process.poll.return_value = 0
        with RenderInterrupts(lambda _: None, force_exit=forced) as controller:
            def launch(*args, **kwargs):
                controller._handle(signal.SIGINT, None)
                controller._handle(signal.SIGINT, None)
                forced.assert_not_called()
                return process
            with (patch("epub_tts_narrator.renderer.interrupts.subprocess.Popen", side_effect=launch),
                  patch.object(controller, "_kill") as kill):
                run_subprocess(["test-child"])
                kill.assert_called_once_with(process)
                forced.assert_called_once_with(130)

    def test_subprocess_run_contract_and_cleanup(self):
        with RenderInterrupts(lambda _: None) as controller:
            result = run_subprocess(
                [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                input="hello", capture_output=True, text=True, check=True, timeout=3,
            )
            self.assertEqual(result.stdout, "HELLO\n")
            self.assertEqual(controller._children, {})
            with self.assertRaises(subprocess.CalledProcessError) as failed:
                run_subprocess([sys.executable, "-c", "raise SystemExit(7)"], check=True)
            self.assertEqual(failed.exception.returncode, 7)
            self.assertEqual(controller._children, {})
            with self.assertRaises(subprocess.TimeoutExpired):
                run_subprocess([sys.executable, "-c", "import time; time.sleep(30)"],
                               capture_output=True, timeout=.1)
            self.assertEqual(controller._children, {})

    def test_without_controller_subprocess_mock_boundary_is_retained(self):
        with patch("subprocess.run") as run:
            result = run_subprocess(["ffmpeg"], timeout=3, check=False)
            self.assertIs(result, run.return_value)
            self.assertEqual(run.call_args.args, (["ffmpeg"],))
            self.assertEqual(run.call_args.kwargs["timeout"], 3)


@unittest.skipUnless(os.name == "posix", "Terminal process-group SIGINT requires POSIX")
class RealRenderSignalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def launch(self, source):
        process = subprocess.Popen(
            [sys.executable, "-c", source, str(self.directory)],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        self.addCleanup(self.cleanup_process, process)
        return process

    @staticmethod
    def cleanup_process(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)

    def wait_for(self, filename, process):
        deadline = time.monotonic() + 5
        path = self.directory / filename
        while time.monotonic() < deadline:
            if path.exists():
                return path
            if process.poll() is not None:
                output, error = process.communicate()
                self.fail(f"Renderer exited before {filename}: {output!r} {error!r}")
            time.sleep(.01)
        self.fail(f"Renderer did not reach {filename}")

    def test_first_terminal_sigint_lets_child_finish_and_cache_be_saved(self):
        process = self.launch('''
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from epub_tts_narrator.renderer.interrupts import (
    RenderInterrupts, RenderInterrupted, run_subprocess, stop_if_requested,
)
root = Path(sys.argv[1])
def active_task():
    run_subprocess([sys.executable, "-c", """
from pathlib import Path
import sys, time
root = Path(sys.argv[1])
(root / 'child-started').write_text('ready')
time.sleep(.5)
(root / 'normalized').write_text('complete')
""", str(root)], timeout=5, check=True)
    (root / 'verified-cache').write_text('verified and saved')
try:
    with RenderInterrupts():
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(active_task).result()
        stop_if_requested()
except RenderInterrupted:
    raise SystemExit(130)
''')
        self.wait_for("child-started", process)
        os.killpg(process.pid, signal.SIGINT)
        _, error = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 130, error)
        self.assertEqual((self.directory / "verified-cache").read_text(), "verified and saved")
        self.assertEqual((self.directory / "normalized").read_text(), "complete")
        self.assertIn("save to cache", error)

    def test_second_terminal_sigint_bypasses_a_blocked_executor(self):
        process = self.launch('''
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys, threading, time
from epub_tts_narrator.renderer.interrupts import RenderInterrupts
with RenderInterrupts():
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(threading.Event().wait)
        (Path(sys.argv[1]) / 'ready').write_text('ready')
        time.sleep(30)
''')
        self.wait_for("ready", process)
        os.killpg(process.pid, signal.SIGINT)
        time.sleep(.05)
        start = time.monotonic()
        os.killpg(process.pid, signal.SIGINT)
        _, error = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 130, error)
        self.assertLess(time.monotonic() - start, 2)
        self.assertIn("Stopping immediately", error)

    def test_second_terminal_sigint_kills_the_shielded_child_process_group(self):
        process = self.launch('''
from pathlib import Path
import sys
from epub_tts_narrator.renderer.interrupts import RenderInterrupts, run_subprocess
with RenderInterrupts():
    run_subprocess([sys.executable, "-c", """
from pathlib import Path
import os, sys, time
root = Path(sys.argv[1])
(root / 'child-pid').write_text(str(os.getpid()))
time.sleep(30)
(root / 'unexpected-cache').write_text('orphan child survived')
""", sys.argv[1]], timeout=40)
''')
        self.assertGreater(int(self.wait_for("child-pid", process).read_text()), 0)
        os.killpg(process.pid, signal.SIGINT)
        time.sleep(.05)
        os.killpg(process.pid, signal.SIGINT)
        # The child inherits these output pipes. communicate cannot reach EOF
        # within this timeout if it survives the parent's immediate exit.
        _, error = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 130, error)
        self.assertFalse((self.directory / "unexpected-cache").exists())


if __name__ == "__main__":
    unittest.main()
