"""Worker progress stays attached to its chunk across queues and inference lanes."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import os
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.renderer.common import digest, file_hash, write_json
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.renderer.engines import EngineError
from epub_tts_narrator.renderer.pool import RenderPool
from epub_tts_narrator.renderer.progress import (
    ChapterProgress, generation_activity, generation_notifier,
    report_worker_activity, set_phase,
)


TOKEN = "fixture-secret-must-not-appear"
SCRIPT = "Private unreleased text must not appear in worker progress."


def configuration(*names, local=False):
    config = deepcopy(DEFAULTS)
    config["engine"]["checkpoint"] = "/unused/drama-3"
    config["render_pool"]["workers"] = [
        {"name": name, "type": "local", "concurrency": 1}
        if local else
        {"name": name, "type": "fish_http", "concurrency": 1,
         "url": f"http://{name}.example/v1/tts", "api_key": TOKEN,
         "model": "drama-3", "timeout_seconds": 30, "multi_speaker": True}
        for name in names
    ]
    return config


class Engine:
    identity = {"engine": "fixture", "api_key": TOKEN}

    def __init__(self, action=None):
        self.action = action

    def synthesize(self, *args, **kwargs):
        if self.action is not None:
            self.action()
        return {"seed": kwargs["seed"], "fixture": True}

    def close(self):
        pass


class WorkerActivityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_pool(self, *names, local=False):
        pool = RenderPool(configuration(*names, local=local))
        self.addCleanup(pool.close)
        return pool

    def generate(self, pool, notify, *, filename="output.wav"):
        with generation_activity(notify):
            return pool.synthesize(SCRIPT, {"reference_text": SCRIPT}, self.root / filename, seed=42)

    def test_nested_context_restores_notifier_even_after_failure(self):
        original = generation_notifier()
        first, second = [], []
        with generation_activity(first.append):
            self.assertEqual(generation_notifier(), first.append)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                with generation_activity(second.append):
                    self.assertEqual(generation_notifier(), second.append)
                    raise RuntimeError("fixture")
            self.assertEqual(generation_notifier(), first.append)
        self.assertIs(generation_notifier(), original)

    def test_concurrent_chunks_keep_separate_worker_notifications(self):
        pool = self.make_pool("first-mac", "second-mac")
        both_started = Barrier(2)
        for worker in pool._workers:
            worker._engine = Engine(lambda: both_started.wait(5))
        first, second = [], []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.generate, pool, messages.append, filename=f"{i}.wav")
                       for i, messages in enumerate((first, second))]
            results = [future.result(5) for future in futures]
        for messages, result in zip((first, second), results):
            output = "\n".join(messages)
            chosen = result["render_worker"]
            other = "second-mac" if chosen == "first-mac" else "first-mac"
            self.assertIn(chosen, output)
            self.assertNotIn(other, output)
            self.assertNotIn(TOKEN, output)
            self.assertNotIn(SCRIPT, output)
        self.assertNotEqual(results[0]["render_worker"], results[1]["render_worker"])

    def test_waiting_for_slot_is_not_reported_as_started(self):
        pool = self.make_pool("second-mac")
        started, release, second_waiting = Event(), Event(), Event()
        pool._workers[0]._engine = Engine(lambda: (started.set(), self.assertTrue(release.wait(5))))
        first, second = [], []
        first_notify, second_notify = first.append, second.append
        events, guard = [], Lock()

        def record(notify, event, worker=None, elapsed=None):
            kwargs = {"worker": worker, "elapsed": elapsed}
            with guard:
                events.append((notify, event, kwargs))
            if notify == second_notify and event == "waiting":
                second_waiting.set()
            return report_worker_activity(notify, event, **kwargs)

        with patch("epub_tts_narrator.renderer.pool.report_worker_activity", side_effect=record):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_job = executor.submit(self.generate, pool, first_notify, filename="first.wav")
                self.assertTrue(started.wait(5))
                second_job = executor.submit(self.generate, pool, second_notify, filename="second.wav")
                try:
                    self.assertTrue(second_waiting.wait(5))
                    with guard:
                        second_events = [event for notify, event, _ in events if notify == second_notify]
                    self.assertEqual(second_events, ["waiting"])
                finally:
                    release.set()
                self.assertEqual(first_job.result(5)["render_worker"], "second-mac")
                self.assertEqual(second_job.result(5)["render_worker"], "second-mac")
        for callback in (first_notify, second_notify):
            selected = [(event, info) for notify, event, info in events if notify == callback]
            self.assertEqual([event for event, _ in selected], ["waiting", "assigned", "started", "completed"])
            self.assertGreaterEqual(selected[-1][1]["elapsed"], 0)

    def test_local_lock_does_not_claim_inference_started_before_lock_acquired(self):
        pool = self.make_pool("local-mac", local=True)
        pool._workers[0]._engine = Engine()
        assigned = Event()
        events = []
        messages = []
        notify = messages.append

        def record(callback, event, worker=None, elapsed=None):
            kwargs = {"worker": worker, "elapsed": elapsed}
            self.assertEqual(callback, notify)
            events.append(event)
            if event == "assigned":
                assigned.set()
            return report_worker_activity(callback, event, **kwargs)

        with patch("epub_tts_narrator.renderer.pool.report_worker_activity", side_effect=record):
            with ThreadPoolExecutor(max_workers=1) as executor:
                with pool.local_lock:
                    pending = executor.submit(self.generate, pool, notify)
                    self.assertTrue(assigned.wait(5))
                    self.assertEqual(events, ["waiting", "assigned"])
                result = pending.result(5)
        self.assertEqual(result["render_worker"], "local-mac")
        self.assertEqual(events, ["waiting", "assigned", "started", "completed"])

    def test_failed_generation_reports_worker_and_returns_slot_for_retry(self):
        pool = self.make_pool("second-mac")
        messages = []

        def fail():
            raise EngineError(f"Fixture synthesis failure with {TOKEN} and {SCRIPT}")

        pool._workers[0]._engine = Engine(fail)
        with self.assertRaisesRegex(EngineError, "Fixture synthesis failure"):
            self.generate(pool, messages.append)
        self.assertEqual(pool._slots.qsize(), 1)
        output = "\n".join(messages)
        self.assertIn("second-mac", output)
        self.assertIn("fail", output.lower())
        self.assertNotIn(TOKEN, output)
        self.assertNotIn(SCRIPT, output)
        pool._workers[0]._engine = Engine()
        result = self.generate(pool, messages.append)
        self.assertEqual(result, {"seed": 42, "fixture": True,
                                 "render_worker": "second-mac", "worker_engine": Engine.identity})
        self.assertEqual(pool._slots.qsize(), 1)

    def test_engine_loading_failure_is_reported_on_assigned_worker(self):
        pool = self.make_pool("second-mac")
        messages = []
        with patch("epub_tts_narrator.renderer.fish_http.FishHTTPEngine",
                   side_effect=EngineError("Fixture initialization failed")):
            with self.assertRaisesRegex(EngineError, "initialization failed"):
                self.generate(pool, messages.append)
        output = "\n".join(messages)
        self.assertIn("Generating speech on second-mac", output)
        self.assertIn("second-mac: speech generation failed", output)
        self.assertEqual(pool._slots.qsize(), 1)
        self.assertIsNone(pool._workers[0]._engine)

    def test_no_notifier_and_plain_callable_are_supported(self):
        pool = self.make_pool("second-mac")
        pool._workers[0]._engine = Engine()
        with generation_activity(None):
            self.assertEqual(pool.synthesize(SCRIPT, {}, self.root / "no-notify.wav", seed=42)["render_worker"],
                             "second-mac")
        messages = []
        report_worker_activity(messages.append, "started", worker="second-mac")
        report_worker_activity(messages.append, "completed", worker="second-mac", elapsed=2.5)
        self.assertTrue(messages)
        self.assertTrue(all("second-mac" in message for message in messages))

    def test_closed_log_callback_does_not_fail_successful_generation(self):
        pool = self.make_pool("second-mac")
        pool._workers[0]._engine = Engine()
        for error in (OSError("closed pipe"), ValueError("closed stream")):
            with self.subTest(error=type(error).__name__):
                def closed(message):
                    raise error
                result = self.generate(pool, closed)
                self.assertEqual(result["render_worker"], "second-mac")
                self.assertEqual(pool._slots.qsize(), 1)

    def test_failed_worker_is_identified_without_claiming_audio_was_created(self):
        messages = []
        with ChapterProgress(3, 12, [{"spoken_text": "Hello"}],
                             notify=messages.append, heartbeat=False) as progress:
            with progress.task(0) as notify:
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                report_worker_activity(notify, "started", "second-mac")
                report_worker_activity(notify, "failed", "second-mac", 0.1)
                notify("Attempt 1/3 rejected: generation_or_verification_error (HTTP 401)")
                rejection = next(message for message in reversed(messages) if "HTTP 401" in message)
                self.assertIn("Chunk 1", rejection)
                self.assertIn("worker second-mac", rejection)
                self.assertNotIn("audio from", rejection)

    def test_compact_live_progress_keeps_generating_worker_and_local_verification_visible(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        output = Terminal()
        with patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
            with ChapterProgress(14, 24, [{"spoken_text": "text"}] * 333,
                                 stream=output, heartbeat=False) as progress:
                with progress.task(198) as local, progress.task(199) as remote:
                    for notify, worker in ((local, "local"), (remote, "second-mac")):
                        set_phase(notify, "generating speech, attempt 1/12", generated=True)
                        report_worker_activity(notify, "started", worker)
                    report_worker_activity(local, "completed", "local", 5)
                    set_phase(local, "verifying speech, attempt 1/12")
                    progress.refresh()
                    live = output.getvalue().rsplit("\033[2K", 1)[-1]
                    self.assertLessEqual(len(live), 79)
                    self.assertIn("second-mac", live)
                    self.assertIn("gen", live)
                    self.assertIn("#200", live)
                    self.assertIn("local", live)
                    self.assertIn("verify", live)
                    self.assertIn("#199", live)

    def test_saved_source_restores_provenance_without_claiming_fresh_generation(self):
        messages = []
        with ChapterProgress(3, 12, [{"spoken_text": "Hello"}],
                             notify=messages.append, heartbeat=False) as progress:
            with progress.task(0) as notify:
                before = list(messages)
                report_worker_activity(notify, "source", "second-mac")
                self.assertEqual(messages, before)
                set_phase(notify, "verifying saved take 4")
                progress.refresh()
                self.assertIn("local", messages[-1])
                self.assertIn("second-mac", messages[-1])
                notify("Saved take 4 rejected: text_mismatch")
                self.assertIn("Chunk 1", messages[-1])
                self.assertIn("audio from second-mac", messages[-1])
                self.assertFalse(any("Generating speech" in message for message in messages))

    def test_invalid_saved_worker_name_cannot_enter_progress_output(self):
        for name in ("bad\nworker", "\033[31mred", "https://user:secret@host/", "a" * 65, 123, None):
            with self.subTest(name=repr(name)):
                messages = []
                with ChapterProgress(3, 12, [{"spoken_text": "Hello"}],
                                     notify=messages.append, heartbeat=False) as progress:
                    with progress.task(0) as notify:
                        report_worker_activity(notify, "source", name)
                        set_phase(notify, "verifying saved take 4")
                        notify("Saved verification finished")
                        progress.refresh()
                text = "\n".join(messages)
                if isinstance(name, str):
                    self.assertNotIn(name, text)
                self.assertNotIn("audio from", text)

    def test_resumed_saved_take_recovers_worker_provenance_before_verification(self):
        from epub_tts_narrator.renderer.workflow import verified_take

        directory = self.root / "saved"
        directory.mkdir()
        wav = directory / "attempt-001.wav"
        wav.write_bytes(b"fixture waveform; recognition is mocked")
        request = {"kind": "speech", "text": "Hello"}
        write_json(directory / "attempt-001.json", {
            "request_hash": digest(request), "request": request, "attempt": 1,
            "validation_key": "old-policy", "audio_file": wav.name,
            "audio_sha256": file_hash(wav), "engine_metadata": {"render_worker": "second-mac"},
            "report": {"passed": False, "retry_on_resume": True,
                       "reasons": ["generation_incomplete"]},
        })
        config = deepcopy(DEFAULTS)
        config["render"]["keep_intermediate_audio"] = True
        messages = []
        generate = Mock(side_effect=AssertionError("saved audio should not regenerate"))
        with ChapterProgress(3, 12, [{"spoken_text": "Hello"}],
                             notify=messages.append, heartbeat=False) as progress:
            with progress.task(0) as notify:
                def verify(*args, **kwargs):
                    progress.refresh()
                    self.assertIn("local", messages[-1])
                    self.assertIn("second-mac", messages[-1])
                    return {"passed": True, "reasons": []}
                with patch("epub_tts_narrator.renderer.workflow.make_validation_key", return_value="new-policy"), \
                     patch("epub_tts_narrator.renderer.workflow.validate_audio", side_effect=verify):
                    result, record = verified_take("Hello", generate, directory, request,
                                                   SimpleNamespace(identity={"backend": "fixture"}),
                                                   config, notify=notify)
                self.assertEqual(result, wav)
                self.assertTrue(record["report"]["passed"])
        generate.assert_not_called()
        self.assertFalse(any("Generating speech" in message for message in messages))

    def test_chapter_logs_identify_chunk_and_local_verification_provenance(self):
        messages = []
        with ChapterProgress(3, 12, [{"spoken_text": "First"}, {"spoken_text": "Second"}],
                             notify=messages.append, heartbeat=False) as progress:
            with progress.task(1) as notify:
                set_phase(notify, "generating speech, attempt 1/3", generated=True)
                report_worker_activity(notify, "waiting")
                progress.refresh()
                self.assertIn("waiting", messages[-1].lower())
                report_worker_activity(notify, "assigned", worker="second-mac")
                report_worker_activity(notify, "started", worker="second-mac")
                progress.refresh()
                self.assertIn("second-mac", messages[-1])
                self.assertIn("generat", messages[-1].lower())
                report_worker_activity(notify, "completed", worker="second-mac", elapsed=1.5)
                for phase in ("normalizing speech", "verifying speech, attempt 1/3"):
                    set_phase(notify, phase)
                    progress.refresh()
                    self.assertIn("local", messages[-1].lower())
                    self.assertIn("second-mac", messages[-1])
                notify("Attempt 1/3 rejected: text_mismatch")
                rejection = next(message for message in reversed(messages) if "rejected" in message)
                self.assertIn("second-mac", rejection)
                self.assertRegex(rejection, r"(?:Chunk 2\b|#2|[Bb]lock 2)")
            with progress.task(0):
                pass


if __name__ == "__main__":
    unittest.main()
