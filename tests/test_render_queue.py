"""Offline tests for the ordered multi-book render queue.

All service contact is a fake ChapterClient and all book loops are faked
``render_remote`` / ``workflow.render`` calls: no network, no sessions, no
models. Threads synchronize on events; the admission poll interval is passed
as a fraction of a second so no wait exceeds a fraction of a second.
"""

import contextlib
from io import StringIO
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer.chapter_client import ChapterClientError
from epub_tts_narrator.renderer.cli import main, parser
from epub_tts_narrator.renderer.render_queue import (
    _book_name, _prefixed, first_completed_chapter, gate_busy, gate_capacity,
    gate_decision, render_queue_command,
)
from audition_fixtures import prepare_document
from test_contracts import valid_document


def scheduling_snapshot(*, tts_slots=4, busy=0, completed=0, coordinators=4, engine=True,
                        stages=True):
    """A GET v1/jobs scheduling block in the server's documented shape."""
    snapshot = {"jobs": {"queued": 0, "generating": busy, "assembling": 0,
                         "complete": completed, "failed": 0, "cancelled": 0},
                "active_workers": min(busy, tts_slots), "coordinators": coordinators,
                "take_identities": 1}
    if engine:
        section = {"available": True, "mode": "pipeline",
                   "capacity": {"tts_slots": tts_slots, "max_inflight": 64}}
        if stages:
            section["stages"] = {"synthesis": {"queued_now": busy, "active_now": 0}}
        snapshot["engine"] = section
    return snapshot


class GateFunctionTests(unittest.TestCase):
    def test_capacity_resolves_engine_slots_then_coordinators_then_capabilities(self):
        self.assertEqual(gate_capacity(scheduling_snapshot(tts_slots=16), {}), 16)
        self.assertEqual(gate_capacity(scheduling_snapshot(engine=False, coordinators=3),
                                       {"parallel_chapters": 9}), 3)
        self.assertEqual(gate_capacity({}, {"parallel_chapters": 5}), 5)
        self.assertIsNone(gate_capacity({}, {}))
        self.assertIsNone(gate_capacity(None, None))

    def test_busy_uses_synthesis_stage_gauges_when_the_engine_reports_them(self):
        self.assertEqual(gate_busy(scheduling_snapshot(tts_slots=16, busy=9)), 9)
        self.assertEqual(gate_busy(scheduling_snapshot(tts_slots=16, busy=0)), 0)

    def test_busy_falls_back_to_job_counts_without_an_engine_section(self):
        snapshot = scheduling_snapshot(engine=False, coordinators=3, busy=2)
        self.assertEqual(gate_busy(snapshot), 2)

    def test_busy_falls_back_to_job_counts_when_stage_gauges_are_absent(self):
        snapshot = scheduling_snapshot(tts_slots=8, busy=0, stages=False)
        snapshot["jobs"]["queued"] = 2
        snapshot["jobs"]["generating"] = 1
        self.assertEqual(gate_busy(snapshot), 3)

    def test_busy_is_unknown_without_any_usable_data(self):
        self.assertIsNone(gate_busy({}))
        self.assertIsNone(gate_busy(None))
        self.assertIsNone(gate_busy({"engine": None}))

    def test_decision_admits_only_below_capacity(self):
        self.assertEqual(gate_decision(scheduling_snapshot(tts_slots=4, busy=3), {}), "admit")
        self.assertEqual(gate_decision(scheduling_snapshot(tts_slots=4, busy=4), {}), "wait")
        self.assertEqual(gate_decision(scheduling_snapshot(tts_slots=4, busy=9), {}), "wait")
        self.assertEqual(gate_decision({}, {}), "unknown")

    def test_first_completed_chapter_reads_counts_then_job_rows(self):
        self.assertTrue(first_completed_chapter(scheduling_snapshot(completed=1)))
        self.assertTrue(first_completed_chapter({}, [{"state": "complete"}]))
        self.assertFalse(first_completed_chapter({}, [{"state": "generating"}]))
        self.assertFalse(first_completed_chapter({}, []))
        self.assertFalse(first_completed_chapter(None, None))


class FakeChapterService:
    """Fake chapter service: scripted or mutable scheduling state, no network."""

    def __init__(self, *, tts_slots=4, coordinators=4, capabilities=None,
                 unreachable=False, script=None, busy=0, completed=0):
        self.tts_slots = tts_slots
        self.coordinators = coordinators
        self.capabilities_payload = ({"parallel_chapters": coordinators or 4}
                                     if capabilities is None else capabilities)
        self.unreachable = unreachable
        self.script = list(script) if script is not None else None
        self.busy = busy
        self.completed = completed
        self.requests = []
        self.constructed = []

    def client_factory(self):
        server = self

        class Client:
            def __init__(self, settings):
                self.settings = settings
                server.constructed.append(settings)

            def request(self, method, path, payload=None, *, timeout=None):
                server.requests.append((method, path))
                if server.unreachable:
                    raise ChapterClientError("fixture service unreachable")
                if path.startswith("v1/capabilities"):
                    if server.capabilities_payload is False:
                        raise ChapterClientError("fixture capabilities unavailable")
                    return server.capabilities_payload
                if server.script:
                    server.busy, server.completed = server.script.pop(0)
                response = {"jobs": [{"state": "complete"} for _ in range(server.completed)]}
                if server.tts_slots is not None or server.coordinators is not None:
                    response["scheduling"] = scheduling_snapshot(
                        tts_slots=server.tts_slots or 0, busy=server.busy,
                        completed=server.completed, coordinators=server.coordinators or 0,
                        engine=server.tts_slots is not None)
                return response

        return Client

    def serve(self):
        return patch("epub_tts_narrator.renderer.chapter_client.ChapterClient",
                     self.client_factory())

    def jobs_requests(self):
        return [request for request in self.requests if request[1].startswith("v1/jobs")]


class QueueFixtureTests(unittest.TestCase):
    """Shared fixtures: one config file and several exported book bundles."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config = self.root / "renderer.toml"
        self.config.write_text(
            '[engine]\ncheckpoint="./tts-model"\n'
            '[verification]\nmodel="./asr-model"\n'
            '[render]\nrequire_cover=false\n'
        )
        self.books = []
        for index in (1, 2, 3):
            directory = self.root / f"book-{index}"
            export_bundle(prepare_document(valid_document()), directory)
            self.books.append(directory)

    def enable_remote_service(self):
        self.config.write_text(self.config.read_text()
                               + '[remote_chapters]\nurl = "https://queue-fixture.example"\n')

    def render_args(self, *books, extra=()):
        paths = [str(self.books[index - 1]) for index in books]
        return parser().parse_args(["render", *paths, "--config", str(self.config), *extra])

    def run_queue_async(self, args, *, notify=None, poll_seconds=0.01, **kwargs):
        messages = [] if notify is None else notify
        result = {}

        def run():
            try:
                result["code"] = render_queue_command(args, self.config,
                                                      notify=messages.append,
                                                      poll_seconds=poll_seconds, **kwargs)
            except BaseException as exc:  # surfaced by the test, never swallowed
                result["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.addCleanup(worker.join, 15)
        return worker, result, messages

    def fake_remote(self, behavior):
        """Patch render_remote with a scripted per-book behavior."""
        calls = []

        def render_remote(bundle, config, *, notify=print, chapters_only=False):
            index = len(calls)
            calls.append({"bundle": bundle, "config": config, "notify": notify,
                          "chapters_only": chapters_only})
            return behavior(index, bundle, config, notify)

        return patch("epub_tts_narrator.renderer.chapter_client.render_remote", render_remote), calls


class RemoteQueueTests(QueueFixtureTests):
    def setUp(self):
        super().setUp()
        self.enable_remote_service()

    def test_book_two_starts_only_after_the_gate_opens_and_never_pauses_afterwards(self):
        server = FakeChapterService(tts_slots=4, busy=8)
        started = [threading.Event(), threading.Event()]
        finished = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]

        def behavior(index, bundle, config, notify):
            started[index].set()
            release[index].wait(timeout=10)
            finished[index].set()
            return {"chapters": []}

        fake, calls = self.fake_remote(behavior)
        with fake, server.serve():
            worker, result, messages = self.run_queue_async(self.render_args(1, 2))
            self.assertTrue(started[0].wait(timeout=5), "book 1 was not admitted immediately")
            self.assertFalse(started[1].wait(timeout=0.3),
                             "book 2 was admitted while the service had no spare capacity")
            server.busy = 0
            self.assertTrue(started[1].wait(timeout=10), "the open gate did not admit book 2")
            polls_at_admission = len(server.jobs_requests())
            server.busy = 8
            release[1].set()
            self.assertTrue(finished[1].wait(timeout=5),
                            "an admitted book paused when the gate closed again")
            release[0].set()
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        self.assertIsNot(calls[0]["config"], calls[1]["config"])
        self.assertNotEqual(calls[0]["config"]["render"]["work_dir"],
                            calls[1]["config"]["render"]["work_dir"])
        self.assertGreaterEqual(polls_at_admission, 2)
        self.assertTrue(any(message.startswith("[book-1]") for message in messages))
        self.assertTrue(any(message.startswith("[book-2]") for message in messages))
        self.assertTrue(any("Rendering book 1 of 2" in message for message in messages))
        self.assertTrue(any("Rendering book 2 of 2" in message for message in messages))

    def test_a_single_capacity_dip_does_not_open_a_book(self):
        server = FakeChapterService(tts_slots=4, script=[(2, 0), (5, 0), (1, 0), (1, 0)])
        polls_at_admission = []

        def behavior(index, bundle, config, notify):
            if index == 1:
                polls_at_admission.append(len(server.jobs_requests()))
            return {"chapters": []}

        fake, _ = self.fake_remote(behavior)
        with fake, server.serve():
            worker, result, _ = self.run_queue_async(self.render_args(1, 2))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        self.assertEqual(polls_at_admission, [4],
                         "book 2 must wait out the dip: two open polls before and after it")

    def test_a_finished_book_admits_the_next_without_waiting_for_the_gate(self):
        server = FakeChapterService(unreachable=True)
        polls_at_admission = []

        def behavior(index, bundle, config, notify):
            if index == 1:
                polls_at_admission.append(len(server.jobs_requests()))
            return {"chapters": []}

        fake, calls = self.fake_remote(behavior)
        with fake, server.serve():
            worker, result, _ = self.run_queue_async(self.render_args(1, 2))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        self.assertEqual(len(calls), 2)
        # Admission must come from the finished-predecessor fallback (the gate
        # can never open); the exact read count races book 1's bundle prep.
        self.assertLessEqual(polls_at_admission, [3],
                             "a finished predecessor must admit the next book without sustained polling")

    def test_no_capacity_data_admits_after_the_first_completed_chapter(self):
        server = FakeChapterService(tts_slots=None, coordinators=None, capabilities=False,
                                    script=[(1, 0), (1, 1)])
        reasons = []

        def behavior(index, bundle, config, notify):
            return {"chapters": []}

        fake, _ = self.fake_remote(behavior)
        with fake, server.serve():
            worker, result, messages = self.run_queue_async(self.render_args(1, 2))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        reasons = [message for message in messages if "first completed chapter" in message]
        self.assertTrue(reasons, "the capacity-unknown fallback was not used")

    def test_a_failed_book_is_recorded_and_the_queue_continues(self):
        server = FakeChapterService(unreachable=True)

        def behavior(index, bundle, config, notify):
            if index == 0:
                raise RuntimeError("book one failed fixture")
            return {"chapters": []}

        fake, calls = self.fake_remote(behavior)
        with fake, server.serve():
            worker, result, messages = self.run_queue_async(self.render_args(1, 2))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 1)
        self.assertEqual(len(calls), 2, "the queue stopped at the failed book")
        joined = "\n".join(messages)
        self.assertIn("book one failed fixture", joined)
        self.assertIn("1 of 2 book(s) failed", joined)
        self.assertIn("book-2", joined)

    def test_every_book_succeeding_exits_zero(self):
        server = FakeChapterService(unreachable=True)
        fake, calls = self.fake_remote(lambda index, bundle, config, notify: {"chapters": []})
        with fake, server.serve():
            worker, result, messages = self.run_queue_async(self.render_args(1, 2, 3))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        self.assertEqual([Path(call["config"]["render"]["work_dir"]).parent.name
                          for call in calls], ["book-1", "book-2", "book-3"])
        self.assertTrue(any("all 3 book(s) rendered" in message for message in messages))

    def test_interruption_stops_admission_and_reports_partial_status(self):
        server = FakeChapterService(tts_slots=4, busy=8)
        gate_calls = {"jobs": 0}
        client_class = server.client_factory()

        class InterruptingClient(client_class):
            def request(self, method, path, payload=None, *, timeout=None):
                if path.startswith("v1/jobs"):
                    gate_calls["jobs"] += 1
                    if gate_calls["jobs"] >= 3:
                        raise KeyboardInterrupt
                return super().request(method, path, payload, timeout=timeout)

        release = threading.Event()

        def behavior(index, bundle, config, notify):
            release.wait(timeout=10)
            return {"chapters": []}

        fake, _ = self.fake_remote(behavior)
        with fake, patch("epub_tts_narrator.renderer.chapter_client.ChapterClient", InterruptingClient):
            worker, result, messages = self.run_queue_async(self.render_args(1, 2),
                                                            join_seconds=0.05)
            worker.join(10)
        release.set()
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 130)
        joined = "\n".join(messages)
        self.assertIn("no further books will be admitted", joined)
        self.assertIn("book-1", joined)
        self.assertIn("book-2: not started", joined)

    def test_duplicate_books_are_rejected_up_front(self):
        for paths in ((1, 1), (1, 2, 1)):
            with self.subTest(paths=paths):
                args = self.render_args(*paths)
                with self.assertRaisesRegex(ValueError, "more than once"):
                    render_queue_command(args, self.config)

    def test_chapters_selection_is_refused_for_a_multi_book_queue(self):
        args = self.render_args(1, 2, extra=("--chapters", "1"))
        with self.assertRaisesRegex(ValueError, "--chapters"):
            render_queue_command(args, self.config)

    def test_book_specific_output_options_are_refused_for_a_multi_book_queue(self):
        for flag in ("--output", "--work-dir", "--epub"):
            with self.subTest(flag=flag):
                args = self.render_args(1, 2, extra=(flag, "somewhere"))
                with self.assertRaisesRegex(ValueError, flag.replace("-", "\\-")):
                    render_queue_command(args, self.config)


class RemoteAcquisitionTests(QueueFixtureTests):
    def external_config(self):
        self.config.write_text(self.config.read_text()
                               + '[remote_chapters]\nurl = "https://fixture.example"\n'
                                 'api_key = "fixture-external-key"\n')

    def test_a_configured_service_url_is_used_directly_without_ensure_service(self):
        self.external_config()
        server = FakeChapterService(unreachable=True)
        fake, calls = self.fake_remote(lambda index, bundle, config, notify: {"chapters": []})
        with fake, server.serve(), \
                patch("epub_tts_narrator.renderer.runpod.ensure_service",
                      side_effect=AssertionError("ensure_service must not run for a configured service")):
            worker, result, _ = self.run_queue_async(self.render_args(1, 2))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        for call in calls:
            self.assertEqual(call["config"]["remote_chapters"]["url"], "https://fixture.example")
            self.assertEqual(call["config"]["remote_chapters"]["api_key"], "fixture-external-key")

    def test_a_configured_service_url_takes_precedence_over_the_remote_flag(self):
        self.external_config()
        server = FakeChapterService(unreachable=True)
        fake, calls = self.fake_remote(lambda index, bundle, config, notify: {"chapters": []})
        with fake, server.serve(), \
                patch("epub_tts_narrator.renderer.runpod.ensure_service",
                      side_effect=AssertionError("--remote must not bootstrap a managed worker "
                                                 "alongside a configured service")):
            worker, result, _ = self.run_queue_async(self.render_args(1, 2, extra=("--remote",)))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        for call in calls:
            self.assertEqual(call["config"]["remote_chapters"]["url"], "https://fixture.example")
            self.assertEqual(call["config"]["remote_chapters"]["api_key"], "fixture-external-key")

    def test_the_gate_client_is_built_from_the_configured_service(self):
        self.external_config()
        server = FakeChapterService(unreachable=True)
        fake, _ = self.fake_remote(lambda index, bundle, config, notify: {"chapters": []})
        with fake, server.serve():
            worker, result, _ = self.run_queue_async(self.render_args(1, 2))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(len(server.constructed), 1)
        self.assertEqual(server.constructed[0]["url"], "https://fixture.example")
        self.assertEqual(server.constructed[0]["api_key"], "fixture-external-key")
        self.assertEqual(server.constructed[0]["timeout_seconds"], 60.0)

    def test_remote_flag_bootstraps_the_managed_service_once(self):
        settings = Mock(name="runpod settings")
        client_config = {"remote_chapters": {"url": "https://managed.fixture",
                                             "api_key": "fixture-managed-secret",
                                             "timeout_seconds": 45}}
        ensure = Mock(return_value=client_config)
        fake, calls = self.fake_remote(lambda index, bundle, config, notify: {"chapters": []})
        server = FakeChapterService(unreachable=True)
        with fake, server.serve(), \
                patch("epub_tts_narrator.renderer.runpod_config.load_settings",
                      return_value=settings) as load, \
                patch("epub_tts_narrator.renderer.runpod.ensure_service", ensure):
            worker, result, _ = self.run_queue_async(self.render_args(1, 2, extra=("--remote",)))
            worker.join(10)
        self.assertNotIn("error", result, str(result.get("error")))
        self.assertEqual(result.get("code"), 0)
        load.assert_called_once_with(self.config)
        ensure.assert_called_once()
        self.assertIs(ensure.call_args.args[0], settings)
        for call in calls:
            remote = call["config"]["remote_chapters"]
            self.assertEqual(remote["url"], "https://managed.fixture")
            self.assertEqual(remote["api_key"], "fixture-managed-secret")
            self.assertEqual(remote["timeout_seconds"], 45)
            self.assertEqual(call["config"]["render"]["chapter_format"], "m4a")

    def test_the_managed_api_key_is_never_printed(self):
        client_config = {"remote_chapters": {"url": "https://managed.fixture",
                                             "api_key": "fixture-managed-secret",
                                             "timeout_seconds": 45}}
        fake, _ = self.fake_remote(lambda index, bundle, config, notify: {"chapters": []})
        server = FakeChapterService(unreachable=True)
        with fake, server.serve(), \
                patch("epub_tts_narrator.renderer.runpod_config.load_settings", return_value=Mock()), \
                patch("epub_tts_narrator.renderer.runpod.ensure_service", return_value=client_config):
            worker, result, messages = self.run_queue_async(self.render_args(1, 2, extra=("--remote",)))
            worker.join(10)
        self.assertNotIn("fixture-managed-secret", "\n".join(messages))


class LocalSequentialTests(QueueFixtureTests):
    def test_queue_creates_one_live_board_for_prefixed_book_progress(self):
        from epub_tts_narrator.renderer.progress import ChapterProgress
        from test_renderer_live_board import TTY, screen
        stream = TTY()
        frames = []

        def render_book(bundle, config, engine, verifier, **kwargs):
            name = Path(bundle["directory"]).name
            with ChapterProgress(1, 2, [{"spoken_text": "Fixture words."}],
                                 notify=kwargs["notify"], heartbeat=False) as chapter:
                with chapter.task(0) as task:
                    task.worker_activity("started", "fixture-worker")
                    chapter.refresh()
                    kwargs["notify"]("ordinary log")
                    rows = [line for line in screen(stream.getvalue())
                            if line.startswith(f"[{name}] Ch")]
                    frames.append(rows[-1])
            return {"chapters": []}

        with (contextlib.redirect_stdout(stream),
              patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value=""),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=render_book)):
            code = render_queue_command(self.render_args(1, 2), self.config)
        self.assertEqual(code, 0)
        self.assertEqual(len(frames), 2)
        for number, row in enumerate(frames, 1):
            self.assertTrue(row.startswith(f"[book-{number}] Ch 1/2 |"), row)
            self.assertIn("fixture-worker", row)

    def test_full_pool_keeps_next_book_pending_until_slots_are_free(self):
        from types import SimpleNamespace
        from epub_tts_narrator.renderer.render_metrics import METRICS
        full_seen = threading.Event()
        drain = threading.Event()
        second_started = threading.Event()
        outcome = []
        engine = SimpleNamespace(max_parallel=16, close=lambda: None)
        verifier = SimpleNamespace(close=lambda: None)
        snapshot = METRICS.snapshot

        def observe():
            result = snapshot()
            if result.get("synthesis", {}).get("active_now") == 16:
                full_seen.set()
            return result

        def render_book(bundle, config, shared_engine, shared_verifier, **kwargs):
            if Path(bundle["directory"]).name == "book-1":
                with contextlib.ExitStack() as stack:
                    for _ in range(16):
                        stack.enter_context(METRICS.active("synthesis"))
                    kwargs["on_synthesis_ready"]()
                    if not drain.wait(3):
                        raise RuntimeError("test did not release the full pool")
                if not second_started.wait(3):
                    raise RuntimeError("next book did not fill the drained pool")
            else:
                kwargs["on_synthesis_ready"]()
                second_started.set()
            return {"chapters": []}

        def run():
            outcome.append(render_queue_command(self.render_args(1, 2), self.config,
                                                poll_seconds=0.01, notify=lambda message: None))

        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=render_book),
              patch.object(METRICS, "snapshot", side_effect=observe)):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(full_seen.wait(3), "gate must observe the saturated pool")
                self.assertFalse(second_started.is_set(), "a full pool must hold the next book")
            finally:
                drain.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertTrue(second_started.is_set())
        self.assertEqual(outcome, [0])

    def test_next_book_starts_while_previous_book_is_assembling_on_shared_pool(self):
        from types import SimpleNamespace
        second_started = threading.Event()
        overlapped = []
        engine = SimpleNamespace(max_parallel=16, close=lambda: None)
        verifier = SimpleNamespace(close=lambda: None)

        def render_book(bundle, config, shared_engine, shared_verifier, **kwargs):
            ready = kwargs.get("on_synthesis_ready")
            if ready is not None:
                ready()
            if Path(bundle["directory"]).name == "book-1":
                overlapped.append(second_started.wait(2))
            else:
                second_started.set()
            self.assertIs(shared_engine, engine)
            self.assertIs(shared_verifier, verifier)
            return {"chapters": []}

        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=render_book)):
            code = render_queue_command(self.render_args(1, 2), self.config,
                                        poll_seconds=0.01, notify=lambda message: None)
        self.assertEqual(overlapped, [True], "book 2 must backfill while book 1 assembles")
        self.assertEqual(code, 0)

    def test_books_render_in_order_through_one_engine_and_verifier(self):
        engine, verifier = Mock(name="engine"), Mock(name="verifier")
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine",
                    return_value=engine) as engine_factory,
              patch("epub_tts_narrator.renderer.verification.create_verifier",
                    return_value=verifier) as verifier_factory,
              patch("epub_tts_narrator.renderer.workflow.render",
                    return_value={"chapters": []}) as render):
            code = render_queue_command(self.render_args(1, 2), self.config)
        self.assertEqual(code, 0)
        engine_factory.assert_called_once()
        verifier_factory.assert_called_once()
        self.assertEqual(render.call_count, 2)
        first, second = (call.args[1] for call in render.call_args_list)
        self.assertIsNot(first, second, "the two books must not share one config dict")
        self.assertNotEqual(first["render"]["work_dir"], second["render"]["work_dir"])
        self.assertEqual([Path(call.args[0]["directory"]).name for call in render.call_args_list],
                         ["book-1", "book-2"])
        self.assertEqual([call.args[2] for call in render.call_args_list], [engine, engine])
        self.assertEqual([call.args[3] for call in render.call_args_list], [verifier, verifier])
        self.assertEqual([call.kwargs.get("chapters_only") for call in render.call_args_list],
                         [False, False])

    def test_a_failed_book_is_recorded_and_the_next_book_still_renders(self):
        calls = []

        def behavior(bundle, config, engine, verifier, **kwargs):
            calls.append(Path(config["render"]["work_dir"]).parent.name)
            if len(calls) == 1:
                raise RuntimeError("local book one failed fixture")
            return {"chapters": []}

        messages = []
        with (patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=behavior)):
            code = render_queue_command(self.render_args(1, 2), self.config,
                                        notify=messages.append)
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["book-1", "book-2"])
        joined = "\n".join(messages)
        self.assertIn("local book one failed fixture", joined)
        self.assertIn("1 of 2 book(s) failed", joined)
        self.assertIn("book-2", joined)


class BookOutcomeBoardTests(QueueFixtureTests):
    """Each final book outcome reaches the shared board's pinned books line."""

    def plain(self, stream):
        return re.sub(r"\033\[[0-9;]*m", "", stream.getvalue())

    def last_visible_line(self, stream):
        from test_renderer_live_board import screen
        return re.sub(r"\033\[[0-9;]*m", "", screen(stream.getvalue())[-1])

    def test_local_outcomes_reach_the_board_as_colored_counts(self):
        from test_renderer_live_board import TTY
        stream = TTY()
        while_book_two_worked = []

        def behavior(bundle, config, engine, verifier, **kwargs):
            if Path(config["render"]["work_dir"]).parent.name == "book-1":
                raise RuntimeError("local book one failed fixture")
            while_book_two_worked.append(self.last_visible_line(stream))
            return {"chapters": []}

        with (contextlib.redirect_stdout(stream),
              patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value=""),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=behavior)):
            code = render_queue_command(self.render_args(1, 2), self.config)
        self.assertEqual(code, 1)
        self.assertEqual(while_book_two_worked, ["Books: 1 failed"])
        self.assertIn("Books: 1 succeeded | 1 failed", self.plain(stream))
        self.assertIn("Books: \033[32m1\033[0m succeeded | \033[31m1\033[0m failed",
                      stream.getvalue())

    def test_each_final_outcome_is_reported_exactly_once(self):
        reported = []

        def behavior(bundle, config, engine, verifier, **kwargs):
            if Path(config["render"]["work_dir"]).parent.name == "book-1":
                raise RuntimeError("local book one failed fixture")
            return {"chapters": []}

        with (patch("epub_tts_narrator.renderer.progress.report_book_outcome",
                    side_effect=lambda notify, key, ok: reported.append((key, ok))),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=behavior)):
            code = render_queue_command(self.render_args(1, 2), self.config,
                                        notify=lambda message: None)
        self.assertEqual(code, 1)
        self.assertEqual(reported, [(str(self.books[0]), False), (str(self.books[1]), True)])

    def run_in_shell_board(self, behavior, on_event, count=3):
        from epub_tts_narrator.renderer.progress import shell_board
        from epub_tts_narrator.run_events import EventBus
        bus = EventBus()
        bus.subscribe(on_event)
        with (shell_board(bus),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=Mock()),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=Mock()),
              patch("epub_tts_narrator.renderer.workflow.render", side_effect=behavior)):
            return render_queue_command(self.render_args(*range(1, count + 1)), self.config, poll_seconds=0.01)

    def test_a_retried_book_goes_back_to_the_front_of_the_queue(self):
        rendered, broken = [], {"book-1"}

        def behavior(bundle, config, engine, verifier, **kwargs):
            name = Path(config["render"]["work_dir"]).parent.name
            rendered.append(name)
            if name in broken:
                raise RuntimeError("fixture: voices were missing")
            return {"chapters": []}

        def on_event(event):
            if event.kind == "book" and event.data["state"] == "failed" and event.data["retry"] is not None:
                broken.discard(event.scope["book"])  # the person fixed it, then pressed r
                self.assertTrue(event.data["retry"]())

        self.assertEqual(self.run_in_shell_board(behavior, on_event), 0)
        self.assertEqual(rendered, ["book-1", "book-1", "book-2", "book-3"])

    def test_a_queue_with_only_failed_books_waits_for_retries_until_released(self):
        holds = []

        def behavior(bundle, config, engine, verifier, **kwargs):
            if Path(config["render"]["work_dir"]).parent.name == "book-2":
                raise RuntimeError("fixture failure")
            return {"chapters": []}

        def on_event(event):
            if event.kind == "hold" and event.data["release"] is not None:
                holds.append(event.data["failed"])
                event.data["release"]()

        self.assertEqual(self.run_in_shell_board(behavior, on_event, count=2), 1)
        self.assertEqual(holds, [1])

    def test_a_stopped_book_reports_stopped_not_failed(self):
        from epub_tts_narrator.renderer.interrupts import RenderInterrupted
        states, lines = [], []

        def behavior(bundle, config, engine, verifier, **kwargs):
            if Path(config["render"]["work_dir"]).parent.name == "book-1":
                raise RenderInterrupted()
            return {"chapters": []}

        def on_event(event):
            if event.kind == "book":
                states.append((event.scope["book"], event.data["state"]))
            elif event.kind == "log":
                lines.append(event.text)
            elif event.kind == "outcome":
                # A stopped book is not counted among the failed books.
                self.assertNotEqual(event.scope.get("book"), "book-1")

        self.assertEqual(self.run_in_shell_board(behavior, on_event, count=2), 130)
        self.assertIn(("book-1", "stopped"), states)
        self.assertNotIn(("book-1", "failed"), states)
        self.assertTrue(any(line.startswith("stopped: stopped before finishing") for line in lines), lines)
        self.assertTrue(any(line.startswith("Render queue stopped: 1 of 2 book(s) did not finish: book-1")
                            for line in lines), lines)

    def test_a_preparation_failure_counts_as_a_failed_book(self):
        from test_renderer_live_board import TTY
        stream = TTY()
        with (contextlib.redirect_stdout(stream),
              patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value=""),
              patch("epub_tts_narrator.renderer.render_queue._prepare_book",
                    side_effect=RuntimeError("fixture preparation failure"))):
            code = render_queue_command(self.render_args(1), self.config)
        self.assertEqual(code, 1)
        plain = self.plain(stream)
        self.assertIn("Books: 1 failed", plain)
        self.assertNotIn(" succeeded", plain)
        self.assertIn("Books: \033[31m1\033[0m failed", stream.getvalue())

    def test_remote_failure_reaches_the_board_counts(self):
        from test_renderer_live_board import TTY
        self.enable_remote_service()
        server = FakeChapterService(unreachable=True)
        stream = TTY()
        while_book_two_worked = []

        def behavior(index, bundle, config, notify):
            if index == 0:
                raise RuntimeError("book one failed fixture")
            while_book_two_worked.append(self.last_visible_line(stream))
            return {"chapters": []}

        fake, _ = self.fake_remote(behavior)
        with fake, server.serve(), contextlib.redirect_stdout(stream), \
                patch("epub_tts_narrator.renderer.progress._scheduling_segment", return_value=""):
            code = render_queue_command(self.render_args(1, 2), self.config, poll_seconds=0.01)
        self.assertEqual(code, 1)
        self.assertEqual(while_book_two_worked, ["Books: 1 failed"])
        self.assertIn("Books: 1 succeeded | 1 failed", self.plain(stream))
        self.assertIn("Books: \033[32m1\033[0m succeeded | \033[31m1\033[0m failed",
                      stream.getvalue())


class QueueCliWiringTests(QueueFixtureTests):
    def run_cli(self, *args):
        stdout, stderr = StringIO(), StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main([str(argument) for argument in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_multi_book_render_dispatches_to_the_queue(self):
        with patch("epub_tts_narrator.renderer.render_queue.render_queue_command",
                   return_value=0) as queue:
            code, _, error = self.run_cli("render", self.books[0], self.books[1],
                                          "--config", self.config)
        self.assertEqual(code, 0, error)
        queue.assert_called_once()
        args, config_path = queue.call_args.args
        self.assertEqual(args.bundle, [self.books[0], self.books[1]])
        self.assertEqual(config_path, self.config)

    def test_single_book_render_keeps_the_ordinary_path(self):
        engine, verifier = Mock(), Mock()
        with (patch("epub_tts_narrator.renderer.render_queue.render_queue_command",
                    side_effect=AssertionError("single-book render must not use the queue")),
              patch("epub_tts_narrator.renderer.model_setup.ModelSetup.ensure"),
              patch("epub_tts_narrator.renderer.engines.create_engine", return_value=engine),
              patch("epub_tts_narrator.renderer.verification.create_verifier", return_value=verifier),
              patch("epub_tts_narrator.renderer.workflow.render") as render):
            render.return_value = {"status": "chapters_complete", "chapter_count": 1,
                                   "chapters": [{"number": 1, "title": "Chapter One",
                                                 "audio_file": "/fixture/chapter.mp3"}]}
            code, output, error = self.run_cli("render", self.books[0], "--config", self.config)
        self.assertEqual(code, 0, error)
        render.assert_called_once()

    def test_single_book_with_remote_flag_dispatches_to_the_queue(self):
        with patch("epub_tts_narrator.renderer.render_queue.render_queue_command",
                   return_value=0) as queue:
            code, _, error = self.run_cli("render", self.books[0], "--remote",
                                          "--config", self.config)
        self.assertEqual(code, 0, error)
        queue.assert_called_once()
        self.assertTrue(queue.call_args.args[0].remote)

    def test_multi_book_with_chapters_is_refused_with_a_clear_message(self):
        code, _, error = self.run_cli("render", self.books[0], self.books[1],
                                      "--chapters", "1", "--config", self.config)
        self.assertEqual(code, 2)
        self.assertIn("--chapters", error)

    def test_duplicate_books_are_refused_with_a_clear_message(self):
        code, _, error = self.run_cli("render", self.books[0], self.books[0],
                                      "--config", self.config)
        self.assertEqual(code, 2)
        self.assertIn("more than once", error)


class SmallUnitTests(unittest.TestCase):
    def test_book_name_uses_the_book_directory(self):
        self.assertEqual(_book_name("/tmp/runs/book-2"), "book-2")

    def test_prefixed_notify_prepends_the_book_prefix(self):
        lines = []
        _prefixed("[book-2] ", lines.append)("hello")
        self.assertEqual(lines, ["[book-2] hello"])


if __name__ == "__main__":
    unittest.main()
