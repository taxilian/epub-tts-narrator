"""Render repairs and the at-a-glance view: listening to a failed passage's takes and accepting one."""

import asyncio
from pathlib import Path
import time
import unittest

from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.failed_take_repair import FailedTakeRepair
from epub_tts_narrator.renderer.heard_variants import effective_variants, recognizer_key, trust_basis
from epub_tts_narrator.run_events import EventBus
from epub_tts_narrator.run_shell_model import STALL_SECONDS, RunModel, repair_hint
from test_renderer_take_selection import TakeSelectionTests


class FixtureRecognizer:
    identity = {"type": "fixture-recognizer", "model": "fixture-v1"}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FailedTakeRepairTests(unittest.TestCase):
    def setUp(self):
        self.fixture = TakeSelectionTests("test_list_takes_returns_all_attempts_with_acceptance_and_eligibility")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.book = self.fixture.root / "book"
        self.book.mkdir()
        write_json(self.book / "source.json", {"book": {"id": "fixture-book"}})
        self.clips, self.directory = self.fixture.clip("render")
        self.work = self.clips.parent
        self.config = {"_heard_variants_directory": str(self.book), "casting": {"player": ["fixture-player"]}}

    def failed_take(self, number, transcript, expected="Kvothe smiled at the fire tonight."):
        record = self.fixture.attempt(self.directory, number)
        record["report"].update({"expected_text": expected, "transcript": transcript, "purpose": "speech"})
        write_json(self.directory / f"attempt-{number:03d}.json", record)
        return record

    def repair(self):
        return FailedTakeRepair(self.work, self.config, FixtureRecognizer(), self.fixture.request_hash)

    def test_takes_list_what_a_listener_needs_and_which_can_teach_the_recognizer(self):
        self.failed_take(1, "quote smiled at the fire tonight.")
        self.failed_take(2, "Kvothe smiled at the fire.")
        takes = self.repair().takes()
        self.assertEqual([(row["attempt"], row["passed"], row["eligible"], row["learnable"]) for row in takes],
                         [(1, False, True, True), (2, False, True, False)])
        self.assertEqual(takes[0]["transcript"], "quote smiled at the fire tonight.")
        self.assertEqual(takes[0]["expected_text"], "Kvothe smiled at the fire tonight.")
        self.assertEqual(self.repair().player_command, ["fixture-player"])

    def test_accepting_and_learning_records_a_manual_selection_and_a_person_approved_variant(self):
        record = self.failed_take(1, "quote smiled at the fire tonight.")
        outcome = self.repair().accept(1, learn=True)
        self.assertEqual((outcome["attempt"], outcome["basis"]), (1, "manual_take_selection"))
        self.assertEqual(outcome["learned"], [("kvothe", "quote")])
        accepted = read_json(self.directory / "accepted.json")
        self.assertEqual(accepted["audio_sha256"], record["audio_sha256"])
        variants = effective_variants(self.book, recognizer_key(FixtureRecognizer()))
        self.assertEqual(trust_basis(variants[("kvothe", "quote")]), "listened")

    def test_accepting_without_learning_or_with_structural_differences_teaches_nothing(self):
        self.failed_take(1, "quote smiled at the fire tonight.")
        self.failed_take(2, "Kvothe smiled at the fire.")
        self.assertEqual(self.repair().accept(1, learn=False)["learned"], [])
        self.assertEqual(self.repair().accept(2, learn=True)["learned"], [])
        self.assertFalse((self.book / "heard-variants.json").exists())

    def test_learning_is_off_when_the_book_turned_heard_variants_off(self):
        self.failed_take(1, "quote smiled at the fire tonight.")
        self.config["_heard_variants_directory"] = None
        self.assertFalse(self.repair().takes()[0]["learnable"])
        self.assertEqual(self.repair().accept(1, learn=True)["learned"], [])

    def test_a_missing_take_is_refused(self):
        self.failed_take(1, "quote smiled at the fire tonight.")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.repair().accept(7)


class GlanceTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.bus = EventBus(clock=self.clock)
        self.model = RunModel("render", clock=self.clock)
        self.bus.subscribe(self.model.on_event)

    def snapshot(self, key, book, chapter, done, active):
        self.bus.publish("status", scope={"book": book, "chapter": chapter}, data={"key": key, "snapshot": {
            "chapter": chapter, "total_chapters": 9, "done": done, "blocks": 40, "elapsed": 60,
            "remaining": 300, "active": active}})

    def test_books_clips_verification_throughput_and_rejections_at_a_glance(self):
        for book in ("b1", "b2", "b3"):
            self.bus.publish("book", "queued", scope={"book": book}, data={"key": f"/books/{book}", "state": "queued"})
        self.bus.publish("book", "rendering", scope={"book": "b1"}, data={"key": "/books/b1", "state": "rendering"})
        self.snapshot("k1", "b1", 3, 4, [
            {"block": 5, "phase": "gen 1/3", "worker": "gpu-a", "worker_stage": "generating", "elapsed": 2},
            {"block": 6, "phase": "verify 1/3", "worker": None, "worker_stage": None, "elapsed": 1},
            {"block": 7, "phase": "waiting for a worker", "worker": "gpu-a", "worker_stage": "waiting", "elapsed": 1}])
        self.clock.now += 60
        self.snapshot("k1", "b1", 3, 10, [])
        self.bus.log("[b1] Ch 3:   Chunk 4 (audio from gpu-a): Attempt 1/6 rejected: text_mismatch")
        self.bus.log("[b1] === Completed chapter 2 / 9 in 03:00; run elapsed 09:00 (2 / 9 selected chapters ready). ===")
        self.bus.publish("outcome", scope={"book": "b3"}, data={"key": "/books/b3", "succeeded": False})
        glance = self.model.glance()
        self.assertEqual(glance["Accepted"][0], "6")
        self.assertIn("1.2 blocks/min", glance["Accepted"][1])
        self.assertEqual(glance["Rejected takes"][:1], ("1",))
        self.assertEqual(glance["Books"][0], "3")
        self.assertIn("1 rendering", glance["Books"][1])
        self.assertIn("1 failed", glance["Books"][1])
        rows = {row[0]: row for row in self.model.book_rows()}
        # 2 of 9 chapters ready plus a quarter of chapter 3: 25%.
        self.assertEqual(rows["b1"][1:6], (25, "rendering", 2, 9, 1))
        self.assertEqual(rows["b3"][2], "failed")
        self.assertEqual(rows["b3"][7], 1)
        self.assertIsNone(rows["b2"][1])

    def test_block_activity_counts_generating_verifying_and_waiting(self):
        self.snapshot("k", None, 1, 0, [
            {"block": 1, "phase": "gen 1/3", "worker_stage": "generating"},
            {"block": 2, "phase": "verify 2/3", "worker_stage": None},
            {"block": 3, "phase": "normalize", "worker_stage": None},
            {"block": 4, "phase": "waiting for a worker", "worker_stage": "waiting"}])
        self.assertEqual(self.model.activity(), {"generating": 1, "verifying": 2, "waiting": 1, "other": 0})
        self.assertEqual(self.model.glance()["Generating"][0], "1")

    def test_health_reports_stalls_failures_holds_and_the_end(self):
        self.snapshot("k", None, 1, 0, [])
        self.assertEqual(self.model.health(), ("ok", "Running normally"))
        self.clock.now += STALL_SECONDS + 1
        self.assertEqual(self.model.health()[0], "warning")
        self.assertIn("No block accepted", self.model.health()[1])
        self.bus.publish("failure", "x", scope={"chapter": 1}, data={"record": {"number": 1}})
        self.bus.publish("hold", data={"release": lambda: None, "failed": 1})
        self.assertEqual(self.model.health(), ("error", "Waiting for retries: 1 failure(s)"))
        self.model.finish(2)
        self.assertEqual(self.model.health()[0], "error")

    def test_a_single_book_s_unprefixed_lines_belong_to_it(self):
        self.bus.publish("book", "rendering", scope={"book": "solo"}, data={"key": "solo", "state": "rendering"})
        self.bus.log("Reusing completed chapter 1 / 4: One (x.m4a)")
        self.snapshot("k", None, 2, 1, [])
        self.assertEqual(self.model.book_rows()[0][:5], ("solo", 25, "rendering", 1, 4))
        self.assertEqual(self.model.logs[-1][1].scope["book"], "solo")

    def test_a_failed_book_can_go_back_into_the_queue_and_clears_once_it_renders(self):
        requested = []
        self.bus.publish("book", "failed", scope={"book": "b2"},
                         data={"key": "/books/b2", "state": "failed", "retry": lambda: requested.append("b2") or True})
        [item] = self.model.current_failures()
        self.assertEqual((item["kind"], item["state"]), ("book", "failed"))
        self.assertTrue(self.model.retry(item))
        self.assertEqual((requested, item["state"], self.model.current_failures()), (["b2"], "retrying", []))
        self.bus.publish("outcome", scope={"book": "b2"}, data={"key": "/books/b2", "succeeded": True})
        self.assertEqual(item["state"], "retried")
        # Failing again after a retry turns the same entry back into a failure, with its new retry.
        item["state"] = "retrying"
        self.bus.log("[b2] failed: fixture failure again")
        self.bus.publish("book", "failed", scope={"book": "b2"},
                         data={"key": "/books/b2", "state": "failed", "retry": lambda: True})
        self.assertEqual([entry["state"] for entry in self.model.failures], ["failed"])
        self.assertEqual(self.model.failures[0]["error"], "failed: fixture failure again")

    def test_node_slots_list_every_slot_with_its_block_and_idle_ones(self):
        from epub_tts_narrator.renderer.node_activity import NodeActivity, job
        activity = NodeActivity(clock=self.clock)
        owner = object()
        activity.register(owner, "local", "speech", 2, local=True)
        activity.register(owner, "local", "verify", 1, local=True)
        activity.register(owner, "second-mac", "speech", 3, local=False)
        with job("b1", 3, 7):
            with activity.busy("second-mac", "speech"):
                self.clock.now += 4
                with activity.busy("local", "verify", {"book": "b2", "chapter": 1, "block": 2}):
                    nodes = activity.snapshot()
        self.assertEqual([(node["name"], node["role"]) for node in nodes],
                         [("local", "speech"), ("local", "verify"), ("second-mac", "speech")])
        rows = self.model.node_rows(nodes)
        self.assertEqual(len(rows), 6)
        busy = [(name, role, slot, item["book"], item["chapter"], item["block"]) for name, role, slot, item in rows if item]
        self.assertEqual(busy, [("local", "verify", 1, "b2", 1, 2), ("second-mac", "speech", 1, "b1", 3, 7)])
        self.assertEqual(self.model.node_summary(nodes)["second-mac"], {"speech": (1, 3)})
        self.assertEqual(activity.snapshot()[2]["jobs"], [])
        activity.unregister(owner)
        self.assertEqual(activity.snapshot(), [])

    def test_repair_hints_name_the_likely_fix(self):
        def item(error, error_type="RenderBlocked", **extra):
            return {"kind": "chapter", "error": error, "record": {"error_type": error_type}, **extra}
        self.assertIn("Listen to the takes", repair_hint(item("Take 3/3 failed", repair=object())))
        self.assertIn("lock", repair_hint(item("Another render may be running; retry after it finishes")))
        self.assertIn("unavailable", repair_hint(item("x", "VerificationUnavailable")))
        self.assertIn("another layout", repair_hint(item("history echo in take 2")))
        self.assertIn("Retry the chapter", repair_hint(item("odd")))


class FakeRepair:
    player_command = ["fixture-player"]

    def __init__(self):
        self.accepted = []

    def takes(self):
        return [{"attempt": 1, "passed": False, "accepted": False, "eligible": True, "ineligible_reason": None,
                 "reasons": ["text_mismatch"], "duration_seconds": 2.5, "transcript": "quote smiled",
                 "expected_text": "Kvothe smiled", "audio_path": "/clips/take-1.wav", "learnable": True},
                {"attempt": 2, "passed": False, "accepted": False, "eligible": False,
                 "ineligible_reason": "history echo", "reasons": ["text_mismatch"], "duration_seconds": 2.0,
                 "transcript": "earlier line", "expected_text": "Kvothe smiled", "audio_path": "/clips/take-2.wav",
                 "learnable": False}]

    def accept(self, attempt, *, learn=False):
        self.accepted.append((attempt, learn))
        return {"attempt": attempt, "basis": "manual_take_selection", "learned": [("kvothe", "quote")] if learn else []}


class FakePlayer:
    def __init__(self):
        self.started, self.stopped = [], 0

    def start(self, path):
        self.started.append(path)

    def stop(self):
        self.stopped += 1


class RepairScreenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    def make_app(self, command):
        from epub_tts_narrator.renderer.progress import shell_board
        from epub_tts_narrator.run_shell import RunApp
        self.bus = EventBus()
        self.model = RunModel("render — fixture")
        self.bus.subscribe(self.model.on_event)
        self.app = RunApp(self.model, self.bus, command, title="render — fixture")
        board = shell_board(self.bus)
        board.__enter__()
        self.addCleanup(board.__exit__, None, None, None)
        return self.app

    async def wait_for(self, pilot, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Timed out waiting for the run shell")
            await pilot.pause(0.05)

    def held_render(self, repair, requested):
        """A render that failed one chapter and waits for a retry, as workflow._render does."""
        from epub_tts_narrator.renderer.progress import (
            live_board, report_book_state, report_chapter_failure, report_retry_hold)
        from epub_tts_narrator.renderer.render_chapter_flow import ChapterRetries

        def command():
            retries = ChapterRetries()
            deadline = time.monotonic() + 10

            def give_up():
                if time.monotonic() > deadline:
                    raise TimeoutError("no retry decision")
            with live_board(print) as notify:
                report_book_state(notify, "/books/fixture-book", "rendering")
                report_chapter_failure(notify, {"number": 4, "error_type": "SpeechVerificationFailed",
                                                "error": "Take 3/3 failed", "request_hash": "f" * 64},
                                       retry=lambda: retries.request(3), repair=repair)
                report_retry_hold(notify, retries.release, 1)
                requested.append(retries.wait_for_request(stop_check=give_up, poll=0.05))
                report_retry_hold(notify, None, 0)
                report_book_state(notify, "/books/fixture-book", "done")
            return 0
        return command

    async def test_listen_to_takes_accept_one_learn_and_continue_rendering(self):
        from epub_tts_narrator.run_shell_screens import FailureScreen, TakesScreen
        from epub_tts_narrator.tui_widgets import ChoiceScreen
        repair, player, requested = FakeRepair(), FakePlayer(), []
        app = self.make_app(self.held_render(repair, requested))
        app._player = player
        async with app.run_test(size=(160, 45)) as pilot:
            await self.wait_for(pilot, lambda: bool(self.model.holds))
            self.assertEqual(app.query_one("#tabs").active, "summary")
            await pilot.press("5")
            errors = app.query_one("#errors")
            errors.focus()
            await pilot.press("enter")
            self.assertIsInstance(app.screen, FailureScreen)
            self.assertEqual(app.screen.actions, ["retry", "takes", "resume_book", "close"])
            await pilot.press("down", "enter")  # "Listen to the takes"
            await self.wait_for(pilot, lambda: isinstance(app.screen, TakesScreen))
            await pilot.press("enter")
            await self.wait_for(pilot, lambda: player.started == ["/clips/take-1.wav"])
            await pilot.press("down", "a")
            self.assertIn("history echo", str(app.screen.query_one("#takes-error").render()))
            await pilot.press("up", "a")
            self.assertIsInstance(app.screen, ChoiceScreen)
            await pilot.press("enter")  # accept and teach the recognizer
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(repair.accepted, [(1, True)])
        self.assertEqual(requested, [True])
        self.assertTrue(any("the recognizer learned: kvothe → quote" in event.text for _, event in self.model.logs))
        self.assertEqual(self.model.book_states["fixture-book"]["state"], "done")

    async def test_the_summary_drills_into_a_book_and_quitting_always_asks(self):
        from epub_tts_narrator.tui_widgets import ConfirmScreen
        repair, requested = FakeRepair(), []
        app = self.make_app(self.held_render(repair, requested))
        async with app.run_test(size=(160, 45)) as pilot:
            await self.wait_for(pilot, lambda: bool(self.model.holds))
            self.assertIn("Waiting for retries", str(app.query_one("#tile-health").render()))
            books = app.query_one("#books")
            self.assertEqual(books.row_count, 1)
            from epub_tts_narrator.run_shell_screens import failure_text
            self.assertTrue(failure_text(self.model.failures[0], "x")[0].startswith("[fixture-book] Chapter 4"))
            books.focus()
            await pilot.press("enter")
            self.assertEqual((app.query_one("#tabs").active, self.model.filter.book), ("chapters", "fixture-book"))
            for key in ("ctrl+q", "q", "ctrl+c"):
                await pilot.press(key)
                self.assertIsInstance(app.screen, ConfirmScreen, key)
                await pilot.press("escape")
                self.assertTrue(self.model.running)
            await pilot.press("F", "enter")  # Enter on a stopping question keeps waiting
            self.assertTrue(self.model.holds)
            await pilot.press("q", "y")
            await self.wait_for(pilot, lambda: not app.is_running)
        self.assertEqual(requested, [False])


if __name__ == "__main__":
    unittest.main()
