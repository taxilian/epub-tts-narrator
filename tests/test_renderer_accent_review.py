"""Reviewing an accent plan: edits, model revisions, saving, the line loop and the full screen."""

import asyncio
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from textual.widgets import DataTable, Input, TextArea

from audition_fixtures import prepare_bundle
from epub_tts_narrator.renderer import accent_assignment
from epub_tts_narrator.renderer.accent_assignment import assigned_accents, guidance, read_groups
from epub_tts_narrator.renderer.accent_review import AccentController, AccentPlanner, ask_line
from epub_tts_narrator.renderer.book_context import resolve_book_config
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.config import DEFAULTS
from epub_tts_narrator.tui_widgets import ChoiceScreen, ConfirmScreen, TextEditScreen


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "last-lantern" / "bundle"


class Replies:
    """Answers requests in turn; each answer is a function of the payload. Records every request."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def run(self, system, payload, validate):
        self.calls.append((system, deepcopy(payload)))
        return validate(self.answers.pop(0)(payload))


def split(first, second, why="From the story"):
    """Voice v1 in one group, every other voice in another."""
    return lambda payload: {"g": [[first[0], first[1], why, ["v1"]],
                                  [second[0], second[1], why, [row[0] for row in payload["voices"][1:]]]]}


class PlannerCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "book"
        shutil.copytree(EXAMPLE, self.directory)
        self.bundle = prepare_bundle(load_bundle(self.directory))
        self.config = resolve_book_config(self.bundle, deepcopy(DEFAULTS))

    def planner(self, *answers, **kwargs):
        self.replies = Replies(*answers)
        return AccentPlanner(self.bundle, self.config, None, requests=self.replies, progress=lambda _: None,
                             **kwargs)


class PlannerTests(PlannerCase):
    def test_a_plan_can_be_edited_by_hand_and_nothing_is_saved_until_asked(self):
        planner = self.planner(split(("the harbor", "Scottish"), ("inland", "Welsh")))
        planner.propose()
        first, *others = planner.pending
        self.assertEqual(planner.accents()[first], "Scottish accent")
        planner.set_accent(0, "broad Yorkshire")
        planner.set_origin(0, "the fishing town")
        planner.set_why(0, "  Fishers   from the coast ")
        self.assertEqual(planner.groups[0]["why"], "Fishers from the coast")
        for bad in ("Mara", "heavy", ""):
            with self.subTest(accent=bad), self.assertRaises(ValueError):
                planner.set_accent(0, bad)
        self.assertFalse((self.directory / accent_assignment.ASSIGNMENTS_FILENAME).exists())
        planner.save()
        self.assertEqual(assigned_accents(self.bundle, self.config)[first], "broad Yorkshire accent")
        self.assertEqual(read_groups(self.directory, self.bundle["source"]["book"]["id"])[0]["origin"],
                         "the fishing town")

    def test_moving_a_character_to_another_group_or_a_new_one(self):
        planner = self.planner(split(("the harbor", "Scottish"), ("inland", "Welsh")))
        planner.propose()
        first, second = planner.pending[:2]
        planner.move(first, 1)
        self.assertEqual(len(planner.groups), 1)  # the emptied group is gone
        self.assertEqual(planner.accents()[first], "Welsh accent")
        planner.move(second, ("the mountains", "Irish"))
        self.assertEqual(planner.groups[-1]["names"], [planner.label(second)])
        self.assertEqual(planner.accents()[second], "Irish accent")
        with self.assertRaises(ValueError):
            planner.move(second, ("the mountains", "Mara"))

    def test_instructions_ask_the_model_to_revise_the_current_plan_and_become_guidance(self):
        planner = self.planner(split(("the harbor", "Scottish"), ("inland", "Welsh")),
                               split(("the harbor", "Irish"), ("inland", "Welsh"), "Told to"))
        planner.propose()
        planner.revise("Harbor folk should sound Irish")
        system, payload = self.replies.calls[1]
        self.assertIn('"revise"', system)
        self.assertEqual(payload["revise"], "Harbor folk should sound Irish")
        self.assertEqual(payload["plan"][0][:2], ["the harbor", "Scottish accent"])
        self.assertEqual(planner.accents()[planner.pending[0]], "Irish accent")
        planner.save()
        self.assertEqual(guidance(self.bundle, self.config), "Harbor folk should sound Irish")
        # A first request never carries revision wording.
        self.assertNotIn('"revise"', self.replies.calls[0][0])

    def test_an_origin_the_series_decided_keeps_its_accent(self):
        planner = self.planner(split(("the harbor", "Scottish"), ("inland", "Welsh")))
        planner.series = {"the harbor": "Scottish accent"}
        planner.propose()
        with self.assertRaisesRegex(ValueError, "already has"):
            planner.set_accent(0, "Irish")
        planner.set_origin(0, "the far harbor")
        planner.set_accent(0, "Irish")


class SavedPlanTests(PlannerCase):
    """A saved plan reopens for editing; the model is asked only when a person asks."""

    def save_plan(self):
        planner = self.planner(split(("the harbor", "Scottish"), ("inland", "Welsh")))
        planner.propose()
        planner.set_why(0, "Fishers")
        planner.save()
        return planner

    def test_the_saved_plan_reopens_without_a_request(self):
        saved = self.save_plan()
        reopened = self.planner()  # no replies: any request would fail
        self.assertTrue(reopened.load_saved())
        self.assertEqual(reopened.groups, saved.groups)
        self.assertEqual((reopened.dirty, reopened.unassigned()), (False, []))
        self.assertEqual(self.replies.calls, [])

    def test_a_character_added_since_is_left_without_an_accent(self):
        self.save_plan()
        newcomer = self.bundle["cast"]["voices"][0]["id"]
        groups = read_groups(self.directory, self.bundle["source"]["book"]["id"])
        for group in groups:
            if newcomer in group["voice_ids"]:
                position = group["voice_ids"].index(newcomer)
                del group["voice_ids"][position], group["names"][position]
        accent_assignment.save_groups(self.bundle, [g for g in groups if g["voice_ids"]], replace=True)
        accent_assignment.save_assignments(self.bundle, self.config, {newcomer: None})
        reopened = self.planner()
        self.assertTrue(reopened.load_saved())
        self.assertEqual(reopened.unassigned(), [newcomer])

    def test_accents_saved_before_groups_were_recorded_show_one_group_per_accent(self):
        first, *others = self.planner().pending
        accent_assignment.save_assignments(self.bundle, self.config, {first: "Welsh accent",
                                                                      **{v: "Irish accent" for v in others}})
        reopened = self.planner()
        self.assertTrue(reopened.load_saved())
        self.assertEqual(sorted((g["accent"], g["origin"]) for g in reopened.groups),
                         [("Irish accent", "(origin not recorded)"), ("Welsh accent", "(origin not recorded)")])

    def test_the_command_reports_a_saved_plan_without_asking_and_revises_only_when_told(self):
        self.save_plan()
        args = SimpleNamespace(instructions=None, apply=False, resume=None, keep=False, plain=True)
        silent = Replies()
        output = io.StringIO()
        with patch.object(accent_assignment, "_requests", return_value=silent), \
                patch.object(accent_assignment, "_book_cache", return_value=self.directory / ".cache"), \
                patch("sys.stdout", output), patch("sys.stderr", io.StringIO()):
            self.assertEqual(accent_assignment.run_accents_command(self.bundle, self.config, args, None), 0)
        self.assertEqual(silent.calls, [])
        self.assertIn("saved plan", output.getvalue())
        self.assertIn("why: Fishers", output.getvalue())
        revising = Replies(split(("the harbor", "Irish"), ("inland", "Welsh")))
        args = SimpleNamespace(instructions="Harbor folk are Irish", apply=True, resume=None, keep=False, plain=True)
        with patch.object(accent_assignment, "_requests", return_value=revising), \
                patch.object(accent_assignment, "_book_cache", return_value=self.directory / ".cache"), \
                patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            accent_assignment.run_accents_command(self.bundle, self.config, args, None)
        self.assertEqual(revising.calls[0][1]["revise"], "Harbor folk are Irish")
        self.assertEqual(revising.calls[0][1]["plan"][0][2], "Fishers")


class LineTests(PlannerCase):
    def test_the_line_choice_reads_save_leave_or_instructions(self):
        for answers, expected in (([" a\n"], ("save", None)), (["L\n"], ("leave", None)),
                                  (["i make them Irish\n"], ("revise", "make them Irish")),
                                  (["I\n", "make them Irish\n"], ("revise", "make them Irish")),
                                  (["?\n", "a\n"], ("save", None)), (["n\n"], ("new", None)),
                                  ([""], ("leave", None))):
            with self.subTest(answers=answers), patch("sys.stdout", io.StringIO()):
                lines = iter(answers)
                self.assertEqual(ask_line(lambda: next(lines), write=lambda _: None), expected)

    def test_a_terminal_can_revise_then_save(self):
        replies = Replies(split(("the harbor", "Scottish"), ("inland", "Welsh")),
                          split(("the harbor", "Irish"), ("inland", "Welsh")))
        answers = iter(["i harbor folk are Irish\n", "a\n"])
        terminal = SimpleNamespace(readline=lambda: next(answers), isatty=lambda: True)
        output = io.StringIO()
        output.isatty = lambda: True
        args = SimpleNamespace(instructions=None, apply=False, resume=None, keep=False, plain=True)
        with patch.object(accent_assignment, "_requests", return_value=replies), \
                patch.object(accent_assignment, "_book_cache", return_value=self.directory / ".cache"), \
                patch("sys.stdin", terminal), patch("sys.stdout", output), patch("sys.stderr", io.StringIO()):
            self.assertEqual(accent_assignment.run_accents_command(self.bundle, self.config, args, None), 0)
        self.assertEqual(replies.calls[1][1]["revise"], "harbor folk are Irish")
        first = next(iter(accent_assignment.voice_entries(self.bundle)))
        self.assertEqual(assigned_accents(self.bundle, self.config)[first], "Irish accent")
        self.assertIn("Saved accents", output.getvalue())


class AccentAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        case = PlannerCase()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.case = case
        self.planner = case.planner(split(("the harbor", "Scottish"), ("inland", "Welsh")),
                                    split(("the harbor", "Irish"), ("inland", "Welsh"), "Told to"))
        self.controller = AccentController(self.planner, describe_error=str)
        self.addCleanup(self.controller.close)
        from epub_tts_narrator.renderer.accent_tui import AccentApp
        self.app = AccentApp(self.controller, title="Fixture Book")

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def settle(self, pilot):
        self.controller.jobs.wait_idle(5)
        await pilot.pause()
        await pilot.pause()

    async def test_the_plan_opens_edits_revises_and_saves(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.settle(pilot)
            groups = self.app.query_one("#groups", DataTable)
            self.assertEqual(groups.row_count, 2)
            self.assertIn("From the story", str(self.app.query_one("#detail").render()))
            # Edit the accent by hand; an invented accent stays in the dialog with its reason.
            await pilot.press("a")
            self.assertIsInstance(self.app.screen, TextEditScreen)
            editor = self.app.screen.query_one("#editor", Input)
            editor.value = "Mara"
            await pilot.press("enter")
            self.assertIsInstance(self.app.screen, TextEditScreen)
            editor.value = "soft Irish"
            await pilot.press("enter")
            await pilot.pause()
            self.assertEqual(self.planner.groups[0]["accent"], "soft Irish accent")
            # Instructions go back to the model, which revises the whole plan.
            await pilot.press("i")
            self.app.screen.query_one("#editor", TextArea).text = "Harbor folk are plain Irish"
            await pilot.press("ctrl+s")
            await self.settle(pilot)
            self.assertEqual(self.case.replies.calls[1][1]["revise"], "Harbor folk are plain Irish")
            self.assertEqual(self.planner.groups[0]["why"], "Told to")
            # Saving asks first.
            await pilot.press("s")
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("y")
            await pilot.pause()
            self.assertTrue(self.planner.saved)
            first = self.planner.pending[0]
            self.assertEqual(assigned_accents(self.case.bundle, self.case.config)[first], "Irish accent")

    async def test_a_saved_plan_opens_without_asking_the_model(self):
        self.planner.propose()
        self.planner.save()
        reopened = self.case.planner()  # no replies: any request would fail
        reopened.load_saved()
        controller = AccentController(reopened, describe_error=str)
        self.addCleanup(controller.close)
        from epub_tts_narrator.renderer.accent_tui import AccentApp
        app = AccentApp(controller, title="Fixture Book")
        async with app.run_test(size=(160, 45)) as pilot:
            controller.jobs.wait_idle(5)
            await pilot.pause()
            self.assertEqual(app.query_one("#groups", DataTable).row_count, 2)
            self.assertEqual(controller.jobs.jobs(), [])
            self.assertIn("saved", app.sub_title)

    async def test_moving_a_character_and_quitting_unsaved_asks_first(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.settle(pilot)
            first = self.planner.pending[0]
            await pilot.press("m")
            self.assertIsInstance(self.app.screen, ChoiceScreen)
            self.app.screen.dismiss(first)
            await pilot.pause()
            self.assertIsInstance(self.app.screen, ChoiceScreen)
            self.app.screen.dismiss(1)
            await pilot.pause()
            self.assertEqual(self.planner.accents()[first], "Welsh accent")
            await pilot.press("q")
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            self.assertFalse(self.planner.saved)


if __name__ == "__main__":
    unittest.main()
