"""Revising voice descriptions from a person's instructions: the request, the cast controller, and the screen."""

import asyncio
from copy import deepcopy
import threading
import unittest

from epub_tts_narrator.renderer.cast_tui_model import CastController
from epub_tts_narrator.renderer.description_revision import BATCH, revise
from epub_tts_narrator.renderer.description_revision_prompts import REVISE
from cast_tui_fixtures import FakePlayer, FakeSession


class FakeRequests:
    """Answers with ``reply(payload)``; records each request."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def run(self, system, payload, validate):
        self.calls.append((system, deepcopy(payload)))
        return validate(self.reply(payload))


def without_accent(payload):
    """Drops "with a ... accent" from every description that has one, as a model following the instructions would."""
    rows = []
    for short, _name, description, _accent in payload["voices"]:
        if " with a " in description:
            rows.append([short, description.split(" with a ")[0] + "."])
    return {"d": rows}


VOICES = [{"voice_id": "v-ava", "name": "Ava", "description": "A calm alto with a soft Irish accent.",
           "accent": "Yorkshire accent"},
          {"voice_id": "v-ben", "name": "Ben", "description": "A dry tenor.", "accent": ""}]


class ReviseTests(unittest.TestCase):
    def test_each_voice_goes_with_its_accent_cue_and_only_changed_descriptions_return(self):
        requests = FakeRequests(without_accent)
        result = revise(requests, VOICES, "Remove accent wording; the cue adds it")
        self.assertEqual(result, {"v-ava": "A calm alto."})
        system, payload = requests.calls[0]
        self.assertEqual(system, REVISE)
        self.assertEqual(payload["instructions"], "Remove accent wording; the cue adds it")
        self.assertEqual(payload["voices"][0], ["v1", "Ava", "A calm alto with a soft Irish accent.",
                                                "Yorkshire accent"])

    def test_a_reply_returning_the_same_words_is_not_a_change(self):
        requests = FakeRequests(lambda payload: {"d": [["v2", "A  dry tenor."]]})
        self.assertEqual(revise(requests, VOICES, "anything"), {})

    def test_markup_unknown_ids_and_empty_instructions_are_refused(self):
        with self.assertRaisesRegex(ValueError, "brackets or markup"):
            revise(FakeRequests(lambda payload: {"d": [["v1", "[calm] An alto."]]}), VOICES, "x")
        with self.assertRaisesRegex(ValueError, "id from"):
            revise(FakeRequests(lambda payload: {"d": [["v9", "An alto."]]}), VOICES, "x")
        with self.assertRaisesRegex(ValueError, "Type what should change"):
            revise(FakeRequests(without_accent), VOICES, "  ")

    def test_a_large_cast_is_sent_in_batches(self):
        voices = [{"voice_id": f"v-{n}", "name": f"N{n}", "description": f"Voice {n} with a Welsh lilt.",
                   "accent": ""} for n in range(BATCH + 3)]
        requests = FakeRequests(without_accent)
        result = revise(requests, voices, "No accents")
        self.assertEqual(len(requests.calls), 2)
        self.assertEqual(len(result), BATCH + 3)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.controller = CastController(self.session, FakePlayer(), describe_error=str)
        self.addCleanup(self.controller.close)
        self.controller.refresh()
        self.requests = FakeRequests(lambda payload: {"d": [[row[0], row[2].replace("calm", "low")]
                                                            for row in payload["voices"]]})
        self.controller.requests_factory = lambda: self.requests

    def test_revising_proposes_without_saving_and_saving_writes_drafts(self):
        got = []
        rows = [self.controller.row("v-ava"), self.controller.row("v-ava-old")]  # the clone is left out
        self.controller.revise_descriptions(rows, "Lower voices", got.append)
        self.assertTrue(self.controller.jobs.wait_idle(5))
        self.assertEqual(got, [{"v-ava": ("A calm voice for Ava.", "A low voice for Ava.")}])
        self.assertEqual(len(self.requests.calls[0][1]["voices"]), 1)
        self.assertEqual(self.session.calls, [])  # nothing saved yet
        self.controller.save_descriptions({"v-ava": "A low voice for Ava."})
        self.assertTrue(self.controller.jobs.wait_idle(5))
        self.assertEqual(self.session.calls, [("update_prompt", ("v-ava",), {"prompt": "A low voice for Ava."})])

    def test_asking_the_model_does_not_wait_for_a_generation(self):
        gate = threading.Event()
        self.session.regenerate = lambda voice_id, voice_designer=None: gate.wait(5)
        self.controller.regenerate(self.controller.row("v-ben"), "local")
        ready = threading.Event()
        self.controller.revise_descriptions([self.controller.row("v-ava")], "Lower", lambda changes: ready.set())
        self.assertTrue(ready.wait(2))
        gate.set()
        self.assertTrue(self.controller.jobs.wait_idle(5))


class ScreenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = FakeSession()
        self.controller = CastController(self.session, FakePlayer(), describe_error=str)
        self.addCleanup(self.controller.close)
        self.controller.requests_factory = lambda: FakeRequests(
            lambda payload: {"d": [[row[0], row[2].replace("calm", "low")] for row in payload["voices"]]})

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def test_instructions_lead_to_a_reviewed_save_of_the_marked_voices(self):
        from textual.widgets import SelectionList, TextArea
        from epub_tts_narrator.renderer.cast_tui import CastApp, RevisionScreen
        app = CastApp(self.controller, title="Fixture Book", startup=False)
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause()
            await pilot.press("i")
            await pilot.pause()
            await pilot.press("down", "enter")  # every voice without an accepted take
            await pilot.pause()
            app.screen.query_one("#editor", TextArea).text = "Make them lower"
            await pilot.press("ctrl+s")
            for _ in range(100):
                if isinstance(app.screen, RevisionScreen):
                    break
                await pilot.pause(0.05)
            self.assertIsInstance(app.screen, RevisionScreen)
            proposed = list(app.screen.changes)
            self.assertGreaterEqual(len(proposed), 2)
            self.assertIn("low", str(app.screen.query_one("#revision-detail").render()))
            await pilot.press("space")  # unmark the first voice
            await pilot.press("enter")
            await pilot.pause()
            self.controller.jobs.wait_idle(5)
        saved = [call[1][0] for call in self.session.calls if call[0] == "update_prompt"]
        self.assertEqual(saved, proposed[1:])


if __name__ == "__main__":
    unittest.main()
