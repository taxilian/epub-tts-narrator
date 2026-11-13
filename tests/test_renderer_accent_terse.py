"""Terse accent names, and a model request on the accent screen that cannot be missed."""

import asyncio
import threading
import unittest

from epub_tts_narrator.renderer.accent_assignment import request_assignment, terse_accent
import test_renderer_accent_assignment as assignment_tests
import test_renderer_accent_review as review_tests


class TerseAccentTests(unittest.TestCase):
    def test_new_accents_use_the_short_house_style(self):
        for given, expected in (("Heavy Russian accent", "heavy Russian accent"),
                                ("German-accented English accent", "German accent"),
                                ("Utah Accent", "Utah accent"), ("upper-class English accent", "upper-class English accent"),
                                ("Scottish accent", "Scottish accent")):
            self.assertEqual(terse_accent(given), expected)


class TerseRepliesTests(assignment_tests.BookTestCase):
    def reply(self, accent, origin="the harbor town"):
        return assignment_tests.FakeRequests(
            lambda payload: {"g": [[origin, accent, "They are islanders", [row[0] for row in payload["voices"]]]]})

    def test_a_reply_is_made_terse(self):
        result = request_assignment(self.bundle, self.reply("Heavy Russian"), self.voices, {})
        self.assertEqual(set(result["accents"].values()), {"heavy Russian accent"})

    def test_an_accent_decided_earlier_keeps_its_exact_words(self):
        origins = [{"origin": "the harbor town", "accent": "Broad Scottish accent", "why": "", "names": ["Old Tam"]}]
        result = request_assignment(self.bundle, self.reply("broad Scottish"), self.voices, {}, origins=origins)
        self.assertEqual(set(result["accents"].values()), {"Broad Scottish accent"})
        keep = {voice_id: "German-accented English accent" for voice_id in self.voices}
        result = request_assignment(self.bundle, self.reply("German"), self.voices, {}, keep=keep)
        self.assertEqual(set(result["accents"].values()), {"German-accented English accent"})


class BusyBannerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.accent_review import AccentController
        case = review_tests.PlannerCase()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.release = threading.Event()

        def slow(payload):
            self.release.wait(10)
            return review_tests.split(("the harbor", "Scottish"), ("inland", "Welsh"))(payload)
        self.planner = case.planner(slow)
        self.controller = AccentController(self.planner, describe_error=str)
        self.addCleanup(self.controller.close)
        self.addCleanup(self.release.set)

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def test_a_running_request_shows_a_banner_and_the_plan_replaces_it(self):
        from textual.widgets import DataTable, Static
        from epub_tts_narrator.renderer.accent_tui import AccentApp
        app = AccentApp(self.controller, title="Fixture Book")
        async with app.run_test(size=(160, 45)) as pilot:
            await pilot.pause(0.6)
            banner = app.query_one("#busy", Static)
            self.assertTrue(banner.has_class("running"))
            self.assertIn("Asking the model: Ask for an accent plan", str(banner.render()))
            self.assertTrue(app.query_one("#groups", DataTable).loading)
            self.release.set()
            self.controller.jobs.wait_idle(5)
            await pilot.pause(0.6)
            self.assertFalse(banner.has_class("running"))
            groups = app.query_one("#groups", DataTable)
            self.assertFalse(groups.loading)
            self.assertEqual(str(groups.get_row_at(0)[0]), "Scottish")  # shown without "accent"


if __name__ == "__main__":
    unittest.main()
