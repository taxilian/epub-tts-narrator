"""The accent list: model replies come from it, and the accent screen offers it with each accent's close neighbours."""

import asyncio
import unittest

from epub_tts_narrator.renderer import accent_catalog as catalog
from epub_tts_narrator.renderer.accent_assignment import accent_cue, request_assignment, terse_accent
from epub_tts_narrator.renderer.accent_prompts import ASSIGN
import test_renderer_accent_assignment as assignment_tests
import test_renderer_accent_review as review_tests


class CatalogTests(unittest.TestCase):
    def test_every_listed_accent_is_a_valid_cue_already_in_house_style(self):
        for name in catalog.ACCENTS:
            cue = terse_accent(accent_cue(name))
            self.assertEqual(catalog.canonical(cue), cue)
        for alias, name in catalog.ALIASES.items():
            self.assertIn(name, catalog.ACCENTS, alias)

    def test_other_words_strength_and_case_come_back_in_the_lists_words(self):
        self.assertEqual(catalog.canonical("Broad glasgow accent"), "broad Glaswegian accent")
        self.assertEqual(catalog.canonical("Liverpool accent"), "Scouse accent")
        self.assertEqual(catalog.canonical("mid - atlantic accent"), "Mid-Atlantic accent")
        self.assertEqual(catalog.canonical("Klingon accent"), "Klingon accent")  # unlisted: unchanged
        self.assertIsNone(catalog.listed("heavy"))

    def test_one_family_or_a_link_is_close_and_far_families_are_not(self):
        self.assertEqual(catalog.relation("Scottish accent", "heavy Glaswegian accent"), "near")
        self.assertEqual(catalog.relation("Northern Irish", "Scottish"), "near")  # linked across families
        self.assertEqual(catalog.relation("broad Welsh accent", "Welsh"), "same")
        self.assertIsNone(catalog.relation("Texan", "Russian"))
        self.assertEqual(catalog.near("Scottish")[0], "Glaswegian")
        self.assertIn("Northern Irish", catalog.near("Scottish"))

    def test_the_prompt_carries_every_family_and_starred_rare_accents(self):
        for family, accents in catalog.FAMILIES:
            self.assertIn(f"{family}: {', '.join(accents)}", ASSIGN)
        self.assertIn("Doric*", ASSIGN)
        self.assertIn("Mid-Atlantic ≈ upper-class English", ASSIGN)


class ListedRepliesTests(assignment_tests.BookTestCase):
    def reply(self, accent):
        return assignment_tests.FakeRequests(
            lambda payload: {"g": [["the harbor town", accent, "Islanders", [row[0] for row in payload["voices"]]]]})

    def test_an_accent_off_the_list_is_refused_so_the_model_repairs_it(self):
        with self.assertRaisesRegex(ValueError, "not on the ACCENTS list"):
            request_assignment(self.bundle, self.reply("Patagonian"), self.voices, {})

    def test_another_word_for_a_listed_accent_is_saved_in_the_lists_words(self):
        result = request_assignment(self.bundle, self.reply("Broad Liverpool"), self.voices, {})
        self.assertEqual(set(result["accents"].values()), {"broad Scouse accent"})

    def test_an_unlisted_accent_the_listener_named_or_the_series_decided_is_allowed(self):
        result = request_assignment(self.bundle, self.reply("Patagonian"), self.voices, {},
                                    notes="The islanders should sound Patagonian.")
        self.assertEqual(set(result["accents"].values()), {"Patagonian accent"})
        origins = [{"origin": "the harbor town", "accent": "Patagonian accent", "why": "", "names": ["Old Tam"]}]
        result = request_assignment(self.bundle, self.reply("Patagonian"), self.voices, {}, origins=origins)
        self.assertEqual(set(result["accents"].values()), {"Patagonian accent"})


def close_plan(payload):
    voices = [row[0] for row in payload["voices"]]  # the fixture book has two speaking voices
    return {"g": [["the harbor", "Scottish", "Coast folk", voices[:1]],
                  ["the hills", "Glaswegian", "Hill folk", voices[1:]]]}


class NeighbourTests(review_tests.PlannerCase):
    def test_groups_with_one_familys_accents_are_close_and_typed_names_are_listed_words(self):
        planner = self.planner(close_plan)
        planner.propose()
        self.assertEqual([(relation, group["origin"]) for relation, group in planner.neighbours("0")],
                         [("near", "the hills")])
        planner.set_accent(1, "scots")
        self.assertEqual(planner.groups[1]["accent"], "Scottish accent")
        self.assertEqual(planner.neighbours("1")[0][0], "same")
        self.assertEqual(planner.used_by()["scottish"], ["the harbor", "the hills"])
        planner.set_accent(1, "Texan")
        self.assertEqual(planner.neighbours("0"), [])


class PickerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from epub_tts_narrator.renderer.accent_review import AccentController
        case = review_tests.PlannerCase()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.controller = AccentController(case.planner(close_plan), describe_error=str)
        self.addCleanup(self.controller.close)

    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def open(self, pilot, app):
        self.controller.jobs.wait_idle(5)
        await pilot.pause(0.4)

    async def test_choosing_from_the_list_shows_close_groups_and_typing_your_own_keeps_a_strength(self):
        from textual.widgets import DataTable, Static
        from epub_tts_narrator.renderer.accent_tui import AccentApp, AccentPickerScreen
        app = AccentApp(self.controller, title="Fixture Book")
        async with app.run_test(size=(160, 45)) as pilot:
            await self.open(pilot, app)
            detail = str(app.query_one("#detail", Static).render())
            self.assertIn("Close to: Glaswegian (the hills)", detail)
            self.assertIn("Similar accents (a chooses): Glaswegian (the hills), Edinburgh", detail)
            await pilot.press("a")
            await pilot.pause(0.2)
            self.assertIsInstance(app.screen, AccentPickerScreen)
            await pilot.press(*"edinb", "enter")
            await pilot.pause(0.2)
            self.assertEqual(self.controller.planner.groups[0]["accent"], "Edinburgh accent")
            await pilot.press("a", *"heavy edinburgh")  # matches nothing listed: only "type your own" remains
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.2)
            self.assertEqual(self.controller.planner.groups[0]["accent"], "heavy Edinburgh accent")
            self.assertEqual(str(app.query_one("#groups", DataTable).get_row_at(0)[0]), "heavy Edinburgh ≈")


if __name__ == "__main__":
    unittest.main()
