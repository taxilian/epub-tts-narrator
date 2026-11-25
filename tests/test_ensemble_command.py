"""A person sets up, resizes or declines a group's ensemble; it is published like any analysis revision."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.bundle import chunk_speakers, load_bundle, make_plan
from epub_tts_narrator.renderer.ensemble_command import apply, plan_decline, plan_ensemble, view
from epub_tts_narrator.series import link_analysis
from shared_fixtures import renderer_config
from test_character_merge import AUDITION, build

PARAGRAPHS = [
    [("Mara waved. ", None), ("“Hello,” ", "mara"), ("she said.", None)],
    [("“We are here,” ", "chorus"), ("they said in unison.", None)],
]


class EnsembleCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name).resolve() / "book"
        document = build("chorus-book", "The Chorus", [("mara", "Mara", ()), ("chorus", "the chorus", ())],
                         PARAGRAPHS)
        chorus = next(item for item in document["cast"]["characters"] if item["id"] == "chorus")
        chorus["group"] = {"size": None, "scale": "several", "members": [], "proposed_voices": 4,
                           "proposed_mode": "unison", "basis": "analysis",
                           "evidence": [{"block_id": "b2", "quote": "they said in unison"}]}
        linked = link_analysis(document, fresh_registry(self.path))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(self.path), self.path, export_bundle)
        names = {item["name"]: item for item in linked["cast"]["characters"]}
        self.chorus, self.mara = names["the chorus"]["voice_id"], names["Mara"]["voice_id"]

    def saved(self):
        return _completed_directory(self.path)[0]

    def lead(self, document):
        return next(voice for voice in document["cast"]["voices"] if voice["id"] == self.chorus)

    def group_chunk(self):
        plan = make_plan(load_bundle(self.path), renderer_config())
        return next(chunk for chapter in plan["chapters"] for chunk in chapter["chunks"]
                    if chunk["segment"].get("voice_id") == self.chorus)

    def test_the_screen_starts_from_the_proposal_and_nothing_changes_until_saved(self):
        shown = view(load_bundle(self.path), self.chorus)
        self.assertEqual((shown["voices"], shown["mode"], shown["ensemble"]), (4, "unison", None))
        self.assertEqual(shown["evidence"], ["they said in unison"])
        plan_ensemble(load_bundle(self.path), self.chorus, voices=4, mode="unison")
        self.assertNotIn("ensemble", self.lead(self.saved()))

    def test_a_saved_ensemble_adds_separately_described_pool_voices_and_renders_them(self):
        apply(plan_ensemble(load_bundle(self.path), self.chorus, voices=4, mode="unison"), progress=lambda _: None)
        document = self.saved()
        self.assertEqual(validate_document(document), [])
        members = self.lead(document)["ensemble"]["members"]
        self.assertEqual(len(members), 3)
        pool = [voice for voice in document["cast"]["voices"] if voice.get("member_of") == self.chorus]
        self.assertEqual([voice["id"] for voice in pool], members)
        self.assertEqual(len({voice["description"] for voice in pool}), 3, "each is designed on its own")
        self.assertEqual([speaker["voice_id"] for speaker in chunk_speakers(self.group_chunk())],
                         [self.chorus, *members])

    def test_resizing_keeps_existing_pool_voices_and_drops_only_the_extra_ones(self):
        apply(plan_ensemble(load_bundle(self.path), self.chorus, voices=4, mode="unison"), progress=lambda _: None)
        first = self.lead(self.saved())["ensemble"]["members"]
        apply(plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="staggered", mix="swell"),
              progress=lambda _: None)
        document = self.saved()
        self.assertEqual(self.lead(document)["ensemble"], {"members": first[:2], "mode": "staggered",
                                                           "mix": "swell", "basis": "user"})
        self.assertNotIn(first[2], {voice["id"] for voice in document["cast"]["voices"]})

    def test_a_plan_names_only_the_voices_it_adds(self):
        first = plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="pool")
        self.assertEqual(first.added, (f"{self.chorus}-pool-2", f"{self.chorus}-pool-3"))
        apply(first, progress=lambda _: None)
        self.assertEqual(plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="unison").added, ())
        self.assertEqual(plan_ensemble(load_bundle(self.path), self.chorus, voices=4, mode="pool").added,
                         (f"{self.chorus}-pool-4",))

    def test_a_named_character_can_be_one_of_the_voices(self):
        apply(plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="alternate", named=[self.mara]),
              progress=lambda _: None)
        members = self.lead(self.saved())["ensemble"]["members"]
        self.assertEqual(members[0], self.mara)
        self.assertEqual(len(members), 2)

    def test_declining_keeps_the_groups_own_voice_and_is_remembered(self):
        apply(plan_ensemble(load_bundle(self.path), self.chorus, voices=4, mode="unison"), progress=lambda _: None)
        apply(plan_decline(load_bundle(self.path), self.chorus), progress=lambda _: None)
        document = self.saved()
        self.assertEqual(self.lead(document)["ensemble"], {"members": [], "mode": "single", "basis": "user"})
        self.assertFalse(any(voice.get("member_of") for voice in document["cast"]["voices"]))
        self.assertEqual([speaker["voice_id"] for speaker in chunk_speakers(self.group_chunk())], [self.chorus])
        plan = make_plan(load_bundle(self.path), renderer_config())
        self.assertEqual(plan["collective_voices"]["proposed"], [])

    def test_a_plan_is_refused_when_the_book_changed_after_it_was_made(self):
        plan = plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="unison")
        apply(plan_decline(load_bundle(self.path), self.chorus), progress=lambda _: None)
        with self.assertRaisesRegex(ValueError, "changed"):
            apply(plan, progress=lambda _: None)

    def test_only_a_characters_own_voice_and_never_the_narrator_leads(self):
        narrator = next(role["voice_id"] for role in self.saved()["cast"]["roles"] if role["kind"] == "narration")
        with self.assertRaises(ValueError):
            plan_ensemble(load_bundle(self.path), narrator, voices=3, mode="unison")
        with self.assertRaisesRegex(ValueError, "2 to 8"):
            plan_ensemble(load_bundle(self.path), self.chorus, voices=9, mode="unison")

    def test_a_revision_that_changes_anything_else_is_refused(self):
        from epub_tts_narrator.analysis_revision import revise_directories
        plan = plan_ensemble(load_bundle(self.path), self.chorus, voices=3, mode="unison")
        plan.candidate["segments"][1]["delivery"]["direction"] = "loud"
        with self.assertRaisesRegex(ValueError, "preserve"):
            revise_directories({plan.path: plan.candidate}, scope="ensembles", progress=lambda _: None)



class EnsembleSeriesTests(unittest.TestCase):
    def test_a_later_book_keeps_an_earlier_books_ensemble_with_its_pool_voices(self):
        from epub_tts_narrator.analysis_directory import continuation_registry
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        first = root / "book-1"
        document = build("chorus-1", "The Chorus", [("mara", "Mara", ()), ("chorus", "the chorus", ())], PARAGRAPHS)
        linked = link_analysis(document, fresh_registry(first))
        linked["provenance"]["voice_samples"] = {
            "version": 1, "source_fingerprint": fingerprint(linked["source"]),
            "cast_fingerprint": fingerprint(linked["cast"]),
            "scripts": {voice["id"]: deepcopy(AUDITION) for voice in linked["cast"]["voices"]}, "samples": {}}
        publish_directory(linked, fresh_registry(first), first, export_bundle)
        chorus = next(item for item in linked["cast"]["characters"] if item["name"] == "the chorus")
        apply(plan_ensemble(load_bundle(first), chorus["voice_id"], voices=3, mode="unison"), progress=lambda _: None)
        pool = [voice["id"] for voice in _completed_directory(first)[0]["cast"]["voices"] if voice.get("member_of")]

        second = build("chorus-2", "The Chorus Returns", [("chorus", "the chorus", ())],
                       [[("“We return,” ", "chorus"), ("they said in unison.", None)]])
        registry = continuation_registry(first, second["source"])
        second["cast"]["characters"][0]["series_identity"] = {
            "character_id": chorus["id"], "basis": "context", "review_required": False,
            "reason": "The same chorus returns.",
            "evidence": [{"block_id": "b1", "quote": second["source"]["blocks"][0]["text"]}]}
        two = link_analysis(second, registry)
        self.assertEqual(validate_document(two), [])
        lead = next(voice for voice in two["cast"]["voices"] if voice["id"] == chorus["voice_id"])
        self.assertEqual(lead["ensemble"]["members"], pool)
        self.assertTrue(set(pool) <= {voice["id"] for voice in two["cast"]["voices"]})


class EnsembleDetailsTests(unittest.TestCase):
    def test_the_details_say_what_a_person_chose_or_what_analysis_proposed(self):
        from epub_tts_narrator.renderer.cast_tui_model import ensemble_text
        rows = {"v-chorus": {"voice_id": "v-chorus", "label": "the chorus"},
                "v-pool": {"voice_id": "v-pool", "label": "the chorus, voice 2"}}
        proposal = {"proposed_voices": 4, "proposed_mode": "unison",
                    "evidence": [{"block_id": "b2", "quote": "they said in unison"}]}
        self.assertEqual(ensemble_text({"group": proposal}, rows),
                         "Proposed as a group of 4 voices (all together, the same words at once) from "
                         "“they said in unison”; speaks in one voice until you press C")
        chosen = {"members": ["v-pool"], "mode": "staggered", "mix": "swell", "basis": "user"}
        self.assertEqual(ensemble_text({"group": proposal, "ensemble": chosen}, rows),
                         "2 voices, the same words, voices entering one after another, mixed swell: "
                         "this voice, the chorus, voice 2")
        self.assertTrue(ensemble_text({"member_of": "v-chorus"}, rows).startswith("One of the voices of the chorus."))
        self.assertEqual(ensemble_text({"ensemble": {"members": [], "mode": "single", "basis": "user"}}, rows),
                         "Kept as one voice (you chose; C changes it)")
        self.assertEqual(ensemble_text({}, rows), "")


class NewVoiceDescriptionTests(unittest.TestCase):
    """Saving a group's new voices has the model describe them apart; without a model the simple ones stay."""

    def setUp(self):
        from types import SimpleNamespace
        from cast_tui_fixtures import FakePlayer, FakeSession
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        self.controller = CastController(FakeSession(), FakePlayer(), describe_error=str)
        self.addCleanup(self.controller.close)
        self.refreshed, self.closed, self.fail = [], [], None
        test = self

        class Session:
            def refresh_prompts(self, *, voice_ids, config_path=None):
                if test.fail:
                    raise test.fail
                test.refreshed.append(voice_ids)

            def close(self):
                test.closed.append(True)
        self.controller.session_for = lambda directory: Session()
        self.plan = SimpleNamespace(name="the chorus", summary="the chorus: 3 voices", path="book", hashes={},
                                    candidate={}, added=("v-pool-2", "v-pool-3"))

    def save(self):
        from unittest.mock import patch
        with patch("epub_tts_narrator.renderer.ensemble_command.apply"):
            self.controller.save_ensemble(self.plan)
            self.controller.jobs.wait_idle(5)
        return " ".join(text for _, _, text in self.controller.messages)

    def test_new_voices_are_described_by_the_model_once_saved(self):
        log = self.save()
        self.assertEqual(self.refreshed, [["v-pool-2", "v-pool-3"]])
        self.assertEqual(self.closed, [True])
        self.assertIn("Described 2 new voice(s) for the chorus so they sound apart", log)
        self.assertTrue(self.controller.reopen)

    def test_without_a_model_the_group_is_still_saved_with_simple_descriptions(self):
        self.fail = ValueError("Voice prompt refresh needs the configured analysis model")
        log = self.save()
        self.assertIn("Kept simple descriptions for the chorus's new voices", log)
        self.assertTrue(self.controller.reopen)

    def test_a_change_that_adds_no_voices_asks_nothing(self):
        self.plan.added = ()
        self.save()
        self.assertEqual((self.refreshed, self.closed), ([], []))


class GroupListTests(unittest.TestCase):
    """The cast list shows a group's voices under it, and which groups are proposed."""

    def setUp(self):
        from cast_tui_fixtures import FakePlayer, FakeSession, voice_row
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        chorus = voice_row("v-chorus", "the chorus") | {
            "ensemble": {"members": ["v-pool-2", "v-pool-3"], "mode": "pool", "basis": "user"}}
        crowd = voice_row("v-crowd", "the crowd") | {"group": {"proposed_voices": 5, "proposed_mode": "unison"}}
        pool = [voice_row(voice, f"the chorus, voice {voice[-1]}") | {"member_of": "v-chorus", "characters": []}
                for voice in ("v-pool-2", "v-pool-3")]
        session = FakeSession(rows=[chorus, voice_row("v-mara", "Mara"), crowd, *pool])
        session.bundle = {"cast": {"characters": [{"id": "c-chorus", "name": "the chorus",
                                                   "voice_id": "v-chorus"}]}}
        self.controller = CastController(session, FakePlayer(), describe_error=str)
        self.addCleanup(self.controller.close)
        self.controller.refresh()

    def listed(self):
        return [(row["voice_id"], note) for row, _, note in self.controller.visible()]

    def test_a_groups_voices_follow_it_and_each_says_whose_it_is(self):
        self.assertEqual(self.listed(), [
            ("v-chorus", "group of 3 voices, pool; no voice yet; g generates one"),
            ("v-pool-2", "one of the chorus's voices; no voice yet; g generates one"),
            ("v-pool-3", "one of the chorus's voices; no voice yet; g generates one"),
            ("v-mara", "no voice yet; g generates one"),
            ("v-crowd", "proposed group of 5: C sets up its voices; no voice yet; g generates one")])

    def test_sorting_by_status_lists_every_voice_on_its_own(self):
        self.controller.filter.by_status = True
        self.assertEqual(len(self.listed()), 5)

    def test_merging_or_renaming_a_groups_voice_points_to_the_group(self):
        row = self.controller.row("v-pool-2")
        with self.assertRaisesRegex(ValueError, "is one of the chorus's voices, not a character: choose the group"):
            self.controller.character(row)


class EnsembleCastScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_shift_c_asks_how_many_voices_how_they_speak_and_saves_after_confirmation(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from cast_tui_fixtures import FakePlayer, FakeSession, voice_row
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        from epub_tts_narrator.tui_widgets import ChoiceScreen, ConfirmScreen, MultiChoiceScreen
        from textual.widgets import DataTable
        asyncio.get_running_loop().slow_callback_duration = 10
        session = FakeSession(rows=[voice_row("v-chorus", "the chorus"), voice_row("v-mara", "Mara")])
        session.store = None
        session.bundle = {"cast": {"characters": []}}
        controller = CastController(session, FakePlayer(), describe_error=str)
        self.addCleanup(controller.close)
        asked = []
        controller.ensemble_view = lambda row: {
            "name": "the chorus", "group": {"proposed_voices": 4}, "ensemble": None, "voices": 4, "mode": "unison",
            "mix": None, "named_members": [], "choosable": [("v-mara", "Mara")],
            "evidence": ["they said in unison"]}

        def plan(row, **choice):
            asked.append(choice)
            return SimpleNamespace(name="the chorus", summary="the chorus: 4 voices", path=None, hashes={},
                                   candidate={}, added=())
        controller.ensemble_plan = plan
        app = CastApp(controller, title="Fixture Book", startup=False)
        with patch("epub_tts_narrator.renderer.ensemble_command.apply") as apply:
            async with app.run_test(size=(160, 45)) as pilot:
                app.query_one("#voices", DataTable).focus()
                await pilot.pause()
                await pilot.press("C")
                self.assertIsInstance(app.screen, ChoiceScreen)
                await pilot.press("down", "enter")              # choose how many and how they speak
                self.assertIsInstance(app.screen, ChoiceScreen)
                await pilot.press("down", "down", "enter")      # 4 voices (the proposal)
                await pilot.press("enter")                      # unison first: the proposed mode
                await pilot.press("down", "down", "down", "enter")  # crowd mix
                self.assertIsInstance(app.screen, ChoiceScreen)
                await pilot.press("down", "enter")              # some are characters in the cast
                self.assertIsInstance(app.screen, MultiChoiceScreen)
                await pilot.press("enter")                      # none of them after all
                self.assertIsInstance(app.screen, ConfirmScreen)
                self.assertFalse(apply.called, "nothing is saved before the person confirms")
                await pilot.press("y")
                controller.jobs.wait_idle(5)
                await pilot.pause(1)
        self.assertEqual(asked, [{"voices": 4, "mode": "unison", "mix": "crowd", "named": []}])
        self.assertTrue(apply.called)
        self.assertTrue(controller.reopen)


    async def test_shift_c_first_offers_the_group_its_own_proposed_voices_in_one_step(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch
        from cast_tui_fixtures import FakePlayer, FakeSession, voice_row
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        from epub_tts_narrator.tui_widgets import ChoiceScreen, ConfirmScreen
        from textual.widgets import DataTable
        asyncio.get_running_loop().slow_callback_duration = 10
        session = FakeSession(rows=[voice_row("v-jotun", "dead jotun"), voice_row("v-mara", "Mara")])
        session.store = None
        session.bundle = {"cast": {"characters": []}}
        controller = CastController(session, FakePlayer(), describe_error=str)
        self.addCleanup(controller.close)
        asked = []
        controller.ensemble_view = lambda row: {
            "name": "dead jotun", "group": {"proposed_voices": 4}, "ensemble": None, "voices": 4, "mode": "unison",
            "mix": None, "named_members": [], "choosable": [("v-mara", "Mara")],
            "evidence": ["several said in unison"]}

        def plan(row, **choice):
            asked.append(choice)
            return SimpleNamespace(name="dead jotun", summary="dead jotun: 4 voices", path=None, hashes={},
                                   candidate={}, added=())
        controller.ensemble_plan = plan
        app = CastApp(controller, title="Fixture Book", startup=False)
        with patch("epub_tts_narrator.renderer.ensemble_command.apply"):
            async with app.run_test(size=(160, 45)) as pilot:
                app.query_one("#voices", DataTable).focus()
                await pilot.pause()
                await pilot.press("C")
                self.assertIsInstance(app.screen, ChoiceScreen)
                self.assertIn("Give it its own 4 voices, unison (proposed from the book)",
                              str(app.screen.options[0][1]))
                await pilot.press("enter")
                self.assertIsInstance(app.screen, ConfirmScreen)
                await pilot.press("y")
                controller.jobs.wait_idle(5)
                await pilot.pause(1)
        self.assertEqual(asked, [{"voices": 4, "mode": "unison", "mix": None, "named": []}])

    async def test_a_groups_voices_are_listed_under_it_and_notes_go_to_the_group(self):
        import asyncio
        from cast_tui_fixtures import FakePlayer, FakeSession, voice_row
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.cast_tui_model import CastController
        from textual.widgets import DataTable
        asyncio.get_running_loop().slow_callback_duration = 10
        chorus = voice_row("v-chorus", "the chorus") | {
            "ensemble": {"members": ["v-pool-2"], "mode": "pool", "basis": "user"}}
        pool = voice_row("v-pool-2", "the chorus, voice 2") | {"member_of": "v-chorus", "characters": []}
        controller = CastController(FakeSession(rows=[chorus, voice_row("v-mara", "Mara"), pool]), FakePlayer(),
                                    describe_error=str)
        self.addCleanup(controller.close)
        app = CastApp(controller, title="Fixture Book", startup=False)
        warnings = []
        notify = app.notify
        app.notify = lambda message, **options: (warnings.append(message), notify(message, **options))[1]
        async with app.run_test(size=(160, 45)) as pilot:
            table = app.query_one("#voices", DataTable)
            names = [str(table.get_row_at(index)[0]) for index in range(table.row_count)]
            self.assertEqual(names, ["the chorus", "  └ the chorus, voice 2", "Mara"])
            table.focus()
            table.move_cursor(row=1)
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            self.assertIs(app.screen, app.screen_stack[0], "no notes editor opens for a group's voice")
        self.assertTrue(any("add them to the chorus" in message for message in warnings))


if __name__ == "__main__":
    unittest.main()
