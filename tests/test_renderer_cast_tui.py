"""The full-screen casting UI, driven headlessly with Textual's test pilot (no terminal, no models)."""

import asyncio
import unittest

from textual.widgets import DataTable, Input, TextArea

from epub_tts_narrator.renderer.cast_tui import CastApp
from epub_tts_narrator.renderer.cast_tui_model import CastController
from epub_tts_narrator.tui_widgets import ChoiceScreen, ConfirmScreen, TextEditScreen
from cast_tui_fixtures import FakePlayer, FakeSession


class CastAppTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = FakeSession()
        self.player = FakePlayer()
        self.controller = CastController(self.session, self.player, describe_error=str)
        self.addCleanup(self.controller.close)
        self.app = CastApp(self.controller, title="Fixture Book", startup=False)

    async def asyncSetUp(self):
        # Debug-mode event loops report every slow screen redraw; that is noise here.
        asyncio.get_running_loop().slow_callback_duration = 10

    async def settle(self, pilot):
        self.controller.jobs.wait_idle(5)
        await pilot.pause()
        await pilot.pause()

    def voice_keys(self):
        table = self.app.query_one("#voices", DataTable)
        return [table.coordinate_to_cell_key((index, 0)).row_key.value for index in range(table.row_count)]

    async def select(self, pilot, voice_id):
        table = self.app.query_one("#voices", DataTable)
        table.focus()
        table.move_cursor(row=self.voice_keys().index(voice_id))
        await pilot.pause()

    async def test_the_cast_list_shows_every_voice_with_its_status_and_detail(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            self.assertEqual(self.voice_keys(), ["v-ava", "v-ben", "v-cole", "v-dee", "v-eli", "v-fay", "v-ava-old"])
            self.assertIn("4/7 ready", self.app.sub_title)
            await self.select(pilot, "v-ava")
            detail = str(self.app.query_one("#detail").render())
            self.assertIn("A calm voice for Ava.", detail)
            self.assertEqual(self.app.query_one("#takes", DataTable).row_count, 2)

    async def test_filter_and_status_keys_narrow_the_list(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await pilot.press("f")
            self.assertEqual(self.voice_keys(), ["v-ben"])
            await pilot.press("f", "f", "f", "f", "f", "f", "f")
            self.assertEqual(len(self.voice_keys()), 7)
            await pilot.press("slash", "c", "o", "l", "e")
            self.assertEqual(self.voice_keys(), ["v-cole"])
            self.assertIsInstance(self.app.focused, Input)

    async def test_editing_a_field_validates_inline_and_saves_through_the_session(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.select(pilot, "v-ava")
            await pilot.press("e")
            screen = self.app.screen
            self.assertIsInstance(screen, TextEditScreen)
            screen.query_one(TextArea).text = ""
            await pilot.press("ctrl+s")
            self.assertIs(self.app.screen, screen)
            self.assertIn("Voice prompt", str(screen.query_one("#error").render()))
            screen.query_one(TextArea).text = "A bright alto with a quick, warm laugh."
            await pilot.press("ctrl+s")
            await self.settle(pilot)
            self.assertNotIsInstance(self.app.screen, TextEditScreen)
            self.assertEqual(self.session.calls[-1],
                             ("update_prompt", ("v-ava",), {"prompt": "A bright alto with a quick, warm laugh."}))

    async def test_escape_discards_an_edit(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.select(pilot, "v-ava")
            await pilot.press("t")
            self.app.screen.query_one(TextArea).text = "Gravelly Scots accent"
            await pilot.press("escape")
            await self.settle(pilot)
            self.assertEqual(self.session.calls, [])

    async def test_generate_asks_for_a_designer_and_confirms_replacing_an_accepted_voice(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.select(pilot, "v-ava")
            await pilot.press("g")
            self.assertIsInstance(self.app.screen, ChoiceScreen)
            await pilot.press("enter")  # Gemini is first when it is the configured designer.
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("y")
            await self.settle(pilot)
            self.assertEqual(self.session.calls, [("regenerate", ("v-ava",), {"voice_designer": "gemini"})])

    async def test_a_missing_voice_generates_without_a_replacement_warning(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.select(pilot, "v-ben")
            await pilot.press("g", "down", "enter")
            await self.settle(pilot)
            self.assertEqual(self.session.calls, [("regenerate", ("v-ben",), {"voice_designer": "local"})])

    async def test_a_take_is_played_and_accepted_from_the_takes_table(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.select(pilot, "v-ava")
            takes = self.app.query_one("#takes", DataTable)
            takes.focus()
            takes.move_cursor(row=0)
            await pilot.press("enter")
            await self.settle(pilot)
            self.assertEqual(self.player.started, ["/voices/c-ava-1.wav"])
            await pilot.press("space")
            self.assertEqual(self.player.stopped, 1)
            await pilot.press("a")
            self.assertIsInstance(self.app.screen, ConfirmScreen)
            await pilot.press("y")
            await self.settle(pilot)
            self.assertEqual(self.session.calls, [("accept", ("v-ava", "c-ava-1"), {})])

    async def test_clip_count_sharing_and_unsharing(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await self.select(pilot, "v-ben")
            await pilot.press("c", "down", "down", "enter")
            await self.settle(pilot)
            await pilot.press("c", "enter")
            await self.settle(pilot)
            await pilot.press("d")
            self.assertIsInstance(self.app.screen, ChoiceScreen)
            await pilot.press("A", "v", "a", "enter")
            await pilot.press("y")
            await self.settle(pilot)
            await self.select(pilot, "v-fay")
            await pilot.press("D")
            await self.settle(pilot)
            self.assertEqual(self.session.calls, [
                ("set_reference_clips", ("v-ben", 2), {}),
                ("set_reference_clips", ("v-ben", None), {}),
                ("share_voice", ("v-ben", "v-ava"), {}),
                ("stop_sharing", ("v-fay",), {}),
            ])

    async def test_a_failed_job_is_retried_with_r_while_the_screen_stays_open(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            self.session.fail["regenerate"] = ValueError("Gemini quota exceeded")
            await self.select(pilot, "v-ben")
            await pilot.press("g", "enter")
            await self.settle(pilot)
            self.assertEqual(len(self.controller.failures()), 1)
            self.assertIn("failed", str(self.app.query_one("#jobs", DataTable).get_row_at(0)[0]))
            jobs = self.app.query_one("#jobs", DataTable)
            jobs.focus()
            await pilot.press("enter")
            from epub_tts_narrator.tui_widgets import DetailScreen
            self.assertIsInstance(self.app.screen, DetailScreen)
            self.assertIn("ValueError: Gemini quota exceeded", self.app.screen.text)
            await pilot.press("escape")
            await pilot.press("R")
            await self.settle(pilot)
            self.assertEqual(self.controller.failures(), [])
            self.assertEqual([call[0] for call in self.session.calls], ["regenerate", "regenerate"])

    async def test_print_output_from_casting_code_lands_in_the_log(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            print("Rechecking v-ava against", "the current verification settings")
            await pilot.pause()
            self.assertIn("Rechecking v-ava against the current verification settings",
                          [text for _, _, text in self.controller.messages])

    async def test_the_keys_screen_lists_keys_the_footer_hides(self):
        from epub_tts_narrator.tui_widgets import KeysScreen
        async with self.app.run_test(size=(160, 45)) as pilot:
            await pilot.press("question_mark")
            self.assertIsInstance(self.app.screen, KeysScreen)
            listed = dict(self.app.screen.rows)
            self.assertIn("Compare", " ".join(listed.values()))
            self.assertIn("D", listed)
            await pilot.press("escape")
            self.assertNotIsInstance(self.app.screen, KeysScreen)

    async def test_quit_exits_when_nothing_runs(self):
        async with self.app.run_test(size=(160, 45)) as pilot:
            await pilot.press("q")
            await pilot.pause()
            self.assertFalse(self.app.is_running)



class CastCommandTests(unittest.TestCase):
    """``cast`` opens the full-screen UI only when wanted and keeps the line menu otherwise."""

    def run_cast(self, *options, tui=True):
        from unittest.mock import Mock, patch
        from epub_tts_narrator.renderer.audition import casting_command
        from epub_tts_narrator.renderer.cli import parser
        args = parser().parse_args(["cast", "book", "--config", "renderer.toml", *options])
        session = Mock()
        session.rows.return_value = []
        bundle = {"directory": "/books/fixture", "source": {"book": {"title": "Fixture Book"}},
                  "cast": {"voices": [{"id": "voice-1"}]}}
        with (patch("epub_tts_narrator.renderer.casting.CastingSession", return_value=session) as factory,
              patch("epub_tts_narrator.renderer.audition.AudioPlayer"),
              patch("epub_tts_narrator.audition_scripts.ensure_bundle_scripts", return_value=bundle),
              patch("epub_tts_narrator.renderer.book_context.voice_store"),
              patch("epub_tts_narrator.renderer.voice_clip_counts.extra_clip_plan", return_value=0),
              patch("epub_tts_narrator.tui_support.textual_available", return_value=True),
              patch("epub_tts_narrator.tui_support.interactive_terminal", return_value=tui),
              patch("epub_tts_narrator.renderer.cast_tui.run_cast_tui", return_value=0) as screen,
              patch("epub_tts_narrator.renderer.audition.run_casting") as menu):
            self.assertEqual(casting_command(args, bundle, {"casting": {"player": ["afplay"],
                                                                         "normalize_samples": False}}), 0)
        return factory, screen, menu, session

    def test_an_interactive_terminal_opens_the_casting_screen(self):
        factory, screen, menu, session = self.run_cast()
        menu.assert_not_called()
        controller = screen.call_args.args[0]
        self.assertIs(controller.session, session)
        self.assertEqual(screen.call_args.kwargs["title"], "Fixture Book")
        session.close.assert_called_once()

    def test_plain_and_non_terminal_runs_keep_the_line_menu(self):
        for options, tui in ((("--plain",), True), ((), False)):
            with self.subTest(options=options, tui=tui):
                factory, screen, menu, _ = self.run_cast(*options, tui=tui)
                screen.assert_not_called()
                menu.assert_called_once()


class ModelPromptBridgeTests(unittest.TestCase):
    """While a full-screen UI is open, model download questions use its streams, not the terminal."""

    def test_model_setup_asks_through_the_open_ui(self):
        from io import StringIO
        from types import SimpleNamespace
        from epub_tts_narrator.renderer.model_setup import ModelSetup
        from epub_tts_narrator.tui_support import active_prompt_streams, prompt_streams

        class Answer(StringIO):
            def isatty(self):
                return True
        answer, question = Answer("y\n"), StringIO()
        model = SimpleNamespace(label="Fixture model", repo="org/fixture", revision="abc", size="1 GB")
        with prompt_streams(answer, question):
            self.assertEqual(ModelSetup({})._confirm("/models/fixture", model, ["weights"]), (True, "Model download declined"))
        self.assertIn("Download this model now?", question.getvalue())
        self.assertIsNone(active_prompt_streams())


if __name__ == "__main__":
    unittest.main()
