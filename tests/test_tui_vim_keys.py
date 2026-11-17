"""Vim movement keys in every full-screen interface (``tui_widgets.VimKeysMixin``), driven by Textual's pilot."""

import asyncio
import unittest

from textual.app import App
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, OptionList, Static, TextArea

from epub_tts_narrator.tui_widgets import ChoiceScreen, KeysScreen, VimKeysMixin


class Fixture(VimKeysMixin, App):
    BINDINGS = [Binding("h", "hello", "Hello")]  # the app's own h keeps its meaning

    def __init__(self):
        super().__init__()
        self.hellos = 0

    def compose(self):
        yield DataTable(id="table", cursor_type="row")
        yield OptionList(*[f"option {index}" for index in range(30)], id="options")
        with VerticalScroll(id="pane"):
            yield Static("\n".join(f"line {index}" for index in range(200)))
        yield TextArea(id="text")

    def on_mount(self):
        table = self.query_one("#table", DataTable)
        table.add_column("Row")
        for index in range(40):
            table.add_row(str(index), key=str(index))
        table.focus()

    def action_hello(self):
        self.hellos += 1


class VimKeysTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 10

    async def press(self, pilot, *keys):
        for key in keys:
            await pilot.press(key)
            await pilot.pause()

    async def test_vim_keys_move_tables_lists_and_panes_like_the_arrow_keys(self):
        app = Fixture()
        async with app.run_test(size=(80, 30)) as pilot:
            table = app.query_one("#table", DataTable)
            await self.press(pilot, "j", "j", "k")
            self.assertEqual(table.cursor_row, 1)
            await self.press(pilot, "G")
            self.assertEqual(table.cursor_row, 39)
            await self.press(pilot, "g")
            self.assertEqual(table.cursor_row, 0)
            await self.press(pilot, "ctrl+d")
            self.assertGreater(table.cursor_row, 1)
            await self.press(pilot, "ctrl+u")
            self.assertEqual(table.cursor_row, 0)
            options = app.query_one("#options", OptionList)
            options.focus()
            options.highlighted = 0
            await self.press(pilot, "j", "j", "G")
            self.assertEqual(options.highlighted, 29)
            await self.press(pilot, "g")
            self.assertEqual(options.highlighted, 0)
            pane = app.query_one("#pane", VerticalScroll)
            pane.focus()
            await self.press(pilot, "j", "j", "j")
            self.assertGreater(pane.scroll_y, 0)
            await self.press(pilot, "G")
            await pilot.pause(0.3)
            self.assertEqual(pane.scroll_y, pane.max_scroll_y)

    async def test_an_apps_own_key_and_typing_keep_their_meaning(self):
        app = Fixture()
        async with app.run_test(size=(80, 30)) as pilot:
            await self.press(pilot, "h")
            self.assertEqual(app.hellos, 1)
            text = app.query_one("#text", TextArea)
            text.focus()
            await self.press(pilot, "j", "k", "g")
            self.assertEqual(text.text, "jkg")

    async def test_a_dialogs_list_moves_with_j_and_k_as_with_the_arrows(self):
        results = []
        for keys in (("j", "j", "j", "k"), ("down", "down", "down", "up")):
            app = Fixture()
            async with app.run_test(size=(80, 30)) as pilot:
                chosen = []
                app.push_screen(ChoiceScreen("Pick", [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")],
                                             filterable=False), chosen.append)
                await pilot.pause()
                await self.press(pilot, *keys, "enter")
                results.append(chosen)
        self.assertEqual(results[0], results[1])
        self.assertNotEqual(results[0], ["a"])

    async def test_the_keys_list_sums_up_the_free_vim_keys(self):
        app = Fixture()
        async with app.run_test(size=(100, 40)) as pilot:
            app.push_screen(KeysScreen(Fixture.BINDINGS))
            await pilot.pause()
            vim = [row for row in app.screen.rows if row[1].startswith("Vim movement")]
            self.assertEqual(len(vim), 1)
            self.assertIn("j/k", vim[0][0])
            self.assertIn("g/G", vim[0][0])
            self.assertNotIn("h/l", vim[0][0])  # h is the app's own key; l alone moves right
            self.assertIn("l", vim[0][0].split(", "))
            self.assertFalse(any(row[1].startswith("Vim ") and row is not vim[0] for row in app.screen.rows))


class EveryAppTests(unittest.TestCase):
    def test_every_full_screen_app_has_vim_movement(self):
        from epub_tts_narrator.renderer.accent_tui import AccentApp
        from epub_tts_narrator.renderer.cast_tui import CastApp
        from epub_tts_narrator.renderer.pronounce_tui import PronounceApp, PronunciationQueueApp
        from epub_tts_narrator.review_tui_app import ReviewTuiApp
        from epub_tts_narrator.run_shell import RunApp, TaskApp
        for app in (AccentApp, CastApp, PronounceApp, PronunciationQueueApp, ReviewTuiApp, RunApp, TaskApp):
            keys = {key for klass in app.__mro__ for binding in klass.__dict__.get("BINDINGS", [])
                    if binding.action.startswith("vim_key") for key in binding.key.split(",")}
            with self.subTest(app=app.__name__):
                self.assertTrue({"j", "k", "ctrl+d", "ctrl+u", "G"} <= keys)

    def test_a_dialogs_own_letter_keeps_its_meaning_and_other_dialogs_move_with_vim_keys(self):
        from epub_tts_narrator.attribution_review_tui import FindingsScreen
        from epub_tts_narrator.run_shell_screens import HistoryTakesScreen, TakesScreen
        from epub_tts_narrator.tui_widgets import ConfirmScreen

        def action(screen, key):
            return screen._merge_bindings().key_to_bindings[key][0].action

        self.assertEqual(action(TakesScreen, "g"), "mark('good')")
        self.assertEqual(action(HistoryTakesScreen, "h"), "immediate")
        self.assertEqual(action(FindingsScreen, "l"), "leave_finding")
        self.assertEqual(action(TakesScreen, "j"), "vim_key('j')")
        self.assertEqual(action(ChoiceScreen, "G"), "vim_key('G')")
        self.assertEqual(action(ConfirmScreen, "n"), "answer(False)")


if __name__ == "__main__":
    unittest.main()
