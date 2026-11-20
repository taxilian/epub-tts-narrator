"""The questions ``analyze BOOK`` reruns ask in the run shell: what each screen says, and its keys and buttons."""

from types import SimpleNamespace
import unittest

from epub_tts_narrator.rerun_questions import NO_SCREEN, ask_on_screen, merge_view


def plan(*, moved=3, fixed="Queen of Sorcery", absorbed_voice="no accepted voice yet", later=()):
    """A ``character_merge_command.MergePlan`` stand-in: two entries both named Ce'Nedra."""
    kept = {"id": "series_char_kept", "name": "Ce'Nedra", "aliases": ["Imperial Princess"], "voice_id": "series_voice_k",
            "description": "Knows  Geran's fate;\nrides with wolf"}
    absorbed = {"id": "char_new", "name": "Ce'Nedra", "aliases": [], "voice_id": "voice_new",
                "description": "Character in the book."}
    books = [SimpleNamespace(title="Sorcerers of Darshiva", moved_lines=moved, joined_lines=0),
             *(SimpleNamespace(title=title, moved_lines=count, joined_lines=0) for title, count in later)]
    return SimpleNamespace(
        kept=kept, absorbed=absorbed, name="Ce'Nedra", names=["Ce'Nedra", "Imperial Princess"], books=books,
        reason=(f"it comes from an earlier book ({fixed}), so it must be kept" if fixed else "it has more lines"),
        fixed=fixed, kept_lines=171, absorbed_lines=moved, kept_voice="accepted voice (from an earlier book)",
        absorbed_voice=absorbed_voice, kept_first="“She didn't have Geran with her,”", absorbed_first="",
        origins={"series_char_kept": fixed} if fixed else {}, earlier=[], text="Keep: series_char_kept")


class MergeViewTests(unittest.TestCase):
    def test_two_entries_with_one_name_are_told_apart_by_side(self):
        view = merge_view(plan(), "the book's text shows they are one person")
        self.assertEqual(view["why"], "The book's text shows they are one person.")
        kept, absorbed = view["entries"]
        self.assertEqual((kept["heading"], kept["from"], kept["voice"], kept["description"]),
                         ("Stays", "From an earlier book: Queen of Sorcery", "Accepted voice (from an earlier book)",
                          "Knows Geran's fate; rides with wolf"))
        self.assertEqual((absorbed["heading"], absorbed["from"], absorbed["first_line"]),
                         ("Merged into it", "New in Sorcerers of Darshiva", ""))
        self.assertEqual(view["summary"],
                         "Merging keeps Ce'Nedra on the left, because an entry from an earlier book always stays. "
                         "Ce'Nedra on the right's 3 lines move to Ce'Nedra on the left's voice and will be rendered "
                         "again.")
        self.assertEqual(view["afterwards"], "Afterwards: Ce'Nedra, also called Imperial Princess")
        self.assertEqual(view["details"], "Keep: series_char_kept")

    def test_what_else_changes_is_said_in_notes(self):
        view = merge_view(plan(moved=1, fixed="", absorbed_voice="accepted voice", later=[("Book 5", 4)]),
                          "your notes say so")
        self.assertIn("Merging keeps Ce'Nedra on the left, because it has more lines.", view["summary"])
        self.assertIn("1 line moves", view["summary"])
        self.assertEqual(view["notes"], ["The recording accepted for Ce'Nedra on the right will no longer be used.",
                                         "Later books follow: 4 more line(s) in Book 5."])

    def test_without_the_run_shell_the_question_stays_on_the_line(self):
        self.assertIs(ask_on_screen("MergeScreen", {}), NO_SCREEN)


MERGE = {"title": "Are these the same character?", "why": "The book's text shows they are one person.",
         "entries": [{"heading": "Stays", "name": "Mara", "aliases": [], "description": "A smith.", "first_line": "“Go.”",
                      "lines": 2, "voice": "Accepted voice", "from": "New in Book"},
                     {"heading": "Merged into it", "name": "Mara Vell", "aliases": [], "description": "", "first_line": "",
                      "lines": 0, "voice": "No accepted voice yet", "from": "New in Book"}],
         "summary": "Merging keeps Mara.", "afterwards": "Afterwards: Mara, also called Mara Vell", "notes": [],
         "details": "Keep: Mara (char_1, voice v1)"}
GENDER = {"title": "Is Mara's voice the right gender?", "name": "Mara", "aliases": [], "current": "Male",
          "wanted": "Female", "why": "The book says: “she”.", "description": "Male, low.", "fixed": "Female, low.",
          "voice": "A voice already exists."}
NARRATOR = {"title": "Read narration in the point-of-view character's voice?", "notes": "Scenes change.",
            "summary": "2 narration passage(s) would be read in a different voice and rendered again.", "moved": 2,
            "sections": [{"label": "Opening", "reader": "Mara", "pov": True, "quote": "Mara", "changed": 2},
                         {"label": "Opening, part 2", "reader": "main narrator", "pov": False, "quote": "", "changed": 0}]}


class QuestionScreenTests(unittest.IsolatedAsyncioTestCase):
    async def answer(self, name, view, *keys, click=None, check=None):
        from textual.app import App
        from epub_tts_narrator import rerun_review_tui
        answers = []

        class Host(App):
            def on_mount(self):
                self.push_screen(getattr(rerun_review_tui, name)(view), answers.append)

        app = Host()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
            if check is not None:
                check(app.screen)
            if click is not None:
                await pilot.click(click)
                await pilot.pause()
        return answers

    async def test_merge_answers_with_its_letters_buttons_and_escape(self):
        self.assertEqual(await self.answer("MergeScreen", MERGE, "m"), ["m"])
        # k and l answer; they are not vim movement here.
        self.assertEqual(await self.answer("MergeScreen", MERGE, "k"), ["k"])
        self.assertEqual(await self.answer("MergeScreen", MERGE, "l"), ["l"])
        self.assertEqual(await self.answer("MergeScreen", MERGE, "escape"), ["l"])
        self.assertEqual(await self.answer("MergeScreen", MERGE, click="#choice-k"), ["k"])
        self.assertEqual(await self.answer("MergeScreen", MERGE, "enter"), ["m"])

    async def test_merge_shows_both_entries_and_hides_the_ids_until_asked(self):
        from textual.widgets import Static

        def hidden(screen):
            self.assertEqual(len(screen.query(".merge-card")), 2)
            self.assertIn("Mara Vell", str(screen.query(".merge-card").last().render()))
            self.assertFalse(screen.query_one("#merge-details", Static).display)

        def shown(screen):
            self.assertTrue(screen.query_one("#merge-details", Static).display)

        self.assertEqual(await self.answer("MergeScreen", MERGE, check=hidden), [])
        self.assertEqual(await self.answer("MergeScreen", MERGE, "d", check=shown), [])

    async def test_gender_fixes_or_leaves(self):
        self.assertEqual(await self.answer("GenderScreen", GENDER, "f"), ["f"])
        self.assertEqual(await self.answer("GenderScreen", GENDER, "l"), ["l"])
        self.assertEqual(await self.answer("GenderScreen", GENDER, "escape"), ["l"])
        self.assertEqual(await self.answer("GenderScreen", GENDER, click="#choice-l"), ["l"])

    async def test_narrator_applies_or_leaves(self):
        def sections(screen):
            text = str(screen.query_one(".narrator-sections").render())
            self.assertIn("Opening, part 2", text)
            self.assertIn("2 passage(s) change", text)

        self.assertEqual(await self.answer("NarratorScreen", NARRATOR, check=sections), [])
        self.assertEqual(await self.answer("NarratorScreen", NARRATOR, "a"), ["a"])
        self.assertEqual(await self.answer("NarratorScreen", NARRATOR, "escape"), ["l"])
        self.assertEqual(await self.answer("NarratorScreen", NARRATOR, click="#choice-a"), ["a"])


if __name__ == "__main__":
    unittest.main()
