"""Choosing speakers on the full-screen chooser: the decisions, and the screen itself."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.contracts import validate_document
from epub_tts_narrator.manual_speaker_revision import validate_manual_speaker_revision
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from test_speaker_decisions import add_lena, missing


def unsettled_lines(count):
    """A book whose last ``count`` paragraphs are each one quoted line nobody is assigned to yet."""
    document = missing(())
    add_lena(document)
    for number in range(count):
        block_id, text = f"q{number}", f"“Line {number}.”"
        document["source"]["blocks"].append({"id": block_id, "chapter_id": "ch1", "text": text,
                                             "source": {"href": "text/ch1.xhtml", "element_id": None, "tag": "p"}})
        document["source"]["chapters"][0]["block_ids"].append(block_id)
        document["segments"].append({
            "id": f"line{number}", "block_id": block_id, "start": 0, "end": len(text), "text": text,
            "kind": "dialogue", "role_id": None, "voice_id": None, "delivery": {"direction": "", "evidence": []},
            "attribution": {"basis": "unresolved", "reason": "Someone said it.", "evidence": [],
                            "candidates": ["mara", "lena"], "review_required": True}})
    return document


def journal(cache):
    [path] = (Path(cache) / "speaker-decisions").glob("*.json")
    return json.loads(path.read_text(encoding="utf-8"))


class ChooserDecisionTests(unittest.TestCase):
    """``prompt_speaker_decisions`` with a chooser in place of the screen."""

    def decide(self, document, answers, **options):
        views = []

        def chooser(view):
            views.append(view)
            return answers.pop(0)
        result = prompt_speaker_decisions(document, chooser=chooser, **options)
        self.assertEqual(validate_document(result), [])
        return result, views

    def test_the_view_shows_the_story_around_the_passage_with_its_speakers(self):
        document = missing()
        add_lena(document)
        result, views = self.decide(document, [{"action": "assign", "roles": ["lena"]}])
        view = views[0]
        self.assertEqual((view["number"], view["total"], view["target"]), (1, 1, 0))
        runs = view["paragraphs"][0]["runs"]
        self.assertEqual([(run["speakers"], run["text"], run["target"]) for run in runs],
                         [(["narrator"], "Mara said softly, ", False), ([], "“Hello.”", True)])
        self.assertEqual(view["paragraphs"][1]["runs"][0]["text"], "They left.")
        self.assertEqual([role["role_id"] for role in view["roles"]], ["narrator", "mara", "lena"])
        self.assertTrue(view["suggestions"])
        self.assertEqual(result["segments"][1]["role_id"], "lena")
        self.assertEqual(result["segments"][1]["attribution"]["basis"], "user")

    def test_a_refused_group_is_asked_again_with_the_reason(self):
        document = missing()
        add_lena(document)
        result, views = self.decide(document, [{"action": "assign", "roles": ["narrator", "mara"]},
                                               {"action": "assign", "roles": ["mara", "lena"]}])
        self.assertIn("narration cannot be a group member", views[1]["error"])
        self.assertEqual([row["role_id"] for row in result["segments"][1]["simultaneous_speakers"]], ["mara", "lena"])

    def test_a_new_character_joins_the_cast_and_a_known_name_is_refused(self):
        document = missing()
        result, views = self.decide(document, [{"action": "new", "name": "Mara", "description": "x"},
                                               {"action": "new", "name": "Orrin", "description": "Gravel bass."}])
        self.assertIsNone(views[0]["error"])
        role = result["segments"][1]["role_id"]
        self.assertTrue(role.startswith("role_prompt_"))
        self.assertIn("Orrin", [character["name"] for character in result["cast"]["characters"]])

    def test_skip_leaves_a_passage_and_stop_saves_what_was_chosen(self):
        document = missing((1,))
        document["segments"][2].update(role_id=None, voice_id=None, kind="dialogue")
        document["segments"][2]["attribution"].update(basis="unresolved", review_required=True,
                                                      reason="No consensus.", candidates=["mara"])
        with tempfile.TemporaryDirectory() as cache:
            result, views = self.decide(document, [{"action": "skip"}, {"action": "assign", "roles": ["mara"]}],
                                        decisions_cache=cache)
            self.assertEqual(len(views), 2)
            self.assertIsNone(result["segments"][1]["role_id"])
            self.assertEqual(result["segments"][2]["role_id"], "mara")
            stopped, _ = self.decide(document, [None], decisions_cache=cache)
        # The journal replays the earlier choice; stopping keeps it.
        self.assertEqual(stopped["segments"][2]["role_id"], "mara")


class GoingBackTests(unittest.TestCase):
    """Back on a passage clears the previous passage's answer and asks it again."""

    def decide(self, document, answers, **options):
        views = []

        def chooser(view):
            views.append(view)
            return answers.pop(0)
        result = prompt_speaker_decisions(document, chooser=chooser, **options)
        self.assertEqual(validate_document(result), [])
        return result, views

    def roles(self, result):
        return {row["id"]: row["role_id"] for row in result["segments"] if row["id"].startswith("line")}

    def test_back_asks_the_previous_passage_again_and_its_new_answer_replaces_the_old_one(self):
        document = unsettled_lines(2)
        with tempfile.TemporaryDirectory() as cache:
            result, views = self.decide(document, [
                {"action": "assign", "roles": ["mara"]},  # the wrong name
                {"action": "back"},
                {"action": "assign", "roles": ["lena"]},
                {"action": "assign", "roles": ["mara"]}], decisions_cache=cache)
            saved = journal(cache)
        self.assertIsNone(views[0]["previous"])
        self.assertEqual((views[1]["number"], views[1]["previous"]["number"], views[1]["previous"]["answer"]),
                         (2, 1, "Mara"))
        self.assertEqual((views[2]["segment_id"], views[2]["number"]), ("line0", 1))
        self.assertIsNone(views[2]["previous"])
        self.assertEqual(views[3]["previous"]["answer"], "Lena")
        self.assertEqual(self.roles(result), {"line0": "lena", "line1": "mara"})
        self.assertEqual(result["segments"][-2]["attribution"]["reason"], "User selected Lena.")
        receipt = result["provenance"]["speaker_decisions"]["decisions"]
        self.assertEqual([(row["segment_id"], row["selected_roles"]) for row in receipt],
                         [("line0", ["lena"]), ("line1", ["mara"])])
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["decisions"], receipt)
        # Publication recomputes the same book from the receipt.
        validate_manual_speaker_revision(document, deepcopy(document), result)

    def test_back_reaches_a_skipped_passage_and_the_first_passage_has_no_back(self):
        document = unsettled_lines(2)
        result, views = self.decide(document, [{"action": "back"}, {"action": "skip"}, {"action": "back"},
                                               {"action": "assign", "roles": ["lena"]},
                                               {"action": "assign", "roles": ["mara"]}])
        self.assertIn("first passage", views[1]["error"])
        self.assertEqual(views[2]["previous"]["answer"], "not answered")
        self.assertEqual(self.roles(result), {"line0": "lena", "line1": "mara"})

    def test_a_character_made_for_the_cleared_answer_leaves_the_cast(self):
        document = unsettled_lines(2)
        result, views = self.decide(document, [{"action": "new", "name": "Orrin", "description": "Gravel bass."},
                                               {"action": "back"},
                                               {"action": "assign", "roles": ["lena"]},
                                               {"action": "assign", "roles": ["mara"]}])
        self.assertIn("Orrin", [role["label"] for role in views[1]["roles"]])
        self.assertNotIn("Orrin", [role["label"] for role in views[2]["roles"]])
        self.assertEqual(result["cast"], document["cast"])
        self.assertEqual(self.roles(result), {"line0": "lena", "line1": "mara"})
        validate_manual_speaker_revision(document, deepcopy(document), result)

    def test_back_into_an_interrupted_runs_answer_changes_it_and_rewrites_the_journal(self):
        document = unsettled_lines(2)
        with tempfile.TemporaryDirectory() as cache:
            self.decide(document, [{"action": "assign", "roles": ["mara"]}, {"action": "stop"}],
                        decisions_cache=cache)
            result, views = self.decide(document, [{"action": "back"}, {"action": "assign", "roles": ["lena"]},
                                                   {"action": "stop"}], decisions_cache=cache)
            saved = journal(cache)
        self.assertEqual((views[0]["segment_id"], views[0]["previous"]["answer"]), ("line1", "Mara"))
        self.assertEqual(views[1]["segment_id"], "line0")
        self.assertEqual(self.roles(result), {"line0": "lena", "line1": None})
        self.assertEqual([(row["segment_id"], row["selected_roles"]) for row in saved["decisions"]],
                         [("line0", ["lena"])])

    def test_a_new_character_another_answer_still_uses_stays_with_that_answer(self):
        # Passage 2 made Orrin and passage 4 chose him; the run stopped. Going back to passage 2
        # from passage 3 clears passage 2 only: Orrin is now made by passage 4's answer.
        document = unsettled_lines(4)
        with tempfile.TemporaryDirectory() as cache:
            first, _ = self.decide(document, [{"action": "skip"},
                                              {"action": "new", "name": "Orrin", "description": "Gravel bass."},
                                              {"action": "skip"}, {"action": "stop"}], decisions_cache=cache)
            orrin = first["segments"][-3]["role_id"]
            self.decide(document, [{"action": "skip"}, {"action": "skip"},
                                   {"action": "assign", "roles": [orrin]}], decisions_cache=cache)
            result, views = self.decide(document, [{"action": "skip"}, {"action": "back"},
                                                   {"action": "assign", "roles": ["lena"]}, {"action": "stop"}],
                                        decisions_cache=cache)
        self.assertEqual(views[1]["previous"]["answer"], "Orrin")
        self.assertEqual(self.roles(result), {"line0": None, "line1": "lena", "line2": None, "line3": orrin})
        receipt = result["provenance"]["speaker_decisions"]["decisions"]
        self.assertEqual([row["segment_id"] for row in receipt], ["line3", "line1"])
        self.assertEqual(receipt[0]["defined_voice"]["name"], "Orrin")
        validate_manual_speaker_revision(document, deepcopy(document), result)


class SpeakerScreenTests(unittest.IsolatedAsyncioTestCase):
    def view(self):
        from epub_tts_narrator.speaker_decisions import _role_labels, passage_view
        document = missing()
        add_lena(document)
        characters = {row["id"]: row for row in document["cast"]["characters"]}
        labels = _role_labels(document["cast"]["roles"], characters, document["source"]["book"])
        return passage_view(document, document["segments"][1], number=1, total=3, labels=labels,
                            suggestions=[{"role_id": "mara", "reason": "Named in the tag."},
                                         {"role_id": "lena", "reason": ""}],
                            roles=document["cast"]["roles"])

    async def answer(self, *keys, previous=None):
        from textual.app import App
        from epub_tts_narrator.speaker_tui import SpeakerScreen
        answers = []
        view = self.view()
        view["previous"] = previous

        class Host(App):
            def on_mount(self):
                self.push_screen(SpeakerScreen(view), answers.append)

        app = Host()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
        return answers

    async def test_enter_takes_the_best_guess_and_digits_pick_a_likely_speaker(self):
        self.assertEqual(await self.answer("enter"), [{"action": "assign", "roles": ["mara"]}])
        self.assertEqual(await self.answer("2"), [{"action": "assign", "roles": ["lena"]}])

    async def test_filtering_finds_any_voice_and_its_letters_are_not_shortcuts(self):
        self.assertEqual(await self.answer("slash", "n", "a", "r", "enter"),
                         [{"action": "assign", "roles": ["narrator"]}])

    async def test_new_character_together_skip_and_stop(self):
        self.assertEqual(await self.answer("n", *"Orrin", "enter", *"Gravel", "enter"),
                         [{"action": "new", "name": "Orrin", "description": "Gravel"}])
        self.assertEqual(await self.answer("t", "space", "down", "space", "enter"),
                         [{"action": "assign", "roles": ["mara", "lena"]}])
        self.assertEqual(await self.answer("s"), [{"action": "skip"}])
        self.assertEqual(await self.answer("escape"), [{"action": "stop"}])

    async def test_b_goes_back_only_when_there_is_a_previous_passage(self):
        previous = {"number": 1, "answer": "Mara", "text": "“Hello.”"}
        self.assertEqual(await self.answer("b", previous=previous), [{"action": "back"}])
        self.assertEqual(await self.answer("b", "enter"), [{"action": "assign", "roles": ["mara"]}])
        # The menu's last choice is the same Back.
        self.assertEqual(await self.answer("end", "enter", previous=previous), [{"action": "back"}])

    async def test_the_passage_is_highlighted_among_colored_speakers(self):
        from epub_tts_narrator.speaker_tui import paragraph_text, speaker_styles
        view = self.view()
        styles = speaker_styles(view)
        self.assertEqual(styles["narrator"], "italic")
        self.assertTrue(styles["mara"].startswith("color("))
        text = paragraph_text(view["paragraphs"][0], view, styles)
        # Narration reads plainly with no name; the passage is marked and highlighted.
        self.assertEqual(text.plain, "Mara said softly,  ▶ “Hello.”")
        self.assertEqual(text.spans[0].style, "italic")
        self.assertIn("on yellow", str(text.spans[-1].style))
        neighbor = {"runs": [{"speakers": ["lena"], "text": "“Go.”", "target": False}]}
        self.assertEqual(paragraph_text(neighbor, view, styles).plain, "Lena  “Go.”")


if __name__ == "__main__":
    unittest.main()
