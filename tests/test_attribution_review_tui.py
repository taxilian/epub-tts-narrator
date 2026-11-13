"""verify-attribution's full-screen review: decisions, new characters, publication, and the screen."""

from unittest.mock import patch
import unittest

from epub_tts_narrator.analysis_directory import _completed_directory
from epub_tts_narrator.contracts import validate_document
from test_attribution_verification_cli import AttributionCLIHarness


def decide(*steps):
    """A stand-in for the run shell's screen host: apply ``steps`` to the review, then answer."""
    seen = {}

    def show(make_screen):
        screen = make_screen()
        review = seen["review"] = screen.review
        answer = "save"
        for step in steps:
            answer = step(review) or answer
        return answer
    return show, seen


class ReviewCLITests(AttributionCLIHarness):
    def run_review(self, *steps):
        show, seen = decide(*steps)
        with patch("epub_tts_narrator.tui_support.active_screen_host", return_value=show):
            code, output, errors, _ = self.invoke(tty=True)
        self.assertEqual(code, 0, errors + output)
        return output, seen["review"]

    def test_accepting_one_and_leaving_one_saves_only_the_accepted_change(self):
        original = self.make_book()
        output, review = self.run_review(lambda r: r.accept(0), lambda r: r.leave(1))
        self.assertEqual(len(review.found), 2)
        revised, _, _ = _completed_directory(self.book)
        self.assertEqual(revised["cast"], original["cast"])
        accepted = review.found[0]["segment_id"]
        self.assertEqual({row["id"]: row for row in revised["segments"]}[accepted],
                         review.proposals[accepted])
        self.assertIn("Review: 1 applied / 1 left unchanged.", output)

    def test_a_new_character_joins_the_cast_through_a_speaker_decision(self):
        original = self.make_book()
        output, review = self.run_review(lambda r: r.new_character(0, "Orrin", "Gravel bass, slow."),
                                         lambda r: r.accept(1))
        revised, _, _ = _completed_directory(self.book)
        self.assertEqual(validate_document(revised), [])
        names = [character["name"] for character in revised["cast"]["characters"]]
        self.assertEqual(len(names), len(original["cast"]["characters"]) + 1)
        self.assertIn("Orrin", names)
        segment = {row["id"]: row for row in revised["segments"]}[review.found[0]["segment_id"]]
        self.assertEqual(segment["attribution"]["basis"], "user")
        orrin = next(role for role in revised["cast"]["roles"] if role["id"] == segment["role_id"])
        self.assertEqual(next(c for c in revised["cast"]["characters"] if c["id"] == orrin["character_id"])["name"],
                         "Orrin")
        self.assertIn("1 new character(s)", output)

    def test_quitting_leaves_the_book_unchanged(self):
        self.make_book()
        before = {name: (self.book / name).read_bytes() for name in ("analysis.json", "manifest.json")}
        output, _ = self.run_review(lambda r: r.accept(0), lambda r: "leave")
        self.assertIn("Leave unchanged", output)
        self.assertEqual({name: (self.book / name).read_bytes() for name in before}, before)


class FindingReviewTests(unittest.TestCase):
    def review(self):
        from epub_tts_narrator.attribution_manual_review import FindingReview
        from epub_tts_narrator.attribution_verification import findings
        from test_attribution_verification import risk_document
        document = risk_document(True)
        audited = __import__("copy").deepcopy(document)
        segment = audited["segments"][1]
        other = next(role for role in document["cast"]["roles"]
                     if role["kind"] == "dialogue" and role["id"] != segment["role_id"])
        segment.update(role_id=other["id"], voice_id=other["voice_id"])
        return FindingReview(document, audited, findings(document, audited)), other

    def test_edits_that_return_to_the_proposal_or_the_saved_line_count_as_accept_or_leave(self):
        review, other = self.review()
        self.assertEqual(review.status(0), "undecided")
        review.set_direction(0, "tired")
        self.assertEqual(review.status(0), "edit")
        review.set_direction(0, review.proposals[review.segment_id(0)]["delivery"]["direction"])
        self.assertIn(review.status(0), {"accept", "edit"})
        review.leave(0)
        self.assertEqual(review.staged(0), review.originals[review.segment_id(0)])

    def test_a_new_name_already_in_the_cast_is_refused_and_a_reused_new_name_keeps_its_description(self):
        review, _ = self.review()
        name = next(iter(review.characters.values()))["name"]
        with self.assertRaisesRegex(ValueError, "already in the cast"):
            review.new_character(0, name, "anything")
        review.new_character(0, "Orrin", "Gravel bass.")
        self.assertEqual(review.new_names(), ["Orrin"])
        self.assertEqual(review.status(0), "new")


class FindingsScreenTests(AttributionCLIHarness, unittest.IsolatedAsyncioTestCase):
    async def test_keys_accept_leave_add_a_character_and_save(self):
        from textual.app import App
        from epub_tts_narrator.attribution_review_tui import FindingsScreen
        from epub_tts_narrator.attribution_manual_review import FindingReview
        from epub_tts_narrator.attribution_verification import findings
        from test_attribution_verification import risk_document
        from copy import deepcopy
        document = risk_document(True)
        audited = deepcopy(document)
        for segment in audited["segments"][1:3]:
            other = next(role for role in document["cast"]["roles"]
                         if role["kind"] == "dialogue" and role["id"] != segment["role_id"])
            segment.update(role_id=other["id"], voice_id=other["voice_id"])
        review = FindingReview(document, audited, findings(document, audited))
        answers = []

        class Host(App):
            def on_mount(self):
                self.push_screen(FindingsScreen(review), answers.append)

        async with Host().run_test(size=(150, 42)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            self.assertEqual(review.status(0), "accept")
            await pilot.press("n", *"Orrin", "enter", *"Gravel", "enter")
            await pilot.pause()
            self.assertEqual(review.status(1), "new")
            await pilot.press("w")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
        self.assertEqual(answers, ["save"])


if __name__ == "__main__":
    unittest.main()
