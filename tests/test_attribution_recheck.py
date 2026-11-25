"""Rerunning analyze on a saved book double-checks every paragraph's speakers and speech split."""

from copy import deepcopy
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from epub_tts_narrator.attribution_recheck import record_checks, recheck_attributions, unchecked_blocks
from epub_tts_narrator.config import DEFAULTS
from test_attribution_verification import AuditClient
from test_attribution_verification_cli import AttributionCLIHarness, NoFindingAuditClient


def config():
    return {"analysis": deepcopy(DEFAULTS["analysis"]),
            "provider": {"model": "test-model", "base_url": "https://example.invalid"}}


class RecheckTests(AttributionCLIHarness):
    def check(self, client):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            published = recheck_attributions(self.book, config(), lambda settings: client, lambda message: None)
        return published, output.getvalue()

    def analysis(self):
        return json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))

    def scanned(self, client):
        return [unit["block_id"] for call in client.calls if call["mode"] == "attribution_review_scan"
                for unit in call["target_units"]]

    def test_every_paragraph_is_checked_once_even_lines_no_risk_rule_selects(self):
        # No context-basis or flagged line: verify-attribution's risk set is empty.
        self.make_book(risk=False)
        blocks = [block["id"] for block in self.analysis()["source"]["blocks"]]
        client = NoFindingAuditClient()
        published, output = self.check(client)
        self.assertTrue(published)
        self.assertEqual(sorted(self.scanned(client)), sorted(blocks))
        self.assertIn(f"Checking speakers and speech splits in {len(blocks)}/{len(blocks)}", output)
        self.assertEqual(unchecked_blocks(self.analysis()), [])

        again = NoFindingAuditClient()
        manifest = (self.book / "manifest.json").read_bytes()
        published, output = self.check(again)
        self.assertFalse(published)
        self.assertEqual(again.calls, [])
        self.assertIn("already checked", output)
        self.assertEqual((self.book / "manifest.json").read_bytes(), manifest)

    def test_agreed_corrections_are_applied_and_the_corrected_paragraphs_count_as_checked(self):
        before = self.make_book(risk=False)
        published, output = self.check(AuditClient())
        self.assertTrue(published)
        after = self.analysis()
        speakers = {row["role_id"] for row in after["segments"] if row["kind"] == "dialogue"}
        self.assertEqual(speakers, {"tom"})
        self.assertNotEqual(after["segments"], before["segments"])
        self.assertEqual(after["source"], before["source"])
        self.assertIn("paragraph(s) corrected", output)
        self.assertIn("revision journal", output)
        self.assertEqual(unchecked_blocks(after), [])

    def added(self, document, name, *, speaks_in=None):
        """``document`` with a person's new character ``name`` (speaking the line in ``speaks_in``)."""
        changed = deepcopy(document)
        character = deepcopy(changed["cast"]["characters"][0])
        character.update(id="char_new", name=name, aliases=[], voice_id="voice_new")
        voice = deepcopy(changed["cast"]["voices"][-1])
        voice.update(id="voice_new")
        role = deepcopy(next(role for role in changed["cast"]["roles"] if role.get("character_id")))
        role.update(id="role_new", character_id="char_new", voice_id="voice_new")
        changed["cast"]["characters"].append(character)
        changed["cast"]["voices"].append(voice)
        changed["cast"]["roles"].append(role)
        if speaks_in is not None:
            row = next(row for row in changed["segments"] if row["block_id"] == speaks_in and row["kind"] == "dialogue")
            row.update(role_id="role_new", voice_id="voice_new")
        return changed

    def test_a_character_added_after_the_check_has_the_lines_around_it_checked_again(self):
        self.make_book(risk=False)
        self.check(NoFindingAuditClient())
        document = self.analysis()
        self.assertEqual(unchecked_blocks(document), [])
        # Mentioned in the second paragraph only ("They left."), speaking nowhere yet.
        self.assertEqual(unchecked_blocks(self.added(document, "Left")), ["b2"])
        # Given a line by a person: the paragraphs around that line are checked again.
        blocks = [block["id"] for block in document["source"]["blocks"]]
        self.assertEqual(unchecked_blocks(self.added(document, "Quill", speaks_in="b2")), blocks)
        # Once checked with the new character in the cast, nothing is checked again.
        changed = self.added(document, "Quill", speaks_in="b2")
        record_checks(changed, unchecked_blocks(changed))
        self.assertEqual(unchecked_blocks(changed), [])

    def test_only_a_paragraph_whose_split_changed_is_checked_again(self):
        self.make_book(risk=False)
        self.check(NoFindingAuditClient())
        document = self.analysis()
        self.assertEqual(unchecked_blocks(document), [])
        changed = deepcopy(document)
        row = next(row for row in changed["segments"] if row["kind"] == "dialogue")
        row["role_id"] = "tom" if row["role_id"] != "tom" else "mara"
        self.assertEqual(unchecked_blocks(changed), [row["block_id"]])
        # A new direction alone is not a change to who speaks or where speech starts.
        directed = deepcopy(document)
        directed["segments"][0]["delivery"] = {"direction": "quietly", "evidence": []}
        self.assertEqual(unchecked_blocks(directed), [])

    def test_a_persons_speaker_choice_is_kept(self):
        def chosen(document):
            for row in document["segments"]:
                if row["kind"] == "dialogue":
                    row["attribution"].update(basis="user", evidence=[], reason="Chosen by a person.")
        self.make_book(risk=False, prepare=chosen)
        before = {row["id"]: row["role_id"] for row in self.analysis()["segments"]
                  if row["attribution"]["basis"] == "user"}
        self.assertTrue(before)
        client = AuditClient()
        self.check(client)
        chosen = {row["block_id"] for row in self.analysis()["segments"] if row["id"] in before}
        self.assertFalse(chosen & set(self.scanned(client)))
        after = {row["id"]: row["role_id"] for row in self.analysis()["segments"] if row["id"] in before}
        self.assertEqual(after, before)

    def test_with_reconsider_choices_a_persons_choice_is_checked_and_may_be_corrected(self):
        def chosen(document):
            for row in document["segments"]:
                if row["kind"] == "dialogue":
                    row["attribution"].update(basis="user", evidence=[], reason="Chosen by a person.")
        self.make_book(risk=False, prepare=chosen)
        before = {row["id"] for row in self.analysis()["segments"] if row["attribution"]["basis"] == "user"}
        client = AuditClient()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            published = recheck_attributions(self.book, config(), lambda settings: client, lambda message: None,
                                              reconsider_choices=True)
        self.assertTrue(published)
        self.assertIn("your own speaker choices may be corrected too", output.getvalue())
        chosen_blocks = {row["block_id"] for row in self.analysis()["segments"] if row["id"] in before}
        self.assertTrue(chosen_blocks <= set(self.scanned(client)), "a person's paragraphs are checked too")
        scans = [call for call in client.calls if call["mode"] == "attribution_review_scan"]
        self.assertTrue(all(call.get("reconsider_choices") for call in scans))
        speakers = {row["role_id"] for row in self.analysis()["segments"] if row["kind"] == "dialogue"}
        self.assertEqual(speakers, {"tom"}, "the agreed correction replaced the person's earlier choice")

    def test_new_instructions_check_every_paragraph_once_and_reach_the_reviewers(self):
        self.make_book(risk=False)
        self.check(NoFindingAuditClient())
        document = self.analysis()
        blocks = [block["id"] for block in document["source"]["blocks"]]
        notes = "There are three whispering voices; the third is not the eaterling."
        self.assertEqual(unchecked_blocks(document), [])
        self.assertEqual(unchecked_blocks(document, notes=notes), blocks)
        client = NoFindingAuditClient()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            recheck_attributions(self.book, config(), lambda settings: client, lambda message: None, notes=notes)
        scans = [call for call in client.calls if call["mode"] == "attribution_review_scan"]
        self.assertTrue(scans and all(call.get("review_notes") == notes for call in scans))
        after = self.analysis()
        self.assertEqual(unchecked_blocks(after, notes=notes), [], "the same notes do not check again")
        self.assertEqual(unchecked_blocks(after), [], "a rerun without notes keeps the record")


if __name__ == "__main__":
    unittest.main()


class WholeBookRecheckTests(AttributionCLIHarness):
    """``analyze BOOK --recheck`` (or yes when asked): every check again with fresh answers, resumable."""

    STEPS = {"cast": "epub_tts_narrator.cast_review.review_cast",
             "names": "epub_tts_narrator.name_review.review_names",
             "genders": "epub_tts_narrator.gender_review.review_genders",
             "group-tags": "epub_tts_narrator.group_check.review_group_tags",
             "joint-lines": "epub_tts_narrator.joint_lines.review_joint_lines",
             "attribution": "epub_tts_narrator.attribution_recheck.recheck_attributions",
             "directions": "epub_tts_narrator.direction_review.review_book_directions"}

    def rerun(self, *arguments, fail=None):
        from unittest.mock import Mock
        from epub_tts_narrator.cli import main
        settings = deepcopy(DEFAULTS)
        settings["provider"].update(model="test-model", api_key="test-key")
        mocks = {step: Mock(return_value=False) for step in self.STEPS}
        if fail:
            mocks[fail].side_effect = RuntimeError("interrupted fixture")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("epub_tts_narrator.cli.load_config", return_value=settings))
            for step, target in self.STEPS.items():
                stack.enter_context(patch(target, mocks[step]))
            for target in ("epub_tts_narrator.group_proposals.review_groups",
                           "epub_tts_narrator.character_split.review_splits",
                           "epub_tts_narrator.narrator_review.review_narrators",
                           "epub_tts_narrator.direction_refresh_command.refresh_outdated_directions"):
                stack.enter_context(patch(target, return_value=False))
            output = io.StringIO()
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            try:
                main(["analyze", str(self.book), *arguments])
            except RuntimeError:
                pass
        caches = {step: mock.call_args.kwargs.get("recheck_cache") if mock.called else "not run"
                  for step, mock in mocks.items()}
        return caches, output.getvalue()

    def test_recheck_gives_every_step_a_fresh_cache_and_resumes_after_an_interruption(self):
        from epub_tts_narrator import recheck
        self.make_book(risk=False)
        caches, _ = self.rerun("--recheck", fail="genders")
        self.assertIsNotNone(caches["cast"])
        self.assertIsNotNone(caches["genders"])
        self.assertEqual(caches["attribution"], "not run")
        pending = recheck.pending(self.book, DEFAULTS["analysis"])
        self.assertEqual(pending.done, ["cast", "names"], "finished steps are remembered, the interrupted one is not")
        run_cache = caches["cast"]

        caches, output = self.rerun()
        self.assertIn("Resuming the recheck of the whole book", output)
        self.assertIsNone(caches["cast"], "a finished step is not asked again")
        self.assertIsNone(caches["names"])
        self.assertEqual(caches["genders"], run_cache, "the interrupted step replays its answers so far")
        self.assertTrue(all(caches[step] == run_cache for step in ("group-tags", "joint-lines", "attribution",
                                                                   "directions")))
        self.assertIsNone(recheck.pending(self.book, DEFAULTS["analysis"]), "a finished recheck is removed")

        caches, output = self.rerun()
        self.assertTrue(all(value is None for value in caches.values()), "an ordinary rerun again")

    def test_a_new_recheck_never_replays_an_earlier_one(self):
        self.make_book(risk=False)
        first, _ = self.rerun("--recheck")
        second, _ = self.rerun("--recheck")
        self.assertNotEqual(first["attribution"], second["attribution"])

    def test_a_terminal_is_asked_and_told_when_nothing_is_due(self):
        from epub_tts_narrator import recheck
        self.make_book(risk=False)
        document = json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))
        record_checks(document, [block["id"] for block in document["source"]["blocks"]])
        from epub_tts_narrator.direction_review import record_reviews, unreviewed_blocks
        record_reviews(document, unreviewed_blocks(document))
        said = []
        self.assertTrue(recheck.ask(document, input_fn=lambda prompt: said.append(prompt) or "y", output=said.append))
        self.assertIn("Everything in this book was already checked", said[0])
        self.assertIn("Check the whole book again with fresh answers", said[1])
        self.assertFalse(recheck.ask(document, input_fn=lambda prompt: "", output=lambda text: None))
        self.assertEqual(unchecked_blocks(document), [])
        self.assertEqual(len(unchecked_blocks(document, everything=True)), len(document["source"]["blocks"]))

    def test_a_recheck_checks_every_paragraph_again_with_fresh_requests(self):
        import tempfile
        from pathlib import Path
        self.make_book(risk=False)
        quiet = contextlib.ExitStack()
        with quiet:
            quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
            quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
            recheck_attributions(self.book, config(), lambda settings: NoFindingAuditClient(), lambda message: None)
            again = NoFindingAuditClient()
            fresh = Path(tempfile.mkdtemp())
            self.addCleanup(__import__("shutil").rmtree, fresh, True)
            recheck_attributions(self.book, config(), lambda settings: again, lambda message: None,
                                 recheck_cache=fresh)
        blocks = [block["id"] for block in json.loads((self.book / "analysis.json").read_text())["source"]["blocks"]]
        scanned = [unit["block_id"] for call in again.calls if call["mode"] == "attribution_review_scan"
                   for unit in call["target_units"]]
        self.assertEqual(sorted(scanned), sorted(blocks))
