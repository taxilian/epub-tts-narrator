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


if __name__ == "__main__":
    unittest.main()
