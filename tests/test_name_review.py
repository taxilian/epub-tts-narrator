"""An entry listed under a description is renamed to the personal name the book gives it (``name_review``)."""

from copy import deepcopy
import contextlib
import io
import json
import unittest

from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.name_review import eligible, name_on_hand, review_names
from test_attribution_verification_cli import AttributionCLIHarness


def described(document):
    """The fixture's Mara, as first met: listed under a description, her name only an alias."""
    mara = next(row for row in document["cast"]["characters"] if row["name"] == "Mara")
    mara.update(name="the soft speaker", aliases=["Mara"])


class NameClient:
    """Proposes ``names`` (listed name -> personal name) and confirms each with the paragraph that has it."""

    def __init__(self, names, verdict="y"):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.names, self.verdict, self.calls = names, verdict, []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if payload["mode"] == "name_review_nominate":
            rows = [line.split("|") for line in payload["cast"].splitlines()]
            return {"r": [[row[0], self.names[row[1]]] for row in rows if row[1] in self.names]}
        if payload["mode"] == "name_review_check":
            answers = []
            for item in payload["items"]:
                if self.verdict != "y":
                    answers.append([item["id"], self.verdict])
                    continue
                line = next(line for line in item["text"].splitlines() if item["name"] in line.partition("|")[2])
                number, _, text = line.partition("|")
                answers.append([item["id"], "y", int(number), text])
            return {"r": answers}
        raise AssertionError(f"Unexpected request mode: {payload['mode']}")


def config():
    return {"analysis": deepcopy(DEFAULTS["analysis"]),
            "provider": {"model": "test-model", "base_url": "https://example.invalid"}}


class NameReviewTests(AttributionCLIHarness):
    def review(self, client):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            changed = review_names(self.book, config(), lambda settings: client, lambda message: None)
        return changed, output.getvalue()

    def analysis(self):
        return json.loads((self.book / "analysis.json").read_text(encoding="utf-8"))

    def test_an_entry_listed_under_a_description_is_renamed_to_the_name_the_book_gives(self):
        self.make_book(risk=False, prepare=described)
        changed, output = self.review(NameClient({"the soft speaker": "Mara"}))
        self.assertTrue(changed, output)
        self.assertIn("renamed the soft speaker to Mara", output)
        mara = next(row for row in self.analysis()["cast"]["characters"] if row["name"] == "Mara")
        self.assertNotIn("Mara", mara["aliases"])
        record = self.analysis()["provenance"]["character_renames"][-1]
        self.assertEqual((record["to"], record["basis"]), ("Mara", "analysis"))

    def test_a_name_the_text_does_not_show_is_not_used(self):
        self.make_book(risk=False, prepare=described)
        changed, _ = self.review(NameClient({"the soft speaker": "Mara"}, verdict="?"))
        self.assertFalse(changed)
        self.assertIn("the soft speaker", [row["name"] for row in self.analysis()["cast"]["characters"]])

    def test_a_character_a_person_renamed_is_never_renamed_again(self):
        from epub_tts_narrator.renderer.character_rename_command import apply_rename, plan_rename
        self.make_book(risk=False, prepare=described)
        apply_rename(plan_rename(self.book, "the soft speaker", "the quiet one"), progress=lambda _: None)
        client = NameClient({"the quiet one": "Mara"})
        changed, _ = self.review(client)
        self.assertFalse(changed)
        self.assertEqual(client.calls, [], "nothing to ask about")
        self.assertIn("the quiet one", [row["name"] for row in self.analysis()["cast"]["characters"]])

    def test_only_entries_with_a_name_on_hand_are_asked_about(self):
        self.assertTrue(name_on_hand({"name": "Eosian Beacon", "aliases": ["the Beacon", "new Beacon", "Aurora"]}))
        self.assertTrue(name_on_hand({"name": "Red Mouse", "aliases": [], "description": "bonded mouse named Nestor"}))
        self.assertFalse(name_on_hand({"name": "Eosian Beacon", "aliases": ["the Beacon", "sleeping Beacon"]}))
        self.assertFalse(name_on_hand({"name": "Mara Venn", "aliases": ["Mara", "Venn"]}),
                         "parts of the listed name are not another name")
        self.make_book(risk=False)
        self.assertEqual(eligible(self.analysis()), [])
        client = NameClient({})
        self.assertFalse(self.review(client)[0])
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
