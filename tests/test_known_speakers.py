"""A line nobody speaks goes to a character the book already knows, named beside it (``known_speakers``)."""

import contextlib
from copy import deepcopy
import io
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.analysis_directory import (_completed_directory, continuation_registry, fresh_registry,
                                                  publish_directory)
from epub_tts_narrator.analysis_revision import related_directories, revise_directories
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.known_speakers import absent_characters, nominate, review_known_speakers
from epub_tts_narrator.series import link_analysis, validate_registry
from epub_tts_narrator.speaker_decisions import prompt_speaker_decisions
from shared_fixtures import Terminal
from test_character_merge import FIRST, build

# Book 2: the keeper returns; Mara (book 1 only) and Wren (new, met in this book's reading) speak in italics
# after an action beat, which the reading took for narration and a later check left unassigned.
SECOND = [
    [("The keeper returned. ", None), ("“Back again,”", "keeper")],
    [("Mara shook out her cloak. ", None), ("Not yet, not tonight.", "?")],
    [("Wren tugged at the rope. ", None), ("Pull, everyone!", "?")],
    [("Rain fell on the stones. ", None), ("Who goes there?", "?")],
]


class Client:
    """Says ``answers`` (character name -> said) for each item, and designs any voice asked for."""

    def __init__(self, says=("Mara", "Wren")):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.says, self.calls = set(says), []

    def complete(self, system, payload):
        self.calls.append(deepcopy(payload))
        if payload["mode"] == "known_speaker_check":
            answers = []
            for item in payload["items"]:
                options = [line.split("|") for line in item["options"].splitlines()]
                marked = item["text"].splitlines()[-1]
                chosen = next((row for row in options if row[1] in self.says and row[1] in marked), None)
                if chosen is None:
                    answers.append([item["id"], "-"])
                    continue
                line = next(line for line in item["text"].splitlines() if chosen[1] in line.partition("|")[2])
                number, _, text = line.partition("|")
                answers.append([item["id"], chosen[0], int(number), text.split(".")[0] + "."])
            return {"r": answers}
        if payload["mode"] == "narrator_voices":
            return {"v": [[entry[0], "Female voice; bright, quick and eager.", "quick, eager", ""] for entry in payload["voices"]]}
        raise AssertionError(f"Unexpected request mode: {payload['mode']}")


def config():
    return {"analysis": deepcopy(DEFAULTS["analysis"]),
            "provider": {"model": "test-model", "base_url": "https://example.invalid"}}


class KnownSpeakerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        first = build("known-book-1", "The Lantern", [("mara", "Mara", ()), ("keeper", "The keeper", ())], FIRST)
        self.paths = [self.root / "book-1", self.root / "book-2"]
        one = link_analysis(first, fresh_registry(self.paths[0]))
        publish_directory(one, fresh_registry(self.paths[0]), self.paths[0], export_bundle)
        self.mara = next(row for row in one["cast"]["characters"] if row["name"] == "Mara")
        keeper = next(row for row in one["cast"]["characters"] if row["name"] == "The keeper")
        second = build("known-book-2", "The Return", [("keeper", "The keeper", ())], SECOND)
        second["cast"]["characters"][0]["series_identity"] = {
            "character_id": keeper["id"], "basis": "context", "review_required": False,
            "reason": "The same keeper returns.", "evidence": [{"block_id": "b1", "quote": "The keeper returned."}]}
        # The reading met Wren (and Mara, matched to book 1) but gave neither a line.
        second["provenance"]["narrator_read"] = {"ledger": {
            "narration": {"mode": "third", "narrator": None},
            "characters": [
                {"id": "7", "name": "Wren", "aliases": [], "sex": "f", "note": "deckhand on the ferry",
                 "voice": [{"block_id": "b3", "quote": "tugged at the rope"}], "series": None, "changes": []},
                {"id": "8", "name": "Mara", "aliases": [], "sex": "f", "note": "returns from the lighthouse",
                 "voice": [], "series": self.mara["id"], "series_confirmed": True, "changes": []}]}}
        registry = continuation_registry(self.paths[0], second["source"])
        publish_directory(link_analysis(second, registry), registry, self.paths[1], export_bundle,
                          continue_from=self.paths[0])

    def review(self, client):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            changed = review_known_speakers(self.paths[1], config(), lambda settings: client, lambda message: None)
        return changed, output.getvalue()

    def speaker(self, document, block_id):
        segment = next(row for row in document["segments"] if row["block_id"] == block_id and row["kind"] == "dialogue")
        role = next((row for row in document["cast"]["roles"] if row["id"] == segment["role_id"]), None)
        return next((row for row in document["cast"]["characters"] if role and row["id"] == role["character_id"]), None)

    def test_known_characters_not_in_the_cast_are_the_reading_and_the_earlier_books(self):
        document, registry, _ = _completed_directory(self.paths[1])
        absent = absent_characters(document, registry)
        self.assertEqual([(row["name"], row["records"] is not None) for row in absent], [("Wren", False), ("Mara", True)],
                         "the reading's Mara is book 1's, listed once; the keeper is in the cast")
        self.assertEqual([segment["block_id"] for segment, *_ in nominate(document, absent)], ["b2", "b3", "b4"],
                         "only lines with one of them named in their paragraph or the two before")

    def test_a_line_beside_an_earlier_books_character_goes_to_it_with_that_books_voice(self):
        changed, output = self.review(Client())
        self.assertTrue(changed, output)
        document, registry, _ = _completed_directory(self.paths[1])
        mara = self.speaker(document, "b2")
        self.assertEqual((mara["id"], mara["voice_id"]), (self.mara["id"], self.mara["voice_id"]))
        voice = next(row for row in document["cast"]["voices"] if row["id"] == mara["voice_id"])
        self.assertEqual(voice["series_origin"]["kind"], "series_casting_reuse", "book 1's recording is reused")
        self.assertIn("Mara from The Lantern", output)
        wren = self.speaker(document, "b3")
        self.assertEqual(wren["name"], "Wren")
        self.assertTrue(next(row for row in document["cast"]["voices"] if row["id"] == wren["voice_id"])
                        ["description"].startswith("Female"), "a new character's voice is designed")
        self.assertIsNone(self.speaker(document, "b4"), "a line nobody known is named beside stays for a person")
        self.assertEqual([row["name"] for row in document["provenance"]["known_speakers"]], ["Mara", "Wren"])
        validate_registry(registry)

    def test_a_line_the_text_does_not_show_is_theirs_is_left_alone(self):
        before = (self.paths[1] / "manifest.json").read_bytes()
        changed, output = self.review(Client(says=()))
        self.assertFalse(changed)
        self.assertIn("stay for the speaker choice", output)
        self.assertEqual((self.paths[1] / "manifest.json").read_bytes(), before)

    def test_a_person_choosing_speakers_can_choose_an_earlier_books_character(self):
        document, registry, _ = _completed_directory(self.paths[1])
        output = StringIO()
        # Passage 1 (b2): Mara is offered first, named beside it; passage 2: skipped by stopping.
        revised = prompt_speaker_decisions(document, input_stream=Terminal("1\nq\n"), output_stream=output,
                                           registry=registry)
        self.assertIn("Mara (from The Lantern)", output.getvalue())
        self.assertEqual(self.speaker(revised, "b2")["id"], self.mara["id"])
        decision = revised["provenance"]["speaker_decisions"]["decisions"][0]
        self.assertEqual(decision["taken_in"][0]["character"]["id"], self.mara["id"])
        paths = related_directories([self.paths[1]])
        revise_directories({self.paths[1]: revised}, directories=paths, scope="speaker-decisions", progress=lambda _: None,
                           speaker_decision_inputs={self.paths[1]: document})
        saved, registry, _ = _completed_directory(self.paths[1])
        self.assertEqual(self.speaker(saved, "b2")["voice_id"], self.mara["voice_id"])
        validate_registry(registry)

    def test_a_forged_character_taken_in_is_refused(self):
        document, registry, _ = _completed_directory(self.paths[1])
        revised = prompt_speaker_decisions(document, input_stream=Terminal("1\nq\n"), output_stream=StringIO(),
                                           registry=registry)
        revised["provenance"]["speaker_decisions"]["decisions"][0]["taken_in"][0]["voice"]["description"] = "Other."
        with self.assertRaisesRegex(ValueError, "earlier records"):
            revise_directories({self.paths[1]: revised}, directories=related_directories([self.paths[1]]),
                               scope="speaker-decisions", progress=lambda _: None,
                               speaker_decision_inputs={self.paths[1]: document})


if __name__ == "__main__":
    unittest.main()
