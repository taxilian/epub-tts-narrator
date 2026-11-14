"""A chosen pronunciation heard with its syllables rearranged is a wrong rendition.

The cases follow takes a listener heard: a name read "seh-NED-ruh" is
transcribed "Sinedra" or "Senedra" when it is right and "Sendera" or
"Sindhara" when the engine moved its consonants, which every word-level
check accepted because Double Metaphone ignores vowels.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.performance_plan import text_sha256
from epub_tts_narrator.renderer.common import read_json, write_json
from epub_tts_narrator.renderer.name_shape import (
    REASON, SWEEP_FILENAME, check, rearranged, reading_shape, retire_misread, spelling_shape)
from epub_tts_narrator.renderer.quality import validate_audio, validation_key
import test_renderer_reference_consensus as consensus

READING = "S EH2 N EH1 D R AH0"


def performance(text, name, phonemes=READING, alphabet="cmu_arpabet"):
    start = text.index(name)
    field = "replacement" if alphabet == "respelling" else "phonemes"
    return {"version": 1, "text_sha256": text_sha256(text),
            "controls": [{"kind": "pronunciation", "start": start, "end": start + len(name), "text": name,
                          "alphabet": alphabet, field: phonemes}]}


class ShapeTests(unittest.TestCase):
    def test_a_name_spelled_differently_keeps_its_shape(self):
        for heard in ("Sinedra", "Senedra", "Sinidra", "Synedra", "Sanhedra", "Sinatra", "Ce'Nedra"):
            with self.subTest(heard=heard):
                self.assertFalse(rearranged(READING, heard))
                self.assertEqual(spelling_shape(heard), reading_shape(READING))

    def test_a_consonant_moved_across_a_vowel_is_rearranged(self):
        for heard in ("Sendera", "Sendara", "Sindera", "Sindhara", "Sendaria", "Cinderia"):
            with self.subTest(heard=heard):
                self.assertTrue(rearranged(READING, heard))
        self.assertEqual(reading_shape(READING), [1, 1, 2, 0])
        self.assertEqual(spelling_shape("Sendera"), [1, 2, 1, 0])

    def test_what_the_shape_cannot_judge_is_left_alone(self):
        # Other consonants, a dropped syllable or another word: the ordinary checks decided these.
        for heard in ("Zira", "Sandra", "sending", "Cinderius"):
            with self.subTest(heard=heard):
                self.assertFalse(rearranged(READING, heard))
        self.assertEqual(reading_shape("M AY1 L | K IY1"), reading_shape("M AY1 L K IY1"))


class CheckTests(unittest.TestCase):
    def test_the_heard_word_in_place_of_the_name_is_checked(self):
        text = "Ce’Nedra demanded, staring at the parchment."
        found = check(text, "Sendera demanded, staring at the parchment.", performance(text, "Ce’Nedra"))
        self.assertEqual([(item["text"], item["heard"]) for item in found], [("Ce’Nedra", "sendera")])
        self.assertEqual(check(text, "Sinedra demanded, staring at the parchment.", performance(text, "Ce’Nedra")), [])
        self.assertEqual(check(text, text, performance(text, "Ce’Nedra")), [])

    def test_possessives_compare_the_name_itself(self):
        text = "They saw Ce’Nedra’s face."
        perf = performance(text, "Ce’Nedra’s", READING + " Z")
        self.assertEqual(len(check(text, "They saw Sendera's face.", perf)), 1)
        self.assertEqual(check(text, "They saw Sinedra's face.", perf), [])

    def test_names_in_a_long_paragraph_are_found_where_they_were_heard(self):
        filler = " ".join(["The road ran on under the trees and the wagons creaked."] * 30)
        text = filler + " Then Ce’Nedra laughed. " + filler
        heard = filler + " Then Sendara laughed. " + filler
        self.assertEqual([item["heard"] for item in check(text, heard, performance(text, "Ce’Nedra"))], ["sendara"])

    def test_respellings_and_passages_without_readings_are_not_checked(self):
        text = "Ce’Nedra swallowed hard."
        self.assertEqual(check(text, "Sendera swallowed hard.", performance(text, "Ce’Nedra", "Sen Edra", "respelling")), [])
        self.assertEqual(check(text, "Sendera swallowed hard.", None), [])


class VerificationTests(unittest.TestCase):
    setUp = consensus.ReferenceConsensusTests.setUp
    write_audio = consensus.ReferenceConsensusTests.write_audio
    TEXT = "Ce’Nedra swallowed hard and looked away."
    POLICY = {"max_close_mismatch_fraction": 0.25}

    def test_a_close_transcript_with_the_name_rearranged_is_rejected(self):
        perf = performance(self.TEXT, "Ce’Nedra")
        report = validate_audio(self.audio, self.TEXT, consensus.PrimaryVerifier(
            "Sendera swallowed hard and looked away."), self.POLICY, performance=perf)
        self.assertFalse(report["passed"])
        self.assertEqual(report["reasons"], [REASON])
        # Double Metaphone alone accepted it: "Sendera" and "Sinedra" share their consonants.
        self.assertEqual(report["name_shape"]["passed_method"], "double_metaphone")
        self.assertEqual(report["name_shape"]["words"][0]["heard"], "sendera")
        right = validate_audio(self.audio, self.TEXT, consensus.PrimaryVerifier(
            "Sinedra swallowed hard and looked away."), self.POLICY, performance=perf)
        self.assertTrue(right["passed"], right)

    def test_without_a_chosen_reading_nothing_changes(self):
        report = validate_audio(self.audio, self.TEXT, consensus.PrimaryVerifier(
            "Sendera swallowed hard and looked away."), self.POLICY)
        self.assertTrue(report["passed"], report)
        verifier = consensus.PrimaryVerifier()
        plain = {"version": 1, "text_sha256": text_sha256(self.TEXT), "controls": []}
        # Only passages with a chosen reading carry the check's version in their validation key.
        self.assertEqual(validation_key(self.TEXT, verifier, self.POLICY, performance=plain),
                         validation_key(self.TEXT, verifier, self.POLICY, performance=plain))
        self.assertNotEqual(validation_key(self.TEXT, verifier, self.POLICY, performance=performance(self.TEXT, "Ce’Nedra")),
                            validation_key(self.TEXT, verifier, self.POLICY, performance=plain))


class SweepTests(unittest.TestCase):
    TEXT = "Ce’Nedra swallowed hard."

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name) / "render-work"

    def accepted(self, name, heard, *, metadata=None, folder=None):
        directory = folder or self.work / "clips" / (name * 64)[:64]
        directory.mkdir(parents=True)
        record = {"attempt": 1, "audio_sha256": name * 64, "audio_file": "take.wav",
                  "request": {"kind": "speech", "chunk": {"spoken_text": self.TEXT, "expressive_direction": "",
                                                          "expressive_performance": performance(self.TEXT, "Ce’Nedra")}},
                  "engine_metadata": metadata or {},
                  "report": {"passed": True, "reasons": [], "expected_text": self.TEXT, "transcript": heard,
                             "text_confirmation_method": "close_match"}}
        write_json(directory / "accepted.json", record)
        return directory

    def test_takes_in_use_heard_rearranged_are_retired_once_and_kept_otherwise(self):
        misread = self.accepted("a", "Sendera swallowed hard.")
        right = self.accepted("b", "Sinedra swallowed hard.")
        chosen = self.accepted("c", "Sendera swallowed hard.", metadata={"manual_selection": {}})
        part = self.accepted("d", "Sindhara swallowed hard.",
                             folder=self.work / "clips" / ("e" * 64) / "parts" / ("d" * 64))
        liked = self.accepted("f", "Sendera swallowed hard.")
        write_json(self.work / "take-feedback.json", {"format": "audiobook-take-feedback", "version": 1, "takes": {
            "x:1": {"request_hash": "f" * 64, "attempt": 1, "verdict": "good", "audio_sha256": "f" * 64}}})
        self.assertEqual(sorted(retire_misread(self.work)), sorted([misread, part]))
        for folder in (misread, part):
            self.assertFalse((folder / "accepted.json").exists())
            evidence = read_json(next(folder.glob("name-shape-*.json")))
            self.assertEqual(evidence["evidence"]["reason"], REASON)
            self.assertEqual(read_json(folder / evidence["retired"][0])["audio_sha256"][:1], folder.name[:1])
        for folder in (right, chosen, liked):
            self.assertTrue((folder / "accepted.json").exists())
        self.assertEqual(read_json(self.work / SWEEP_FILENAME)["version"], 1)

    def test_a_later_sweep_reads_only_takes_accepted_since(self):
        old = self.accepted("a", "Sendera swallowed hard.")
        retire_misread(self.work)
        self.accepted("b", "Sendera swallowed hard.")
        # A take accepted before the last sweep is not read again.
        stale = self.accepted("c", "Sendara swallowed hard.")
        os.utime(stale / "accepted.json", (1, 1))
        self.assertEqual(retire_misread(self.work), [self.work / "clips" / ("b" * 64)])
        self.assertTrue((stale / "accepted.json").exists())
        self.assertFalse((old / "accepted.json").exists())


if __name__ == "__main__":
    unittest.main()
