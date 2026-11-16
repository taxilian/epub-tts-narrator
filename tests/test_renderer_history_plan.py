"""One decision for what history the next take of a failing passage hears."""

from unittest.mock import patch
import unittest

from epub_tts_narrator.renderer import continuity
from epub_tts_narrator.renderer.history_echo import apply_recovery
from epub_tts_narrator.renderer.history_plan import plan


def attempt(reasons=(), *, applied=None, echo=False):
    metadata = {"history_recovery": {"version": 1, "requested": applied, "applied": applied}} if applied else {}
    report = {"passed": not reasons, "reasons": list(reasons)}
    if echo:
        report["history_echo"] = {"heard_share": 0.9}
    return {"report": report, "engine_metadata": metadata}


def mode(records, *, readings=True):
    return plan(records, "Marendra nodded.", has_readings=readings, heard=[("h", "An earlier line.")])[0]


class HistoryPlanTests(unittest.TestCase):
    def test_ordinary_failures_keep_the_chapter_history(self):
        self.assertIsNone(mode([]))
        self.assertIsNone(mode([attempt(["text_mismatch"])]))

    def test_after_a_tone_a_line_keeps_only_its_chosen_names_turn_or_none(self):
        self.assertEqual(mode([attempt(["steady_tone"])]), "reading")
        self.assertEqual(mode([attempt(["steady_tone"])], readings=False), "none")

    def test_a_tone_again_under_the_reading_turn_keeps_it(self):
        records = [attempt(["steady_tone"]), attempt(["steady_tone"], applied="reading")]
        self.assertEqual(mode(records), "reading")

    def test_the_reading_turn_is_dropped_once_it_was_echoed_or_said_nothing(self):
        base = [attempt(["steady_tone"])]
        self.assertEqual(mode(base + [attempt(["text_mismatch"], applied="reading", echo=True)]), "none")
        self.assertEqual(mode(base + [attempt(["silent_audio"], applied="reading")]), "none")

    def test_takes_that_said_nothing_cut_history_but_keep_the_names_turn_first(self):
        silent = [attempt(["audio_too_quiet"]), attempt(["excessive_silence"])]
        self.assertIsNone(mode(silent[:1]))
        self.assertEqual(mode(silent), "reading")
        self.assertEqual(mode(silent, readings=False), "none")

    def test_a_line_with_a_chosen_name_hears_the_turn_saying_it_where_the_ladder_would_leave_none(self):
        echoes = [attempt(["text_mismatch"], echo=True), attempt(["text_mismatch"], applied="role_tag", echo=True),
                  attempt(["text_mismatch"], applied="streaming", echo=True)]
        self.assertEqual(mode(echoes), "reading")
        self.assertEqual(mode(echoes, readings=False), "none")
        # Takes misreading the name with no history do not keep it there.
        stuck = echoes + [attempt(["pronunciation_mismatch"], applied="none")] * 3
        self.assertEqual(mode(stuck), "reading")
        self.assertEqual(mode(echoes + [attempt(["text_mismatch"], applied="reading", echo=True)]), "none")

    def test_an_echo_climbs_the_ladder_but_never_after_a_tone(self):
        self.assertEqual(mode([attempt(["text_mismatch"], echo=True)]), "role_tag")
        self.assertEqual(mode([attempt(["text_mismatch"], echo=True), attempt(["steady_tone"])]), "reading")


class KeptStartTests(unittest.TestCase):
    """A take completion's kept start survives every recovery mode."""

    def test_no_history_still_keeps_the_kept_start(self):
        history = [("chapter turn", [[1]]), ("kept start", [[2]])]
        self.assertEqual(apply_recovery("none", "t", history, (), kind=None, continuation=True)[2], history[-1:])
        self.assertEqual(apply_recovery("none", "t", history, (), kind=None)[2], [])
        # A ladder rung that cannot run falls to none, still keeping the kept start.
        self.assertEqual(apply_recovery("streaming", "t", history, (), kind=None, continuation=True)[2][-1],
                         history[-1])

    def test_compose_keeps_the_names_turn_and_the_kept_start(self):
        voice = {"asset_hash": "n"}
        chapter = type("Chapter", (), {"voices_by_asset": {"n": voice}})()
        takes = [{"marker": {"speakers": ["n"], "text": "<|speaker:0|> first"}, "path": "a", "readings": frozenset()},
                 {"marker": {"speakers": ["n"], "text": "<|speaker:0|> Marendra"}, "path": "b",
                  "readings": frozenset({"m"})}]
        kept = {"text": "<|speaker:0|> kept start", "codes": [[3]], "speakers": ["n"]}

        def composed(mode):
            token = continuity._CURRENT.set({"chapter": chapter, "takes": takes, "readings": frozenset({"m"})})
            try:
                with continuity.history_recovery(mode), continuity.continuation_turn(kept):
                    return [text for text, _ in continuity.compose("<|speaker:0|> line", [voice])[1]]
            finally:
                continuity._CURRENT.reset(token)
        with patch("epub_tts_narrator.renderer.continuity_codes.load_codes", return_value={"codes": [[1]]}):
            self.assertEqual(composed("reading"), ["<|speaker:0|> Marendra", "<|speaker:0|> kept start"])
            self.assertEqual(composed("none"), ["<|speaker:0|> kept start"])
            self.assertEqual(composed(None), ["<|speaker:0|> first", "<|speaker:0|> kept start"])


if __name__ == "__main__":
    unittest.main()
