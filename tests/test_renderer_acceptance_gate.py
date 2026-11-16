"""One gate decides whether any take may become its passage's accepted take."""

import ast
from pathlib import Path
import tempfile
import unittest

from epub_tts_narrator.renderer.acceptance_gate import refusal, refused_report
from epub_tts_narrator.renderer.take_feedback import record_verdict
from test_renderer_name_shape import performance

RENDERER = Path(__file__).resolve().parents[1] / "epub_tts_narrator" / "renderer"
TEXT = "Marendra nodded."
READING = "M AH0 R EH1 N D R AH0"


def take(heard, *, method="close_match", metadata=None, reasons=(), audio="a" * 64, **report):
    return {"audio_sha256": audio, "engine_metadata": metadata or {},
            "request": {"kind": "speech", "chunk": {"spoken_text": TEXT, "expressive_direction": "",
                                                    "expressive_performance": performance(TEXT, "Marendra", READING)}},
            "report": {"passed": not reasons, "reasons": list(reasons), "expected_text": TEXT, "transcript": heard,
                       "text_confirmation_method": method, **report}}


class AcceptanceGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name) / "render-work"
        self.folder = self.work / "clips" / ("c" * 64)
        self.folder.mkdir(parents=True)

    def test_an_ordinary_take_saying_the_name_in_its_shape_is_accepted(self):
        self.assertIsNone(refusal(take(" Morendra nodded."), self.folder))

    def test_a_rearranged_chosen_name_is_refused_on_every_path(self):
        for method in ("close_match", "heard_variant", "skipped_source_word", "recurring_resolution"):
            with self.subTest(method=method):
                why = refusal(take(" Marnedra nodded.", method=method), self.folder)
                self.assertIn("Marendra heard as marnedra", why)

    def test_a_missing_name_is_refused_only_where_word_checks_were_bypassed(self):
        # Ordinary verification already decided the words: an unplaced name is its call.
        self.assertIsNone(refusal(take(" Nodded."), self.folder))
        for method in ("heard_variant", "skipped_source_word", "recurring_resolution"):
            with self.subTest(method=method):
                self.assertIn("Marendra not heard", refusal(take(" Nodded.", method=method), self.folder))
        best_effort = take(None, metadata={"exhaustion_selection": {}})
        self.assertIn("not heard", refusal(best_effort, self.folder))

    def test_a_take_a_person_marked_bad_is_refused_and_their_own_choice_never_is(self):
        record_verdict(self.work, "r" * 64, {"attempt": 1, "record": take(" Morendra nodded.")}, "bad")
        self.assertEqual(refusal(take(" Morendra nodded."), self.folder), "a person marked it bad")
        chosen = take(" Marnedra nodded.", metadata={"manual_selection": {}})
        self.assertIsNone(refusal(chosen, self.folder))

    def test_an_echo_or_a_tone_is_refused_but_the_filtered_copy_is_not(self):
        echo = take(" Morendra nodded.", history_echo={"heard_share": 0.9})
        self.assertIn("history turn", refusal(echo, self.folder))
        tonal = take(" Morendra nodded.", reasons=["steady_tone"])
        self.assertIn("steady", refusal(tonal, self.folder))
        filtered = take(" Morendra nodded.", reasons=["steady_tone"], metadata={"tone_filter": {}})
        self.assertIsNone(refusal(filtered, self.folder))

    def test_a_refused_new_take_fails_with_the_reason_downstream_steps_know(self):
        report = refused_report(take(" Marnedra nodded.")["report"], "it misread a chosen pronunciation (x)")
        self.assertEqual((report["passed"], report["reasons"]), (False, ["pronunciation_mismatch"]))

    def test_every_writer_of_an_accepted_take_asks_the_gate(self):
        # A person's own choice (take_selection) is theirs; every other path
        # that makes a take the accepted one must call acceptance_gate.refusal.
        exempt = {"take_selection.py"}
        missing = []
        for path in sorted(RENDERER.glob("*.py")):
            if path.name in exempt:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                writes = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                          and getattr(node.func, "id", None) == "write_json" and node.args
                          and "accepted" in ast.unparse(node.args[0])]
                gated = any(isinstance(node, ast.Call) and getattr(node.func, "id", None) == "refusal"
                            for node in ast.walk(function))
                if writes and not gated:
                    missing.append(f"{path.name}:{function.name}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
