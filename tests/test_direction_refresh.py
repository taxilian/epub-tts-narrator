"""Direction-only refresh: new performance directions on a published book, speakers kept."""

import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import tempfile
from threading import Lock
import unittest
from unittest.mock import patch

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.direction_policy import VERSION as DIRECTION_POLICY
from epub_tts_narrator.direction_refresh import refresh_directions
from epub_tts_narrator.performance_plan import text_sha256, validate_performance
from epub_tts_narrator.pipeline import CachedRequests, fingerprint
from epub_tts_narrator.series import link_analysis
from test_attribution_verification import risk_document


SCENE = [
    ("b1", "ch1", "Mara laughed. “Fine. But you never listen.”", [("narration", 14), ("mara", None)]),
    ("b2", "ch1", "The door slammed somewhere below.", [("narration", None)]),
    ("b3", "ch1", "“Who is there?” Tom whispered.", [("tom", 15), ("narration", None)]),
    ("b4", "ch2", "“Run,” Mara said.", [("mara", 6), ("narration", None)]),
]


def scene_document():
    """A two-chapter published-shape book with old directions and one old control."""
    document = risk_document(False)
    roles = {role["id"]: role for role in document["cast"]["roles"]}
    blocks, segments = [], []
    for block_id, chapter_id, text, pieces in SCENE:
        blocks.append({"id": block_id, "chapter_id": chapter_id, "text": text,
                       "source": {"href": f"text/{chapter_id}.xhtml", "element_id": None, "tag": "p"}})
        start = 0
        for role_id, end in pieces:
            end = len(text) if end is None else end
            role = roles["narrator_main" if role_id == "narration" else role_id]
            segments.append({"id": f"{block_id}:{start}:{end}", "block_id": block_id, "start": start, "end": end,
                             "text": text[start:end], "kind": "narration" if role_id == "narration" else "dialogue",
                             "role_id": role["id"], "voice_id": role["voice_id"],
                             "delivery": {"direction": "", "evidence": []},
                             "attribution": {"basis": "explicit", "reason": "Fixture.", "evidence": [],
                                             "candidates": [], "review_required": False}})
            start = end
    blocks[0]["inline_spans"] = [{"start": 29, "end": 34, "kind": "emphasis", "origin": "tag"}]
    document["source"]["blocks"] = blocks
    document["segments"] = segments

    def requote(value):
        if isinstance(value, dict):
            if set(value) >= {"block_id", "quote"}:
                value.update(block_id="b1", quote="Mara")
            for item in value.values():
                requote(item)
        elif isinstance(value, list):
            for item in value:
                requote(item)
    requote(document["cast"])
    document["source"]["chapters"] = [
        {"id": chapter, "title": title, "source_href": f"text/{chapter}.xhtml",
         "block_ids": [block["id"] for block in blocks if block["chapter_id"] == chapter]}
        for chapter, title in (("ch1", "Chapter One"), ("ch2", "Chapter Two"))]
    mara = segments[1]
    mara["delivery"] = {"direction": "calm", "evidence": [{"block_id": "b1", "quote": "Mara laughed."}]}
    tom = segments[3]
    tom["performance_controls"] = validate_performance(
        {"version": 1, "text_sha256": text_sha256(tom["text"]),
         "controls": [{"kind": "delivery", "start": 1, "end": 4, "text": "Who", "tag": "emphasis"}]}, tom["text"])
    document["provenance"]["inline_performance"] = {"version": 1, "policy": "remove_only", "counts": {"kept": 1},
                                                    "batches": [], "outcomes": []}
    document["provenance"]["source_fingerprint"] = fingerprint(document["source"])
    return document


class DirectorClient:
    """Directs every marked line; answers the inline review with "keep"."""

    identity = {"model": "test", "base_url": "https://example.invalid"}

    def __init__(self, model="test-model", base_url="https://example.invalid", skip_once=None, **kwargs):
        self.model, self.base_url = model, base_url
        self.calls, self.lock = [], Lock()
        self.skip_once = skip_once

    def complete(self, system, payload):
        with self.lock:
            self.calls.append(deepcopy(payload))
        if payload["mode"] == "inline_performance_review":
            ids = [proposal["id"] for row in payload["rows"] for proposal in row["proposals"]] \
                if "rows" in payload else re.findall(r'"id": "(r\d+)"', json.dumps(payload))
            return {"decisions": [{"id": identifier, "verdict": "keep"} for identifier in dict.fromkeys(ids)]}
        assert payload["mode"] == "direction_refresh", payload["mode"]
        markers = re.findall(r"‹(q\d+) ([^›]*)›", payload["text"])
        answers = {"q1": ["q1", "flat, weary", [["But you", "hurt, rising"]]],
                   "q2": ["q2", "hushed, afraid"], "q3": ["q3", "urgent, breathless"]}
        rows = [answers[qid] for qid, _ in markers if not (qid == self.skip_once and "correction" not in payload)]
        notes = [["2", "read", "tense, hushed"], ["q2", "whisper", "Who is there?"]]
        return {"q": rows, "p": [note for note in notes if f"‹{note[0]} " in payload["text"]
                                 or (note[0] == "2" and "\n2|" in "\n" + payload["text"])]}


class DirectionRefreshTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)

    def refresh(self, document, client=None, **options):
        client = client or DirectorClient()
        requests = CachedRequests(client, self.cache, repair_attempts=1)
        revised, report = refresh_directions(document, requests, **options)
        return revised, report, client

    def test_lines_get_new_directions_and_in_line_changes_while_speakers_stay(self):
        document = scene_document()
        original = deepcopy(document)
        revised, report, client = self.refresh(document)

        self.assertEqual(document, original)
        self.assertEqual(readiness(revised)["errors"], [])
        asked = client.calls[0]
        self.assertIn("‹q1 Mara›", asked["text"])
        self.assertIn("‹q2 Tom›", asked["text"])
        self.assertNotIn("cast", asked)
        # The narration about each line's own speaker travels with the line.
        self.assertEqual(asked["cues"], {"q1": "Mara laughed.", "q2": "Tom whispered."})
        self.assertEqual([(row["id"], row["role_id"], row["voice_id"]) for row in revised["segments"]],
                         [(row["id"], row["role_id"], row["voice_id"]) for row in document["segments"]])
        rows = {row["id"]: row for row in revised["segments"]}
        mara = rows["b1:14:43"]
        self.assertEqual(mara["delivery"], {"direction": "flat, weary", "evidence": []})
        controls = mara["performance_controls"]["controls"]
        scoped = [(c["tag"], c["text"], c.get("direction")) for c in controls if c["kind"] == "delivery"]
        self.assertIn(("style", "But you never listen.”", "hurt, rising"), scoped)
        self.assertIn(("emphasis", "never", ""), scoped)
        # "Mara laughed." beside her own line: a laugh as she starts it.
        self.assertIn({"kind": "vocalization", "offset": 1, "sound": "laugh"}, controls)
        self.assertEqual(rows["b2:0:33"]["delivery"]["direction"], "tense, hushed")
        tom = rows["b3:0:15"]
        self.assertEqual(tom["delivery"]["direction"], "hushed, afraid")
        # The old emphasis is replaced: "Tom whispered." whispers his whole line, and the
        # model's narrower whisper inside it is the one dropped.
        self.assertEqual([(c["tag"], c["text"]) for c in tom["performance_controls"]["controls"]],
                         [("whisper", "“Who is there?”")])
        self.assertEqual(revised["provenance"]["inline_performance"], document["provenance"]["inline_performance"])
        self.assertEqual((report["lines"], report["directed"], report["changes"], report["narration_reads"],
                          report["italic_emphasis"], report["cue_sounds"], report["cue_scoped"]), (3, 3, 1, 1, 1, 1, 1))
        self.assertEqual(report["review"], {"direction": 4})

    def test_a_chapter_refresh_asks_only_for_its_lines_with_book_wide_ids(self):
        document = scene_document()
        revised, report, client = self.refresh(document, block_ids=["b4"])

        self.assertEqual(len(client.calls), 1)
        self.assertIn("‹q3 Mara›", client.calls[0]["text"])
        self.assertNotIn("‹q1", client.calls[0]["text"])
        self.assertEqual(revised["segments"][:5], document["segments"][:5])
        self.assertEqual(revised["segments"][5]["delivery"]["direction"], "urgent, breathless")
        self.assertEqual(report["lines"], 1)

    def test_a_reply_that_skips_a_line_is_asked_again(self):
        _, report, client = self.refresh(scene_document(), client=DirectorClient(skip_once="q2"))
        modes = [call["mode"] for call in client.calls if call["mode"] == "direction_refresh"]
        self.assertEqual(len(modes), 2)
        self.assertIn("correction", client.calls[1])
        self.assertEqual(report["directed"], 3)

    def test_notes_are_sent_as_hints_only_when_given(self):
        _, _, plain = self.refresh(scene_document())
        _, report, hinted = self.refresh(scene_document(), notes="Mara is sarcastic.")
        self.assertNotIn("hints", plain.calls[0])
        self.assertEqual(hinted.calls[0]["hints"], "Mara is sarcastic.")
        self.assertTrue(report["hints"])

    def test_pronunciation_controls_are_kept_and_block_new_notes_on_that_line(self):
        document = scene_document()
        tom = document["segments"][3]
        tom["performance_controls"] = validate_performance(
            {"version": 1, "text_sha256": text_sha256(tom["text"]),
             "controls": [{"kind": "pronunciation", "start": 8, "end": 13, "text": "there",
                           "alphabet": "cmu_arpabet", "phonemes": "DH EH1 R"},
                          {"kind": "delivery", "start": 1, "end": 4, "text": "Who", "tag": "emphasis"}]},
            tom["text"])
        revised, report, _ = self.refresh(document)
        tom = revised["segments"][3]
        self.assertEqual([c["kind"] for c in tom["performance_controls"]["controls"]], ["pronunciation"])
        self.assertEqual(tom["delivery"]["direction"], "hushed, afraid")
        self.assertEqual(report["kept_pronunciation_rows"], 1)
        self.assertEqual(readiness(revised)["errors"], [])


    def test_a_typed_direction_is_kept_and_its_line_is_not_asked_about(self):
        from epub_tts_narrator.attribution_edits import edit_direction
        document = scene_document()
        document["segments"][1] = edit_direction(document["segments"][1], "my own reading")
        revised, report, client = self.refresh(document)
        self.assertNotIn("‹q1 Mara›", client.calls[0]["text"])
        mara = {row["id"]: row for row in revised["segments"]}["b1:14:43"]
        self.assertEqual((mara["delivery"]["direction"], mara["delivery"]["basis"]), ("my own reading", "user"))
        self.assertNotIn("performance_controls", mara)
        self.assertEqual(report["kept_typed_directions"], 1)
        self.assertEqual(report["policy"], DIRECTION_POLICY)


class LeftoverSpokenTextTests(DirectionRefreshTests.__base__):
    def test_a_line_that_loses_its_only_control_loses_the_spoken_text_written_for_it(self):
        document = scene_document()
        run = document["segments"][5]
        run["performance_controls"] = validate_performance(
            {"version": 1, "text_sha256": text_sha256(run["text"]),
             "controls": [{"kind": "delivery", "start": 1, "end": 4, "text": "Run", "tag": "emphasis"}]}, run["text"])
        run["spoken_text"] = run["text"]
        with tempfile.TemporaryDirectory() as cache:
            revised, _ = refresh_directions(document, CachedRequests(DirectorClient(), cache), block_ids=["b4"])
        self.assertEqual(readiness(revised)["errors"], [])
        self.assertNotIn("performance_controls", revised["segments"][5])
        self.assertNotIn("spoken_text", revised["segments"][5])


class TerminalIO(io.StringIO):
    def isatty(self):
        return True


class DirectionRefreshCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.book = Path(temporary.name).resolve() / "book"
        registry = fresh_registry(self.book)
        publish_directory(link_analysis(scene_document(), registry), registry, self.book, export_bundle)

    def invoke(self, *arguments, client=None, tty=False, input_text=""):
        config = {"analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {"model": "test-model", "base_url": "https://example.invalid"}}
        client = client or DirectorClient()
        output, errors = (TerminalIO() if tty else io.StringIO()), io.StringIO()
        stdin = TerminalIO(input_text) if tty else io.StringIO(input_text)
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client),
              patch("sys.stdin", stdin),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["refresh-directions", str(self.book), *map(str, arguments)])
        return code, output.getvalue(), errors.getvalue(), client

    def test_report_only_leaves_the_book_and_every_run_asks_afresh(self):
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, first = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertIn("3/3 spoken line(s) directed", output)
        self.assertIn("b1:14:43: calm -> flat, weary", output)
        self.assertIn("Report only: no changes applied.", output)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)
        self.assertTrue(first.calls)

        code, output, errors, second = self.invoke("--apply")
        self.assertEqual(code, 0, errors)
        self.assertEqual(len(second.calls), len(first.calls))
        document, _, _ = _completed_directory(self.book)
        rows = {row["id"]: row for row in document["segments"]}
        self.assertEqual(rows["b1:14:43"]["delivery"]["direction"], "flat, weary")
        self.assertEqual(rows["b1:14:43"]["role_id"], "mara")
        self.assertEqual(document["provenance"]["direction_refreshes"][-1]["directed"], 3)
        self.assertIn("Original analysis files and revision journal:", output)
        self.assertTrue((self.book / ".analysis-revisions").is_dir())

    def test_a_terminal_run_asks_before_saving_and_resume_replays_only_on_request(self):
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, first = self.invoke(tty=True, input_text="a\n")
        self.assertEqual(code, 0, errors)
        self.assertIn("[A] apply the new directions / [L] leave the book unchanged", output)
        self.assertNotEqual((self.book / "analysis.json").read_bytes(), before)
        run_id = re.search(r"Fresh direction run: ([0-9a-f]{32})", errors).group(1)

        code, _, errors, replay = self.invoke("--resume", run_id)
        self.assertEqual(code, 0, errors)
        self.assertEqual(replay.calls, [])
        self.assertIn(f"Resuming direction run: {run_id}", errors)

        code, _, errors, _ = self.invoke("--resume", "0" * 32)
        self.assertEqual(code, 1)
        self.assertIn("Unknown direction run ID", errors)

    def test_a_terminal_answer_of_leave_changes_nothing(self):
        before = (self.book / "analysis.json").read_bytes()
        code, output, errors, _ = self.invoke("--chapter", "2", tty=True, input_text="l\n")
        self.assertEqual(code, 0, errors)
        self.assertIn("No changes applied.", output)
        self.assertEqual((self.book / "analysis.json").read_bytes(), before)

    def test_unknown_chapters_are_refused_before_any_request(self):
        code, _, errors, client = self.invoke("--chapter", "9")
        self.assertEqual(code, 1)
        self.assertIn("Unknown chapter number(s): 9", errors)
        self.assertEqual(client.calls, [])


class AnalyzeRefreshesOutdatedDirectionsTests(unittest.TestCase):
    """Rerunning analyze on a saved book directs its lines again when older rules made them."""

    def publish(self, *, refreshed):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        book = Path(temporary.name).resolve() / "book"
        document = scene_document()
        if refreshed:
            # A whole-book refresh made before direction rules were versioned.
            document["provenance"]["direction_refreshes"] = [{"version": 2, "lines": 3, "chapters": None}]
        registry = fresh_registry(book)
        publish_directory(link_analysis(document, registry), registry, book, export_bundle)
        return book

    def analyze(self, book):
        config = {**deepcopy(DEFAULTS), "analysis": deepcopy(DEFAULTS["analysis"]),
                  "provider": {**deepcopy(DEFAULTS["provider"]), "model": "test-model",
                               "base_url": "https://example.invalid"}}
        client = DirectorClient()
        output, errors = io.StringIO(), io.StringIO()
        with (patch("epub_tts_narrator.cli.load_config", return_value=config),
              patch("epub_tts_narrator.cli.OpenAICompatibleClient", return_value=client),
              patch("sys.stdin", io.StringIO()),
              contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors)):
            code = main(["analyze", str(book)])
        return code, output.getvalue(), errors.getvalue(), client

    def test_outdated_directions_are_refreshed_once(self):
        book = self.publish(refreshed=True)
        code, output, errors, client = self.analyze(book)
        self.assertIn(code, (0, 2), errors + output)
        self.assertIn("made with older direction rules", output)
        self.assertTrue(any(call["mode"] == "direction_refresh" for call in client.calls))
        document, _, _ = _completed_directory(book)
        self.assertEqual(document["provenance"]["direction_refreshes"][-1]["policy"], DIRECTION_POLICY)
        rows = {row["id"]: row for row in document["segments"]}
        self.assertEqual(rows["b1:14:43"]["delivery"]["direction"], "flat, weary")
        self.assertEqual(rows["b1:14:43"]["role_id"], "mara")

        published = (book / "manifest.json").read_bytes()
        code, output, errors, again = self.analyze(book)
        self.assertIn(code, (0, 2), errors + output)
        self.assertEqual(again.calls, [])
        self.assertEqual((book / "manifest.json").read_bytes(), published)

    def test_an_interrupted_refresh_replays_its_answers_on_the_next_analyze(self):
        book = self.publish(refreshed=True)
        with patch("epub_tts_narrator.direction_refresh_command.revise_directories",
                   side_effect=OSError("disk interrupted")):
            code, output, errors, first = self.analyze(book)
        self.assertEqual(code, 1)
        self.assertIn("rerunning analyze", errors)
        self.assertTrue(first.calls)
        code, output, errors, second = self.analyze(book)
        self.assertIn(code, (0, 2), errors + output)
        self.assertEqual(second.calls, [])
        document, _, _ = _completed_directory(book)
        self.assertEqual(document["provenance"]["direction_refreshes"][-1]["policy"], DIRECTION_POLICY)

    def test_a_never_refreshed_consensus_book_keeps_its_directions(self):
        book = self.publish(refreshed=False)
        before = (book / "manifest.json").read_bytes()
        code, output, errors, client = self.analyze(book)
        self.assertIn(code, (0, 2), errors + output)
        self.assertIn("run refresh-directions on this book", output)
        self.assertEqual(client.calls, [])
        self.assertEqual((book / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
