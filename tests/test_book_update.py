"""Serially published books: rerunning analyze with more chapters updates the book."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.book_update import compare_sources
from epub_tts_narrator.cli import main, read_json, source_from
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.bundle import chunk_identity_fingerprint, load_bundle
from epub_tts_narrator.series import validate_registry
from shared_fixtures import make_serial_epub, snapshot
from test_pipeline import FakeClient

CHAPTER_ONE = ("One", ["Mara, the lighthouse keeper, said softly, “Hello.”", "They left."])
CHAPTER_TWO = ("Two", ["Mara said, “The lamp is lit.”"])
CHAPTER_THREE = ("Three", ["Tobin the ferryman called out, “Wait for me.”",
                           "Mara, the lighthouse keeper, answered, “Hurry.”"])
LONG_ONE = ("One", ["Mara, the lighthouse keeper, said softly, “Hello.”", "They left.", "The night was cold.",
                    "Waves broke on the rocks.", "Mara said, “Bring the oil.”", "Morning came.",
                    "The tide turned.", "Boats came home."])
# Source context, beyond the shared name, that links a returning character.
CONTEXT = {"Mara": "Mara, the lighthouse keeper", "Tobin": "Tobin the ferryman"}


class SerialClient(FakeClient):
    """Discovers Mara and Tobin; links Mara to her prior record with source context."""

    def __init__(self, model, base_url, *, misread=None, **kwargs):
        super().__init__()
        self.model, self.base_url = model, base_url
        self.identity = {"model": model, "base_url": base_url}
        # Text whose speaker this client reads as plain narration.
        self.misread = misread

    def complete(self, system, payload):
        if payload.get("mode") == "speech_spelling_scan":
            self.calls.append((system, deepcopy(payload)))
            return {"segments": [{"segment_id": row["segment_id"], "edits": []}
                                 for row in payload["target_segments"]]}
        return super().complete(system, payload)

    def discover(self, payload):
        if payload["known_cast"]["characters"]:
            return self.patch()
        ids, units = payload["available_ids"], payload["target_units"]
        prior = {character["name"]: character
                 for character in payload.get("series_cast", {}).get("characters", [])}
        characters, roles, voices = [], [], []
        names = [name for name in ("Mara", "Tobin") if any(name in unit["text"] for unit in units)]
        for index, name in enumerate(names):
            unit = next(unit for unit in units if name in unit["text"])
            character = {"id": ids["characters"][index], "name": name, "aliases": [],
                         "description": f"{name}, who speaks in this story.",
                         "evidence": [{"block_id": unit["block_id"], "quote": name}],
                         "voice_id": ids["voices"][index]}
            context = CONTEXT[name]
            keeper = next((unit for unit in units if context in unit["text"]), None)
            if name in prior and keeper is not None:
                character["series_identity"] = {
                    "character_id": prior[name]["id"], "basis": "context", "review_required": False,
                    "reason": "The same described role establishes continuity beyond the name.",
                    "evidence": [{"block_id": keeper["block_id"], "quote": context}]}
            characters.append(character)
            roles.append({"id": ids["roles"][index], "kind": "dialogue", "character_id": character["id"],
                          "voice_id": character["voice_id"], "description": f"Dialogue spoken by {name}."})
            voices.append({"id": character["voice_id"], "description": f"Proposed distinct voice for {name}.",
                           "traits": [{"name": "timbre", "value": "light and clear" if name == "Mara" else "deep",
                                       "basis": "casting", "evidence": []}]})
        return self.patch(characters=characters, roles=roles, voices=voices)

    def reconcile(self, payload):
        if not any(character["name"] == "Mara" for character in payload["known_cast"]["characters"]):
            return self.patch()
        return super().reconcile(payload)

    def attribute(self, payload):
        cast = payload["final_cast"]
        dialogue = {character["name"]: next(role for role in cast["roles"] if role["kind"] == "dialogue"
                                            and role["character_id"] == character["id"])
                    for character in cast["characters"]}
        units = []
        for unit in payload["target_units"]:
            text = unit["text"]
            speaker = next((name for name in ("Tobin", "Mara") if name in text and name in dialogue), None)
            if self.misread is not None and self.misread in text:
                speaker = None
            start, finish = text.find("“"), text.find("”")
            spans = [(text, "narration")]
            if speaker is not None and start >= 0 and finish > start:
                spans = [(text[:start], "narration"), (text[start:finish + 1], "dialogue"),
                         (text[finish + 1:], "narration")]
            pieces = []
            for exact, kind in spans:
                if not exact:
                    continue
                pieces.append({
                    "text": exact, "kind": kind,
                    "role_id": dialogue[speaker]["id"] if kind == "dialogue" else "narrator_main",
                    "basis": "explicit", "reason": f"The narration names {speaker} as speaker.",
                    "evidence": [{"block_id": unit["block_id"], "quote": exact}] if kind == "dialogue" else [],
                    "candidates": [], "review_required": False,
                    "delivery": {"direction": "", "evidence": []},
                })
            units.append({"unit_id": unit["unit_id"], "segments": pieces})
        return {"units": units}


class BookUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = self.root / "epub-tts-narrator.toml"
        self.config.write_text(
            '[provider]\nmodel = "fixture-model"\nbase_url = "http://offline.invalid/v1"\n'
            '[analysis]\nidentity_consensus = false\nattribution_review = false\n'
            'repair_attempts = 0\nattribution_workers = 1\n', encoding="utf-8")
        self.clients = []
        self.book = self.root / "serial"
        self.misread = None

    def client(self, *args, **kwargs):
        client = SerialClient(*args, misread=self.misread, **kwargs)
        self.clients.append(client)
        return client

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        previous = Path.cwd()
        os.chdir(self.root)
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                 patch("epub_tts_narrator.cli.OpenAICompatibleClient", side_effect=self.client):
                try:
                    # The default ./epub-tts-narrator.toml is this test's config.
                    status = main([str(arg) for arg in args])
                except SystemExit as error:
                    status = error.code
        finally:
            os.chdir(previous)
        return status, stdout.getvalue(), stderr.getvalue()

    def epub(self, name, chapters, identifier):
        return make_serial_epub(self.root / f"{name}.epub", "The Serial", identifier, chapters)

    def analyze(self, epub):
        return self.run_cli("analyze", epub, "-o", self.book)

    def seed(self):
        first = self.epub("serial-v1", [CHAPTER_ONE, CHAPTER_TWO], "download-1")
        status, stdout, stderr = self.analyze(first)
        self.assertEqual(status, 0, stderr)
        return first

    def model_payloads(self):
        return [json.dumps(payload) for client in self.clients for _, payload in client.calls]

    def test_appended_chapter_updates_the_book_and_reports_its_new_character(self):
        first = self.seed()
        before = snapshot(self.book)
        original = read_json(self.book / "analysis.json")
        calls_before = len(self.model_payloads())
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")

        status, stdout, stderr = self.analyze(updated)

        self.assertEqual(status, 0, stderr + stdout)
        self.assertIn("Added 1 chapter", stdout)
        self.assertIn("New character: Tobin", stdout)
        self.assertIn("Returning: Mara", stdout)
        bundle = load_bundle(self.book)
        document = read_json(self.book / "analysis.json")
        self.assertEqual(len(bundle["chapters"]), 3)
        # The saved book identity, earlier chapters and existing cast are untouched.
        self.assertEqual(document["source"]["book"], original["source"]["book"])
        self.assertEqual(document["source"]["blocks"][:len(original["source"]["blocks"])],
                         original["source"]["blocks"])
        self.assertEqual(document["segments"][:len(original["segments"])], original["segments"])
        for name in ("scripts/chapter-0001.jsonl", "scripts/chapter-0002.jsonl"):
            self.assertEqual((self.book / name).read_bytes(), before[name])
        for key in ("characters", "roles", "voices"):
            self.assertEqual(document["cast"][key][:len(original["cast"][key])], original["cast"][key])
        names = [character["name"] for character in document["cast"]["characters"]]
        self.assertEqual(names, ["Mara", "Tobin"])
        tobin = document["cast"]["characters"][1]
        tobin_rows = [row for row in document["segments"] if row["voice_id"] == tobin["voice_id"]]
        self.assertEqual([row["text"] for row in tobin_rows], ["“Wait for me.”"])
        mara_role = next(role["id"] for role in original["cast"]["roles"] if role["kind"] == "dialogue")
        self.assertIn("“Hurry.”", [row["text"] for row in document["segments"] if row["role_id"] == mara_role])
        # Only the new chapter was sent to the model.
        sent = self.model_payloads()[calls_before:]
        self.assertTrue(sent)
        self.assertFalse(any("The lamp is lit" in payload for payload in sent))
        self.assertTrue(any("Wait for me" in payload for payload in sent))
        # Update provenance, render identity and series snapshot.
        update = document["provenance"]["book_updates"][-1]
        self.assertEqual(update["previous_source_fingerprint"], fingerprint(original["source"]))
        self.assertEqual([chapter["title"] for chapter in update["added_chapters"]], ["Three"])
        self.assertEqual(update["new_characters"], [tobin["id"]])
        self.assertEqual(update["epub_sha256"], hashlib.sha256(updated.read_bytes()).hexdigest())
        self.assertEqual(chunk_identity_fingerprint(bundle["manifest"]), fingerprint(original["source"]))
        registry = validate_registry(read_json(self.book / "series-cast.json"))
        self.assertEqual(len(registry["books"]), 1)
        self.assertEqual(registry["books"][-1]["source_fingerprint"], fingerprint(document["source"]))
        self.assertFalse((self.book / ".analysis-update").exists())
        self.assertTrue(list((self.book / ".analysis-revisions").glob("*/revision.json")))
        # Rerunning with the same EPUB is a resume of the updated book.
        status, stdout, _ = self.analyze(updated)
        self.assertEqual(status, 0)
        self.assertIn("Saved analysis is complete", stdout)
        self.assertIsNotNone(first)

    def test_reordered_chapters_are_refused_with_their_location_and_book_unchanged(self):
        self.seed()
        before = snapshot(self.book)
        calls = len(self.model_payloads())
        changed = self.epub("serial-reordered", [CHAPTER_TWO, CHAPTER_ONE, CHAPTER_THREE], "download-3")
        status, stdout, stderr = self.analyze(changed)
        self.assertEqual(status, 1)
        self.assertIn("Chapter 1 ('One')", stderr)
        self.assertIn("removed, inserted or reordered", stderr)
        self.assertEqual(snapshot(self.book), before)
        self.assertEqual(len(self.model_payloads()), calls)

    def seed_long(self):
        self.seed_epub = self.epub("serial-v1", [LONG_ONE, CHAPTER_TWO], "download-1")
        status, _, stderr = self.analyze(self.seed_epub)
        self.assertEqual(status, 0, stderr)
        return read_json(self.book / "analysis.json")

    def test_edited_paragraph_is_analyzed_with_its_neighbours_and_everything_else_is_kept(self):
        original = self.seed_long()
        before_ids = self.plan_chunk_ids()
        calls = len(self.model_payloads())
        paragraphs = list(LONG_ONE[1])
        paragraphs[4] = "Mara said, “Bring the lamp oil.”"
        paragraphs.insert(6, "Gulls cried overhead.")
        edited = self.epub("serial-v2", [("One", paragraphs), CHAPTER_TWO, CHAPTER_THREE], "download-2")

        status, stdout, stderr = self.analyze(edited)

        self.assertEqual(status, 0, stderr + stdout)
        self.assertIn("Edited chapter 1 (One): 1 edited, 1 inserted paragraph(s)", stdout)
        self.assertIn("Added 1 chapter: Three", stdout)
        self.assertIn("New character: Tobin", stdout)
        sent = self.model_payloads()[calls:]
        # The edit is read with two unchanged paragraphs on each side; the rest
        # of the chapter and unchanged chapters are never sent again.
        self.assertTrue(any("Bring the lamp oil" in payload for payload in sent))
        self.assertTrue(any("Waves broke on the rocks." in payload for payload in sent))
        self.assertFalse(any("They left." in payload for payload in sent))
        self.assertFalse(any("The lamp is lit" in payload for payload in sent))
        document = read_json(self.book / "analysis.json")
        old_blocks = {block["text"]: block for block in original["source"]["blocks"]}
        new_blocks = {block["text"]: block for block in document["source"]["blocks"]}
        old_rows = {row["block_id"]: row for row in original["segments"]}
        for text, block in old_blocks.items():
            if text == LONG_ONE[1][4]:
                self.assertNotIn(text, new_blocks)
                continue
            # Unchanged paragraphs keep their IDs and their saved segments.
            self.assertEqual(new_blocks[text]["id"], block["id"])
            self.assertEqual([row for row in document["segments"] if row["block_id"] == block["id"]],
                             [row for row in original["segments"] if row["block_id"] == block["id"]])
        highest = max(int(block["id"][1:]) for block in original["source"]["blocks"])
        fresh = [new_blocks[paragraphs[4]]["id"], new_blocks["Gulls cried overhead."]["id"]]
        self.assertTrue(all(int(identifier[1:]) > highest for identifier in fresh))
        self.assertNotIn(fresh[0], old_rows)
        mara_role = next(role["id"] for role in original["cast"]["roles"] if role["kind"] == "dialogue")
        self.assertIn("“Bring the lamp oil.”",
                      [row["text"] for row in document["segments"] if row["role_id"] == mara_role])
        update = document["provenance"]["book_updates"][-1]
        self.assertEqual(update["edited_chapters"], [{"id": "ch0001", "title": "One", "number": 1,
            "edited_paragraphs": 1, "inserted_paragraphs": 1, "removed_paragraphs": 0}])
        # Every unchanged line keeps its render identity, so its audio is reused.
        after_ids = self.plan_chunk_ids()
        self.assertTrue(set(before_ids[1]) <= set(after_ids[1]))
        edited_rows = [row for row in original["segments"]
                       if row["block_id"] == old_blocks[LONG_ONE[1][4]]["id"]]
        self.assertEqual(len(set(before_ids[0]) - set(after_ids[0])), len(edited_rows))

    def test_neighbour_read_differently_keeps_its_saved_speakers_and_is_flagged_for_review(self):
        original = self.seed_long()
        neighbour = next(block["id"] for block in original["source"]["blocks"] if block["text"] == LONG_ONE[1][4])
        saved_rows = [row for row in original["segments"] if row["block_id"] == neighbour]
        self.misread = "Bring the oil"
        paragraphs = list(LONG_ONE[1])
        paragraphs[6] = "The tide turned slowly."

        status, stdout, stderr = self.analyze(self.epub("serial-v2", [("One", paragraphs), CHAPTER_TWO], "download-2"))

        self.assertEqual(status, 0, stderr + stdout)
        self.assertIn("1 unchanged paragraph(s) next to an edit now read differently", stdout)
        document = read_json(self.book / "analysis.json")
        self.assertEqual([row for row in document["segments"] if row["block_id"] == neighbour], saved_rows)
        issue = next(issue for issue in document["issues"] if issue["kind"] == "update_context_attribution")
        self.assertEqual(issue["target"], saved_rows[0]["id"])
        self.assertFalse(issue["resolved"])
        self.assertEqual(document["provenance"]["book_updates"][-1]["context_reviews"], [neighbour])

    def test_removed_paragraph_needs_no_model_and_moves_pronunciation_evidence(self):
        from epub_tts_narrator import pronunciations
        from epub_tts_narrator.pipeline import write_json
        original = self.seed_long()
        removed_block = next(block["id"] for block in original["source"]["blocks"] if block["text"] == LONG_ONE[1][4])
        record = pronunciations._empty(original["source"])
        for word, block_id in (("Mara", removed_block), ("oil", removed_block)):
            record["entries"].append({"id": pronunciations.word_id(word), "word": word, "status": "confirmed",
                                      "kind": "other", "reason": "Chosen by the listener.",
                                      "evidence": [{"block_id": block_id, "quote": word}],
                                      "pronunciation": {"alphabet": "respelling", "value": word + "a"}})
        write_json(self.book / pronunciations.FILENAME, record)
        clients = len(self.clients)
        paragraphs = [paragraph for index, paragraph in enumerate(LONG_ONE[1]) if index != 4]

        status, stdout, stderr = self.analyze(self.epub("serial-v2", [("One", paragraphs), CHAPTER_TWO], "download-2"))

        self.assertEqual(status, 0, stderr + stdout)
        self.assertIn("Edited chapter 1 (One): 1 removed paragraph(s)", stdout)
        self.assertIn("no model is needed", stderr + stdout)
        self.assertEqual(len(self.clients), clients)
        document = read_json(self.book / "analysis.json")
        self.assertNotIn(removed_block, [row["block_id"] for row in document["segments"]])
        entries = {entry["word"]: entry for entry in pronunciations._load_local(self.book)["entries"]}
        # "Mara" still occurs, so her choice cites another paragraph; "oil" left the book.
        self.assertEqual(list(entries), ["Mara"])
        cited = entries["Mara"]["evidence"][0]["block_id"]
        self.assertIn("Mara", next(block["text"] for block in document["source"]["blocks"] if block["id"] == cited))

    def test_new_download_without_new_chapters_changes_nothing(self):
        self.seed()
        before = snapshot(self.book)
        again = self.epub("serial-again", [CHAPTER_ONE, CHAPTER_TWO], "download-4")
        status, stdout, stderr = self.analyze(again)
        self.assertEqual(status, 0, stderr)
        self.assertIn("No new or edited chapters", stdout)
        self.assertEqual(snapshot(self.book), before)

    def test_interrupted_publication_resumes_without_repeating_model_work(self):
        self.seed()
        before = {name: data for name, data in snapshot(self.book).items()
                  if not name.startswith((".analysis-update", ".analysis-revisions", ".analysis-cache"))
                  and not name.endswith(".lease")}
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        with patch("epub_tts_narrator.book_update._install", side_effect=OSError("disk interrupted")):
            status, _, stderr = self.analyze(updated)
        self.assertEqual(status, 1)
        self.assertIn("disk interrupted", stderr)
        # Lease files are coordination artifacts that are kept, never deleted.
        self.assertEqual({name: data for name, data in snapshot(self.book).items()
                          if not name.startswith((".analysis-update", ".analysis-revisions", ".analysis-cache"))
                          and not name.endswith(".lease")}, before)
        self.assertTrue((self.book / ".analysis-update").is_dir())
        calls = len(self.model_payloads())
        status, stdout, stderr = self.analyze(updated)
        self.assertEqual(status, 0, stderr)
        self.assertIn("New character: Tobin", stdout)
        self.assertEqual(len(self.model_payloads()), calls)
        self.assertEqual(len(load_bundle(self.book)["chapters"]), 3)

    def plan_chunk_ids(self):
        from epub_tts_narrator.renderer.bundle import make_plan
        from epub_tts_narrator.renderer.config import DEFAULTS
        bundle = load_bundle(self.book)
        plan = make_plan(bundle, deepcopy(DEFAULTS))
        return [[chunk["chunk_id"] for chunk in chapter["chunks"]] for chapter in plan["chapters"]]

    def test_earlier_chapters_keep_their_exact_render_identity(self):
        self.seed()
        before = self.plan_chunk_ids()
        status, _, stderr = self.analyze(self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2"))
        self.assertEqual(status, 0, stderr)
        after = self.plan_chunk_ids()
        self.assertEqual(len(after), 3)
        self.assertEqual(after[:2], before)

    def test_source_bound_sidecars_are_rebound_to_the_updated_source(self):
        from epub_tts_narrator import audition_scripts, pronunciations, scoped_pronunciations
        from epub_tts_narrator.pipeline import write_json
        self.seed()
        document = read_json(self.book / "analysis.json")
        write_json(self.book / pronunciations.FILENAME, pronunciations._empty(document["source"]))
        write_json(self.book / scoped_pronunciations.FILENAME,
                   scoped_pronunciations.local(self.book, source=document["source"]))
        write_json(self.book / audition_scripts.FILENAME, {
            "version": audition_scripts.VERSION, "source_fingerprint": fingerprint(document["source"]),
            "cast_fingerprint": fingerprint(document["cast"]), "scripts": {}})
        status, _, stderr = self.analyze(self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2"))
        self.assertEqual(status, 0, stderr)
        updated = read_json(self.book / "analysis.json")
        self.assertEqual(pronunciations._load_local(self.book)["source_fingerprint"], fingerprint(updated["source"]))
        self.assertEqual(scoped_pronunciations.local(self.book)["source_fingerprint"], fingerprint(updated["source"]))
        sidecar = read_json(self.book / audition_scripts.FILENAME)
        self.assertEqual((sidecar["source_fingerprint"], sidecar["cast_fingerprint"]),
                         (fingerprint(updated["source"]), fingerprint(updated["cast"])))

    def test_updated_epub_is_accepted_for_cover_and_metadata(self):
        from epub_tts_narrator.renderer.book_metadata import accepted_source_epubs, source_epub_digest
        from epub_tts_narrator.source_epub import source_epub_hint
        first = self.seed()
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        self.assertEqual(self.analyze(updated)[0], 0)
        bundle = load_bundle(self.book)
        digest = hashlib.sha256(updated.read_bytes()).hexdigest()
        self.assertEqual(accepted_source_epubs(bundle), {bundle["source"]["book"]["id"], digest})
        self.assertEqual(source_epub_digest(bundle, updated), digest)
        self.assertEqual(source_epub_digest(bundle, first), bundle["source"]["book"]["id"])
        unrelated = self.epub("unrelated", [CHAPTER_THREE], "other")
        self.assertEqual(source_epub_digest(bundle, unrelated), bundle["source"]["book"]["id"])
        self.assertEqual(source_epub_hint(self.book, bundle["source"]["book"]["id"]), updated.resolve())

    def test_character_from_an_earlier_book_returns_with_that_books_voice(self):
        earlier = self.root / "book-one"
        status, _, stderr = self.run_cli("analyze", self.epub("book-one", [CHAPTER_ONE, CHAPTER_THREE], "book-one"),
                                         "-o", earlier)
        self.assertEqual(status, 0, stderr)
        morning = ("Four", ["Mara, the lighthouse keeper, said, “Morning.”"])
        status, _, stderr = self.run_cli("analyze", self.epub("serial-v1", [morning], "serial-1"),
                                         "-o", self.book, "--continue-from", earlier)
        self.assertEqual(status, 0, stderr)
        tobin = next(character for character in read_json(earlier / "analysis.json")["cast"]["characters"]
                     if character["name"] == "Tobin")
        self.assertNotIn("Tobin", [character["name"] for character in
                                   read_json(self.book / "analysis.json")["cast"]["characters"]])
        updated = self.epub("serial-v2", [morning, ("Five", ["Tobin the ferryman called out, “Back again.”"])],
                            "serial-2")

        status, stdout, stderr = self.analyze(updated)

        self.assertEqual(status, 0, stderr)
        self.assertIn("Returning from an earlier book: Tobin", stdout)
        self.assertNotIn("need casting", stdout)
        document = read_json(self.book / "analysis.json")
        returned = next(character for character in document["cast"]["characters"] if character["name"] == "Tobin")
        self.assertEqual((returned["id"], returned["voice_id"]), (tobin["id"], tobin["voice_id"]))
        self.assertEqual(document["provenance"]["book_updates"][-1]["returning_series_characters"], [tobin["id"]])
        # The continuation link to the earlier book still validates.
        from epub_tts_narrator.analysis_directory import continuation_directories
        self.assertEqual(continuation_directories(self.book), [earlier.resolve()])
        # The earlier book cannot grow in place: a later book continues from it.
        before = snapshot(earlier)
        status, _, stderr = self.run_cli(
            "analyze", self.epub("book-one-v2", [CHAPTER_ONE, CHAPTER_THREE, ("Six", ["They rowed home."])],
                                 "book-one-2"), "-o", earlier)
        self.assertEqual(status, 1)
        self.assertIn("continues from this one", stderr)
        self.assertEqual(snapshot(earlier), before)

    def test_plan_estimates_only_the_appended_chapters(self):
        self.seed()
        calls = len(self.model_payloads())
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        status, stdout, stderr = self.run_cli("plan", updated, "-o", self.book)
        self.assertEqual(status, 0, stderr)
        planned = json.loads(stdout)
        self.assertEqual(planned["book_update"]["unchanged_chapters"], 2)
        self.assertEqual([row["title"] for row in planned["book_update"]["added_chapters"]], ["Three"])
        self.assertEqual(len(self.model_payloads()), calls)
        status, stdout, _ = self.run_cli("plan", self.epub("serial-same", [CHAPTER_ONE, CHAPTER_TWO], "same"),
                                         "-o", self.book)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout)["book_update"]["added_chapters"], [])

    def test_update_is_refused_while_the_book_is_rendering(self):
        self.seed()
        (self.book / ".audio-render.lock").mkdir()
        before = snapshot(self.book)
        calls = len(self.model_payloads())
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        status, _, stderr = self.analyze(updated)
        self.assertEqual(status, 1)
        self.assertIn("Rendering is active", stderr)
        self.assertEqual(snapshot(self.book), before)
        # Refused before any book text is sent to the model.
        self.assertEqual(len(self.model_payloads()), calls)


class SourceComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def source(self, name, chapters):
        return source_from(make_serial_epub(self.root / f"{name}.epub", "The Serial", name, chapters))

    def test_appended_chapters_are_identified_after_an_unchanged_prefix(self):
        saved = self.source("one", [CHAPTER_ONE, CHAPTER_TWO])
        current = self.source("two", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE])
        comparison = compare_sources(saved, current)
        self.assertEqual([chapter["title"] for chapter in comparison["appended_chapters"]], ["Three"])
        self.assertEqual(comparison["unchanged_chapters"], 2)

    def test_removed_inserted_or_reordered_chapters_are_refused(self):
        saved = self.source("one", [CHAPTER_ONE, CHAPTER_TWO])
        cases = {
            "removed": [CHAPTER_ONE],
            "reordered": [CHAPTER_TWO, CHAPTER_ONE, CHAPTER_THREE],
            "inserted": [CHAPTER_ONE, CHAPTER_THREE, CHAPTER_TWO],
        }
        for label, chapters in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, "removed, inserted or reordered"):
                    compare_sources(saved, self.source(label, chapters))

    def test_edits_inside_saved_chapters_keep_every_unchanged_paragraph_id(self):
        saved = self.source("one", [CHAPTER_ONE, CHAPTER_TWO])
        cases = {
            "rewritten": ([CHAPTER_ONE, ("Two", ["Mara said, “The lamp is out.”"])], (1, 0, 0)),
            "paragraph added": ([CHAPTER_ONE, ("Two", [*CHAPTER_TWO[1], "Night fell."])], (0, 1, 0)),
            "paragraph removed": ([("One", CHAPTER_ONE[1][:1]), CHAPTER_TWO], (0, 0, 1)),
        }
        for label, (chapters, counts) in cases.items():
            with self.subTest(label):
                comparison = compare_sources(saved, self.source(label.replace(" ", "-"), chapters))
                (edit,) = comparison["edited_chapters"]
                self.assertEqual((edit["edited_paragraphs"], edit["inserted_paragraphs"],
                                  edit["removed_paragraphs"]), counts)
                old = {block["text"]: block["id"] for block in saved["blocks"]}
                for block in comparison["source"]["blocks"]:
                    if block["text"] in old:
                        self.assertEqual(block["id"], old[block["text"]])
                    else:
                        self.assertIn(block["id"], comparison["changed_blocks"])
                self.assertEqual(comparison["source"]["book"], saved["book"])
                self.assertEqual([chapter["id"] for chapter in comparison["source"]["chapters"]],
                                 [chapter["id"] for chapter in saved["chapters"]])

    def test_a_paragraph_moved_to_another_epub_file_is_not_an_edit(self):
        saved = self.source("one", [CHAPTER_ONE, CHAPTER_TWO])
        current = deepcopy(saved)
        for block in current["blocks"]:
            block["source"]["href"] = "renamed/" + block["source"]["href"]
        self.assertEqual(compare_sources(saved, current)["edited_chapters"], [])

    def test_changed_extraction_policy_is_refused(self):
        saved = self.source("one", [CHAPTER_ONE, CHAPTER_TWO])
        current = deepcopy(self.source("two", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE]))
        current["extraction_policy"] += "; different"
        with self.assertRaisesRegex(ValueError, "extraction"):
            compare_sources(saved, current)


if __name__ == "__main__":
    unittest.main()
