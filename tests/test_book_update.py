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
import zipfile
from xml.sax.saxutils import escape

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
# A later download of the same chapter with one typo corrected in place.
CORRECTED_THREE = ("Three", ["Tobin the ferryman called out, “Wait for me, please.”",
                             "Mara, the lighthouse keeper, answered, “Hurry.”"])
LONG_ONE = ("One", ["Mara, the lighthouse keeper, said softly, “Hello.”", "They left.", "The night was cold.",
                    "Waves broke on the rocks.", "Mara said, “Bring the oil.”", "Morning came.",
                    "The tide turned.", "Boats came home."])
# Source context, beyond the shared name, that links a returning character.
CONTEXT = {"Mara": "Mara, the lighthouse keeper", "Tobin": "Tobin the ferryman"}


def with_auxiliary_chapter(epub):
    """Rewrite a serial fixture with one non-linear spine chapter appended.

    The chapter belongs to the book only under extraction.include_auxiliary,
    which is exactly the saved policy content acceptance must mirror.
    """
    with zipfile.ZipFile(epub) as archive:
        entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(epub, "w") as archive:
        for name, data in entries:
            if name == "content.opf":
                text = data.decode("utf-8")
                text = text.replace(
                    "</manifest>",
                    '<item id="aux" href="auxiliary.xhtml" media-type="application/xhtml+xml"/></manifest>')
                text = text.replace("</spine>", '<itemref idref="aux" linear="no"/></spine>')
                data = text.encode("utf-8")
            archive.writestr(name, data)
        archive.writestr(
            "auxiliary.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Notes</title></head>'
            "<body><h1>Notes</h1><p>The keeper's log recorded the night.</p></body></html>")
    return epub


def with_navigation_titles(epub, titles):
    """Add NCX labels without changing any XHTML heading or paragraph."""
    with zipfile.ZipFile(epub) as archive:
        entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    points = "".join(
        f'<navPoint id="n{number}"><navLabel><text>{escape(title)}</text></navLabel>'
        f'<content src="chapter-{number:03d}.xhtml"/></navPoint>'
        for number, title in enumerate(titles, 1))
    with zipfile.ZipFile(epub, "w") as archive:
        for name, data in entries:
            if name == "content.opf":
                text = data.decode("utf-8").replace(
                    "</manifest>", '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest>')
                data = text.replace("<spine>", '<spine toc="ncx">').encode("utf-8")
            archive.writestr(name, data)
        archive.writestr("toc.ncx", '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>'
                         + points + "</navMap></ncx>")
    return epub


def rename_chapter_resource(epub, old_name, new_name):
    """Move one XHTML resource while preserving its extracted text and markup."""
    with zipfile.ZipFile(epub) as archive:
        entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(epub, "w") as archive:
        for name, data in entries:
            if name in {"content.opf", "toc.ncx"}:
                data = data.replace(old_name.encode("utf-8"), new_name.encode("utf-8"))
            archive.writestr(new_name if name == old_name else name, data)
    return epub


def publication_snapshot(directory):
    """Book payload, excluding revision journals and retained lock ownership."""
    return {name: data for name, data in snapshot(directory).items()
            if not name.startswith((".analysis-revisions", ".analysis-update.lock"))
            and not name.endswith(".lease")}


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

    def analyze(self, epub, *, fresh=False):
        args = ["analyze", epub, "-o", self.book]
        if fresh:
            args += ["--strategy", "consensus"]
        return self.run_cli(*args)

    def seed(self):
        first = self.epub("serial-v1", [CHAPTER_ONE, CHAPTER_TWO], "download-1")
        status, stdout, stderr = self.analyze(first, fresh=True)
        self.assertEqual(status, 0, stderr)
        return first

    def seed_series(self):
        earlier = (self.root / "book-one").resolve()
        status, _, stderr = self.run_cli(
            "analyze", "--strategy", "consensus",
            self.epub("book-one", [CHAPTER_ONE, CHAPTER_THREE], "book-one"), "-o", earlier)
        self.assertEqual(status, 0, stderr)
        morning = ("Four", ["Mara, the lighthouse keeper, said, “Morning.”"])
        status, _, stderr = self.run_cli(
            "analyze", self.epub("serial-v1", [morning], "serial-1"), "-o", self.book,
            "--continue-from", earlier, "--strategy", "consensus")
        self.assertEqual(status, 0, stderr)
        return earlier

    def prepare_title_sidecars(self, book):
        from epub_tts_narrator import audition_scripts, name_pronunciations, pronunciations, scoped_pronunciations
        from epub_tts_narrator.config import DEFAULTS
        from epub_tts_narrator.performance_planning import prepare_performance
        from epub_tts_narrator.pipeline import write_json
        from test_performance_planning import FakeClient as PerformanceClient

        document = read_json(book / "analysis.json")
        source_fingerprint = fingerprint(document["source"])
        block = next(block for block in document["source"]["blocks"] if "Mara" in block["text"])
        registry = pronunciations._empty(document["source"])
        registry["entries"].append({
            "id": pronunciations.word_id("Mara"), "word": "Mara", "status": "confirmed", "kind": "name",
            "reason": "Chosen by the listener.", "evidence": [{"block_id": block["id"], "quote": "Mara"}],
            "pronunciation": {"alphabet": "respelling", "value": "MAH-rah"}})
        write_json(book / pronunciations.FILENAME, registry)
        write_json(book / scoped_pronunciations.FILENAME,
                   scoped_pronunciations.local(book, source=document["source"]))
        write_json(book / audition_scripts.FILENAME, {
            "version": audition_scripts.VERSION, "source_fingerprint": source_fingerprint,
            "cast_fingerprint": fingerprint(document["cast"]), "scripts": {},
            "samples": {"fixture": "keep samples"}, "pending": {"fixture": {"error": "keep pending"}}})
        config = deepcopy(DEFAULTS)
        config["analysis"].update(attribution_workers=1, repair_attempts=0)
        plan = prepare_performance(load_bundle(book), config, book / "performance-plan.json",
                                   client=PerformanceClient(), notify=lambda _: None)
        write_json(book / name_pronunciations.FILENAME, name_pronunciations.load_local(book))
        for name in ("heard-variants.json", "voice-aliases.json", "voice-traits.json"):
            (book / name).write_text(json.dumps({"fixture": f"keep {name}"}) + "\n", encoding="utf-8")
        (book / "voices").mkdir(exist_ok=True)
        (book / "voices" / "mara-reference.wav").write_bytes(b"fixture reference remains")
        (book / "render-work").mkdir(exist_ok=True)
        (book / "render-work" / "history.json").write_text('{"fixture": "keep history"}\n', encoding="utf-8")
        bound_names = ["audition-scripts.json", "pronunciations.json", "scoped-pronunciations.json",
                       "performance-plan.json"]
        preserved_names = ["name-pronunciations.json", "heard-variants.json", "voice-aliases.json",
                           "voice-traits.json", "voices/mara-reference.wav", "render-work/history.json"]
        preserved = {name: (book / name).read_bytes() for name in preserved_names}
        preserved.update({str(path.relative_to(book)): path.read_bytes() for path in book.rglob("*")
                          if path.is_file() and ({".analysis-cache", "performance-plan.json.work"} & set(path.parts))})
        return {"plan": plan,
                "bound_bytes": {name: (book / name).read_bytes() for name in bound_names},
                "preserved_bytes": preserved}

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
        self.assertEqual(update["strategy"], "consensus")
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
        status, _, stderr = self.analyze(self.seed_epub, fresh=True)
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

    def test_title_only_download_republishes_metadata_without_model_work_and_rerun_is_a_no_op(self):
        from epub_tts_narrator.source_epub import source_epub_hint
        self.seed()
        original = read_json(self.book / "analysis.json")
        before_chunks = self.plan_chunk_ids()
        before_scripts = {path.name: path.read_bytes() for path in (self.book / "scripts").glob("*.jsonl")}
        titled = with_navigation_titles(
            self.epub("serial-titles", [CHAPTER_ONE, CHAPTER_TWO], "download-2"),
            ["Chapter One", "Chapter Two"])

        with (patch.object(self, "client", side_effect=AssertionError("title update created a model client")),
              patch("epub_tts_narrator.audition_scripts.prepare_document_samples",
                    side_effect=AssertionError("title update prepared auditions")),
              patch("epub_tts_narrator.speaker_decisions.prompt_speaker_decisions",
                    side_effect=AssertionError("title update prompted for speakers"))):
            status, stdout, stderr = self.analyze(titled)

        self.assertEqual(status, 0, stderr + stdout)
        document = read_json(self.book / "analysis.json")
        expected_source = deepcopy(original["source"])
        for chapter, title in zip(expected_source["chapters"], ["Chapter One", "Chapter Two"]):
            chapter["title"] = title
        self.assertEqual(document["source"], expected_source)
        for key in ("cast", "segments", "issues"):
            self.assertEqual(document[key], original[key])
        self.assertEqual(document["provenance"]["series"], original["provenance"]["series"])
        expected_samples = deepcopy(original["provenance"]["voice_samples"])
        expected_samples["source_fingerprint"] = fingerprint(expected_source)
        self.assertEqual(document["provenance"]["voice_samples"], expected_samples)
        expected_chunk_source = original["provenance"].get(
            "chunk_identity_source_fingerprint", fingerprint(original["source"]))
        self.assertEqual(document["provenance"]["chunk_identity_source_fingerprint"], expected_chunk_source)
        self.assertEqual(chunk_identity_fingerprint(load_bundle(self.book)["manifest"]), expected_chunk_source)
        self.assertEqual(self.plan_chunk_ids(), before_chunks)
        self.assertEqual({path.name: path.read_bytes() for path in (self.book / "scripts").glob("*.jsonl")},
                         before_scripts)
        update = document["provenance"]["book_updates"][-1]
        self.assertEqual(update["retitled_chapters"], [
            {"id": "ch0001", "number": 1, "previous_title": "One", "title": "Chapter One"},
            {"id": "ch0002", "number": 2, "previous_title": "Two", "title": "Chapter Two"}])
        self.assertEqual((update["added_chapters"], update["edited_chapters"]), ([], []))
        self.assertIsNone(update["delta_source_fingerprint"])
        self.assertIsNone(update["analysis"])
        self.assertEqual(update["epub_sha256"], hashlib.sha256(titled.read_bytes()).hexdigest())
        self.assertEqual(source_epub_hint(self.book, document["source"]["book"]["id"]), titled.resolve())
        self.assertFalse((self.book / ".analysis-pending").exists())
        self.assertFalse((self.book / ".analysis-update").exists())
        after_first = snapshot(self.book)

        with (patch.object(self, "client", side_effect=AssertionError("no-op created a model client")),
              patch("epub_tts_narrator.audition_scripts.prepare_document_samples",
                    side_effect=AssertionError("no-op prepared auditions")),
              patch("epub_tts_narrator.speaker_decisions.prompt_speaker_decisions",
                    side_effect=AssertionError("no-op prompted for speakers"))):
            status, stdout, stderr = self.analyze(titled)
        self.assertEqual(status, 0, stderr + stdout)
        self.assertIn("Saved analysis is complete", stdout)
        self.assertEqual(snapshot(self.book), after_first)

    def test_title_and_text_changes_still_use_the_ordinary_analysis_pipeline(self):
        self.seed()
        calls = len(self.model_payloads())
        changed_two = ("Two", ["Mara said, “The lamp is lit tonight.”"])
        titled = with_navigation_titles(
            self.epub("serial-title-and-text", [CHAPTER_ONE, changed_two], "download-2"),
            ["Chapter One", "Two"])

        status, _, stderr = self.analyze(titled)

        self.assertEqual(status, 0, stderr)
        self.assertGreater(len(self.model_payloads()), calls)
        document = read_json(self.book / "analysis.json")
        update = document["provenance"]["book_updates"][-1]
        self.assertEqual([row["id"] for row in update["edited_chapters"]], ["ch0002"])
        self.assertEqual(update["retitled_chapters"], [
            {"id": "ch0001", "number": 1, "previous_title": "One", "title": "Chapter One"}])
        self.assertEqual([chapter["title"] for chapter in document["source"]["chapters"]],
                         ["Chapter One", "Two"])
        self.assertTrue(any("lit tonight" in payload for payload in self.model_payloads()[calls:]))

    def test_title_update_refuses_changed_resource_layout_or_an_invalid_sidecar_without_changes(self):
        for sidecar_case in (False, True):
            with self.subTest(sidecar_case=sidecar_case):
                self.book = self.root / f"serial-unsafe-book-{sidecar_case}"
                self.seed()
                if sidecar_case:
                    (self.book / "performance-plan.json").write_text('{"format": "unrecognized"}\n',
                                                                      encoding="utf-8")
                before = snapshot(self.book)
                titled = with_navigation_titles(
                    self.epub(f"serial-unsafe-{sidecar_case}", [CHAPTER_ONE, CHAPTER_TWO], "download-2"),
                    ["Chapter One", "Chapter Two"])
                if not sidecar_case:
                    titled = rename_chapter_resource(titled, "chapter-001.xhtml", "renamed/chapter-001.xhtml")
                with patch.object(self, "client", side_effect=AssertionError("unsafe update created a client")):
                    status, _, stderr = self.analyze(titled)
                self.assertEqual(status, 1)
                self.assertIn("Performance plan" if sidecar_case else "title-only", stderr)
                self.assertEqual(snapshot(self.book), before)

    def test_title_update_rebinds_sidecars_and_descendant_links_without_touching_descendant_content(self):
        from epub_tts_narrator import audition_scripts, pronunciations, scoped_pronunciations
        from epub_tts_narrator.analysis_directory import _completed_directory
        from epub_tts_narrator.performance_planning import load_prepared_performance
        earlier = self.seed_series()
        fixtures = self.prepare_title_sidecars(earlier)
        child_before = {name: (self.book / name).read_bytes() for name in
                        ["analysis.json", "source.json", "cast.json", "review.json", "scripts/chapter-0001.jsonl"]}
        original = read_json(earlier / "analysis.json")
        titled = with_navigation_titles(
            self.epub("book-one-titles", [CHAPTER_ONE, CHAPTER_THREE], "book-one-2"),
            ["Chapter One", "Chapter Three"])

        with patch.object(self, "client", side_effect=AssertionError("title update created a model client")):
            status, stdout, stderr = self.run_cli("analyze", titled, "-o", earlier)

        self.assertEqual(status, 0, stderr + stdout)
        self.assertIn("Original analysis files and revision journal", stdout)
        document = read_json(earlier / "analysis.json")
        new_source = fingerprint(document["source"])
        self.assertEqual([chapter["title"] for chapter in document["source"]["chapters"]],
                         ["Chapter One", "Chapter Three"])
        for name, data in child_before.items():
            self.assertEqual((self.book / name).read_bytes(), data, name)
        _completed_directory(earlier)
        _completed_directory(self.book)
        for name, data in fixtures["preserved_bytes"].items():
            self.assertEqual((earlier / name).read_bytes(), data, name)
        for name, module in (("pronunciations.json", pronunciations),
                             ("scoped-pronunciations.json", scoped_pronunciations),
                             ("audition-scripts.json", audition_scripts)):
            expected = json.loads(fixtures["bound_bytes"][name])
            expected["source_fingerprint"] = new_source
            self.assertEqual(read_json(earlier / name), expected)
        self.assertEqual(pronunciations._load_local(earlier)["source_fingerprint"], new_source)
        self.assertEqual(scoped_pronunciations.local(earlier)["source_fingerprint"], new_source)
        expected_plan = deepcopy(fixtures["plan"])
        expected_plan["source_fingerprint"] = new_source
        provenance_without_inputs = {key: deepcopy(value) for key, value in
                                     expected_plan["provenance"].items() if key != "inputs_sha256"}
        expected_plan["provenance"]["inputs_sha256"] = fingerprint({
            "source": new_source, "chapters": expected_plan["chapters"], **provenance_without_inputs})
        self.assertEqual(read_json(earlier / "performance-plan.json"), expected_plan)
        self.assertEqual(load_prepared_performance(load_bundle(earlier), earlier / "performance-plan.json"),
                         expected_plan)
        transactions = list((earlier / ".analysis-revisions").glob("*/revision.json"))
        self.assertEqual(len(transactions), 1)
        backup = transactions[0].parent / "book-0001" / "before"
        for name, data in fixtures["bound_bytes"].items():
            self.assertEqual((backup / name).read_bytes(), data, name)

    def test_title_update_rolls_back_selected_book_descendant_and_sidecars_together(self):
        from epub_tts_narrator.analysis_directory import _completed_directory
        from epub_tts_narrator.analysis_revision import _install
        earlier = self.seed_series()
        self.prepare_title_sidecars(earlier)
        before_parent, before_child = publication_snapshot(earlier), publication_snapshot(self.book)
        titled = with_navigation_titles(
            self.epub("book-one-rollback", [CHAPTER_ONE, CHAPTER_THREE], "book-one-2"),
            ["Chapter One", "Chapter Three"])
        failed = False

        def fail_once_at_descendant_registry(source, destination):
            nonlocal failed
            if not failed and Path(destination) == (self.book / "series-cast.json").resolve():
                failed = True
                raise OSError("fixture interrupted title publication")
            return _install(source, destination)

        with (patch("epub_tts_narrator.analysis_revision._install", side_effect=fail_once_at_descendant_registry),
              patch.object(self, "client", side_effect=AssertionError("rollback created a model client"))):
            status, _, stderr = self.run_cli("analyze", titled, "-o", earlier)

        self.assertEqual(status, 1)
        self.assertIn("fixture interrupted title publication", stderr)
        self.assertTrue(failed)
        self.assertEqual(publication_snapshot(earlier), before_parent)
        self.assertEqual(publication_snapshot(self.book), before_child)
        _completed_directory(earlier)
        _completed_directory(self.book)
        journal = json.loads(next((earlier / ".analysis-revisions").glob("*/revision.json")).read_text())
        self.assertEqual(journal["status"], "rolled_back")

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

    def test_repacked_or_corrected_update_epub_still_supplies_cover_and_metadata(self):
        from epub_tts_narrator.renderer.book_metadata import publication_assets, source_epub_digest
        from epub_tts_narrator.renderer.config import DEFAULTS
        from shared_fixtures import repack_epub
        self.seed_long()
        updated = self.epub("serial-v2", [LONG_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        self.assertEqual(self.analyze(updated)[0], 0)
        bundle = load_bundle(self.book)
        book_id = bundle["source"]["book"]["id"]
        updated_digest = hashlib.sha256(updated.read_bytes()).hexdigest()

        # A rebuilt edition of the updated download: different archive bytes,
        # identical extracted book.
        repacked = repack_epub(updated, self.root / "serial-v2-repacked.epub")
        repacked_digest = hashlib.sha256(repacked.read_bytes()).hexdigest()
        self.assertNotEqual(repacked_digest, updated_digest)
        self.assertEqual(source_epub_digest(bundle, repacked), repacked_digest)

        # A typo-corrected revision at the remembered location is the same book:
        # one edited paragraph inside an otherwise unchanged chapter.
        paragraphs = list(LONG_ONE[1])
        paragraphs[4] = "Mara said, “Bring the lamp oil.”"
        corrected = self.epub("serial-v2-corrected",
                              [("One", paragraphs), CHAPTER_TWO, CHAPTER_THREE], "download-2")
        corrected_digest = hashlib.sha256(corrected.read_bytes()).hexdigest()
        self.assertNotIn(corrected_digest, {book_id, updated_digest})
        self.assertEqual(source_epub_digest(bundle, corrected), corrected_digest)

        # Publication uses the revision's metadata under the saved book identity.
        config = deepcopy(DEFAULTS)
        config["render"].update(epub=str(corrected), require_cover=False)
        notes = []
        book, cover, artwork = publication_assets(bundle, config, self.root / "artwork",
                                                  notify=notes.append)
        self.assertIsNone(cover)
        self.assertIsNone(artwork)
        self.assertEqual(book["id"], book_id)
        self.assertTrue(any("newer revision" in note for note in notes))

        # A genuinely different book, and a reordered download, still fail closed.
        unrelated = self.epub("unrelated", [CHAPTER_THREE], "other")
        self.assertEqual(source_epub_digest(bundle, unrelated), book_id)
        reordered = self.epub("reordered", [CHAPTER_TWO, LONG_ONE, CHAPTER_THREE], "download-2")
        self.assertEqual(source_epub_digest(bundle, reordered), book_id)

    def test_auxiliary_spine_book_accepts_repacked_and_corrected_cover_sources(self):
        from epub_tts_narrator.renderer.book_metadata import source_epub_digest
        from shared_fixtures import repack_epub
        # Analyzed with extraction.include_auxiliary = true: the saved book
        # keeps its non-linear spine chapter, so only an extraction with the
        # same policy can match it.
        self.config.write_text(self.config.read_text(encoding="utf-8")
                               + "[extraction]\ninclude_auxiliary = true\n", encoding="utf-8")
        analyzed = with_auxiliary_chapter(self.epub("auxiliary-v1", [CHAPTER_ONE, CHAPTER_TWO], "download-1"))
        status, _, stderr = self.analyze(analyzed, fresh=True)
        self.assertEqual(status, 0, stderr)
        bundle = load_bundle(self.book)
        saved = bundle["source"]
        book_id = saved["book"]["id"]
        self.assertIn("auxiliary-spine=include", saved["extraction_policy"])
        self.assertEqual([chapter["title"] for chapter in saved["chapters"]], ["One", "Two", "Notes"])

        repacked = repack_epub(analyzed, self.root / "auxiliary-v1-repacked.epub")
        self.assertEqual(source_epub_digest(bundle, repacked),
                         hashlib.sha256(repacked.read_bytes()).hexdigest())
        corrected = with_auxiliary_chapter(
            self.epub("auxiliary-v2", [CHAPTER_ONE, ("Two", ["Mara said, “The lamp is lit tonight.”"])],
                      "download-2"))
        self.assertEqual(source_epub_digest(bundle, corrected),
                         hashlib.sha256(corrected.read_bytes()).hexdigest())

        # An unsupported saved policy stays fail-closed, and another book is refused.
        unknown = deepcopy(bundle)
        unknown["source"]["extraction_policy"] = saved["extraction_policy"].replace(
            "auxiliary-spine=include", "auxiliary-spine=auto")
        self.assertEqual(source_epub_digest(unknown, repacked), book_id)
        unrelated = with_auxiliary_chapter(self.epub("auxiliary-other", [CHAPTER_THREE], "other"))
        self.assertEqual(source_epub_digest(bundle, unrelated), book_id)

    def test_corrected_revision_of_the_updated_epub_supplies_cover_and_metadata(self):
        from epub_tts_narrator.renderer.book_metadata import (
            accepted_source_epubs, check_publication_source, publication_assets, source_epub_digest)
        from epub_tts_narrator.source_epub import remember_source_epub, source_epub_hint
        from shared_fixtures import renderer_config
        self.seed()
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        self.assertEqual(self.analyze(updated)[0], 0)
        bundle = load_bundle(self.book)
        fingerprint_before = fingerprint(bundle["source"])
        corrected = self.epub("serial-v3", [CHAPTER_ONE, CHAPTER_TWO, CORRECTED_THREE], "download-2")
        digest = hashlib.sha256(corrected.read_bytes()).hexdigest()
        self.assertNotIn(digest, accepted_source_epubs(bundle))
        # The revision was never analyzed, but it extracts to the same book.
        self.assertEqual(source_epub_digest(bundle, corrected), digest)
        config = renderer_config(epub=str(corrected))
        messages = []
        book, cover, artwork = publication_assets(bundle, config, self.root / "artwork",
                                                  notify=messages.append)
        self.assertIsNone(cover)
        self.assertIsNone(artwork)
        self.assertEqual(book["id"], bundle["source"]["book"]["id"])
        self.assertTrue(any("newer revision" in message for message in messages))
        check_publication_source(bundle, config)
        self.assertEqual(fingerprint(load_bundle(self.book)["source"]), fingerprint_before)
        remember_source_epub(self.book, corrected, bundle["source"]["book"]["id"],
                             saved_source=bundle["source"])
        self.assertEqual(source_epub_hint(self.book, bundle["source"]["book"]["id"]), corrected.resolve())

    def test_repacked_identical_epub_is_accepted_by_content_for_fresh_and_updated_books(self):
        from epub_tts_narrator.renderer.book_metadata import source_epub_digest
        self.seed()
        bundle = load_bundle(self.book)
        repacked = self.epub("serial-v1-repacked", [CHAPTER_ONE, CHAPTER_TWO], "download-1-repacked")
        self.assertNotEqual(hashlib.sha256(repacked.read_bytes()).hexdigest(),
                            bundle["source"]["book"]["id"])
        self.assertEqual(source_epub_digest(bundle, repacked),
                         hashlib.sha256(repacked.read_bytes()).hexdigest())
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        self.assertEqual(self.analyze(updated)[0], 0)
        bundle = load_bundle(self.book)
        repacked = self.epub("serial-v2-repacked", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE],
                             "download-2-repacked")
        self.assertEqual(source_epub_digest(bundle, repacked),
                         hashlib.sha256(repacked.read_bytes()).hexdigest())

    def test_unrelated_and_rewritten_epubs_still_stop_publication(self):
        from epub_tts_narrator.renderer.book_metadata import check_publication_source, source_epub_digest
        from epub_tts_narrator.source_epub import remember_source_epub
        from shared_fixtures import renderer_config
        self.seed()
        updated = self.epub("serial-v2", [CHAPTER_ONE, CHAPTER_TWO, CHAPTER_THREE], "download-2")
        self.assertEqual(self.analyze(updated)[0], 0)
        bundle = load_bundle(self.book)
        book_id = bundle["source"]["book"]["id"]
        unrelated = self.epub("unrelated", [CHAPTER_THREE], "other")
        rewritten = self.epub("rewritten", [("One", ["Completely different words appear here."]),
                                            ("Two", ["Nothing in common with the saved text."])], "other-2")
        for stranger in (unrelated, rewritten):
            self.assertEqual(source_epub_digest(bundle, stranger), book_id)
            with self.assertRaisesRegex(ValueError, "does not match"):
                check_publication_source(bundle, renderer_config(epub=str(stranger)))
            with self.assertRaisesRegex(ValueError, "does not match"):
                remember_source_epub(self.book, stranger, book_id, saved_source=bundle["source"])

    def test_character_from_an_earlier_book_returns_with_that_books_voice(self):
        earlier = self.root / "book-one"
        status, _, stderr = self.run_cli("analyze", "--strategy", "consensus", self.epub("book-one", [CHAPTER_ONE, CHAPTER_THREE], "book-one"),
                                         "-o", earlier)
        self.assertEqual(status, 0, stderr)
        morning = ("Four", ["Mara, the lighthouse keeper, said, “Morning.”"])
        status, _, stderr = self.run_cli("analyze", "--strategy", "consensus", self.epub("serial-v1", [morning], "serial-1"),
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

    def test_source_typo_updates_parent_and_rebinds_descendant_without_reanalysis(self):
        earlier = self.root / "book-one"
        status, _, stderr = self.run_cli("analyze", "--strategy", "consensus", self.epub("book-one", [CHAPTER_ONE, CHAPTER_THREE], "book-one"),
                                         "-o", earlier)
        self.assertEqual(status, 0, stderr)
        status, _, stderr = self.run_cli(
            "analyze", self.epub("serial-typo-source", [("Four", ["Mara said, “Morning.”"])], "serial-typo-1"),
            "-o", self.book, "--continue-from", earlier, "--strategy", "consensus")
        self.assertEqual(status, 0, stderr)
        script_names = [f"scripts/{path.name}" for path in (self.book / "scripts").glob("*.jsonl")]
        before = {name: (self.book / name).read_bytes() for name in ["analysis.json", *script_names]}
        fixed = self.epub("book-one-fixed", [
            ("One", [CHAPTER_ONE[1][0], "They leIt."]), CHAPTER_THREE], "book-one-2")
        status, stdout, stderr = self.run_cli("analyze", "--strategy", "consensus", fixed, "-o", earlier)
        self.assertEqual(status, 0, stderr)
        self.assertIn("Original analysis files and revision journal", stdout)
        for name, data in before.items():
            self.assertEqual((self.book / name).read_bytes(), data)
        from epub_tts_narrator.analysis_directory import _completed_directory, continuation_directories
        _completed_directory(earlier)
        _completed_directory(self.book)
        self.assertEqual(continuation_directories(self.book), [earlier.resolve()])

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

    def test_title_changes_are_reported_separately_from_paragraph_edits(self):
        saved = self.source("one", [CHAPTER_ONE, CHAPTER_TWO])
        epub = with_navigation_titles(
            make_serial_epub(self.root / "two.epub", "The Serial", "two", [CHAPTER_ONE, CHAPTER_TWO]),
            ["Chapter One", "Two"])
        current = source_from(epub)
        comparison = compare_sources(saved, current)
        self.assertEqual(comparison["retitled_chapters"], [
            {"id": "ch0001", "number": 1, "previous_title": "One", "title": "Chapter One"}])
        self.assertEqual((comparison["appended_chapters"], comparison["edited_chapters"]), ([], []))
        self.assertEqual(comparison["changed_blocks"], [])

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
