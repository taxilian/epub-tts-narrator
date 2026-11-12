from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from epub_tts_narrator.analysis_directory import fresh_registry, publish_directory
from epub_tts_narrator.chapter_title_update import source_for_title_refresh
from epub_tts_narrator.cli import export_bundle, main
from epub_tts_narrator.epub import extract_epub
from epub_tts_narrator.front_matter import drop_front_matter
from epub_tts_narrator.name_pronunciations import save_definition
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.assembly import probe_audio
from epub_tts_narrator.renderer.cli import main as renderer_main
from epub_tts_narrator.renderer.config import load_config as renderer_config
from epub_tts_narrator.renderer.workflow import render
from epub_tts_narrator.series import link_analysis
from audition_fixtures import prepare_bundle, prepare_document
from shared_fixtures import make_epub, repack_epub
from test_contracts import valid_document
from test_epub import xhtml
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


class NarratorChapterTitleRefreshTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.book = self.root / "book"
        self.epub = make_epub(
            self.root / "original.epub", "A Valley", "valley",
            xhtml("<p>Chapter Eight</p><p>All rights reserved.</p><p>The valley was quiet.</p>", "c114"))
        with zipfile.ZipFile(self.epub) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["content.opf"] = members["content.opf"].replace(
            b"</manifest>", b'<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest>'
        ).replace(b"<spine>", b'<spine toc="ncx">')
        members["toc.ncx"] = (
            b'<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap><navPoint>'
            b'<navLabel><text>Chapter Eight</text></navLabel><content src="one.xhtml"/>'
            b'</navPoint></navMap></ncx>')
        with zipfile.ZipFile(self.epub, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)

        source = drop_front_matter(extract_epub(self.epub))
        source["chapters"][0]["title"] = "c114"
        document = valid_document()
        template = deepcopy(document["segments"][0])
        document["source"] = source
        document["cast"]["characters"][0]["evidence"] = []
        document["cast"]["voices"][1]["traits"] = []
        document["segments"] = [dict(
            deepcopy(template), id=f"s{index}", block_id=block["id"], start=0,
            end=len(block["text"]), text=block["text"], kind="narration",
            role_id="narrator", voice_id="vn") for index, block in enumerate(source["blocks"], 1)]
        document["provenance"].update(strategy="narrator_read", source_fingerprint=fingerprint(source))
        document = prepare_document(document)
        registry = fresh_registry(self.book)
        linked = link_analysis(document, registry)
        publish_directory(linked, registry, self.book, export_bundle, source_epub=self.epub)
        self.config = self.root / "config.toml"
        self.config.write_text('[provider]\nmodel = "must-not-call"\n', encoding="utf-8")

    def refresh(self, expected_status=0):
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors), patch(
                "epub_tts_narrator.cli.OpenAICompatibleClient",
                side_effect=AssertionError("Title repair must not construct a model client")):
            status = main(["analyze", str(self.epub), "-o", str(self.book), "--config", str(self.config)])
        self.assertEqual(status, expected_status, errors.getvalue())

    def test_title_refresh_keeps_recorded_publisher_omissions_without_model_calls(self):
        before = read_json(self.book / "analysis.json")
        script = (self.book / "scripts/chapter-0001.jsonl").read_bytes()

        self.refresh()

        after = read_json(self.book / "analysis.json")
        self.assertEqual(after["source"]["chapters"][0]["title"], "Chapter Eight")
        self.assertEqual(after["source"]["blocks"], before["source"]["blocks"])
        self.assertEqual(after["source"]["exclusions"], before["source"]["exclusions"])
        self.assertEqual(after["segments"], before["segments"])
        self.assertEqual(after["cast"], before["cast"])
        self.assertEqual((self.book / "scripts/chapter-0001.jsonl").read_bytes(), script)
        self.assertEqual(load_bundle(self.book)["chapters"][0]["entry"]["title"], "Chapter Eight")
        self.assertEqual(after["provenance"]["chunk_identity_source_fingerprint"], fingerprint(before["source"]))

    def test_title_refresh_does_not_label_pending_name_uses(self):
        save_definition(self.book, "valley", "V AE1 L IY0", status="confirmed", source="manual",
                        parts_of_speech=["proper_noun"])
        choices = self.book / "name-pronunciations.json"
        before = choices.read_bytes()

        self.refresh(expected_status=2)

        self.assertEqual(choices.read_bytes(), before)
        self.assertEqual(read_json(self.book / "source.json")["chapters"][0]["title"], "Chapter Eight")

    def test_repeating_the_repair_preserves_all_book_files_without_model_calls(self):
        self.refresh()
        before = {path.relative_to(self.book): path.read_bytes()
                  for path in self.book.rglob("*") if path.is_file()}

        self.refresh()

        self.assertEqual({path.relative_to(self.book): path.read_bytes()
                          for path in self.book.rglob("*") if path.is_file()}, before)

    @unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
    def test_analyze_then_assemble_repairs_a_rendered_book_without_models(self):
        bundle = prepare_bundle(load_bundle(self.book))
        self.config.write_text(self.config.read_text()
                               + "[engine]\ncheckpoint = " + json.dumps(str(self.root / "unused-tts"))
                               + "\n[verification]\nmodel = " + json.dumps(str(self.root / "unused-asr"))
                               + "\n[render]\nrequire_cover = false\n", encoding="utf-8")
        config = renderer_config(self.config)
        engine = FixtureEngine({})
        verifier = FixtureVerifier(engine.fixture_transcripts)
        original = render(bundle, config, engine, verifier, notify=lambda _: None)
        work = self.book / "render-work"
        checkpoint = work / "completed-chapters/chapter-0001.json"
        before = read_json(checkpoint)
        pcm_before = read_json(work / "chapters" / before["cache_key"] / "chapter.json")["pcm_sha256"]
        accepted = {path.relative_to(work): path.read_bytes()
                    for path in (work / "clips").rglob("*") if path.is_file()}
        model_calls = len(engine.speech_calls), len(verifier.calls)

        self.refresh()
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors), \
             patch("epub_tts_narrator.renderer.engines.create_engine", side_effect=AssertionError("No TTS model")), \
             patch("epub_tts_narrator.renderer.pool.create_render_engine", side_effect=AssertionError("No TTS pool")), \
             patch("epub_tts_narrator.renderer.verification.create_verifier", side_effect=AssertionError("No ASR model")):
            status = renderer_main(["assemble", str(self.book), "--config", str(self.config),
                                    "--epub", str(self.epub)])

        self.assertEqual(status, 0, errors.getvalue())
        self.assertEqual((len(engine.speech_calls), len(verifier.calls)), model_calls)
        self.assertEqual({path.relative_to(work): path.read_bytes()
                          for path in (work / "clips").rglob("*") if path.is_file()}, accepted)
        after = read_json(checkpoint)
        self.assertEqual(after["acceptance"], before["acceptance"])
        self.assertEqual(read_json(work / "chapters" / after["cache_key"] / "chapter.json")["pcm_sha256"], pcm_before)
        book_path = self.book / "audiobook" / original["audio"]["audiobook"]["path"]
        self.assertEqual(probe_audio(book_path, ffprobe=FFPROBE)["chapters"][0]["tags"]["title"], "Chapter Eight")

    def test_recorded_omissions_cannot_hide_changed_text_or_resource_locations(self):
        original = read_json(self.book / "analysis.json")
        fresh = extract_epub(self.epub)
        for change in ("text", "href", "extra", "missing"):
            with self.subTest(change=change):
                source = deepcopy(fresh)
                if change == "text":
                    source["blocks"][-1]["text"] = "A different scene."
                elif change == "href":
                    source["blocks"][1]["source"]["href"] = "another.xhtml"
                elif change == "extra":
                    block = dict(deepcopy(source["blocks"][-1]), id="b000009", text="New story text.")
                    source["blocks"].append(block)
                    source["chapters"][0]["block_ids"].append(block["id"])
                else:
                    source["blocks"].pop(1)
                    source["chapters"][0]["block_ids"].remove("b000002")
                before = deepcopy(source)

                projected = source_for_title_refresh(original, source, epub_sha256=fresh["book"]["id"])

                self.assertIsNone(projected)
                self.assertEqual(source, before)

    def test_repacked_epub_does_not_authorize_dropping_blocks_by_saved_ids(self):
        original = read_json(self.book / "analysis.json")
        repacked = repack_epub(self.epub, self.root / "repacked.epub")
        source = extract_epub(repacked)
        self.assertNotEqual(source["book"]["id"], original["source"]["book"]["id"])

        projected = source_for_title_refresh(original, source, epub_sha256=source["book"]["id"])

        self.assertIsNone(projected)


if __name__ == "__main__":
    unittest.main()
