"""Every whole-book render opens and ends with a short narrator line, added without touching any chapter."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.renderer import spoken_ending
from epub_tts_narrator.renderer.assembly import probe_audio
from epub_tts_narrator.renderer.bundle import load_bundle, make_plan
from epub_tts_narrator.renderer.common import read_json
from epub_tts_narrator.renderer.config import DEFAULTS, load_config
from epub_tts_narrator.renderer.quality import validate_policy
from epub_tts_narrator.renderer.workflow import RenderBlocked, assemble, make_render_plan, render
from audition_fixtures import prepare_document
from test_chapter_export import chapter_document
from test_renderer_workflow import FFMPEG, FFPROBE, FixtureEngine, FixtureVerifier


class EndingTextTests(unittest.TestCase):
    def text(self, title, authors, **render):
        return spoken_ending.ending_text({"title": title, "authors": authors},
                                         {**spoken_ending.SETTINGS, **render})

    def test_the_line_names_the_title_and_every_author_in_natural_order(self):
        cases = [
            ("The Talisman", ["Stephen King"],
             "This has been The Talisman by Stephen King. Thank you for listening!"),
            ("Good Omens", ["Terry Pratchett", "Neil Gaiman"],
             "This has been Good Omens by Terry Pratchett and Neil Gaiman. Thank you for listening!"),
            ("Three Hands", ["A. One", "B. Two", "C. Three", "a. one", " "],
             "This has been Three Hands by A. One, B. Two, and C. Three. Thank you for listening!"),
        ]
        for title, authors, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(self.text(title, authors), expected)

    def test_a_series_suffix_in_parentheses_is_read_as_an_aside(self):
        self.assertEqual(self.text("Pawn of Prophecy (The Belgariad Book 1)", ["David Eddings"]),
                         "This has been Pawn of Prophecy, The Belgariad Book 1, by David Eddings. "
                         "Thank you for listening!")
        # Without authors the aside's comma does not stand before the full stop.
        self.assertEqual(self.text("Pawn of Prophecy [Belgariad 1]", []),
                         "This has been Pawn of Prophecy, Belgariad 1. Thank you for listening!")

    def test_a_book_without_authors_uses_the_line_without_authors(self):
        for authors in ([], None, ["", "  "]):
            with self.subTest(authors=authors):
                self.assertEqual(self.text("Night Harbor", authors),
                                 "This has been Night Harbor. Thank you for listening!")

    def test_a_reworded_line_fills_its_placeholders_and_a_book_without_a_title_has_no_ending(self):
        self.assertEqual(self.text("Night Harbor", ["Ann Lee"], ending_text="The end of {title}, by {authors}."),
                         "The end of Night Harbor, by Ann Lee.")
        self.assertEqual(self.text("Night Harbor", ["Ann Lee"], ending_text="Thanks for listening."),
                         "Thanks for listening.")
        self.assertIsNone(self.text("  ", ["Ann Lee"]))

    def test_invalid_settings_name_the_setting(self):
        cases = [
            ({"ending": "yes"}, "render.ending must be boolean"),
            ({"ending_text": "This was {book}."}, r"render.ending_text may use only \{authors\} and \{title\}"),
            ({"ending_text": "This was {title!r}."}, "render.ending_text may use only"),
            ({"ending_text": "This was {title."}, "render.ending_text has unbalanced braces"),
            ({"ending_text": ""}, "render.ending_text must be nonempty"),
            ({"ending_text": "x" * 401}, "at most 400"),
            ({"ending_text": "This was [title]."}, "without brackets"),
            ({"ending_text_without_authors": "By {authors}."}, r"may use only \{title\}"),
            ({"ending_title": ""}, "render.ending_title must be"),
            ({"ending_title": "End\nCredits"}, "render.ending_title must be"),
        ]
        for render, message in cases:
            with self.subTest(render=render), self.assertRaisesRegex(ValueError, message):
                spoken_ending.validate_settings({**spoken_ending.SETTINGS, **render})

    def test_configuration_turns_it_on_by_default_and_checks_a_rewording(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audiobook-render.toml"
            path.write_text("", encoding="utf-8")
            config = load_config(path, require_verifier=False, require_engine=False)
            self.assertIs(config["render"]["ending"], True)
            self.assertEqual(config["render"]["ending_title"], "End Credits")
            path.write_text('[render]\nending_text = "Goodbye from {narrator}."\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "render.ending_text may use only"):
                load_config(path, require_verifier=False, require_engine=False)


class EndingBookFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "output"
        self.work = self.root / "work"
        self.bundle_dir = self.root / "bundle"
        self.document = prepare_document(chapter_document([
            ("CHAPTER ONE", "heading", "p"),
            ("Let us leave before the morning bell.", "narration", "p"),
            ("CHAPTER TWO", "heading", "p"),
            ("At dawn the boat returned safely to shore.", "narration", "p"),
        ]))
        self.document["source"]["book"].update(title="Night Harbor (Harbor Tales Book 2)",
                                               authors=["Ann Lee", "Bo Park"])
        self.save_revision()
        self.config = deepcopy(DEFAULTS)
        self.config["engine"]["checkpoint"] = str(self.root / "unused-tts")
        self.config["verification"]["model"] = str(self.root / "unused-asr")
        self.config["voices"].update(directory=str(self.root / "voices"), series_id="ending-fixture")
        # The ending tests leave the opening out; OpeningTests turn it on.
        self.config["render"].update(work_dir=str(self.work), output_dir=str(self.output), max_attempts=2,
                                     ffmpeg=FFMPEG, ffprobe=FFPROBE, require_cover=False, opening=False)
        self.config["quality"] = validate_policy({})
        self.engine = FixtureEngine({})
        self.verifier = FixtureVerifier(self.engine.fixture_transcripts)
        self.messages = []

    def save_revision(self):
        replacement = self.root / "next-bundle"
        export_bundle(self.document, replacement)
        if self.bundle_dir.exists():
            shutil.rmtree(self.bundle_dir)
        replacement.rename(self.bundle_dir)
        self.bundle = load_bundle(self.bundle_dir)

    def run_render(self, **options):
        return render(self.bundle, self.config, self.engine, self.verifier, notify=self.messages.append, **options)

    def components(self):
        self.loaded = getattr(self, "loaded", 0) + 1
        return self.engine, self.verifier

    def package(self, components=True):
        return assemble(self.bundle, self.config, notify=self.messages.append,
                        ending_components=self.components if components else None)

    def checkpoint(self, name):
        return self.work / "completed-chapters" / f"{name}.json"

    def spoken(self):
        return [call["text"] for call in self.engine.speech_calls]

    def m4b_chapters(self, result):
        book = self.output / result["audio"]["audiobook"]["path"]
        return [chapter["tags"]["title"] for chapter in probe_audio(book, ffprobe=FFPROBE)["chapters"]]


LINE = "This has been Night Harbor, Harbor Tales Book 2, by Ann Lee and Bo Park. Thank you for listening!"


class EndingPlanTests(EndingBookFixture, unittest.TestCase):
    def test_a_whole_book_plan_ends_with_the_narrator_line_and_keeps_every_chapter_identity(self):
        plan = make_render_plan(self.bundle, self.config)
        source = make_plan(self.bundle, self.config)
        self.assertEqual(plan["chapters"][:-1], make_render_plan(self.bundle, self.config, ending=False)["chapters"])
        self.assertEqual([[chunk["chunk_id"] for chunk in chapter["chunks"]] for chapter in plan["chapters"][:-1]],
                         [[chunk["chunk_id"] for chunk in chapter["chunks"]] for chapter in source["chapters"]])
        ending = plan["chapters"][-1]
        self.assertEqual((ending["id"], ending["number"], ending["title"]), ("spoken-ending", 3, "End Credits"))
        [chunk] = ending["chunks"]
        self.assertEqual(chunk["spoken_text"], LINE)
        narrator = next(role for role in self.bundle["cast"]["roles"] if role["kind"] == "narration")
        self.assertEqual((chunk["segment"]["role_id"], chunk["segment"]["voice_id"]),
                         (narrator["id"], narrator["voice_id"]))
        self.assertIn(narrator["voice_id"], plan["voice_ids"])

    def test_no_ending_for_a_partial_book_a_disabled_setting_or_a_remote_chapter_service(self):
        partial = deepcopy(self.config)
        partial["render"]["chapter_numbers"] = [1]
        disabled = deepcopy(self.config)
        disabled["render"]["ending"] = False
        for name, config, options in (("partial", partial, {}), ("disabled", disabled, {}),
                                      ("remote", self.config, {"ending": False})):
            with self.subTest(name):
                plan = make_render_plan(self.bundle, config, **options)
                self.assertNotIn("spoken-ending", [chapter["id"] for chapter in plan["chapters"]])

    def test_a_reworded_line_is_a_new_passage(self):
        before = make_render_plan(self.bundle, self.config)["chapters"][-1]["chunks"][0]["chunk_id"]
        self.config["render"]["ending_text"] = "That was {title}."
        after = make_render_plan(self.bundle, self.config)["chapters"][-1]["chunks"]
        self.assertEqual([chunk["spoken_text"] for chunk in after], ["That was Night Harbor, Harbor Tales Book 2."])
        self.assertNotEqual(after[0]["chunk_id"], before)


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class EndingRenderTests(EndingBookFixture, unittest.TestCase):
    def test_render_speaks_verifies_and_publishes_the_ending_as_the_last_chapter(self):
        result = self.run_render()

        self.assertEqual(self.spoken()[-1], LINE)
        self.assertEqual(self.m4b_chapters(result), ["CHAPTER ONE", "CHAPTER TWO", "End Credits"])
        last = result["audio"]["chapters"][-1]
        self.assertEqual((last["id"], last["number"], last["path"]),
                         ("spoken-ending", 3, "chapters/spoken-ending.m4a"))
        self.assertTrue((self.output / last["path"]).is_file())
        checkpoint = read_json(self.checkpoint("spoken-ending"))
        [accepted] = checkpoint["acceptance"]
        # Verified like any take: the recognizer heard the exact line.
        self.assertIs(accepted["report"]["passed"], True)
        self.assertEqual(Path(checkpoint["audio_file"]).name, "spoken-ending.m4a")
        sync = read_json(self.output / "sync.json")
        self.assertEqual(sync["segments"][-1]["spoken_text"], LINE)

    def test_an_already_rendered_book_gets_the_ending_on_the_next_render_without_rerendering_chapters(self):
        self.config["render"]["ending"] = False
        first = self.run_render()
        self.assertEqual(self.m4b_chapters(first), ["CHAPTER ONE", "CHAPTER TWO"])
        saved = {number: self.checkpoint(f"chapter-{number:04d}").read_bytes() for number in (1, 2)}
        spoken = len(self.engine.speech_calls)

        self.config["render"]["ending"] = True
        result = self.run_render()

        self.assertEqual(self.spoken()[spoken:], [LINE])
        self.assertEqual({number: self.checkpoint(f"chapter-{number:04d}").read_bytes() for number in (1, 2)},
                         saved)
        self.assertEqual(self.m4b_chapters(result), ["CHAPTER ONE", "CHAPTER TWO", "End Credits"])
        # A later render reuses it like any finished chapter.
        self.run_render()
        self.assertEqual(self.spoken()[spoken:], [LINE])

    def test_assemble_renders_only_the_missing_ending_and_without_an_engine_says_to_render(self):
        self.config["render"]["ending"] = False
        self.run_render()
        saved = {number: self.checkpoint(f"chapter-{number:04d}").read_bytes() for number in (1, 2)}
        spoken = len(self.engine.speech_calls)
        self.config["render"]["ending"] = True

        with self.assertRaisesRegex(RenderBlocked, "spoken ending has not been rendered yet. Run render"):
            self.package(components=False)
        result = self.package()

        self.assertEqual(self.loaded, 1)
        self.assertEqual(self.spoken()[spoken:], [LINE])
        self.assertEqual({number: self.checkpoint(f"chapter-{number:04d}").read_bytes() for number in (1, 2)},
                         saved)
        self.assertEqual(self.m4b_chapters(result), ["CHAPTER ONE", "CHAPTER TWO", "End Credits"])
        # Once rendered, assembly needs no engine again.
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            self.package(components=False)

    def test_assemble_renders_a_reworded_ending_again_and_retitles_without_models(self):
        self.run_render()
        spoken = len(self.engine.speech_calls)
        self.config["render"]["ending_title"] = "Closing"
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            retitled = self.package(components=False)
        self.assertEqual(self.m4b_chapters(retitled)[-1], "Closing")

        self.config["render"]["ending_text"] = "That was {title}, read for you."
        result = self.package()

        self.assertEqual(self.spoken()[spoken:], ["That was Night Harbor, Harbor Tales Book 2, read for you."])
        self.assertEqual(self.m4b_chapters(result), ["CHAPTER ONE", "CHAPTER TWO", "Closing"])

    def test_a_failed_ending_take_blocks_publication_like_any_failed_chapter(self):
        from epub_tts_narrator.renderer.workflow import ChaptersRenderFailed
        self.config["render"]["ending_text"] = "Goodbye from {title}."
        self.verifier.wrong_texts.add("Goodbye from Night Harbor, Harbor Tales Book 2.")
        with self.assertRaises(ChaptersRenderFailed) as raised:
            self.run_render()
        self.assertEqual([failure["number"] for failure in raised.exception.failed_chapters], [3])
        self.assertFalse((self.output / "audio-manifest.json").exists())
        self.assertTrue(self.checkpoint("chapter-0002").exists(), "the book's chapters still finished")


class OpeningTextTests(unittest.TestCase):
    def text(self, title, authors, **render):
        return spoken_ending.opening_text({"title": title, "authors": authors},
                                          {**spoken_ending.SETTINGS, **render})

    def test_the_opening_names_the_title_and_authors(self):
        self.assertEqual(self.text("The Talisman", ["Stephen King"]),
                         "You are listening to The Talisman, by Stephen King.")
        self.assertEqual(self.text("Pawn of Prophecy (The Belgariad Book 1)", ["David Eddings", "Leigh Eddings"]),
                         "You are listening to Pawn of Prophecy, The Belgariad Book 1, by David Eddings and "
                         "Leigh Eddings.")

    def test_a_book_without_a_known_author_is_named_by_its_title_alone(self):
        for authors in ([], None, ["Unknown"]):
            with self.subTest(authors=authors):
                self.assertEqual(self.text("Pawn of Prophecy (The Belgariad Book 1)", authors),
                                 "You are listening to Pawn of Prophecy, The Belgariad Book 1.")
        self.assertIsNone(self.text(" ", ["Ann Lee"]))

    def test_invalid_opening_settings_name_the_setting(self):
        cases = [
            ({"opening": 1}, "render.opening must be boolean"),
            ({"opening_text": "Now {narrator} reads."}, "render.opening_text may use only"),
            ({"opening_text_without_authors": "By {authors}."}, "render.opening_text_without_authors may use only"),
            ({"opening_title": ""}, "render.opening_title must be"),
        ]
        for render, message in cases:
            with self.subTest(render=render), self.assertRaisesRegex(ValueError, message):
                spoken_ending.validate_settings({**spoken_ending.SETTINGS, **render})

    def test_configuration_turns_it_on_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audiobook-render.toml"
            path.write_text("", encoding="utf-8")
            config = load_config(path, require_verifier=False, require_engine=False)
            self.assertIs(config["render"]["opening"], True)
            self.assertEqual(config["render"]["opening_title"], "Opening Credits")


OPENING = "You are listening to Night Harbor, Harbor Tales Book 2, by Ann Lee and Bo Park."


class OpeningPlanTests(EndingBookFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config["render"]["opening"] = True

    def test_a_whole_book_plan_opens_with_the_narrator_line_and_keeps_every_number_and_identity(self):
        plan = make_render_plan(self.bundle, self.config)
        source = make_plan(self.bundle, self.config)
        self.assertEqual([chapter["id"] for chapter in plan["chapters"]],
                         ["spoken-opening", *[chapter["id"] for chapter in source["chapters"]], "spoken-ending"])
        self.assertEqual(plan["chapters"][1:-1], make_render_plan(self.bundle, self.config, ending=False)["chapters"])
        opening, ending = plan["chapters"][0], plan["chapters"][-1]
        # Every source chapter and the ending keep their numbers; the opening takes the next one.
        self.assertEqual((opening["number"], opening["title"], ending["number"]), (4, "Opening Credits", 3))
        [chunk] = opening["chunks"]
        self.assertEqual(chunk["spoken_text"], OPENING)
        narrator = next(role for role in self.bundle["cast"]["roles"] if role["kind"] == "narration")
        self.assertEqual((chunk["segment"]["role_id"], chunk["segment"]["voice_id"]),
                         (narrator["id"], narrator["voice_id"]))
        self.assertNotEqual(chunk["chunk_id"], ending["chunks"][0]["chunk_id"])

    def test_no_opening_when_disabled_for_a_partial_book_or_a_remote_chapter_service(self):
        disabled = deepcopy(self.config)
        disabled["render"]["opening"] = False
        partial = deepcopy(self.config)
        partial["render"]["chapter_numbers"] = [1]
        for name, config, options in (("disabled", disabled, {}), ("partial", partial, {}),
                                      ("remote", self.config, {"ending": False})):
            with self.subTest(name):
                plan = make_render_plan(self.bundle, config, **options)
                self.assertNotIn("spoken-opening", [chapter["id"] for chapter in plan["chapters"]])
        # Turning the opening off leaves the ending as it was.
        self.assertEqual(make_render_plan(self.bundle, disabled)["chapters"][-1],
                         make_render_plan(self.bundle, self.config)["chapters"][-1])


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required")
class OpeningRenderTests(EndingBookFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.config["render"]["opening"] = True

    def test_render_publishes_the_opening_as_the_first_chapter(self):
        result = self.run_render()

        self.assertIn(OPENING, self.spoken())
        self.assertEqual(self.m4b_chapters(result), ["Opening Credits", "CHAPTER ONE", "CHAPTER TWO", "End Credits"])
        first = result["audio"]["chapters"][0]
        self.assertEqual((first["id"], first["number"], first["ordinal"], first["path"]),
                         ("spoken-opening", 4, 1, "chapters/spoken-opening.m4a"))
        [accepted] = read_json(self.checkpoint("spoken-opening"))["acceptance"]
        self.assertIs(accepted["report"]["passed"], True)
        self.assertEqual(read_json(self.output / "sync.json")["segments"][0]["spoken_text"], OPENING)

    def test_assemble_adds_a_missing_opening_and_ending_loading_the_engine_once(self):
        self.config["render"].update(opening=False, ending=False)
        self.run_render()
        saved = {number: self.checkpoint(f"chapter-{number:04d}").read_bytes() for number in (1, 2)}
        spoken = len(self.engine.speech_calls)
        self.config["render"].update(opening=True, ending=True)

        with self.assertRaisesRegex(RenderBlocked, "spoken opening has not been rendered yet. Run render"):
            self.package(components=False)
        result = self.package()

        self.assertEqual(self.loaded, 1)
        self.assertEqual(self.spoken()[spoken:], [OPENING, LINE])
        self.assertEqual({number: self.checkpoint(f"chapter-{number:04d}").read_bytes() for number in (1, 2)},
                         saved)
        self.assertEqual(self.m4b_chapters(result), ["Opening Credits", "CHAPTER ONE", "CHAPTER TWO", "End Credits"])
        with patch("epub_tts_narrator.renderer.workflow.verified_take", side_effect=AssertionError("No TTS/ASR")):
            again = self.package(components=False)
        self.assertEqual(self.m4b_chapters(again)[0], "Opening Credits")


if __name__ == "__main__":
    unittest.main()
