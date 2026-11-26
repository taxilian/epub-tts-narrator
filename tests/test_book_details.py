"""A book's title and author: placeholders are no author, a missing one is found or asked, review-ui edits them."""

import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from epub_tts_narrator.analysis_directory import _completed_directory, fresh_registry, publish_directory
from epub_tts_narrator.analysis_revision import related_directories, revise_directories
from epub_tts_narrator.author_review import find_authors, needs_author, review_author
from epub_tts_narrator.book_details import (
    bundle_book, document_book, placeholder, real_authors, save_details, split_authors, with_details,
)
from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.config import DEFAULTS
from epub_tts_narrator.epub import extract_epub
from epub_tts_narrator.pipeline import fingerprint
from epub_tts_narrator.renderer.book_metadata import book_tags, publication_assets
from epub_tts_narrator.renderer.book_output import resolve_audiobook_filename
from epub_tts_narrator.renderer.bundle import load_bundle
from epub_tts_narrator.renderer.config import DEFAULTS as RENDER_DEFAULTS
from epub_tts_narrator.renderer.spoken_ending import ending_text
from epub_tts_narrator.review_tui import ReviewApp
from epub_tts_narrator.series import link_analysis
from tests.test_review_tui import FakeScreen, review_document


def publish(root, authors=("A. Writer",), name="book"):
    directory = Path(root) / name
    document = review_document()
    document["source"]["book"]["authors"] = list(authors)
    registry = fresh_registry(directory)
    linked = link_analysis(document, registry)
    publish_directory(linked, registry, directory, export_bundle)
    return directory, linked


def epub_with_creator(path, creator):
    opf = ('<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata '
           'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Killing Time</dc:title>'
           f'<dc:creator>{creator}</dc:creator><dc:language>en</dc:language></metadata>'
           '<manifest><item id="one" href="one.xhtml" media-type="application/xhtml+xml"/></manifest>'
           '<spine><itemref idref="one"/></spine></package>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:'
                         'container"><rootfiles><rootfile full-path="book.opf" media-type="application/'
                         'oebps-package+xml"/></rootfiles></container>')
        archive.writestr("book.opf", opf)
        archive.writestr("one.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>One</title></head>'
                         '<body><p>Copyright © 2002 by Jo Penn.</p><p>It began.</p></body></html>')
    return path


def config():
    return {"analysis": deepcopy(DEFAULTS["analysis"]),
            "provider": {"model": "test-model", "base_url": "https://example.invalid"}}


class AuthorClient:
    """Answers the author request with ``answer(paragraphs)``: ``{number: text}`` -> ``{"a": [...]}``."""

    def __init__(self, answer):
        self.model, self.base_url = "test-model", "https://example.invalid"
        self.identity = {"model": self.model, "base_url": self.base_url}
        self.answer, self.calls = answer, []

    def complete(self, system, payload):
        if payload["mode"] != "author_find":
            raise AssertionError(f"Unexpected request mode: {payload['mode']}")
        self.calls.append(deepcopy(payload))
        paragraphs = {int(line.partition("|")[0]): line.partition("|")[2] for line in payload["text"].splitlines()}
        return self.answer(paragraphs)


class PlaceholderAuthorTests(unittest.TestCase):
    def test_a_conversion_tools_placeholder_is_no_author(self):
        for name in ("Unknown", " unknown ", "[Unknown]", "Unknown Author", "N/A", "", "?"):
            self.assertTrue(placeholder(name), name)
        for name in ("Anne McCaffrey", "Anonymous", "Unknownia Smith"):
            self.assertFalse(placeholder(name), name)
        self.assertEqual(real_authors(["Unknown", "Jo  Penn", "jo penn", ""]), ["Jo Penn"])

    def test_extraction_drops_a_placeholder_creator(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = extract_epub(epub_with_creator(Path(temporary) / "calibre.epub", "Unknown"))
        self.assertEqual(source["book"]["authors"], [])

    def test_a_book_saved_with_unknown_reads_as_having_no_author_everywhere(self):
        book = {"id": "b", "title": "Killing Time", "authors": ["Unknown"], "language": "en"}
        self.assertEqual(document_book({"source": {"book": book}, "provenance": {}})["authors"], [])
        bundle = {"source": {"book": book}, "provenance": {}}
        self.assertEqual(ending_text(bundle_book(bundle), {}), "This has been Killing Time. Thank you for listening!")
        self.assertEqual(ending_text(book, {}), "This has been Killing Time. Thank you for listening!")
        self.assertEqual(book_tags(book)["artist"], "")

    def test_typed_authors_refuse_placeholders(self):
        self.assertEqual(split_authors(" Jo Penn ; Al Roe "), ["Jo Penn", "Al Roe"])
        self.assertEqual(split_authors(""), [])
        with self.assertRaisesRegex(ValueError, "not an author's name"):
            split_authors("Unknown")


class BookDetailsRevisionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_a_chosen_title_and_author_change_no_source_identity_and_reach_the_renderer(self):
        directory, document = publish(self.root, authors=["Unknown"])
        before = fingerprint(document["source"])
        save_details(directory, title="The Long Walk", authors=["Jo Penn"], progress=lambda _: None)
        saved = _completed_directory(directory)[0]
        self.assertEqual(fingerprint(saved["source"]), before)
        self.assertEqual(saved["source"]["book"]["authors"], ["Unknown"])
        self.assertEqual(saved["provenance"]["book_details"]["authors"], {"value": ["Jo Penn"], "basis": "user"})
        bundle = load_bundle(directory)
        self.assertEqual(bundle["manifest"]["source_fingerprint"], before)
        book = bundle_book(bundle)
        self.assertEqual((book["title"], book["authors"]), ("The Long Walk", ["Jo Penn"]))
        self.assertEqual(ending_text(book, {}), "This has been The Long Walk by Jo Penn. Thank you for listening!")

    def test_the_scope_refuses_any_other_change(self):
        directory, document = publish(self.root)
        candidate = with_details(document, title="New")
        candidate["segments"][0]["delivery"]["direction"] = "Loudly"
        hashes = {str(directory): hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "only provenance.book_details"):
            revise_directories({directory: candidate}, directories=[directory], expected_manifest_hashes=hashes,
                               scope="book-details", progress=lambda _: None)

    def test_a_value_equal_to_the_epubs_clears_the_choice_but_no_author_is_kept(self):
        _, document = publish(self.root)
        chosen = with_details(document, authors=["Jo Penn"])
        self.assertNotIn("book_details", with_details(chosen, authors=["A. Writer"])["provenance"])
        _, anonymous = publish(self.root, authors=[], name="other")
        none = with_details(anonymous, authors=[])
        self.assertEqual(none["provenance"]["book_details"]["authors"], {"value": [], "basis": "user"})
        self.assertFalse(needs_author(none))

    def test_publication_keeps_a_chosen_author_over_the_epubs_placeholder(self):
        epub = epub_with_creator(self.root / "calibre.epub", "Unknown")
        book = {"id": hashlib.sha256(epub.read_bytes()).hexdigest(), "title": "Killing Time", "authors": [],
                "language": "en"}
        render = deepcopy(RENDER_DEFAULTS)
        render["render"].update(epub=str(epub), require_cover=False)
        bundle = {"source": {"book": book}, "provenance": {}}
        published, _, _ = publication_assets(bundle, render, self.root / "artwork")
        self.assertEqual(published["authors"], [])
        bundle["provenance"] = {"book_details": {"version": 1, "title": {"value": "Time Kept", "basis": "user"},
                                                 "authors": {"value": ["Jo Penn"], "basis": "analysis"}}}
        published, _, _ = publication_assets(bundle, render, self.root / "artwork")
        self.assertEqual((published["title"], published["authors"]), ("Time Kept", ["Jo Penn"]))

    def test_a_retitled_book_gets_a_new_file_name_unless_its_name_is_its_own(self):
        previous = {"book": {"title": "Old Title"}, "audiobook": {"path": "Old Title.m4b"}}
        self.assertEqual(resolve_audiobook_filename({"title": "New Title"}, previous), "New Title.m4b")
        self.assertEqual(resolve_audiobook_filename({"title": "Old Title"}, previous), "Old Title.m4b")
        renamed = {"book": {"title": "Old Title"}, "audiobook": {"path": "My copy.m4b"}}
        self.assertEqual(resolve_audiobook_filename({"title": "New Title"}, renamed), "My copy.m4b")


class AuthorReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def run_review(self, directory, client, **options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            changed = review_author(directory, config(), lambda settings: client, lambda message: None, **options)
        return changed, output.getvalue()

    def test_an_author_the_books_pages_name_is_saved_with_its_quote(self):
        directory, _ = publish(self.root, authors=["Unknown"])

        def answer(paragraphs):
            number, text = next((number, text) for number, text in paragraphs.items() if "Mara" in text)
            return {"a": [["Mara", number, text]]}
        client = AuthorClient(answer)
        changed, output = self.run_review(directory, client, is_interactive=lambda: False)
        self.assertTrue(changed, output)
        record = _completed_directory(directory)[0]["provenance"]["book_details"]["authors"]
        self.assertEqual((record["value"], record["basis"]), (["Mara"], "analysis"))
        self.assertIn("untrusted", self.system_prompt())
        self.assertEqual(client.calls[0]["title"], "A Short Walk")
        # Found once: the next rerun neither asks the model nor the person.
        self.assertFalse(self.run_review(directory, client, ask=self.fail)[0])
        self.assertEqual(len(client.calls), 1)

    @staticmethod
    def system_prompt():
        from epub_tts_narrator.rerun_review_prompts import AUTHOR_FIND
        return AUTHOR_FIND

    def test_a_name_the_quote_does_not_contain_is_refused(self):
        paragraphs = ["Copyright © 2002 by Jo Penn.", "It began."]
        self.assertEqual(find_authors("T", paragraphs, lambda system, payload, validate: validate(
            {"a": [["Jo Penn", 1, "Copyright © 2002 by Jo Penn."]]})), [("Jo Penn", "Copyright © 2002 by Jo Penn.")])
        with self.assertRaisesRegex(ValueError, "contain the author's name"):
            find_authors("T", paragraphs, lambda system, payload, validate: validate(
                {"a": [["Anne McCaffrey", 1, "Copyright © 2002 by Jo Penn."]]}))
        with self.assertRaisesRegex(ValueError, "exact text"):
            find_authors("T", paragraphs, lambda system, payload, validate: validate(
                {"a": [["Jo Penn", 2, "by Jo Penn"]]}))

    def test_without_an_answer_a_terminal_is_asked_and_otherwise_nothing_is_made_up(self):
        directory, _ = publish(self.root, authors=[])
        client = AuthorClient(lambda paragraphs: {"a": []})
        changed, output = self.run_review(directory, client, is_interactive=lambda: False)
        self.assertFalse(changed)
        self.assertIn("Author: unknown", output)
        self.assertNotIn("book_details", _completed_directory(directory)[0]["provenance"])
        answers = iter(["Unknown", "Jo Penn; Al Roe"])
        changed, output = self.run_review(directory, client, is_interactive=lambda: True,
                                          ask=lambda prompt: next(answers))
        self.assertTrue(changed, output)
        record = _completed_directory(directory)[0]["provenance"]["book_details"]["authors"]
        self.assertEqual(record, {"value": ["Jo Penn", "Al Roe"], "basis": "user"})

    def test_a_provider_failure_leaves_the_author_unknown_without_ending_the_run(self):
        from epub_tts_narrator.provider import ProviderError
        directory, _ = publish(self.root, authors=[])

        def fail(paragraphs):
            raise ProviderError("offline")
        changed, output = self.run_review(directory, AuthorClient(fail), is_interactive=lambda: False)
        self.assertFalse(changed)
        self.assertIn("could not be asked", output)

    def test_enter_leaves_it_unknown_and_a_book_with_an_author_is_not_checked(self):
        directory, _ = publish(self.root, authors=[])
        changed, _ = self.run_review(directory, AuthorClient(lambda paragraphs: {"a": []}),
                                     is_interactive=lambda: True, ask=lambda prompt: "")
        self.assertFalse(changed)
        named, _ = publish(self.root, name="named")
        client = AuthorClient(lambda paragraphs: self.fail("no request for a book with an author"))
        self.assertFalse(self.run_review(named, client)[0])


class ReviewUiBookDetailsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def app(self, authors=("Unknown",)):
        directory, document = publish(self.root, authors=authors)
        return directory, ReviewApp(directory, document, revise_directories=revise_directories,
                                    related_directories=related_directories, player=object())

    def test_curses_keys_edit_the_title_and_author_and_save_at_once(self):
        directory, app = self.app()
        screen = FakeScreen()
        app.handle_key("b")
        app.draw(screen)
        self.assertIn("Author(s): (unknown)", screen.text)
        app.handle_key("DOWN")
        app.handle_key("ENTER")
        for key in "Jo Penn":
            app.handle_key("SPACE" if key == " " else key)
        app.handle_key("ENTER")
        self.assertEqual(app.view, "book", app.notice)
        app.handle_key("UP")
        app.handle_key("ENTER")
        app.book_buffer = ""
        for key in "Home":
            app.handle_key(key)
        app.handle_key("ENTER")
        saved = _completed_directory(directory)[0]["provenance"]["book_details"]
        self.assertEqual((saved["title"]["value"], saved["authors"]["value"]), ("Home", ["Jo Penn"]))
        app.handle_key("ESC")
        app.draw(screen)
        self.assertIn("Home", screen.text)

    def test_a_placeholder_is_refused_and_pending_attribution_edits_still_save_after(self):
        directory, app = self.app()
        self.assertFalse(app.set_book_details(authors="Unknown"))
        self.assertIn("not an author's name", app.notice)
        app.handle_key("ENTER")
        app.handle_key("DOWN")
        app.handle_key("ENTER")
        self.assertTrue(app.set_direction("Quietly"))
        self.assertTrue(app.set_book_details(authors="Jo Penn"))
        self.assertTrue(app.save(), app.notice)
        saved = _completed_directory(directory)[0]
        self.assertEqual(saved["provenance"]["book_details"]["authors"]["value"], ["Jo Penn"])
        self.assertIn("Quietly", json.dumps(saved["segments"]))



class ReviewTuiBookDetailsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import asyncio
        asyncio.get_running_loop().slow_callback_duration = 10

    async def test_b_edits_the_author_in_the_full_screen_review(self):
        from epub_tts_narrator.review_tui_app import ReviewTuiApp
        from epub_tts_narrator.tui_widgets import ChoiceScreen, TextEditScreen
        with tempfile.TemporaryDirectory() as temporary:
            directory, document = publish(Path(temporary).resolve(), authors=["Unknown"])
            controller = ReviewApp(directory, document, revise_directories=revise_directories,
                                   related_directories=related_directories, player=object())
            app = ReviewTuiApp(controller)
            async with app.run_test(size=(150, 45)) as pilot:
                await pilot.press("b")
                await pilot.pause()
                self.assertIsInstance(app.screen, ChoiceScreen)
                self.assertIn("(unknown)", app.screen.options[1][1])
                app.screen.dismiss("authors")
                await pilot.pause()
                self.assertIsInstance(app.screen, TextEditScreen)
                await pilot.press(*"Unknown", "enter")
                await pilot.pause()
                self.assertIsInstance(app.screen, TextEditScreen)  # A placeholder is refused; the dialog stays.
                await pilot.press("ctrl+a", "ctrl+k", *"Jo Penn", "enter")
                await pilot.pause()
            saved = _completed_directory(directory)[0]["provenance"]["book_details"]["authors"]
            self.assertEqual(saved, {"value": ["Jo Penn"], "basis": "user"})


if __name__ == "__main__":
    unittest.main()
