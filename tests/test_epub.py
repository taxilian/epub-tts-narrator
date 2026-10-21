"""EPUB extractor checks using real, tiny ZIP fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from epub_tts_narrator.epub import extract_epub


def xhtml(body: str, title: str = "Document", *, head: str = "") -> str:
    return (f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
            f"<head><title>{title}</title>{head}</head><body>{body}</body></html>")


class EpubExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.epub"

    def make_book(self, *, documents=None, manifest=None, spine=None, extras=None,
                  package_path="OPS/book.opf", rootfile=None, compression=zipfile.ZIP_STORED):
        documents = documents if documents is not None else {"OPS/one.xhtml": xhtml("<p>Hello.</p>")}
        manifest = manifest if manifest is not None else '<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
        spine = spine if spine is not None else '<itemref idref="one"/>'
        package = ('<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:title>A Test Book</dc:title><dc:creator>First Author</dc:creator>'
                   '<dc:creator>Second Author</dc:creator><dc:language>en</dc:language>'
                   '</metadata><manifest>' + manifest + '</manifest><spine>' + spine + '</spine></package>')
        container = ('<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                     f'<rootfiles><rootfile full-path="{rootfile or package_path}" '
                     'media-type="application/oebps-package+xml"/></rootfiles></container>')
        with zipfile.ZipFile(self.path, "w", compression=compression) as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("META-INF/container.xml", container)
            archive.writestr(package_path, package)
            for name, content in {**documents, **(extras or {})}.items():
                archive.writestr(name, content)
        return self.path

    def assert_inline_coverage(self, block, *, kind, origin, expected_text):
        """Allow merged spans, but verify offsets and exactly which prose is cued."""
        text = block["text"]
        covered = set()
        for span in block.get("inline_spans", []):
            self.assertIs(type(span["start"]), int)
            self.assertIs(type(span["end"]), int)
            self.assertGreaterEqual(span["start"], 0)
            self.assertGreater(span["end"], span["start"])
            self.assertLessEqual(span["end"], len(text))
            self.assertIn(span["kind"], {"emphasis", "strong"})
            self.assertIn(span["origin"], {"tag", "style"})
            if (span["kind"], span["origin"]) == (kind, origin):
                covered.update(range(span["start"], span["end"]))
        # Collapsed whitespace at cue boundaries may belong to either neighbor.
        # Every nonwhitespace codepoint must still map to the right source cue.
        start = text.index(expected_text)
        expected = {index for index in range(start, start + len(expected_text))
                    if not text[index].isspace()}
        self.assertEqual({index for index in covered if not text[index].isspace()}, expected)
        return covered

    def test_spine_order_and_metadata_are_preserved(self):
        path = self.make_book(
            documents={"OPS/one.xhtml": xhtml("<h1>First</h1><p>One.</p>"),
                       "OPS/two.xhtml": xhtml("<h1>Second</h1><p>Two.</p>")},
            manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="two" href="two.xhtml" media-type="application/xhtml+xml"/>',
            spine='<itemref idref="two"/><itemref idref="one"/>')
        result = extract_epub(path)
        self.assertEqual([ch["title"] for ch in result["chapters"]], ["Second", "First"])
        self.assertEqual([b["text"] for b in result["blocks"]], ["Second", "Two.", "First", "One."])
        self.assertEqual(result["book"]["authors"], ["First Author", "Second Author"])
        self.assertEqual(result["book"]["language"], "en")
        self.assertEqual(result["book"]["id"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["chapters"][1]["block_ids"], ["b000003", "b000004"])

    def test_nested_inline_text_and_tails_are_not_duplicated(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<div><p id="sentence">“Hello, <em>dear <b>friend</b></em>,” she said.<br/> Again.</p></div>')})
        result = extract_epub(self.path)
        self.assertEqual([row["text"] for row in result["blocks"]], ['“Hello, dear friend,” she said. Again.'])
        self.assertEqual(result["blocks"][0]["source"],
                         {"href": "OPS/one.xhtml", "element_id": "sentence", "tag": "p"})
        emphasis = self.assert_inline_coverage(result["blocks"][0], kind="emphasis", origin="tag",
                                              expected_text="dear friend")
        strong = self.assert_inline_coverage(result["blocks"][0], kind="strong", origin="tag",
                                            expected_text="friend")
        self.assertTrue(strong & emphasis)

    def test_semantic_tag_aliases_preserve_text_and_source(self):
        for tag, kind in [("em", "emphasis"), ("i", "emphasis"),
                          ("strong", "strong"), ("b", "strong")]:
            with self.subTest(tag=tag):
                self.make_book(documents={"OPS/one.xhtml": xhtml(
                    f'<p id="line">Before <{tag}>cued words</{tag}> after.</p>')})
                block = extract_epub(self.path)["blocks"][0]
                self.assertEqual(block["text"], "Before cued words after.")
                self.assertEqual(block["source"],
                                 {"href": "OPS/one.xhtml", "element_id": "line", "tag": "p"})
                self.assert_inline_coverage(block, kind=kind, origin="tag", expected_text="cued words")

    def test_inline_span_offsets_use_normalized_unicode_codepoints(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<p id="unicode"> \n 🚀  Before\t<em>\n quiet <em>𝄞</em>\t'
            '<strong>words</strong>  </em> after. \n </p>')})
        block = extract_epub(self.path)["blocks"][0]
        self.assertEqual(block["text"], "🚀 Before quiet 𝄞 words after.")
        self.assertEqual(block["source"],
                         {"href": "OPS/one.xhtml", "element_id": "unicode", "tag": "p"})
        self.assert_inline_coverage(block, kind="emphasis", origin="tag", expected_text="quiet 𝄞 words")
        self.assert_inline_coverage(block, kind="strong", origin="tag", expected_text="words")

    def test_emphasis_crossing_block_boundaries_has_block_local_offsets(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<em> Before <p id="middle"> middle <b>🚀</b> words </p> After </em>')})
        result = extract_epub(self.path)
        self.assertEqual([block["text"] for block in result["blocks"]],
                         ["Before", "middle 🚀 words", "After"])
        self.assertEqual([block["source"] for block in result["blocks"]], [
            {"href": "OPS/one.xhtml", "element_id": None, "tag": "body"},
            {"href": "OPS/one.xhtml", "element_id": "middle", "tag": "p"},
            {"href": "OPS/one.xhtml", "element_id": None, "tag": "body"},
        ])
        for block in result["blocks"]:
            self.assert_inline_coverage(block, kind="emphasis", origin="tag", expected_text=block["text"])
        self.assert_inline_coverage(result["blocks"][1], kind="strong", origin="tag", expected_text="🚀")

    def test_inline_style_cues_preserve_text_and_source(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<p id="styled">Before <span style="font-style: italic">quiet '
            '<span style="font-weight: bold">words</span></span> after.</p>')})
        block = extract_epub(self.path)["blocks"][0]
        self.assertEqual(block["text"], "Before quiet words after.")
        self.assertEqual(block["source"],
                         {"href": "OPS/one.xhtml", "element_id": "styled", "tag": "p"})
        self.assert_inline_coverage(block, kind="emphasis", origin="style", expected_text="quiet words")
        self.assert_inline_coverage(block, kind="strong", origin="style", expected_text="words")

    def test_linked_local_class_style_cues_are_resolved_relative_to_xhtml(self):
        self.make_book(
            documents={"OPS/text/one.xhtml": xhtml(
                '<p id="styled">Before <span class="calibre11">quiet words</span> after.</p>',
                head='<link rel="stylesheet" type="text/css" href="../styles/book.css"/>'
                     '<link rel="stylesheet" type="text/css" href="../styles/pages.css"/>')},
            manifest='<item id="one" href="text/one.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="css" href="styles/book.css" media-type="text/css"/>',
            extras={"OPS/styles/book.css": '.calibre11 { font-style: italic; }',
                    "OPS/styles/pages.css": '@page { margin: 5pt; }'})
        result = extract_epub(self.path)
        block = result["blocks"][0]
        self.assertEqual(block["text"], "Before quiet words after.")
        self.assertEqual(block["source"],
                         {"href": "OPS/text/one.xhtml", "element_id": "styled", "tag": "p"})
        self.assert_inline_coverage(block, kind="emphasis", origin="style", expected_text="quiet words")
        self.assertFalse(any("unresolved" in warning or "unavailable" in warning for warning in result["warnings"]))

    def test_conflicting_class_style_warns_without_claiming_resolved_emphasis(self):
        self.make_book(
            documents={"OPS/one.xhtml": xhtml('<p id="conflict">Before <span class="x">words</span> after.</p>',
                head='<link rel="stylesheet" type="text/css" href="book.css"/>')},
            manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="css" href="book.css" media-type="text/css"/>',
            extras={"OPS/book.css": '.x { font-style: italic; } .x { font-style: normal; }'})
        result = extract_epub(self.path)
        block = result["blocks"][0]
        self.assertEqual(block["text"], "Before words after.")
        self.assertEqual(block["source"],
                         {"href": "OPS/one.xhtml", "element_id": "conflict", "tag": "p"})
        self.assertFalse(any(span["kind"] == "emphasis" for span in block.get("inline_spans", [])))
        self.assertTrue(any("ambig" in warning.lower() or "conflict" in warning.lower()
                            for warning in result["warnings"]))

    def test_quoted_editorial_metadata_does_not_discard_linked_class_emphasis(self):
        self.make_book(
            documents={"OPS/one.xhtml": xhtml(
                '<p id="styled">Before <span class="calibre11">quiet</span> '
                '<span class="calibre28">strong words</span> after.</p>',
                head='<link rel="stylesheet" href="book.css"/>')},
            extras={"OPS/book.css": '''
                .calibre11 { font-size: 1em; font-style: italic }
                .calibre28 { font-style: italic; font-weight: bold; font-family: "A Font" }
                .calibre49 { -aw-comment-author: "An Editor";
                    -aw-comment-datetime: "2022-07-20T19:23:40"; display: block }
            '''})
        result = extract_epub(self.path)
        block = result["blocks"][0]
        self.assertEqual(block["id"], "b000001")
        self.assertEqual(block["text"], "Before quiet strong words after.")
        self.assertEqual(block["source"],
                         {"href": "OPS/one.xhtml", "element_id": "styled", "tag": "p"})
        self.assertEqual(result["chapters"][0]["block_ids"], ["b000001"])
        self.assert_inline_coverage(block, kind="emphasis", origin="style",
                                    expected_text="quiet strong words")
        self.assert_inline_coverage(block, kind="strong", origin="style",
                                    expected_text="strong words")
        self.assertFalse(any("unresolved" in warning or "unavailable" in warning
                             for warning in result["warnings"]))

    def test_quoted_css_syntax_and_comments_cannot_create_or_disable_rules(self):
        self.make_book(
            documents={"OPS/one.xhtml": xhtml(
                '<p><span class="x">quiet</span> <span class="fake">plain</span></p>',
                head='<link rel="stylesheet" href="book.css"/>')},
            extras={"OPS/book.css": r'''
                .x { font-style: italic;
                    -aw-comment-author: "escaped \" quote; /* not a comment */ @import 'x'; }";
                    font-family: 'A { Font }';
                }
                .metadata { content: ".fake { font-weight: bold; }"; }
                /* .fake { font-style: italic; } @import "ignored.css"; */
            '''})
        result = extract_epub(self.path)
        block = result["blocks"][0]
        self.assertEqual(block["text"], "quiet plain")
        self.assert_inline_coverage(block, kind="emphasis", origin="style", expected_text="quiet")
        self.assertFalse(any(span["kind"] == "strong" for span in block.get("inline_spans", [])))
        self.assertFalse(any("unresolved" in warning or "unavailable" in warning
                             for warning in result["warnings"]))

    def test_inline_quoted_font_family_keeps_explicit_emphasis(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<p>Before <span style="font-family: &quot;A Font&quot;; font-style: italic">'
            'quiet words</span> after.</p>')})
        block = extract_epub(self.path)["blocks"][0]
        self.assert_inline_coverage(block, kind="emphasis", origin="style", expected_text="quiet words")

    def test_quoted_emphasis_values_are_not_css_keywords(self):
        for declaration in ['font-style: "italic"', 'font-style: italic "extra"',
                            'font-weight: "bold"', 'font-weight: bold "extra"']:
            for inline in (False, True):
                with self.subTest(declaration=declaration, inline=inline):
                    if inline:
                        body = '<p><span style="' + declaration.replace('"', "&quot;") + '">plain</span></p>'
                        head = ""
                    else:
                        body = '<p><span class="x">plain</span></p>'
                        head = f"<style>.x {{{declaration}}}</style>"
                    self.make_book(documents={"OPS/one.xhtml": xhtml(body, head=head)})
                    result = extract_epub(self.path)
                    self.assertFalse(result["blocks"][0].get("inline_spans"))
                    self.assertTrue(any("unresolved" in warning for warning in result["warnings"]))

    def test_real_unsupported_css_cascade_still_blocks_class_emphasis(self):
        for unsupported in [
            '@import "other.css";',
            '@media print { .x { font-style: normal; } }',
            '@supports (display: grid) { .x { font-style: normal; } }',
            '@layer alternate { .x { font-style: normal; } }',
            '.outer { .x { font-style: normal; } }',
            r'.\78 { font-style: normal; }',
            r'.x { f\6fnt-style: normal; }',
            '.x { font: 1em serif; }',
            '.x { all: initial; }',
            'span.x { font-style: normal; }',
        ]:
            with self.subTest(unsupported=unsupported):
                self.make_book(
                    documents={"OPS/one.xhtml": xhtml(
                        '<p><span class="x">plain</span></p>',
                        head='<style>.x { font-style: italic; }</style>'
                             '<link rel="stylesheet" href="book.css"/>')},
                    extras={"OPS/book.css": unsupported})
                result = extract_epub(self.path)
                self.assertFalse(result["blocks"][0].get("inline_spans"))
                self.assertTrue(any("unresolved" in warning or "unavailable" in warning
                                    or "not resolved" in warning for warning in result["warnings"]))

    def test_conditional_linked_and_embedded_stylesheets_remain_unresolved(self):
        for conditional in [
            '<link rel="stylesheet" href="book.css" media="print"/>',
            '<style media="print">.x { font-style: normal; }</style>',
        ]:
            with self.subTest(conditional=conditional):
                self.make_book(
                    documents={"OPS/one.xhtml": xhtml(
                        '<p><span class="x">plain</span></p>',
                        head='<style>.x { font-style: italic; }</style>' + conditional)},
                    extras={"OPS/book.css": '.x { font-style: normal; }'})
                result = extract_epub(self.path)
                self.assertFalse(result["blocks"][0].get("inline_spans"))
                self.assertTrue(any("conditional stylesheet" in warning for warning in result["warnings"]))

    def test_malformed_css_string_or_comment_cannot_certify_other_class_cues(self):
        for malformed in ['.metadata { content: "unterminated; }',
                          '.metadata { content: "bad\nnewline"; }',
                          '.metadata { /* unterminated',
                          '.metadata { content: "trailing\\']:
            with self.subTest(malformed=malformed):
                self.make_book(
                    documents={"OPS/one.xhtml": xhtml(
                        '<p><span class="x">plain</span></p>',
                        head='<style>.x { font-style: italic; }</style>'
                             '<link rel="stylesheet" href="book.css"/>')},
                    extras={"OPS/book.css": malformed})
                result = extract_epub(self.path)
                self.assertFalse(result["blocks"][0].get("inline_spans"))
                self.assertTrue(any("not resolved" in warning for warning in result["warnings"]))

    def test_empty_paragraphs_mark_only_the_next_emitted_block(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<p id="first">First.</p><p> \n <span> <span/> </span><br/></p><p/>'
            '<p id="second">Second.</p><p id="third">Third.</p><p> </p>')})
        result = extract_epub(self.path)
        self.assertEqual([block["text"] for block in result["blocks"]], ["First.", "Second.", "Third."])
        self.assertEqual([block["id"] for block in result["blocks"]], ["b000001", "b000002", "b000003"])
        self.assertEqual(result["chapters"][0]["block_ids"], ["b000001", "b000002", "b000003"])
        self.assertNotIn("break_before", result["blocks"][0])
        self.assertEqual(result["blocks"][1]["break_before"], "blank_paragraph")
        self.assertNotIn("break_before", result["blocks"][2])
        self.assertEqual(result["blocks"][1]["source"],
                         {"href": "OPS/one.xhtml", "element_id": "second", "tag": "p"})

    def test_leading_empty_paragraph_marks_the_first_text_block(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml('<p><span> </span><br/></p><h1>Chapter One</h1><p>Text.</p>')})
        blocks = extract_epub(self.path)["blocks"]
        self.assertEqual([block["text"] for block in blocks], ["Chapter One", "Text."])
        self.assertEqual(blocks[0]["break_before"], "blank_paragraph")
        self.assertNotIn("break_before", blocks[1])

    def test_nonempty_paragraph_line_break_does_not_mark_a_scene_break(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml('<p>First.<br/>Still first.</p><p>Second.</p>')})
        blocks = extract_epub(self.path)["blocks"]
        self.assertEqual([block["text"] for block in blocks], ["First. Still first.", "Second."])
        self.assertTrue(all("break_before" not in block for block in blocks))

    def test_mixed_content_is_retained_in_dom_order(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            'Before <em>the</em> section.<div id="group">Opening <b>words</b>.'
            '<p>Middle.</p>Closing.</div>After.<ul><li>Outer<ul><li>Inner.</li></ul>Tail.</li></ul>')})
        result = extract_epub(self.path)
        self.assertEqual([row["text"] for row in result["blocks"]],
                         ["Before the section.", "Opening words.", "Middle.", "Closing.",
                          "After.", "Outer", "Inner.", "Tail."])

    def test_footnotes_are_retained_and_media_are_warned(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml(
            '<p>See note<a href="#n">1</a>.</p><aside epub:type="footnote" id="n"><p>A footnote.</p></aside>'
            '<p>A<img src="pic.png" alt="Portrait"/>B.</p><script>bad()</script>')})
        result = extract_epub(self.path)
        self.assertEqual([row["text"] for row in result["blocks"]], ["See note1.", "A footnote.", "AB."])
        self.assertTrue(any("<img>" in message for message in result["warnings"]))
        self.assertTrue(any("<script>" in message for message in result["warnings"]))

    def test_navigation_and_non_linear_are_excluded_with_audit(self):
        self.make_book(
            documents={"OPS/one.xhtml": xhtml("<p>Main.</p>"), "OPS/nav.xhtml": xhtml("<p>Contents.</p>"),
                       "OPS/extra.xhtml": xhtml("<p>Extra.</p>"), "OPS/orphan.xhtml": xhtml("<p>Orphan.</p>")},
            manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
                     '<item id="extra" href="extra.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="orphan" href="orphan.xhtml" media-type="application/xhtml+xml"/>',
            spine='<itemref idref="nav"/><itemref idref="one"/><itemref idref="extra" linear="no"/>')
        result = extract_epub(self.path)
        self.assertEqual([row["text"] for row in result["blocks"]], ["Main."])
        self.assertEqual(len(result["exclusions"]), 3)
        included = extract_epub(self.path, include_auxiliary=True)
        self.assertEqual([row["text"] for row in included["blocks"]], ["Contents.", "Main.", "Extra."])
        self.assertEqual(len(included["exclusions"]), 1)

    def test_repeated_spine_resource_is_not_silently_deduplicated(self):
        self.make_book(spine='<itemref idref="one"/><itemref idref="one"/>')
        result = extract_epub(self.path)
        self.assertEqual([row["text"] for row in result["blocks"]], ["Hello.", "Hello."])
        self.assertEqual([ch["id"] for ch in result["chapters"]], ["ch0001", "ch0002"])
        self.assertTrue(any("Repeated spine" in message for message in result["warnings"]))

    def test_percent_decoding_and_relative_resolution(self):
        self.make_book(package_path="OPS/package/book.opf", rootfile="OPS/package/book%2Eopf",
                       documents={"OPS/text/one chapter.xhtml": xhtml("<p>Resolved.</p>")},
                       manifest='<item id="one" href="../text/one%20chapter.xhtml" media-type="application/xhtml+xml"/>')
        result = extract_epub(self.path)
        self.assertEqual(result["blocks"][0]["source"]["href"], "OPS/text/one chapter.xhtml")

    def test_local_xhtml_manifest_fallback(self):
        self.make_book(manifest='<item id="graphic" href="cover.svg" media-type="image/svg+xml" fallback="one"/>'
                               '<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>',
                       spine='<itemref idref="graphic"/>', extras={"OPS/cover.svg": "<svg/>"})
        result = extract_epub(self.path)
        self.assertEqual(result["blocks"][0]["text"], "Hello.")
        self.assertTrue(any("fallback" in message for message in result["warnings"]))
        self.assertEqual(result["exclusions"], [])

    def test_unsupported_primary_without_fallback_fails(self):
        self.make_book(manifest='<item id="one" href="cover.svg" media-type="image/svg+xml"/>',
                       documents={"OPS/cover.svg": "<svg/>"})
        with self.assertRaisesRegex(ValueError, "without a local XHTML fallback"):
            extract_epub(self.path)

    def test_fallback_cycle_fails(self):
        self.make_book(manifest='<item id="one" href="one.svg" media-type="image/svg+xml" fallback="two"/>'
                               '<item id="two" href="two.svg" media-type="image/svg+xml" fallback="one"/>')
        with self.assertRaisesRegex(ValueError, "cycle"):
            extract_epub(self.path)

    def test_missing_primary_content_fails(self):
        self.make_book(documents={})
        with self.assertRaisesRegex(ValueError, "Missing EPUB resource"):
            extract_epub(self.path)

    def test_unknown_spine_id_fails(self):
        self.make_book(spine='<itemref idref="missing"/>')
        with self.assertRaisesRegex(ValueError, "unknown manifest item"):
            extract_epub(self.path)

    def test_malformed_archive_fails(self):
        self.path.write_bytes(b"not a zip")
        with self.assertRaisesRegex(ValueError, "Unable to open EPUB"):
            extract_epub(self.path)

    def test_unsafe_archive_members_fail(self):
        for name in ["../escape", "/absolute", "OPS/../../escape", "OPS\\escape"]:
            with self.subTest(name=name):
                self.make_book(extras={name: "bad"})
                with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                    extract_epub(self.path)

    def test_unsafe_manifest_href_fails(self):
        for href in ["../../outside.xhtml", "%2e%2e/%2e%2e/outside.xhtml", "/absolute.xhtml", "one%00.xhtml", "one%XX.xhtml"]:
            with self.subTest(href=href):
                self.make_book(manifest=f'<item id="one" href="{href}" media-type="application/xhtml+xml"/>')
                with self.assertRaises(ValueError):
                    extract_epub(self.path)

    def test_duplicate_zip_member_fails(self):
        self.make_book()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.path, "a") as archive:
                archive.writestr("OPS/one.xhtml", xhtml("<p>Other.</p>"))
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP member"):
            extract_epub(self.path)

    def test_encrypted_spine_fails_but_obfuscated_fonts_do_not(self):
        def encryption(uri):
            return ('<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                    '<EncryptedData xmlns="http://www.w3.org/2001/04/xmlenc#"><CipherData>'
                    f'<CipherReference URI="{uri}"/></CipherData></EncryptedData></encryption>')
        self.make_book(extras={"META-INF/encryption.xml": encryption("OPS/one.xhtml")})
        with self.assertRaisesRegex(ValueError, "Encrypted EPUB content"):
            extract_epub(self.path)
        self.make_book(extras={"META-INF/encryption.xml": encryption("OPS/font.otf"), "OPS/font.otf": b"font"})
        self.assertEqual(extract_epub(self.path)["blocks"][0]["text"], "Hello.")

    def test_entity_expansion_is_rejected_including_utf16(self):
        document = '<!DOCTYPE html [<!ENTITY bomb "expanded">]>' + xhtml("<p>&bomb;</p>")
        for content in [document, document.encode("utf-16")]:
            with self.subTest(encoding=type(content).__name__):
                self.make_book(documents={"OPS/one.xhtml": content})
                with self.assertRaisesRegex(ValueError, "entity declarations"):
                    extract_epub(self.path)

    def test_safe_external_doctype_is_allowed(self):
        self.make_book(documents={"OPS/one.xhtml": '<!DOCTYPE html SYSTEM "http://example.invalid/xhtml.dtd">' + xhtml("<p>Works.</p>")})
        self.assertEqual(extract_epub(self.path)["blocks"][0]["text"], "Works.")

    def test_malformed_xhtml_fails(self):
        self.make_book(documents={"OPS/one.xhtml": "<html><body><p>Broken</body></html>"})
        with self.assertRaisesRegex(ValueError, "Malformed XML"):
            extract_epub(self.path)

    def test_empty_text_fails(self):
        self.make_book(documents={"OPS/one.xhtml": xhtml('<img src="scan.png"/>')})
        with self.assertRaisesRegex(ValueError, "no extractable text"):
            extract_epub(self.path)

    def test_zip_member_and_total_size_limits(self):
        self.make_book()
        with patch("epub_tts_narrator.epub.MAX_MEMBER_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "safety size"):
                extract_epub(self.path)
        with patch("epub_tts_narrator.epub.MAX_TOTAL_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "expanded size"):
                extract_epub(self.path)

    def test_suspicious_compression_ratio_fails(self):
        self.make_book(extras={"OPS/bomb.txt": b"A" * (2 * 1024 * 1024)}, compression=zipfile.ZIP_DEFLATED)
        with self.assertRaisesRegex(ValueError, "compression ratio"):
            extract_epub(self.path)

    def test_remote_nonspine_assets_are_not_fetched(self):
        self.make_book(manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
                               '<item id="remote" href="https://example.invalid/image.jpg" media-type="image/jpeg"/>')
        self.assertEqual(extract_epub(self.path)["blocks"][0]["text"], "Hello.")


if __name__ == "__main__":
    unittest.main()
