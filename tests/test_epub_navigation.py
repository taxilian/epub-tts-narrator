"""EPUB navigation-label checks using real, tiny ZIP fixtures."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import zipfile

from epub_tts_narrator.epub import extract_epub
from test_epub import xhtml


class EpubNavigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "navigation.epub"

    def make_book(self, *, documents, manifest, spine, extras=None,
                  package_path="OPS/book.opf", spine_attributes=""):
        package = ('<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   '<dc:title>A Test Book</dc:title><dc:language>en</dc:language></metadata>'
                   '<manifest>' + manifest + '</manifest><spine' + spine_attributes + '>' + spine + '</spine></package>')
        container = ('<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                     f'<rootfiles><rootfile full-path="{package_path}" '
                     'media-type="application/oebps-package+xml"/></rootfiles></container>')
        with zipfile.ZipFile(self.path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("META-INF/container.xml", container)
            archive.writestr(package_path, package)
            for name, content in {**documents, **(extras or {})}.items():
                archive.writestr(name, content)

    def test_ncx_navmap_labels_replace_gibberish_titles_without_changing_source(self):
        self.make_book(
            documents={
                "OPS/text/c5K.xhtml": xhtml('<p style="font-weight: bold">Opening One</p><p id="first">First text.</p>', "c5K"),
                "OPS/text/c114.xhtml": xhtml('<p class="chapter">Opening Eight</p><p id="eighth">Eighth text.</p>', "c114"),
            },
            manifest='<item id="one" href="text/c5K.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="eight" href="text/c114.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="ncx" href="toc.ncx" media-type="application/dtbncx+xml"/>',
            spine='<itemref idref="one"/><itemref idref="eight"/>',
            spine_attributes=' toc="ncx"',
            extras={"OPS/toc.ncx": '''<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
                <navMap>
                  <navPoint id="chapter-one"><navLabel><text>Chapter One</text></navLabel><content src="text/c5K.xhtml"/></navPoint>
                  <navPoint id="chapter-eight"><navLabel><text>Chapter Eight</text></navLabel><content src="text/c114.xhtml"/></navPoint>
                </navMap>
                <pageList>
                  <pageTarget id="page-one"><navLabel><text>1</text></navLabel><content src="text/c5K.xhtml#page-one"/></pageTarget>
                  <pageTarget id="page-eight"><navLabel><text>8</text></navLabel><content src="text/c114.xhtml#page-eight"/></pageTarget>
                </pageList>
            </ncx>'''})

        result = extract_epub(self.path)

        self.assertEqual([chapter["title"] for chapter in result["chapters"]], ["Chapter One", "Chapter Eight"])
        self.assertEqual([chapter["id"] for chapter in result["chapters"]], ["ch0001", "ch0002"])
        self.assertEqual([chapter["source_href"] for chapter in result["chapters"]],
                         ["OPS/text/c5K.xhtml", "OPS/text/c114.xhtml"])
        self.assertEqual([chapter["block_ids"] for chapter in result["chapters"]],
                         [["b000001", "b000002"], ["b000003", "b000004"]])
        self.assertEqual([(block["id"], block["chapter_id"], block["text"]) for block in result["blocks"]], [
            ("b000001", "ch0001", "Opening One"), ("b000002", "ch0001", "First text."),
            ("b000003", "ch0002", "Opening Eight"), ("b000004", "ch0002", "Eighth text."),
        ])
        self.assertEqual(result["extraction_policy"],
                         "spine-xhtml-dom-v1; whitespace=collapse; footnotes=retain; auxiliary-spine=exclude; "
                         "inline-cues=source-v1; blank-paragraphs=mark")

    def test_epub3_toc_label_ignores_landmarks_page_list_and_structural_heading(self):
        self.make_book(
            documents={"OPS/one.xhtml": xhtml("<h1>Structural Heading</h1><p>Main text.</p>", "c5K")},
            manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            spine='<itemref idref="one"/>',
            extras={"OPS/nav.xhtml": xhtml('''
                <nav epub:type="landmarks"><ol><li><a href="one.xhtml">Landmark Label</a></li></ol></nav>
                <nav epub:type="toc"><ol><li><a href="one.xhtml">Actual Chapter</a></li></ol></nav>
                <nav epub:type="page-list"><ol><li><a href="one.xhtml#page-one">1</a></li></ol></nav>
            ''', "Contents")})

        result = extract_epub(self.path)

        self.assertEqual(result["chapters"][0]["title"], "Actual Chapter")
        self.assertEqual([block["text"] for block in result["blocks"]], ["Structural Heading", "Main text."])
        self.assertEqual(result["chapters"][0]["block_ids"], ["b000001", "b000002"])
        self.assertEqual(result["exclusions"], [{"href": "OPS/nav.xhtml", "reason": "navigation document outside spine"}])

    def test_epub3_nested_relative_percent_encoded_fragment_targets_keep_first_label(self):
        self.make_book(
            package_path="OPS/package/book.opf",
            documents={"OPS/text/one chapter.xhtml": xhtml("<p>Nested text.</p>", "c4Z")},
            manifest='<item id="one" href="../text/one%20chapter.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="nav" href="../nav/contents.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            spine='<itemref idref="one"/>',
            extras={"OPS/nav/contents.xhtml": xhtml('''
                <nav epub:type="toc"><ol><li>
                  <a href="../text/one%20chapter.xhtml#opening">Part One: <span>Maragor</span></a>
                  <ol><li><a href="../text/one%20chapter.xhtml#later">Later Subheading</a></li></ol>
                </li></ol></nav>
            ''', "Contents")})

        result = extract_epub(self.path)

        self.assertEqual(result["chapters"][0]["title"], "Part One: Maragor")
        self.assertEqual(result["chapters"][0]["source_href"], "OPS/text/one chapter.xhtml")
        self.assertEqual([(block["id"], block["text"], block["source"]["href"]) for block in result["blocks"]],
                         [("b000001", "Nested text.", "OPS/text/one chapter.xhtml")])

    def test_epub3_labels_take_precedence_and_ncx_fills_uncovered_resources(self):
        self.make_book(
            documents={"OPS/one.xhtml": xhtml("<p>One.</p>", "HTML One"),
                       "OPS/two.xhtml": xhtml("<p>Two.</p>", "HTML Two")},
            manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="two" href="two.xhtml" media-type="application/xhtml+xml"/>'
                     '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
                     '<item id="ncx" href="toc.ncx" media-type="application/dtbncx+xml"/>',
            spine='<itemref idref="one"/><itemref idref="two"/>',
            spine_attributes=' toc="ncx"',
            extras={
                "OPS/nav.xhtml": xhtml('<nav epub:type="toc"><ol><li><a href="one.xhtml">EPUB3 One</a></li></ol></nav>'),
                "OPS/toc.ncx": '''<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
                    <navPoint><navLabel><text>NCX One</text></navLabel><content src="one.xhtml"/></navPoint>
                    <navPoint><navLabel><text>NCX Two</text></navLabel><content src="two.xhtml"/></navPoint>
                </navMap></ncx>''',
            })

        result = extract_epub(self.path)

        self.assertEqual([chapter["title"] for chapter in result["chapters"]], ["EPUB3 One", "NCX Two"])
        self.assertEqual([block["text"] for block in result["blocks"]], ["One.", "Two."])

    def test_heading_html_title_and_filename_fallbacks_remain_without_toc(self):
        cases = [
            ("<h1>Structural Heading</h1><p>Text.</p>", "HTML Title", "Structural Heading"),
            ("<p>Text.</p>", "HTML Title", "HTML Title"),
            ("<p>Text.</p>", "", "one"),
        ]
        for body, document_title, expected in cases:
            with self.subTest(expected=expected):
                self.make_book(
                    documents={"OPS/one.xhtml": xhtml(body, document_title)},
                    manifest='<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>',
                    spine='<itemref idref="one"/>')

                result = extract_epub(self.path)

                self.assertEqual(result["chapters"][0]["title"], expected)
                self.assertEqual(result["chapters"][0]["source_href"], "OPS/one.xhtml")


if __name__ == "__main__":
    unittest.main()
