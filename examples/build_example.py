"""Create a fictional, hand-annotated example. No model or TTS requests."""

from pathlib import Path
import sys
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epub_tts_narrator.cli import export_bundle
from epub_tts_narrator.contracts import readiness
from epub_tts_narrator.epub import extract_epub
from epub_tts_narrator.pipeline import apply_edits, base_document, fingerprint, segment, write_json


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paragraphs = [
        "My name is Mara. My voice is low and rough.",
        "“Wait,” I said, “the lantern is still warm.”",
        "Lena answered in a bright, clear voice. “Then someone was here.”",
        "Lena and I watched the harbor wake.",
        "“Did you hear that?”",
    ]
    resources = {
        "mimetype": "application/epub+zip",
        "META-INF/container.xml": '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        "content.opf": '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="id">urn:example:the-last-lantern</dc:identifier><dc:title>The Last Lantern</dc:title><dc:creator>Fictional demonstration</dc:creator><dc:language>en</dc:language><meta property="dcterms:modified">2026-10-19T00:00:00Z</meta></metadata><manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/></manifest><spine><itemref idref="chapter"/></spine></package>',
        "chapter.xhtml": '<html xmlns="http://www.w3.org/1999/xhtml" lang="en"><head><title>The Last Lantern</title></head><body><h1>The Last Lantern</h1>' + ''.join(f'<p id="p{i}">{escape(p)}</p>' for i, p in enumerate(paragraphs, 1)) + '</body></html>',
        "nav.xhtml": '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head><body><nav epub:type="toc"><ol><li><a href="chapter.xhtml">The Last Lantern</a></li></ol></nav></body></html>',
    }
    epub = output / "the-last-lantern.epub"
    with ZipFile(epub, "x") as archive:
        for name, content in resources.items():
            info = ZipInfo(name, date_time=(2026, 10, 19, 0, 0, 0))
            info.compress_type = 0 if name == "mimetype" else ZIP_DEFLATED
            archive.writestr(info, content)
    source = extract_epub(epub)
    blocks = source["blocks"]

    def evidence(index, quote):
        return [{"block_id": blocks[index]["id"], "quote": quote}]

    cast = {
        "characters": [
            {"id": "mara", "name": "Mara", "aliases": [], "description": "First-person narrator and character.", "evidence": evidence(1, "My name is Mara."), "voice_id": "voice_mara"},
            {"id": "lena", "name": "Lena", "aliases": [], "description": "Mara's companion at the harbor.", "evidence": evidence(3, "Lena answered"), "voice_id": "voice_lena"},
        ],
        "roles": [
            {"id": "narrator_main", "kind": "narration", "character_id": "mara", "voice_id": "voice_mara", "description": "Mara narrates in first person."},
            {"id": "mara_dialogue", "kind": "dialogue", "character_id": "mara", "voice_id": "voice_mara", "description": "Mara speaking."},
            {"id": "lena_dialogue", "kind": "dialogue", "character_id": "lena", "voice_id": "voice_lena", "description": "Lena speaking."},
        ],
        "voices": [
            {"id": "voice_mara", "description": "Low-pitched, rough-textured voice. Measured pacing is a proposed casting choice.", "traits": [
                {"name": "pitch", "value": "low", "basis": "text", "evidence": evidence(1, "low and rough")},
                {"name": "texture", "value": "rough", "basis": "text", "evidence": evidence(1, "low and rough")},
                {"name": "pace", "value": "measured", "basis": "casting", "evidence": []},
            ]},
            {"id": "voice_lena", "description": "Bright, clear voice; keep it audibly distinct from Mara.", "traits": [
                {"name": "tone", "value": "bright and clear", "basis": "text", "evidence": evidence(3, "bright, clear voice")},
            ]},
        ],
    }
    roles = {r["id"]: r for r in cast["roles"]}
    segments = []
    for index, block in enumerate(blocks):
        text = block["text"]
        if index == 2:
            middle = " I said, "
            split = text.index(middle)
            pieces = [(text[:split], "mara_dialogue", "dialogue"),
                      (middle, "narrator_main", "narration"),
                      (text[split + len(middle):], "mara_dialogue", "dialogue")]
        elif index == 3:
            split = text.index("“")
            pieces = [(text[:split], "narrator_main", "narration"), (text[split:], "lena_dialogue", "dialogue")]
        elif index == 5:
            pieces = [(text, None, "dialogue")]
        else:
            pieces = [(text, "narrator_main", "heading" if index == 0 else "narration")]
        position = 0
        for text, role_id, kind in pieces:
            unresolved = role_id is None
            segments.append(segment(source, block["id"], position, text, roles.get(role_id), kind,
                basis="unresolved" if unresolved else "user",
                reason="No speaker is named in the passage." if unresolved else "Hand-authored demonstration annotation.",
                candidates=["mara_dialogue", "lena_dialogue"] if unresolved else [], review_required=unresolved))
            position += len(text)
    document = base_document(source, cast, segments, provenance={
        "method": "Hand-authored fictional example; no model requests or generated audio.",
        "source_fingerprint": fingerprint(source), "corrections": [],
    })
    document["readiness"] = readiness(document)
    assert not document["readiness"]["errors"]
    write_json(output / "analysis.json", document)
    corrections = {"source_fingerprint": fingerprint(source), "cast_fingerprint": fingerprint(cast),
                   "assignments": [{"segment_id": segments[-1]["id"], "role_id": "mara_dialogue"}]}
    write_json(output / "corrections.json", corrections)
    reviewed = apply_edits(document, corrections)
    write_json(output / "reviewed.json", reviewed)
    export_bundle(reviewed, output / "bundle")
    print(f"Created fictional EPUB, draft, corrections, reviewed analysis, and bundle in {output}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3.11 examples/build_example.py NEW_OUTPUT_DIRECTORY")
    build(sys.argv[1])
