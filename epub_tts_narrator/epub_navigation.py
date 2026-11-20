"""Read chapter labels from EPUB navigation without extracting navigation text."""

from __future__ import annotations

from urllib.parse import unquote
import xml.etree.ElementTree as ET

from .epub import _normalize, _resolve_href, _tag


_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"


def _collect_entries(base: str, links, entries: list[tuple[str, str | None, str]]) -> None:
    """Append (resource, fragment or None, label) for each labelled local link, in order."""
    for href, text in links:
        label = _normalize(text)
        target, _, fragment = href.partition("#")
        if not label or not target:
            continue
        path = _resolve_href(base, target, remote_ok=True)
        if path is not None:
            entries.append((path, unquote(fragment) or None, label))


def _collect_labels(base: str, links, labels: dict[str, str]) -> None:
    """Keep the first nonempty label for each local archive resource."""
    for href, text in links:
        label = _normalize(text)
        target = href.split("#", 1)[0]
        if not label or not target:
            continue
        path = _resolve_href(base, target, remote_ok=True)
        if path is not None:
            labels.setdefault(path, label)


def _ncx_links(items: dict[str, dict], spine: ET.Element, xml) -> list[tuple[str, list]]:
    """(NCX path, [(src, label)]) for each navMap of the spine-selected NCX only, in document order."""
    ncx_id = spine.get("toc")
    if not ncx_id:
        return []
    if ncx_id not in items:
        raise ValueError(f"OPF spine toc references an unknown manifest item: {ncx_id}")
    path = items[ncx_id]["path"]
    if path is None:
        return []
    root = xml(path)
    maps = []
    for nav_map in (node for node in root.iter() if _tag(node) == "navmap"):
        links = []
        for point in (node for node in nav_map.iter() if _tag(node) == "navpoint"):
            nav_label = next((node for node in point if _tag(node) == "navlabel"), None)
            content = next((node for node in point if _tag(node) == "content"), None)
            text = next((node for node in nav_label.iter() if _tag(node) == "text"), None) if nav_label is not None else None
            if content is not None and text is not None:
                links.append((content.get("src", ""), "".join(text.itertext())))
        maps.append((path, links))
    return maps


def _epub3_links(items: dict[str, dict], xml) -> list[tuple[str, list]]:
    """(nav path, [(href, label)]) for anchors belonging to epub:type=toc, not other navigation lists."""
    navs = []
    for item in items.values():
        path = item["path"]
        if "nav" not in item["properties"] or path is None:
            continue
        root = xml(path)
        for nav in (node for node in root.iter()
                    if _tag(node) == "nav" and "toc" in node.get(_EPUB_TYPE, "").split()):
            navs.append((path, [(node.get("href", ""), "".join(node.itertext()))
                                for node in nav.iter() if _tag(node) == "a"]))
    return navs


def _ncx_labels(items: dict[str, dict], spine: ET.Element, xml) -> dict[str, str]:
    """Read navPoint labels from the spine-selected NCX navMap only."""
    labels: dict[str, str] = {}
    for path, links in _ncx_links(items, spine, xml):
        _collect_labels(path, links, labels)
    return labels


def _epub3_labels(items: dict[str, dict], xml) -> dict[str, str]:
    """Read anchors belonging to epub:type=toc, not other navigation lists."""
    labels: dict[str, str] = {}
    for path, links in _epub3_links(items, xml):
        _collect_labels(path, links, labels)
    return labels


def navigation_titles(items: dict[str, dict], spine: ET.Element, xml) -> dict[str, str]:
    """Return resource titles, preferring EPUB3 toc entries and filling from NCX."""
    labels = _ncx_labels(items, spine, xml)
    labels.update(_epub3_labels(items, xml))
    return labels


def navigation_entries(items: dict[str, dict], spine: ET.Element, xml) -> dict[str, list[tuple[str | None, str]]]:
    """Each resource's table-of-contents entries in reading order: [(fragment or None, label)].

    The EPUB3 toc is used when it lists any entry, else the NCX navMap, so one
    book's chapters never mix two tables of contents.
    """
    entries: list[tuple[str, str | None, str]] = []
    for path, links in _epub3_links(items, xml):
        _collect_entries(path, links, entries)
    if not entries:
        for path, links in _ncx_links(items, spine, xml):
            _collect_entries(path, links, entries)
    by_path: dict[str, list[tuple[str | None, str]]] = {}
    for path, fragment, label in entries:
        by_path.setdefault(path, []).append((fragment, label))
    return by_path
