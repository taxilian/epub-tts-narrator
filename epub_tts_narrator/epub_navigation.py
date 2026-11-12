"""Read chapter labels from EPUB navigation without extracting navigation text."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .epub import _normalize, _resolve_href, _tag


_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"


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


def _ncx_labels(items: dict[str, dict], spine: ET.Element, xml) -> dict[str, str]:
    """Read navPoint labels from the spine-selected NCX navMap only."""
    ncx_id = spine.get("toc")
    if not ncx_id:
        return {}
    if ncx_id not in items:
        raise ValueError(f"OPF spine toc references an unknown manifest item: {ncx_id}")
    path = items[ncx_id]["path"]
    if path is None:
        return {}
    root = xml(path)
    labels: dict[str, str] = {}
    for nav_map in (node for node in root.iter() if _tag(node) == "navmap"):
        links = []
        for point in (node for node in nav_map.iter() if _tag(node) == "navpoint"):
            nav_label = next((node for node in point if _tag(node) == "navlabel"), None)
            content = next((node for node in point if _tag(node) == "content"), None)
            text = next((node for node in nav_label.iter() if _tag(node) == "text"), None) if nav_label is not None else None
            if content is not None and text is not None:
                links.append((content.get("src", ""), "".join(text.itertext())))
        _collect_labels(path, links, labels)
    return labels


def _epub3_labels(items: dict[str, dict], xml) -> dict[str, str]:
    """Read anchors belonging to epub:type=toc, not other navigation lists."""
    labels: dict[str, str] = {}
    for item in items.values():
        path = item["path"]
        if "nav" not in item["properties"] or path is None:
            continue
        root = xml(path)
        for nav in (node for node in root.iter()
                    if _tag(node) == "nav" and "toc" in node.get(_EPUB_TYPE, "").split()):
            links = ((node.get("href", ""), "".join(node.itertext()))
                     for node in nav.iter() if _tag(node) == "a")
            _collect_labels(path, links, labels)
    return labels


def navigation_titles(items: dict[str, dict], spine: ET.Element, xml) -> dict[str, str]:
    """Return resource titles, preferring EPUB3 toc entries and filling from NCX."""
    labels = _ncx_labels(items, spine, xml)
    labels.update(_epub3_labels(items, xml))
    return labels
