"""One input-only DOCX projection; offsets are Unicode code points, not UTF-16."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

VERSION = "sc35-corrupted-text-v1"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
NS = {"w": W[1:-1], "m": M[1:-1]}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def intervals(values):
    """Normalize unions; overlapping predictions cannot multiply character credit."""
    merged = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def span_dicts(values):
    return [{"start": start, "end": end} for start, end in intervals(values)]


class Projection:
    def __init__(self, path):
        self.path = Path(path)
        self.source_sha256 = sha_bytes(self.path.read_bytes())
        with ZipFile(self.path) as package:
            self.root = etree.fromstring(
                package.read("word/document.xml"),
                etree.XMLParser(resolve_entities=False, no_network=True),
            )
        self.body = self.root.find("w:body", NS)
        if self.root.tag != W + "document" or self.body is None:
            raise ValueError("Not a Word document body")
        self.tree = self.root.getroottree()
        self.ns = {**{k: v for k, v in self.root.nsmap.items() if k}, **NS}
        self.paragraphs = self.root.findall(".//w:p", NS)
        self.direct = self.body.findall("w:p", NS)
        self.records, self.by_element = [], {}
        chunks, cursor = [], 0
        for index, paragraph in enumerate(self.paragraphs):
            parts = []
            for element in paragraph.iter():
                ancestors = list(element.iterancestors())
                if any(a.tag in {W + "del", W + "moveFrom"} for a in ancestors):
                    continue
                if next((a for a in ancestors if a.tag == W + "p"), None) is not paragraph:
                    continue
                if element.tag in {W + "t", M + "t"}:
                    parts.append(element.text or "")
                elif element.tag == W + "tab":
                    parts.append("\t")
                elif element.tag == W + "cr":
                    parts.append("\n")
                elif element.tag == W + "br":
                    parts.append("\f" if element.get(W + "type") == "page" else "\n")
            text = "".join(parts)
            # Every paragraph has a separator: empty/non-text objects have an anchor.
            record = {
                "paragraph_index": index,
                "xpath": self.tree.getpath(paragraph),
                "start": cursor,
                "end": cursor + len(text),
                "anchor_end": cursor + len(text) + 1,
                "text": text,
            }
            self.records.append(record)
            self.by_element[paragraph] = record
            chunks.append(text + "\n")
            cursor = record["anchor_end"]
        if not self.records:
            raise ValueError("No paragraph objects; no canonical span anchors")
        self.text = "".join(chunks)
        self.text_sha256 = sha_bytes(self.text.encode("utf-8"))

    def paragraph_span(self, paragraph):
        record = self.by_element[paragraph]
        return (record["start"], max(record["end"], record["start"] + 1))

    def node_spans(self, node):
        if not isinstance(node, etree._Element):
            raise ValueError("Locator must select XML objects")
        if node in self.by_element:
            return [self.paragraph_span(node)]
        parent = next((p for p in node.iterancestors() if p in self.by_element), None)
        if parent is not None:
            return [self.paragraph_span(parent)]
        descendants = [p for p in node.iter(W + "p") if p in self.by_element]
        if descendants:
            return [self.paragraph_span(p) for p in descendants]
        if node.tag == W + "sectPr" or any(p.tag == W + "sectPr" for p in node.iterancestors()):
            # Final section settings have no text: use the final paragraph separator.
            return [(len(self.text) - 1, len(self.text))]
        raise ValueError("Object has no supported canonical anchor")

    def resolve(self, locator):
        if not isinstance(locator, dict):
            raise ValueError("Missing locator")
        path = locator.get("xpath")
        if path and path != "/w:document/w:body":
            selected = self.root.xpath(path, namespaces=self.ns)
            if not selected:
                raise ValueError("XPath selects no objects: " + path)
            return span_dicts([span for node in selected for span in self.node_spans(node)])
        if locator.get("bookmark"):
            selected = self.root.xpath(
                ".//w:bookmarkStart[@w:name=$name]", namespaces=NS, name=locator["bookmark"]
            )
            if len(selected) != 1:
                raise ValueError("Bookmark is absent or ambiguous")
            paragraph = next((p for p in selected[0].iterancestors() if p in self.by_element), None)
            if paragraph is None:
                raise ValueError("Bookmark has no paragraph")
            index = self.by_element[paragraph]["paragraph_index"]
            count = locator.get("region_paragraph_count", 1)
            if type(count) is not int or count < 1 or index + count > len(self.paragraphs):
                raise ValueError("Invalid bookmark region")
            return span_dicts(
                [self.paragraph_span(p) for p in self.paragraphs[index : index + count]]
            )
        index = locator.get("paragraph_index")
        if type(index) is int:
            scope = locator.get("paragraph_index_scope")
            if scope not in {"all_document_paragraphs", "direct_body_paragraphs"}:
                raise ValueError("paragraph_index requires an explicit index scope")
            values = self.paragraphs if scope == "all_document_paragraphs" else self.direct
            if not 0 <= index < len(values):
                raise ValueError("Paragraph index out of bounds")
            return span_dicts([self.paragraph_span(values[index])])
        quote = locator.get("text")
        if isinstance(quote, str) and quote:
            first = self.text.find(quote)
            if first < 0 or self.text.find(quote, first + 1) >= 0:
                raise ValueError("Quote is absent or repeated; an address is required")
            return [{"start": first, "end": first + len(quote)}]
        raise ValueError("No unambiguous location; never substitute the whole document")

    def property_anchors(self):
        return span_dicts(
            [
                span
                for node in self.root.findall(".//w:sectPr", NS)
                for span in self.node_spans(node)
            ]
        )

    def to_dict(self):
        return {
            "schema_version": "sc35-canonical-document-v1",
            "extraction_version": VERSION,
            "document_sha256": self.source_sha256,
            "text_sha256": self.text_sha256,
            "offset_unit": "unicode_codepoint",
            "interval": "half_open",
            "part": "word/document.xml",
            "text": self.text,
            "paragraphs": self.records,
            "limits": [
                "XML paragraph order, not rendered page order.",
                "Main document only; headers, footers and images have no text projection.",
                "Empty paragraphs and non-text properties use declared object anchors.",
            ],
        }
