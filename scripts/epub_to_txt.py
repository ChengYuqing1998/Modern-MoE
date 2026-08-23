"""Convert an EPUB spine to clean UTF-8 plain text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree


BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "div", "dl", "dt", "dd",
    "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
    "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
IGNORED_TAGS = {"head", "script", "style", "svg"}
EOD = "<|endoftext|>"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            self.ignored_depth += 1
        elif not self.ignored_depth and (tag in BLOCK_TAGS or tag == "br"):
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if not self.ignored_depth and (
            tag.lower() in BLOCK_TAGS or tag.lower() == "br"
        ):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in IGNORED_TAGS:
            self.ignored_depth = max(0, self.ignored_depth - 1)
        elif not self.ignored_depth and tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        text = unicodedata.normalize("NFC", "".join(self.parts))
        text = (
            text.replace("\ufeff", "")
            .replace("\u200b", "")
            .replace("\xa0", " ")
        )
        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r"[ \t\u3000]+", " ", line).strip()
            if cleaned:
                lines.append(cleaned)
            elif lines and lines[-1] != "":
                lines.append("")
        return "\n".join(lines).strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def container_rootfile(archive: zipfile.ZipFile) -> PurePosixPath:
    root = ElementTree.fromstring(
        archive.read("META-INF/container.xml")
    )
    for element in root.iter():
        if local_name(element.tag) == "rootfile":
            return PurePosixPath(element.attrib["full-path"])
    raise ValueError("EPUB container has no rootfile")


def spine_documents(
    archive: zipfile.ZipFile, opf_path: PurePosixPath
) -> tuple[list[PurePosixPath], dict[str, str]]:
    root = ElementTree.fromstring(archive.read(str(opf_path)))
    manifest: dict[str, tuple[str, str]] = {}
    spine_ids: list[str] = []
    excluded_hrefs: set[str] = set()
    metadata: dict[str, str] = {}
    for element in root.iter():
        name = local_name(element.tag)
        if name == "item":
            manifest[element.attrib["id"]] = (
                element.attrib["href"],
                element.attrib.get("media-type", ""),
            )
        elif name == "itemref":
            spine_ids.append(element.attrib["idref"])
        elif name == "reference" and element.attrib.get("type") == "toc":
            excluded_hrefs.add(element.attrib.get("href", "").split("#")[0])
        elif name in {"title", "creator", "language", "publisher", "date"}:
            value = (element.text or "").strip()
            if value and name not in metadata:
                metadata[name] = value

    base = opf_path.parent
    documents = []
    for item_id in spine_ids:
        href, media_type = manifest[item_id]
        href_without_fragment = href.split("#")[0]
        if (
            media_type in {"application/xhtml+xml", "text/html"}
            and href_without_fragment not in excluded_hrefs
        ):
            documents.append(base / href_without_fragment)
    return documents, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(".txt")
    chapters: list[str] = []
    with zipfile.ZipFile(args.input) as archive:
        opf_path = container_rootfile(archive)
        document_paths, metadata = spine_documents(archive, opf_path)
        for path in document_paths:
            payload = archive.read(str(path))
            html = payload.decode("utf-8-sig", errors="strict")
            extractor = TextExtractor()
            extractor.feed(html)
            chapter = extractor.text()
            if chapter:
                chapters.append(chapter)

    # Keep one EPUB spine item as one training document. The tokenizer reads
    # this sentinel as a boundary and emits its configured EOS token; it is
    # not intended to become ordinary model-visible prose.
    text = "".join(
        f"{chapter.strip()}\n{EOD}\n" for chapter in chapters
    )
    output.write_text(text, encoding="utf-8", newline="\n")
    payload = output.read_bytes()
    report = {
        "input": str(args.input),
        "output": str(output),
        "encoding": "UTF-8 without BOM",
        "metadata": metadata,
        "spine_documents": len(document_paths),
        "written_documents": len(chapters),
        "document_separator": EOD,
        "document_separators": text.count(EOD),
        "characters": len(text),
        "bytes": len(payload),
        "replacement_characters": text.count("\ufffd"),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    output.with_suffix(".txt.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
