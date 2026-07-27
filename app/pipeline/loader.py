"""PDF loading and text cleaning -- step 1 of the pipeline.

Level 2 of the deck lists "Embedding raw junk" as one of four ways people break
RAG: navigation menus, cookie banners and page footers embed perfectly well and
then pollute every result set. So cleaning here is a visible feature, not a
hidden detail -- every removal is counted and reported to the UI.

Two entry points, deliberately separated:

  clean_pages()  pure, no I/O -- the interesting logic, exactly testable
  load_pdf()     extracts with LangChain's PyPDFLoader, then delegates
"""

from __future__ import annotations

import bisect
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

# Zero-width and byte-order marks. Documents exported from Google Docs are
# littered with U+200B: invisible on screen, yet it burns tokens and can split a
# word in the middle as far as a tokeniser is concerned.
_INVISIBLE_RE = re.compile("[​‌‍﻿]")

# A table-of-contents line: text, a run of dots or spaces, then a trailing page
# number. "Companies Management .......... 16"
_TOC_LINE_RE = re.compile(r"^\s*\S.*?[\s.…]{2,}\d{1,4}\s*$")

# Lines this long are prose, not running headers, whatever their frequency.
_MAX_BOILERPLATE_LEN = 120

# Below this many pages, "appears on most pages" is not evidence of boilerplate.
_MIN_PAGES_FOR_BOILERPLATE = 4

# Fraction of pages a line must appear on before it counts as boilerplate.
_BOILERPLATE_PAGE_FRACTION = 0.5

# Only strip TOC lines from the front, where a TOC actually lives. A mid-document
# line that happens to end in a number is probably data.
_TOC_LEADING_FRACTION = 0.15


class EmptyDocumentError(ValueError):
    """Raised when a PDF yields no usable text.

    Almost always a scanned document: the pages are images, so there is no text
    layer to extract. Worth failing loudly -- it is a real RAG gotcha, and a
    silently empty collection is far more confusing than an error.
    """


@dataclass
class CleanResult:
    pages: list[str]
    boilerplate_lines_removed: int
    invisible_chars_removed: int


@dataclass
class LoadResult:
    text: str
    page_count: int
    char_count: int
    pages_without_text: int
    boilerplate_lines_removed: int
    invisible_chars_removed: int
    doc_id: str
    # (start_char, page_number) pairs, ascending by start_char.
    page_offsets: list[tuple[int, int]]

    def page_for_offset(self, offset: int) -> int:
        """Which source page does this character offset fall on?

        Cleaning concatenates every page into one string, so a chunk can straddle
        a page boundary. Chunks are attributed to the page containing their
        *start* offset: that keeps splitters free to cross boundaries (recursive
        and semantic must) while still giving every chunk something citable.
        """
        if not self.page_offsets:
            return 1
        starts = [start for start, _ in self.page_offsets]
        index = max(bisect.bisect_right(starts, offset) - 1, 0)
        return self.page_offsets[index][1]


def _find_boilerplate(pages: list[str]) -> set[str]:
    """Lines recurring across most pages -- running headers and footers."""
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        # Count each distinct line once per page, so a line repeated many times
        # on a single page does not masquerade as a running header.
        counts.update({
            line.strip()
            for line in page.splitlines()
            if line.strip() and len(line.strip()) <= _MAX_BOILERPLATE_LEN
        })

    threshold = len(pages) * _BOILERPLATE_PAGE_FRACTION
    return {line for line, count in counts.items() if count >= threshold}


def _squash_whitespace(text: str) -> str:
    """Collapse run-on spaces and blank lines left behind by extraction."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def clean_pages(raw_pages: list[str]) -> CleanResult:
    """Strip the parts of a document that should never reach an embedding model.

    Returns cleaned pages plus counts, because the UI displays them: watching
    "removed 412 boilerplate lines, stripped 1,847 invisible characters" appear
    on screen is what turns the deck's gotcha into something the room has seen.
    """
    invisible_removed = sum(len(_INVISIBLE_RE.findall(page)) for page in raw_pages)
    pages = [_INVISIBLE_RE.sub("", page) for page in raw_pages]

    boilerplate = _find_boilerplate(pages)
    toc_cutoff = max(1, int(len(pages) * _TOC_LEADING_FRACTION))

    lines_removed = 0
    kept_pages: list[str] = []
    for page_index, page in enumerate(pages):
        kept_lines: list[str] = []
        for line in page.splitlines():
            stripped = line.strip()
            if stripped and stripped in boilerplate:
                lines_removed += 1
                continue
            if page_index < toc_cutoff and _TOC_LINE_RE.match(line):
                lines_removed += 1
                continue
            kept_lines.append(line)
        kept_pages.append(_squash_whitespace("\n".join(kept_lines)))

    return CleanResult(
        pages=kept_pages,
        boilerplate_lines_removed=lines_removed,
        invisible_chars_removed=invisible_removed,
    )


def load_pdf(path: str | Path) -> LoadResult:
    """Extract and clean a PDF, returning text plus everything the UI reports."""
    documents = PyPDFLoader(str(path)).load()
    raw_pages = [doc.page_content or "" for doc in documents]

    cleaned = clean_pages(raw_pages)
    pages_without_text = sum(1 for page in cleaned.pages if not page.strip())

    # Assemble one string, recording where each page starts so chunks produced
    # downstream can be attributed back to a page number.
    separator = "\n\n"
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_index, page in enumerate(cleaned.pages):
        offsets.append((cursor, page_index + 1))
        parts.append(page)
        cursor += len(page) + len(separator)

    text = separator.join(parts).strip()
    if not text:
        raise EmptyDocumentError(
            f"0 characters extracted from {Path(path).name}. This looks like a "
            "scanned PDF -- the pages are images, so there is no text layer. RAG "
            "needs extractable text; run OCR over the document first."
        )

    return LoadResult(
        text=text,
        page_count=len(raw_pages),
        char_count=len(text),
        pages_without_text=pages_without_text,
        boilerplate_lines_removed=cleaned.boilerplate_lines_removed,
        invisible_chars_removed=cleaned.invisible_chars_removed,
        doc_id=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        page_offsets=offsets,
    )
