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

# A table-of-contents line: text, then a genuine leader, then a trailing page
# number. "Companies Management .......... 16"
#
# The leader must be a real one -- either a run of 3+ dots (optionally spaced
# out, "Introduction . . . . . 12") or a single ellipsis character, or, for
# leaderless columnar layouts, a substantially wide whitespace gap (6+
# characters). Two whitespace-or-dot characters is NOT enough: ordinary prose
# routinely ends a line in a sentence-final period plus a couple of spaces
# plus a number ("All rights reserved.  2024") or a label with a couple of
# aligning spaces ("Room number:   204"), and neither is a TOC entry.
_TOC_LINE_RE = re.compile(r"^\s*\S.*?(?:(?:\.[ \t]*){3,}|…+|[ \t]{6,})\d{1,4}\s*$")

# Lines this long are prose, not running headers, whatever their frequency.
_MAX_BOILERPLATE_LEN = 120

# Below this many pages, "appears on most pages" is not evidence of boilerplate.
_MIN_PAGES_FOR_BOILERPLATE = 4

# Fraction of pages a line must appear on before it counts as boilerplate.
_BOILERPLATE_PAGE_FRACTION = 0.5

# Running headers and footers live within this many lines of the top or bottom
# of a page. A line recurring across most pages but sitting in the middle of
# the body (a repeated "Notes:" subheading, a recurring disclaimer) is content,
# not boilerplate, however often it repeats -- frequency alone cannot tell the
# two apart, only position can.
_BOILERPLATE_EDGE_LINES = 3

# The middle region a page's top/bottom windows must leave untouched needs to
# be at least this many lines before those windows are trusted to be that
# wide. A naive top-N/bottom-N split doesn't scale down: on a page with only
# 2*N or fewer lines, "top 3" and "bottom 3" overlap and cover the *entire*
# page, leaving no middle at all -- which silently resurrects the exact bug
# the positional constraint above was meant to fix (a genuine mid-page line
# like "Notes:" getting caught in one of the windows and stripped as if it
# were a header/footer). So the edge window shrinks on short pages to
# guarantee this many lines survive in the middle before it is allowed to grow
# toward _BOILERPLATE_EDGE_LINES.
#
# Below 2 lines there is no "top" and "bottom" to distinguish in the first
# place -- both windows would collapse onto the same single line -- so that
# case is handled separately (see _find_boilerplate): the positional filter
# does not apply to it at all, rather than applying it degenerately.
_BOILERPLATE_MIN_MIDDLE_LINES = 3

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
        return page_for_offset(self.page_offsets, offset)


def page_for_offset(page_offsets: list[tuple[int, int]], offset: int) -> int:
    """Which page does a character offset fall on, given (start, page) pairs?

    Module-level, not just a LoadResult method: SessionState (app/session.py)
    needs the identical lookup after a JSON round-trip, once the LoadResult
    object itself no longer exists. Character-offset attribution has
    repeatedly produced bugs in this codebase, so this gets one home rather
    than two copies free to drift apart.
    """
    if not page_offsets:
        return 1
    starts = [start for start, _ in page_offsets]
    # bisect_right, not bisect_left: an offset equal to a page's recorded
    # start must resolve to *that* page, not the one before it. bisect_left
    # would put the insertion point before equal entries, and subtracting 1
    # would then land one page too early for any offset sitting exactly on
    # a boundary.
    index = max(bisect.bisect_right(starts, offset) - 1, 0)
    return page_offsets[index][1]


def _find_boilerplate(pages: list[str]) -> set[str]:
    """Lines recurring across most pages, near the top or bottom -- running
    headers and footers.

    Frequency alone is not enough: a legitimate line can recur on every page
    just as easily as a footer can (a repeated subheading, a standard
    disclaimer paragraph). What actually marks a line as a running header or
    footer is *where* it sits -- within a few lines of the top or bottom of the
    page -- so only those lines are eligible to count, however often a
    mid-page line repeats.
    """
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = page.splitlines()
        if len(lines) < 2:
            # A single line has no "top" separate from its "bottom" -- both
            # windows would just be that same line, so the positional check
            # is meaningless here rather than merely small. Exempting it
            # entirely means a recurring one-line page (a slide with just a
            # title, say) can never have its only line silently deleted,
            # which a window of size >=1 would otherwise allow.
            edge_count = 0
        else:
            # Shrink the edge window so top-N and bottom-N leave at least
            # _BOILERPLATE_MIN_MIDDLE_LINES of genuine middle between them --
            # see that constant's comment for why. The window never shrinks
            # below 1, though: with >=2 lines a real top and bottom always
            # exist, and pinning the floor at 1 (rather than letting it hit 0)
            # keeps the long-standing, tested behavior on very short pages
            # where a window of exactly 1 still spans the whole page (e.g. a
            # 2-line page) -- there is no middle to protect there either way,
            # so shrinking further would only stop catching a real running
            # header/footer without protecting anything in return.
            max_edge_for_middle = (len(lines) - _BOILERPLATE_MIN_MIDDLE_LINES) // 2
            edge_count = min(_BOILERPLATE_EDGE_LINES, max(1, max_edge_for_middle))
        edge_indexes = set(range(edge_count)) | set(
            range(len(lines) - edge_count, len(lines))
        )
        # Count each distinct edge line once per page, so a line repeated many
        # times on a single page does not masquerade as a running header.
        counts.update({
            lines[i].strip()
            for i in edge_indexes
            if lines[i].strip() and len(lines[i].strip()) <= _MAX_BOILERPLATE_LEN
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


def _join_pages(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate cleaned pages into one string, with a correct offset per page.

    A naive `separator.join(parts).strip()` looks harmless but is not: when a
    leading page cleans to "" (a logo-only cover, an all-boilerplate title
    page -- entirely realistic), the joined string *starts* with the separator
    itself ("\\n\\n..."), and .strip() eats it. That shifts every later page's
    real start left by the stripped amount, while offsets recorded during the
    join assumed the separator was still there -- a silent, systematic drift
    that only gets worse the more leading pages are empty. One empty leading
    page is already enough to make every citation in the document point one
    page too early.

    Rather than assume the drift is zero, this measures exactly how much
    leading whitespace .strip() removes and subtracts that same amount from
    every recorded offset, so the two are correct by construction instead of
    by coincidence. Trailing whitespace needs no such correction: stripping the
    *end* of the string cannot move anything that comes before it.
    """
    separator = "\n\n"
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for page_index, page in enumerate(pages):
        offsets.append((cursor, page_index + 1))
        parts.append(page)
        cursor += len(page) + len(separator)

    joined = separator.join(parts)
    text = joined.strip()
    leading_trim = len(joined) - len(joined.lstrip())
    offsets = [(max(0, start - leading_trim), page_num) for start, page_num in offsets]
    return text, offsets


def load_pdf(path: str | Path) -> LoadResult:
    """Extract and clean a PDF, returning text plus everything the UI reports."""
    documents = PyPDFLoader(str(path)).load()
    raw_pages = [doc.page_content or "" for doc in documents]

    cleaned = clean_pages(raw_pages)
    pages_without_text = sum(1 for page in cleaned.pages if not page.strip())

    # Assemble one string, recording where each page starts so chunks produced
    # downstream can be attributed back to a page number.
    text, offsets = _join_pages(cleaned.pages)
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
