"""Test fixtures.

No PDF is committed to this repository -- the presenter's document is internal.
Every PDF the suite uses is generated here at runtime with reportlab, which also
makes these fixtures executable documentation of what "dirty input" means.
"""

from __future__ import annotations

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

FOOTER = "Internal Handbook -- Confidential"

TOC_ENTRIES = [
    "Introduction .......................... 1",
    "Getting Started ....................... 2",
    "Companies Management .................. 3",
    "People Management ..................... 4",
    "Reporting ............................. 5",
]

SECTIONS = [
    ("1. Introduction", [
        "This handbook describes the platform and its day to day operation.",
        "Each section covers one area of the product in practical terms.",
    ]),
    ("2. Companies Management", [
        "Companies are the top level record in the system.",
        "Every company holds an address, operating hours and custom fields.",
    ]),
    ("3. People Management", [
        "People belong to one or more companies as contacts.",
        "A person record holds an address, notes and custom fields.",
    ]),
]


def _draw_lines(pdf: canvas.Canvas, lines: list[str], start_y: int = 750) -> None:
    y = start_y
    for line in lines:
        pdf.drawString(72, y, line)
        y -= 16


@pytest.fixture(scope="session")
def structured_pdf(tmp_path_factory) -> str:
    """A PDF with a TOC, numbered headings, a repeated footer, and a blank page.

    Deliberately messy, mirroring what real exported documents look like.
    """
    path = tmp_path_factory.mktemp("pdfs") / "structured.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)

    # Page 1: title + table of contents (retrieval poison; the loader strips it)
    _draw_lines(pdf, ["Sample Handbook", ""] + TOC_ENTRIES)
    pdf.drawString(72, 40, FOOTER)
    pdf.showPage()

    # Pages 2-4: content, each carrying the same footer
    for heading, body in SECTIONS:
        _draw_lines(pdf, [heading, ""] + body)
        pdf.drawString(72, 40, FOOTER)
        pdf.showPage()

    # Final page: intentionally blank -- no text layer at all
    pdf.showPage()
    pdf.save()
    return str(path)


@pytest.fixture(scope="session")
def flat_pdf(tmp_path_factory) -> str:
    """A PDF with no headings whatsoever.

    Used to prove structure-aware chunking degrades to recursive rather than
    returning one section containing the whole document.
    """
    path = tmp_path_factory.mktemp("pdfs") / "flat.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    sentence = "The quick brown fox jumps over the lazy dog and keeps running."
    _draw_lines(pdf, [sentence] * 30)
    pdf.save()
    return str(path)


@pytest.fixture
def toc_near_miss_pages() -> list[str]:
    """A leading page mixing a genuine TOC entry with two lines that only look
    like one.

    "All rights reserved.  2024" and "Room number:   204" both end in a run of
    1-4 digits preceded by nothing more than a couple of whitespace
    characters -- exactly what a too-loose TOC regex keys on. Neither is a
    table-of-contents entry; a real one needs an actual leader (a genuine
    dot-leader, or a much wider gap than two characters).
    """
    return [
        "Sample Handbook\n"
        "Companies Management .................. 3\n"
        "All rights reserved.  2024\n"
        "Room number:   204"
    ]


@pytest.fixture
def mid_page_boilerplate_pages() -> list[str]:
    """Four pages sharing a repeated footer *and* a repeated mid-page line.

    Frequency alone cannot tell these apart -- "Notes:" and the footer both
    recur on every page. Position can: a running header or footer lives within
    a few lines of the top or bottom of the page, while "Notes:" sits in the
    middle of the body, where a recurring subheading or disclaimer would.
    """
    return [
        f"Header info page {n}\n"
        f"More header page {n}\n"
        f"Even more header page {n}\n"
        f"Some unique body about page {n}.\n"
        "Notes:\n"
        f"Another unique body line about page {n}.\n"
        f"Body filler A page {n}\n"
        f"Body filler B page {n}\n"
        f"Body filler C page {n}\n"
        f"Footer line one page {n}\n"
        f"Footer line two page {n}\n"
        "Repeated Footer Text"
        for n in range(1, 5)
    ]


@pytest.fixture
def short_mid_page_boilerplate_pages() -> list[str]:
    """Four *short* (5-line) pages sharing a repeated footer and a repeated
    mid-page "Notes:" line -- the same shape as mid_page_boilerplate_pages,
    but at the length where naive top-3/bottom-3 edge windows overlap and
    swallow the whole page. Regression fixture for that overlap bug.
    """
    return [
        f"Heading page {n}\n"
        "\n"
        f"Some unique body about page {n}.\n"
        "Notes:\n"
        "Repeated Footer Text"
        for n in range(1, 5)
    ]


@pytest.fixture
def dirty_pages() -> list[str]:
    """Raw page text as an extractor hands it over, including U+200B.

    Exactly three zero-width spaces, marked below, so the count assertion in
    test_loader.py is exact.
    """
    return [
        "Sample Handbook\n"
        "Introduction .......................... 1\n"
        "Getting Started ....................... 2\n"
        "Companies Management .................. 3\n" + FOOTER,
        "1.​ Introduction\n"                       # zero-width #1
        "This handbook describes    the platform.\n"
        "\n\n\n"
        "It covers day to day operation.\n" + FOOTER,
        "2.​ Companies​ Management\n"          # zero-width #2 and #3
        "Companies are the top level record.\n" + FOOTER,
        "3. People Management\nPeople belong to companies.\n" + FOOTER,
        "",
    ]
