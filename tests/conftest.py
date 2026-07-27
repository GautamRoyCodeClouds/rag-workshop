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
