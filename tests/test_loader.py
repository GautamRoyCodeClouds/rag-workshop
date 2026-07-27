"""Loader tests.

clean_pages is tested against strings so the zero-width assertions are exact.
load_pdf is tested against a generated PDF for file-level properties.
Page-offset attribution is tested by driving clean_pages() + the offset-join
logic directly on strings (see TestPageForOffset) rather than through a
generated PDF, so the exact character positions in every assertion are known
rather than dependent on reportlab's output.
"""

import pytest

from app.pipeline.loader import (
    EmptyDocumentError,
    LoadResult,
    _join_pages,
    clean_pages,
    load_pdf,
)


class TestCleanPages:
    def test_strips_zero_width_characters(self, dirty_pages):
        result = clean_pages(dirty_pages)
        assert result.invisible_chars_removed == 3
        assert "​" not in "".join(result.pages)

    def test_removes_table_of_contents_lines(self, dirty_pages):
        joined = "\n".join(clean_pages(dirty_pages).pages)
        assert "Companies Management .................. 3" not in joined
        # ...but the real heading of the same name survives
        assert "Companies Management" in joined

    def test_near_miss_toc_lines_survive(self, toc_near_miss_pages):
        # A real dot-leader TOC line is removed, but a line that merely ends in
        # a couple of whitespace characters and a number is not a TOC entry --
        # the old regex ([\s.…]{2,}) could not tell these apart and deleted
        # legitimate content. Requiring an actual leader run (3+ dots, or a
        # substantially wider whitespace gap) fixes that.
        joined = "\n".join(clean_pages(toc_near_miss_pages).pages)
        assert "Companies Management .................. 3" not in joined
        # _squash_whitespace legitimately collapses the runs of spaces here
        # down to one -- that is unrelated to TOC stripping -- so check for the
        # squashed form rather than the original spacing.
        assert "All rights reserved. 2024" in joined
        assert "Room number: 204" in joined

    def test_removes_the_repeated_footer(self, dirty_pages):
        result = clean_pages(dirty_pages)
        assert "Confidential" not in "\n".join(result.pages)
        assert result.boilerplate_lines_removed >= 4

    def test_mid_page_recurring_line_survives_but_repeated_footer_is_removed(
        self, mid_page_boilerplate_pages
    ):
        # Frequency alone cannot distinguish a running footer from a recurring
        # mid-content line (a repeated "Notes:" subheading, a disclaimer) --
        # both appear on every page. Only position can: running headers and
        # footers live at the top or bottom of a page, not in the middle.
        joined = "\n".join(clean_pages(mid_page_boilerplate_pages).pages)
        assert "Notes:" in joined
        assert "Repeated Footer Text" not in joined

    def test_squashes_whitespace_runs(self, dirty_pages):
        joined = "\n".join(clean_pages(dirty_pages).pages)
        assert "    " not in joined
        assert "\n\n\n" not in joined

    def test_keeps_body_text(self, dirty_pages):
        assert "top level record" in "\n".join(clean_pages(dirty_pages).pages)

    def test_short_documents_have_no_boilerplate(self):
        # With two pages, "appears on most pages" is not evidence of a running
        # header -- it may simply be a two-page document that repeats a line.
        result = clean_pages(["Alpha line\nBody one", "Alpha line\nBody two"])
        assert result.boilerplate_lines_removed == 0

    def test_three_pages_still_suppresses_boilerplate_detection(self):
        # One page short of _MIN_PAGES_FOR_BOILERPLATE (4): still not enough
        # evidence, even though the line recurs on every page present.
        pages = [
            "Alpha line\nBody one",
            "Alpha line\nBody two",
            "Alpha line\nBody three",
        ]
        result = clean_pages(pages)
        assert result.boilerplate_lines_removed == 0

    def test_four_pages_activates_boilerplate_detection(self):
        # Exactly at _MIN_PAGES_FOR_BOILERPLATE: now the same repeated line is
        # evidence of a running header. A mutation of 4 to 3 would make the
        # test above fail; a mutation of 4 to 6 would make this one fail.
        pages = [
            "Alpha line\nBody one",
            "Alpha line\nBody two",
            "Alpha line\nBody three",
            "Alpha line\nBody four",
        ]
        result = clean_pages(pages)
        assert result.boilerplate_lines_removed == 4

    def test_page_count_is_preserved(self, dirty_pages):
        assert len(clean_pages(dirty_pages).pages) == len(dirty_pages)


class TestLoadPdf:
    def test_reports_page_count_and_blank_pages(self, structured_pdf):
        result = load_pdf(structured_pdf)
        assert result.page_count == 5
        assert result.pages_without_text == 1

    def test_doc_id_is_stable_across_identical_loads(self, structured_pdf):
        assert load_pdf(structured_pdf).doc_id == load_pdf(structured_pdf).doc_id

    def test_doc_id_differs_between_documents(self, structured_pdf, flat_pdf):
        assert load_pdf(structured_pdf).doc_id != load_pdf(flat_pdf).doc_id

    def test_char_count_matches_text_length(self, structured_pdf):
        result = load_pdf(structured_pdf)
        assert result.char_count == len(result.text)

    def test_page_attribution_starts_at_one(self, structured_pdf):
        assert load_pdf(structured_pdf).page_for_offset(0) == 1

    def test_offset_before_the_first_page_clamps(self, structured_pdf):
        assert load_pdf(structured_pdf).page_for_offset(-5) == 1

    def test_rejects_a_document_with_no_text_layer(self, tmp_path):
        # A PDF with pages but no extractable text is the scanned-document case.
        # It must fail loudly, not silently produce an empty collection.
        from reportlab.pdfgen import canvas

        path = tmp_path / "scanned.pdf"
        pdf = canvas.Canvas(str(path))
        pdf.showPage()
        pdf.save()

        with pytest.raises(EmptyDocumentError, match="text layer"):
            load_pdf(path)


class TestPageForOffset:
    """page_for_offset(), exercised directly against strings.

    Each case is built by running clean_pages() and the same offset-join logic
    load_pdf() uses (_join_pages), then constructing a LoadResult by hand. That
    keeps every assertion exact -- these do not depend on reportlab's PDF
    output or pypdf's extraction of it, only on our own code.
    """

    @staticmethod
    def _load_result(pages: list[str]) -> LoadResult:
        cleaned = clean_pages(pages)
        text, offsets = _join_pages(cleaned.pages)
        return LoadResult(
            text=text,
            page_count=len(pages),
            char_count=len(text),
            pages_without_text=sum(1 for p in cleaned.pages if not p.strip()),
            boilerplate_lines_removed=cleaned.boilerplate_lines_removed,
            invisible_chars_removed=cleaned.invisible_chars_removed,
            doc_id="test",
            page_offsets=offsets,
        )

    def test_offset_on_an_interior_page_returns_that_page(self):
        result = self._load_result(
            ["Page one text.", "Page two text.", "Page three text."]
        )
        p2_start, page_number = result.page_offsets[1]
        assert page_number == 2
        # Comfortably inside page two's content, not on its boundary.
        assert result.page_for_offset(p2_start + 3) == 2

    def test_offset_exactly_on_a_page_start_returns_that_page_not_the_previous_one(
        self,
    ):
        # This is exactly what distinguishes bisect_right from bisect_left: an
        # offset equal to a page's recorded start belongs to that page, not the
        # page before it. bisect_left would return the *previous* page here.
        result = self._load_result(
            ["Page one text.", "Page two text.", "Page three text."]
        )
        p2_start, page_number = result.page_offsets[1]
        assert page_number == 2
        assert result.page_for_offset(p2_start) == 2

    def test_leading_empty_page_does_not_shift_later_offsets(self):
        # Regression test for the offset-drift bug: when the first page cleans
        # to "", the recorded offsets must still line up with the real text
        # after the leading whitespace that produces is stripped away.
        result = self._load_result(
            ["", "Page two content here.", "Page three content."]
        )
        p3_start, page_number = result.page_offsets[2]
        assert page_number == 3
        assert result.text[p3_start : p3_start + len("Page three content.")] == (
            "Page three content."
        )
        assert result.page_for_offset(p3_start) == 3

    def test_several_leading_empty_pages_in_a_row(self):
        result = self._load_result(["", "", "", "Real content starts here."])
        p4_start, page_number = result.page_offsets[3]
        assert page_number == 4
        assert result.text[p4_start:].startswith("Real content starts here.")
        assert result.page_for_offset(p4_start) == 4

    def test_offset_past_the_end_returns_the_last_page(self):
        result = self._load_result(["Page one.", "Page two.", "Page three."])
        assert result.page_for_offset(len(result.text) + 1000) == 3
