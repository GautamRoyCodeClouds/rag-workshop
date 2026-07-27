"""Loader tests.

clean_pages is tested against strings so the zero-width assertions are exact.
load_pdf is tested against a generated PDF for file-level properties.
"""

import pytest

from app.pipeline.loader import EmptyDocumentError, clean_pages, load_pdf


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

    def test_removes_the_repeated_footer(self, dirty_pages):
        result = clean_pages(dirty_pages)
        assert "Confidential" not in "\n".join(result.pages)
        assert result.boilerplate_lines_removed >= 4

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

    def test_page_attribution_starts_at_one_and_advances(self, structured_pdf):
        result = load_pdf(structured_pdf)
        assert result.page_for_offset(0) == 1
        assert result.page_for_offset(len(result.text) - 1) >= 1

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
