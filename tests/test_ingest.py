import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from german_buddy.ingest import chunk_text, extract_pdf_pages  # noqa: E402
from make_sample_pdf import make_sample_pdf  # noqa: E402

SAMPLE_PDF = Path(__file__).parent / "sample_grammar.pdf"


def _ensure_sample_pdf():
    if not SAMPLE_PDF.exists():
        make_sample_pdf(SAMPLE_PDF)


def test_extract_pdf_pages_strips_bare_page_numbers():
    _ensure_sample_pdf()
    pages = extract_pdf_pages(SAMPLE_PDF)
    assert len(pages) == 2
    for _, text in pages:
        # the footer page numbers ("42", "43") should not survive as
        # standalone lines
        assert "42" not in text.split("\n")
        assert "43" not in text.split("\n")


def test_extract_pdf_pages_keeps_real_content():
    _ensure_sample_pdf()
    pages = extract_pdf_pages(SAMPLE_PDF)
    page1_text = pages[0][1]
    assert "accusative" in page1_text.lower()


def test_extract_pdf_pages_missing_file_returns_empty():
    assert extract_pdf_pages(Path("does_not_exist.pdf")) == []


def test_chunk_text_respects_size_limit_roughly():
    text = "\n\n".join([f"Paragraph {i} " + ("x" * 50) for i in range(20)])
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) < 400 for c in chunks)


def test_chunk_text_falls_back_to_lines_when_no_paragraphs():
    # simulates PDF-extracted text with no blank-line paragraph breaks
    text = "\n".join([f"Line {i}" for i in range(10)])
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    combined = " ".join(chunks)
    for i in range(10):
        assert f"Line {i}" in combined
