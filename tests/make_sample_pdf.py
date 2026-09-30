"""One-off helper (not part of the package) to generate a tiny sample PDF
for testing extract_pdf_pages()/chunk_text() without needing your real
grammar/vocab books."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_sample_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)

    # Page 1: some running header + page number noise + real content.
    c.setFont("Helvetica", 10)
    c.drawString(72, 800, "German Grammar Reference")
    c.setFont("Helvetica", 12)
    y = 760
    for line in [
        "Chapter 3: The Accusative Case",
        "",
        "The accusative case is used for the direct object of a sentence.",
        "It answers the question 'wen?' or 'was?' (whom? or what?).",
        "",
        "Der Mann sieht den Hund.",
        "The man sees the dog.",
    ]:
        c.drawString(72, y, line)
        y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(300, 40, "42")  # bare page number, like a footer
    c.showPage()

    # Page 2
    c.setFont("Helvetica", 10)
    c.drawString(72, 800, "German Grammar Reference")
    c.setFont("Helvetica", 12)
    y = 760
    for line in [
        "Masculine nouns change from 'der' to 'den' in the accusative.",
        "Feminine, neuter, and plural articles do not change.",
        "",
        "Examples:",
        "Ich sehe den Mann. (I see the man.)",
        "Ich sehe die Frau. (I see the woman.)",
        "Ich sehe das Kind. (I see the child.)",
    ]:
        c.drawString(72, y, line)
        y -= 20
    c.setFont("Helvetica", 9)
    c.drawString(300, 40, "43")
    c.showPage()

    c.save()


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_grammar.pdf"
    make_sample_pdf(out)
    print(f"wrote {out}")
