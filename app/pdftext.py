"""Fallback for papers without LaTeX source: one section per PDF page, text extracted with PyMuPDF."""
from pathlib import Path

import pymupdf


def pdf_sections(folder: Path) -> list[tuple[list[str], str]]:
    pdf = folder / "paper.pdf"
    if not pdf.exists():
        return []
    sections = []
    with pymupdf.open(pdf) as doc:
        for number, page in enumerate(doc, start=1):
            text = " ".join(page.get_text().split())
            if text:  # scanned pages have no text layer
                sections.append(([f"p. {number}"], text))
    return sections
