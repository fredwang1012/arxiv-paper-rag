"""Split a downloaded paper into cited sections: from LaTeX when it parses, else from the PDF pages."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.latex import latex_sections
from app.pdftext import pdf_sections


log = logging.getLogger(__name__)


@dataclass
class Section:
    paper_id: str
    path: list[str]  # e.g. ["3 Model", "3.1 Inference"]
    text: str
    source: Literal["latex", "pdf"]


def _paper_id(paper_dir: Path) -> str:
    try:
        return json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))["id"]
    except (OSError, ValueError, KeyError):
        return paper_dir.name  # the folder name is the id with "/" replaced by "_"


def load_sections(paper_dir: Path) -> list[Section]:
    paper_id = _paper_id(paper_dir)
    try:
        parsed, source = latex_sections(paper_dir), "latex"
    except Exception as e:  # one odd source (even a MemoryError) must not lose the paper: use its PDF
        log.warning("LaTeX parsing failed for %s (%s); using the PDF", paper_id, type(e).__name__)
        parsed, source = [], "latex"
    if not parsed:
        parsed, source = pdf_sections(paper_dir), "pdf"
    return [Section(paper_id, path, text, source) for path, text in parsed]
