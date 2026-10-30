"""Find a cited passage in the paper's PDF and serve the PDF with it highlighted.

The passages come from LaTeX, so a quote rarely matches the PDF text exactly (ligatures, hyphenation, macros). The
whole quote is tried first, then six-word windows of it, then windows of the passage; the page with the most window
matches wins, and every match on that page is highlighted.
"""
import functools
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pymupdf

WINDOW, STEP, MAX_WINDOWS = 6, 3, 24
_LATEX_WORD = re.compile(r"[\\$^_{}~]")


@dataclass
class Location:
    page: int  # 1-based
    needles: list[str]  # the strings that matched on that page, for highlighting


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("-\n", "")
    return " ".join(text.lower().split())


@functools.lru_cache(maxsize=16)
def _page_texts(path: str, mtime_ns: int) -> tuple[str, ...]:
    with pymupdf.open(path) as doc:
        return tuple(_norm(page.get_text()) for page in doc)


def page_texts(pdf: Path) -> tuple[str, ...]:
    return _page_texts(str(pdf), pdf.stat().st_mtime_ns)


def _windows(text: str) -> list[str]:
    """Six-word windows over runs of words that can appear in a PDF (no LaTeX commands or math)."""
    out: list[str] = []
    run: list[str] = []
    for word in _norm(text).split() + [""]:
        if word and not _LATEX_WORD.search(word):
            run.append(word)
            continue
        if len(run) >= WINDOW:
            out += [" ".join(run[i:i + WINDOW]) for i in range(0, len(run) - WINDOW + 1, STEP)]
        elif len(run) >= 3 and not out:
            out.append(" ".join(run))
        run = []
    return out[:MAX_WINDOWS]


def locate(pdf: Path, quote: str, passage: str) -> Location | None:
    """The page holding the quote (or, failing that, the passage), or None when nothing matches."""
    if not pdf.is_file():
        return None
    pages = page_texts(pdf)
    whole = _norm(quote)
    if len(whole.split()) >= 3 and not _LATEX_WORD.search(whole):
        for number, text in enumerate(pages, start=1):
            if whole in text:
                return Location(number, [whole])
    for source in (quote, passage):
        hits: dict[int, list[str]] = {}
        for needle in _windows(source):
            for number, text in enumerate(pages, start=1):
                if needle in text:
                    hits.setdefault(number, []).append(needle)
        if hits:
            page = max(hits, key=lambda n: (len(hits[n]), -n))
            return Location(page, hits[page])
    return None


def highlighted(pdf: Path, location: Location) -> bytes:
    """The PDF with every match of the location's strings highlighted on its page."""
    with pymupdf.open(pdf) as doc:
        page = doc[location.page - 1]
        for needle in location.needles:
            quads = page.search_for(needle, quads=True)
            if quads:  # one annotation per match, even when it wraps onto several lines
                page.add_highlight_annot(quads)
        return doc.tobytes()
