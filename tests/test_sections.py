"""load_sections: LaTeX when it parses, otherwise one section per PDF page."""
import gzip
import json

import pymupdf

from app.sections import load_sections


def make_pdf(path, pages: list[str]):
    pdf = pymupdf.open()
    for text in pages:
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_latex_preferred_over_pdf(tmp_path):
    folder = tmp_path / "1511.06279"
    folder.mkdir()
    (folder / "meta.json").write_text(json.dumps({"id": "1511.06279"}))
    (folder / "source.gz").write_bytes(
        gzip.compress(b"\\documentclass{article}\\begin{document}\\section{Model}hello\\end{document}"))
    make_pdf(folder / "paper.pdf", ["page one"])
    s = load_sections(folder)
    assert len(s) == 1
    assert (s[0].paper_id, s[0].path, s[0].text, s[0].source) == ("1511.06279", ["1 Model"], "hello", "latex")


def test_pdf_fallback_one_section_per_page(tmp_path):
    folder = tmp_path / "cs_0112017"
    folder.mkdir()
    (folder / "source.none").write_bytes(b"")
    make_pdf(folder / "paper.pdf", ["first page words", "second page words"])
    s = load_sections(folder)
    assert [(x.paper_id, x.path, x.source) for x in s] == [
        ("cs_0112017", ["p. 1"], "pdf"), ("cs_0112017", ["p. 2"], "pdf")]
    assert "first page words" in s[0].text and "second page words" in s[1].text


def test_unparseable_latex_falls_back_to_pdf(tmp_path):
    (tmp_path / "source.gz").write_bytes(gzip.compress(b"plain text, not latex"))
    make_pdf(tmp_path / "paper.pdf", ["only page"])
    assert [x.source for x in load_sections(tmp_path)] == ["pdf"]


def test_corrupt_source_gz_falls_back_to_pdf(tmp_path):
    good = gzip.compress(b"\\documentclass{article}\\begin{document}\\section{S}" + b"hello world " * 500)
    (tmp_path / "source.gz").write_bytes(good[:20] + b"\xff" * 30 + good[50:])  # valid header, mangled stream
    make_pdf(tmp_path / "paper.pdf", ["only page"])
    assert [x.source for x in load_sections(tmp_path)] == ["pdf"]


def test_nothing_usable_gives_empty_list(tmp_path):
    assert load_sections(tmp_path) == []
