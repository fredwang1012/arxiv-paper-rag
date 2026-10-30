"""Finding a cited passage in the PDF, highlighting it, and the PDF endpoint. PDFs are made in the test."""
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import answer as answer_mod
from app import arxiv, main, pdflocate
from app.llm import RawAnswer, RawCitation
from tests.test_answer import FakeLLM, make_corpus


def write_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(72, 72, 320, 700), text, fontsize=11)
    doc.save(path)
    return path


PAGES = ["Widgets are small devices. They hum quietly in the dark.",
         "We train the widget with gradient descent on synthetic gadgets over many steps of the program."]


def test_whole_quote_wins_even_across_line_breaks(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", PAGES)
    loc = pdflocate.locate(pdf, "train the widget with gradient descent on synthetic gadgets over many", "")
    assert loc.page == 2 and len(loc.needles) == 1


def test_latex_quote_falls_back_to_windows_then_the_passage(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", PAGES)
    loc = pdflocate.locate(pdf, r"we \emph{train} the widget with gradient descent on synthetic gadgets", "")
    assert loc.page == 2 and all("\\" not in n for n in loc.needles)
    loc = pdflocate.locate(pdf, "$x^2$", r"Widgets are small devices. They hum quietly in the \textbf{dark}.")
    assert loc.page == 1


def test_ligatures_and_hyphenation_are_normalised():
    assert pdflocate._norm("ﬁne-\ntuning   the  Model") == "finetuning the model"


def test_nothing_found_or_no_pdf(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", PAGES)
    assert pdflocate.locate(pdf, "completely unrelated words that are nowhere", "also absent from every page here") is None
    assert pdflocate.locate(tmp_path / "missing.pdf", "x", "y") is None


def test_windows_skip_latex_and_are_capped():
    words = " ".join(f"w{i}" for i in range(200))
    assert len(pdflocate._windows(words)) == pdflocate.MAX_WINDOWS
    assert pdflocate._windows(r"a b \cite{x} c d e f g h") == ["c d e f g h"]


def test_highlighted_adds_annotations_on_the_page(tmp_path):
    pdf = write_pdf(tmp_path / "p.pdf", PAGES)
    out = pdflocate.highlighted(pdf, pdflocate.Location(2, ["gradient descent"]))
    doc = pymupdf.open(stream=out)
    assert [a.type[1] for a in doc[1].annots()] == ["Highlight"] and list(doc[0].annots()) == []


def test_citations_carry_the_pdf_page():
    c = make_corpus()
    write_pdf(arxiv.paper_dir("1111.0001") / "paper.pdf", PAGES)
    llm = FakeLLM(RawAnswer(answer="Trained by gradient descent [1].",
                            citations=[RawCitation(passage=1, quote="gradient descent on synthetic gadgets")]))
    a = answer_mod.ask("gradient descent", c, llm)
    assert a.citations[0].pdf_page == 2


def test_damaged_pdf_leaves_the_page_empty():
    c = make_corpus()
    path = arxiv.paper_dir("1111.0001") / "paper.pdf"
    path.write_bytes(b"%PDF-1.4 not really")
    llm = FakeLLM(RawAnswer(answer="x [1].", citations=[RawCitation(passage=1, quote="hum")]))
    assert answer_mod.ask("widgets hum", c, llm).citations[0].pdf_page is None


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "_corpus", None)
    return TestClient(main.app, base_url="http://localhost")


def test_pdf_endpoint_highlights_the_quote(client):
    make_corpus()
    write_pdf(arxiv.paper_dir("1111.0001") / "paper.pdf", PAGES)
    r = client.get("/api/papers/1111.0001/pdf", params={"quote": "hum quietly in the dark"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert "inline" in r.headers["content-disposition"]
    doc = pymupdf.open(stream=r.content)
    assert [a.type[1] for a in doc[0].annots()] == ["Highlight"]


def test_pdf_endpoint_uses_the_chunk_when_the_quote_is_missing(client):
    c = make_corpus()
    write_pdf(arxiv.paper_dir("1111.0001") / "paper.pdf", PAGES)
    chunk = next(ch for ch in c.chunks if "gradient" in ch.text)
    r = client.get("/api/papers/1111.0001/pdf", params={"chunk": chunk.id})
    assert len(list(pymupdf.open(stream=r.content)[1].annots())) >= 1
    plain = client.get("/api/papers/1111.0001/pdf")  # nothing to find: the PDF as it is
    assert plain.content == (arxiv.paper_dir("1111.0001") / "paper.pdf").read_bytes()


@pytest.mark.parametrize("paper", ["9999.9999", "..", "1111.0001%2F..%2F..", "%2Fetc%2Fpasswd"])
def test_pdf_endpoint_refuses_unlisted_ids(client, paper):
    make_corpus()
    assert client.get(f"/api/papers/{paper}/pdf").status_code == 404


def test_pdf_endpoint_without_a_downloaded_pdf(client):
    make_corpus()
    r = client.get("/api/papers/1111.0001/pdf")
    assert r.status_code == 404 and "not been downloaded" in r.json()["detail"]


def test_page_links_citations_to_the_pdf():
    html = (main.STATIC / "index.html").read_text(encoding="utf-8")
    assert "function pdfLink" in html and 'a.rel = "noopener"' in html and "encodeURIComponent(cit.paper_id)" in html
