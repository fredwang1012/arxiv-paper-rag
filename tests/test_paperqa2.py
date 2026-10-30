"""The PaperQA2 adapter's conversions and staging, with sessions shaped like paper-qa's; paper-qa is not needed."""
import csv
from types import SimpleNamespace

import pytest

from app import answer as answer_mod
from app import arxiv, paperqa2
from app.llm import LLMError
from tests.test_agent import two_papers
from tests.test_answer import FakeLLM
from tests.test_pdflocate import PAGES, write_pdf
from app.llm import RawAnswer, RawVerdict, RawVerification


def ctx(cid, docname, name, text, score=7):
    doc = SimpleNamespace(docname=docname, file_location=None, dockey="k", citation="c")
    return SimpleNamespace(id=cid, score=score, text=SimpleNamespace(name=name, text=text, doc=doc))


def session(raw, contexts, ok=True, answer=None):
    return SimpleNamespace(raw_answer=raw, answer=answer or raw, contexts=contexts, has_successful_answer=ok)


def test_to_answer_numbers_used_contexts_in_citation_order():
    c = two_papers()
    s = session("Gears evolve (pqac-2). Widgets hum (pqac-1, pqac-2). See (Fig. 3).", [
        ctx("pqac-1", "1111.0001", "widgets pages 1-2", "Widgets hum quietly."),
        ctx("pqac-2", "2222.0002", "gears pages 3", "Gears are trained with evolutionary search."),
        ctx("pqac-9", "1111.0001", "unused", "never cited"),
    ])
    a = paperqa2.to_answer(s, c)
    assert a.text == "Gears evolve [1]. Widgets hum [1, 2]. See (Fig. 3)."
    assert [(p.number, p.chunk.paper_id, p.section) for p in a.passages] == [(1, "2222.0002", "gears pages 3"),
                                                                             (2, "1111.0001", "widgets pages 1-2")]
    assert [x.paper_id for x in a.citations] == ["2222.0002", "1111.0001"]
    assert all(x.quote == "" for x in a.citations) and a.unresolved_markers == [] and a.unmarked_citations == []


def test_unknown_documents_are_dropped_and_failed_answers_cite_nothing():
    c = two_papers()
    s = session("Claim (pqac-1). Other (pqac-x).", [ctx("pqac-1", "1111.0001", "w", "t"),
                                                    ctx("pqac-x", "somebook2020", "b", "t")])
    a = paperqa2.to_answer(s, c)
    assert a.text == "Claim [1]. Other (pqac-x)." and len(a.passages) == 1
    a = paperqa2.to_answer(session("", [], ok=False, answer="I cannot answer."), c)
    assert (a.text, a.citations, a.passages) == ("I cannot answer.", [], [])


def test_paper_id_of_looks_in_every_field():
    known = {"1511.06279", "2003.06085"}
    assert paperqa2.paper_id_of(SimpleNamespace(docname="reed2015", file_location="/x/1511.06279.pdf"), known) == \
        "1511.06279"
    assert paperqa2.paper_id_of(SimpleNamespace(docname="x", citation="… arXiv:2003.06085"), known) == "2003.06085"
    assert paperqa2.paper_id_of(SimpleNamespace(docname="x"), known) is None


def test_claim_check_runs_on_converted_answers():
    c = two_papers()
    a = paperqa2.to_answer(session("Gears evolve (pqac-2).", [ctx("pqac-2", "2222.0002", "g", "Gears evolve.")]), c)
    llm = FakeLLM(RawAnswer(answer="", citations=[]),
                  RawVerification(verdicts=[RawVerdict(claim=1, passage=1, verdict="supported", reason="")]))
    answer_mod.check_support(a, llm)
    assert [(s.claim, s.verdict) for s in a.support] == [("Gears evolve [1].", "supported")]
    (claims, passages), = llm.verify_calls
    assert passages[0].chunk.text == "Gears evolve."


def test_stage_copies_pdfs_and_writes_a_manifest(tmp_path):
    c = two_papers()
    write_pdf(arxiv.paper_dir("1111.0001") / "paper.pdf", PAGES)  # 2222.0002 has no PDF: left out
    papers = paperqa2.stage(c, tmp_path / "pqa")
    assert (papers / "1111.0001.pdf").read_bytes() == (arxiv.paper_dir("1111.0001") / "paper.pdf").read_bytes()
    with (papers / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"file_location": "1111.0001.pdf", "title": "Neural Widgets", "docname": "1111.0001",
                     "citation": "A. Author (2020). Neural Widgets. arXiv:1111.0001"}]


def test_llm_config_sends_no_sampling_parameters():
    cfg = paperqa2._llm_config("anthropic/claude-opus-5-5")
    assert cfg["model_list"][0]["litellm_params"] == {"model": "anthropic/claude-opus-5-5"}


def test_missing_package_and_failures_are_clear(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_paperqa(name, *args, **kwargs):
        if name == "paperqa":
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_paperqa)
    with pytest.raises(RuntimeError, match="uv run --with paper-qa=="):
        paperqa2.PaperQA2(two_papers())

    pqa = object.__new__(paperqa2.PaperQA2)

    async def boom(question, settings):
        raise ValueError("secret payload")

    pqa._query, pqa.settings, pqa.corpus = boom, None, None
    with pytest.raises(LLMError) as info:
        pqa.ask("q")
    assert "ValueError" in str(info.value) and "secret" not in str(info.value)
