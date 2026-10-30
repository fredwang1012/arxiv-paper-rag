"""PaperQA2 on the same PDFs, so the answer eval can compare it with this project on the same questions and metrics.

paper-qa is not a dependency of the app; run the comparison with it added for that run only:

    uv run --with paper-qa==2026.8.12 python -m app.evaluate --answers --systems plain agentic paperqa2

PaperQA2's own agent does the work (paper search over a local index of the PDFs, evidence gathering, answer). Its
answer cites contexts by id; to_answer turns those into numbered passages and [n] markers, so citation precision and
recall, keyword hits and the claim check are computed exactly as for the other systems.
"""
import asyncio
import csv
import re
import shutil
from pathlib import Path

from app import arxiv
from app.answer import Answer, Citation, make_passage, marker_numbers
from app.chunks import Chunk
from app.config import settings
from app.corpus import Corpus
from app.llm import LLMError
from app.retrieval import Hit

PIN = "paper-qa==2026.8.12"
EMBEDDING = "st-intfloat/multilingual-e5-small"  # the same embedder as this project's hybrid search
_GROUP = re.compile(r"\(([^()]*)\)")


def paper_id_of(doc, known: set[str]) -> str | None:
    """The arXiv id of a PaperQA2 doc, from whichever of its fields carries it."""
    for field in ("docname", "file_location", "dockey", "citation"):
        value = str(getattr(doc, field, "") or "")
        for pid in known:
            if pid in value:
                return pid
    return None


def marker_text(raw_answer: str, numbers: dict[str, int]) -> str:
    """"(pqac-1, pqac-2)" -> "[1, 2]" for the context ids we numbered; other parentheses are left alone."""
    def swap(m: re.Match) -> str:
        found = sorted({n for cid, n in numbers.items() if cid in m.group(1)})
        return f"[{', '.join(map(str, found))}]" if found else m.group(0)
    return _GROUP.sub(swap, raw_answer)


def to_answer(session, corpus: Corpus) -> Answer:
    """A PaperQA2 session as an Answer: its used contexts become passages, numbered in the order first cited."""
    raw = session.raw_answer or ""
    if not session.has_successful_answer:
        return Answer(session.answer or raw)
    used = sorted((c for c in session.contexts if c.id in raw), key=lambda c: raw.find(c.id))
    passages, numbers = [], {}
    for c in used:
        pid = paper_id_of(c.text.doc, set(corpus.metas))
        if pid is None or c.id in numbers:
            continue
        numbers[c.id] = len(passages) + 1
        chunk = Chunk(f"pqa:{c.id}", pid, [c.text.name], c.text.text)
        passages.append(make_passage(corpus, numbers[c.id], Hit(chunk, float(c.score))))
    text = marker_text(raw, numbers)
    citations = []
    for p in passages:
        meta = corpus.paper(p.chunk.paper_id)
        # PaperQA2 does not quote: an empty quote is left out of the quote-found rate
        citations.append(Citation(meta.id, meta.title, meta.authors, meta.year, p.section, p.chunk.id, "", False,
                                  p.number))
    marked = marker_numbers(text)
    cited = {c.passage for c in citations}
    return Answer(text, citations, passages, sorted(marked - cited), sorted(cited - marked))


def _llm_config(name: str) -> dict:
    # no temperature: current Claude models reject sampling parameters, and PaperQA2's default config sends one
    return {"name": name, "model_list": [{"model_name": name, "litellm_params": {"model": name}}]}


def stage(corpus: Corpus, root: Path) -> Path:
    """Copy each paper's PDF to <root>/papers/<id>.pdf with a manifest, so PaperQA2 never looks up metadata online."""
    papers = root / "papers"
    papers.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, meta in corpus.metas.items():
        src = arxiv.paper_dir(pid) / "paper.pdf"
        if not src.is_file():
            continue
        name = pid.replace("/", "_") + ".pdf"
        if not (papers / name).exists():
            shutil.copyfile(src, papers / name)
        authors = ", ".join(meta.authors[:3]) + (" et al." if len(meta.authors) > 3 else "")
        rows.append({"file_location": name, "title": meta.title, "docname": pid,
                     "citation": f"{authors} ({meta.year}). {meta.title}. arXiv:{pid}"})
    with (papers / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_location", "title", "docname", "citation"])
        writer.writeheader()
        writer.writerows(rows)
    return papers


class PaperQA2:
    def __init__(self, corpus: Corpus, root: Path | None = None):
        try:
            from paperqa import Settings, agent_query
        except ImportError as e:
            raise RuntimeError(f"paper-qa is not installed; run with `uv run --with {PIN} ...`") from e
        self.corpus, self._query = corpus, agent_query
        root = root or settings.data_dir / "paperqa2"
        papers = stage(corpus, root)
        llm = f"anthropic/{settings.llm_model}"
        self.settings = Settings(
            llm=llm, llm_config=_llm_config(llm), summary_llm=llm, summary_llm_config=_llm_config(llm),
            embedding=EMBEDDING,
            parsing={"use_doc_details": False},
            agent={"agent_llm": llm, "agent_llm_config": _llm_config(llm),
                   "index": {"paper_directory": str(papers), "index_directory": str(root / "index"),
                             "manifest_file": str(papers / "manifest.csv")}},
        )

    def ask(self, question: str) -> Answer:
        try:
            response = asyncio.run(self._query(question, self.settings))
        except Exception as e:  # PaperQA2 and LiteLLM raise their own types; record the class, not the payload
            raise LLMError(f"PaperQA2 failed ({type(e).__name__})") from e
        return to_answer(response.session, self.corpus)
