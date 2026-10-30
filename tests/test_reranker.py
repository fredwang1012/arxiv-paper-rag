"""Reranking: the reordering itself, paper filters, failure fallback, settings and the eval rows. No real model."""
import json

import numpy as np

from app import corpus as corpus_mod
from app import evaluate, reranker
from app.chunks import Chunk
from app.config import settings
from app.retrieval import Index
from tests.test_agent import two_papers


class WordReranker:
    """Scores a text by how often it contains `favourite`, so the order it wants is easy to state."""

    def __init__(self, favourite="gears"):
        self.favourite, self.calls = favourite, []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return np.array([t.lower().count(self.favourite) for t in texts], dtype=float)


def small_index():
    chunks = [Chunk("p#c0", "p", ["1 A"], "widgets widgets hum"), Chunk("p#c1", "p", ["2 B"], "widgets and gears"),
              Chunk("q#c0", "q", ["1 A"], "widgets gears gears")]
    return Index(chunks, [c.text for c in chunks])


def test_reranked_reorders_the_first_stage_candidates():
    index = small_index()
    first = [h.chunk.id for h in index.search("widgets", 3)]
    assert first[0] == "p#c0"
    rr = WordReranker()
    hits = index.reranked("widgets", 2, rr, depth=3)
    assert [h.chunk.id for h in hits] == ["q#c0", "p#c1"] and hits[0].score == 2.0
    (query, texts), = rr.calls
    assert query == "widgets" and len(texts) == 3  # depth candidates, not k


def test_reranked_respects_the_paper_filter_and_depth():
    index = small_index()
    rr = WordReranker()
    assert [h.chunk.id for h in index.reranked("widgets", 5, rr, depth=2, paper="p")] == ["p#c1", "p#c0"]
    assert [h.chunk.id for h in index.search("widgets", 5, paper="q")] == ["q#c0"]
    assert index.reranked("zzz", 3, rr, depth=3) == []


def test_reranker_failure_keeps_first_stage_order():
    class Broken:
        def score(self, query, texts):
            raise RuntimeError("boom")

    index = small_index()
    assert [h.chunk.id for h in index.reranked("widgets", 2, Broken(), 3)] == \
        [h.chunk.id for h in index.search("widgets", 2)]


def test_corpus_search_reranks_only_when_enabled(monkeypatch):
    c = two_papers()
    rr = WordReranker()
    monkeypatch.setattr(corpus_mod, "get_reranker", lambda: rr if settings.rerank == "auto" else None)
    plain = [h.chunk.id for h in c.search("widgets gears trained", 4)]
    assert rr.calls == []
    monkeypatch.setattr(settings, "rerank", "auto")
    ranked = c.search("widgets gears trained", 4)
    assert len(rr.calls) == 1 and len(ranked) == len(plain)
    assert "gears" in ranked[0].chunk.text.lower() and ranked[0].score >= ranked[-1].score


def test_get_reranker_obeys_the_setting_and_remembers_failure(monkeypatch):
    built = []

    class Fake:
        def __init__(self, name):
            built.append(name)
            raise OSError("offline")

    monkeypatch.setattr(reranker, "CrossEncoderReranker", Fake)
    assert reranker.get_reranker() is None and built == []  # RERANK=off: never loaded
    assert reranker.get_reranker(force=True) is None and built == [settings.rerank_model]
    assert reranker.get_reranker(force=True) is None and len(built) == 1  # a failure is not retried


def test_eval_scores_rerank_rows_when_the_model_loads(tmp_path, monkeypatch, capsys):
    from tests.test_evaluate import QUESTIONS
    two_papers()
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in QUESTIONS) + "\n", encoding="utf-8")
    out = tmp_path / "out.json"
    assert evaluate.main(["--questions", str(path), "--json", str(out)]) == 0
    assert "rerank: skipped" in capsys.readouterr().out
    monkeypatch.setattr(evaluate, "get_reranker", lambda force=False: WordReranker() if force else None)
    assert evaluate.main(["--questions", str(path), "--json", str(out)]) == 0
    assert "| bm25+rerank | section |" in capsys.readouterr().out
    assert set(json.loads(out.read_text(encoding="utf-8"))["retrieval"]) == {"bm25", "bm25+rerank"}
