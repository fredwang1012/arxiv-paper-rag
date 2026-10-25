"""Chunking, hybrid search and the on-disk index cache."""
import gzip
import json
import os
from dataclasses import asdict

import numpy as np

from app import corpus
from app.arxiv import PaperMeta, paper_dir
from app.chunks import Chunk, chunk_sections, search_text, split_text
from app.config import settings
from app.retrieval import Index
from app.sections import Section


def meta(paper_id="1111.0001", title="Neural Widgets") -> PaperMeta:
    return PaperMeta(paper_id, 1, title, ["A. Author"], 2020, "abstract", "cs.LG")


def add_paper(m: PaperMeta, sections: dict[str, str]):
    folder = paper_dir(m.id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").write_text(json.dumps(asdict(m)))
    body = "".join(f"\\section{{{title}}}\n{text}\n" for title, text in sections.items())
    (folder / "source.gz").write_bytes(
        gzip.compress(f"\\documentclass{{article}}\\begin{{document}}{body}\\end{{document}}".encode()))
    return folder


class FakeEmbedder:
    """One dimension per concept; synonyms share a dimension, which plain BM25 cannot see."""
    CONCEPTS = [("car", "automobile", "vehicle"), ("cat", "feline"), ("tree", "oak")]

    def vec(self, text: str) -> np.ndarray:
        words = text.lower().split()
        v = np.array([sum(w.strip(".,") in group for w in words) for group in self.CONCEPTS], dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def embed_query(self, text):
        return self.vec(text)

    def embed_passages(self, texts):
        return np.stack([self.vec(t) for t in texts])


# ---------- chunking ----------

def test_chunk_sizes_and_ids():
    sentence = "This sentence has a fairly ordinary length of about sixty chars. "
    section = Section("2001.00001", ["1 Intro"], sentence * 60, "latex")
    other = Section("2001.00001", ["2 Method"], "Short one. Another short one.", "latex")
    chunks = chunk_sections([section, other])
    assert [c.id for c in chunks] == [f"2001.00001#c{i}" for i in range(len(chunks))]
    assert len(chunks) > 3
    assert all(len(c.text) <= 1400 for c in chunks)
    assert all(len(c.text) >= 800 for c in chunks[:-2])  # packed, not one sentence each
    assert all(c.text.endswith(".") for c in chunks)  # cut at sentence ends
    assert chunks[-1].section_path == ["2 Method"] and chunks[-1].text == "Short one. Another short one."
    assert " ".join(c.text for c in chunks[:-1]) == section.text.strip()


def test_text_without_long_runs_stays_one_chunk():
    assert split_text("First one. (Second) one! [Third] one? fourth stays.") == [
        "First one. (Second) one! [Third] one? fourth stays."]


def test_overlong_sentence_is_split_on_whitespace():
    text = "word " * 800  # 4000 characters and no sentence end
    pieces = split_text(text)
    assert len(pieces) >= 4 and all(len(p) <= 1400 for p in pieces)
    assert " ".join(pieces) == text.strip()


def test_search_text_has_title_and_path_but_chunk_text_stays_raw():
    c = Chunk("p#c0", "p", ["3 Model", "3.1 Inference"], "raw words")
    assert search_text(c, "Big Title") == "Big Title | 3 Model > 3.1 Inference | raw words"
    assert c.text == "raw words"


# ---------- search ----------

def test_section_name_query_hits_that_section(tmp_path):
    add_paper(meta(), {"Alpha": "Plain filler about nothing in particular. " * 3,
                       "Zebra Crossing": "Plain filler about nothing in particular. " * 3})
    c = corpus.build()
    hits = c.search("zebra crossing")
    assert hits and hits[0].chunk.section_path == ["2 Zebra Crossing"]
    assert "zebra" not in hits[0].chunk.text.lower()  # found through the section path, not the stored text


def test_title_words_are_searchable_and_scores_are_ordered():
    add_paper(meta(title="Quantum Gardening"), {"Intro": "Some words here."})
    add_paper(meta("1111.0002", "Other Topic"), {"Intro": "Different words."})
    hits = corpus.build().search("quantum gardening", k=5)
    assert hits[0].chunk.paper_id == "1111.0001"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_bm25_only_hits_must_share_a_query_term():
    add_paper(meta(), {"Intro": "Cats purr."})
    assert corpus.build().search("completely unrelated xylophone") == []


def test_hybrid_finds_a_paraphrase_that_bm25_misses():
    chunks = [Chunk("p#c0", "p", ["1 A"], "The automobile stopped."), Chunk("p#c1", "p", ["1 B"], "The oak grew tall.")]
    texts = ["T | 1 A | The automobile stopped.", "T | 1 B | The oak grew tall."]
    plain = Index(chunks, texts)
    assert plain.search("car") == []  # no shared word
    fake = FakeEmbedder()
    hybrid = Index(chunks, texts, fake.embed_passages(texts), fake)
    assert hybrid.mode == "hybrid"
    assert hybrid.search("car")[0].chunk.id == "p#c0"


def test_embedding_count_mismatch_falls_back_to_bm25():
    chunks = [Chunk("p#c0", "p", ["1 A"], "x")]
    fake = FakeEmbedder()
    assert Index(chunks, ["x"], np.zeros((2, 3)), fake).mode == "bm25"


# ---------- cache ----------

def test_cache_is_reused_then_rebuilt_when_files_change(monkeypatch):
    folder = add_paper(meta(), {"Intro": "First version text."})
    calls = []
    real = corpus.chunk_sections
    monkeypatch.setattr(corpus, "chunk_sections", lambda s: calls.append(1) or real(s))

    corpus.build()
    assert len(calls) == 1
    assert {p.name for p in (settings.data_dir / "index").iterdir()} == {"chunks.json", "manifest.json"}
    again = corpus.build()
    assert len(calls) == 1  # reused
    assert again.search("first version")[0].chunk.text == "First version text."

    add_paper(meta(), {"Intro": "Second version text, longer now."})
    changed = corpus.build()
    assert len(calls) == 2
    assert changed.search("second version")[0].chunk.text == "Second version text, longer now."

    corpus.build()
    assert len(calls) == 2
    os.utime(folder / "source.gz", ns=(1, 1))  # same size, different mtime
    corpus.build()
    assert len(calls) == 3


def test_cache_is_rebuilt_when_a_paper_is_added_or_removed(monkeypatch):
    add_paper(meta(), {"Intro": "One."})
    assert {c.paper_id for c in corpus.build().chunks} == {"1111.0001"}
    add_paper(meta("1111.0002", "Two"), {"Intro": "Two."})
    assert {c.paper_id for c in corpus.build().chunks} == {"1111.0001", "1111.0002"}
    (paper_dir("1111.0001") / "meta.json").unlink()
    assert {c.paper_id for c in corpus.build().chunks} == {"1111.0002"}


def test_corrupt_cache_is_rebuilt():
    add_paper(meta(), {"Intro": "Some text."})
    corpus.build()
    (settings.data_dir / "index" / "chunks.json").write_text("not json")
    assert corpus.build().search("some text")


def test_embeddings_are_cached_and_embed_mode_change_rebuilds(monkeypatch):
    add_paper(meta(), {"Cars": "The automobile stopped.", "Trees": "The oak grew tall."})
    fake = FakeEmbedder()
    monkeypatch.setattr(corpus, "get_embedder", lambda: fake)
    first = corpus.build()
    assert first.index.mode == "hybrid"
    assert first.search("car")[0].chunk.section_path == ["1 Cars"]
    assert (settings.data_dir / "index" / "embeddings.npy").exists()

    calls = []
    monkeypatch.setattr(fake, "embed_passages", lambda t: calls.append(1))
    assert corpus.build().index.mode == "hybrid"  # cached embeddings, no new embedding call
    assert calls == []

    monkeypatch.setattr(corpus, "get_embedder", lambda: None)
    assert corpus.build().index.mode == "bm25"  # mode differs from the cached one


def test_embedding_failure_falls_back_to_bm25(monkeypatch, caplog):
    add_paper(meta(), {"Intro": "Some text."})

    class Broken(FakeEmbedder):
        def embed_passages(self, texts):
            raise RuntimeError("boom")

    monkeypatch.setattr(corpus, "get_embedder", lambda: Broken())
    with caplog.at_level("WARNING"):
        c = corpus.build()
    assert c.index.mode == "bm25" and c.search("some text")
    assert "boom" in caplog.text


def test_paper_returns_meta_and_empty_corpus_is_fine():
    assert corpus.build().search("anything") == []
    add_paper(meta(title="Neural Widgets"), {"Intro": "Text."})
    c = corpus.build()
    assert c.paper("1111.0001").title == "Neural Widgets"
    assert c.paper("1111.0001").year == 2020


# ---------- robustness ----------

def test_one_bad_paper_does_not_break_the_build(monkeypatch, caplog):
    add_paper(meta("1111.0001"), {"Intro": "Bad paper words."})
    add_paper(meta("1111.0002", "Good"), {"Intro": "Good paper words."})
    real = corpus.load_sections

    def flaky(folder):
        if folder.name == "1111.0001":
            raise MemoryError
        return real(folder)

    monkeypatch.setattr(corpus, "load_sections", flaky)
    with caplog.at_level("WARNING"):
        c = corpus.build()
    assert {x.paper_id for x in c.chunks} == {"1111.0002"}
    assert "1111.0001" in caplog.text and "MemoryError" in caplog.text


def test_latex_failure_falls_back_to_the_pdf(tmp_path, monkeypatch):
    from app import sections
    from tests.test_sections import make_pdf
    folder = add_paper(meta(), {"Intro": "latex words"})
    make_pdf(folder / "paper.pdf", ["pdf page words"])

    def boom(_):
        raise MemoryError

    monkeypatch.setattr(sections, "latex_sections", boom)
    got = sections.load_sections(folder)
    assert [(s.source, s.path) for s in got] == [("pdf", ["p. 1"])]


def test_query_time_embedding_failure_falls_back_to_bm25():
    class Flaky(FakeEmbedder):
        def embed_query(self, text):
            raise RuntimeError("boom")

    chunks = [Chunk("p#c0", "p", ["1 A"], "The automobile stopped.")]
    texts = ["T | 1 A | The automobile stopped."]
    fake = FakeEmbedder()
    index = Index(chunks, texts, fake.embed_passages(texts), Flaky())
    assert [h.chunk.id for h in index.search("automobile")] == ["p#c0"]
    assert index.mode == "bm25"


def test_a_single_huge_token_is_hard_split():
    pieces = split_text("x" * 5000)
    assert [len(p) for p in pieces] == [1400, 1400, 1400, 800]


def test_manifest_records_schema_and_a_mismatch_rebuilds(monkeypatch):
    add_paper(meta(), {"Intro": "Some text."})
    calls = []
    real = corpus.chunk_sections
    monkeypatch.setattr(corpus, "chunk_sections", lambda s: calls.append(1) or real(s))
    corpus.build()
    manifest_path = settings.data_dir / "index" / "manifest.json"
    assert json.loads(manifest_path.read_text())["schema"] == corpus.SCHEMA == 1
    corpus.build()
    assert len(calls) == 1
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = 0  # an index written by an older parser
    manifest_path.write_text(json.dumps(manifest))
    corpus.build()
    assert len(calls) == 2
    assert json.loads(manifest_path.read_text())["schema"] == 1
