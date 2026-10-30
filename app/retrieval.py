"""Hybrid search over chunks: BM25 (scaled by the best raw score) blended with cosine similarity.

Falls back to BM25 alone when there are no embeddings or no embedder is loaded.
"""
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Plus

from app.chunks import Chunk

DEFAULT_K = 8
W_EMBED, W_BM25 = 0.6, 0.4

log = logging.getLogger(__name__)
_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray: ...


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Index:
    def __init__(self, chunks: list[Chunk], texts: list[str], embeddings: np.ndarray | None = None,
                 embedder: Embedder | None = None):
        """texts[i] is the search text of chunks[i] (title and section path included); embeddings are of those texts."""
        self.chunks = chunks
        self.texts = texts
        usable = embeddings is not None and embedder is not None and len(embeddings) == len(chunks)
        self.embeddings = embeddings if usable else None
        self.embedder = embedder if usable else None
        # BM25Plus rather than Okapi: Okapi's idf is 0 for a term in half the chunks, so a small corpus would score
        # its own words 0 and never match.
        tokens = [tokenize(t) for t in texts]
        self._words = [set(t) for t in tokens]
        self._bm25 = BM25Plus(tokens) if chunks else None

    @property
    def mode(self) -> str:
        return "hybrid" if self.embeddings is not None else "bm25"

    def scores(self, query: str) -> np.ndarray:
        n = len(self.chunks)
        terms = tokenize(query)
        bm = np.asarray(self._bm25.get_scores(terms), dtype=float) if self._bm25 and terms else np.zeros(n)
        hi = bm.max() if n else 0.0
        bm_norm = bm / hi if hi > 0 else np.zeros(n)
        if self.mode == "bm25" or not query.strip():
            return bm_norm
        try:
            q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        except Exception as e:  # e.g. the model fails at query time: stay usable with BM25 alone
            log.warning("embedding the query failed (%s): BM25-only search", type(e).__name__)
            self.embeddings = self.embedder = None
            return bm_norm
        cos = self.embeddings @ (q / (np.linalg.norm(q) or 1.0))
        return W_EMBED * cos + W_BM25 * bm_norm

    def search(self, query: str, k: int = DEFAULT_K, paper: str | None = None) -> list[Hit]:
        """Top-k chunks, best first, optionally from one paper only. BM25-only search returns only chunks that contain
        a query word."""
        if not self.chunks or not query.strip() or k < 1:
            return []
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")
        if paper is not None:
            order = [i for i in order if self.chunks[i].paper_id == paper]
        order = order[:k]
        if self.mode == "bm25":
            # BM25Plus gives absent terms a constant positive score, so a score says nothing about a match
            terms = set(tokenize(query))
            order = [i for i in order if self._words[i] & terms]
        return [Hit(self.chunks[i], float(s[i])) for i in order]

    def reranked(self, query: str, k: int, reranker: Reranker, depth: int, paper: str | None = None) -> list[Hit]:
        """The first `depth` hits reordered by the reranker's score; the hit score becomes that score."""
        hits = self.search(query, max(k, depth), paper)
        if not hits:
            return []
        pos = {c.id: i for i, c in enumerate(self.chunks)}
        try:
            scores = np.asarray(reranker.score(query, [self.texts[pos[h.chunk.id]] for h in hits]), dtype=float)
        except Exception as e:  # stay usable with the first-stage order
            log.warning("reranking failed (%s): first-stage order kept", type(e).__name__)
            return hits[:k]
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(hits[i].chunk, float(scores[i])) for i in order]
