"""Agentic retrieval: a first search on the question, then follow-up searches the model chooses, pooled into one
numbered passage list for the answer."""
from app.corpus import Corpus
from app.llm import LLM, Passage, render_results, research_message
from app.retrieval import Hit

PER_SEARCH = 5  # passages returned to the model per search
MAX_PASSAGES = 16  # pooled passages sent to the answer step
MAX_SEARCHES = 8  # tool calls served per question; later ones are told the budget is spent


class Pool:
    """Passages found so far, numbered in the order they were first found, without duplicates."""

    def __init__(self, corpus: Corpus, make_passage):
        self.corpus, self._make = corpus, make_passage
        self.passages: list[Passage] = []
        self._seen: set[str] = set()
        self.searches: list[str] = []  # queries the model ran, "paper: query" when it named a paper

    def add(self, hits: list[Hit]) -> list[Passage]:
        """Add new hits (up to MAX_PASSAGES in total); returns every hit as a passage, old ones with their number."""
        shown = []
        for hit in hits:
            if hit.chunk.id in self._seen:
                shown.append(next(p for p in self.passages if p.chunk.id == hit.chunk.id))
            elif len(self.passages) < MAX_PASSAGES:
                self._seen.add(hit.chunk.id)
                self.passages.append(self._make(len(self.passages) + 1, hit))
                shown.append(self.passages[-1])
        return shown

    def search(self, query: str, paper: str | None) -> str:
        if len(self.searches) >= MAX_SEARCHES:
            return "Search budget spent: answer with the passages found so far."
        if not isinstance(query, str) or not query.strip():
            return "Empty query: give some words to search for."
        if paper is not None and paper not in self.corpus.metas:
            return f"Unknown paper id {paper!r}."
        self.searches.append(f"{paper}: {query}" if paper else query)
        return render_results(self.add(self.corpus.search(query, PER_SEARCH, paper)))


def gather(question: str, corpus: Corpus, llm: LLM, k: int, rounds: int, make_passage) -> tuple[list[Passage], list[str]]:
    """The question's own top k, then whatever the model's searches add; returns the passages and the queries run."""
    pool = Pool(corpus, make_passage)
    first = pool.add(corpus.search(question, k))
    papers = {pid: m.title for pid, m in corpus.metas.items()}
    llm.research(research_message(question, papers, render_results(first)), papers, pool.search, rounds)
    return pool.passages, pool.searches
