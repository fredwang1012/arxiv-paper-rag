"""Ask a question: search, number the passages, call the LLM, and map its citations back to papers and sections."""
import re
from dataclasses import dataclass, field

from app.corpus import Corpus
from app.llm import LLM, Passage, RawAnswer

# [1], [1, 2], [1-3] (hyphen or en dash)
_MARKER = re.compile(r"\[\s*(\d+(?:\s*[,–-]\s*\d+)*)\s*\]")
_MAX_RANGE = 50  # a marker like [1-99999] must not expand into a huge set
NOTHING_FOUND = "I couldn't find anything about that in the reading list."
NO_PAPERS = "No papers indexed yet — run `python -m app.cli fetch` then `index`."
_PARTICLES = {"de", "van", "von", "di", "da", "le"}
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


@dataclass
class Citation:
    paper_id: str
    title: str
    authors: list[str]
    year: int
    section: str
    chunk_id: str
    quote: str
    quote_found: bool
    passage: int  # the 1-based passage number it came from


@dataclass
class Reference:
    """A distinct (paper, section) that was cited, with every passage number that points at it."""
    paper_id: str
    title: str
    authors: list[str]
    year: int
    section: str
    passages: list[int]


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    # [n] markers in the text whose n is out of range or was never cited; they stay in the text
    unresolved_markers: list[int] = field(default_factory=list)
    # passage numbers that were cited but never appear as a marker in the text
    unmarked_citations: list[int] = field(default_factory=list)

    def references(self) -> list[Reference]:
        """One entry per distinct (paper, section), in the order first cited."""
        out: dict[tuple[str, str], Reference] = {}
        for c in self.citations:
            ref = out.setdefault((c.paper_id, c.section),
                                 Reference(c.paper_id, c.title, c.authors, c.year, c.section, []))
            if c.passage not in ref.passages:
                ref.passages.append(c.passage)
        return list(out.values())


def _normalise(text: str) -> str:
    return _EDGE_PUNCT.sub("", " ".join(text.lower().split()))


def quote_in(quote: str, text: str) -> bool:
    q = _normalise(quote)
    return bool(q) and q in _normalise(text)


def surname(name: str) -> str:
    """Last word of a name, with lower-case particles kept: "Nando de Freitas" -> "de Freitas"."""
    words = name.split()
    if not words:
        return ""
    i = len(words) - 1
    while i > 0 and words[i - 1].lower() in _PARTICLES:
        i -= 1
    return " ".join(words[i:])


def _surname(authors: list[str]) -> str:
    return surname(authors[0]) or "Unknown" if authors else "Unknown"


def marker_numbers(text: str) -> set[int]:
    """Every passage number the text points at with [n], [n, m] or [n-m]."""
    out: set[int] = set()
    for body in _MARKER.findall(text):
        for part in body.split(","):
            ends = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
            if len(ends) == 2 and 0 <= ends[1] - ends[0] <= _MAX_RANGE:
                out.update(range(ends[0], ends[1] + 1))
            else:
                out.update(ends)
    return out


def build_passages(question: str, corpus: Corpus, k: int) -> list[Passage]:
    passages = []
    for n, hit in enumerate(corpus.search(question, k), start=1):
        meta = corpus.paper(hit.chunk.paper_id)
        passages.append(Passage(n, hit.chunk, meta.title, _surname(meta.authors), meta.year,
                                " > ".join(hit.chunk.section_path)))
    return passages


def map_citations(raw: RawAnswer, passages: list[Passage], corpus: Corpus) -> list[Citation]:
    out = []
    for rc in raw.citations:
        if not 1 <= rc.passage <= len(passages):
            continue
        p = passages[rc.passage - 1]
        meta = corpus.paper(p.chunk.paper_id)
        out.append(Citation(meta.id, meta.title, meta.authors, meta.year, p.section, p.chunk.id, rc.quote,
                            quote_in(rc.quote, p.chunk.text), rc.passage))
    return out


def ask(question: str, corpus: Corpus, llm: LLM, k: int = 8) -> Answer:
    if not corpus.chunks:
        return Answer(NO_PAPERS)
    passages = build_passages(question, corpus, k)
    if not passages:
        return Answer(NOTHING_FOUND)
    raw = llm.answer(question, passages)
    citations = map_citations(raw, passages, corpus)
    cited = {c.passage for c in citations}
    marked = marker_numbers(raw.answer)
    return Answer(raw.answer, citations, passages, sorted(marked - cited), sorted(cited - marked))
