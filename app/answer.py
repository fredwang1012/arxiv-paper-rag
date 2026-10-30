"""Ask a question: search, number the passages, call the LLM, map its citations back to papers and sections, and
optionally check that each cited passage supports the sentence that cites it."""
import re
from dataclasses import dataclass, field

from app.corpus import Corpus
from app.llm import LLM, ClaimCheck, LLMError, Passage, RawAnswer, RawVerification, error_message

# [1], [1, 2], [1-3] (hyphen or en dash)
_MARKER = re.compile(r"\[\s*(\d+(?:\s*[,–-]\s*\d+)*)\s*\]")
_MAX_RANGE = 50  # a marker like [1-99999] must not expand into a huge set
NOTHING_FOUND = "I couldn't find anything about that in the reading list."
NO_PAPERS = "No papers indexed yet — run `python -m app.cli fetch` then `index`."
_PARTICLES = {"de", "van", "von", "di", "da", "le"}
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")
# a sentence ends at . ! or ? before whitespace and a capital, digit or opening mark; always after a marker and a
# full stop ("... [2]. next"); and after a trailing marker before a capital ("... stops. [1] Then")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'*\-•<])|(?<=\][.!?])\s+|(?<=\])\s+(?=[A-Z])|\n+")
_ONLY_MARKERS = re.compile(r"^(?:\s*\[[\d\s,–-]+\]\s*[.,;:]?)+$")
MAX_CHECKS = 40  # claim-passage pairs per answer: bounds the cost of the check
UNCHECKED = "unchecked"


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
class Support:
    """Whether one cited passage supports the sentence that cites it."""
    claim: str
    passage: int
    verdict: str  # supported, partial, unsupported, or unchecked when the check gave no verdict for this pair
    reason: str = ""


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    # [n] markers in the text whose n is out of range or was never cited; they stay in the text
    unresolved_markers: list[int] = field(default_factory=list)
    # passage numbers that were cited but never appear as a marker in the text
    unmarked_citations: list[int] = field(default_factory=list)
    # one entry per (sentence, passage it marks), in text order; empty when the check was not run
    support: list[Support] = field(default_factory=list)
    verify_error: str | None = None  # why the check failed; the answer itself is still good

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


def split_claims(text: str, n_passages: int) -> list[ClaimCheck]:
    """Sentences (or lines) of the answer that mark at least one existing passage, with the passages they mark."""
    parts = [x.strip() for x in _SENTENCE_END.split(text) if x and x.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and _ONLY_MARKERS.match(part):  # "... stops. [1]": the marker belongs to the sentence before
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    claims = []
    for part in merged:
        nums = sorted(n for n in marker_numbers(part) if 1 <= n <= n_passages)
        if nums:
            claims.append(ClaimCheck(len(claims) + 1, part, nums))
    return claims


def _limit(claims: list[ClaimCheck], budget: int) -> list[ClaimCheck]:
    out = []
    for c in claims:
        if budget <= 0:
            break
        out.append(ClaimCheck(c.number, c.text, c.passages[:budget]))
        budget -= len(out[-1].passages)
    return out


def map_support(claims: list[ClaimCheck], raw: RawVerification) -> list[Support]:
    """One Support per (claim, passage) asked about; a missing or out-of-range verdict leaves the pair unchecked."""
    got = {(v.claim, v.passage): v for v in raw.verdicts}
    out = []
    for c in claims:
        for n in c.passages:
            v = got.get((c.number, n))
            out.append(Support(c.text, n, v.verdict, v.reason) if v else Support(c.text, n, UNCHECKED))
    return out


def check_support(answer: Answer, llm: LLM) -> None:
    """Fill answer.support; an LLM failure is recorded in answer.verify_error rather than raised."""
    claims = _limit(split_claims(answer.text, len(answer.passages)), MAX_CHECKS)
    if not claims:
        return
    try:
        raw = llm.verify(claims, answer.passages)
    except LLMError as e:
        answer.verify_error = error_message(e)
        return
    answer.support = map_support(claims, raw)


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


def ask(question: str, corpus: Corpus, llm: LLM, k: int = 8, verify: bool = False) -> Answer:
    if not corpus.chunks:
        return Answer(NO_PAPERS)
    passages = build_passages(question, corpus, k)
    if not passages:
        return Answer(NOTHING_FOUND)
    raw = llm.answer(question, passages)
    citations = map_citations(raw, passages, corpus)
    cited = {c.passage for c in citations}
    marked = marker_numbers(raw.answer)
    answer = Answer(raw.answer, citations, passages, sorted(marked - cited), sorted(cited - marked))
    if verify:
        check_support(answer, llm)
    return answer
