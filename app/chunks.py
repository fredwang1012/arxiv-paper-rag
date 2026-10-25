"""Cut each section into chunks of about 900 characters (never more than 1400) at sentence boundaries."""
import re
from dataclasses import dataclass

from app.sections import Section

TARGET_CHARS = 900
MAX_CHARS = 1400

# a sentence ends at . ! ? followed by whitespace and a capital letter, "(" or "["
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass
class Chunk:
    id: str  # "<paper_id>#c<n>", n counting from 0 within the paper
    paper_id: str
    section_path: list[str]
    text: str


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Join consecutive pieces with spaces, starting a new string when the next piece would pass the limit."""
    out: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > limit:
            out.append(current)
            current = piece
        else:
            current = f"{current} {piece}" if current else piece
    if current:
        out.append(current)
    return out


def split_text(text: str) -> list[str]:
    """Sentences packed up to about TARGET_CHARS; a sentence over MAX_CHARS is split on whitespace first."""
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if len(sentence) > MAX_CHARS:
            words = [w[i:i + MAX_CHARS] for w in sentence.split() for i in range(0, len(w), MAX_CHARS)]
            pieces += _pack(words, TARGET_CHARS)
        elif sentence:
            pieces.append(sentence)
    return _pack(pieces, TARGET_CHARS)


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """Chunks for one paper's sections, numbered in reading order."""
    chunks: list[Chunk] = []
    for section in sections:
        for text in split_text(section.text):
            chunks.append(Chunk(f"{section.paper_id}#c{len(chunks)}", section.paper_id, section.path, text))
    return chunks


def search_text(chunk: Chunk, title: str) -> str:
    """What is indexed: the title and section path in front of the chunk, so section names are searchable."""
    return f"{title} | {' > '.join(chunk.section_path)} | {chunk.text}"
