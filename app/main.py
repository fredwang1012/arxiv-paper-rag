"""The web app: a JSON API over the reading list and a single static page."""
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app import corpus as corpus_mod
from app.answer import Answer, ask
from app.corpus import Corpus
from app.llm import LLM, AnthropicLLM, LLMError, LLMRefusal, LLMTransient, LLMUnavailable, error_message

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="arxiv-paper-rag")
_lock = threading.Lock()
_corpus: Corpus | None = None
_llm: LLM | None = None


def get_corpus() -> Corpus:
    """Built on first use, once, even when requests arrive together; rebuilt while it is empty (papers fetched since)."""
    global _corpus
    with _lock:
        if _corpus is None or not _corpus.chunks:
            _corpus = corpus_mod.build()
        return _corpus


def get_llm() -> LLM:
    global _llm
    with _lock:
        if _llm is None:
            _llm = AnthropicLLM()
        return _llm


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=8, ge=1, le=20)


def answer_json(a: Answer) -> dict:
    return {
        "text": a.text,
        "references": [r.__dict__ for r in a.references()],
        "citations": [c.__dict__ for c in a.citations],
        "unresolved_markers": a.unresolved_markers,
        "unmarked_citations": a.unmarked_citations,
    }


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Field locations and messages only: the default body echoes the submitted input."""
    detail = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    return JSONResponse({"error": "invalid_request", "detail": detail}, status_code=422)


@app.get("/api/papers")
def papers() -> list[dict]:
    c = get_corpus()
    counts: dict[str, int] = {}
    for ch in c.chunks:
        counts[ch.paper_id] = counts.get(ch.paper_id, 0) + 1
    return [{"id": m.id, "title": m.title, "authors": m.authors, "year": m.year, "chunks": counts.get(m.id, 0)}
            for m in c.metas.values()]


@app.post("/api/ask")
def ask_endpoint(req: AskRequest) -> dict:
    try:
        return answer_json(ask(req.question, get_corpus(), get_llm(), req.k))
    except LLMUnavailable as e:
        raise HTTPException(503, error_message(e))
    except LLMTransient as e:
        raise HTTPException(503, error_message(e))
    except LLMRefusal as e:
        return JSONResponse({"error": "refused", "detail": error_message(e)}, status_code=422)
    except LLMError as e:
        raise HTTPException(502, error_message(e))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
