"""The web app: a JSON API over the reading list and a single static page."""
import secrets
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import corpus as corpus_mod
from app.answer import Answer, ask
from app.config import settings
from app.corpus import Corpus
from app.llm import LLM, AnthropicLLM, LLMError, LLMRefusal, LLMTransient, LLMUnavailable, error_message

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="arxiv-paper-rag")


class _SettingsHosts(TrustedHostMiddleware):
    """TrustedHostMiddleware whose allowed hosts follow settings.allowed_hosts at request time, not import time."""

    def __init__(self, app, **kwargs) -> None:
        super().__init__(app, allowed_hosts=settings.allowed_hosts, **kwargs)
        self._hosts = tuple(settings.allowed_hosts)

    async def __call__(self, scope, receive, send) -> None:
        if tuple(settings.allowed_hosts) != self._hosts:
            self.__init__(self.app, www_redirect=self.www_redirect)
        await super().__call__(scope, receive, send)


# a DNS-rebinding page reaches this server under its own hostname; refuse any Host that is not listed
app.add_middleware(_SettingsHosts)
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
    verify: bool = True  # also check each citation supports its sentence (a second model call; off when VERIFY=false)


def answer_json(a: Answer) -> dict:
    return {
        "text": a.text,
        "references": [r.__dict__ for r in a.references()],
        "citations": [c.__dict__ for c in a.citations],
        "unresolved_markers": a.unresolved_markers,
        "unmarked_citations": a.unmarked_citations,
        "support": [s.__dict__ for s in a.support],
        "verify_error": a.verify_error,
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


_DEFAULT_PORTS = {"http": ":80", "https": ":443"}


def _origin_key(scheme: str, netloc: str) -> str:
    netloc = netloc.lower()
    default = _DEFAULT_PORTS.get(scheme)
    if default and netloc.endswith(default):
        netloc = netloc[:-len(default)]
    return f"{scheme}://{netloc}"


def guard_ask(request: Request, x_api_token: str | None = Header(default=None)) -> None:
    """Refuse cross-site browser requests, then require the shared token when one is configured."""
    origin = request.headers.get("origin")
    if origin is not None:
        scheme, sep, netloc = origin.partition("://")
        same = sep and _origin_key(scheme.lower(), netloc) == _origin_key(request.url.scheme, request.headers.get("host", ""))
        if not same:
            raise HTTPException(403, "Cross-origin requests are not allowed.")
    if settings.api_token is not None:
        expected = settings.api_token.get_secret_value().encode()
        if x_api_token is None or not secrets.compare_digest(x_api_token.encode(), expected):
            raise HTTPException(401, "A valid X-API-Token header is required.")


@app.post("/api/ask", dependencies=[Depends(guard_ask)])
def ask_endpoint(req: AskRequest) -> dict:
    try:
        return answer_json(ask(req.question, get_corpus(), get_llm(), req.k, verify=settings.verify and req.verify))
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
