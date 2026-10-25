"""Fetch paper metadata, LaTeX source and PDFs from arXiv, politely (rate limited, identified, cached on disk)."""
import json
import logging
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
SOURCE_URL = "https://arxiv.org/e-print/{id}"
PDF_URL = "https://arxiv.org/pdf/{id}"
USER_AGENT = "arxiv-paper-rag/0.1 (+https://github.com/fredwang1012/arxiv-paper-rag)"

_NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_last_request: float | None = None
RETRY_WAIT_S = 30.0  # before the single retry after HTTP 429 or 503
RETRY_CAP_S = 120.0  # longest Retry-After we honour


@dataclass
class PaperMeta:
    id: str  # without version, e.g. "1511.06279"
    version: int
    title: str
    authors: list[str]
    year: int
    abstract: str
    primary_category: str


def _http_get(url: str) -> bytes:
    """The only place that touches the network; waits so requests are at least arxiv_delay_s apart."""
    global _last_request
    if _last_request is not None:
        wait = _last_request + settings.arxiv_delay_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    finally:
        _last_request = time.monotonic()


class FetchError(OSError):
    """arXiv answered with something that is not the file we asked for (e.g. an HTML throttle page)."""


def _get(url: str) -> bytes:
    """_http_get, retried once after HTTP 429/503 (waiting for Retry-After when given, capped)."""
    try:
        return _http_get(url)
    except urllib.error.HTTPError as e:
        if e.code not in (429, 503):
            raise
        try:
            wait = float(e.headers.get("Retry-After", "")) if e.headers is not None else RETRY_WAIT_S
        except ValueError:
            wait = RETRY_WAIT_S
        log.warning("arXiv returned HTTP %s; retrying in %.0f s", e.code, min(max(wait, 0.0), RETRY_CAP_S))
        time.sleep(min(max(wait, 0.0), RETRY_CAP_S))
        return _http_get(url)


def _text(el: ET.Element, path: str) -> str:
    found = el.find(path, _NS)
    return " ".join((found.text or "").split()) if found is not None else ""


def parse_atom(xml: bytes) -> list[PaperMeta]:
    papers = []
    for entry in ET.fromstring(xml).findall("a:entry", _NS):
        if "/api/errors" in _text(entry, "a:id") or _text(entry, "a:title") == "Error":
            raise ValueError(f"arXiv returned an error: {_text(entry, 'a:summary')}")
        # <id> looks like http://arxiv.org/abs/1511.06279v3 (or .../abs/cs/0112017v1 for old ids)
        m = re.search(r"/abs/(.+?)(?:v(\d+))?$", _text(entry, "a:id"))
        if not m:
            raise ValueError(f"unexpected arXiv entry id: {_text(entry, 'a:id')!r}")
        primary = entry.find("x:primary_category", _NS)
        papers.append(PaperMeta(
            id=m.group(1),
            version=int(m.group(2) or 1),
            title=_text(entry, "a:title"),
            authors=[_text(a, "a:name") for a in entry.findall("a:author", _NS)],
            year=int(_text(entry, "a:published")[:4]),  # first submission, which is what citations use
            abstract=_text(entry, "a:summary"),
            primary_category=primary.get("term", "") if primary is not None else "",
        ))
    return papers


def fetch_meta(ids: list[str]) -> list[PaperMeta]:
    """One batched API request; the result follows the order of ids."""
    if not ids:
        return []
    url = f"{API_URL}?id_list={','.join(ids)}&max_results={len(ids)}"
    by_id = {p.id: p for p in parse_atom(_get(url))}
    bare = [re.sub(r"v\d+$", "", i) for i in ids]  # a requested "1511.06279v3" matches the entry's versionless id
    missing = [i for i, b in zip(ids, bare) if b not in by_id]
    if missing:
        raise ValueError(f"arXiv returned no metadata for: {', '.join(missing)}")
    return [by_id[b] for b in bare]


def paper_dir(paper_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or settings.data_dir) / "papers" / paper_id.replace("/", "_")


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a .part file so an interrupted write is never mistaken for a complete file."""
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    part.replace(path)


def download(meta: PaperMeta, data_dir: Path | None = None) -> Path:
    """Save meta.json, the source archive and paper.pdf under data/papers/<id>/; files already there are kept."""
    folder = paper_dir(meta.id, data_dir)
    folder.mkdir(parents=True, exist_ok=True)

    meta_path = folder / "meta.json"
    if not meta_path.exists():
        _write_atomic(meta_path, json.dumps(asdict(meta), indent=2).encode("utf-8"))

    # source.none records that the paper has no LaTeX source, so later runs do not ask again
    if not any((folder / n).exists() for n in ("source.tar.gz", "source.gz", "source.none")):
        raw = _get(SOURCE_URL.format(id=meta.id))
        if raw[:2] == bytes([0x1F, 0x8B]):  # gzip magic number
            # a multi-file submission is a gzipped tarball, a single-file one a gzipped .tex; the tar magic
            # ("ustar" at offset 257) is within the first 512 bytes, so there is no need to unpack the rest
            try:
                head = zlib.decompressobj(wbits=31).decompress(raw, 512)
            except zlib.error as e:
                raise FetchError(f"arXiv source for {meta.id} is a corrupt gzip file") from e
            _write_atomic(folder / ("source.tar.gz" if head[257:262] == b"ustar" else "source.gz"), raw)
        elif raw.startswith(b"%PDF"):
            log.warning("arXiv source for %s is a PDF (PDF-only submission); skipping it", meta.id)
            _write_atomic(folder / "source.none", b"")
        else:  # e.g. an HTML error or throttle page: keep nothing, so a later run asks again
            raise FetchError(f"arXiv returned something other than the source for {meta.id} (throttled?)")

    pdf_path = folder / "paper.pdf"
    if not pdf_path.exists():
        _write_atomic(pdf_path, _get(PDF_URL.format(id=meta.id)))
    return folder
