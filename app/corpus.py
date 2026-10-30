"""The searchable reading list: every downloaded paper chunked and indexed, cached under data/index/."""
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.arxiv import PaperMeta
from app.chunks import Chunk, chunk_sections, search_text
from app.config import settings
from app.embedder import get_embedder
from app.retrieval import DEFAULT_K, Hit, Index
from app.sections import load_sections

log = logging.getLogger(__name__)

SCHEMA = 1  # bump when the parser or chunker output changes, so cached chunks are rebuilt


class Corpus:
    def __init__(self, metas: dict[str, PaperMeta], index: Index):
        self.metas = metas
        self.index = index

    @property
    def chunks(self) -> list[Chunk]:
        return self.index.chunks

    def search(self, query: str, k: int = DEFAULT_K) -> list[Hit]:
        return self.index.search(query, k)

    def paper(self, paper_id: str) -> PaperMeta:
        return self.metas[paper_id]


def _paper_folders(data_dir: Path) -> list[Path]:
    """Downloaded papers (folders with a meta.json), sorted by folder name."""
    root = data_dir / "papers"
    return sorted(p for p in root.iterdir() if (p / "meta.json").is_file()) if root.is_dir() else []


def _read_meta(folder: Path) -> PaperMeta:
    return PaperMeta(**json.loads((folder / "meta.json").read_text(encoding="utf-8")))


def _fingerprint(folders: list[Path]) -> dict[str, list]:
    """Folder name -> [file name, size, mtime] for each file, so any change to a paper's files shows up."""
    out = {}
    for folder in folders:
        files = sorted(f for f in folder.iterdir() if f.is_file())
        out[folder.name] = [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]
    return out


def _load_cached(index_dir: Path, manifest: dict) -> tuple[list[Chunk], np.ndarray | None] | None:
    try:
        if json.loads((index_dir / "manifest.json").read_text(encoding="utf-8")) != manifest:
            return None
        chunks = [Chunk(**c) for c in json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))]
        embeddings = np.load(index_dir / "embeddings.npy") if manifest["embed"] == "e5" else None
    except (OSError, ValueError, TypeError, KeyError):  # missing, unreadable or from an older layout
        return None
    return chunks, embeddings


def _save_cache(index_dir: Path, manifest: dict, chunks: list[Chunk], embeddings: np.ndarray | None) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "manifest.json").unlink(missing_ok=True)  # written last, so a half-written cache is never trusted
    (index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks]), encoding="utf-8")
    if embeddings is not None:
        np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def build(data_dir: Path | None = None) -> Corpus:
    """Load the downloaded papers, reusing the cached chunks and embeddings when nothing has changed."""
    data_dir = data_dir or settings.data_dir
    folders = _paper_folders(data_dir)
    metas = {m.id: m for m in map(_read_meta, folders)}
    embedder = get_embedder()
    manifest = {"schema": SCHEMA, "papers": sorted(metas), "files": _fingerprint(folders), "embed": "e5" if embedder else "off"}
    index_dir = data_dir / "index"

    cached = _load_cached(index_dir, manifest)
    if cached:
        chunks, embeddings = cached
    else:
        chunks = []
        for folder in folders:
            try:
                chunks += chunk_sections(load_sections(folder))
            except Exception as e:  # skip the paper, keep the rest of the list searchable
                log.warning("skipping paper %s: %s", folder.name, type(e).__name__)
        embeddings = None
    titles = {m.id: m.title for m in metas.values()}
    texts = [search_text(c, titles.get(c.paper_id, "")) for c in chunks]

    if not cached:
        if embedder and chunks:
            try:
                embeddings = np.asarray(embedder.embed_passages(texts), dtype=np.float32)
            except Exception as e:  # the cache then records bm25 so a later run tries again
                log.warning("embedding the chunks failed (%s): BM25-only search", e)
                embedder, manifest["embed"] = None, "off"
        elif embedder:
            manifest["embed"] = "off"
        _save_cache(index_dir, manifest, chunks, embeddings)
    return Corpus(metas, Index(chunks, texts, embeddings, embedder))
