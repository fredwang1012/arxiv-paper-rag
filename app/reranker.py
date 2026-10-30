"""Cross-encoder reranking of the first-stage candidates. Imported lazily, like the embedder; off by default."""
import logging
import threading

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # heavy import: only when actually used
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, t) for t in texts]), dtype=float)


_loaded: CrossEncoderReranker | None = None
_failed = False
_lock = threading.Lock()


def get_reranker(force: bool = False) -> CrossEncoderReranker | None:
    """The shared reranker when RERANK=auto (or force), loaded on first use; None when off or it cannot be loaded."""
    global _loaded, _failed
    if settings.rerank == "off" and not force:
        return None
    with _lock:
        if _failed:
            return None
        if _loaded is None:
            try:
                _loaded = CrossEncoderReranker(settings.rerank_model)
            except ImportError:
                log.info("sentence-transformers not installed (uv sync --extra embed): no reranking")
                _failed = True
            except Exception as e:  # e.g. the model cannot be downloaded offline
                log.warning("could not load %s (%s): no reranking", settings.rerank_model, e)
                _failed = True
        return _loaded
