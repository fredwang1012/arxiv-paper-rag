"""Shared fixtures: every test gets its own data folder and no real delays."""
import pytest

from app import evaluate, reranker
from app.config import settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "embed", "off")
    monkeypatch.setattr(settings, "rerank", "off")
    # the eval forces the reranker on; never let a test load (or download) a real model
    monkeypatch.setattr(evaluate, "get_reranker", lambda force=False: None)
    monkeypatch.setattr(reranker, "_loaded", None)
    monkeypatch.setattr(reranker, "_failed", False)
    monkeypatch.setattr(settings, "arxiv_delay_s", 0.0)
    # TestClient talks to the host "testserver"
    monkeypatch.setattr(settings, "allowed_hosts", ["testserver", "localhost", "127.0.0.1", "[::1]"])
    monkeypatch.setattr(settings, "api_token", None)
