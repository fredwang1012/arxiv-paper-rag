"""Shared fixtures: every test gets its own data folder and no real delays."""
import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "embed", "off")
    monkeypatch.setattr(settings, "arxiv_delay_s", 0.0)
