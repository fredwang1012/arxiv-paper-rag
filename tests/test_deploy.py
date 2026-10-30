"""Spend limits on /api/ask and the guarded production entry point."""
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import main, serve
from app.config import settings
from app.ratelimit import DAY, HOUR, RateLimiter
from tests.test_answer import make_corpus
from tests.test_surfaces import good_llm


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_per_client_hourly_limit_slides():
    clock = Clock()
    rl = RateLimiter(per_client_hour=2, per_day=0, clock=clock)
    assert rl.check("a") is None and rl.check("a") is None
    assert rl.check("a") == pytest.approx(HOUR)
    assert rl.check("b") is None  # other clients are counted apart
    clock.t += HOUR - 10
    assert rl.check("a") == pytest.approx(10)
    clock.t += 10
    assert rl.check("a") is None


def test_daily_limit_covers_every_client():
    clock = Clock()
    rl = RateLimiter(per_client_hour=0, per_day=3, clock=clock)
    assert [rl.check(c) for c in "abc"] == [None, None, None]
    assert rl.check("d") == pytest.approx(DAY)
    clock.t += DAY
    assert rl.check("d") is None


def test_refused_questions_are_not_counted():
    clock = Clock()
    rl = RateLimiter(per_client_hour=1, per_day=2, clock=clock)
    assert rl.check("a") is None
    for _ in range(5):
        assert rl.check("a") is not None
    assert rl.check("b") is None  # a's refusals did not use up the daily budget


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "_corpus", None)
    monkeypatch.setattr(main, "_limiter", None)
    monkeypatch.setattr(main, "_llm", good_llm())
    make_corpus()
    return TestClient(main.app, base_url="http://localhost")


def test_api_answers_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "ask_limit_per_hour", 1)
    ok = client.post("/api/ask", json={"question": "widgets hum", "verify": False})
    assert ok.status_code == 200
    r = client.post("/api/ask", json={"question": "widgets hum", "verify": False})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 3000
    assert client.get("/api/papers").status_code == 200  # only questions are limited


def test_bad_token_is_refused_before_it_counts(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", SecretStr("t"))
    monkeypatch.setattr(settings, "ask_limit_per_hour", 1)
    for _ in range(3):
        assert client.post("/api/ask", json={"question": "widgets"}).status_code == 401
    r = client.post("/api/ask", json={"question": "widgets hum", "verify": False}, headers={"X-API-Token": "t"})
    assert r.status_code == 200


def test_no_limits_by_default(client):
    for _ in range(5):
        assert client.post("/api/ask", json={"question": "widgets hum", "verify": False}).status_code == 200


def test_serve_refuses_without_a_spend_guard_or_public_host(monkeypatch, capsys):
    monkeypatch.setattr(settings, "allowed_hosts", serve.DEFAULT_HOSTS)
    assert serve.main() == 1
    err = capsys.readouterr().err
    assert "API_TOKEN" in err and "ALLOWED_HOSTS" in err
    monkeypatch.setattr(settings, "allowed_hosts", ["rag.example.org"])
    monkeypatch.setattr(settings, "ask_limit_per_hour", 10)
    assert serve.problems() and "ASK_LIMIT_PER_DAY" in serve.problems()[0]  # one limit alone is not enough
    monkeypatch.setattr(settings, "ask_limit_per_day", 200)
    assert serve.problems() == []
    monkeypatch.setattr(settings, "ask_limit_per_hour", 0)
    monkeypatch.setattr(settings, "api_token", SecretStr("t"))
    assert serve.problems() == []


def test_serve_starts_uvicorn_on_all_interfaces(monkeypatch):
    import uvicorn
    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    monkeypatch.setattr(settings, "allowed_hosts", ["rag.example.org"])
    monkeypatch.setattr(settings, "api_token", SecretStr("t"))
    monkeypatch.setenv("PORT", "9000")
    assert serve.main() == 0
    assert started["app"] == "app.main:app" and started["host"] == "0.0.0.0" and started["port"] == 9000
    assert started["proxy_headers"] is True
