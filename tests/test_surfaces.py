"""The CLI and the web API, with a fake LLM and no network."""
import gzip
import json
import re
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr
from fastapi.testclient import TestClient

from app import arxiv, cli, main
from app import llm as llm_mod
from app.answer import ask
from app.config import Settings, settings
from app.llm import (AnthropicLLM, LLMError, LLMRefusal, LLMTransient, LLMUnavailable, RawAnswer, RawCitation,
                     RawVerdict, RawVerification)
from tests.test_answer import FakeLLM, make_corpus


class RaisingLLM:
    def __init__(self, exc):
        self.exc = exc

    def answer(self, question, passages):
        raise self.exc


@pytest.fixture(autouse=True)
def _reset_main(monkeypatch):
    monkeypatch.setattr(main, "_corpus", None)
    monkeypatch.setattr(main, "_llm", None)


def good_llm():
    return FakeLLM(RawAnswer(answer="Widgets hum [1] and fly [2].", citations=[
        RawCitation(passage=1, quote="They hum quietly"), RawCitation(passage=1, quote="made up words")]))


def test_cli_ask_prints_answer_references_and_warnings(capsys):
    make_corpus()
    assert cli.main(["ask", "widgets hum"], llm=good_llm()) == 0
    out = capsys.readouterr().out
    assert "Widgets hum [1] and fly [2]." in out
    assert "References" in out
    assert re.search(r"\[1\] Author 2020, Neural Widgets, §", out)
    assert '"They hum quietly"' in out and "⚠" not in out
    assert "Warning" in out and "[2]" in out.split("Warning")[1]


def test_cli_ask_flags_missing_quote(capsys):
    make_corpus()
    llm = FakeLLM(RawAnswer(answer="x [1]", citations=[RawCitation(passage=1, quote="not there")]))
    cli.main(["ask", "widgets"], llm=llm)
    assert "⚠" in capsys.readouterr().out


@pytest.mark.parametrize("exc,word", [
    (LLMUnavailable("x"), "ANTHROPIC_API_KEY"), (LLMTransient("x"), "try again"),
    (LLMRefusal("x"), "declined"), (LLMError("boom"), "boom")])
def test_cli_ask_maps_errors(capsys, exc, word):
    make_corpus()
    assert cli.main(["ask", "widgets"], llm=RaisingLLM(exc)) == 1
    err = capsys.readouterr().err
    assert word in err and len(err.strip().splitlines()) == 1


def test_cli_papers_and_index(capsys):
    make_corpus()
    assert cli.main(["index"]) == 0 and "Indexed" in capsys.readouterr().out
    assert cli.main(["papers"]) == 0
    assert "1111.0001" in capsys.readouterr().out


def test_cli_papers_empty(capsys):
    assert cli.main(["papers"]) == 0
    assert "No papers" in capsys.readouterr().out


def test_reading_list_has_five_ids():
    assert len(cli.reading_list_ids()) == 5


def client(llm, monkeypatch):
    monkeypatch.setattr(main, "_llm", llm)
    return TestClient(main.app)


def test_api_papers(monkeypatch):
    make_corpus()
    (p,) = client(good_llm(), monkeypatch).get("/api/papers").json()
    assert p["id"] == "1111.0001" and p["chunks"] >= 1


def test_api_ask(monkeypatch):
    make_corpus()
    r = client(good_llm(), monkeypatch).post("/api/ask", json={"question": "widgets hum", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"text", "references", "citations", "unresolved_markers", "unmarked_citations", "support",
                         "verify_error"}
    assert body["references"][0]["passages"] == [1] and body["unresolved_markers"] == [2]
    assert body["citations"][0]["quote_found"] is True
    # [2] is marked but not in the model's citations: it is still checked, since passage 2 exists
    assert [(s["passage"], s["verdict"]) for s in body["support"]] == [(1, "supported"), (2, "supported")]
    assert body["support"][0]["claim"] == "Widgets hum [1] and fly [2]."
    assert body["verify_error"] is None


@pytest.mark.parametrize("payload", [
    {"question": ""}, {"question": "x" * 1001}, {"question": "ok", "k": 0}, {"question": "ok", "k": 21}, {}])
def test_api_validation(monkeypatch, payload):
    make_corpus()
    assert client(good_llm(), monkeypatch).post("/api/ask", json=payload).status_code == 422


@pytest.mark.parametrize("exc,status,word", [
    (LLMUnavailable("x"), 503, "ANTHROPIC_API_KEY"), (LLMTransient("x"), 503, "try again"),
    (LLMError("boom"), 502, "boom")])
def test_api_error_mapping(monkeypatch, exc, status, word):
    make_corpus()
    r = client(RaisingLLM(exc), monkeypatch).post("/api/ask", json={"question": "widgets"})
    assert r.status_code == status and word in r.json()["detail"]


def test_corpus_built_once(monkeypatch):
    make_corpus()
    calls = []
    real = main.corpus_mod.build
    monkeypatch.setattr(main.corpus_mod, "build", lambda: calls.append(1) or real())
    c = client(good_llm(), monkeypatch)
    c.get("/api/papers")
    c.get("/api/papers")
    assert len(calls) == 1


def test_page_served_and_never_uses_innerhtml(monkeypatch):
    r = client(good_llm(), monkeypatch).get("/")
    assert r.status_code == 200 and "<textarea" in r.text
    html = (Path(main.STATIC) / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", html, re.IGNORECASE)
    assert not re.search(r"<script[^>]+src=", html, re.IGNORECASE)
    assert "prefers-color-scheme: dark" in html and "textContent" in html


# ---------- credentials, fetch failures, empty corpus, validation ----------

@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    """The real SDK client, no key, no profile, pointed at a dead local port: nothing can leave the machine."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    empty = tmp_path / "empty-home"
    empty.mkdir()
    for var in ("ANTHROPIC_CONFIG_DIR", "APPDATA", "HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(empty))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(settings, "anthropic_api_key", None)


def ask_with(llm):
    return ask("widgets hum", make_corpus(), llm)


def test_real_sdk_without_credentials_is_llm_unavailable(no_credentials):
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        ask_with(AnthropicLLM())


def test_cli_without_credentials_prints_friendly_line(no_credentials, capsys):
    make_corpus()
    assert cli.main(["ask", "widgets hum"]) == 1
    err = capsys.readouterr().err
    assert "No Anthropic credentials found" in err
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_api_without_credentials_is_503(no_credentials, monkeypatch):
    make_corpus()
    r = client(AnthropicLLM(), monkeypatch).post("/api/ask", json={"question": "widgets hum"})
    assert r.status_code == 503 and "No Anthropic credentials" in r.json()["detail"]


def test_other_type_errors_are_not_swallowed():
    def parse(**kwargs):
        raise TypeError("something else")

    llm = AnthropicLLM()
    llm._client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    with pytest.raises(TypeError, match="something else"):
        ask_with(llm)


def test_error_message_lives_in_llm():
    assert cli.error_message is llm_mod.error_message and main.error_message is llm_mod.error_message


# fetch

def write_meta(paper_id):
    folder = arxiv.paper_dir(paper_id)
    folder.mkdir(parents=True)
    m = arxiv.PaperMeta(paper_id, 1, "T " + paper_id, ["A B"], 2020, "abs", "cs.LG")
    (folder / "meta.json").write_text(json.dumps(m.__dict__))
    return m


@pytest.fixture
def two_ids(monkeypatch):
    monkeypatch.setattr(cli, "reading_list_ids", lambda: ["1111.0001", "1111.0002"])


def test_fetch_with_everything_cached_makes_no_request(two_ids, monkeypatch, capsys):
    for i in ("1111.0001", "1111.0002"):
        write_meta(i)
        folder = arxiv.paper_dir(i)
        (folder / "source.none").write_bytes(b"")
        (folder / "paper.pdf").write_bytes(b"%PDF")

    def fail(url):
        raise AssertionError("network used")

    monkeypatch.setattr(arxiv, "_http_get", fail)
    assert cli.main(["fetch"]) == 0
    assert "Fetched 2 papers" in capsys.readouterr().out


@pytest.mark.parametrize("exc", [
    urllib.error.HTTPError("u", 429, "Too Many Requests", None, None), urllib.error.URLError("no route"),
    OSError("reset"), EOFError(), gzip.BadGzipFile("bad"), TimeoutError("slow")])
def test_fetch_failures_are_one_line_exit_1(two_ids, monkeypatch, capsys, exc):
    write_meta("1111.0001")  # already saved: kept
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)

    def broken(url):
        raise exc

    monkeypatch.setattr(arxiv, "_http_get", broken)
    assert cli.main(["fetch"]) == 1
    err = capsys.readouterr().err
    assert "arXiv unreachable or throttling" in err and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert (arxiv.paper_dir("1111.0001") / "meta.json").exists()


def test_fetch_asks_only_for_missing_metadata(two_ids, monkeypatch, capsys):
    write_meta("1111.0001")
    requested = []
    new = arxiv.PaperMeta("1111.0002", 1, "New", ["C D"], 2021, "a", "cs.LG")

    def fake_meta(ids):
        requested.append(ids)
        return [new]

    monkeypatch.setattr(arxiv, "fetch_meta", fake_meta)
    monkeypatch.setattr(arxiv, "download", lambda m: None)
    assert cli.main(["fetch"]) == 0
    assert requested == [["1111.0002"]]
    assert "T 1111.0001" in capsys.readouterr().out


# -k, empty corpus

@pytest.mark.parametrize("k", ["0", "21", "-1"])
def test_cli_k_must_be_1_to_20(capsys, k):
    with pytest.raises(SystemExit):
        cli.main(["ask", "q", "-k", k])
    assert "invalid choice" in capsys.readouterr().err


def test_search_with_k_below_one_is_empty():
    c = make_corpus()
    assert c.search("widgets", 0) == [] and c.search("widgets", -3) == []


def test_cli_ask_and_index_on_empty_corpus(capsys):
    llm = RaisingLLM(AssertionError("must not be called"))
    assert cli.main(["ask", "widgets"], llm=llm) == 0
    assert "No papers indexed yet" in capsys.readouterr().out
    assert cli.main(["index"]) == 0
    assert "No papers indexed yet" in capsys.readouterr().out


def test_api_ask_empty_corpus_then_corpus_rebuilds(monkeypatch):
    c = client(good_llm(), monkeypatch)
    r = c.post("/api/ask", json={"question": "widgets hum"})
    assert r.status_code == 200 and r.json()["text"].startswith("No papers indexed yet")
    make_corpus()  # papers arrive while the server runs
    r = c.post("/api/ask", json={"question": "widgets hum"})
    assert "Widgets hum" in r.json()["text"]


# typed error bodies

def test_validation_error_body_has_locations_not_input(monkeypatch):
    make_corpus()
    r = client(good_llm(), monkeypatch).post("/api/ask", json={"question": "secret text", "k": 99})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_request"
    assert body["detail"][0]["loc"] == ["body", "k"] and body["detail"][0]["msg"]
    assert "secret text" not in r.text and "input" not in body["detail"][0]


def test_refusal_body_is_typed(monkeypatch):
    make_corpus()
    r = client(RaisingLLM(LLMRefusal("x")), monkeypatch).post("/api/ask", json={"question": "widgets"})
    assert r.status_code == 422
    assert r.json() == {"error": "refused", "detail": "The model declined to answer this question."}


# ---------- access control on /api/ask ----------

def _ask(c, **kw):
    return c.post("/api/ask", json={"question": "widgets hum"}, **kw)


def test_unlisted_host_is_rejected(monkeypatch):
    make_corpus()
    c = client(good_llm(), monkeypatch)
    for path in ("/", "/api/papers"):
        assert c.get(path, headers={"host": "evil.example"}).status_code == 400
    assert _ask(c, headers={"host": "evil.example:8000"}).status_code == 400
    assert _ask(c).status_code == 200


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1:8000", "[::1]:8000"])
def test_default_local_hosts_are_accepted(monkeypatch, host):
    make_corpus()
    monkeypatch.setattr(settings, "allowed_hosts", Settings(_env_file=None).allowed_hosts)
    assert _ask(client(good_llm(), monkeypatch), headers={"host": host}).status_code == 200


def test_allowed_hosts_env_is_comma_separated(monkeypatch):
    monkeypatch.setenv("ALLOWED_HOSTS", "a.example, b.example,")
    assert Settings(_env_file=None).allowed_hosts == ["a.example", "b.example"]


def test_cross_origin_request_is_rejected(monkeypatch):
    make_corpus()
    c = client(good_llm(), monkeypatch)
    for origin in ("http://evil.example", "https://testserver", "http://testserver:9999", "null"):
        r = _ask(c, headers={"origin": origin})
        assert r.status_code == 403, origin
    assert _ask(c, headers={"origin": "http://testserver"}).status_code == 200
    assert _ask(c).status_code == 200  # no Origin header: curl, the CLI, tests


def test_ask_requires_json_content_type(monkeypatch):
    make_corpus()
    c = client(good_llm(), monkeypatch)
    r = c.post("/api/ask", content='{"question": "widgets hum"}', headers={"content-type": "text/plain"})
    assert r.status_code == 422
    r = c.post("/api/ask", data={"question": "widgets hum"})  # form-encoded, as a cross-site <form> would send
    assert r.status_code == 422


def test_api_token_required_when_configured(monkeypatch):
    make_corpus()
    monkeypatch.setattr(settings, "api_token", SecretStr("s3cret-token"))
    c = client(good_llm(), monkeypatch)
    missing, wrong = _ask(c), _ask(c, headers={"X-API-Token": "nope"})
    assert missing.status_code == 401 and wrong.status_code == 401
    assert "s3cret-token" not in missing.text + wrong.text and "nope" not in wrong.text
    assert _ask(c, headers={"X-API-Token": "café".encode()}).status_code == 401  # non-ASCII must not crash
    assert _ask(c, headers={"X-API-Token": "s3cret-token"}).status_code == 200
    assert c.get("/api/papers").status_code == 200  # only /api/ask is guarded


def test_token_check_comes_before_the_model_is_called(monkeypatch):
    make_corpus()
    monkeypatch.setattr(settings, "api_token", SecretStr("t"))
    llm = good_llm()
    c = client(llm, monkeypatch)
    assert _ask(c).status_code == 401
    assert _ask(c, headers={"origin": "http://evil.example", "X-API-Token": "t"}).status_code == 403
    assert not llm.calls if hasattr(llm, "calls") else True


def test_no_token_configured_local_requests_work(monkeypatch):
    make_corpus()
    assert settings.api_token is None
    assert _ask(client(good_llm(), monkeypatch)).status_code == 200


def test_empty_api_token_means_none(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "")
    assert Settings(_env_file=None).api_token is None


def test_page_sends_token_from_session_storage_not_url():
    html = (Path(main.STATIC) / "index.html").read_text(encoding="utf-8")
    assert "X-API-Token" in html and "sessionStorage" in html
    assert "localStorage" not in html and "location" not in html


# ---------- claim check on the surfaces ----------

def _graded_llm():
    return FakeLLM(RawAnswer(answer="Widgets hum [1]. They fly [1].", citations=[RawCitation(passage=1, quote="hum")]),
                   RawVerification(verdicts=[
                       RawVerdict(claim=1, passage=1, verdict="supported", reason=""),
                       RawVerdict(claim=2, passage=1, verdict="unsupported", reason="flying is never mentioned")]))


def test_cli_prints_the_claim_check(capsys):
    make_corpus()
    assert cli.main(["ask", "widgets hum"], llm=_graded_llm()) == 0
    out = capsys.readouterr().out
    assert "Claim check: 1 of 2 cited passages support their sentence" in out
    assert "✓ [1] supported: Widgets hum [1]." in out
    assert "✗ [1] unsupported: They fly [1]. (flying is never mentioned)" in out


def test_cli_no_verify_and_setting_off_skip_the_check(capsys, monkeypatch):
    make_corpus()
    llm = _graded_llm()
    assert cli.main(["ask", "widgets hum", "--no-verify"], llm=llm) == 0
    monkeypatch.setattr(settings, "verify", False)
    assert cli.main(["ask", "widgets hum"], llm=llm) == 0
    assert llm.verify_calls == [] and "Claim check" not in capsys.readouterr().out


def test_cli_reports_a_failed_check_but_still_answers(capsys):
    make_corpus()
    llm = FakeLLM(RawAnswer(answer="Widgets hum [1].", citations=[]), LLMTransient("down"))
    assert cli.main(["ask", "widgets hum"], llm=llm) == 0
    out = capsys.readouterr().out
    assert "Widgets hum [1]." in out and "Claim check failed:" in out and "try again" in out


def test_api_verify_flag_and_failed_check(monkeypatch):
    make_corpus()
    llm = _graded_llm()
    body = client(llm, monkeypatch).post("/api/ask", json={"question": "widgets hum", "verify": False}).json()
    assert body["support"] == [] and llm.verify_calls == []
    llm = FakeLLM(RawAnswer(answer="Widgets hum [1].", citations=[]), LLMRefusal("no"))
    r = client(llm, monkeypatch).post("/api/ask", json={"question": "widgets hum"})
    assert r.status_code == 200 and r.json()["support"] == [] and "declined" in r.json()["verify_error"]


def test_page_renders_the_claim_check_as_text_only():
    html = (main.STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="support"' in html and "renderSupport" in html
    assert "innerHTML" not in html
