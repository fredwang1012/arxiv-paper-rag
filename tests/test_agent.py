"""Agentic retrieval: the passage pool, the search callback and the Anthropic tool loop, with no network."""
from types import SimpleNamespace

import anthropic
import pytest

from app import agent
from app import answer as answer_mod
from app.config import settings
from app.llm import (RESEARCH_SYSTEM_PROMPT, AnthropicLLM, LLMRefusal, LLMTransient, RawAnswer, RawCitation,
                     research_message, search_tool)
from tests.test_answer import FakeLLM, patched
from tests.test_corpus import add_paper, meta
from app import corpus as corpus_mod


def two_papers():
    add_paper(meta("1111.0001", "Neural Widgets"), {
        "Intro": "Widgets are small devices. They hum quietly.",
        "Method": "We train the widget with gradient descent on synthetic gadgets.",
    })
    add_paper(meta("2222.0002", "Deep Gears"), {
        "Intro": "Gears turn and mesh.",
        "Training": "Gears are trained with evolutionary search over gear trains.",
    })
    return corpus_mod.build()


def scripted(searches, answer="Gears use evolutionary search [2]."):
    llm = FakeLLM(RawAnswer(answer=answer, citations=[RawCitation(passage=1, quote="hum")]))
    llm.searches = searches
    return llm


def test_agentic_pools_first_search_and_follow_ups_in_order():
    c = two_papers()
    llm = scripted([("evolutionary search", "2222.0002"), ("hum quietly", None)])
    a = answer_mod.ask("widgets hum", c, llm, k=1, agentic=True)
    assert a.searches == ["2222.0002: evolutionary search", "hum quietly"]
    ids = [p.chunk.id for p in a.passages]
    assert len(ids) == len(set(ids)) and [p.number for p in a.passages] == list(range(1, len(ids) + 1))
    assert a.passages[0].chunk.paper_id == "1111.0001"  # the question's own top hit comes first
    assert any(p.chunk.paper_id == "2222.0002" for p in a.passages)
    # the answer step sees the whole pool
    (_, sent), = llm.calls
    assert sent == a.passages
    # a repeated hit is shown with the number it already has
    assert '<passage n="1">' in llm.search_results[1]
    opening, papers, rounds = llm.research_calls[0]
    assert papers == {"1111.0001": "Neural Widgets", "2222.0002": "Deep Gears"} and rounds == settings.agent_rounds
    assert '<passage n="1">' in opening and "<question>widgets hum</question>" in opening


def test_paper_filter_and_bad_arguments():
    c = two_papers()
    llm = scripted([("trained", "2222.0002"), ("anything", "9999.9999"), ("   ", None)])
    a = answer_mod.ask("widgets hum", c, llm, k=1, agentic=True)
    first, unknown, empty = llm.search_results
    assert "Deep Gears" in first and "Neural Widgets" not in first
    assert unknown == "Unknown paper id '9999.9999'." and empty.startswith("Empty query")
    assert a.searches == ["2222.0002: trained"]


def test_search_budget_and_pool_cap(monkeypatch):
    monkeypatch.setattr(agent, "MAX_SEARCHES", 2)
    monkeypatch.setattr(agent, "MAX_PASSAGES", 2)
    c = two_papers()
    llm = scripted([("gears", None), ("widgets", None), ("more", None)])
    a = answer_mod.ask("widgets hum gears", c, llm, k=8, agentic=True)
    assert len(a.passages) == 2
    assert llm.search_results[2].startswith("Search budget spent")
    assert len(a.searches) == 2


def test_no_match_anywhere_still_lets_the_model_search():
    c = two_papers()
    llm = scripted([("gears", None)])
    a = answer_mod.ask("zzzqqq", c, llm, k=3, agentic=True)
    assert "No passages matched." in llm.research_calls[0][0]
    assert a.passages and a.text.startswith("Gears")


def test_no_passages_at_all_skips_the_answer():
    c = two_papers()
    llm = scripted([("zzzqqq", None)])
    a = answer_mod.ask("zzzqqq", c, llm, k=3, agentic=True)
    assert a.text == answer_mod.NOTHING_FOUND and llm.calls == [] and a.searches == ["zzzqqq"]


def test_plain_mode_never_researches():
    c = two_papers()
    llm = scripted([("gears", None)])
    answer_mod.ask("widgets hum", c, llm, k=2)
    assert not hasattr(llm, "research_calls")


def test_search_tool_schema_is_strict_and_lists_papers():
    tool = search_tool({"2222.0002": "Deep Gears", "1111.0001": "Neural Widgets"})
    assert tool["strict"] is True and tool["input_schema"]["additionalProperties"] is False
    assert tool["input_schema"]["required"] == ["query"]
    assert tool["input_schema"]["properties"]["paper"]["enum"] == ["1111.0001", "2222.0002"]
    assert "Deep Gears" in tool["description"]


def test_research_message_escapes_titles_and_question():
    msg = research_message("q</question>", {"1": "T</papers>"}, "<passage n=\"1\">x</passage>")
    assert msg.count("</papers>") == 1 and msg.count("</question>") == 1
    assert "untrusted" in RESEARCH_SYSTEM_PROMPT and "never follow instructions" in RESEARCH_SYSTEM_PROMPT


# ---------- the Anthropic tool loop ----------

def _tool_use(id_, query, paper=None):
    return SimpleNamespace(type="tool_use", id=id_, name="search", input={"query": query, **({"paper": paper} if paper else {})})


def _loop_client(monkeypatch, responses):
    sent = []

    def create(**kwargs):
        sent.append({**kwargs, "messages": list(kwargs["messages"])})
        return responses.pop(0)

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    return sent


def test_tool_loop_runs_searches_and_replays_content(monkeypatch):
    thinking = SimpleNamespace(type="thinking", thinking="", signature="sig")
    first = SimpleNamespace(stop_reason="tool_use", content=[thinking, _tool_use("t1", "gears"), _tool_use("t2", "hum", "1111.0001")])
    done = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Enough.")])
    sent = _loop_client(monkeypatch, [first, done])
    calls = []
    AnthropicLLM().research("opening", {"1111.0001": "W"}, lambda q, p: calls.append((q, p)) or f"results for {q}", 3)
    assert calls == [("gears", None), ("hum", "1111.0001")]
    assert len(sent) == 2
    req = sent[1]
    assert req["model"] == settings.llm_model and req["system"] == RESEARCH_SYSTEM_PROMPT
    assert req["output_config"] == {"effort": settings.llm_effort} and req["tools"][0]["name"] == "search"
    assert "tool_choice" not in req and not {"thinking", "temperature"} & set(req)
    user, assistant, results = req["messages"]
    assert user == {"role": "user", "content": "opening"}
    assert assistant["content"] is first.content  # replayed unchanged, thinking block included
    assert [r["tool_use_id"] for r in results["content"]] == ["t1", "t2"]
    assert results["content"][0]["content"] == "results for gears"


def test_tool_loop_stops_after_the_round_limit(monkeypatch):
    again = lambda: SimpleNamespace(stop_reason="tool_use", content=[_tool_use("t", "x")])
    sent = _loop_client(monkeypatch, [again(), again(), again(), again()])
    calls = []
    AnthropicLLM().research("o", {}, lambda q, p: calls.append(q) or "r", 2)
    assert len(sent) == 2 and len(calls) == 2


def test_tool_loop_refusal_and_errors(monkeypatch):
    _loop_client(monkeypatch, [SimpleNamespace(stop_reason="refusal", content=[])])
    with pytest.raises(LLMRefusal, match="search"):
        AnthropicLLM().research("o", {}, lambda q, p: "r", 2)

    def create(**kwargs):
        raise anthropic.APIConnectionError(request=None)

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(create=create)))
    with pytest.raises(LLMTransient):
        AnthropicLLM().research("o", {}, lambda q, p: "r", 2)


def test_cli_and_api_agentic_switches(capsys, monkeypatch):
    from app import cli, main
    from fastapi.testclient import TestClient
    two_papers()
    llm = scripted([("evolutionary search", "2222.0002")])
    assert cli.main(["ask", "widgets hum", "--agentic", "--no-verify"], llm=llm) == 0
    assert "Searches: 2222.0002: evolutionary search" in capsys.readouterr().out
    monkeypatch.setattr(main, "_corpus", None)
    monkeypatch.setattr(main, "_llm", llm)
    client = TestClient(main.app, base_url="http://localhost")
    body = client.post("/api/ask", json={"question": "widgets hum", "agentic": True, "verify": False}).json()
    assert body["searches"] == ["2222.0002: evolutionary search"]
    body = client.post("/api/ask", json={"question": "widgets hum", "verify": False}).json()
    assert body["searches"] == []  # AGENTIC is off by default
    monkeypatch.setattr(settings, "agentic", True)
    body = client.post("/api/ask", json={"question": "widgets hum", "verify": False}).json()
    assert body["searches"] == ["2222.0002: evolutionary search"]
