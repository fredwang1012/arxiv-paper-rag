"""The eval harness: metric functions on hand-made data and end-to-end runs on a tiny corpus with a FakeLLM."""
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app import evaluate
from app.answer import Answer, Citation
from app.chunks import Chunk
from app.config import settings
from app.evaluate import Question
from app.llm import (AnthropicLLM, LLMRefusal, LLMTransient, LLMUnavailable, RawAnswer, RawCitation, RawVerdict,
                     RawVerification)
from app.retrieval import Hit
from tests.test_corpus import add_paper, meta


def q(papers=("1111.0001",), sections=(), keywords=(), qid="q1", question="widgets"):
    sections = sections if isinstance(sections, dict) else list(sections)
    return Question(qid, question, list(papers), sections, list(keywords), "factual")


def hit(paper_id, *path):
    return Hit(Chunk(f"{paper_id}#c0", paper_id, list(path), "text"), 1.0)


# ---------- metrics ----------

def test_section_relevance_needs_gold_paper_and_substring_ignoring_case():
    question = q(sections=["3.1 inference"])
    assert evaluate.section_relevant(question, "1111.0001", ["3 Model", "3.1 Inference"])
    assert not evaluate.section_relevant(question, "1111.0001", ["3 Model", "3.2 Training"])
    assert not evaluate.section_relevant(question, "2222.0002", ["3 Model", "3.1 Inference"])


def test_substrings_match_the_section_own_title_not_its_parents():
    assert evaluate.section_matches(["3 Model"], ["3 model"])
    assert not evaluate.section_matches(["3 Model", "3.1 Inference"], ["3 Model"])
    assert not evaluate.section_matches([], ["3 Model"])


def test_per_paper_sections_must_belong_to_the_hit_paper():
    question = q(papers=["1111.0001", "2222.0002"], sections={"1111.0001": ["Intro"], "2222.0002": ["Training"]})
    assert evaluate.section_relevant(question, "1111.0001", ["1 Intro"])
    assert evaluate.section_relevant(question, "2222.0002", ["2 Training"])
    assert not evaluate.section_relevant(question, "2222.0002", ["1 Intro"])  # Intro is gold only for 1111
    assert not evaluate.section_relevant(question, "1111.0001", ["2 Training"])
    # a gold paper without its own sections falls back to the paper
    assert evaluate.section_relevant(q(papers=["1111.0001", "2222.0002"], sections={"1111.0001": ["Intro"]}),
                                     "2222.0002", ["9 Anything"])


def test_all_papers_rank_is_where_the_last_gold_paper_first_appears():
    question = q(papers=["a", "b"])
    assert evaluate.all_papers_rank(question, ["x", "b", "b", "a"]) == 4
    assert evaluate.all_papers_rank(question, ["a", "a", "x"]) is None


def test_section_relevance_falls_back_to_paper_without_gold_sections():
    assert evaluate.section_relevant(q(), "1111.0001", ["Anything"])
    assert not evaluate.section_relevant(q(), "2222.0002", ["Anything"])


def test_first_rank_recall_and_mrr():
    assert evaluate.first_rank([False, False, True, True]) == 3
    assert evaluate.first_rank([False]) is None
    ranks = [1, 3, None, 2]
    assert evaluate.recall_at(ranks, 1) == 0.25
    assert evaluate.recall_at(ranks, 3) == 0.75
    assert evaluate.mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    assert evaluate.recall_at([], 3) == 0.0 and evaluate.mrr([]) == 0.0


def test_retrieval_eval_ranks_paper_and_section_separately():
    questions = [q(sections=["Method"], qid="a"),
                 q(papers=["3333.0003", "1111.0001"], sections={"1111.0001": ["Intro"]}, qid="b")]
    hits = [hit("2222.0002", "1 Intro"), hit("1111.0001", "1 Intro"), hit("1111.0001", "2 Method")]
    seen = []

    def search(text, k):
        seen.append(k)
        return hits

    res = evaluate.retrieval_eval(questions, search, [1, 3])
    assert seen == [3, 3]  # searched once per question, to the deepest cut-off
    a, b = res["questions"]
    assert (a["paper_rank"], a["section_rank"], a["gold_papers_found"], a["all_papers_rank"]) == (2, 3, 1, None)
    assert (b["paper_rank"], b["section_rank"], b["gold_papers_found"], b["all_papers_rank"]) == (2, 2, 1, None)
    assert res["summary"]["all_papers"] == {"recall@1": 0.0, "recall@3": 0.0, "questions": 1}
    assert a["hits"][0] == {"chunk_id": "2222.0002#c0", "paper_id": "2222.0002", "section": "1 Intro", "score": 1.0}
    assert res["summary"]["paper"] == {"recall@1": 0.0, "recall@3": 1.0, "mrr": 0.5}
    assert res["summary"]["section"]["recall@3"] == 1.0
    assert res["summary"]["section"]["mrr"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def citation(paper_id, found=True):
    return Citation(paper_id, "T", ["A. Author"], 2020, "1 Intro", f"{paper_id}#c0", "quote", found, 1)


def test_answer_metrics():
    question = q(papers=["1111.0001", "2222.0002"], keywords=["Widget", "gadget", "sprocket"])
    a = Answer("Widgets hum [1] and gadgets whir [2].",
               [citation("1111.0001"), citation("1111.0001", found=False), citation("9999.0009")])
    m = evaluate.answer_metrics(question, a)
    assert m["cited_papers"] == ["1111.0001", "9999.0009"]
    assert m["citation_precision"] == 0.5
    assert m["citation_recall"] == 0.5
    assert (m["quotes"], m["quotes_found"]) == (3, 2)
    assert m["keyword_hit"] == pytest.approx(2 / 3)
    assert m["cannot_answer"] is False


def test_answer_without_citations_counts_as_cannot_answer():
    m = evaluate.answer_metrics(q(keywords=[]), Answer("The passages do not say."))
    assert m["cannot_answer"] and m["citation_precision"] is None and m["citation_recall"] == 0.0
    assert m["keyword_hit"] is None


def test_summarize_answers_skips_errors_and_missing_values():
    rows = [
        {"citation_precision": 1.0, "citation_recall": 1.0, "quotes": 2, "quotes_found": 1, "keyword_hit": 1.0,
         "cannot_answer": False},
        {"citation_precision": None, "citation_recall": 0.0, "quotes": 0, "quotes_found": 0, "keyword_hit": 0.0,
         "cannot_answer": True},
        {"id": "x", "error": "LLMTransient: down"},
    ]
    s = evaluate.summarize_answers(rows)
    assert (s["answered"], s["errors"]) == (2, 1)
    assert s["citation_precision"] == 1.0
    assert s["citation_recall"] == 0.5
    assert s["quote_found_rate"] == 0.5
    assert s["keyword_hit_rate"] == 0.5
    assert s["cannot_answer_rate"] == 0.5
    assert evaluate.summarize_answers([])["citation_recall"] is None


# ---------- end to end ----------

class ScriptedLLM:
    """Answers by citing passage 1, or raises the error scripted for a question."""

    def __init__(self, errors=None):
        self.errors, self.calls = errors or {}, []

    def answer(self, question, passages):
        self.calls.append(question)
        if question in self.errors:
            raise self.errors[question]
        return RawAnswer(answer=f"{passages[0].chunk.text} [1]",
                         citations=[RawCitation(passage=1, quote=passages[0].chunk.text[:20])])

    def research(self, opening, papers, search, rounds):
        self.research_calls = getattr(self, "research_calls", 0) + 1
        search("gradient", None)

    def verify(self, claims, passages):
        self.verify_calls = getattr(self, "verify_calls", 0) + 1
        return RawVerification(verdicts=[RawVerdict(claim=c.number, passage=n, verdict="supported", reason="")
                                         for c in claims for n in c.passages])


QUESTIONS = [
    {"id": "hum", "question": "Why do widgets hum?", "papers": ["1111.0001"], "sections": ["Intro"],
     "keywords": ["hum"], "type": "factual"},
    {"id": "gears", "question": "How are gears trained?", "papers": ["2222.0002"], "sections": ["Training"],
     "keywords": ["gradient", "missing"], "type": "method"},
    {"id": "both", "question": "Compare widgets and gears", "papers": ["1111.0001", "2222.0002"], "sections": [],
     "keywords": ["widget"], "type": "cross"},
]


@pytest.fixture
def setup(tmp_path):
    add_paper(meta("1111.0001", "Neural Widgets"), {
        "Intro": "Widgets hum quietly because their coils vibrate.",
        "Method": "We build widgets from synthetic parts.",
    })
    add_paper(meta("2222.0002", "Deep Gears"), {
        "Intro": "Gears turn.",
        "Training": "Gears are trained with gradient descent on gear trains.",
    })
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in QUESTIONS) + "\n", encoding="utf-8")
    return path, tmp_path / "out.json"


def test_retrieval_only_run_prints_table_and_writes_json(setup, capsys):
    questions, out = setup
    assert evaluate.main(["--questions", str(questions), "--k", "1", "3", "--json", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "hybrid: skipped" in stdout
    assert "| Retriever | Level | R@1 | R@3 | MRR |" in stdout
    assert "| bm25 | section |" in stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_questions"] == 3 and report["ks"] == [1, 3] and "answers" not in report
    rows = {r["id"]: r for r in report["retrieval"]["bm25"]["questions"]}
    assert rows["hum"]["section_rank"] == 1
    assert rows["gears"]["section_rank"] == 1
    assert rows["both"]["gold_papers_found"] == 2 and rows["both"]["all_papers_rank"] is not None
    assert rows["hum"]["all_papers_rank"] is None  # single-paper questions are left out of that metric
    assert report["retrieval"]["bm25"]["summary"]["all_papers"]["questions"] == 1
    assert "| bm25 | all gold papers (1 cross) |" in stdout
    assert set(report["retrieval"]) == {"bm25"}


def test_hybrid_is_scored_when_the_corpus_has_embeddings(setup, monkeypatch, capsys):
    from app import corpus as corpus_mod
    from tests.test_corpus import FakeEmbedder
    monkeypatch.setattr(corpus_mod, "get_embedder", lambda: FakeEmbedder())
    questions, out = setup
    assert evaluate.main(["--questions", str(questions), "--json", str(out)]) == 0
    assert "| hybrid | paper |" in capsys.readouterr().out
    assert set(json.loads(out.read_text(encoding="utf-8"))["retrieval"]) == {"bm25", "hybrid"}


def test_answers_run_scores_each_question(setup, capsys):
    questions, out = setup
    llm = ScriptedLLM()
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--json", str(out)], llm=llm) == 0
    stdout = capsys.readouterr().out
    assert "plain: 3 calls (one per question)" in stdout and settings.llm_model in stdout
    assert "| Model | Answered | Errors |" in stdout
    assert len(llm.calls) == 3
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]
    assert answers["summary"]["answered"] == 3 and answers["summary"]["errors"] == 0
    assert answers["summary"]["quote_found_rate"] == 1.0
    rows = {r["id"]: r for r in answers["questions"]}
    assert rows["hum"]["citation_precision"] == 1.0 and rows["hum"]["keyword_hit"] == 1.0
    assert rows["gears"]["keyword_hit"] == 0.5
    assert rows["both"]["citation_recall"] == 0.5


def test_transient_and_refusal_are_recorded_per_question(setup, capsys):
    questions, out = setup
    llm = ScriptedLLM({"Why do widgets hum?": LLMTransient("timeout"), "How are gears trained?": LLMRefusal("no")})
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--json", str(out)], llm=llm) == 0
    rows = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]["questions"]}
    assert rows["hum"]["error"] == "LLMTransient: timeout"
    assert rows["gears"]["error"] == "LLMRefusal: no"
    assert "error" not in rows["both"]


def test_unavailable_llm_stops_the_run_and_keeps_partial_results(setup, capsys):
    questions, out = setup
    llm = ScriptedLLM({"How are gears trained?": LLMUnavailable("No valid Anthropic credentials")})
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--json", str(out)], llm=llm) == 2
    assert "No valid Anthropic credentials" in capsys.readouterr().err
    assert llm.calls == ["Why do widgets hum?", "How are gears trained?"]  # stopped: the third is never asked
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]
    assert [r["id"] for r in answers["questions"]] == ["hum"]
    assert answers["aborted"] == "No valid Anthropic credentials"


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("busy", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize("error, expected", [
    (_status_error(anthropic.RateLimitError, 429), "LLMTransient: The Anthropic API is busy or failing (HTTP 429"),
    (_status_error(anthropic.InternalServerError, 500), "LLMTransient: The Anthropic API is busy or failing (HTTP 500"),
])
def test_rate_limits_and_server_errors_from_the_sdk_are_recorded_and_the_run_goes_on(
        setup, monkeypatch, capsys, error, expected):
    def parse(**kwargs):
        if kwargs["output_format"] is RawVerification:
            return SimpleNamespace(stop_reason="end_turn", parsed_output=RawVerification(verdicts=[]))
        if "widgets hum" in kwargs["messages"][0]["content"]:
            raise error
        return SimpleNamespace(stop_reason="end_turn", parsed_output=RawAnswer(
            answer="See [1].", citations=[RawCitation(passage=1, quote="x")]))

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    questions, out = setup
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--json", str(out)],
                         llm=AnthropicLLM()) == 0
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]
    rows = {r["id"]: r for r in answers["questions"]}
    assert rows["hum"]["error"].startswith(expected)
    assert "error" not in rows["gears"] and "error" not in rows["both"]
    assert answers["summary"]["errors"] == 1 and answers["summary"]["answered"] == 2


def test_unexpected_exception_aborts_cleanly_and_keeps_partial_results(setup, capsys):
    questions, out = setup
    llm = ScriptedLLM({"How are gears trained?": KeyError("secret payload")})
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--json", str(out)], llm=llm) == 1
    captured = capsys.readouterr()
    assert "unexpected KeyError" in captured.err and "secret payload" not in captured.err + captured.out
    assert "| Model | Answered | Errors |" in captured.out  # the partial summary
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]
    assert [r["id"] for r in answers["questions"]] == ["hum"]
    assert answers["aborted"] == "unexpected KeyError"
    assert answers["summary"]["answered"] == 1


def test_json_is_written_after_every_question(setup):
    from app import corpus as corpus_mod
    questions = evaluate.load_questions(setup[0])
    snapshots = []
    evaluate.answers_eval(questions, corpus_mod.build(), ScriptedLLM(),
                          on_progress=lambda r: snapshots.append(len(r["questions"])))
    assert snapshots == [0, 1, 2, 3]


def test_answers_need_confirmation(setup, monkeypatch, capsys):
    questions, out = setup
    llm = ScriptedLLM()
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert evaluate.main(["--questions", str(questions), "--answers", "--json", str(out)], llm=llm) == 1
    assert llm.calls == [] and not out.exists()
    assert "Cancelled" in capsys.readouterr().out


def test_no_papers_is_a_clear_error(tmp_path, capsys):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(QUESTIONS[0]) + "\n", encoding="utf-8")
    assert evaluate.main(["--questions", str(path)]) == 1
    assert "No papers indexed" in capsys.readouterr().err


@pytest.mark.parametrize("line, message", [
    ({"id": "x", "question": "?", "papers": ["a", "b"], "sections": ["Intro"]}, "per paper"),
    ({"id": "x", "question": "?", "papers": ["a"], "sections": {"b": ["Intro"]}}, "not gold"),
])
def test_cross_paper_sections_are_given_per_gold_paper(tmp_path, line, message):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        evaluate.load_questions(path)


def test_bad_question_line_names_the_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(QUESTIONS[0]) + "\n\n{\"id\": \"x\"}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"q.jsonl:3"):
        evaluate.load_questions(path)


def test_answers_run_grades_citations_and_no_verify_skips_it(setup, capsys):
    questions, out = setup
    llm = ScriptedLLM()
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--json", str(out)], llm=llm) == 0
    stdout = capsys.readouterr().out
    assert "The claim check adds up to 3 more" in stdout and "| Supported | Unsupported |" in stdout
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]
    assert llm.verify_calls == 3 and answers["verify"]["effort"] == settings.verify_effort
    assert answers["summary"]["support_rate"] == 1.0 and answers["summary"]["claims_graded"] == 3
    assert all(r["support"] and r["verify_error"] is None for r in answers["questions"])

    llm = ScriptedLLM()
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--no-verify", "--json", str(out)],
                         llm=llm) == 0
    assert "claim check" not in capsys.readouterr().out
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]["plain"]
    assert not hasattr(llm, "verify_calls") and answers["verify"] is None
    assert answers["summary"]["support_rate"] is None


def test_summarize_answers_rates_verdicts():
    rows = [{"citation_precision": 1.0, "citation_recall": 1.0, "quotes": 1, "quotes_found": 1, "keyword_hit": 1.0,
             "cannot_answer": False, "supported": 3, "partial": 1, "unsupported": 0, "verify_error": None},
            {"citation_precision": 1.0, "citation_recall": 1.0, "quotes": 1, "quotes_found": 1, "keyword_hit": 1.0,
             "cannot_answer": False, "supported": 0, "partial": 0, "unsupported": 0, "verify_error": "down"}]
    s = evaluate.summarize_answers(rows)
    assert (s["claims_graded"], s["support_rate"], s["unsupported_rate"], s["verify_errors"]) == (4, 0.75, 0.0, 1)


def test_systems_are_compared_side_by_side(setup, capsys):
    questions, out = setup
    llm = ScriptedLLM()
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--no-verify", "--systems", "plain",
                          "agentic", "--json", str(out)], llm=llm) == 0
    stdout = capsys.readouterr().out
    assert "agentic: up to 12 calls" in stdout
    assert "| plain |" in stdout and "| agentic |" in stdout
    answers = json.loads(out.read_text(encoding="utf-8"))["answers"]
    assert list(answers) == ["plain", "agentic"]
    assert answers["agentic"]["system"] == "agentic" and answers["agentic"]["agent_rounds"] == settings.agent_rounds
    assert llm.research_calls == 3 and len(llm.calls) == 6
    assert all(r["searches"] == ["gradient"] for r in answers["agentic"]["questions"])
    assert all(r["searches"] == [] for r in answers["plain"]["questions"])


def test_paperqa2_without_the_package_stops_before_spending(setup, capsys, monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "paperqa" else real(name, *a))
    questions, out = setup
    llm = ScriptedLLM()
    assert evaluate.main(["--questions", str(questions), "--answers", "--yes", "--systems", "paperqa2"], llm=llm) == 1
    assert "uv run --with paper-qa==" in capsys.readouterr().err and llm.calls == []


def test_runner_for_an_injected_system_and_unknown_names(setup):
    from app import corpus as corpus_mod
    questions = evaluate.load_questions(setup[0])
    c = corpus_mod.build()
    result = evaluate.answers_eval(questions, c, ScriptedLLM(), verify=False, system="paperqa2",
                                   run=lambda q: Answer("The passages do not say."))
    assert result["system"] == "paperqa2" and result["summary"]["cannot_answer_rate"] == 1.0
    with pytest.raises(ValueError):
        evaluate.system_runner("other", c, ScriptedLLM(), False)


def test_empty_quotes_are_left_out_of_the_quote_rate():
    a = Answer("x [1].", [citation("1111.0001"), citation("1111.0001")])
    a.citations[1].quote = ""
    m = evaluate.answer_metrics(q(), a)
    assert (m["quotes"], m["quotes_found"]) == (1, 1)
