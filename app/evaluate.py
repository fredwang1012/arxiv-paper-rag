"""Score retrieval, and optionally cited answers, against the labelled questions in eval/questions.jsonl.

    python -m app.evaluate [--k 1 3 5 8] [--answers [--yes] [--no-verify]] [--json out.json] [--questions path]

Retrieval metrics need no LLM. --answers asks the LLM once per question, plus once more to grade the citations unless
--no-verify is given, and needs an API key.
"""
import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from app import corpus as corpus_mod
from app.answer import Answer, ask
from app.chunks import search_text
from app.config import ROOT, settings
from app.llm import LLM, AnthropicLLM, LLMError, LLMUnavailable
from app.retrieval import Hit, Index

QUESTIONS = ROOT / "eval" / "questions.jsonl"
DEFAULT_KS = [1, 3, 5, 8]
ANSWER_K = 8  # passages per answer, as in ask()

Search = Callable[[str, int], list[Hit]]


@dataclass
class Question:
    id: str
    question: str
    papers: list[str]  # arXiv ids that answer it
    # section-title substrings that answer it: one list for a single paper, or {paper id: list} per paper
    sections: list[str] | dict[str, list[str]] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)  # words a correct answer should contain
    type: str = ""  # factual, method, term or cross

    def gold_sections(self, paper_id: str) -> list[str]:
        if isinstance(self.sections, dict):
            return self.sections.get(paper_id, [])
        return self.sections if paper_id in self.papers else []


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                q = Question(**json.loads(line))
                if isinstance(q.sections, dict) and not set(q.sections) <= set(q.papers):
                    raise ValueError(f"sections name papers that are not gold: {sorted(set(q.sections) - set(q.papers))}")
                if isinstance(q.sections, list) and q.sections and len(q.papers) > 1:
                    raise ValueError("a question with several papers must give its sections per paper")
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{n}: {e}") from e
            out.append(q)
    return out


# ---------- metrics ----------

def section_matches(path: list[str], substrings: list[str]) -> bool:
    """Some substring occurs, ignoring case, in the section's own title (the last element of its path).

    The own title, not the joined path: "3 Model" must not also match its subsections "3 Model > 3.1 Inference".
    """
    title = path[-1].lower() if path else ""
    return any(s.lower() in title for s in substrings)


def paper_relevant(q: Question, paper_id: str) -> bool:
    return paper_id in q.papers


def section_relevant(q: Question, paper_id: str, path: list[str]) -> bool:
    """A gold paper and one of that paper's gold sections; a paper without gold sections falls back to the paper."""
    if not paper_relevant(q, paper_id):
        return False
    subs = q.gold_sections(paper_id)
    return not subs or section_matches(path, subs)


def all_papers_rank(q: Question, hit_papers: list[str]) -> int | None:
    """1-based rank by which every gold paper has appeared, None if some gold paper never does."""
    ranks = [first_rank([p == gold for p in hit_papers]) for gold in q.papers]
    return None if None in ranks else max(ranks)


def first_rank(flags: list[bool]) -> int | None:
    """1-based rank of the first relevant hit, None if there is none."""
    return next((i for i, f in enumerate(flags, start=1) if f), None)


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def retrieval_eval(questions: list[Question], search: Search, ks: list[int]) -> dict:
    """Paper and section recall@k and MRR (over the top max(ks) hits), with the hits behind them per question."""
    depth = max(ks)
    rows = []
    for q in questions:
        hits = search(q.question, depth)
        rows.append({
            "id": q.id, "type": q.type,
            "paper_rank": first_rank([paper_relevant(q, h.chunk.paper_id) for h in hits]),
            "section_rank": first_rank([section_relevant(q, h.chunk.paper_id, h.chunk.section_path) for h in hits]),
            # cross-paper questions: how many gold papers show up, and by which rank all of them have
            "gold_papers_found": len({h.chunk.paper_id for h in hits} & set(q.papers)),
            "all_papers_rank": all_papers_rank(q, [h.chunk.paper_id for h in hits]) if len(q.papers) > 1 else None,
            "hits": [{"chunk_id": h.chunk.id, "paper_id": h.chunk.paper_id,
                      "section": " > ".join(h.chunk.section_path), "score": round(h.score, 4)} for h in hits],
        })
    summary = {}
    for level in ("paper", "section"):
        ranks = [r[f"{level}_rank"] for r in rows]
        summary[level] = {**{f"recall@{k}": recall_at(ranks, k) for k in ks}, "mrr": mrr(ranks)}
    # fraction of the cross-paper questions whose gold papers all appear in the top k
    multi = [r["all_papers_rank"] for q, r in zip(questions, rows) if len(q.papers) > 1]
    summary["all_papers"] = {**{f"recall@{k}": recall_at(multi, k) for k in ks}, "questions": len(multi)}
    return {"summary": summary, "questions": rows}


def answer_metrics(q: Question, a: Answer) -> dict:
    cited = {c.paper_id for c in a.citations}
    gold = set(q.papers)
    text = a.text.lower()
    return {
        "cited_papers": sorted(cited),
        "citation_precision": len(cited & gold) / len(cited) if cited else None,
        "citation_recall": len(cited & gold) / len(gold),
        "quotes": len(a.citations),
        "quotes_found": sum(c.quote_found for c in a.citations),
        "keyword_hit": sum(k.lower() in text for k in q.keywords) / len(q.keywords) if q.keywords else None,
        # the prompt tells the model to return no citations when the passages do not answer the question
        "cannot_answer": not a.citations,
        # claim check: (sentence, passage) pairs graded, by verdict; unchecked pairs are left out
        **{v: sum(x.verdict == v for x in a.support) for v in VERDICTS},
        "verify_error": a.verify_error,
    }


VERDICTS = ("supported", "partial", "unsupported")


def summarize_answers(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    quotes = sum(r["quotes"] for r in ok)
    graded = {v: sum(r.get(v, 0) for r in ok) for v in VERDICTS}
    n_graded = sum(graded.values())
    return {
        "answered": len(ok),
        "errors": len(rows) - len(ok),
        "citation_precision": _mean([r["citation_precision"] for r in ok if r["citation_precision"] is not None]),
        "citation_recall": _mean([r["citation_recall"] for r in ok]),
        "quote_found_rate": sum(r["quotes_found"] for r in ok) / quotes if quotes else None,
        "keyword_hit_rate": _mean([r["keyword_hit"] for r in ok if r["keyword_hit"] is not None]),
        "cannot_answer_rate": _mean([float(r["cannot_answer"]) for r in ok]),
        "claims_graded": n_graded,
        "support_rate": graded["supported"] / n_graded if n_graded else None,
        "unsupported_rate": graded["unsupported"] / n_graded if n_graded else None,
        "verify_errors": sum(1 for r in ok if r.get("verify_error")),
    }


def answers_eval(questions: list[Question], corpus: corpus_mod.Corpus, llm: LLM,
                 on_progress: Callable[[dict], None] = lambda result: None, verify: bool = True) -> dict:
    """Ask every question; a refusal or transient failure is that question's error. LLMUnavailable propagates.

    With verify, each answer's citations are also graded (a second call); a failed grading is recorded on the row."""
    rows: list[dict] = []
    result = {"model": settings.llm_model, "effort": settings.llm_effort, "k": ANSWER_K,
              "verify": {"model": settings.verify_model or settings.llm_model, "effort": settings.verify_effort}
              if verify else None,
              "summary": summarize_answers(rows), "questions": rows}
    on_progress(result)
    for q in questions:
        try:
            a = ask(q.question, corpus, llm, k=ANSWER_K, verify=verify)
        except LLMUnavailable:
            raise
        except LLMError as e:  # LLMTransient, LLMRefusal or a malformed response
            row = {"id": q.id, "type": q.type, "error": f"{type(e).__name__}: {e}"}
        else:
            row = {"id": q.id, "type": q.type, "answer": a.text, **answer_metrics(q, a),
                   "citations": [{"paper_id": c.paper_id, "section": c.section, "quote": c.quote,
                                  "quote_found": c.quote_found} for c in a.citations],
                   "support": [s.__dict__ for s in a.support]}
        rows.append(row)
        result["summary"] = summarize_answers(rows)
        on_progress(result)
    return result


# ---------- output ----------

def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def retrieval_table(results: dict[str, dict], ks: list[int]) -> str:
    lines = ["| Retriever | Level | " + " | ".join(f"R@{k}" for k in ks) + " | MRR |",
             "| --- | --- | " + " | ".join("---" for _ in ks) + " | --- |"]
    for name, res in results.items():
        for level in ("paper", "section"):
            s = res["summary"][level]
            lines.append(f"| {name} | {level} | " + " | ".join(_fmt(s[f"recall@{k}"]) for k in ks)
                         + f" | {_fmt(s['mrr'])} |")
        s = res["summary"]["all_papers"]
        if s["questions"]:
            lines.append(f"| {name} | all gold papers ({s['questions']} cross) | "
                         + " | ".join(_fmt(s[f"recall@{k}"]) for k in ks) + " | n/a |")
    return "\n".join(lines)


def answers_table(result: dict) -> str:
    s = result["summary"]
    return "\n".join([
        "| Model | Answered | Errors | Citation precision | Citation recall | Quote found | Supported | Unsupported "
        "| Keyword hit | Cannot answer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {result['model']} ({result['effort']}) | {s['answered']} | {s['errors']} | {_fmt(s['citation_precision'])} "
        f"| {_fmt(s['citation_recall'])} | {_fmt(s['quote_found_rate'])} | {_fmt(s.get('support_rate'))} "
        f"| {_fmt(s.get('unsupported_rate'))} | {_fmt(s['keyword_hit_rate'])} | {_fmt(s['cannot_answer_rate'])} |",
    ])


def _write_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)  # a crash mid-write never leaves a half-written report


def _display(path: Path) -> str:
    """Repo-relative when possible, so a committed report does not carry a local absolute path."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv: list[str] | None = None, llm: LLM | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS, help="cut-offs for recall@k")
    parser.add_argument("--answers", action="store_true", help="also score LLM answers (one API call per question)")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation for --answers")
    parser.add_argument("--no-verify", action="store_true",
                        help="with --answers, skip the claim check (one fewer call per question)")
    parser.add_argument("--json", type=Path, help="write per-question detail here")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args(argv)
    ks = sorted(set(args.k))
    if ks[0] < 1:
        parser.error("--k values must be at least 1")

    questions = load_questions(args.questions)
    corpus = corpus_mod.build()
    if not corpus.chunks:
        print(f"No papers indexed under {settings.data_dir / 'papers'}: download the reading list first.",
              file=sys.stderr)
        return 1

    titles = {pid: m.title for pid, m in corpus.metas.items()}
    bm25 = Index(corpus.chunks, [search_text(c, titles.get(c.paper_id, "")) for c in corpus.chunks])
    retrievers: dict[str, Search] = {"bm25": bm25.search}
    if corpus.index.mode == "hybrid":
        retrievers["hybrid"] = corpus.search
    else:
        print("hybrid: skipped, no embedder (install it with `uv sync --extra embed`)")

    results = {name: retrieval_eval(questions, search, ks) for name, search in retrievers.items()}
    report: dict = {"date": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "questions_file": _display(args.questions), "n_questions": len(questions), "ks": ks,
                    "chunks": len(corpus.chunks), "retrieval": results}
    print(f"Retrieval over {len(questions)} questions, {len(corpus.chunks)} chunks:\n")
    print(retrieval_table(results, ks))
    for name, res in results.items():
        misses = [r["id"] for r in res["questions"] if r["section_rank"] is None]
        print(f"\n{name}: no gold section in the top {max(ks)} for {len(misses)} question(s)"
              + (f": {', '.join(misses)}" if misses else ""))

    json_path = args.json
    if args.answers:
        json_path = json_path or settings.data_dir / "eval" / "answers.json"
        verify = settings.verify and not args.no_verify
        print(f"\n--answers makes {len(questions)} LLM calls (one per question) to {settings.llm_model} "
              f"at effort {settings.llm_effort}, each with {ANSWER_K} passages. This costs money.")
        if verify:
            print(f"The claim check adds up to {len(questions)} more, to {settings.verify_model or settings.llm_model} "
                  f"at effort {settings.verify_effort} (skip it with --no-verify).")
        if not args.yes and not _confirm("Continue? [y/N] "):
            print("Cancelled; no LLM calls made.")
            return 1
        llm = llm or AnthropicLLM()

        def save(partial: dict) -> None:
            report["answers"] = partial
            _write_json(json_path, report)

        try:
            answers_eval(questions, corpus, llm, on_progress=save, verify=verify)
        except LLMUnavailable as e:
            report["answers"]["aborted"] = str(e)
            _write_json(json_path, report)
            print(f"\n{e}\nPartial results saved to {json_path}", file=sys.stderr)
            return 2
        except Exception as e:  # anything unforeseen: keep what was scored, name the class only (no payloads)
            report["answers"]["aborted"] = f"unexpected {type(e).__name__}"
            _write_json(json_path, report)
            print(f"\nStopped by an unexpected {type(e).__name__}. Partial results so far:\n", file=sys.stderr)
            print(answers_table(report["answers"]))
            print(f"\nPartial results saved to {json_path}", file=sys.stderr)
            return 1
        print("\nAnswers:\n")
        print(answers_table(report["answers"]))

    if json_path:
        _write_json(json_path, report)
        print(f"\nPer-question detail: {json_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
