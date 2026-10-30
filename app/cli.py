"""Command line: fetch the reading list, build the index, list the papers and ask a question."""
import argparse
import gzip
import json
import re
import sys
import tomllib
import urllib.error

from app import arxiv, corpus as corpus_mod
from app.answer import NO_PAPERS, UNCHECKED, Answer, ask, surname
from app.config import ROOT, settings
from app.llm import LLM, AnthropicLLM, LLMError, error_message

READING_LIST = ROOT / "readinglist.toml"


def reading_list_ids() -> list[str]:
    with READING_LIST.open("rb") as f:
        return [p["id"] for p in tomllib.load(f)["paper"]]


FETCH_ERRORS = (urllib.error.URLError, OSError, EOFError, gzip.BadGzipFile)  # HTTPError is a URLError
FETCH_FAILED = ("arXiv unreachable or throttling (HTTP 429) \u2014 papers already saved are kept; "
                "try again in a few minutes")


def _saved_meta(paper_id: str) -> arxiv.PaperMeta | None:
    path = arxiv.paper_dir(paper_id) / "meta.json"
    try:
        return arxiv.PaperMeta(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def cmd_fetch(_args) -> int:
    ids = reading_list_ids()
    try:
        bare = [re.sub(r"v\d+$", "", i) for i in ids]
        saved = {b: _saved_meta(b) for b in bare}
        # no metadata request at all when every paper is already saved
        new = {m.id: m for m in arxiv.fetch_meta([i for i, b in zip(ids, bare) if saved[b] is None])}
        metas = [saved[b] or new[b] for b in bare]
        for m in metas:
            arxiv.download(m)
            print(f"{m.id}  {m.title}")
    except FETCH_ERRORS:
        print(FETCH_FAILED, file=sys.stderr)
        return 1
    print(f"Fetched {len(metas)} papers into {arxiv.settings.data_dir / 'papers'}")
    return 0


def cmd_index(_args) -> int:
    c = corpus_mod.build()
    if not c.metas:
        print(NO_PAPERS)
        return 0
    mode = "hybrid" if c.index.embedder else "BM25 only"
    print(f"Indexed {len(c.chunks)} chunks from {len(c.metas)} papers ({mode}).")
    return 0


def cmd_papers(_args) -> int:
    c = corpus_mod.build()
    if not c.metas:
        print("No papers downloaded yet. Run: python -m app.cli fetch")
        return 0
    counts: dict[str, int] = {}
    for ch in c.chunks:
        counts[ch.paper_id] = counts.get(ch.paper_id, 0) + 1
    for m in c.metas.values():
        print(f"{m.id}  {_authors(m.authors)} {m.year}  {m.title}  ({counts.get(m.id, 0)} chunks)")
    return 0


def _authors(authors: list[str]) -> str:
    surnames = [surname(a) for a in authors if a.split()]
    if not surnames:
        return "Unknown"
    return surnames[0] if len(surnames) == 1 else f"{surnames[0]} & {surnames[1]}" if len(surnames) == 2 \
        else f"{surnames[0]} et al."


def format_answer(a: Answer) -> str:
    lines = [a.text]
    if a.searches:
        lines += ["", "Searches: " + "; ".join(a.searches)]
    refs = a.references()
    if refs:
        lines += ["", "References"]
        for r in refs:
            nums = ", ".join(str(n) for n in r.passages)
            lines.append(f"[{nums}] {_authors(r.authors)} {r.year}, {r.title}, §{r.section}")
            first = next((c for c in a.citations if c.paper_id == r.paper_id and c.section == r.section), None)
            if first and first.quote:
                warn = "" if first.quote_found else " ⚠ quote not found in the passage"
                lines.append(f'    "{first.quote}"{warn}')
    if a.unresolved_markers:
        marks = ", ".join(f"[{n}]" for n in a.unresolved_markers)
        lines += ["", f"Warning: the answer uses {marks} with no matching citation."]
    if a.unmarked_citations:
        marks = ", ".join(f"[{n}]" for n in a.unmarked_citations)
        lines += ["", f"Warning: {marks} cited but never marked in the answer text."]
    lines += support_lines(a)
    return "\n".join(lines)


SUPPORT_MARK = {"supported": "✓", "partial": "~", "unsupported": "✗", UNCHECKED: "?"}


def _short(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width - 1] + "…"


def support_lines(a: Answer) -> list[str]:
    if a.verify_error:
        return ["", f"Claim check failed: {a.verify_error}"]
    if not a.support:
        return []
    ok = sum(s.verdict == "supported" for s in a.support)
    lines = ["", f"Claim check: {ok} of {len(a.support)} cited passages support their sentence"]
    for s in a.support:
        reason = f" ({s.reason})" if s.reason and s.verdict != "supported" else ""
        lines.append(f"  {SUPPORT_MARK.get(s.verdict, '?')} [{s.passage}] {s.verdict}: {_short(s.claim)}{reason}")
    return lines


def cmd_ask(args, llm: LLM | None = None) -> int:
    c = corpus_mod.build()
    try:
        a = ask(args.question, c, llm or AnthropicLLM(), args.k, verify=settings.verify and not args.no_verify,
                agentic=settings.agentic or args.agentic)
    except LLMError as e:
        print(error_message(e), file=sys.stderr)
        return 1
    print(format_answer(a))
    return 0


def main(argv: list[str] | None = None, llm: LLM | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the reading list from arXiv (cached)").set_defaults(func=cmd_fetch)
    sub.add_parser("index", help="chunk and index the downloaded papers").set_defaults(func=cmd_index)
    sub.add_parser("papers", help="list the downloaded papers").set_defaults(func=cmd_papers)
    p = sub.add_parser("ask", help="ask a question and print the answer with its references")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=8, choices=range(1, 21), metavar="K",
                   help="passages to retrieve, 1 to 20 (default 8)")
    p.add_argument("--agentic", action="store_true",
                   help="let the model run follow-up searches before answering (more calls; on when AGENTIC=true)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the claim check (a second model call that grades each citation)")
    p.set_defaults(func=lambda a: cmd_ask(a, llm))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
