# arXiv Paper RAG

Ask a question about a fixed reading list of arXiv papers and get an answer where every claim cites the exact paper
and section, for example "[2] Reed & de Freitas 2015, §3.1 Inference". The seed list is the CSC2626 Week 5 papers on
learning programs and task structure from demonstrations. Papers are fetched from arXiv, parsed from their LaTeX
source, chunked by section and searched with BM25 plus optional embeddings. The answer comes from the Anthropic API.

Motion Grammar (Dantam & Stilman) is on the course list but is not on arXiv, so it is not in the reading list here.

## Features

- **Citations to the section.** Each answer has numbered references: authors and year, title, section path, and a
  short quote from the passage.
- **Quotes are checked.** The model must quote the passage it cites. A quote that is not found in that passage is
  marked with a warning, and so is a `[n]` (or `[n, m]`, `[n-m]`) in the text that has no matching citation, or a
  citation whose passage number is never marked in the text.
- **Claims are checked.** A second model call grades every cited sentence against each passage it marks:
  supported, partial or unsupported, with a one-line reason. A quote can be in the passage while the sentence
  around it says more than the passage does; this catches that.
- **Jump to the passage.** Each citation is looked up in the paper's PDF. The CLI prints its page; on the web page,
  "Open PDF at p. N" opens the PDF at that page with the passage highlighted.
- **Agentic search (optional).** With `--agentic` (or `AGENTIC=true`), the model sees the first results and runs
  up to 8 follow-up searches of its own, on the whole list or one paper, before the answer is written. Off by default
  until the comparison numbers are in.
- **Hybrid search.** BM25 keyword scoring, plus multilingual-e5 embeddings when the `embed` extra is installed.
  An optional cross-encoder reranker (`RERANK=auto`, same extra) reorders the top 30 candidates.
- **Three ways in.** A command line, a JSON API and a one-page web UI.
- **Polite to arXiv.** One request every 3 seconds, a project User-Agent, and everything cached under `data/`, so a
  paper is downloaded once.

## Quick start

Prerequisites: Python 3.12 with [uv](https://docs.astral.sh/uv/), and an Anthropic API key for answers.

```bash
py -3.12 -m uv sync                     # or plain: uv sync
# optional, for hybrid search (a large PyTorch download): py -3.12 -m uv sync --extra embed
# the first run then also downloads the embedding model (about 470 MB)
cp .env.example .env                    # then set ANTHROPIC_API_KEY, or run `ant auth login` instead
py -3.12 -m uv run python -m app.cli fetch    # download the papers from arXiv (about 1 minute, cached)
py -3.12 -m uv run python -m app.cli index    # chunk and index them
py -3.12 -m uv run python -m app.cli ask "How does the NPI core decide when to stop?"
```

`ask` prints the answer, its references and the claim check; `--no-verify` skips the check (one call instead of
two), and `--agentic` lets the model search again before answering. `papers` lists what is downloaded. `fetch`, `index` and `papers` need no API key; only `ask` calls the model. Without
credentials `ask` prints one line saying so and exits with status 1. `fetch` skips papers already saved; if arXiv is
unreachable or throttling it prints one line and exits with status 1, keeping what was saved.

For the web page and API:

```bash
py -3.12 -m uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Every `/api/ask` call spends API credit, so this command only listens on this machine: `--host 127.0.0.1` is
uvicorn's default, and you should not change it; to serve others, use `python -m app.serve` (see Deploying). It also refuses requests whose `Host` header is not in `ALLOWED_HOSTS`
(comma-separated, default `localhost,127.0.0.1,[::1]`; this blocks DNS rebinding) and `POST /api/ask` requests from
another origin (403). For any use beyond your own machine, add your hostname to `ALLOWED_HOSTS` and set `API_TOKEN`
(clients must then send it in the `X-API-Token` header, 401 otherwise; the page has a field for it, kept in
`sessionStorage` for the tab) or the spend limits `ASK_LIMIT_PER_HOUR` and `ASK_LIMIT_PER_DAY`. Use HTTPS in front of
the server, since the token is sent in the clear otherwise.

Open **http://localhost:8000**. `GET /api/papers` lists the papers. `POST /api/ask` takes
`{"question": "...", "k": 8, "verify": true, "agentic": false}` (question 1 to 1000 characters, `k` 1 to 20;
`agentic` defaults to the `AGENTIC` setting) and returns
`{text, references, citations, unresolved_markers, unmarked_citations, support, verify_error, searches}`. `support` has one
`{claim, passage, verdict, reason}` per cited sentence and passage; if the check itself fails, `verify_error` says
why and the answer is still returned. Each citation also has `pdf_page` (or `null`). `GET
/api/papers/{id}/pdf?quote=...&chunk=...` returns the paper's PDF with the quote, or failing that the cited chunk,
highlighted; only ids on the reading list are served. With no credentials at all (no
`ANTHROPIC_API_KEY`, no `ant auth login` profile) or when the API is unreachable it returns 503 with a message. A
model refusal is 422 with `{"error": "refused", "detail": ...}`, a bad request is 422 with
`{"error": "invalid_request", "detail": [{"loc": ..., "msg": ...}]}`, and any other model error is 502.

Settings come from the environment or a `.env` file at the repo root; see [.env.example](.env.example). They are
`ANTHROPIC_API_KEY`, `LLM_MODEL` (default `claude-opus-5-5`), `LLM_EFFORT` (default `medium`), `VERIFY` (default
`true`), `VERIFY_MODEL` (default: same as `LLM_MODEL`), `VERIFY_EFFORT` (default `low`), `AGENTIC` (default `false`), `AGENT_ROUNDS` (default `3`), `RERANK` (`off` or `auto`, default `off`),
`RERANK_MODEL` (default `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), `RERANK_DEPTH` (default `30`), `EMBED` (`auto` or `off`),
`DATA_DIR`, `ARXIV_DELAY_S`, `ALLOWED_HOSTS`, `API_TOKEN`, `ASK_LIMIT_PER_HOUR` and `ASK_LIMIT_PER_DAY` (both
default `0`, no limit).

## Deploying

The `Dockerfile` builds an image with the papers already fetched and indexed (arXiv is only called at build time) and
runs `python -m app.serve`, which listens on `0.0.0.0:$PORT` with proxy headers on. It refuses to start unless:

- `API_TOKEN` is set, or both `ASK_LIMIT_PER_HOUR` (questions per visitor address) and `ASK_LIMIT_PER_DAY`
  (questions in total) are, so strangers cannot spend your API credit without limit; over a limit, `/api/ask`
  answers 429 with `Retry-After`. A public demo would use the limits, since visitors have no token.
- `ALLOWED_HOSTS` names the public hostname.

```bash
docker build -t arxiv-paper-rag .                                   # BM25 only, small image
docker build -t arxiv-paper-rag --build-arg EXTRAS="--extra embed" .   # hybrid search (adds PyTorch)
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -e ALLOWED_HOSTS=rag.example.org \
  -e ASK_LIMIT_PER_HOUR=10 -e ASK_LIMIT_PER_DAY=200 arxiv-paper-rag
```

Any host that runs a container works (Fly.io, Render, Railway, Cloud Run); put HTTPS in front, and set
`FORWARDED_ALLOW_IPS` to the proxy's address so the limits count visitors rather than the proxy. The limits are kept
in memory, so run one instance. Vercel's Python functions are not a good fit: the app is a long-running server and the
optional embedder needs PyTorch.

## How it works

1. **Fetch.** `readinglist.toml` holds the arXiv ids. One batched API request gets the metadata; then each paper's
   LaTeX source and PDF are saved under `data/papers/<id>/`. Requests are 3 seconds apart.
2. **Parse.** The LaTeX is read into a section tree, with macros expanded and figures and bibliography dropped. A
   paper with no usable source falls back to the PDF text.
3. **Chunks.** Each section is split into chunks of about 900 characters (max 1400). A chunk remembers its section path.
4. **Search.** Chunks are scored with BM25Plus, scaled by the highest score. With embeddings, a chunk also gets a
   cosine similarity against the query, combined as 0.6 cosine + 0.4 BM25. Without them, a chunk must share a query
   term to count as a hit. With `RERANK=auto`, the top `RERANK_DEPTH` hits are rescored by a cross-encoder that reads
   the query and the chunk together, and the top k of that order are kept.
5. **Search again (agentic only).** The model gets the first results and a `search(query, paper?)` tool. For up to
   `AGENT_ROUNDS` turns it can run follow-up searches (5 passages each, 8 searches at most); new passages join a pool
   of at most 16, numbered in the order found.
6. **Answer.** The top passages are numbered and sent to the model with the question. It returns structured output:
   the answer text with `[n]` markers and a quote for each passage it used. The prompt tells it that passages are
   untrusted text and that it must not follow instructions inside them.
7. **Check quotes.** Each citation is mapped back to its paper and section, and its quote is looked for in that
   passage.
8. **Find it in the PDF.** The quote is searched for in the PDF's text (ligatures and line-end hyphens normalised);
   when the LaTeX-derived quote does not match exactly, six-word windows of the quote and then of the passage are
   tried, and the page with the most matches wins.
9. **Check claims.** The answer is split into sentences; each sentence with a `[n]` marker is sent with the passages
   it marks to a second call, which grades every (sentence, passage) pair as supported, partial or unsupported.
   Passages and the answer are both treated as untrusted text. At most 40 pairs are graded per answer.

```
app/        fetch, parse, search, answer, claim check, agentic search (agent.py), PDF lookup (pdflocate.py),
            CLI (cli.py), API (main.py), public server (serve.py), PaperQA2 adapter (paperqa2.py), page (static/)
eval/       question set, baseline numbers and their README
tests/      pytest suite
data/       downloaded papers and the index (gitignored)
```

## Evaluation

`app/evaluate.py` scores retrieval on 32 hand-labelled questions about the five papers, with no model call. Section
recall counts a hit when its own section title contains a gold substring for that hit's paper. Setup, metrics and how
the labels changed are in [eval/README.md](eval/README.md).

```bash
py -3.12 -m uv run python -m app.evaluate                # retrieval only, free
py -3.12 -m uv run python -m app.evaluate --answers      # model calls per question; asks first
py -3.12 -m uv run python -m app.evaluate --answers --systems plain agentic   # side by side
```

`--systems plain agentic paperqa2` compares this project's one-shot and agentic modes with PaperQA2 on the same
questions and metrics; see [eval/README.md](eval/README.md).

Baseline, run on 2026-10-25 over 380 chunks (all parsed from LaTeX), hybrid with `intfloat/multilingual-e5-small`:

| Retriever | Level | R@1 | R@3 | R@5 | R@8 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 | paper | 0.97 | 1.00 | 1.00 | 1.00 | 0.98 |
| bm25 | section | 0.59 | 0.94 | 0.97 | 0.97 | 0.75 |
| bm25 | all gold papers (7 cross) | 0.00 | 0.71 | 0.86 | 0.86 | n/a |
| hybrid | paper | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid | section | 0.66 | 0.94 | 0.97 | 1.00 | 0.79 |
| hybrid | all gold papers (7 cross) | 0.00 | 0.71 | 0.86 | 0.86 | n/a |

The question labels are a draft: nobody who knows the papers has reviewed them yet. Answer-level numbers (citation
precision and recall, quote found, claims supported) need an API key and come from the `--answers` run; they are not
in yet.

## Tests

```bash
py -3.12 -m uv run pytest -q
```

No test uses the network or a paid API; arXiv responses are recorded or built in the test and the model is faked.

## Limitations

- Only the five arXiv papers are searchable. Motion Grammar is not on arXiv.
- Answers need the Anthropic API and cost money per question; `LLM_EFFORT` trades cost for care.
- A section path is only as good as the parser. LaTeX with unusual macros can land text under the wrong heading.
- Equations and tables are kept as source text, which embeds and quotes poorly, and is hard to find in the PDF: such
  citations may open at the right page without a highlight, or at the first page.
- The claim check is the model grading its own citations. It catches overreach, but it can be wrong too, and it
  only judges the passages the answer cites, not ones it should have cited.
- Without the `embed` extra, search is keyword-based only.

## Roadmap

Tiers follow the project plan.

- **Tier 1: MVP (done).** Fixed reading list, LaTeX parsing, hybrid search, section citations, a 32-question eval with
  retrieval baselines. Left: review the eval labels and publish answer-level numbers.
- **Tier 2: better than plain RAG.**
  - [x] LaTeX-source parsing (done in Tier 1)
  - [x] Claim check: each cited passage is graded against the sentence that cites it
  - [x] Agentic multi-step retrieval: the model searches again and refines its query (built; not yet measured)
  - [x] Reranking (built, off by default; the retrieval eval scores it as `+rerank` rows once the model can load)
  - [x] Compare against plain RAG and PaperQA2 on the same papers and questions (harness built; not yet run)
- **Tier 3: reader integration.**
  - [x] A citation opens the PDF at its page with the passage highlighted (built in, with PyMuPDF)
  - [ ] Share the viewer with the Semantic PDF search reader project
- **Tier 4: ship.**
  - [x] Container image and a guarded server (token or spend limits, public host) for deployment
  - [ ] Deploy it (needs a hosting account and an API key with a spend cap)
  - [ ] A writeup with the results table, and a demo video
- Later: users add their own arXiv ids; a daily digest of new papers in chosen arXiv categories.

## License

Copyright (c) 2026 Frederick Wang. All rights reserved. See [LICENSE](LICENSE).
