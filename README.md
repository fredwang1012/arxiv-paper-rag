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
- **Hybrid search.** BM25 keyword scoring, plus multilingual-e5 embeddings when the `embed` extra is installed.
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

`papers` lists what is downloaded. `fetch`, `index` and `papers` need no API key; only `ask` calls the model. Without
credentials `ask` prints one line saying so and exits with status 1. `fetch` skips papers already saved; if arXiv is
unreachable or throttling it prints one line and exits with status 1, keeping what was saved.

For the web page and API:

```bash
py -3.12 -m uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Every `/api/ask` call spends API credit, so the server only listens on this machine: `--host 127.0.0.1` is uvicorn's
default, and you should not change it. It also refuses requests whose `Host` header is not in `ALLOWED_HOSTS`
(comma-separated, default `localhost,127.0.0.1,[::1]`; this blocks DNS rebinding) and `POST /api/ask` requests from
another origin (403). For any use beyond your own machine, add your hostname to `ALLOWED_HOSTS` and set `API_TOKEN`:
clients must then send it in the `X-API-Token` header (401 otherwise); the page has a field for it, kept in
`sessionStorage` for the tab. Use HTTPS in front of the server, since the token is sent in the clear otherwise.

Open **http://localhost:8000**. `GET /api/papers` lists the papers. `POST /api/ask` takes
`{"question": "...", "k": 8}` (question 1 to 1000 characters, `k` 1 to 20) and returns
`{text, references, citations, unresolved_markers, unmarked_citations}`. With no credentials at all (no
`ANTHROPIC_API_KEY`, no `ant auth login` profile) or when the API is unreachable it returns 503 with a message. A
model refusal is 422 with `{"error": "refused", "detail": ...}`, a bad request is 422 with
`{"error": "invalid_request", "detail": [{"loc": ..., "msg": ...}]}`, and any other model error is 502.

Settings come from the environment or a `.env` file at the repo root; see [.env.example](.env.example). They are
`ANTHROPIC_API_KEY`, `LLM_MODEL` (default `claude-opus-5-5`), `LLM_EFFORT` (default `medium`), `EMBED` (`auto` or
`off`), `DATA_DIR`, `ARXIV_DELAY_S`, `ALLOWED_HOSTS` and `API_TOKEN`.

## How it works

1. **Fetch.** `readinglist.toml` holds the arXiv ids. One batched API request gets the metadata; then each paper's
   LaTeX source and PDF are saved under `data/papers/<id>/`. Requests are 3 seconds apart.
2. **Parse.** The LaTeX is read into a section tree, with macros expanded and figures and bibliography dropped. A
   paper with no usable source falls back to the PDF text.
3. **Chunks.** Each section is split into chunks of about 900 characters (max 1400). A chunk remembers its section path.
4. **Search.** Chunks are scored with BM25Plus, scaled by the highest score. With embeddings, a chunk also gets a
   cosine similarity against the query, combined as 0.6 cosine + 0.4 BM25. Without them, a chunk must share a query
   term to count as a hit.
5. **Answer.** The top passages are numbered and sent to the model with the question. It returns structured output:
   the answer text with `[n]` markers and a quote for each passage it used. The prompt tells it that passages are
   untrusted text and that it must not follow instructions inside them.
6. **Check.** Each citation is mapped back to its paper and section, and its quote is looked for in that passage.

```
app/        fetch, parse, search, answer, CLI (cli.py), API (main.py), static page (static/)
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
py -3.12 -m uv run python -m app.evaluate --answers      # one model call per question; asks first
```

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
precision and recall, quote found) need an API key and come from the `--answers` run; they are not in yet.

## Tests

```bash
py -3.12 -m uv run pytest -q
```

No test uses the network or a paid API; arXiv responses are recorded or built in the test and the model is faked.

## Limitations

- Only the five arXiv papers are searchable. Motion Grammar is not on arXiv.
- Answers need the Anthropic API and cost money per question; `LLM_EFFORT` trades cost for care.
- A section path is only as good as the parser. LaTeX with unusual macros can land text under the wrong heading.
- Equations and tables are kept as source text, which embeds and quotes poorly.
- Quote checking confirms the quote is in the passage, not that the passage supports the claim.
- Without the `embed` extra, search is keyword-based only.

## Roadmap

Tier 1 (this repo) is done once the eval labels are reviewed and answer-level numbers are in. Next:

- **Tier 2: better retrieval and checking.** Agentic multi-step retrieval, reranking, and citation verification that
  checks the passage supports the claim, not only that the quote is in it.
- **Tier 3: comparison.** Compare against plain RAG and PaperQA2 on the eval set.
- **Tier 4: reader and release.** Reader integration, so a citation opens the PDF at the highlighted passage via
  semantic-pdf-reader; public deployment; a writeup and demo video.
- Review the eval labels and publish answer-level numbers.
- Let users add their own arXiv ids to the reading list.

## License

Copyright (c) 2026 Frederick Wang. All rights reserved. See [LICENSE](LICENSE).
