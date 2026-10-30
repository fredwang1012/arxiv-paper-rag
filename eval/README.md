# Eval set

> **Draft for the owner to check.** The 32 questions and their gold labels in `questions.jsonl` were written from the
> parsed text of the papers and checked by script, but nobody who knows the papers has reviewed them yet. Fix any
> label you disagree with before treating the numbers below as final.

The questions cover the CSC2626 Week 5 papers that are on arXiv:

| arXiv id | Paper |
| --- | --- |
| 1511.06279 | Neural Programmer-Interpreters (NPI), Reed & de Freitas 2015 |
| 1710.01813 | Neural Task Programming (NTP), Xu et al. 2017 |
| 1803.01840 | TACO: Learning Task Decomposition via Temporal Alignment for Control, Shiarlis et al. 2018 |
| 2003.06085 | Learning to Generalize Across Long-Horizon Tasks from Human Demonstrations (GTI), Mandlekar et al. 2020 |
| 2206.11795 | Video PreTraining (VPT), Baker et al. 2022 |

Motion Grammar (Dantam & Stilman) is on the reading list but not on arXiv, so it has no questions.

## Questions

One JSON object per line:

```json
{"id": "npi-core-stop", "type": "term", "question": "How does the NPI core decide when to stop ...?",
 "papers": ["1511.06279"], "sections": ["3 Model", "3.1 Inference"], "keywords": ["threshold", "call stack"]}
```

- `papers`: the arXiv ids whose text answers the question.
- `sections`: section-title substrings, matched without case against the section's own title (the last part of its
  path, so `3 Model` matches the "3 Model" section but not its subsection "3 Model > 3.1 Inference"). A single-paper
  question gives a list; a cross-paper question gives a list per paper, `{"1511.06279": ["3.2 Training"], ...}`, and a
  hit counts only if its section is gold for its own paper. Labels name sections, never chunk ids, because chunk ids
  shift whenever the parser changes.
- `keywords`: 1 to 3 words a correct answer should contain. None may appear in the question text, so a keyword
  cannot be matched by an answer that only repeats the question.
- `type`: `factual` (a fact or number), `method` (how something works), `term` (a named component or acronym, such
  as "the NPI core", "TACO's alignment", "ACT", "IDM") or `cross` (the answer spans several papers).

| Type | Count |
| --- | --- |
| factual | 12 |
| method | 6 |
| term | 7 |
| cross | 7 |
| **total** | **32** |

Each paper has 5 single-paper questions. The 7 cross-paper questions have 2 or 3 gold papers.

`tests/test_eval_set.py` checks every label against the downloaded papers:
- each section substring matches at least 1 and at most 3 sections of its paper
- each gold paper has a gold section that contains one of the keywords
- each keyword appears in the text of some gold section

It skips when `data/papers/` is absent.

## Running it

```bash
py -3.12 -m uv run python -m app.evaluate                       # retrieval only, no LLM, free
py -3.12 -m uv run python -m app.evaluate --k 1 3 5 8 --json out.json
py -3.12 -m uv run python -m app.evaluate --answers              # one LLM call per question, asks first
py -3.12 -m uv run python -m app.evaluate --answers --yes --json out.json
```

Hybrid retrieval is scored only when the optional embedder is installed
(`py -3.12 -m uv sync --extra embed`); otherwise the run says it skipped it. With the same extra, each retriever also
gets a `+rerank` row: its top `RERANK_DEPTH` (30) hits reordered by the cross-encoder `RERANK_MODEL`. These rows are
scored whatever `RERANK` is set to, so the comparison exists before reranking is turned on.

`--answers` needs `ANTHROPIC_API_KEY` (or an `ant auth login` profile). It prints the number of calls, the model and
the effort, and asks before spending anything unless `--yes` is given. The claim check adds up to one more call per
question, at `VERIFY_EFFORT` (default `low`); `--no-verify` skips it. The JSON is rewritten after every question
(default `data/eval/answers.json`), so an interrupted run keeps what it has. A refusal or a network failure is
recorded as that question's error and the run goes on; missing credentials stop the run after saving.

## Metrics

Retrieval (no LLM), over the top `max(k)` hits of each question:

- **Paper recall@k**: some gold paper is in the top k.
- **Section recall@k**: some hit in the top k has its own section title (the last part of its path) containing a gold
  substring for that hit's paper.
- **MRR**: mean of 1 / rank of the first relevant hit (0 if none), for each of the two levels.
- **All gold papers@k** (cross-paper questions only): the share of questions where every gold paper appears in the
  top k.

Answers (`--answers`, k = 8 passages):

- **Citation precision**: share of the cited papers that are gold (questions with no citations are left out).
- **Citation recall**: share of the gold papers that are cited. An answer that says it cannot answer cites nothing
  and so counts as 0.
- **Quote found**: share of all citations whose quote was found in the cited passage.
- **Supported / Unsupported** (claim check, on unless `--no-verify`): each sentence of the answer is graded against
  every passage it marks, by a second model call. Supported is the share of graded (sentence, passage) pairs judged
  "supported"; Unsupported the share judged "unsupported" (the rest are "partial"). Pairs the check gave no verdict
  for are left out. A failed check is recorded on the question's row (`verify_error`) and does not make it an error.
- **Keyword hit**: share of a question's keywords that appear in the answer.
- **Cannot answer**: share of answers with no citations. The prompt tells the model to cite nothing when the passages
  do not answer the question, so this counts the answers where it said so.

## Baseline (retrieval)

Run on 2026-10-25 over 380 chunks from the 5 papers (all parsed from LaTeX), with the `embed` extra installed
(`intfloat/multilingual-e5-small`):

```bash
py -3.12 -m uv run python -m app.evaluate --json eval/baseline.json
```

| Retriever | Level | R@1 | R@3 | R@5 | R@8 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| bm25 | paper | 0.97 | 1.00 | 1.00 | 1.00 | 0.98 |
| bm25 | section | 0.59 | 0.94 | 0.97 | 0.97 | 0.75 |
| bm25 | all gold papers (7 cross) | 0.00 | 0.71 | 0.86 | 0.86 | n/a |
| hybrid | paper | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hybrid | section | 0.66 | 0.94 | 0.97 | 1.00 | 0.79 |
| hybrid | all gold papers (7 cross) | 0.00 | 0.71 | 0.86 | 0.86 | n/a |

Per-question ranks and hits are in `baseline.json`. Answer-level numbers are not in yet: they need an API key and come
from the `--answers` run.

How to read it:

- Paper recall is close to 1 because most questions name their paper ("NPI", "TACO", "VPT") and the paper's acronym
  appears throughout its own chunks. It says little about retrieval quality; section recall@1 is the number to watch.
- All gold papers@1 is 0 by construction: one hit cannot come from two papers. In the top 8, 6 of the 7 cross-paper
  questions find every gold paper. "How do TACO, GTI and VPT each use behavioral cloning?" never surfaces TACO,
  probably because TACO always writes "behavioural", which BM25 treats as a different word.
- Weak spots:
  - `npi-act` (what is the ACT program?) has no gold section in BM25's top 8 and only reaches rank 8 with hybrid. NPI's
    introduction and NTP's summary of NPI outrank NPI's own definition, which sits in the middle of the long
    "3.1 Inference" section.
  - `taco-dial` finds its gold section only at rank 5. The Dial domain is described in "5.3 Dial Domain", and the
    higher-ranked "5.3.1 Joint Space Dial Domain" chunks do not name the JACO arm or the PIN task.
  - The TACO method questions (`taco-sketch`, `taco-alignment`, `taco-stop`) find the right section only at rank 2,
    behind chunks that mention the same terms without defining them.

### Label history

The numbers moved three times, all on the same 32 questions; no question was reworded.

| Labels | BM25 section R@1 / R@8 / MRR | Hybrid section R@1 / R@8 / MRR |
| --- | --- | --- |
| 1. First draft | 0.59 / 0.94 / 0.73 | 0.62 / 0.97 / 0.76 |
| 2. Widened | 0.62 / 0.97 / 0.78 | 0.69 / 1.00 / 0.82 |
| 3. Tightened (current) | 0.59 / 0.97 / 0.75 | 0.66 / 1.00 / 0.79 |

1. The first draft matched substrings against the whole joined path, so a label like "3 Model" or "5 Experiments" also
   counted every subsection under it, and cross-paper labels were one shared list.
2. After the first run, four labels were widened to sections that answer the question but had been missed:
   - `vpt-kl-loss`: appendix G.1
   - `taco-sketch`: Conclusion
   - `gti-pandakitchen`: the PandaKitchen caption that the parser places in "Stage 2"
   - `x-decomposition`: NTP's Abstract, Introduction and Background

   That review looked only at misses, so it could only raise the numbers.
3. The labels were then tightened using only the paper text, never the retriever's output:
   - Substrings now match a section's own title.
   - Cross-paper questions name their gold sections per paper.
   - Each label names the most specific section whose text contains the answer.
   - The weakest keywords were replaced ("60" became "size 60", "20" became "native human interface", "stop" became "STOP action").

   The only rank that moved was `taco-dial`, from 1 to 5: its subsection "5.3.1 Joint Space Dial Domain" no longer counts.
