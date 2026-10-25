# arXiv paper RAG

Ask questions over a reading list of arXiv papers and get answers that cite the exact paper, section and passage.

The bar is [PaperQA2](https://github.com/Future-House/paper-qa). The angle here is arXiv-native: parse the LaTeX
source instead of the PDF, support reading-list and digest modes, and open every citation at the highlighted passage
in [semantic-pdf-reader](https://github.com/fredwang1012/semantic-pdf-reader).

## Plan

| Stage | Done when |
| --- | --- |
| MVP | Answers over a fixed arXiv reading list with correct paper and section citations, plus baseline numbers on a ~30-question eval. |
| Better than plain RAG | Agentic retrieval, LaTeX parsing, reranking and citation verification, compared against plain RAG and PaperQA2 on the same eval. |
| Reader integration | Citations open the PDF at the highlighted passage. |
| Ship | Public deployment, writeup with a results table, demo video. |

## Status

Not started. Retrieval will build on semantic-pdf-reader's hybrid index.
