# ArxivLens to-do

Nothing is open.

The product is a paper search tool for research students. The live corpus is 5,000 papers from 2024 onward in `cs.CL` and `cs.LG`, stored in `data/vector_store_corpus`. Each paper is one chunk: title, then the full abstract. The full archive in `data/vector_store` stays untouched and vector-only.

Grow the corpus later with `MAX_DOCS`, `INGEST_CATEGORIES`, and `INGEST_YEAR_MIN`. Keep it under 50,000 papers so keyword search stays on.

Query rewriting stays off (`ENABLE_QUERY_REWRITE=false`). The answer model stays `gpt-4o-mini`.

## Done

1. One paper per result. A top-5 answer lists five different paper ids.
2. Category and author filters run before the final cut. A `cs.CL` filter still returns five chunks, and each one includes `cs.CL`.
3. The reranker sees a wider set (24) and the answer still returns `top_k` chunks.
4. The live corpus is 5,000 recent `cs.CL` / `cs.LG` papers, one chunk each, with clean category codes and author names.
5. Keyword and vector ranks are fused with reciprocal rank fusion. `hybrid_score` stays that fusion score. `reranker_score` is separate. The reranker stays on.
6. Each citation includes the arXiv id (`metadata.doc_id`), chunk id, and title. The prompt still uses `[Document N]`. The app opens `https://arxiv.org/abs/{id}` from that id. A citation is kept only when its quote appears in that chunk.
7. Chunks with a raw reranker score below `RERANKER_MIN_RAW_SCORE` (1) are dropped before the model writes. A score near 0 goes away. A score around 5 stays. If nothing remains, the API skips the model and returns “I do not have enough information in the provided context.” with no citations.
8. Query rewriting stays off. On “What is retrieval-augmented generation?” the rewrite lowered the top paper’s raw score and replaced a stronger paper with a weaker one. A loosely worded question was rescued only by changing its meaning. `ENABLE_QUERY_REWRITE` stays `false`. The answer model stays `gpt-4o-mini`.
9. Ten abstract questions in `evals/abstract_questions.jsonl` score Recall@8 = 1.0 and MRR = 1.0 in `evals/results/latest_eval.json`. Each gold paper is rank 1. The embedding stays `all-MiniLM-L6-v2`. The answer model stays `gpt-4o-mini`.

## Notes

- Keyword search loads every chunk into memory, so it stays off for the full archive.
- `RRF_K=60` controls rank fusion. Higher values make the top ranks matter less.
- The confidence label in the app uses `reranker_score`, which runs from 0 to 1. `hybrid_score` is the smaller fusion score and is not a confidence percentage.
