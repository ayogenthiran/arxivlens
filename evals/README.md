# Retrieval And Groundedness Eval

This folder provides a production-style offline evaluation harness for:

- `Recall@k` (retrieval completeness against known relevant chunks)
- `MRR` (ranking quality)
- Groundedness checks (answer support + citation validity)

## Eval Set Format (JSONL)

Each line is a JSON object:

```json
{
  "query": "What is retrieval-augmented generation?",
  "relevant_chunk_ids": ["chunk_123", "chunk_456"],
  "top_k": 8
}
```

- `query` (required): query text
- `relevant_chunk_ids` (required): chunk IDs that should be retrieved
- `top_k` (optional): query-specific `k` override

## Run Retrieval-Only Metrics

```bash
python -m evals.retrieval_eval \
  --eval-set evals/sample_eval_set.jsonl \
  --top-k 8
```

## Run Retrieval + Groundedness

Groundedness requires `OPENAI_API_KEY` in your environment.

```bash
python -m evals.retrieval_eval \
  --eval-set evals/sample_eval_set.jsonl \
  --top-k 8 \
  --run-groundedness \
  --rerank \
  --output evals/results/latest_eval.json
```

## Groundedness Metrics

- `answer_support_ratio_avg`: average fraction of answer tokens supported by retrieved context tokens
- `citation_groundedness_avg`: average fraction of citations with quote text found in cited chunk
- `grounded_pass_rate`: fraction of queries passing:
  - support ratio >= `--support-threshold` (default `0.55`)
  - citation groundedness >= `0.8`

## CI Recommendation

- Run this script on every main branch merge and nightly.
- Keep a fixed benchmark set for regressions.
- Track trendline of:
  - `Recall@k`
  - `MRR`
  - `grounded_pass_rate`
