# Retrieval eval

`retrieval_eval.py` scores a JSONL query set against the local Chroma index. It reads the index and does not write to it.

## What it measures

Each query is embedded, then searched with the same vector lookup the app uses. Retrieved chunk ids are compared with the gold ids on that line.

- **Recall@k.** Share of gold chunk ids present in that query's top k. The printed mean is labeled `recall@{k}` using the CLI `--top-k` (default 8, from `UI_DEFAULT_TOP_K`). A line's own `top_k` is the depth used for that query.
- **MRR.** Mean reciprocal rank of the first gold chunk in the retrieved list. A query with no gold hit scores 0.

`--rerank` runs the cross-encoder on the retrieved list before those scores are computed.

`--run-groundedness` also asks the chat model for an answer and adds three scores:

- **Answer support ratio.** Fraction of answer tokens that also appear in the retrieved chunk text.
- **Citation groundedness.** Fraction of citations whose quote appears in the cited chunk.
- **Grounded pass rate.** Share of queries where answer support is at least `EVAL_SUPPORT_THRESHOLD` (default 0.55) and citation groundedness is at least `CITATION_GROUNDEDNESS_MIN` (default 0.8).

## How to run

The vector store must already be ingested. The default path is `data/vector_store` (`VECTOR_STORE_DIR`). The embedding model downloads on first use. `--run-groundedness` needs `OPENAI_API_KEY`.

```bash
python evals/retrieval_eval.py --eval-set evals/abstract_questions.jsonl
```

`--eval-set` is required. Write the summary and per-query rows as JSON with `--output`:

```bash
python evals/retrieval_eval.py \
  --eval-set evals/abstract_questions.jsonl \
  --output evals/results/latest_eval.json
```

Other flags: `--vector-store-dir`, `--top-k`, `--rerank`, `--run-groundedness`, `--support-threshold`, `--citation-threshold`.

## Eval file format

UTF-8 JSONL, one JSON object per line. Blank lines are skipped.

| Field | Required | Meaning |
| --- | --- | --- |
| `query` | yes | Non-empty query string |
| `relevant_chunk_ids` | yes | Non-empty list of gold chunk ids (`{arxiv id}-{index}`, for example `2401.02329-0`) |
| `top_k` | no | Retrieval depth for that query. Defaults to `DEFAULT_TOP_K` |

```json
{"query":"Which paper discusses FedVLS for vacant classes in label-skewed federated learning?","relevant_chunk_ids":["2401.02329-0"],"top_k":8}
```

The checked-in set is `abstract_questions.jsonl`.

## Limitations

`abstract_questions.jsonl` queries contain each paper's unique title terms, so it is a known-item lookup sanity check, not a measure of discovery quality. A perfect score shows the index can return that paper when the query already includes those terms.
