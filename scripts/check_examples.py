"""Run the empty-state examples against the live API.

Start the API first (``python main.py``). This must pass after any corpus change.
A query fails when ``POST /api/search`` returns fewer than ``MIN_EXAMPLE_PAPERS``
or when the top ``reranker_score_raw`` is below
``RERANKER_MIN_RAW_SCORE + EXAMPLE_RAW_SCORE_MARGIN``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402

from src.settings import (  # noqa: E402
    DEFAULT_TOP_K,
    MAX_QUERY_TOP_K,
    RERANKER_MIN_RAW_SCORE,
)
from src.ui.examples import (  # noqa: E402
    EXAMPLE_QUESTIONS,
    MIN_EXAMPLE_PAPERS,
    example_filter_suffix,
    example_result_problem,
    example_retrieval_filters,
    top_reranker_raw,
)
from src.ui.search_flow import search_papers  # noqa: E402


def _top_k() -> int:
    return min(MAX_QUERY_TOP_K, max(DEFAULT_TOP_K, MIN_EXAMPLE_PAPERS))


def _format_raw(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _query_label(example: dict) -> str:
    query = str(example.get("query") or "")
    suffix = example_filter_suffix(example.get("filters") or {})
    if suffix:
        return f"{query} · {suffix}"
    return query


def _print_table(rows: list[tuple[str, str, str, str]]) -> None:
    headers = ("papers", "top_raw", "status", "query")
    papers_w = max(len(headers[0]), *(len(row[0]) for row in rows))
    raw_w = max(len(headers[1]), *(len(row[1]) for row in rows))
    status_w = max(len(headers[2]), *(len(row[2]) for row in rows))
    print(f"{headers[0]:>{papers_w}}  {headers[1]:>{raw_w}}  {headers[2]:<{status_w}}  {headers[3]}")
    for papers, top_raw, status, query in rows:
        print(f"{papers:>{papers_w}}  {top_raw:>{raw_w}}  {status:<{status_w}}  {query}")


def main() -> int:
    failed = False
    rows: list[tuple[str, str, str, str]] = []
    top_k = _top_k()
    for example in EXAMPLE_QUESTIONS:
        label = _query_label(example)
        try:
            payload = search_papers(
                str(example.get("query") or ""),
                top_k,
                example_retrieval_filters(example.get("filters") or {}),
            )
        except requests.RequestException as exc:
            rows.append(("-", "-", "request failed", f"{label} ({exc})"))
            failed = True
            continue
        chunks = payload.get("context_chunks") if isinstance(payload, dict) else None
        if not isinstance(chunks, list):
            chunks = []
        top_raw = top_reranker_raw(chunks)
        problem = example_result_problem(len(chunks), top_raw, RERANKER_MIN_RAW_SCORE)
        rows.append((str(len(chunks)), _format_raw(top_raw), problem or "ok", label))
        failed = failed or problem is not None
    _print_table(rows)
    if failed:
        print("Example check failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
