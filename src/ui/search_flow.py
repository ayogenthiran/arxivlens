"""Search papers first, then ask the model to summarize those chunks."""

from collections.abc import Callable

import requests

from src.settings import API_BASE_URL, API_QUERY_TIMEOUT_S

SEARCH_URL = f"{API_BASE_URL}/api/search"
ANSWER_URL = f"{API_BASE_URL}/api/answer"
ANSWER_FAILURE_MESSAGE = "Could not generate the summary. The papers below are still available."


def search_papers(query: str, top_k: int, filters: dict) -> dict:
    response = requests.post(
        SEARCH_URL,
        json={"query": query, "top_k": top_k, **filters},
        timeout=API_QUERY_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def request_answer(query: str, context_chunks: list) -> dict:
    response = requests.post(
        ANSWER_URL,
        json={"query": query, "context_chunks": context_chunks},
        timeout=API_QUERY_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def search_then_answer(
    query: str,
    top_k: int,
    filters: dict,
    on_papers: Callable[[dict], None],
    *,
    search: Callable[[str, int, dict], dict] = search_papers,
    answer: Callable[[str, list], dict] = request_answer,
) -> tuple[dict, dict | None, str | None]:
    """Retrieve papers, hand them to the UI, then summarize. A failed summary keeps the papers."""
    papers = search(query, top_k, filters)
    on_papers(papers)
    chunks = papers.get("context_chunks") or []
    if not isinstance(chunks, list):
        chunks = []
    try:
        summary = answer(query, chunks)
    except (requests.RequestException, ValueError):
        return papers, None, ANSWER_FAILURE_MESSAGE
    return papers, summary, None
