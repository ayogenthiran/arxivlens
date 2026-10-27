"""The UI asks for papers before it asks for a summary."""

from pathlib import Path

import pytest
import requests

from src.ui.search_flow import ANSWER_FAILURE_MESSAGE, search_then_answer

ROOT = Path(__file__).resolve().parents[1]


def test_papers_are_delivered_before_the_answer_call():
    order = []

    def search(query, top_k, filters):
        order.append("search")
        assert query == "topic"
        assert top_k == 4
        assert filters["categories"] == ["cs.CL"]
        return {"context_chunks": [{"id": "a", "text": "alpha"}], "facets": {}, "query": query}

    def on_papers(papers):
        order.append("papers")
        assert papers["context_chunks"] == [{"id": "a", "text": "alpha"}]

    def answer(query, chunks):
        order.append("answer")
        assert query == "topic"
        assert chunks == [{"id": "a", "text": "alpha"}]
        return {"answer": "ok", "summary": "ok", "points": [], "citations": []}

    papers, summary, error = search_then_answer(
        "topic",
        4,
        {"categories": ["cs.CL"]},
        on_papers,
        search=search,
        answer=answer,
    )

    assert order == ["search", "papers", "answer"]
    assert error is None
    assert summary["answer"] == "ok"
    assert papers["query"] == "topic"


def test_answer_failure_keeps_the_papers():
    def search(query, top_k, filters):
        return {"context_chunks": [{"id": "a"}], "facets": {}, "query": query}

    def answer(query, chunks):
        raise requests.ConnectionError("down")

    seen = []
    papers, summary, error = search_then_answer(
        "topic",
        3,
        {},
        seen.append,
        search=search,
        answer=answer,
    )

    assert seen[0]["context_chunks"] == [{"id": "a"}]
    assert papers["context_chunks"] == [{"id": "a"}]
    assert summary is None
    assert error == ANSWER_FAILURE_MESSAGE


def test_search_failure_propagates():
    def search(query, top_k, filters):
        raise requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        search_then_answer("topic", 3, {}, lambda papers: None, search=search, answer=lambda query, chunks: {})


def test_client_posts_search_then_answer(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        if url.endswith("/api/search"):
            return Response({"context_chunks": [{"id": "a"}], "facets": {}, "query": json["query"]})
        return Response({"answer": "ok", "summary": "ok", "points": [], "citations": []})

    monkeypatch.setattr("src.ui.search_flow.requests.post", fake_post)

    _papers, summary, error = search_then_answer(
        "topic",
        4,
        {"authors": ["Hopper"], "categories": [], "year_min": None, "year_max": 2024},
        lambda papers: None,
    )

    assert calls[0][0].endswith("/api/search")
    assert calls[0][1]["query"] == "topic"
    assert calls[0][1]["top_k"] == 4
    assert calls[0][1]["authors"] == ["Hopper"]
    assert calls[0][1]["year_max"] == 2024
    assert calls[1][0].endswith("/api/answer")
    assert calls[1][1] == {"query": "topic", "context_chunks": [{"id": "a"}]}
    assert error is None
    assert summary["summary"] == "ok"
    assert calls[0][2] >= 1


def test_summary_expander_is_open_above_the_papers():
    source = (ROOT / "streamlit_app.py").read_text()
    summary_slot = source.index("summary_slot = st.empty()")
    papers_slot = source.index("papers_slot = st.empty()")
    render_papers = source.index("render_papers(papers,")
    answer_call = source.index("answer=answer_with_spinner")

    assert 'SUMMARY_TITLE = "AI summary of these papers"' in source
    assert "st.expander(SUMMARY_TITLE, expanded=True)" in source
    assert "render_summary_error" in source
    assert summary_slot < papers_slot
    assert render_papers < answer_call
    assert "/api/query" not in source
