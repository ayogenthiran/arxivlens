"""Split search and answer routes share one retrieval pipeline."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.api.server import APIServer
from src.llm.llm_interface import INSUFFICIENT_CONTEXT_ANSWER, LLMInterface

CHUNKS = [
    {
        "id": "1706.03762-0",
        "text": "Attention is all you need for sequence transduction.",
        "metadata": {
            "doc_id": "1706.03762",
            "title": "Attention Is All You Need",
            "chunk_id": "1706.03762-0",
        },
    },
    {
        "id": "1810.04805-0",
        "text": "BERT pretrains deep bidirectional representations.",
        "metadata": {"doc_id": "1810.04805", "title": "BERT"},
    },
]

SCRIPTED_ANSWER = json.dumps(
    {
        "summary": "Transformers use attention.",
        "points": [
            {
                "title": "Attention",
                "detail": "Sequence models use attention.",
                "doc_ids": [1, 9],
            }
        ],
        "citations": [
            {"doc_id": 1, "quote": "Attention is all you need"},
            {"doc_id": 1, "quote": "not in the chunk"},
            {"doc_id": 9, "quote": "Attention is all you need"},
        ],
    }
)

QUERY = {
    "query": "How does attention work?",
    "top_k": 2,
    "authors": ["Vaswani"],
    "categories": ["cs.CL"],
    "year_min": 2023,
    "year_max": 2024,
}

ATTENTION_CITATION = {
    "doc_id": 1,
    "arxiv_id": "1706.03762",
    "chunk_id": "1706.03762-0",
    "title": "Attention Is All You Need",
    "quote": "Attention is all you need",
}


class FakeEmbedding:
    def __init__(self):
        self.queries = []
        self.model = object()

    def load_model(self):
        return None

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.queries = []
        self.by_id_calls = []
        self.collection = object()

    def initialize(self):
        return None

    def query(self, query_embedding, query_text, n_results, filters):
        self.queries.append(
            {
                "query_embedding": query_embedding,
                "query_text": query_text,
                "n_results": n_results,
                "filters": filters,
            }
        )
        return list(self.chunks)

    def get_facets(self, chunks):
        return {
            "years": [{"value": 2017, "count": len(chunks)}],
            "categories": [],
            "authors": [],
        }

    def get_chunks_by_ids(self, chunk_ids):
        self.by_id_calls.append(list(chunk_ids))
        by_id = {chunk["id"]: chunk for chunk in self.chunks}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


class FakeReranker:
    def __init__(self):
        self.max_candidates = 6
        self.calls = []

    def rerank(self, query_text, candidates, final_top_k):
        self.calls.append(
            {
                "query_text": query_text,
                "count": len(candidates),
                "final_top_k": final_top_k,
            }
        )
        return list(candidates)[:final_top_k]


class RecordingLLM(LLMInterface):
    def __init__(self):
        super().__init__()
        self.events = []
        self.raw = SCRIPTED_ANSWER
        self.generate_chunks = None

    def rewrite_query(self, query: str) -> str:
        self.events.append(("rewrite", query))
        return super().rewrite_query(query)

    def generate_response(self, query, context_chunks):
        self.events.append(("generate", query))
        self.generate_chunks = context_chunks
        return super().generate_response(query, context_chunks)

    def _call_llm_api(self, **kwargs):
        self.events.append(("api", kwargs.get("prompt", "")))
        return self.raw


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr("src.llm.llm_interface.ENABLE_QUERY_REWRITE", False)
    embedding = FakeEmbedding()
    store = FakeVectorStore(CHUNKS)
    reranker = FakeReranker()
    llm = RecordingLLM()
    server = APIServer(embedding, store, llm, reranker)
    with TestClient(server.app) as client:
        yield client, embedding, store, reranker, llm


def _event_names(llm):
    return [name for name, _query in llm.events]


def test_search_never_calls_the_llm(api, monkeypatch):
    monkeypatch.setattr("src.llm.llm_interface.ENABLE_QUERY_REWRITE", True)
    client, embedding, store, reranker, llm = api

    response = client.post("/api/search", json=QUERY)

    assert response.status_code == 200
    assert llm.events == []
    assert embedding.queries == [QUERY["query"]]
    assert store.queries[0]["query_text"] == QUERY["query"]
    assert store.queries[0]["n_results"] == 6
    assert store.queries[0]["filters"] == {
        "authors": ["Vaswani"],
        "categories": ["cs.CL"],
        "year_min": 2023,
        "year_max": 2024,
    }
    assert reranker.calls[0]["final_top_k"] == 2
    assert reranker.calls[0]["query_text"] == QUERY["query"]
    assert response.json() == {
        "context_chunks": CHUNKS,
        "facets": {"years": [{"value": 2017, "count": 2}], "categories": [], "authors": []},
        "query": QUERY["query"],
    }


def test_query_output_is_unchanged(api):
    client, embedding, _store, _reranker, llm = api

    response = client.post("/api/query", json=QUERY)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Transformers use attention.\n1. Attention: Sequence models use attention.",
        "summary": "Transformers use attention.",
        "points": [
            {
                "title": "Attention",
                "detail": "Sequence models use attention.",
                "sources": [
                    {
                        "doc_id": 1,
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                    }
                ],
            }
        ],
        "citations": [ATTENTION_CITATION],
        "context_chunks": CHUNKS,
        "facets": {"years": [{"value": 2017, "count": 2}], "categories": [], "authors": []},
        "query": QUERY["query"],
        "rewritten_query": QUERY["query"],
    }
    assert _event_names(llm) == ["rewrite", "generate", "api"]
    assert embedding.queries == [QUERY["query"]]
    assert llm.generate_chunks == CHUNKS


def test_query_retrieves_with_rewritten_text_and_answers_the_original(api):
    client, embedding, store, reranker, llm = api

    def rewrite(query):
        llm.events.append(("rewrite", query))
        return "dense retrieval of attention"

    llm.rewrite_query = rewrite
    response = client.post("/api/query", json={"query": QUERY["query"]})

    assert response.status_code == 200
    body = response.json()
    assert body["rewritten_query"] == "dense retrieval of attention"
    assert body["query"] == QUERY["query"]
    assert body["citations"] == [ATTENTION_CITATION]
    assert embedding.queries == ["dense retrieval of attention"]
    assert store.queries[0]["query_text"] == "dense retrieval of attention"
    assert reranker.calls[0]["query_text"] == "dense retrieval of attention"
    assert [query for name, query in llm.events if name == "generate"] == [QUERY["query"]]


def test_answer_validates_citations_against_supplied_chunks(api):
    client, embedding, store, reranker, llm = api
    supplied = [
        {
            "id": "custom-0",
            "text": "A custom sentence supplied by the client.",
            "metadata": {
                "doc_id": "9999.00001",
                "title": "Custom Paper",
                "chunk_id": "custom-0",
            },
        }
    ]
    llm.raw = json.dumps(
        {
            "summary": "The client supplied a sentence.",
            "points": [],
            "citations": [
                {"doc_id": 1, "quote": "custom sentence supplied"},
                {"doc_id": 1, "quote": "Attention is all you need"},
                {"doc_id": 2, "quote": "custom sentence supplied"},
            ],
        }
    )

    response = client.post(
        "/api/answer",
        json={
            "query": "What was supplied?",
            "chunk_ids": ["1706.03762-0"],
            "context_chunks": supplied,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The client supplied a sentence.",
        "summary": "The client supplied a sentence.",
        "points": [],
        "citations": [
            {
                "doc_id": 1,
                "arxiv_id": "9999.00001",
                "chunk_id": "custom-0",
                "title": "Custom Paper",
                "quote": "custom sentence supplied",
            }
        ],
    }
    assert store.by_id_calls == []
    assert embedding.queries == []
    assert reranker.calls == []
    assert llm.generate_chunks == supplied


def test_answer_loads_chunk_ids_when_chunks_are_omitted(api):
    client, _embedding, store, _reranker, llm = api
    llm.raw = json.dumps(
        {
            "summary": "Transformers use attention.",
            "points": [],
            "citations": [
                {"doc_id": 1, "quote": "Attention is all you need"},
                {"doc_id": 1, "quote": "BERT pretrains deep bidirectional representations."},
            ],
        }
    )

    response = client.post(
        "/api/answer",
        json={"query": "attention", "chunk_ids": ["1706.03762-0", "missing"]},
    )

    assert response.status_code == 200
    assert response.json()["citations"] == [ATTENTION_CITATION]
    assert store.by_id_calls == [["1706.03762-0", "missing"]]
    assert [chunk["id"] for chunk in llm.generate_chunks] == ["1706.03762-0"]


def test_answer_with_empty_chunks_skips_the_model(api):
    client, _embedding, store, _reranker, llm = api

    response = client.post(
        "/api/answer",
        json={"query": "attention", "context_chunks": [], "chunk_ids": ["1706.03762-0"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert body["summary"] == INSUFFICIENT_CONTEXT_ANSWER
    assert body["points"] == []
    assert body["citations"] == []
    assert llm.events == []
    assert store.by_id_calls == []


def test_answer_requires_chunks_or_ids(api):
    client, *_rest = api

    response = client.post("/api/answer", json={"query": "attention"})

    assert response.status_code == 422


def test_search_rejects_an_inverted_year_range(api):
    client, *_rest = api

    response = client.post("/api/search", json={"query": "attention", "year_min": 2024, "year_max": 2023})

    assert response.status_code == 422


def test_query_skips_the_model_when_no_chunks_remain(api):
    client, _embedding, _store, reranker, llm = api

    def drop_all(query_text, candidates, final_top_k):
        reranker.calls.append({"query_text": query_text, "count": len(candidates), "final_top_k": final_top_k})
        return []

    reranker.rerank = drop_all
    response = client.post("/api/query", json={"query": "nothing here"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert body["citations"] == []
    assert body["context_chunks"] == []
    assert body["rewritten_query"] == "nothing here"
    assert "generate" not in _event_names(llm)
    assert "rewrite" in _event_names(llm)


def test_stage_timings_are_logged(api, caplog):
    client, *_rest = api

    with caplog.at_level(logging.INFO, logger="src.api.server"):
        assert client.post("/api/search", json={"query": "How does attention work?"}).status_code == 200
        assert client.post("/api/query", json={"query": "How does attention work?"}).status_code == 200
        answer = client.post(
            "/api/answer",
            json={"query": "How does attention work?", "context_chunks": CHUNKS},
        )
        assert answer.status_code == 200

    assert "Search served in" in caplog.text
    assert "Query served in" in caplog.text
    assert "rewrite=" in caplog.text
    assert "embed=" in caplog.text
    assert "retrieval=" in caplog.text
    assert "rerank=" in caplog.text
    assert "generation=" in caplog.text
    assert "Answer served in" in caplog.text
