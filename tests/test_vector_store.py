import logging

import pytest

from src.vector_store.vector_store import VectorStore


@pytest.fixture
def store():
    created = VectorStore()
    assert created.client is None
    assert created.collection is None
    return created


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Ada Lovelace and Alan Turing", ["Ada Lovelace", "Alan Turing"]),
        ("Doe, Jane and Roe, Richard", ["Doe, Jane", "Roe, Richard"]),
        ("Ada Lovelace, Alan Turing", ["Ada Lovelace", "Alan Turing"]),
        ([" Ada Lovelace ", "", "Alan Turing"], ["Ada Lovelace", "Alan Turing"]),
    ],
)
def test_normalize_authors(store, raw, expected):
    assert store._normalize_chunk_metadata({"authors": raw})["authors"] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cs.CL   cs.AI", ["cs.CL", "cs.AI"]),
        ([" cs.CL ", "", "cs.AI"], ["cs.CL", "cs.AI"]),
    ],
)
def test_normalize_categories(store, raw, expected):
    assert store._normalize_chunk_metadata({"categories": raw})["categories"] == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024", 2024),
        (2024, 2024),
        (None, None),
    ],
)
def test_normalize_year(store, raw, expected):
    assert store._normalize_chunk_metadata({"year": raw})["year"] == expected


def test_normalize_keeps_other_fields_and_does_not_mutate_input(store):
    raw = {
        "doc_id": "1706.03762",
        "title": "Attention",
        "authors": "A and B",
        "categories": "cs.CL",
        "year": "2017",
    }
    normalized = store._normalize_chunk_metadata(raw)
    assert normalized["doc_id"] == "1706.03762"
    assert normalized["title"] == "Attention"
    assert normalized["authors"] == ["A", "B"]
    assert normalized["categories"] == ["cs.CL"]
    assert normalized["year"] == 2017
    assert raw["authors"] == "A and B"
    assert raw["year"] == "2017"


@pytest.mark.parametrize(
    ("metadata", "filters", "expected"),
    [
        ({"categories": "cs.CL cs.LG"}, {"categories": ["cs.LG"]}, True),
        ({"categories": ["cs.CL", "cs.AI"]}, {"categories": ["cs.AI"]}, True),
        ({"categories": ["cs.CL"]}, {"categories": ["cs.AI"]}, False),
        ({"authors": "Grace Hopper and John Smith"}, {"authors": ["hop"]}, True),
        ({"authors": ["Grace Hopper"]}, {"authors": ["HOPPER"]}, True),
        ({"authors": ["Grace Hopper"]}, {"authors": ["turing"]}, False),
        ({"year": 2020}, {"year_min": 2020, "year_max": 2020}, True),
        ({"year": "2020"}, {"year_min": 2019, "year_max": 2020}, True),
        ({"year": 2019}, {"year_min": 2020}, False),
        ({"year": 2021}, {"year_max": 2020}, False),
        ({"year": None}, {"year_min": 2020, "year_max": 2024}, False),
        ({"categories": "cs.CL", "authors": "Ada Lovelace", "year": 2017}, None, True),
        ({"categories": "cs.AI", "year": 2020}, {"categories": ["cs.CL"], "year_min": 2010}, False),
    ],
)
def test_metadata_matches_filters(store, metadata, filters, expected):
    assert store._metadata_matches_filters(metadata, filters) is expected


def test_take_best_chunk_keeps_one_per_paper_in_first_seen_order(store):
    ranked = [
        {"id": "b-low", "metadata": {"doc_id": "paper-b"}, "vector_score": 0.1},
        {"id": "a-low", "metadata": {"doc_id": "paper-a"}, "vector_score": 0.2},
        {"id": "b-high", "metadata": {"doc_id": "paper-b"}, "vector_score": 0.9},
        {"id": "a-tie", "metadata": {"doc_id": "paper-a"}, "vector_score": 0.2},
        {"id": "c-only", "metadata": {"doc_id": "paper-c"}, "vector_score": 0.8},
    ]
    kept = store._take_best_chunk_per_paper(ranked, n_results=2, score_key="vector_score")
    assert [item["id"] for item in kept] == ["b-high", "a-low"]


def test_take_best_chunk_keeps_chunks_without_a_doc_id(store):
    ranked = [
        {"id": "loose-1", "metadata": {}, "vector_score": 0.4},
        {"id": "loose-2", "metadata": {"doc_id": ""}, "vector_score": 0.9},
    ]
    kept = store._take_best_chunk_per_paper(ranked, n_results=5, score_key="vector_score")
    assert [item["id"] for item in kept] == ["loose-1", "loose-2"]


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (None, None),
        ({}, None),
        ({"categories": ["cs.CL"], "authors": ["Hopper"]}, None),
        ({"year_min": "2018"}, None),
        ({"year_min": 2018}, {"year": {"$gte": 2018}}),
        ({"year_max": 2022}, {"year": {"$lte": 2022}}),
        (
            {"year_min": 2018, "year_max": 2022, "categories": ["cs.CL"]},
            {"$and": [{"year": {"$gte": 2018}}, {"year": {"$lte": 2022}}]},
        ),
    ],
)
def test_build_where_clause(store, filters, expected):
    assert store._build_where_clause(filters) == expected


class _FakeCollection:
    def __init__(self, distances, metadata):
        self.metadata = metadata
        self.distances = list(distances)

    def query(self, **kwargs):
        count = len(self.distances)
        return {
            "ids": [[f"id-{index}" for index in range(count)]],
            "documents": [[f"doc-{index}" for index in range(count)]],
            "metadatas": [[{"doc_id": f"paper-{index}"} for index in range(count)]],
            "distances": [self.distances],
        }


class _FakeClient:
    def __init__(self, existing=None):
        self.existing = existing
        self.created_kwargs = None
        self.deleted = False

    def get_collection(self, name):
        if self.existing is None:
            raise RuntimeError(f"missing collection {name}")
        return self.existing

    def create_collection(self, name, metadata=None):
        self.created_kwargs = {"name": name, "metadata": metadata}
        return _FakeCollection(distances=[], metadata=dict(metadata or {}))

    def delete_collection(self, name):
        self.deleted = True


def test_new_collection_uses_cosine_space(store, caplog):
    client = _FakeClient()
    store.client = client
    with caplog.at_level(logging.WARNING, logger="src.vector_store.vector_store"):
        collection = store._get_or_create_collection()

    assert client.created_kwargs == {
        "name": store.collection_name,
        "metadata": {"hnsw:space": "cosine"},
    }
    assert collection.metadata == {"hnsw:space": "cosine"}
    assert client.deleted is False
    assert "re-ingest" not in caplog.text.lower()


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"hnsw:space": "l2"}, {"hnsw:space": "ip"}],
)
def test_existing_non_cosine_collection_warns_and_is_kept(store, caplog, metadata):
    existing = _FakeCollection(distances=[0.2], metadata=metadata)
    client = _FakeClient(existing=existing)
    store.client = client
    with caplog.at_level(logging.WARNING, logger="src.vector_store.vector_store"):
        opened = store._get_or_create_collection()

    assert opened is existing
    assert client.created_kwargs is None
    assert client.deleted is False
    assert "re-ingest" in caplog.text.lower()


def test_existing_cosine_collection_does_not_warn(store, caplog):
    existing = _FakeCollection(distances=[0.2], metadata={"hnsw:space": "Cosine"})
    client = _FakeClient(existing=existing)
    store.client = client
    with caplog.at_level(logging.WARNING, logger="src.vector_store.vector_store"):
        opened = store._get_or_create_collection()

    assert opened is existing
    assert client.created_kwargs is None
    assert "re-ingest" not in caplog.text.lower()


@pytest.mark.parametrize(
    ("metadata", "expected_scores"),
    [
        ({"hnsw:space": "cosine"}, [0.8, 1.0, -0.5, 0.5]),
        ({"hnsw:space": "l2"}, [1.0 / 1.2, 1.0, 1.0 / 2.5, 1.0 / 1.5]),
        (None, [1.0 / 1.2, 1.0, 1.0 / 2.5, 1.0 / 1.5]),
    ],
)
def test_vector_score_follows_collection_space(store, metadata, expected_scores):
    distances = [0.2, 0.0, 1.5, 0.5]
    store.collection = _FakeCollection(distances, metadata)

    results = store._vector_search([1.0, 0.0], n_results=len(distances))

    assert [item["distance"] for item in results] == distances
    assert [item["vector_score"] for item in results] == pytest.approx(expected_scores)


def test_missing_distance_scores_zero(store):
    store.collection = _FakeCollection([None], metadata={"hnsw:space": "cosine"})

    results = store._vector_search([1.0, 0.0], n_results=1)

    assert results[0]["vector_score"] == 0.0


def test_ranking_order_is_unchanged_between_cosine_and_l2(store):
    distances = [0.4, 0.1, 1.2, 0.0]
    closest_first = [f"id-{index}" for index in sorted(range(len(distances)), key=lambda index: distances[index])]

    ranked_ids = {}
    for metadata in ({"hnsw:space": "cosine"}, {"hnsw:space": "l2"}, None):
        store.collection = _FakeCollection(distances, metadata)
        ranked = store.query([1.0, 0.0], n_results=len(distances))
        ranked_ids[str(metadata)] = [item["id"] for item in ranked]

    assert {tuple(ids) for ids in ranked_ids.values()} == {tuple(closest_first)}
