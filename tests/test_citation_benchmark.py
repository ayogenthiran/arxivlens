"""Citation benchmark builder, exercised with fake Semantic Scholar responses."""

import json
from datetime import date

import pytest

from scripts.build_citation_benchmark import (
    S2_BATCH_URL,
    S2_FIELDS,
    BuiltQuery,
    S2Client,
    benchmark_stats,
    build_queries,
    collect_queries,
    months_before,
    papers_from_cache,
    query_candidates,
    relevant_paper_ids,
    seeded_candidate_order,
    select_corpus,
    summarize_references,
)


class _Response:
    def __init__(self, status_code: int, text: str, headers: dict | None = None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


def _clocked_client(cache_dir, post, **kwargs) -> tuple[S2Client, dict]:
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    def sleep(seconds: float) -> None:
        now["t"] += seconds
        now.setdefault("slept", []).append(seconds)

    client = S2Client(cache_dir=cache_dir, post=post, sleep=sleep, clock=clock, **kwargs)
    return client, now


def test_select_corpus_uses_fixed_year_category_and_cap():
    documents = [
        {"id": "2301.00001", "categories": ["cs.CL"], "year": 2022},
        {"id": "2212.00001", "categories": ["cs.LG"], "year": 2023},
        {"id": "2302.00002", "categories": ["cs.AI"], "year": 2023},
        {"id": "2303.00003", "categories": ["cs.AI", "cs.CL"], "year": 2024},
        {"id": "2304.00004", "categories": ["cs.LG"], "year": "2024"},
        {"id": "", "categories": ["cs.CL"], "year": 2024},
    ]

    assert [doc["id"] for doc in select_corpus(documents, max_docs=10)] == [
        "2212.00001",
        "2303.00003",
    ]
    assert [doc["id"] for doc in select_corpus(documents, max_docs=1)] == ["2212.00001"]


def test_query_candidates_are_the_most_recent_six_months():
    papers = [
        {"id": "early", "submission_date": "2023-07-14"},
        {"id": "cutoff", "submission_date": "2023-07-15"},
        {"id": "latest", "submission_date": "2024-01-15"},
        {"id": "undated", "submission_date": None},
    ]

    assert [paper["id"] for paper in query_candidates(papers)] == ["cutoff", "latest"]
    assert months_before(date(2024, 8, 31), 6) == date(2024, 2, 29)
    assert months_before(date(2023, 8, 31), 6) == date(2023, 2, 28)


def test_relevant_filtering_reads_s2_external_ids():
    paper = {
        "externalIds": {"ArXiv": "2406.00001"},
        "references": [
            {"paperId": "a", "externalIds": {"ArXiv": "2301.00001v2"}},
            {"paperId": "b", "externalIds": {"DOI": "10.1000/xyz"}},
            {"paperId": "c", "externalIds": {"ArXiv": "https://arxiv.org/abs/2306.00006"}},
            {"paperId": "d", "externalIds": None},
            None,
            {"citedPaper": {"externalIds": {"ArXiv": "2302.00002"}}},
            {"externalIds": {"arXiv": "2301.00001"}},
        ],
    }

    summary = summarize_references(paper, {"2301.00001", "2302.00002"})

    assert summary.arxiv_ids == ["2301.00001", "2306.00006", "2302.00002", "2301.00001"]
    assert summary.total == 7
    assert summary.on_arxiv == 4
    assert summary.in_corpus == 3


def test_date_filter_keeps_only_earlier_corpus_papers():
    corpus_ids = {"2301.00001", "2306.00001", "2401.00001", "undated"}
    corpus_dates = {
        "2301.00001": date(2023, 1, 1),
        "2306.00001": date(2023, 6, 1),
        "2401.00001": date(2024, 1, 1),
        "undated": None,
    }
    reference_ids = [
        "2301.00001",
        "2301.00001",
        "2306.00001",
        "2401.00001",
        "2406.00001",
        "not-in-corpus",
        "undated",
        "query",
    ]

    assert relevant_paper_ids("query", date(2024, 1, 1), reference_ids, corpus_ids, corpus_dates) == [
        "2301.00001",
        "2306.00001",
    ]
    assert relevant_paper_ids("query", date(2023, 6, 1), reference_ids, corpus_ids, corpus_dates) == [
        "2301.00001",
    ]


def test_min_three_relevant_papers():
    corpus_ids = {"2301.00001", "2302.00002", "2303.00003", "2407.00007"}
    corpus_dates = {
        "2301.00001": date(2023, 1, 1),
        "2302.00002": date(2023, 2, 1),
        "2303.00003": date(2023, 3, 1),
        "2407.00007": date(2024, 7, 1),
        "kept": date(2024, 6, 1),
        "short": date(2024, 6, 1),
    }
    candidates = [
        {
            "id": "kept",
            "title": "Kept paper",
            "abstract": "Uses three earlier papers.",
            "submission_date": "2024-06-01",
        },
        {
            "id": "short",
            "title": "Short paper",
            "abstract": "Only two earlier papers.",
            "submission_date": "2024-06-01",
        },
    ]
    papers_by_id = {
        "kept": {
            "references": [
                {"externalIds": {"ArXiv": "2301.00001"}},
                {"externalIds": {"ArXiv": "2302.00002"}},
                {"externalIds": {"ArXiv": "2303.00003"}},
                {"externalIds": {"ArXiv": "2407.00007"}},
                {"externalIds": {"DOI": "10.1/only-doi"}},
            ]
        },
        "short": {
            "references": [
                {"externalIds": {"ArXiv": "2301.00001"}},
                {"externalIds": {"ArXiv": "2302.00002"}},
            ]
        },
    }

    built = build_queries(candidates, papers_by_id, corpus_ids, corpus_dates)

    assert [query.record["query_id"] for query in built] == ["kept"]
    record = built[0].record
    assert record["query_type"] == "paper"
    assert record["query_text"] == "Kept paper\n\nUses three earlier papers."
    assert record["query_date"] == "2024-06-01"
    assert record["relevant_paper_ids"] == ["2301.00001", "2302.00002", "2303.00003"]
    assert built[0].n_references == 5
    assert built[0].n_references_on_arxiv == 4
    assert built[0].n_references_in_corpus == 4


def test_seeded_sample_stops_when_enough_queries_qualify():
    papers = [
        {
            "id": f"q{index:02d}",
            "title": f"Title {index}",
            "abstract": "Abstract",
            "submission_date": "2024-06-01",
        }
        for index in range(12)
    ]
    corpus_ids = {"r0", "r1", "r2"}
    corpus_dates = {
        "r0": date(2023, 1, 1),
        "r1": date(2023, 2, 1),
        "r2": date(2023, 3, 1),
    }
    calls: list[list[str]] = []

    def fetch_ids(ids: list[str]) -> dict:
        calls.append(list(ids))
        return {
            paper_id: {
                "references": [
                    {"externalIds": {"ArXiv": "r0"}},
                    {"externalIds": {"ArXiv": "r1"}},
                    {"externalIds": {"ArXiv": "r2"}},
                ]
            }
            for paper_id in ids
        }

    ordered = seeded_candidate_order(papers, seed=0)
    again = seeded_candidate_order(papers, seed=0)
    other = seeded_candidate_order(papers, seed=1)
    assert [paper["id"] for paper in ordered] == [paper["id"] for paper in again]
    assert [paper["id"] for paper in ordered] != [paper["id"] for paper in other]

    chosen, examined = collect_queries(
        ordered,
        fetch_ids,
        corpus_ids,
        corpus_dates,
        num_queries=3,
        batch_size=2,
    )
    chosen_again, _examined_again = collect_queries(
        again,
        fetch_ids,
        corpus_ids,
        corpus_dates,
        num_queries=3,
        batch_size=2,
    )

    assert examined == 4
    assert [query.record["query_id"] for query in chosen] == [paper["id"] for paper in ordered[:3]]
    assert [query.record["query_id"] for query in chosen_again] == [query.record["query_id"] for query in chosen]
    assert calls[:2] == [
        [paper["id"] for paper in ordered[0:2]],
        [paper["id"] for paper in ordered[2:4]],
    ]
    assert len(calls) == 4


def test_benchmark_stats_percentages_use_every_reference():
    queries = [
        BuiltQuery(
            record={"relevant_paper_ids": ["a", "b", "c"]},
            n_references=10,
            n_references_on_arxiv=4,
            n_references_in_corpus=3,
        ),
        BuiltQuery(
            record={"relevant_paper_ids": ["d", "e", "f", "g"]},
            n_references=10,
            n_references_on_arxiv=6,
            n_references_in_corpus=1,
        ),
    ]

    stats = benchmark_stats(queries)

    assert stats["n_queries"] == 2
    assert stats["mean_relevant_per_query"] == 3.5
    assert stats["median_relevant_per_query"] == 3.5
    assert stats["pct_references_on_arxiv"] == 50.0
    assert stats["pct_references_in_corpus"] == 20.0
    assert stats["n_references"] == 20
    empty = benchmark_stats([])
    assert empty["n_queries"] == 0
    assert empty["mean_relevant_per_query"] is None
    assert empty["median_relevant_per_query"] is None
    assert empty["pct_references_on_arxiv"] == 0.0


def test_papers_from_cache_indexes_arxiv_ids(tmp_path):
    body = [{"externalIds": {"ArXiv": "2301.00001v1"}, "references": []}, None]
    (tmp_path / "batch.json").write_text(json.dumps(body), encoding="utf-8")

    assert list(papers_from_cache(tmp_path)) == ["2301.00001"]


def test_cache_reuses_raw_response_without_a_second_request(tmp_path):
    raw = '[\n  {"externalIds": {"ArXiv": "2301.00001"}, "references": []}\n]\n'
    calls = {"n": 0, "headers": None, "params": None, "body": None, "url": None}

    def post(url, params=None, json=None, headers=None, timeout=None):
        calls["n"] += 1
        calls["url"] = url
        calls["params"] = params
        calls["body"] = json
        calls["headers"] = headers
        return _Response(200, raw)

    client, _now = _clocked_client(tmp_path, post, api_key="test-key")
    first = client.fetch_papers(["2301.00001"])
    cached_files = list(tmp_path.glob("*.json"))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cache hit must not call Semantic Scholar")

    rerun, _rerun_now = _clocked_client(tmp_path, fail_if_called, api_key="test-key")
    second = rerun.fetch_papers(["2301.00001"])

    assert first == {"2301.00001": {"externalIds": {"ArXiv": "2301.00001"}, "references": []}}
    assert second == first
    assert calls["n"] == 1
    assert calls["url"] == S2_BATCH_URL
    assert calls["params"] == {"fields": S2_FIELDS}
    assert calls["body"] == {"ids": ["ARXIV:2301.00001"]}
    assert calls["headers"]["x-api-key"] == "test-key"
    assert len(cached_files) == 1
    assert cached_files[0].read_text(encoding="utf-8") == raw
    assert rerun.network_requests == 0
    assert rerun.cache_hits == 1


def test_backoff_uses_retry_after_then_caches(tmp_path):
    raw = '[{"externalIds": {"ArXiv": "2301.00001"}, "references": []}]'
    calls = {"n": 0}

    def post(url, params=None, json=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Response(429, "slow down", headers={"Retry-After": "25"})
        return _Response(200, raw)

    client, now = _clocked_client(tmp_path, post)
    found = client.fetch_papers(["2301.00001"])
    client.fetch_papers(["2301.00001"])

    assert found["2301.00001"]["externalIds"]["ArXiv"] == "2301.00001"
    assert calls["n"] == 2
    assert now["slept"] == [25.0]
    assert client.cache_hits == 1


def test_oversized_batch_splits_and_rerun_skips_the_network(tmp_path):
    def post(url, **kwargs):
        del url
        ids = kwargs["json"]["ids"]
        if len(ids) > 2:
            return _Response(400, '{"error":"Requested too many citations"}')
        body = [
            {"externalIds": {"ArXiv": paper_id.removeprefix("ARXIV:")}, "references": []}
            for paper_id in ids
        ]
        return _Response(200, json.dumps(body))

    client, _now = _clocked_client(tmp_path, post, batch_size=4)
    found = client.fetch_papers(["2301.00001", "2301.00002", "2301.00003", "2301.00004"])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("split batches are cached and must not be requested again")

    rerun, _rerun_now = _clocked_client(tmp_path, fail_if_called, batch_size=4)
    again = rerun.fetch_papers(["2301.00001", "2301.00002", "2301.00003", "2301.00004"])

    assert list(found) == ["2301.00001", "2301.00002", "2301.00003", "2301.00004"]
    assert again == found
    assert rerun.network_requests == 0
    cached = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("*.json")]
    assert cached
    assert all(len(batch) <= 2 for batch in cached)
    assert all("error" not in json.dumps(batch) for batch in cached)


def test_unknown_single_id_is_cached_as_null(tmp_path):
    def post(url, **kwargs):
        del url, kwargs
        return _Response(400, '{"error":"No valid paper ids given"}')

    client, _now = _clocked_client(tmp_path, post)
    assert client.fetch_papers(["not-a-paper"]) == {"not-a-paper": None}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("an unrecognized id must be reused from the cache")

    rerun, _rerun_now = _clocked_client(tmp_path, fail_if_called)
    assert rerun.fetch_papers(["not-a-paper"]) == {"not-a-paper": None}
    assert rerun.network_requests == 0
    assert json.loads((next(tmp_path.glob("*.json"))).read_text(encoding="utf-8")) == [None]


def test_batch_size_cannot_exceed_the_api_cap(tmp_path):
    with pytest.raises(ValueError):
        S2Client(tmp_path, batch_size=501)
