"""reranker_score is sigmoid(logit), not a per-query min-max rescale."""

import math

import pytest

from src.reranker.cross_encoder_reranker import CrossEncoderReranker


class _FakeCrossEncoder:
    """predict() returns fixed logits in the same order as the pairs."""

    def __init__(self, logits: list[float]):
        self._logits = list(logits)

    def predict(self, pairs):
        if len(pairs) != len(self._logits):
            raise AssertionError(f"expected {len(self._logits)} pairs, got {len(pairs)}")
        return list(self._logits)


def _sigmoid(raw: float) -> float:
    return 1.0 / (1.0 + math.exp(-raw))


def _reranker(logits: list[float], min_raw_score: float = 1.0) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker()
    reranker.enabled = True
    reranker.min_raw_score = min_raw_score
    reranker._model = _FakeCrossEncoder(logits)
    return reranker


def _chunks() -> list[dict]:
    return [
        {"id": "low", "text": "unrelated abstract"},
        {"id": "high", "text": "relevant abstract"},
        {"id": "mid", "text": "partial abstract"},
    ]


def test_reranker_score_is_sigmoid_of_raw_and_top_is_not_one():
    logits = [-2.0, 3.0, 1.0]
    results = _reranker(logits, min_raw_score=-10.0).rerank("query", _chunks(), final_top_k=3)

    by_id = {item["id"]: item for item in results}
    for chunk, raw in zip(_chunks(), logits, strict=True):
        scored = by_id[chunk["id"]]
        assert scored["reranker_score_raw"] == raw
        assert scored["reranker_score"] == pytest.approx(_sigmoid(raw))

    top = results[0]
    assert top["id"] == "high"
    assert top["reranker_score_raw"] == 3.0
    assert top["reranker_score"] == pytest.approx(_sigmoid(3.0))
    assert top["reranker_score"] < 1.0
    assert by_id["low"]["reranker_score"] == pytest.approx(_sigmoid(-2.0))
    assert by_id["low"]["reranker_score"] > 0.0


def test_orders_by_raw_score_and_drops_weak_hits():
    # Candidate order is not score order. 0.4 is below the raw cutoff of 1.
    logits = [0.4, 2.5, 1.0]
    results = _reranker(logits, min_raw_score=1.0).rerank("query", _chunks(), final_top_k=3)

    assert [item["id"] for item in results] == ["high", "mid"]
    assert [item["reranker_score_raw"] for item in results] == [2.5, 1.0]
    assert results[0]["reranker_score"] == pytest.approx(_sigmoid(2.5))
    assert results[0]["reranker_score"] < 1.0
    assert results[1]["reranker_score"] == pytest.approx(_sigmoid(1.0))
