import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from src.settings import (
    ENABLE_RERANKER,
    RERANKER_MAX_CANDIDATES,
    RERANKER_MIN_RAW_SCORE,
    RERANKER_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def _sigmoid(raw_score: float) -> float:
    """Map one cross-encoder logit to a relevance probability in (0, 1)."""
    if raw_score >= 0.0:
        return float(1.0 / (1.0 + np.exp(-raw_score)))
    exp_raw = float(np.exp(raw_score))
    return exp_raw / (1.0 + exp_raw)


class CrossEncoderReranker:
    """
    Cross-encoder reranker that reorders retrieved chunks after first-pass retrieval.

    ``reranker_score`` is sigmoid(raw logit), so the same number means the same
    relevance on every query. It is not rescaled inside one result list.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or RERANKER_MODEL_NAME
        self.enabled = ENABLE_RERANKER
        self.max_candidates = RERANKER_MAX_CANDIDATES
        self.min_raw_score = RERANKER_MIN_RAW_SCORE
        self._model: Optional[CrossEncoder] = None

    def _drop_weak_hits(self, reranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep chunks whose raw cross-encoder score is at or above the cutoff."""
        kept: List[Dict[str, Any]] = []
        for item in reranked:
            raw_score = item.get("reranker_score_raw")
            if raw_score is None:
                continue
            if float(raw_score) >= self.min_raw_score:
                kept.append(item)
        return kept

    def _load_model(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        try:
            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder reranker model: %s", self.model_name)
            return True
        except Exception as exc:
            logger.warning("Failed to load reranker model '%s': %s", self.model_name, exc)
            return False

    def rerank(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        final_top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks for a query and return the final top_k results.
        """
        if not candidates:
            return []
        logger.info(
            "Reranker received %s candidates (final_top_k=%s)",
            len(candidates),
            final_top_k,
        )
        if not self._load_model():
            return candidates[:final_top_k]
        model = self._model
        if model is None:
            return candidates[:final_top_k]

        capped_candidates = candidates[: self.max_candidates]
        pairs = [(query_text, item.get("text", "")) for item in capped_candidates]

        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker scoring failed, using original ranking: %s", exc)
            return candidates[:final_top_k]

        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size == 0:
            return candidates[:final_top_k]

        reranked: List[Dict[str, Any]] = []
        for idx, item in enumerate(capped_candidates):
            item_copy = dict(item)
            raw = float(scores[idx])
            item_copy["reranker_score_raw"] = raw
            item_copy["reranker_score"] = _sigmoid(raw)
            reranked.append(item_copy)

        reranked.sort(key=lambda x: x.get("reranker_score_raw", 0.0), reverse=True)
        kept = self._drop_weak_hits(reranked)
        logger.info(
            "Reranker kept %s of %s candidates with raw score >= %.3f",
            len(kept),
            len(reranked),
            self.min_raw_score,
        )
        return kept[:final_top_k]
