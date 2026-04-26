import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker that reorders retrieved chunks after first-pass retrieval.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.enabled = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        self.max_candidates = int(os.getenv("RERANKER_MAX_CANDIDATES", "24"))
        self._model: Optional[CrossEncoder] = None

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
        if not self._load_model():
            return candidates[:final_top_k]

        capped_candidates = candidates[: self.max_candidates]
        pairs = [(query_text, item.get("text", "")) for item in capped_candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker scoring failed, using original ranking: %s", exc)
            return candidates[:final_top_k]

        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size == 0:
            return candidates[:final_top_k]

        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        score_denom = (score_max - score_min) if score_max > score_min else 1.0

        reranked: List[Dict[str, Any]] = []
        for idx, item in enumerate(capped_candidates):
            item_copy = dict(item)
            raw = float(scores[idx])
            norm = (raw - score_min) / score_denom
            item_copy["reranker_score_raw"] = raw
            item_copy["reranker_score"] = norm
            item_copy["hybrid_score"] = norm
            reranked.append(item_copy)

        reranked.sort(key=lambda x: x.get("reranker_score_raw", 0.0), reverse=True)

        if final_top_k <= len(reranked):
            return reranked[:final_top_k]

        # If final_top_k is larger than reranked list, fill from remaining original candidates.
        seen_ids = {item.get("id") for item in reranked if item.get("id")}
        tail: List[Dict[str, Any]] = []
        for item in candidates[len(capped_candidates) :]:
            if item.get("id") in seen_ids:
                continue
            tail.append(item)
            if len(reranked) + len(tail) >= final_top_k:
                break
        return reranked + tail
