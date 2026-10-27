import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.settings import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_ENCODE_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_WINDOW_OVERLAP_TOKENS,
)

logger = logging.getLogger(__name__)


def split_token_ids(token_ids: Sequence[int], window_size: int, overlap: int) -> List[List[int]]:
    """Split token ids into windows no longer than ``window_size``.

    The last window ends on the final token and is padded back to a full
    window when the text is longer than one window. A short tail would
    otherwise contribute almost nothing once the windows are averaged.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    tokens = [int(token_id) for token_id in token_ids]
    if len(tokens) <= window_size:
        return [tokens]

    overlap = min(max(int(overlap), 0), window_size - 1)
    stride = window_size - overlap
    windows: List[List[int]] = []
    start = 0
    while start < len(tokens):
        end = start + window_size
        if end >= len(tokens):
            start = len(tokens) - window_size
            windows.append(tokens[start:])
            break
        windows.append(tokens[start:end])
        start += stride
    return windows


def pool_window_embeddings(embeddings: np.ndarray, weights: Sequence[int]) -> np.ndarray:
    """Length-weighted mean of window vectors, one vector per paper."""
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("embeddings must be a 2-D array of window vectors")
    if len(weights) != len(vectors):
        raise ValueError("weights must match the number of window vectors")

    weight = np.asarray(list(weights), dtype=np.float64)
    total = float(weight.sum())
    if total <= 0:
        pooled = vectors.mean(axis=0)
    else:
        pooled = (vectors * (weight / total)[:, None]).sum(axis=0)
    return pooled.astype(np.float32)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit L2 norm. A zero vector is unchanged."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return array
    return np.asarray(array / np.float32(norm), dtype=np.float32)


class EmbeddingModel:
    """
    Handles text embedding using Sentence Transformers.

    ``all-MiniLM-L6-v2`` keeps only the first 256 tokens. A title plus an
    abstract often runs past that, so a single ``encode`` call drops the end
    of the abstract. Texts that fit are embedded exactly as before. Longer
    texts are split into token windows, embedded separately, and averaged
    into the one vector stored for that paper.
    """

    def __init__(self, model_name: Optional[str] = None):
        configured_name = model_name or EMBEDDING_MODEL_NAME
        self.model_name = self._normalize_model_name(configured_name)
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)

    def embed_texts(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """One L2-normalized vector per text. Texts past the model limit are windowed, then pooled."""
        self._ensure_model_loaded()
        return self._embed_texts(
            texts,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        logger.info("Generated embeddings for %s chunks.", len(chunks))
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """L2-normalized query vector, using the same windowing as document text."""
        self._ensure_model_loaded()
        embedding = self._embed_texts(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embedding[0])

    def _embed_texts(
        self,
        texts: List[str],
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        if not texts:
            return self._encode(texts, show_progress_bar=False, normalize_embeddings=normalize_embeddings)

        limit = self._content_token_limit()
        if limit is None:
            return self._encode(texts, show_progress_bar=show_progress_bar, normalize_embeddings=normalize_embeddings)

        token_rows = [self._content_token_ids(text) for text in texts]
        long_indexes = [index for index, token_ids in enumerate(token_rows) if len(token_ids) > limit]
        if not long_indexes:
            return self._encode(texts, show_progress_bar=show_progress_bar, normalize_embeddings=normalize_embeddings)

        overlap = min(EMBEDDING_WINDOW_OVERLAP_TOKENS, limit - 1)
        flat_windows: List[List[int]] = []
        spans: List[Optional[tuple]] = [None] * len(texts)
        weights: List[Optional[List[int]]] = [None] * len(texts)
        for index in long_indexes:
            pieces = split_token_ids(token_rows[index], limit, overlap)
            start = len(flat_windows)
            flat_windows.extend(pieces)
            spans[index] = (start, len(flat_windows))
            weights[index] = [len(piece) for piece in pieces]

        pooled = self._encode_content_windows(flat_windows)
        short_indexes = [index for index in range(len(texts)) if spans[index] is None]
        short_vectors = None
        if short_indexes:
            short_vectors = self._encode(
                [texts[index] for index in short_indexes],
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )

        rows: List[np.ndarray] = []
        short_cursor = 0
        for index, span in enumerate(spans):
            if span is None:
                rows.append(np.asarray(short_vectors[short_cursor], dtype=np.float32))
                short_cursor += 1
                continue
            start, end = span
            pooled_vector = pool_window_embeddings(pooled[start:end], weights[index])
            if normalize_embeddings:
                pooled_vector = l2_normalize(pooled_vector)
            rows.append(pooled_vector)

        over_long = len(long_indexes)
        noun = "text" if over_long == 1 else "texts"
        logger.info(
            "Pooled %s %s past the %s-token content limit across %s windows.",
            over_long,
            noun,
            limit,
            len(flat_windows),
        )
        return np.vstack(rows)

    def _encode(self, texts: List[str], show_progress_bar: bool, normalize_embeddings: bool) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            )
        )

    def _encode_content_windows(self, windows: List[List[int]]) -> np.ndarray:
        batches = [
            self._forward_content_windows(windows[start : start + EMBEDDING_ENCODE_BATCH_SIZE])
            for start in range(0, len(windows), EMBEDDING_ENCODE_BATCH_SIZE)
        ]
        if len(batches) == 1:
            return batches[0]
        return np.vstack(batches)

    def _forward_content_windows(self, windows: List[List[int]]) -> np.ndarray:
        """Embed token windows that already fit, including special tokens."""
        tokenizer = self.model.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        sequences = [self._with_special_tokens(window) for window in windows]
        width = max(len(sequence) for sequence in sequences)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        for sequence in sequences:
            padding = width - len(sequence)
            input_ids.append(sequence + [pad_id] * padding)
            attention_mask.append([1] * len(sequence) + [0] * padding)

        features = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
            features["token_type_ids"] = torch.zeros_like(features["input_ids"])

        device = self.model.device
        features = {name: value.to(device) for name, value in features.items()}
        self.model.eval()
        with torch.no_grad():
            output = self.model.forward(features)
        return np.asarray(output["sentence_embedding"].detach().cpu().numpy(), dtype=np.float32)

    def _with_special_tokens(self, content_ids: Sequence[int]) -> List[int]:
        tokenizer = self.model.tokenizer
        max_length = int(self.model.max_seq_length)
        content = [int(token_id) for token_id in content_ids]
        sequence = list(tokenizer.build_inputs_with_special_tokens(content))
        if len(sequence) <= max_length:
            return sequence

        overflow = len(sequence) - max_length
        trimmed = content[:-overflow] if overflow < len(content) else []
        return list(tokenizer.build_inputs_with_special_tokens(trimmed))

    def _content_token_limit(self) -> Optional[int]:
        """How many content tokens fit once the tokenizer adds special tokens."""
        max_length = getattr(self.model, "max_seq_length", None)
        tokenizer = getattr(self.model, "tokenizer", None)
        if not isinstance(max_length, int) or max_length <= 0 or tokenizer is None:
            return None
        if not hasattr(tokenizer, "num_special_tokens_to_add"):
            return None

        limit = max_length - int(tokenizer.num_special_tokens_to_add(pair=False))
        if limit < 1:
            return None
        return limit

    def _content_token_ids(self, text: str) -> List[int]:
        """Token ids for the text the model would actually embed, without truncation."""
        prepared = str(text).strip()
        transformer = self.model[0]
        if getattr(transformer, "do_lower_case", False):
            prepared = prepared.lower()
        # truncation stays off so the length check sees the whole abstract.
        # verbose stays off: this count is not a model forward pass.
        token_ids = self.model.tokenizer.encode(
            prepared,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        return [int(token_id) for token_id in token_ids]

    def _ensure_model_loaded(self):
        if self.model is None:
            self.load_model()

    def _normalize_model_name(self, model_name: str) -> str:
        normalized = model_name.strip()

        # OpenAI embedding model names are not valid SentenceTransformer model IDs.
        if normalized.startswith("text-embedding-"):
            fallback = DEFAULT_EMBEDDING_MODEL
            logger.warning(
                "EMBEDDING_MODEL_NAME='%s' is an OpenAI embedding model. "
                "Falling back to sentence-transformers model '%s'.",
                normalized,
                fallback,
            )
            return fallback

        return normalized
