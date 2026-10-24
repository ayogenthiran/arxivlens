import unittest

import numpy as np

from src.embeddings.embedding_model import (
    EMBEDDING_WINDOW_OVERLAP_TOKENS,
    EmbeddingModel,
    l2_normalize,
    pool_window_embeddings,
    split_token_ids,
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


class SplitTokenIdsTests(unittest.TestCase):
    def test_short_input_is_one_window(self):
        self.assertEqual(split_token_ids([1, 2, 3], 5, 1), [[1, 2, 3]])

    def test_empty_input_is_one_empty_window(self):
        self.assertEqual(split_token_ids([], 4, 1), [[]])

    def test_windows_cover_every_token_and_the_tail_is_full(self):
        token_ids = list(range(100))
        windows = split_token_ids(token_ids, 30, 5)

        covered = set()
        for window in windows:
            covered.update(window)

        self.assertEqual(covered, set(token_ids))
        self.assertTrue(all(len(window) == 30 for window in windows))
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][-1], 99)
        self.assertLessEqual(max(len(window) for window in windows), 30)

    def test_overlap_repeats_the_boundary_token(self):
        windows = split_token_ids(list(range(10)), 4, 1)
        self.assertIn(3, windows[0])
        self.assertIn(3, windows[1])
        self.assertEqual(windows[-1], [6, 7, 8, 9])

    def test_window_size_must_be_positive(self):
        with self.assertRaises(ValueError):
            split_token_ids([1, 2], 0, 0)


class PoolWindowEmbeddingsTests(unittest.TestCase):
    def test_equal_weights_are_a_mean(self):
        embeddings = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=np.float32)
        pooled = pool_window_embeddings(embeddings, [1, 1])
        np.testing.assert_allclose(pooled, [2.0, 3.0])

    def test_longer_window_counts_for_more(self):
        embeddings = np.array([[0.0, 0.0], [4.0, 8.0]], dtype=np.float32)
        pooled = pool_window_embeddings(embeddings, [1, 3])
        np.testing.assert_allclose(pooled, [3.0, 6.0])

    def test_single_window_is_unchanged(self):
        embeddings = np.array([[1.5, -2.0]], dtype=np.float32)
        pooled = pool_window_embeddings(embeddings, [8])
        np.testing.assert_allclose(pooled, embeddings[0])


class EmbeddingModelWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EmbeddingModel()
        cls.model.load_model()

    def test_model_limit_leaves_room_for_special_tokens(self):
        limit = self.model._content_token_limit()
        self.assertIsNotNone(limit)
        self.assertLess(limit, int(self.model.model.max_seq_length))
        self.assertGreaterEqual(limit, 1)

    def test_short_text_matches_a_single_encode(self):
        text = "Attention is all you need."
        raw = self.model.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )[0]
        normalized = self.model.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        via_texts = self.model.embed_texts([text], show_progress_bar=False)[0]
        via_query = self.model.embed_query(text)
        manual = self.model._encode_content_windows([self.model._content_token_ids(text)])[0]

        np.testing.assert_allclose(via_texts, normalized, atol=1e-5)
        np.testing.assert_allclose(via_query, normalized, atol=1e-5)
        np.testing.assert_allclose(manual, raw, atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(via_texts)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(via_query)), 1.0, places=5)

    def test_text_past_the_limit_includes_the_tail(self):
        limit = self.model._content_token_limit()
        head_sentence = "This paper studies graph neural networks for citation ranking. "
        tail_sentence = "Zebra quark distillation separates martian spectrograms from noise. "
        head = ""
        while len(self.model._content_token_ids(head + head_sentence)) <= limit:
            head += head_sentence
        text = head + (tail_sentence * 12)

        token_ids = self.model._content_token_ids(text)
        self.assertGreater(len(token_ids), limit)
        windows = split_token_ids(
            token_ids,
            limit,
            min(EMBEDDING_WINDOW_OVERLAP_TOKENS, limit - 1),
        )
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[-1][-1], token_ids[-1])
        self.assertTrue(all(len(window) <= limit for window in windows))

        truncated = self.model.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        pooled = self.model.embed_texts([text], show_progress_bar=False)[0]
        query = self.model.embed_query(tail_sentence.strip())

        self.assertFalse(np.allclose(truncated, pooled, atol=1e-4))
        self.assertGreater(_cosine(query, pooled), _cosine(query, truncated))

    def test_short_row_in_a_mixed_batch_still_matches_encode(self):
        short = "A short title and abstract."
        long = "token " * 400
        self.assertGreater(len(self.model._content_token_ids(long)), self.model._content_token_limit())

        batch = self.model.embed_texts([short, long, short], show_progress_bar=False)
        direct = self.model.model.encode(
            [short],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        self.assertEqual(batch.shape[0], 3)
        np.testing.assert_allclose(batch[0], direct, atol=1e-5)
        np.testing.assert_allclose(batch[2], direct, atol=1e-5)
        self.assertFalse(np.allclose(batch[1], direct, atol=1e-3))
        self.assertAlmostEqual(float(np.linalg.norm(batch[1])), 1.0, places=5)


class _RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        raw = np.tile(np.array([3.0, 4.0], dtype=np.float32), (len(texts), 1))
        if kwargs.get("normalize_embeddings"):
            raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        return raw


class _LongTextModel(EmbeddingModel):
    """Forces the window path without loading sentence-transformers."""

    def __init__(self):
        super().__init__()
        self.model = object()

    def _content_token_limit(self):
        return 4

    def _content_token_ids(self, text):
        return [1, 2, 3, 4, 5]

    def _encode_content_windows(self, windows):
        return np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)


class NormalizeEmbeddingsTests(unittest.TestCase):
    def test_embed_texts_and_query_pass_normalize_embeddings(self):
        model = EmbeddingModel()
        encoder = _RecordingEncoder()
        model.model = encoder

        texts = model.embed_texts(["one", "two"], show_progress_bar=False)
        query = model.embed_query("one")

        self.assertEqual(len(encoder.calls), 2)
        self.assertTrue(all(call["normalize_embeddings"] is True for call in encoder.calls))
        self.assertFalse(encoder.calls[0]["show_progress_bar"])
        self.assertFalse(encoder.calls[1]["show_progress_bar"])
        np.testing.assert_allclose(np.linalg.norm(texts, axis=1), [1.0, 1.0], atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0, places=5)
        np.testing.assert_allclose(texts[0], [0.6, 0.8], atol=1e-5)

    def test_pooled_long_text_is_l2_normalized(self):
        model = _LongTextModel()

        pooled = model.embed_texts(["a long abstract"], show_progress_bar=False)[0]
        query = model.embed_query("a long abstract")

        np.testing.assert_allclose(pooled, [0.6, 0.8], atol=1e-5)
        np.testing.assert_allclose(query, pooled, atol=1e-5)
        self.assertAlmostEqual(float(np.linalg.norm(pooled)), 1.0, places=5)

    def test_l2_normalize_leaves_a_zero_vector_unchanged(self):
        zero = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(l2_normalize(zero), zero)


if __name__ == "__main__":
    unittest.main()
