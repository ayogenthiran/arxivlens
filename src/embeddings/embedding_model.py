# src/embeddings/embedding_model.py

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Handles text embedding using Sentence Transformers.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        configured_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.model_name = self._normalize_model_name(configured_name)
        self.model = None

    def load_model(self):
        """
        Load the embedding model if not already loaded.
        """
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings
        """
        self._ensure_model_loaded()

        try:
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document chunks.

        Args:
            chunks: List of chunk dictionaries with text and metadata

        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        logger.info(f"Generated embeddings for {len(chunks)} chunks.")
        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.

        Args:
            query: Query text

        Returns:
            Query embedding (1D numpy array)
        """
        self._ensure_model_loaded()

        try:
            embedding = self.model.encode(query, convert_to_numpy=True)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise

    def _ensure_model_loaded(self):
        """Helper to make sure the model is loaded before using."""
        if self.model is None:
            self.load_model()

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize user-provided model names to valid sentence-transformers IDs.
        """
        normalized = model_name.strip()

        # OpenAI embedding model names are not valid SentenceTransformer model IDs.
        if normalized.startswith("text-embedding-"):
            fallback = "all-MiniLM-L6-v2"
            logger.warning(
                "EMBEDDING_MODEL_NAME='%s' is an OpenAI embedding model. "
                "Falling back to sentence-transformers model '%s'.",
                normalized,
                fallback,
            )
            return fallback

        return normalized
