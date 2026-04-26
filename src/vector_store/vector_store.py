# src/vector_store/vector_store.py

import os
import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.
    """

    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "documents"):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._bm25 = None
        self._bm25_doc_ids: List[str] = []
        self._bm25_docs: List[str] = []

    def initialize(self):
        """
        Initialize the ChromaDB client and collection.
        """
        if self.client is None:
            try:
                if self.persist_directory:
                    os.makedirs(self.persist_directory, exist_ok=True)
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    logger.info(f"Initialized PersistentClient at {self.persist_directory}")
                else:
                    self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
                    logger.info(f"Initialized In-Memory ChromaDB Client")
                
                self.collection = self._get_or_create_collection()
            except Exception as e:
                logger.error(f"Error initializing VectorStore: {e}")
                raise

    def _get_or_create_collection(self):
        """
        Get or create the collection.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")
        return collection

    def add_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks with embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with text, metadata, and embeddings
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return

        if self.collection is None:
            self.initialize()

        ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
        embeddings = [chunk["embedding"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            # Keep BM25 index in sync with newly added chunks.
            self._bm25 = None
            self._bm25_doc_ids = []
            self._bm25_docs = []
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding embeddings to vector store: {e}")
            raise

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _ensure_bm25_index(self) -> None:
        if self.collection is None:
            self.initialize()
        if self._bm25 is not None:
            return

        try:
            max_bm25_docs = int(os.getenv("HYBRID_BM25_MAX_DOCS", "50000"))
            total_docs = int(self.collection.count())
            if total_docs > max_bm25_docs:
                logger.info(
                    "Skipping BM25 index build: collection size=%s exceeds HYBRID_BM25_MAX_DOCS=%s. "
                    "Using vector-only retrieval.",
                    total_docs,
                    max_bm25_docs,
                )
                self._bm25 = None
                self._bm25_doc_ids = []
                self._bm25_docs = []
                return

            all_docs = self.collection.get(include=["documents"])
            ids = all_docs.get("ids", [])
            documents = all_docs.get("documents", [])
            if not ids or not documents:
                self._bm25 = None
                self._bm25_doc_ids = []
                self._bm25_docs = []
                return

            tokenized_corpus = [self._tokenize(doc or "") for doc in documents]
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_doc_ids = ids
            self._bm25_docs = documents
        except Exception as e:
            logger.warning(f"BM25 index initialization failed; falling back to vector-only search: {e}")
            self._bm25 = None
            self._bm25_doc_ids = []
            self._bm25_docs = []

    def _vector_search(self, query_embedding: np.ndarray | List[float], n_results: int) -> List[Dict[str, Any]]:
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            distance = results.get("distances", [[]])[0][i] if "distances" in results else None
            vector_score = 1.0 / (1.0 + distance) if distance is not None else 0.0
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "vector_score": vector_score,
                "bm25_score": 0.0,
            })
        return formatted_results

    def query(self, query_embedding: np.ndarray, query_text: Optional[str] = None, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents using hybrid retrieval.

        Args:
            query_embedding: Query embedding vector
            query_text: Original user query for BM25 lexical retrieval
            n_results: Number of results to return

        Returns:
            List of similar document chunks
        """
        if self.collection is None:
            self.initialize()

        try:
            vector_results = self._vector_search(query_embedding=query_embedding, n_results=n_results * 3)
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise

        # If no query text is available, return vector-only retrieval.
        if not query_text:
            formatted_results = vector_results[:n_results]
            logger.info(f"Vector-only query returned {len(formatted_results)} results")
            return formatted_results

        self._ensure_bm25_index()
        if self._bm25 is None or not self._bm25_doc_ids:
            formatted_results = vector_results[:n_results]
            logger.info(f"Vector-only fallback returned {len(formatted_results)} results")
            return formatted_results

        bm25_scores = self._bm25.get_scores(self._tokenize(query_text))
        if len(bm25_scores) == 0:
            formatted_results = vector_results[:n_results]
            logger.info(f"Vector-only fallback returned {len(formatted_results)} results")
            return formatted_results

        bm25_score_min = float(np.min(bm25_scores))
        bm25_score_max = float(np.max(bm25_scores))
        bm25_denom = (bm25_score_max - bm25_score_min) if bm25_score_max > bm25_score_min else 1.0

        top_bm25_idx = np.argsort(bm25_scores)[::-1][: n_results * 3]
        hybrid_candidates: Dict[str, Dict[str, Any]] = {item["id"]: item for item in vector_results}

        for idx in top_bm25_idx:
            chunk_id = self._bm25_doc_ids[int(idx)]
            score_raw = float(bm25_scores[int(idx)])
            score_norm = (score_raw - bm25_score_min) / bm25_denom
            if chunk_id in hybrid_candidates:
                hybrid_candidates[chunk_id]["bm25_score"] = score_norm
                continue

            doc_text = self._bm25_docs[int(idx)]
            hybrid_candidates[chunk_id] = {
                "id": chunk_id,
                "text": doc_text,
                "metadata": {},
                "distance": None,
                "vector_score": 0.0,
                "bm25_score": score_norm,
            }

        alpha = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7"))
        alpha = min(max(alpha, 0.0), 1.0)
        for item in hybrid_candidates.values():
            item["hybrid_score"] = (alpha * item.get("vector_score", 0.0)) + ((1.0 - alpha) * item.get("bm25_score", 0.0))

        ranked = sorted(hybrid_candidates.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        formatted_results = ranked[:n_results]

        logger.info(f"Hybrid query returned {len(formatted_results)} results")
        return formatted_results
