# src/vector_store/vector_store.py

import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.settings import (
    ANONYMIZED_TELEMETRY,
    CHROMA_COLLECTION_NAME,
    DEFAULT_TOP_K,
    FACET_TOP_AUTHORS,
    HYBRID_BM25_MAX_DOCS,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RETRIEVAL_FILTER_CANDIDATE_FLOOR,
    RETRIEVAL_FILTER_CANDIDATE_MULTIPLIER,
    RRF_K,
)

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.
    """

    def __init__(self, persist_directory: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.client = None
        self.collection = None
        self._bm25 = None
        self._bm25_doc_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metadata_by_id: Dict[str, Dict[str, Any]] = {}

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
                        settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY),
                    )
                    logger.info(f"Initialized PersistentClient at {self.persist_directory}")
                else:
                    self.client = chromadb.Client(settings=Settings(anonymized_telemetry=ANONYMIZED_TELEMETRY))
                    logger.info("Initialized In-Memory ChromaDB Client")
                
                self.collection = self._get_or_create_collection()
            except Exception as e:
                logger.error(f"Error initializing VectorStore: {e}")
                raise

    def _get_or_create_collection(self):
        """
        Open the collection, or create it with cosine distance.

        An existing collection is left in place. If its distance is not cosine,
        a warning tells the user to re-ingest.
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            self._warn_if_not_cosine(collection)
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Created new collection: {self.collection_name}")
        return collection

    def _warn_if_not_cosine(self, collection) -> None:
        space = self._hnsw_space(collection)
        if space == "cosine":
            return
        shown = space if space else "unset (Chroma default l2)"
        logger.warning(
            "Collection '%s' uses hnsw:space=%s, not cosine. "
            "Re-ingest into a new collection so vector_score is cosine similarity (1 - distance). "
            "Existing data was not deleted or recreated.",
            self.collection_name,
            shown,
        )

    def _hnsw_space(self, collection=None) -> Optional[str]:
        target = collection if collection is not None else self.collection
        metadata = getattr(target, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        space = metadata.get("hnsw:space")
        if space is None:
            return None
        text = str(space).strip().lower()
        return text or None

    def _vector_score(self, distance: Optional[float]) -> float:
        """Cosine similarity for a cosine collection; the legacy L2 map otherwise."""
        if distance is None:
            return 0.0
        distance_value = float(distance)
        if self._hnsw_space() == "cosine":
            return 1.0 - distance_value
        return 1.0 / (1.0 + distance_value)

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
        metadatas = [self._metadata_for_chroma(chunk["metadata"]) for chunk in chunks]

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
            self._bm25_metadata_by_id = {}
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
            max_bm25_docs = HYBRID_BM25_MAX_DOCS
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
                self._bm25_metadata_by_id = {}
                return

            all_docs = self.collection.get(include=["documents", "metadatas"])
            ids = all_docs.get("ids", [])
            documents = all_docs.get("documents", [])
            metadatas = all_docs.get("metadatas", [])
            if not ids or not documents:
                self._bm25 = None
                self._bm25_doc_ids = []
                self._bm25_docs = []
                self._bm25_metadata_by_id = {}
                return

            tokenized_corpus = [self._tokenize(doc or "") for doc in documents]
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_doc_ids = ids
            self._bm25_docs = documents
            self._bm25_metadata_by_id = {
                doc_id: self._normalize_chunk_metadata(metadata or {})
                for doc_id, metadata in zip(ids, metadatas)
            }
        except Exception as e:
            logger.warning(f"BM25 index initialization failed; falling back to vector-only search: {e}")
            self._bm25 = None
            self._bm25_doc_ids = []
            self._bm25_docs = []
            self._bm25_metadata_by_id = {}

    def _vector_search(
        self,
        query_embedding: np.ndarray | List[float],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(
            **query_kwargs
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            distance = results.get("distances", [[]])[0][i] if "distances" in results else None
            vector_score = self._vector_score(distance)
            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": self._normalize_chunk_metadata(results["metadatas"][0][i]),
                "distance": distance,
                "vector_score": vector_score,
                "bm25_score": 0.0,
            })
        return formatted_results

    def query(
        self,
        query_embedding: np.ndarray,
        query_text: Optional[str] = None,
        n_results: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
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

        # Ask for more neighbors than n_results so one chunk per paper can
        # still fill the list. Category and author filters run after Chroma
        # returns, so those queries need a wider pool. Year bounds stay in
        # the Chroma where clause.
        candidate_limit = self._candidate_limit(n_results, filters)

        try:
            where_clause = self._build_where_clause(filters)
            vector_results = self._vector_search(
                query_embedding=query_embedding,
                n_results=candidate_limit,
                where=where_clause,
            )
            vector_results = [
                item for item in vector_results if self._metadata_matches_filters(item.get("metadata", {}), filters)
            ]
            vector_results = sorted(
                vector_results,
                key=lambda item: float(item.get("vector_score") or 0.0),
                reverse=True,
            )
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise

        # If no query text is available, return vector-only retrieval.
        if not query_text:
            formatted_results = self._take_best_chunk_per_paper(vector_results, n_results, "vector_score")
            logger.info(f"Vector-only query returned {len(formatted_results)} results")
            return formatted_results

        self._ensure_bm25_index()
        if self._bm25 is None or not self._bm25_doc_ids:
            formatted_results = self._take_best_chunk_per_paper(vector_results, n_results, "vector_score")
            logger.info(f"Vector-only fallback returned {len(formatted_results)} results")
            return formatted_results

        bm25_scores = self._bm25.get_scores(self._tokenize(query_text))
        if len(bm25_scores) == 0:
            formatted_results = self._take_best_chunk_per_paper(vector_results, n_results, "vector_score")
            logger.info(f"Vector-only fallback returned {len(formatted_results)} results")
            return formatted_results

        bm25_score_min = float(np.min(bm25_scores))
        bm25_score_max = float(np.max(bm25_scores))
        bm25_denom = (bm25_score_max - bm25_score_min) if bm25_score_max > bm25_score_min else 1.0
        rrf_k = RRF_K

        vector_rank = {item["id"]: rank for rank, item in enumerate(vector_results, start=1)}
        hybrid_candidates: Dict[str, Dict[str, Any]] = {item["id"]: item for item in vector_results}
        bm25_rank: Dict[str, int] = {}
        lexical_rank = 0

        for idx in np.argsort(bm25_scores)[::-1]:
            chunk_id = self._bm25_doc_ids[int(idx)]
            score_raw = float(bm25_scores[int(idx)])
            score_norm = (score_raw - bm25_score_min) / bm25_denom
            bm25_metadata = self._bm25_metadata_by_id.get(chunk_id, {})
            if not self._metadata_matches_filters(bm25_metadata, filters):
                continue

            lexical_rank += 1
            if lexical_rank > candidate_limit:
                break
            bm25_rank[chunk_id] = lexical_rank

            if chunk_id in hybrid_candidates:
                hybrid_candidates[chunk_id]["bm25_score"] = score_norm
                continue

            hybrid_candidates[chunk_id] = {
                "id": chunk_id,
                "text": self._bm25_docs[int(idx)],
                "metadata": bm25_metadata,
                "distance": None,
                "vector_score": 0.0,
                "bm25_score": score_norm,
            }

        for item in hybrid_candidates.values():
            fused = 0.0
            v_rank = vector_rank.get(item["id"])
            b_rank = bm25_rank.get(item["id"])
            if v_rank is not None:
                fused += 1.0 / (rrf_k + v_rank)
            if b_rank is not None:
                fused += 1.0 / (rrf_k + b_rank)
            item["hybrid_score"] = fused

        ranked = sorted(
            hybrid_candidates.values(),
            key=lambda item: float(item.get("hybrid_score") or 0.0),
            reverse=True,
        )
        formatted_results = self._take_best_chunk_per_paper(ranked, n_results, "hybrid_score")

        logger.info(f"Hybrid query returned {len(formatted_results)} results")
        return formatted_results

    def _take_best_chunk_per_paper(
        self,
        ranked: List[Dict[str, Any]],
        n_results: int,
        score_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Keep the highest-scoring chunk for each paper, then cut to n_results.

        Chunk ids are left unchanged so citations can still point at one chunk.
        """
        best_by_paper: Dict[str, Dict[str, Any]] = {}
        paper_order: List[str] = []

        for item in ranked:
            metadata = item.get("metadata") or {}
            paper_id = metadata.get("doc_id")
            paper_key = str(paper_id) if paper_id else f"chunk:{item.get('id')}"
            current = best_by_paper.get(paper_key)
            if current is None:
                best_by_paper[paper_key] = item
                paper_order.append(paper_key)
                continue

            current_score = float(current.get(score_key) or 0.0)
            candidate_score = float(item.get(score_key) or 0.0)
            if candidate_score > current_score:
                best_by_paper[paper_key] = item

        return [best_by_paper[paper_key] for paper_key in paper_order[:n_results]]

    def _candidate_limit(self, n_results: int, filters: Optional[Dict[str, Any]]) -> int:
        """How many neighbors to pull before paper dedupe and the final cut."""
        categories = (filters or {}).get("categories") or []
        authors = (filters or {}).get("authors") or []
        if categories or authors:
            return max(n_results * RETRIEVAL_FILTER_CANDIDATE_MULTIPLIER, RETRIEVAL_FILTER_CANDIDATE_FLOOR)
        return n_results * RETRIEVAL_CANDIDATE_MULTIPLIER

    def get_facets(self, chunks: List[Dict[str, Any]], top_n_authors: Optional[int] = None) -> Dict[str, Any]:
        """Build facet counts from retrieved chunks."""
        if top_n_authors is None:
            top_n_authors = FACET_TOP_AUTHORS
        years: Counter[int] = Counter()
        categories: Counter[str] = Counter()
        authors: Counter[str] = Counter()

        for chunk in chunks:
            metadata = self._normalize_chunk_metadata(chunk.get("metadata", {}))
            year = metadata.get("year")
            if isinstance(year, int):
                years[year] += 1

            for category in metadata.get("categories", []):
                categories[category] += 1

            for author in metadata.get("authors", []):
                authors[author] += 1

        return {
            "years": [{"value": year, "count": count} for year, count in sorted(years.items(), reverse=True)],
            "categories": [{"value": cat, "count": count} for cat, count in categories.most_common()],
            "authors": [{"value": author, "count": count} for author, count in authors.most_common(top_n_authors)],
        }

    def _build_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        clauses: List[Dict[str, Any]] = []

        year_min = filters.get("year_min")
        if isinstance(year_min, int):
            clauses.append({"year": {"$gte": year_min}})

        year_max = filters.get("year_max")
        if isinstance(year_max, int):
            clauses.append({"year": {"$lte": year_max}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _metadata_matches_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True

        normalized_metadata = self._normalize_chunk_metadata(metadata)
        categories = filters.get("categories") or []
        if categories:
            chunk_categories = set(normalized_metadata.get("categories", []))
            if not chunk_categories.intersection(categories):
                return False

        year_min = filters.get("year_min")
        year_max = filters.get("year_max")
        year = normalized_metadata.get("year")
        if isinstance(year_min, int):
            if not isinstance(year, int) or year < year_min:
                return False
        if isinstance(year_max, int):
            if not isinstance(year, int) or year > year_max:
                return False

        authors = filters.get("authors") or []
        if authors:
            filter_terms = [term.lower().strip() for term in authors if term and term.strip()]
            chunk_authors = " ".join(normalized_metadata.get("authors", [])).lower()
            if not any(term in chunk_authors for term in filter_terms):
                return False

        return True

    def _normalize_chunk_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(metadata or {})
        authors = normalized.get("authors", [])
        categories = normalized.get("categories", [])
        year = normalized.get("year")

        if isinstance(authors, str):
            # Stored records use either "and" (arXiv) or commas (older Chroma writes).
            separator = " and " if " and " in authors else ","
            authors = [name.strip() for name in authors.split(separator) if name.strip()]
        elif isinstance(authors, list):
            authors = [str(name).strip() for name in authors if str(name).strip()]
        else:
            authors = []

        if isinstance(categories, str):
            categories = [category.strip() for category in categories.split() if category.strip()]
        elif isinstance(categories, list):
            categories = [str(category).strip() for category in categories if str(category).strip()]
        else:
            categories = []

        if isinstance(year, str) and year.isdigit():
            year = int(year)
        if not isinstance(year, int):
            year = None

        normalized["authors"] = authors
        normalized["categories"] = categories
        normalized["year"] = year
        return normalized

    def _metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten metadata to the scalar types Chroma accepts."""
        normalized = self._normalize_chunk_metadata(metadata)
        chroma_metadata: Dict[str, Any] = {}
        for key, value in normalized.items():
            if key == "authors":
                joined = " and ".join(value)
                if joined:
                    chroma_metadata[key] = joined
                continue
            if key == "categories":
                joined = " ".join(value)
                if joined:
                    chroma_metadata[key] = joined
                continue
            if isinstance(value, (str, int, float, bool)):
                chroma_metadata[key] = value
        return chroma_metadata
