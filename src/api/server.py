import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from src.llm.llm_interface import EMPTY_ANSWER, INSUFFICIENT_CONTEXT_ANSWER
from src.settings import (
    APP_VERSION,
    CORS_ALLOW_ORIGINS,
    DEFAULT_TOP_K,
    HOST,
    MAX_QUERY_TOP_K,
    PORT,
    YEAR_MAX_BOUND,
    YEAR_MIN_BOUND,
)

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_QUERY_TOP_K)
    authors: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    year_min: Optional[int] = Field(default=None, ge=YEAR_MIN_BOUND, le=YEAR_MAX_BOUND)
    year_max: Optional[int] = Field(default=None, ge=YEAR_MIN_BOUND, le=YEAR_MAX_BOUND)

    @model_validator(mode="after")
    def validate_year_range(self):
        if self.year_min is not None and self.year_max is not None and self.year_min > self.year_max:
            raise ValueError("year_min must be less than or equal to year_max")
        return self

class QueryResponse(BaseModel):
    answer: str
    summary: str = ""
    points: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    facets: Dict[str, Any]
    query: str
    rewritten_query: str


class SearchResponse(BaseModel):
    """Retrieved papers and facet counts. No model output."""
    context_chunks: List[Dict[str, Any]]
    facets: Dict[str, Any]
    query: str


class AnswerRequest(BaseModel):
    """The user query plus chunks from /api/search, or the ids of those chunks."""
    query: str
    chunk_ids: Optional[List[str]] = None
    context_chunks: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def require_chunks_or_ids(self):
        if self.context_chunks is None and not self.chunk_ids:
            raise ValueError("Provide context_chunks or chunk_ids")
        return self


class AnswerResponse(BaseModel):
    """Model answer grounded in the supplied chunks."""
    answer: str
    summary: str = ""
    points: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]]

class APIServer:
    def __init__(self, embedding_model, vector_store, llm_interface, reranker=None):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.reranker = reranker
        self.app = FastAPI(
            title="ArxivLens API",
            description="API for ArxivLens",
            version=APP_VERSION,
            lifespan=self._lifespan,
        )
        self._register_routes()
        self._configure_cors()

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        """Open the index and load the embedding model before the server accepts traffic."""
        logger.info("Loading embedding model and vector store")
        self.embedding_model.load_model()
        self.vector_store.initialize()
        yield

    def _register_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "ArxivLens API is running"}

        @self.app.post("/api/query", response_model=QueryResponse)
        def query(request: QueryRequest):
            try:
                started_at = time.perf_counter()
                rewritten_query, rewrite_elapsed = self._rewrite_query(request.query)
                context_chunks, facets, stage_times = self._retrieve_papers(
                    query_text=rewritten_query,
                    top_k=request.top_k,
                    filters=self._request_filters(request),
                )
                generated, generation_elapsed = self._generate_answer(request.query, context_chunks)
                total_elapsed = time.perf_counter() - started_at
                logger.info(
                    "Query served in %.2fs (rewrite=%.2fs, embed=%.2fs, retrieval=%.2fs, rerank=%.2fs, generation=%.2fs)",
                    total_elapsed,
                    rewrite_elapsed,
                    stage_times["embed"],
                    stage_times["retrieval"],
                    stage_times["rerank"],
                    generation_elapsed,
                )
                return {
                    **generated,
                    "context_chunks": context_chunks,
                    "facets": facets,
                    "query": request.query,
                    "rewritten_query": rewritten_query,
                }
            except Exception as exc:
                logger.exception("Query failed")
                raise HTTPException(status_code=500, detail="Query failed.") from exc

        @self.app.post("/api/search", response_model=SearchResponse)
        def search(request: QueryRequest):
            try:
                started_at = time.perf_counter()
                context_chunks, facets, stage_times = self._retrieve_papers(
                    query_text=request.query,
                    top_k=request.top_k,
                    filters=self._request_filters(request),
                )
                total_elapsed = time.perf_counter() - started_at
                logger.info(
                    "Search served in %.2fs (embed=%.2fs, retrieval=%.2fs, rerank=%.2fs)",
                    total_elapsed,
                    stage_times["embed"],
                    stage_times["retrieval"],
                    stage_times["rerank"],
                )
                return {
                    "context_chunks": context_chunks,
                    "facets": facets,
                    "query": request.query,
                }
            except Exception as exc:
                logger.exception("Search failed")
                raise HTTPException(status_code=500, detail="Search failed.") from exc

        @self.app.post("/api/answer", response_model=AnswerResponse)
        def answer(request: AnswerRequest):
            try:
                started_at = time.perf_counter()
                context_chunks = self._chunks_for_answer(request)
                generated, generation_elapsed = self._generate_answer(request.query, context_chunks)
                total_elapsed = time.perf_counter() - started_at
                logger.info(
                    "Answer served in %.2fs (generation=%.2fs)",
                    total_elapsed,
                    generation_elapsed,
                )
                return generated
            except Exception as exc:
                logger.exception("Answer failed")
                raise HTTPException(status_code=500, detail="Answer failed.") from exc

        @self.app.get("/api/health")
        async def health_check():
            return {"status": "healthy"}

        @self.app.get("/api/readiness")
        async def readiness_check():
            try:
                # Avoid full initialization inside readiness probe; only verify component state.
                model_loaded = self.embedding_model.model is not None
                collection_ready = self.vector_store.collection is not None

                if not model_loaded or not collection_ready:
                    raise HTTPException(
                        status_code=503,
                        detail={
                            "status": "not_ready",
                            "embedding_model_loaded": model_loaded,
                            "vector_collection_ready": collection_ready,
                        },
                    )

                return {
                    "status": "ready",
                    "embedding_model_loaded": model_loaded,
                    "vector_collection_ready": collection_ready,
                }
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Readiness check failed")
                raise HTTPException(status_code=503, detail={"status": "not_ready"}) from exc

    def _request_filters(self, request: QueryRequest) -> Dict[str, Any]:
        return {
            "authors": request.authors,
            "categories": request.categories,
            "year_min": request.year_min,
            "year_max": request.year_max,
        }

    def _rewrite_query(self, query: str) -> Tuple[str, float]:
        started_at = time.perf_counter()
        rewritten_query = self.llm_interface.rewrite_query(query)
        return rewritten_query, time.perf_counter() - started_at

    def _retrieve_papers(
        self,
        query_text: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, float]]:
        """Embed, retrieve, rerank, and count facets. Does not call the LLM."""
        effective_top_k = min(top_k, MAX_QUERY_TOP_K)
        embed_started_at = time.perf_counter()
        query_embedding = self.embedding_model.embed_query(query_text)
        embed_elapsed = time.perf_counter() - embed_started_at

        # Retrieve a wider set than top_k so the reranker can reorder
        # unique papers, then cut the response back to top_k.
        retrieval_started_at = time.perf_counter()
        reranker_max_candidates = (
            self.reranker.max_candidates if self.reranker is not None else effective_top_k
        )
        retrieve_k = max(effective_top_k, reranker_max_candidates)
        context_chunks = self.vector_store.query(
            query_embedding=query_embedding,
            query_text=query_text,
            n_results=retrieve_k,
            filters=filters,
        )
        retrieval_elapsed = time.perf_counter() - retrieval_started_at

        rerank_started_at = time.perf_counter()
        if self.reranker is not None:
            context_chunks = self.reranker.rerank(
                query_text=query_text,
                candidates=context_chunks,
                final_top_k=effective_top_k,
            )
        rerank_elapsed = time.perf_counter() - rerank_started_at
        facets = self.vector_store.get_facets(context_chunks)
        return context_chunks, facets, {
            "embed": embed_elapsed,
            "retrieval": retrieval_elapsed,
            "rerank": rerank_elapsed,
        }

    def _chunks_for_answer(self, request: AnswerRequest) -> List[Dict[str, Any]]:
        """Use the chunks the client sent. Look up ids only when chunks were not sent."""
        if request.context_chunks is not None:
            return list(request.context_chunks)
        return self.vector_store.get_chunks_by_ids(request.chunk_ids or [])

    def _generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], float]:
        """Ask the model for a grounded answer, or skip it when there are no chunks."""
        generation_started_at = time.perf_counter()
        if not context_chunks:
            answer = INSUFFICIENT_CONTEXT_ANSWER
            summary = answer
            points: List[Dict[str, Any]] = []
            citations: List[Dict[str, Any]] = []
            logger.info("Skipped answer generation because no chunks passed the reranker cutoff")
        else:
            llm_output = self.llm_interface.generate_response(
                query=query,
                context_chunks=context_chunks,
            )
            answer = llm_output.get("answer", EMPTY_ANSWER)
            summary = str(llm_output.get("summary", answer))
            points = llm_output.get("points") or []
            if not isinstance(points, list):
                points = []
            citations = llm_output.get("citations", [])
        generation_elapsed = time.perf_counter() - generation_started_at
        return {
            "answer": answer,
            "summary": summary,
            "points": points,
            "citations": citations,
        }, generation_elapsed

    def _configure_cors(self):
        allow_origins = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        uvicorn.run(self.app, host=host or HOST, port=port or PORT)
