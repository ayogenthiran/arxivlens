# src/api/server.py

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import uvicorn

from src.llm.llm_interface import INSUFFICIENT_CONTEXT_ANSWER
from src.settings import (
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
    """Request model for query endpoint"""
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
    """Response model for query endpoint"""
    answer: str
    citations: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    facets: Dict[str, Any]
    query: str
    rewritten_query: str

class APIServer:
    """
    FastAPI server for the RAG application.
    """
    
    def __init__(self, embedding_model, vector_store, llm_interface, reranker=None):
        """Initialize the API server with injected components"""
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_interface = llm_interface
        self.reranker = reranker
        self.app = FastAPI(
            title="ArxivLens API",
            description="API for ArxivLens",
            version="1.0.0",
            lifespan=self._lifespan,
        )

        # Register routes
        self._register_routes()

        # Configure CORS (important for frontend React app)
        self._configure_cors()

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI):
        """Open the index and load the embedding model before the server accepts traffic."""
        logger.info("Loading embedding model and vector store")
        self.embedding_model.load_model()
        self.vector_store.initialize()
        yield

    def _register_routes(self):
        """Register API routes"""

        @self.app.get("/")
        async def root():
            return {"message": "ArxivLens API is running"}

        @self.app.post("/api/query", response_model=QueryResponse)
        def query(request: QueryRequest):
            try:
                effective_top_k = min(request.top_k, MAX_QUERY_TOP_K)
                started_at = time.perf_counter()
                rewritten_query = self.llm_interface.rewrite_query(request.query)
                rewrite_elapsed = time.perf_counter() - started_at

                # Generate query embedding
                embed_started_at = time.perf_counter()
                query_embedding = self.embedding_model.embed_query(rewritten_query)
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
                    query_text=rewritten_query,
                    n_results=retrieve_k,
                    filters={
                        "authors": request.authors,
                        "categories": request.categories,
                        "year_min": request.year_min,
                        "year_max": request.year_max,
                    },
                )
                retrieval_elapsed = time.perf_counter() - retrieval_started_at

                rerank_started_at = time.perf_counter()
                if self.reranker is not None:
                    context_chunks = self.reranker.rerank(
                        query_text=rewritten_query,
                        candidates=context_chunks,
                        final_top_k=effective_top_k,
                    )
                rerank_elapsed = time.perf_counter() - rerank_started_at
                facets = self.vector_store.get_facets(context_chunks)

                # Weak chunks are already gone. If none remain, do not ask the model to write.
                generation_started_at = time.perf_counter()
                if not context_chunks:
                    answer = INSUFFICIENT_CONTEXT_ANSWER
                    citations = []
                    logger.info("Skipped answer generation because no chunks passed the reranker cutoff")
                else:
                    llm_output = self.llm_interface.generate_response(
                        query=request.query,
                        context_chunks=context_chunks
                    )
                    answer = llm_output.get("answer", "No answer returned.")
                    citations = llm_output.get("citations", [])
                generation_elapsed = time.perf_counter() - generation_started_at
                total_elapsed = time.perf_counter() - started_at
                logger.info(
                    "Query served in %.2fs (rewrite=%.2fs, embed=%.2fs, retrieval=%.2fs, rerank=%.2fs, generation=%.2fs)",
                    total_elapsed,
                    rewrite_elapsed,
                    embed_elapsed,
                    retrieval_elapsed,
                    rerank_elapsed,
                    generation_elapsed,
                )

                return {
                    "answer": answer,
                    "citations": citations,
                    "context_chunks": context_chunks,
                    "facets": facets,
                    "query": request.query,
                    "rewritten_query": rewritten_query,
                }
            except Exception as exc:
                logger.exception("Query failed")
                raise HTTPException(status_code=500, detail="Query failed.") from exc

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

    def _configure_cors(self):
        """Configure CORS to allow frontend requests"""
        allow_origins = [origin.strip() for origin in CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        """Run the API server"""
        uvicorn.run(self.app, host=host or HOST, port=port or PORT)

# For ASGI servers
app_instance = None
app = None

def create_app(embedding_model, vector_store, llm_interface, reranker=None):
    global app_instance, app
    app_instance = APIServer(embedding_model, vector_store, llm_interface, reranker=reranker)
    app = app_instance.app
    return app
