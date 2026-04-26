# src/api/server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import time
import logging
import uvicorn

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str
    top_k: int = Field(default=8, ge=1, le=50)

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    context_chunks: List[Dict[str, Any]]
    query: str
    rewritten_query: str

class APIServer:
    """
    FastAPI server for the RAG application.
    """
    
    def __init__(self, embedding_model, vector_store, llm_interface):
        """Initialize the API server with injected components"""
        self.app = FastAPI(
            title="ArxivLens API",
            description="API for ArxivLens",
            version="1.0.0"
        )

        # Inject components
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm_interface = llm_interface

        # Register routes
        self._register_routes()

        # Configure CORS (important for frontend React app)
        self._configure_cors()

    def _register_routes(self):
        """Register API routes"""

        @self.app.get("/")
        async def root():
            return {"message": "ArxivLens API is running"}

        @self.app.post("/api/query", response_model=QueryResponse)
        def query(request: QueryRequest):
            try:
                max_top_k = int(os.getenv("MAX_QUERY_TOP_K", "20"))
                effective_top_k = min(request.top_k, max_top_k)
                started_at = time.perf_counter()
                rewritten_query = self.llm_interface.rewrite_query(request.query)
                rewrite_elapsed = time.perf_counter() - started_at

                # Generate query embedding
                embed_started_at = time.perf_counter()
                query_embedding = self.embedding_model.embed_query(rewritten_query)
                embed_elapsed = time.perf_counter() - embed_started_at

                # Retrieve relevant chunks
                retrieval_started_at = time.perf_counter()
                context_chunks = self.vector_store.query(
                    query_embedding=query_embedding,
                    query_text=rewritten_query,
                    n_results=effective_top_k
                )
                retrieval_elapsed = time.perf_counter() - retrieval_started_at

                # Generate response
                generation_started_at = time.perf_counter()
                answer = self.llm_interface.generate_response(
                    query=request.query,
                    context_chunks=context_chunks
                )
                generation_elapsed = time.perf_counter() - generation_started_at
                total_elapsed = time.perf_counter() - started_at
                logger.info(
                    "Query served in %.2fs (rewrite=%.2fs, embed=%.2fs, retrieval=%.2fs, generation=%.2fs)",
                    total_elapsed,
                    rewrite_elapsed,
                    embed_elapsed,
                    retrieval_elapsed,
                    generation_elapsed,
                )

                return {
                    "answer": answer,
                    "context_chunks": context_chunks,
                    "query": request.query,
                    "rewritten_query": rewritten_query,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

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
            except Exception as e:
                raise HTTPException(status_code=503, detail={"status": "not_ready", "error": str(e)})

    def _configure_cors(self):
        """Configure CORS to allow frontend requests"""
        raw_origins = os.getenv(
            "CORS_ALLOW_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        )
        allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port)

# For ASGI servers
app_instance = None
app = None

def create_app(embedding_model, vector_store, llm_interface):
    global app_instance, app
    app_instance = APIServer(embedding_model, vector_store, llm_interface)
    app = app_instance.app
    return app
