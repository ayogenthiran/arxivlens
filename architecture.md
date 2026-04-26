# ArxivLens Architecture

## Runtime Components

1. **Data Processing**
   - Loads source documents from `data/raw/`
   - Splits documents into chunks
   - Persists processed chunks to `data/processed/`

2. **Embedding Layer**
   - Generates embeddings for chunks and queries
   - Uses sentence-transformers model configured via environment

3. **Vector Store**
   - Stores and retrieves embeddings using Chroma
   - Persists data under `data/vector_store/`

4. **LLM Response Layer**
   - Builds prompt from retrieved context chunks
   - Generates final answer through configured LLM API

5. **FastAPI Service (`main.py`)**
   - Exposes:
     - `GET /`
     - `GET /api/health`
     - `POST /api/query`
   - Coordinates embedding, retrieval, and generation

6. **Streamlit UI (`streamlit_app.py`)**
   - Simple query input + response view
   - Calls backend endpoint `POST /api/query`
   - Shows retrieved context chunks for transparency

## Request Flow

1. User submits question in Streamlit UI
2. UI sends `query` and `top_k` to FastAPI
3. FastAPI embeds the query
4. Vector store returns most relevant chunks
5. LLM interface generates grounded answer
6. API returns answer + context chunks to UI

## Directory Ownership

- `main.py`: backend process entrypoint
- `streamlit_app.py`: frontend process entrypoint
- `src/api`: API server wiring
- `src/data_processing`: ingestion + chunking
- `src/embeddings`: embedding generation
- `src/vector_store`: vector DB integration
- `src/llm`: response generation logic
