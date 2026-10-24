# ArxivLens

ArxivLens is a paper-discovery tool for Master's and PhD students. It searches arXiv metadata (title, abstract, authors, categories, and year), reranks the short list, and answers with citations that open the paper on arxiv.org.

## Status

The live corpus is 5,000 recent `cs.CL` and `cs.LG` papers, one chunk each, in `data/vector_store_corpus`. Ingest kept papers whose modern arXiv id is from 2024 onward and that list `cs.CL` or `cs.LG`. The full Kaggle archive in `data/vector_store` is optional. The app reads whichever directory `VECTOR_STORE_DIR` points at. `data/` is gitignored, so a clone does not include either index.

## Architecture

Ingest and query are drawn in [architecture.md](architecture.md).

## Quickstart

Python 3.11 or newer. Docker is optional. The API listens on port 8000. The Streamlit UI listens on port 8501.

### Local

```bash
git clone https://github.com/ayogenthiran/arxivlens.git
cd arxivlens
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
uv sync --frozen
# or, without uv: pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. To query the live corpus, also set:

```env
OUTPUT_DIR=data/processed_corpus
VECTOR_STORE_DIR=data/vector_store_corpus
```

Start the API and the UI in two terminals:

```bash
python main.py
streamlit run streamlit_app.py --server.port 8501
```

API: http://127.0.0.1:8000

UI: http://localhost:8501

Ask a question in the UI. Advanced settings change how many sources to retrieve and can filter by category, author, and year.

### Docker

`docker-compose.yml` builds one image and runs the API and the UI. The API container sets `HOST=0.0.0.0` so the published port is reachable. The UI calls the API at `http://api:8000`. Both services mount `./data`.

```bash
cp .env.example .env
docker compose up --build
```

API: http://127.0.0.1:8000

UI: http://localhost:8501

Ingest is a separate Compose profile and does not start with `up`:

```bash
docker compose --profile ingest run --rm ingest
```

## Configuration

Defaults live in `src/settings.py`. A `.env` file overrides them. `load_dotenv()` does not replace variables already set in the process.

| Variable | Default | What it does |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | API bind address. Compose sets `0.0.0.0`. |
| `PORT` | `8000` | API port. |
| `UI_PORT` | `8501` | Streamlit port in Compose. |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Where the UI sends queries. Compose sets `http://api:8000`. |
| `OPENAI_API_KEY` | | Key for the chat model. Answers fail without it. |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible base URL. |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Answer model. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Document and query embeddings. |
| `DATA_DIR` | `data/raw` | Directory of the metadata snapshot. |
| `OUTPUT_DIR` | `data/processed` | Checkpoint and `processed_chunks.jsonl`. |
| `VECTOR_STORE_DIR` | `data/vector_store` | Chroma directory. The live corpus is `data/vector_store_corpus`. |
| `MAX_DOCS` | `0` | Ingest cap. `0` means no cap. The live corpus used `5000`. |
| `INGEST_CATEGORIES` | empty | Comma-separated arXiv categories. Empty means every category. |
| `INGEST_YEAR_MIN` | `0` | Skip papers submitted before this year. `0` means any year. |
| `RESET_INGEST` | `false` | Delete the checkpoint and the chunk sidecar. Does not delete Chroma. |
| `ENABLE_RERANKER` | `true` | Cross-encoder rerank after retrieval. |
| `RERANKER_MIN_RAW_SCORE` | `1` | Drop chunks whose raw cross-encoder logit is below this. |
| `ENABLE_QUERY_REWRITE` | `false` | When false, retrieval uses the query as typed. |
| `HYBRID_BM25_MAX_DOCS` | `50000` | Above this size, search is vector-only. |
| `UI_DEFAULT_TOP_K` | `8` | Default number of sources. The API caps this at `MAX_QUERY_TOP_K` (20). |

`requirements.txt` is the install source for pip and Docker. Keep its pins aligned with `pyproject.toml`, including the PyTorch CPU index. `chromadb==1.0.7` stays paired with `posthog==3.25.0`.

## Ingest

Put the Kaggle arXiv metadata file at `data/raw/arxiv-metadata-oai-snapshot.json` ([Cornell arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)). `main.py` either ingests or serves the API. One process does not do both.

The settings that built the live corpus:

```env
OUTPUT_DIR=data/processed_corpus
VECTOR_STORE_DIR=data/vector_store_corpus
MAX_DOCS=5000
INGEST_CATEGORIES=cs.CL,cs.LG
INGEST_YEAR_MIN=2024
RESET_INGEST=false
```

```bash
python main.py --process-data
```

That writes one chunk per paper into Chroma and appends the same batch to `processed_chunks.jsonl`. `ingest_checkpoint.json` stores how many source lines have been read. The next run skips lines below that count. `MAX_DOCS=0` walks the snapshot with no paper cap. A run that can pass 50,000 papers prints a warning that BM25 will turn off.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

Tests do not need the network, `OPENAI_API_KEY`, or a model download. Missing `chromadb`, `sentence-transformers`, and `torch` are stubbed in `tests/conftest.py`. CI on Python 3.11 runs the same ruff and pytest commands.

## Evaluation

`evals/retrieval_eval.py` scores a JSONL query set against the local index. It does not write to the index. The embedding model downloads on first use. `--run-groundedness` also calls the chat model and needs `OPENAI_API_KEY`.

```bash
python evals/retrieval_eval.py --eval-set evals/abstract_questions.jsonl
python evals/retrieval_eval.py \
  --eval-set evals/abstract_questions.jsonl \
  --output evals/results/latest_eval.json
```

`evals/results/latest_eval.json` records 10 queries from `evals/abstract_questions.jsonl`: Recall@8 = 1.0 and MRR = 1.0. Each gold chunk is rank 1. Those queries include distinctive title terms from the target paper, so this is a known-item lookup check. A perfect score shows the index can return that paper when the query already contains those terms. It is not a measure of discovery quality. Field definitions are in `evals/README.md`.

## Project structure

```
.
├── .github/workflows/     # ruff and pytest
├── assets/                # UI logo
├── data/                  # gitignored; present after you add a snapshot or index
│   ├── arxiv/             # extra local copy of the snapshot
│   ├── raw/
│   ├── processed/         # full-archive checkpoint and sidecar
│   ├── processed_corpus/  # live-corpus checkpoint and sidecar
│   ├── vector_store/      # optional full archive
│   └── vector_store_corpus/
├── evals/
│   └── results/
├── src/
│   ├── api/
│   ├── data_processing/
│   ├── embeddings/
│   ├── llm/
│   ├── reranker/
│   ├── ui/
│   └── vector_store/
└── tests/
```

Entry points: `main.py` (ingest or API), `streamlit_app.py` (UI), `docker-compose.yml`, `Dockerfile`.

## Limitations

- Search covers title and abstract metadata, not the PDF.
- The live set is 5,000 papers, not the archive. 273 of them store `year` 2023 because `year` is the first submission date, which can be earlier than the 2024 id.
- The full archive still has multiple chunks per paper from an older ingest (5,209,090 chunk ids, 3,021,737 of them ending in `-0`). It is above 50,000 documents, so BM25 stays off.
- The live corpus uses cosine distance. The full archive is still L2, and an existing collection is not converted in place.
- Query rewriting is implemented and left off.
- Keyword search loads every chunk into memory.

## Roadmap

- Grow the live corpus with `MAX_DOCS`, `INGEST_CATEGORIES`, and `INGEST_YEAR_MIN`, and keep it under 50,000 papers so keyword search stays on.
- Add eval questions that do not repeat the paper's distinctive title terms.
- Turn query rewriting on only if a rewrite improves retrieval.

## Acknowledgements

<!-- TODO(Anojan): credit original template if this repo was derived from one -->
