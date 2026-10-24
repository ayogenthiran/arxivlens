"""Runtime configuration.

Defaults live here. Override any of them with an environment variable or `.env`.
`load_dotenv()` does not replace variables that are already set in the process.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def env_str(name: str, default: str) -> str:
    return _raw(name) or default


def env_int(name: str, default: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _warn(f"Invalid integer for {name}={raw!r}. Using {default}.")
        return default


def env_float(name: str, default: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        _warn(f"Invalid number for {name}={raw!r}. Using {default}.")
        return default


def env_bool(name: str, default: bool) -> bool:
    """True only when the variable is the word `true`. Missing keeps the default."""
    raw = _raw(name)
    if raw is None:
        return default
    return raw.lower() == "true"


def env_int_min(name: str, default: int, minimum: int) -> int:
    raw = _raw(name)
    value = env_int(name, default)
    if value >= minimum:
        return value
    if raw is not None:
        _warn(f"{name}={value} is below {minimum}. Using {minimum}.")
    return minimum


def env_unit_float(name: str, default: float) -> float:
    """A fraction from 0 through 1. Invalid values keep the default."""
    raw = _raw(name)
    value = env_float(name, default)
    if 0.0 <= value <= 1.0:
        return value
    if raw is not None:
        _warn(f"{name}={value} must be between 0 and 1. Using {default}.")
    return default


def _port(name: str, default: int) -> int:
    raw = _raw(name)
    value = env_int(name, default)
    if 1 <= value <= 65535:
        return value
    if raw is not None:
        _warn(f"Invalid {name}={value}. Using {default}.")
    return default


# Paths
DATA_DIR = env_str("DATA_DIR", "data/raw")
OUTPUT_DIR = env_str("OUTPUT_DIR", "data/processed")
VECTOR_STORE_DIR = env_str("VECTOR_STORE_DIR", "data/vector_store")
CHROMA_COLLECTION_NAME = env_str("CHROMA_COLLECTION_NAME", "documents")

# Server
HOST = env_str("HOST", "127.0.0.1")
PORT = _port("PORT", 8000)
API_BASE_URL = env_str("API_BASE_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
CORS_ALLOW_ORIGINS = env_str(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8501,http://127.0.0.1:8501",
)
API_QUERY_TIMEOUT_S = env_int_min("API_QUERY_TIMEOUT_S", 60, 1)

# Retrieval size. The UI slider cannot ask for more than the API accepts.
MAX_QUERY_TOP_K = env_int_min("MAX_QUERY_TOP_K", 20, 1)
_configured_ui_max = env_int("UI_MAX_TOP_K", 20)
if _raw("UI_MAX_TOP_K") is not None and _configured_ui_max > MAX_QUERY_TOP_K:
    _warn(
        f"UI_MAX_TOP_K={_configured_ui_max} is above MAX_QUERY_TOP_K={MAX_QUERY_TOP_K}. "
        f"The slider max is {MAX_QUERY_TOP_K}."
    )
UI_MAX_TOP_K = min(max(_configured_ui_max, 1), MAX_QUERY_TOP_K)
UI_MIN_TOP_K = min(env_int_min("UI_MIN_TOP_K", 3, 1), UI_MAX_TOP_K)
DEFAULT_TOP_K = min(max(env_int("UI_DEFAULT_TOP_K", 8), UI_MIN_TOP_K), UI_MAX_TOP_K)
YEAR_MIN_BOUND = env_int("YEAR_MIN_BOUND", 1900)
YEAR_MAX_BOUND = env_int("YEAR_MAX_BOUND", 2100)
if YEAR_MIN_BOUND > YEAR_MAX_BOUND:
    _warn(
        f"YEAR_MIN_BOUND={YEAR_MIN_BOUND} is above YEAR_MAX_BOUND={YEAR_MAX_BOUND}. Swapping them."
    )
    YEAR_MIN_BOUND, YEAR_MAX_BOUND = YEAR_MAX_BOUND, YEAR_MIN_BOUND

# Models
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = env_str("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
LLM_MODEL_NAME = env_str("LLM_MODEL_NAME", "gpt-4o-mini")
QUERY_REWRITE_MODEL = env_str("QUERY_REWRITE_MODEL", "")
ENABLE_QUERY_REWRITE = env_bool("ENABLE_QUERY_REWRITE", False)
OPENAI_API_BASE = env_str("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.2)
LLM_MAX_TOKENS = env_int_min("LLM_MAX_TOKENS", 1000, 1)
LLM_REQUEST_TIMEOUT_S = env_int_min("LLM_REQUEST_TIMEOUT_S", 15, 1)
QUERY_REWRITE_TEMPERATURE = env_float("QUERY_REWRITE_TEMPERATURE", 0.0)
QUERY_REWRITE_MAX_TOKENS = env_int_min("QUERY_REWRITE_MAX_TOKENS", 64, 1)

# Hybrid search
HYBRID_BM25_MAX_DOCS = env_int_min("HYBRID_BM25_MAX_DOCS", 50000, 1)
RRF_K = env_int_min("RRF_K", 60, 1)
RETRIEVAL_CANDIDATE_MULTIPLIER = env_int_min("RETRIEVAL_CANDIDATE_MULTIPLIER", 8, 1)
RETRIEVAL_FILTER_CANDIDATE_MULTIPLIER = env_int_min("RETRIEVAL_FILTER_CANDIDATE_MULTIPLIER", 40, 1)
RETRIEVAL_FILTER_CANDIDATE_FLOOR = env_int_min("RETRIEVAL_FILTER_CANDIDATE_FLOOR", 200, 1)
FACET_TOP_AUTHORS = env_int_min("FACET_TOP_AUTHORS", 10, 1)
ANONYMIZED_TELEMETRY = env_bool("ANONYMIZED_TELEMETRY", False)

# Reranker
ENABLE_RERANKER = env_bool("ENABLE_RERANKER", True)
RERANKER_MODEL_NAME = env_str("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_MAX_CANDIDATES = env_int_min("RERANKER_MAX_CANDIDATES", 24, 1)
RERANKER_MIN_RAW_SCORE = env_float("RERANKER_MIN_RAW_SCORE", 1.0)

# Ingest
INGEST_BATCH_DOCS = env_int_min("INGEST_BATCH_DOCS", 250, 1)
MAX_DOCS = env_int_min("MAX_DOCS", 0, 0)
INGEST_YEAR_MIN = env_int_min("INGEST_YEAR_MIN", 0, 0)
INGEST_CATEGORIES = env_str("INGEST_CATEGORIES", "")
INGEST_RETRY_MAX_ATTEMPTS = env_int_min("INGEST_RETRY_MAX_ATTEMPTS", 4, 1)
INGEST_RETRY_BASE_DELAY_S = env_float("INGEST_RETRY_BASE_DELAY_S", 1.0)
if INGEST_RETRY_BASE_DELAY_S < 0:
    _warn(f"INGEST_RETRY_BASE_DELAY_S={INGEST_RETRY_BASE_DELAY_S} is negative. Using 1.0.")
    INGEST_RETRY_BASE_DELAY_S = 1.0
RESET_INGEST = env_bool("RESET_INGEST", False)
INGEST_JSONL_STREAM_MIN_BYTES = env_int_min("INGEST_JSONL_STREAM_MIN_BYTES", 50_000_000, 1)
INGEST_YEAR_SEEK_WINDOW_BYTES = env_int_min("INGEST_YEAR_SEEK_WINDOW_BYTES", 1_000_000, 1)
_snapshot_fraction = env_float("INGEST_MODERN_SNAPSHOT_FRACTION", 0.90)
if 0.0 < _snapshot_fraction <= 1.0:
    INGEST_MODERN_SNAPSHOT_FRACTION = _snapshot_fraction
else:
    if _raw("INGEST_MODERN_SNAPSHOT_FRACTION") is not None:
        _warn(
            f"INGEST_MODERN_SNAPSHOT_FRACTION={_snapshot_fraction} must be greater than 0 and at most 1. "
            "Using 0.90."
        )
    INGEST_MODERN_SNAPSHOT_FRACTION = 0.90

# UI
UI_CONFIDENCE_HIGH = env_unit_float("UI_CONFIDENCE_HIGH", 0.72)
UI_CONFIDENCE_MEDIUM = env_unit_float("UI_CONFIDENCE_MEDIUM", 0.45)
UI_FACET_LIMIT = env_int_min("UI_FACET_LIMIT", 5, 1)
UI_MAX_AUTHORS_SHOWN = env_int_min("UI_MAX_AUTHORS_SHOWN", 3, 1)
ARXIV_ABS_BASE_URL = env_str("ARXIV_ABS_BASE_URL", "https://arxiv.org/abs").rstrip("/")

# Eval
EVAL_SUPPORT_THRESHOLD = env_unit_float("EVAL_SUPPORT_THRESHOLD", 0.55)
CITATION_GROUNDEDNESS_MIN = env_unit_float("CITATION_GROUNDEDNESS_MIN", 0.8)
# Optional Semantic Scholar Graph API key. scripts/build_citation_benchmark.py
# sends it as the x-api-key header. Empty uses the shared unauthenticated limit.
S2_API_KEY = _raw("S2_API_KEY") or ""
