"""Empty-state example searches.

Each example is a query plus optional sidebar presets. ``published`` values such
as ``Since 2025`` become a year filter. The chip shows that preset as a short
suffix, for example ``· since 2025``.
"""

from collections.abc import Mapping
from typing import Any

from src.settings import YEAR_MAX_BOUND, YEAR_MIN_BOUND

EXAMPLE_PROMPT = "Not sure where to start? Try one of these:"
MIN_EXAMPLE_PAPERS = 5
# A kept hit at the reranker cutoff is not enough. The top paper must clear it by this margin.
EXAMPLE_RAW_SCORE_MARGIN = 1.0
EXAMPLE_QUESTIONS: list[dict[str, Any]] = [
    {
        "label": "Contrastive pretraining for low-resource machine translation",
        "query": "Contrastive pretraining for low-resource machine translation",
        "filters": {},
    },
    {
        "label": "Evaluating LLM reasoning without benchmark contamination",
        "query": "Evaluating LLM reasoning without benchmark contamination",
        "filters": {"published": "Since 2025"},
    },
    {
        "label": "Retrieval-augmented generation for scientific question answering",
        "query": "Retrieval-augmented generation for scientific question answering",
        "filters": {},
    },
]


def example_filter_suffix(filters: Mapping[str, Any] | None) -> str:
    """Short chip suffix for a preset, such as ``since 2025``."""
    published = _published_text(filters)
    if not published:
        return ""
    return published.lower()


def example_chip_label(example: Mapping[str, Any]) -> str:
    """Button label. A preset suffix is italic so the chip can draw it smaller."""
    label = str(example.get("label") or "").strip()
    suffix = example_filter_suffix(example.get("filters") or {})
    if not suffix:
        return label
    return f"{label} · *{suffix}*"


def example_retrieval_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Map an example preset onto the search request fields."""
    raw = filters or {}
    year_min = _optional_year(raw.get("year_min"))
    year_max = _optional_year(raw.get("year_max"))
    published = _published_text(raw)
    if published and year_min is None and year_max is None:
        year_min = _since_year(published)
    return {
        "categories": _terms(raw.get("categories")),
        "authors": _terms(raw.get("authors")),
        "year_min": year_min,
        "year_max": year_max,
    }


def top_reranker_raw(chunks: list) -> float | None:
    """Highest ``reranker_score_raw`` in a search response."""
    best: float | None = None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        raw = chunk.get("reranker_score_raw")
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        if best is None or score > best:
            best = score
    return best


def example_result_problem(paper_count: int, top_raw: float | None, min_raw: float) -> str | None:
    """Why an example search fails, or None when it meets the corpus bar."""
    if paper_count < MIN_EXAMPLE_PAPERS:
        return f"{paper_count} papers"
    if top_raw is None or top_raw < min_raw + EXAMPLE_RAW_SCORE_MARGIN:
        shown = "missing" if top_raw is None else f"{top_raw:.2f}"
        return f"top raw {shown}"
    return None


def _published_text(filters: Mapping[str, Any] | None) -> str:
    if not filters:
        return ""
    return " ".join(str(filters.get("published") or "").split())


def _since_year(published: str) -> int:
    parts = published.split()
    if len(parts) == 2 and parts[0].lower() == "since" and parts[1].isdigit():
        year = int(parts[1])
        if YEAR_MIN_BOUND <= year <= YEAR_MAX_BOUND:
            return year
        raise ValueError(f"Published year {year} is outside {YEAR_MIN_BOUND}-{YEAR_MAX_BOUND}.")
    raise ValueError(f"Unrecognized published preset: {published!r}.")


def _optional_year(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or not text.isdigit():
        return None
    return int(text)


def _terms(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    else:
        return []
    return [part.strip() for part in raw_items if part.strip()]
