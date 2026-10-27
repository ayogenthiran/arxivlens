"""Encode and decode the Streamlit search URL.

Query params use the same names as the retrieval request: ``query``, ``top_k``,
``categories``, ``authors``, ``year_min``, and ``year_max``.
"""

from collections.abc import Mapping
from typing import Any

from src.settings import (
    DEFAULT_TOP_K,
    UI_MAX_TOP_K,
    UI_MIN_TOP_K,
    YEAR_MAX_BOUND,
    YEAR_MIN_BOUND,
)

QUERY_PARAM = "query"
TOP_K_PARAM = "top_k"
CATEGORIES_PARAM = "categories"
AUTHORS_PARAM = "authors"
YEAR_MIN_PARAM = "year_min"
YEAR_MAX_PARAM = "year_max"

_LIST_PARAMS = (CATEGORIES_PARAM, AUTHORS_PARAM)
_YEAR_PARAMS = (YEAR_MIN_PARAM, YEAR_MAX_PARAM)


def encode_search_params(query: str, top_k: int, filters: Mapping[str, Any]) -> dict[str, str]:
    """Build the query string that reproduces this search."""
    params: dict[str, str] = {}
    text = str(query or "").strip()
    if text:
        params[QUERY_PARAM] = text
    params[TOP_K_PARAM] = str(_clamp_top_k(top_k))
    categories = _clean_terms(filters.get(CATEGORIES_PARAM))
    if categories:
        params[CATEGORIES_PARAM] = ", ".join(categories)
    authors = _clean_terms(filters.get(AUTHORS_PARAM))
    if authors:
        params[AUTHORS_PARAM] = ", ".join(authors)
    for key in _YEAR_PARAMS:
        year = _coerce_year(filters.get(key))
        if year is not None:
            params[key] = str(year)
    return params


def decode_search_params(params: object) -> tuple[str, int, dict[str, Any]]:
    """Read a search from a mapping or from ``st.query_params``.

    Missing or invalid ``top_k`` falls back to the default, then clamps to the
    slider range. Invalid years are dropped. Category and author values may be
    repeated or comma-separated.
    """
    query_values = _param_values(params, QUERY_PARAM)
    query = query_values[-1].strip() if query_values else ""
    top_k_values = _param_values(params, TOP_K_PARAM)
    top_k = _clamp_top_k(top_k_values[-1] if top_k_values else None)
    filters = {
        CATEGORIES_PARAM: _clean_terms(_param_values(params, CATEGORIES_PARAM)),
        AUTHORS_PARAM: _clean_terms(_param_values(params, AUTHORS_PARAM)),
        YEAR_MIN_PARAM: _last_year(params, YEAR_MIN_PARAM),
        YEAR_MAX_PARAM: _last_year(params, YEAR_MAX_PARAM),
    }
    return query, top_k, filters


def present_search_params(params: object) -> dict[str, str]:
    """Normalize the search keys that are actually present.

    List values are joined the same way ``encode_search_params`` writes them.
    ``top_k`` and years stay as written so an out-of-range URL is not treated
    as already matching the clamped search.
    """
    found: dict[str, str] = {}
    query_values = [value.strip() for value in _param_values(params, QUERY_PARAM) if value.strip()]
    if query_values:
        found[QUERY_PARAM] = query_values[-1]
    top_k_values = [value.strip() for value in _param_values(params, TOP_K_PARAM) if value.strip()]
    if top_k_values:
        found[TOP_K_PARAM] = top_k_values[-1]
    for key in _LIST_PARAMS:
        terms = _clean_terms(_param_values(params, key))
        if terms:
            found[key] = ", ".join(terms)
    for key in _YEAR_PARAMS:
        values = [value.strip() for value in _param_values(params, key) if value.strip()]
        if values:
            found[key] = values[-1]
    return found


def _param_values(params: object, key: str) -> list[str]:
    get_all = getattr(params, "get_all", None)
    if callable(get_all):
        return [str(value) for value in get_all(key)]
    if not isinstance(params, Mapping) or key not in params:
        return []
    raw = params[key]
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    if raw is None:
        return []
    return [str(raw)]


def _clean_terms(value: object) -> list[str]:
    if value is None:
        return []
    parts: list[str] = []
    items = value if isinstance(value, (list, tuple)) else [value]
    for item in items:
        parts.extend(str(item).split(","))
    return [part.strip() for part in parts if part.strip()]


def _parse_int(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.startswith("+"):
        text = text[1:]
    if not text:
        return None
    digits = text[1:] if text.startswith("-") else text
    if not digits.isdigit():
        return None
    return int(text)


def _clamp_top_k(raw: object) -> int:
    parsed = _parse_int(raw)
    if parsed is None:
        return DEFAULT_TOP_K
    return min(max(parsed, UI_MIN_TOP_K), UI_MAX_TOP_K)


def _coerce_year(raw: object) -> int | None:
    if isinstance(raw, str) and not raw.strip():
        return None
    parsed = _parse_int(raw)
    if parsed is None:
        return None
    if YEAR_MIN_BOUND <= parsed <= YEAR_MAX_BOUND:
        return parsed
    return None


def _last_year(params: object, key: str) -> int | None:
    values = [value.strip() for value in _param_values(params, key) if value.strip()]
    if not values:
        return None
    return _coerce_year(values[-1])
