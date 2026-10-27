"""Sidebar filter count, reset, and the empty-author message.

Three controls count: categories, a published range other than the full
"Any time" span, and author. Each one counts once.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from src.settings import YEAR_MAX_BOUND, YEAR_MIN_BOUND
from src.ui.search_params import AUTHORS_PARAM, CATEGORIES_PARAM, YEAR_MAX_PARAM, YEAR_MIN_PARAM

AUTHOR_INPUT_HINT = "Matches any author whose name contains this text."
EMPTY_RESULTS_MESSAGE = "No papers matched this search."
_FILTER_QUERY_PARAMS = (CATEGORIES_PARAM, AUTHORS_PARAM, YEAR_MIN_PARAM, YEAR_MAX_PARAM)


def active_filter_count(filters: Mapping[str, Any]) -> int:
    """How many of categories, published range, and author currently narrow results."""
    count = 0
    if _terms(filters.get(CATEGORIES_PARAM)):
        count += 1
    if _published_range_is_set(filters):
        count += 1
    if _terms(filters.get(AUTHORS_PARAM)):
        count += 1
    return count


def filters_heading(count: int) -> str:
    """Sidebar title. A zero count stays 'Filters'."""
    if count > 0:
        return f"Filters ({count})"
    return "Filters"


def author_filter_name(filters: Mapping[str, Any]) -> str:
    """Author text used in the empty-results message."""
    return ", ".join(_terms(filters.get(AUTHORS_PARAM)))


def results_empty_message(paper_count: int, filters: Mapping[str, Any]) -> str | None:
    """Copy for an empty paper list. Papers present means there is nothing to say."""
    if paper_count > 0:
        return None
    name = author_filter_name(filters)
    if name:
        return f"No papers matched author '{name}'. Try a last name only."
    return EMPTY_RESULTS_MESSAGE


def clear_all_filters(
    state: MutableMapping[str, Any],
    query_params: MutableMapping[str, Any],
    *,
    categories_key: str,
    authors_key: str,
    year_key: str,
    extra_blank_keys: tuple[str, ...] = (),
    pop_keys: tuple[str, ...] = (),
) -> None:
    """Reset categories, the year span, and author, plus those query params.

    Query text, result count, and sort are left as they are.
    """
    state[categories_key] = []
    state[authors_key] = ""
    state[year_key] = (YEAR_MIN_BOUND, YEAR_MAX_BOUND)
    for key in extra_blank_keys:
        state[key] = ""
    for key in pop_keys:
        state.pop(key, None)
    _drop_query_params(query_params, _FILTER_QUERY_PARAMS)


def clear_author_filter(
    state: MutableMapping[str, Any],
    query_params: MutableMapping[str, Any],
    *,
    authors_key: str,
) -> None:
    """Clear only the author field and its query param."""
    state[authors_key] = ""
    _drop_query_params(query_params, (AUTHORS_PARAM,))


def _terms(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items: list[object] = [value]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        return []
    found: list[str] = []
    for item in raw_items:
        for part in str(item).split(","):
            text = part.strip()
            if text:
                found.append(text)
    return found


def _published_range_is_set(filters: Mapping[str, Any]) -> bool:
    """True when the year span is narrower than the slider's full range."""
    start = _year_edge(filters.get(YEAR_MIN_PARAM), YEAR_MIN_BOUND)
    end = _year_edge(filters.get(YEAR_MAX_PARAM), YEAR_MAX_BOUND)
    return start > YEAR_MIN_BOUND or end < YEAR_MAX_BOUND


def _year_edge(value: object, bound: int) -> int:
    if value is None or isinstance(value, bool):
        return bound
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text or not text.lstrip("-").isdigit():
        return bound
    return int(text)


def _drop_query_params(query_params: MutableMapping[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in query_params:
            del query_params[key]
