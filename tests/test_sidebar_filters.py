"""Active sidebar filters, Clear all, and the empty author search message."""

from src.settings import YEAR_MAX_BOUND, YEAR_MIN_BOUND
from src.ui.sidebar_filters import (
    AUTHOR_INPUT_HINT,
    active_filter_count,
    author_filter_name,
    clear_all_filters,
    clear_author_filter,
    filters_heading,
    results_empty_message,
)

CATEGORIES_KEY = "categories_input"
AUTHORS_KEY = "authors_input"
YEAR_KEY = "year_range_input"
CUSTOM_KEY = "category_custom_input"
PENDING_KEY = "pending_categories"


def _empty_filters() -> dict:
    return {"categories": [], "authors": [], "year_min": None, "year_max": None}


def test_active_filter_count_covers_categories_published_range_and_author():
    assert active_filter_count(_empty_filters()) == 0
    assert filters_heading(0) == "Filters"

    categories = _empty_filters()
    categories["categories"] = ["cs.CL", "cs.IR"]
    assert active_filter_count(categories) == 1

    author = _empty_filters()
    author["authors"] = ["Yann LeCun"]
    assert active_filter_count(author) == 1
    blank_author = _empty_filters()
    blank_author["authors"] = ["  ", ""]
    assert active_filter_count(blank_author) == 0

    any_time = _empty_filters()
    any_time["year_min"] = YEAR_MIN_BOUND
    any_time["year_max"] = YEAR_MAX_BOUND
    assert active_filter_count(any_time) == 0

    from_year = _empty_filters()
    from_year["year_min"] = 2018
    assert active_filter_count(from_year) == 1
    until_year = _empty_filters()
    until_year["year_max"] = 2020
    assert active_filter_count(until_year) == 1

    all_three = {
        "categories": ["cs.LG"],
        "authors": ["LeCun"],
        "year_min": 2019,
        "year_max": 2024,
    }
    assert active_filter_count(all_three) == 3
    assert filters_heading(3) == "Filters (3)"
    assert filters_heading(active_filter_count(all_three)) == "Filters (3)"


def test_clear_all_resets_filter_state_and_matching_query_params():
    state = {
        CATEGORIES_KEY: ["cs.CL", "cs.IR"],
        AUTHORS_KEY: "LeCun",
        YEAR_KEY: (2018, 2024),
        CUSTOM_KEY: "stat.ML",
        PENDING_KEY: ["stat.ML"],
        "query_input": "attention",
        "top_k_input": 8,
        "sort_by_input": "Newest",
    }
    params = {
        "query": "attention",
        "top_k": "8",
        "categories": "cs.CL, cs.IR",
        "authors": "LeCun",
        "year_min": "2018",
        "year_max": "2024",
    }

    clear_all_filters(
        state,
        params,
        categories_key=CATEGORIES_KEY,
        authors_key=AUTHORS_KEY,
        year_key=YEAR_KEY,
        extra_blank_keys=(CUSTOM_KEY,),
        pop_keys=(PENDING_KEY,),
    )

    assert state[CATEGORIES_KEY] == []
    assert state[AUTHORS_KEY] == ""
    assert state[YEAR_KEY] == (YEAR_MIN_BOUND, YEAR_MAX_BOUND)
    assert state[CUSTOM_KEY] == ""
    assert PENDING_KEY not in state
    assert state["query_input"] == "attention"
    assert state["top_k_input"] == 8
    assert state["sort_by_input"] == "Newest"
    assert params == {"query": "attention", "top_k": "8"}


def test_zero_results_author_message_and_clear_author_only():
    filters = {
        "categories": ["cs.CL"],
        "authors": ["Yann LeCun"],
        "year_min": 2020,
        "year_max": None,
    }

    assert AUTHOR_INPUT_HINT == "Matches any author whose name contains this text."
    assert author_filter_name(filters) == "Yann LeCun"
    assert results_empty_message(0, filters) == (
        "No papers matched author 'Yann LeCun'. Try a last name only."
    )
    assert results_empty_message(0, {"authors": ["LeCun", "Hinton"], "categories": []}) == (
        "No papers matched author 'LeCun, Hinton'. Try a last name only."
    )
    assert results_empty_message(4, filters) is None
    assert results_empty_message(0, _empty_filters()) == "No papers matched this search."

    state = {
        AUTHORS_KEY: "Yann LeCun",
        CATEGORIES_KEY: ["cs.CL"],
        YEAR_KEY: (2020, 2024),
    }
    params = {
        "authors": "Yann LeCun",
        "categories": "cs.CL",
        "year_min": "2020",
        "query": "transformers",
    }

    clear_author_filter(state, params, authors_key=AUTHORS_KEY)

    assert state[AUTHORS_KEY] == ""
    assert state[CATEGORIES_KEY] == ["cs.CL"]
    assert state[YEAR_KEY] == (2020, 2024)
    assert "authors" not in params
    assert params["categories"] == "cs.CL"
    assert params["year_min"] == "2020"
    assert params["query"] == "transformers"
