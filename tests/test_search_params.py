import re
import tomllib
from pathlib import Path

from src.settings import DEFAULT_TOP_K, UI_MAX_TOP_K, UI_MIN_TOP_K, YEAR_MAX_BOUND, YEAR_MIN_BOUND
from src.ui.search_params import (
    decode_search_params,
    encode_search_params,
    present_search_params,
)

ROOT = Path(__file__).resolve().parents[1]


class _RepeatedParams:
    def __init__(self, **kwargs):
        self._data = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                self._data[key] = [str(item) for item in value]
            else:
                self._data[key] = [str(value)]

    def get_all(self, key):
        return list(self._data.get(key, []))


def test_round_trip_keeps_query_top_k_and_filters():
    filters = {
        "categories": ["cs.CL", "cs.AI"],
        "authors": ["Yann LeCun"],
        "year_min": 2019,
        "year_max": 2024,
    }
    encoded = encode_search_params("  reasoning in language models  ", 8, filters)
    query, top_k, decoded = decode_search_params(encoded)

    assert encoded["query"] == "reasoning in language models"
    assert encoded["top_k"] == "8"
    assert encoded["categories"] == "cs.CL, cs.AI"
    assert query == "reasoning in language models"
    assert top_k == 8
    assert decoded == filters


def test_missing_optionals_use_defaults():
    query, top_k, filters = decode_search_params({"query": "rag"})

    assert query == "rag"
    assert top_k == DEFAULT_TOP_K
    assert filters == {
        "categories": [],
        "authors": [],
        "year_min": None,
        "year_max": None,
    }


def test_invalid_top_k_and_year_are_dropped_or_clamped():
    query, top_k, filters = decode_search_params(
        {
            "query": "rag",
            "top_k": "nope",
            "year_min": "12",
            "year_max": "2024",
        }
    )

    assert query == "rag"
    assert top_k == DEFAULT_TOP_K
    assert filters["year_min"] is None
    assert filters["year_max"] == 2024

    _, clamped_high, _ = decode_search_params({"query": "x", "top_k": "999"})
    _, clamped_low, _ = decode_search_params({"query": "x", "top_k": "1"})
    assert clamped_high == UI_MAX_TOP_K
    assert clamped_low == UI_MIN_TOP_K

    _, _, out_of_range = decode_search_params(
        {"year_min": str(YEAR_MIN_BOUND - 1), "year_max": str(YEAR_MAX_BOUND + 1)}
    )
    assert out_of_range["year_min"] is None
    assert out_of_range["year_max"] is None


def test_repeated_and_comma_separated_filters():
    params = _RepeatedParams(
        query="topic",
        top_k="5",
        categories=["cs.CL", "cs.LG"],
        authors="LeCun, Hinton",
    )
    query, top_k, filters = decode_search_params(params)

    assert query == "topic"
    assert top_k == 5
    assert filters["categories"] == ["cs.CL", "cs.LG"]
    assert filters["authors"] == ["LeCun", "Hinton"]


def test_encode_omits_empty_filters_and_blank_query():
    encoded = encode_search_params(
        "   ",
        DEFAULT_TOP_K,
        {"categories": [], "authors": ["  "], "year_min": None, "year_max": None},
    )

    assert "query" not in encoded
    assert "categories" not in encoded
    assert "authors" not in encoded
    assert "year_min" not in encoded
    assert encoded["top_k"] == str(DEFAULT_TOP_K)


def test_present_params_match_encoded_search_after_spacing_normalization():
    encoded = encode_search_params(
        "topic",
        5,
        {"categories": ["cs.CL", "cs.AI"], "authors": [], "year_min": 2020, "year_max": None},
    )
    current = {
        "query": "topic",
        "top_k": "5",
        "categories": "cs.CL,cs.AI",
        "year_min": "2020",
    }

    assert present_search_params(current) == encoded
    assert "top_k" not in present_search_params({"query": "topic"})


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]

    def linearize(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearize(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(_relative_luminance(foreground), _relative_luminance(background))
    darker = min(_relative_luminance(foreground), _relative_luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def test_placeholder_meets_wcag_aa_on_input_background():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    theme = config["theme"]
    source = (ROOT / "streamlit_app.py").read_text()
    match = re.search(
        r"::placeholder\s*\{[^}]*color:\s*(#[0-9a-fA-F]{6})",
        source,
    )

    assert theme["base"] == "dark"
    assert theme["backgroundColor"].lower() == "#0f172a"
    assert theme["primaryColor"]
    assert theme["textColor"]
    assert theme["font"]
    assert config["client"]["toolbarMode"] == "minimal"
    assert match
    assert "opacity: 1" in source.split("::placeholder", 1)[1].split("}", 1)[0]
    assert ".stApp {" not in source
    ratio = _contrast_ratio(match.group(1), theme["secondaryBackgroundColor"])
    assert ratio >= 4.5
