"""The year slider runs from 2023 through 2026."""

from src.settings import YEAR_MAX_BOUND, YEAR_MIN_BOUND


def test_year_slider_runs_from_2023_through_2026():
    assert YEAR_MIN_BOUND == 2023
    assert YEAR_MAX_BOUND == 2026
