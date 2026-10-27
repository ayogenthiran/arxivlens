"""The category dropdown lists every current arXiv category."""

from src.settings import COMMON_ARXIV_CATEGORIES


def test_category_dropdown_includes_the_full_arxiv_taxonomy():
    codes = COMMON_ARXIV_CATEGORIES

    assert len(codes) == len(set(codes)) == 155
    assert "cs.CR" in codes
    assert "math.PR" in codes
    assert "quant-ph" in codes
    assert codes.index("physics.space-ph") < codes.index("quant-ph") < codes.index("q-bio.BM")
