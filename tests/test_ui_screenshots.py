"""The screenshot script serves the search fixture, and the UI keeps the mobile layout readable."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _screenshots():
    path = ROOT / "scripts" / "ui_screenshots.py"
    spec = importlib.util.spec_from_file_location("ui_screenshots", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mock_search_returns_fixture_fields_and_the_posted_query():
    module = _screenshots()
    fixture = module.load_fixture(module.FIXTURE_PATH)

    search = module.search_body(fixture, {"query": "typed question", "top_k": 3})
    answer = module.answer_body(fixture)

    assert module.WIDTHS == (1280, 390)
    assert module.OUTPUT_DIR == ROOT / "artifacts" / "ui"
    assert search["query"] == "typed question"
    assert set(search) == {"context_chunks", "facets", "query"}
    assert search["context_chunks"][0]["metadata"]["doc_id"] == "2401.12345"
    assert set(answer) == {"answer", "summary", "points", "citations"}
    assert answer["points"][0]["sources"][0]["arxiv_id"] == "2401.12345"
    assert module.search_body(fixture, {})["query"] == fixture["query"]


def test_screenshot_output_is_gitignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "/artifacts/" in ignored


def test_results_page_clamps_titles_and_keeps_tap_targets():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    mobile = source.split("@media (max-width: 640px)", 1)[1]

    assert "-webkit-line-clamp: 3 !important" in source
    assert "max-height: calc(1.35em * 3) !important" in source
    assert "min-height: 40px" in source
    assert 'button[data-testid="stBaseButton-headerNoPadding"]' in source
    assert "a.source-chip, span.source-chip" in source
    assert ".passage summary" in source
    assert "max-width: min(280px, 100%)" in source
    slider_thumb = source.split('div[data-testid="stSlider"] [role="slider"]', 1)[1].split("}", 1)[0]
    assert "width: 16px !important" in slider_thumb
    assert "height: 16px !important" in slider_thumb
    assert 'div[data-testid="stSliderThumbValue"]' in source
    assert "background: #0f172a !important" in source
    assert "min-width: 0" in mobile
    assert "display: none" in mobile
    label_rule = source.split('label[data-testid="stWidgetLabel"]', 1)[1].split("}", 1)[0]
    assert "display:" not in label_rule
