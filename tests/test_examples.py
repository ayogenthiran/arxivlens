"""Empty-state example chips and the corpus bar for those queries."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.settings import RERANKER_MIN_RAW_SCORE, YEAR_MAX_BOUND
from src.ui.examples import (
    EXAMPLE_PROMPT,
    EXAMPLE_QUESTIONS,
    EXAMPLE_RAW_SCORE_MARGIN,
    MIN_EXAMPLE_PAPERS,
    example_chip_label,
    example_result_problem,
    example_retrieval_filters,
)

ROOT = Path(__file__).resolve().parents[1]
YEAR_EXAMPLE = "Evaluating LLM reasoning without benchmark contamination"


def _param(params: dict, key: str) -> str | None:
    if key not in params:
        return None
    value = params[key]
    if isinstance(value, list):
        return str(value[-1]) if value else None
    return str(value)


def _markdown(at: AppTest) -> str:
    return "\n".join(str(block.value) for block in at.markdown)


def _fake_papers(query: str) -> dict:
    return {
        "context_chunks": [{"text": "A retrieved abstract.", "metadata": {"title": "A paper"}}],
        "facets": {},
        "query": query,
    }


def _install_fakes(monkeypatch):
    calls = []

    def fake_search(query, top_k, filters):
        calls.append({"query": query, "top_k": top_k, "filters": dict(filters)})
        return _fake_papers(query)

    def fake_answer(query, chunks):
        return {"answer": "ok", "summary": "ok", "points": [], "citations": []}

    def blocked(*args, **kwargs):
        raise AssertionError("example test tried to use the network")

    monkeypatch.setattr("src.ui.search_flow.search_papers", fake_search)
    monkeypatch.setattr("src.ui.search_flow.request_answer", fake_answer)
    monkeypatch.setattr("src.ui.search_flow.requests.post", blocked)
    return calls


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)


def test_example_presets_and_corpus_bar():
    assert [item["query"] for item in EXAMPLE_QUESTIONS] == [
        "Contrastive pretraining for low-resource machine translation",
        YEAR_EXAMPLE,
        "Retrieval-augmented generation for scientific question answering",
    ]
    assert EXAMPLE_QUESTIONS[0]["filters"] == {}
    assert EXAMPLE_QUESTIONS[1]["filters"] == {"published": "Since 2026"}
    assert EXAMPLE_QUESTIONS[2]["filters"] == {}
    assert example_chip_label(EXAMPLE_QUESTIONS[0]) == EXAMPLE_QUESTIONS[0]["label"]
    assert example_chip_label(EXAMPLE_QUESTIONS[1]).replace("*", "") == f"{YEAR_EXAMPLE} · since 2026"
    preset = example_retrieval_filters(EXAMPLE_QUESTIONS[1]["filters"])
    assert preset["year_min"] == 2026
    assert preset["year_max"] is None
    assert preset["categories"] == []
    assert preset["authors"] == []
    assert example_retrieval_filters({}) == {
        "categories": [],
        "authors": [],
        "year_min": None,
        "year_max": None,
    }

    cutoff = RERANKER_MIN_RAW_SCORE + EXAMPLE_RAW_SCORE_MARGIN
    assert example_result_problem(MIN_EXAMPLE_PAPERS, cutoff, RERANKER_MIN_RAW_SCORE) is None
    assert example_result_problem(MIN_EXAMPLE_PAPERS - 1, cutoff, RERANKER_MIN_RAW_SCORE) is not None
    assert example_result_problem(MIN_EXAMPLE_PAPERS, cutoff - 0.01, RERANKER_MIN_RAW_SCORE) is not None
    assert example_result_problem(MIN_EXAMPLE_PAPERS, None, RERANKER_MIN_RAW_SCORE) is not None


def test_example_chip_click_sets_query_and_filters(monkeypatch):
    calls = _install_fakes(monkeypatch)
    at = _app()
    at.session_state["categories_input"] = ["cs.CL"]
    at.session_state["authors_input"] = "LeCun"
    at.session_state["year_range_input"] = (2023, 2023)
    at.run()

    assert not at.exception
    assert EXAMPLE_PROMPT in _markdown(at)
    at.button("example-1").click().run()

    assert not at.exception
    assert at.text_area[0].value == YEAR_EXAMPLE
    assert at.session_state["categories_input"] == []
    assert at.session_state["authors_input"] == ""
    year = at.sidebar.slider[0].value
    assert int(year[0]) == 2026
    assert int(year[1]) == YEAR_MAX_BOUND
    assert _param(at.query_params, "query") == YEAR_EXAMPLE
    assert _param(at.query_params, "year_min") == "2026"
    assert _param(at.query_params, "year_max") is None
    assert _param(at.query_params, "categories") is None
    assert _param(at.query_params, "authors") is None
    assert calls[0]["query"] == YEAR_EXAMPLE
    assert calls[0]["filters"]["year_min"] == 2026
    assert calls[0]["filters"]["year_max"] is None
    assert calls[0]["filters"]["categories"] == []
    assert calls[0]["filters"]["authors"] == []


def test_examples_hidden_after_results(monkeypatch):
    _install_fakes(monkeypatch)
    at = _app()
    at.run()

    assert not at.exception
    assert at.button("example-0").label == example_chip_label(EXAMPLE_QUESTIONS[0])
    assert EXAMPLE_PROMPT in _markdown(at)

    at.button("example-0").click().run()

    assert not at.exception
    keys = [button.key for button in at.button]
    assert "example-0" not in keys
    assert "example-1" not in keys
    assert "example-2" not in keys
    assert EXAMPLE_PROMPT not in _markdown(at)
    assert at.text_area[0].value == EXAMPLE_QUESTIONS[0]["query"]
