import html
from datetime import datetime

import requests
import streamlit as st

from src.settings import (
    API_BASE_URL,
    API_QUERY_TIMEOUT_S,
    ARXIV_ABS_BASE_URL,
    DEFAULT_TOP_K,
    UI_CONFIDENCE_HIGH,
    UI_CONFIDENCE_MEDIUM,
    UI_FACET_LIMIT,
    UI_MAX_AUTHORS_SHOWN,
    UI_MAX_TOP_K,
    UI_MIN_TOP_K,
    YEAR_MAX_BOUND,
    YEAR_MIN_BOUND,
)

QUERY_URL = f"{API_BASE_URL}/api/query"

GLOBAL_STYLES = """
<style>
.stApp {
    background: #0f172a;
    color: #e2e8f0;
}
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 1.5rem;
}
.app-header h1 {
    margin-bottom: 0.3rem;
    font-size: 2.2rem;
    color: #f8fafc;
}
.app-header p {
    margin-top: 0;
    margin-bottom: 1.6rem;
    color: #94a3b8;
}
.query-label {
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.query-hint {
    margin-top: 0.35rem;
    color: #94a3b8;
    font-size: 0.88rem;
}
.stTextInput input {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}
.stButton > button {
    background: #2563eb !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
}
.error-box {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.6);
    color: #fecaca;
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    margin-top: 1rem;
}
.result-container {
    margin-top: 1.4rem;
}
.result-container h2 {
    margin-bottom: 0.4rem;
    color: #f8fafc;
}
.result-container h3 {
    margin: 0;
    color: #e2e8f0;
    font-size: 1.05rem;
}
.answer-box, .source-card {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.trust-note {
    color: #cbd5e1;
    font-size: 0.86rem;
    margin-top: 0.45rem;
}
.source-meta {
    margin-top: 0.25rem;
    color: #94a3b8;
    font-size: 0.9rem;
}
.source-card a {
    color: #93c5fd;
}
.source-score {
    margin-top: 0.35rem;
    color: #bfdbfe;
    font-size: 0.86rem;
}
.chunk-text {
    margin-top: 0.6rem;
    color: #cbd5e1;
}
.app-footer {
    margin-top: 1.25rem;
    padding-top: 0.75rem;
    border-top: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.9rem;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def _html(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def _arxiv_url(paper_id: object) -> str | None:
    paper_id = str(paper_id or "").strip()
    if not paper_id:
        return None
    return f"{ARXIV_ABS_BASE_URL}/{paper_id}"


def _parse_year_input(raw_value: str) -> tuple[int | None, str | None]:
    value = raw_value.strip()
    if not value:
        return None, None
    if not value.isdigit():
        return None, "Year must be a number."
    year = int(value)
    if YEAR_MIN_BOUND <= year <= YEAR_MAX_BOUND:
        return year, None
    return None, f"Year must be between {YEAR_MIN_BOUND} and {YEAR_MAX_BOUND}."


def query_rag_api(query: str, top_k: int, filters: dict) -> dict:
    payload = {"query": query, "top_k": top_k, **filters}
    response = requests.post(
        QUERY_URL,
        json=payload,
        timeout=API_QUERY_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()


def render_header() -> None:
    st.markdown(
        """
        <header class="app-header">
            <h1>ArxivLens</h1>
            <p>Ask questions about scientific papers from ArXiv</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_query_form() -> tuple[str, int, dict, bool, str | None]:
    with st.form("query-form", clear_on_submit=False):
        st.markdown('<div class="query-label">Ask a question</div>', unsafe_allow_html=True)
        query = st.text_area(
            "What do you want to learn from arXiv papers?",
            placeholder="Enter your question...",
            label_visibility="collapsed",
            height=110,
        )
        st.markdown(
            '<div class="query-hint">Tip: Ask specific questions to get better cited answers.</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Advanced retrieval settings", expanded=False):
            top_k = st.slider(
                "Sources to retrieve (higher may improve coverage but can be slower)",
                min_value=UI_MIN_TOP_K,
                max_value=UI_MAX_TOP_K,
                value=DEFAULT_TOP_K,
            )
            categories_raw = st.text_input(
                "Categories (comma-separated, e.g. cs.CL, cs.AI)",
                placeholder="cs.CL, cs.AI",
            )
            authors_raw = st.text_input(
                "Author keywords (comma-separated, optional)",
                placeholder="Yann LeCun, Geoffrey Hinton",
            )
            year_col_min, year_col_max = st.columns(2)
            with year_col_min:
                year_min_raw = st.text_input("Year from (optional)", placeholder="2019")
            with year_col_max:
                year_max_raw = st.text_input("Year to (optional)", placeholder="2024")
        ask_clicked = st.form_submit_button("Ask")
    year_min, year_min_error = _parse_year_input(year_min_raw)
    year_max, year_max_error = _parse_year_input(year_max_raw)
    filters = {
        "categories": [value.strip() for value in categories_raw.split(",") if value.strip()],
        "authors": [value.strip() for value in authors_raw.split(",") if value.strip()],
        "year_min": year_min,
        "year_max": year_max,
    }
    return query, top_k, filters, ask_clicked, year_min_error or year_max_error


def _chunk_confidence(chunk: dict) -> tuple[str, str]:
    score = chunk.get("reranker_score")
    if score is None:
        score = chunk.get("vector_score")
    if score is None:
        return "N/A", "Unavailable"

    score = float(score)
    score = min(max(score, 0.0), 1.0)
    if score >= UI_CONFIDENCE_HIGH:
        band = "High"
    elif score >= UI_CONFIDENCE_MEDIUM:
        band = "Medium"
    else:
        band = "Low"
    return f"{score:.2f}", band


def render_result(result: dict) -> None:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown("<h2>Answer</h2>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{_html(result.get("answer", "No answer returned."))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="trust-note">AI-generated synthesis. Verify details with linked sources.</div>',
        unsafe_allow_html=True,
    )
    rewritten_query = str(result.get("rewritten_query") or "").strip()
    original_query = str(result.get("query") or "").strip()
    if rewritten_query and rewritten_query != original_query:
        st.markdown(
            f'<div class="source-meta"><strong>Retrieval query:</strong> {_html(rewritten_query)}</div>',
            unsafe_allow_html=True,
        )
    citations = result.get("citations", [])
    if citations:
        st.markdown("<h2>Citations</h2>", unsafe_allow_html=True)
        for item in citations:
            quote = str(item.get("quote") or "").strip()
            if not quote:
                continue
            arxiv_id = str(item.get("arxiv_id") or "").strip()
            title = str(item.get("title") or "").strip() or arxiv_id or "Untitled"
            arxiv_url = _arxiv_url(arxiv_id)
            link_html = (
                f'<div class="source-meta"><a href="{html.escape(arxiv_url, quote=True)}" target="_blank" rel="noopener noreferrer">Open on arXiv ({_html(arxiv_id)})</a></div>'
                if arxiv_url
                else ""
            )
            st.markdown(
                (
                    '<div class="source-card">'
                    f"<h3>{_html(title)}</h3>"
                    f"{link_html}"
                    f'<div class="chunk-text">"{_html(quote)}"</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    facets = result.get("facets", {})
    if facets:
        st.markdown("<h2>Facets</h2>", unsafe_allow_html=True)
        facet_lines = []
        top_years = facets.get("years", [])[:UI_FACET_LIMIT]
        top_categories = facets.get("categories", [])[:UI_FACET_LIMIT]
        top_authors = facets.get("authors", [])[:UI_FACET_LIMIT]
        if top_years:
            facet_lines.append(
                "Years: " + ", ".join(
                    f"{_html(item['value'])} ({_html(item['count'])})" for item in top_years
                )
            )
        if top_categories:
            facet_lines.append(
                "Categories: " + ", ".join(
                    f"{_html(item['value'])} ({_html(item['count'])})" for item in top_categories
                )
            )
        if top_authors:
            facet_lines.append(
                "Authors: " + ", ".join(
                    f"{_html(item['value'])} ({_html(item['count'])})" for item in top_authors
                )
            )
        for line in facet_lines:
            st.markdown(f'<div class="source-meta">{line}</div>', unsafe_allow_html=True)

    st.markdown("<h2>Sources</h2>", unsafe_allow_html=True)
    for chunk in result.get("context_chunks", []):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown")
        paper_id = metadata.get("doc_id") or metadata.get("id") or metadata.get("paper_id")
        arxiv_url = _arxiv_url(paper_id)
        year = metadata.get("year")
        categories = metadata.get("categories") or []
        authors = metadata.get("authors") or []
        if isinstance(categories, str):
            categories = [categories]
        if isinstance(authors, str):
            authors = [authors]
        category_text = ", ".join(str(category) for category in categories)
        author_text = ", ".join(str(author) for author in list(authors)[:UI_MAX_AUTHORS_SHOWN])
        text = chunk.get("text", "")
        score_value, score_band = _chunk_confidence(chunk)
        metadata_segments = [f"Source: {_html(source)}"]
        if year:
            metadata_segments.append(f"Year: {_html(year)}")
        if category_text:
            metadata_segments.append(f"Categories: {_html(category_text)}")
        if author_text:
            metadata_segments.append(f"Authors: {_html(author_text)}")
        source_link_html = (
            f'<div class="source-meta"><a href="{html.escape(arxiv_url, quote=True)}" target="_blank" rel="noopener noreferrer">Open on arXiv</a></div>'
            if arxiv_url
            else ""
        )
        st.markdown(
            (
                '<div class="source-card">'
                f"<h3>{_html(title)}</h3>"
                f"{source_link_html}"
                f'<div class="source-meta">{" | ".join(metadata_segments)}</div>'
                f'<div class="source-score">Relevance: {_html(score_value)} ({_html(score_band)})</div>'
                f'<div class="chunk-text">{_html(text)}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        f'<footer class="app-footer">ArxivLens &copy; {datetime.now().year}</footer>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="ArxivLens", page_icon=":books:", layout="centered")
    inject_styles()
    render_header()

    result = None
    error = None

    query, top_k, filters, ask_clicked, year_error = render_query_form()

    if ask_clicked:
        if not query.strip():
            error = "Please enter a question first."
        elif year_error:
            error = year_error
        elif (
            filters.get("year_min") is not None
            and filters.get("year_max") is not None
            and filters["year_min"] > filters["year_max"]
        ):
            error = "Year from must be less than or equal to Year to."
        else:
            try:
                with st.status("Running pipeline...", expanded=True) as status:
                    st.write("Retrieving papers...")
                    st.write("Reranking sources...")
                    st.write("Generating answer...")
                    result = query_rag_api(query.strip(), top_k, filters)
                    status.update(label="Completed", state="complete")
            except requests.exceptions.RequestException as exc:
                error = (
                    "Could not reach backend API. Make sure `python main.py` is running "
                    f"on `{API_BASE_URL}`. Details: {exc}"
                )

    if error:
        st.markdown(f'<div class="error-box">Error: {_html(error)}</div>', unsafe_allow_html=True)

    if result:
        render_result(result)
    render_footer()


if __name__ == "__main__":
    main()
