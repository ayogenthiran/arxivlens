import os
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
QUERY_URL = f"{API_BASE_URL}/api/query"
DEFAULT_TOP_K = int(os.getenv("UI_DEFAULT_TOP_K", "8"))
MIN_TOP_K = int(os.getenv("UI_MIN_TOP_K", "3"))
MAX_TOP_K = int(os.getenv("UI_MAX_TOP_K", "20"))

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
.source-meta {
    margin-top: 0.25rem;
    color: #94a3b8;
    font-size: 0.9rem;
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


def _parse_year_input(raw_value: str) -> int | None:
    value = raw_value.strip()
    if not value:
        return None
    if not value.isdigit():
        return None
    year = int(value)
    if 1900 <= year <= 2100:
        return year
    return None


def query_rag_api(query: str, top_k: int, filters: dict) -> dict:
    payload = {"query": query, "top_k": top_k, **filters}
    response = requests.post(
        QUERY_URL,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def render_header() -> None:
    st.markdown(
        """
        <header class="app-header">
            <h1>Arxix Lens</h1>
            <p>Ask questions about scientific papers from ArXiv</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_query_form() -> tuple[str, int, dict, bool]:
    with st.form("query-form", clear_on_submit=False):
        st.markdown('<div class="query-label">Ask a question</div>', unsafe_allow_html=True)
        query = st.text_input(
            "Enter your question...",
            placeholder="Enter your question...",
            label_visibility="collapsed",
        )
        top_k = st.slider("Sources to retrieve", min_value=MIN_TOP_K, max_value=MAX_TOP_K, value=DEFAULT_TOP_K)
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
        ask_clicked = st.form_submit_button("Search")
    year_min = _parse_year_input(year_min_raw)
    year_max = _parse_year_input(year_max_raw)
    filters = {
        "categories": [value.strip() for value in categories_raw.split(",") if value.strip()],
        "authors": [value.strip() for value in authors_raw.split(",") if value.strip()],
        "year_min": year_min,
        "year_max": year_max,
    }
    return query, top_k, filters, ask_clicked


def _chunk_confidence(chunk: dict) -> tuple[str, str]:
    score = chunk.get("hybrid_score")
    if score is None:
        score = chunk.get("vector_score")
    if score is None:
        return "N/A", "Unavailable"

    score = float(score)
    score = min(max(score, 0.0), 1.0)
    if score >= 0.72:
        band = "High"
    elif score >= 0.45:
        band = "Medium"
    else:
        band = "Low"
    return f"{score:.2f}", band


def render_result(result: dict) -> None:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown("<h2>Answer</h2>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{result.get("answer", "No answer returned.")}</div>',
        unsafe_allow_html=True,
    )
    rewritten_query = result.get("rewritten_query")
    if rewritten_query:
        st.markdown(
            f'<div class="source-meta"><strong>Retrieval query:</strong> {rewritten_query}</div>',
            unsafe_allow_html=True,
        )

    facets = result.get("facets", {})
    if facets:
        st.markdown("<h2>Facets</h2>", unsafe_allow_html=True)
        facet_lines = []
        top_years = facets.get("years", [])[:5]
        top_categories = facets.get("categories", [])[:5]
        top_authors = facets.get("authors", [])[:5]
        if top_years:
            facet_lines.append(
                "Years: " + ", ".join(f"{item['value']} ({item['count']})" for item in top_years)
            )
        if top_categories:
            facet_lines.append(
                "Categories: " + ", ".join(f"{item['value']} ({item['count']})" for item in top_categories)
            )
        if top_authors:
            facet_lines.append(
                "Authors: " + ", ".join(f"{item['value']} ({item['count']})" for item in top_authors)
            )
        for line in facet_lines:
            st.markdown(f'<div class="source-meta">{line}</div>', unsafe_allow_html=True)

    st.markdown("<h2>Sources</h2>", unsafe_allow_html=True)
    for chunk in result.get("context_chunks", []):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown")
        year = metadata.get("year")
        categories = ", ".join(metadata.get("categories", []))
        authors = ", ".join(metadata.get("authors", [])[:3])
        text = chunk.get("text", "")
        score_value, score_band = _chunk_confidence(chunk)
        metadata_segments = [f"Source: {source}"]
        if year:
            metadata_segments.append(f"Year: {year}")
        if categories:
            metadata_segments.append(f"Categories: {categories}")
        if authors:
            metadata_segments.append(f"Authors: {authors}")
        st.markdown(
            (
                '<div class="source-card">'
                f"<h3>{title}</h3>"
                f'<div class="source-meta">{" | ".join(metadata_segments)}</div>'
                f'<div class="source-score">Relevance: {score_value} ({score_band})</div>'
                f'<div class="chunk-text">{text}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        '<footer class="app-footer">Arxix Lens &copy; 2025</footer>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="ArxivLens", page_icon=":books:", layout="centered")
    inject_styles()
    render_header()

    result = None
    error = None

    query, top_k, filters, ask_clicked = render_query_form()

    if ask_clicked:
        if not query.strip():
            error = "Please enter a question first."
        elif (
            filters.get("year_min") is not None
            and filters.get("year_max") is not None
            and filters["year_min"] > filters["year_max"]
        ):
            error = "Year from must be less than or equal to Year to."
        else:
            try:
                with st.spinner("Searching..."):
                    result = query_rag_api(query.strip(), top_k, filters)
            except requests.exceptions.RequestException as exc:
                error = (
                    "Could not reach backend API. Make sure `python main.py` is running "
                    f"on `{API_BASE_URL}`. Details: {exc}"
                )

    if error:
        st.markdown(f'<div class="error-box">Error: {error}</div>', unsafe_allow_html=True)

    if result:
        render_result(result)
    render_footer()


if __name__ == "__main__":
    main()
