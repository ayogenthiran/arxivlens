import base64
import html
from datetime import datetime

import requests
import streamlit as st

from src.settings import (
    API_BASE_URL,
    API_QUERY_TIMEOUT_S,
    ARXIV_ABS_BASE_URL,
    ARXIV_HOME_URL,
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
from src.ui.answer_display import render_answer_html
from src.ui.paper_visuals import (
    arxiv_logo_path,
    category_color,
    clean_arxiv_id,
    load_figure_urls,
    paper_media_html,
    passage_body,
    passage_preview,
    primary_category,
    uncached_figure_ids,
)

QUERY_URL = f"{API_BASE_URL}/api/query"
QUERY_INPUT_KEY = "query_input"
RESULT_KEY = "query_result"
ERROR_KEY = "query_error"
EXAMPLE_QUESTIONS = (
    "How do recent papers evaluate reasoning in large language models?",
    "What techniques reduce hallucination in language models?",
    "How do retrieval systems select evidence for an answer?",
)
BAND_COLORS = {
    "High": "#34d399",
    "Medium": "#fbbf24",
    "Low": "#f87171",
}

GLOBAL_STYLES = """
<style>
.stApp {
    background: #0f172a;
    color: #e2e8f0;
}
.block-container {
    max-width: 960px;
    padding-top: 2rem;
    padding-bottom: 1.5rem;
}
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    min-width: 0;
}
.brand-copy {
    min-width: 0;
}
.lens-mark {
    flex: 0 0 auto;
    line-height: 0;
}
.brand-copy h1 {
    margin: 0 0 0.2rem 0;
    font-size: 2.1rem;
    color: #f8fafc;
    line-height: 1.1;
}
.brand-copy p {
    margin: 0;
    color: #94a3b8;
}
.arxiv-mark {
    flex: 0 0 auto;
    line-height: 0;
    padding: 0.4rem;
}
.arxiv-mark img {
    height: 42px;
    width: auto;
    display: block;
}
.examples h2 {
    margin: 0.4rem 0 0.25rem 0;
    color: #f8fafc;
    font-size: 1.15rem;
}
.examples p {
    margin: 0 0 0.7rem 0;
    color: #94a3b8;
    font-size: 0.9rem;
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
.stTextInput input, .stTextArea textarea {
    background: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
}
button[kind="secondaryFormSubmit"],
button[kind="primary"] {
    background: #2563eb !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
}
button[kind="secondaryFormSubmit"]:hover,
button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    color: #f8fafc !important;
}
button[kind="secondary"] {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 0.8rem 0.95rem !important;
    margin-bottom: 0.15rem !important;
}
button[kind="secondary"] p {
    text-align: left !important;
    width: 100%;
}
button[kind="secondary"]:hover {
    background: #1e293b !important;
    border-color: #60a5fa !important;
    color: #f8fafc !important;
}
h1 [data-testid="stHeaderActionElements"] {
    display: none;
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
.answer-box, .source-card, .paper-card {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.answer-lead {
    margin: 0;
    color: #e2e8f0;
    line-height: 1.5;
}
ol.answer-points {
    margin: 0.85rem 0 0 0;
    padding: 0;
    list-style: none;
}
ol.answer-points li {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.75rem 0 0.15rem 0;
    border-top: 1px solid #1e293b;
}
.point-index {
    flex: 0 0 1.6rem;
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 999px;
    background: #1e293b;
    color: #93c5fd;
    font-size: 0.82rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
.point-copy {
    min-width: 0;
    flex: 1;
}
.point-title {
    color: #f8fafc;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.point-detail {
    color: #cbd5e1;
    line-height: 1.45;
}
.point-sources {
    margin-top: 0.45rem;
}
a.source-chip, span.source-chip {
    display: inline-block;
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: middle;
    margin: 0 0.35rem 0.35rem 0;
    padding: 0.16rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #334155;
    background: #1e293b;
    color: #93c5fd;
    font-size: 0.8rem;
    text-decoration: none;
}
a.source-chip:hover {
    border-color: #60a5fa;
    color: #bfdbfe;
}
.paper-card {
    display: flex;
    gap: 0.9rem;
    align-items: stretch;
    position: relative;
    overflow: hidden;
    border-left: 3px solid var(--band, #334155);
}
.paper-body {
    min-width: 0;
    flex: 1;
}
.paper-body h3 {
    margin-top: 0;
}
a.paper-media, .paper-media {
    flex: 0 0 132px;
    width: 132px;
    height: 156px;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    display: block;
    text-decoration: none;
    background: #0b1220;
}
.paper-media img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #f8fafc;
    display: block;
}
.paper-card:has(.has-figure) {
    flex-direction: column;
}
a.paper-media.has-figure, .paper-media.has-figure {
    flex: none;
    width: 100%;
    height: auto;
    overflow: visible;
    background: transparent;
}
.paper-media.has-figure img {
    width: 100%;
    height: 320px;
    border-radius: 8px;
    object-fit: contain;
    object-position: center top;
}
.media-caption {
    display: block;
    margin-top: 0.35rem;
    color: #94a3b8;
    font-size: 0.75rem;
    letter-spacing: 0.01em;
}
.passage-block:has(details[open]) .chunk-preview {
    display: none;
}
.passage {
    margin-top: 0.45rem;
}
.passage summary {
    cursor: pointer;
    color: #93c5fd;
    font-size: 0.86rem;
}
.passage .chunk-text {
    margin-top: 0.45rem;
}
.paper-cover {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    padding: 0.55rem 0.5rem;
    background: linear-gradient(165deg, var(--cover, #334155), #0f172a 78%);
    color: #f8fafc;
}
.cover-cat {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.cover-year {
    color: rgba(248, 250, 252, 0.82);
    font-size: 0.75rem;
}
.cover-id {
    font-size: 0.68rem;
    line-height: 1.25;
    color: rgba(248, 250, 252, 0.9);
    word-break: break-all;
}
.relevance-track {
    margin-top: 0.45rem;
    height: 4px;
    border-radius: 999px;
    background: #1e293b;
    overflow: hidden;
}
.relevance-fill {
    height: 100%;
    border-radius: 999px;
}
.facet-row {
    margin: 0.35rem 0 0.55rem 0;
}
.facet-label {
    display: block;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-bottom: 0.3rem;
}
.facet-chip {
    display: inline-block;
    margin: 0 0.35rem 0.35rem 0;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    border: 1px solid #334155;
    background: #1e293b;
    color: #e2e8f0;
    font-size: 0.82rem;
}
.chip-count {
    color: #94a3b8;
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
.ack {
    margin-top: 0.45rem;
    color: #cbd5e1;
    font-size: 0.82rem;
    line-height: 1.45;
}
@media (max-width: 640px) {
    .app-header {
        flex-wrap: nowrap;
        gap: 0.65rem;
    }
    .brand-copy h1 {
        font-size: 1.35rem;
    }
    .brand-copy p {
        font-size: 0.75rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .arxiv-mark {
        padding: 0.15rem;
    }
    .arxiv-mark img {
        height: 28px;
    }
    .lens-mark svg {
        width: 32px;
        height: 32px;
    }
    .paper-card:not(:has(.has-figure)) a.paper-media,
    .paper-card:not(:has(.has-figure)) .paper-media {
        flex-basis: 84px;
        width: 84px;
        height: 112px;
    }
    .paper-media.has-figure img {
        height: 200px;
    }
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


def _arxiv_logo_src() -> str:
    """Official single-color white arXiv logo, unchanged, for the dark header."""
    path = arxiv_logo_path()
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def render_header() -> None:
    logo_src = _arxiv_logo_src()
    if logo_src:
        logo_html = (
            f'<a class="arxiv-mark" href="{ARXIV_HOME_URL}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{logo_src}" alt="arXiv">'
            "</a>"
        )
    else:
        logo_html = (
            f'<a class="arxiv-mark" href="{ARXIV_HOME_URL}" target="_blank" rel="noopener noreferrer">arXiv</a>'
        )
    st.markdown(
        f"""
        <header class="app-header">
            <div class="brand-row">
                <div class="lens-mark" aria-hidden="true">
                    <svg viewBox="0 0 48 48" width="46" height="46">
                        <rect x="7" y="8" width="20" height="26" rx="2" fill="none" stroke="#93c5fd" stroke-width="1.6"/>
                        <path d="M11 15h12M11 19h10M11 23h8" fill="none" stroke="#94a3b8" stroke-width="1.4" stroke-linecap="round"/>
                        <circle cx="31" cy="29" r="8" fill="none" stroke="#60a5fa" stroke-width="2"/>
                        <path d="M36.5 34.5L42 40" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="brand-copy">
                    <h1>ArxivLens</h1>
                    <p>Ask questions about scientific papers from arXiv</p>
                </div>
            </div>
            {logo_html}
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_query_form() -> tuple[str, int, dict, bool, str | None]:
    pending_query = st.session_state.pop("pending_query", None)
    if pending_query is not None:
        st.session_state[QUERY_INPUT_KEY] = pending_query
    with st.form("query-form", clear_on_submit=False):
        st.markdown('<div class="query-label">Ask a question</div>', unsafe_allow_html=True)
        query = st.text_area(
            "What do you want to learn from arXiv papers?",
            placeholder="Enter your question...",
            label_visibility="collapsed",
            height=110,
            key=QUERY_INPUT_KEY,
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


def _as_list(value: object) -> list:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return value
    return []


def _paper_id_from_metadata(metadata: dict) -> str:
    return str(metadata.get("doc_id") or metadata.get("id") or metadata.get("paper_id") or "").strip()


def _metadata_by_paper(chunks: list) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        paper_id = _paper_id_from_metadata(metadata)
        if paper_id and paper_id not in found:
            found[paper_id] = metadata
    return found


def _relevance_html(score_value: str, score_band: str) -> str:
    color = BAND_COLORS.get(score_band)
    if color is None:
        return f'<div class="source-score">Relevance: {_html(score_value)}</div>'
    try:
        width = max(0.0, min(float(score_value), 1.0)) * 100
    except ValueError:
        width = 0.0
    return (
        '<div class="relevance">'
        f'<div class="relevance-track"><div class="relevance-fill" style="width:{width:.0f}%;background:{color}"></div></div>'
        f'<div class="source-score">Relevance {_html(score_value)} · {_html(score_band)}</div>'
        "</div>"
    )


def _passage_html(text: str, title: str) -> str:
    preview, truncated = passage_preview(text, title)
    if not preview:
        return ""
    if not truncated:
        return f'<div class="chunk-text">{_html(preview)}</div>'
    full = passage_body(text, title)
    return (
        '<div class="passage-block">'
        f'<div class="chunk-text chunk-preview">{_html(preview)}</div>'
        '<details class="passage">'
        "<summary>Show passage</summary>"
        f'<div class="chunk-text">{_html(full)}</div>'
        "</details>"
        "</div>"
    )


def _paper_card_html(
    *,
    title: str,
    paper_id: str,
    metadata: dict,
    figure_urls: dict[str, str | None],
    body_html: str,
    score_band: str | None = None,
    include_figure: bool = True,
) -> str:
    clean_id = clean_arxiv_id(paper_id)
    figure_url = figure_urls.get(clean_id) if clean_id and include_figure else None
    media = paper_media_html(
        paper_id=clean_id or paper_id,
        category=primary_category(metadata.get("categories")),
        year=metadata.get("year"),
        figure_url=figure_url,
        abs_url=_arxiv_url(paper_id),
    )
    band_color = BAND_COLORS.get(score_band or "", "#334155")
    return (
        f'<article class="paper-card" style="--band:{band_color}">'
        f"{media}"
        '<div class="paper-body">'
        f"<h3>{_html(title)}</h3>"
        f"{body_html}"
        "</div></article>"
    )


def _open_arxiv_html(arxiv_url: str | None, label: str) -> str:
    if not arxiv_url:
        return ""
    href = html.escape(arxiv_url, quote=True)
    return (
        f'<div class="source-meta"><a href="{href}" target="_blank" rel="noopener noreferrer">'
        f"{_html(label)}</a></div>"
    )


def _facet_row(label: str, items: list, colored: bool) -> str:
    chips = []
    for item in items:
        value = item.get("value")
        count = item.get("count")
        style = ""
        if colored:
            color = category_color(str(value or ""))
            style = f' style="border-color:{color};color:{color};background:{color}22"'
        chips.append(
            f'<span class="facet-chip"{style}>{_html(value)} '
            f'<span class="chip-count">{_html(count)}</span></span>'
        )
    if not chips:
        return ""
    return f'<div class="facet-row"><span class="facet-label">{label}</span>{"".join(chips)}</div>'


def _load_result_figures(paper_ids: list[str]) -> dict[str, str | None]:
    if uncached_figure_ids(paper_ids):
        with st.spinner("Fetching figures from arXiv..."):
            return load_figure_urls(paper_ids)
    return load_figure_urls(paper_ids)


def render_result(result: dict) -> None:
    chunks = result.get("context_chunks", [])
    papers = _metadata_by_paper(chunks)
    citations = result.get("citations", [])
    citation_ids = [
        str(item.get("arxiv_id") or "").strip()
        for item in citations
        if str(item.get("quote") or "").strip()
    ]
    figure_urls = _load_result_figures(citation_ids)

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown("<h2>Answer</h2>", unsafe_allow_html=True)
    st.markdown(
        render_answer_html(result, abs_base_url=ARXIV_ABS_BASE_URL),
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
    if citations:
        st.markdown("<h2>Citations</h2>", unsafe_allow_html=True)
        for item in citations:
            quote = str(item.get("quote") or "").strip()
            if not quote:
                continue
            arxiv_id = str(item.get("arxiv_id") or "").strip()
            metadata = papers.get(arxiv_id, {})
            title = str(item.get("title") or "").strip() or arxiv_id or "Untitled"
            body = (
                f"{_open_arxiv_html(_arxiv_url(arxiv_id), f'Open on arXiv ({arxiv_id})' if arxiv_id else 'Open on arXiv')}"
                f'<div class="chunk-text">"{_html(quote)}"</div>'
            )
            st.markdown(
                _paper_card_html(
                    title=title,
                    paper_id=arxiv_id,
                    metadata=metadata,
                    figure_urls=figure_urls,
                    body_html=body,
                ),
                unsafe_allow_html=True,
            )

    facets = result.get("facets", {})
    if facets:
        st.markdown("<h2>Facets</h2>", unsafe_allow_html=True)
        facet_html = "".join(
            (
                _facet_row("Years", facets.get("years", [])[:UI_FACET_LIMIT], False),
                _facet_row("Categories", facets.get("categories", [])[:UI_FACET_LIMIT], True),
                _facet_row("Authors", facets.get("authors", [])[:UI_FACET_LIMIT], False),
            )
        )
        if facet_html:
            st.markdown(facet_html, unsafe_allow_html=True)

    st.markdown("<h2>Sources</h2>", unsafe_allow_html=True)
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown")
        paper_id = _paper_id_from_metadata(metadata)
        year = metadata.get("year")
        categories = _as_list(metadata.get("categories"))
        authors = _as_list(metadata.get("authors"))
        category_text = ", ".join(str(category) for category in categories)
        author_text = ", ".join(str(author) for author in authors[:UI_MAX_AUTHORS_SHOWN])
        score_value, score_band = _chunk_confidence(chunk)
        metadata_segments = []
        source_text = str(source)
        if source_text and "/" not in source_text and "\\" not in source_text:
            metadata_segments.append(f"Source: {_html(source_text)}")
        if year:
            metadata_segments.append(f"Year: {_html(year)}")
        if category_text:
            metadata_segments.append(f"Categories: {_html(category_text)}")
        if author_text:
            metadata_segments.append(f"Authors: {_html(author_text)}")
        meta_html = (
            f'<div class="source-meta">{" | ".join(metadata_segments)}</div>'
            if metadata_segments
            else ""
        )
        already_cited = clean_arxiv_id(paper_id) in {
            clean_arxiv_id(arxiv_id) for arxiv_id in citation_ids
        }
        body = (
            f"{_open_arxiv_html(_arxiv_url(paper_id), 'Open on arXiv')}"
            f"{meta_html}"
            f"{_relevance_html(score_value, score_band)}"
            f"{_passage_html(str(chunk.get('text', '')), str(title))}"
        )
        st.markdown(
            _paper_card_html(
                title=str(title),
                paper_id=paper_id,
                metadata=metadata,
                figure_urls=figure_urls,
                body_html=body,
                score_band=score_band,
                include_figure=not already_cited,
            ),
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_examples() -> None:
    st.markdown(
        """
        <div class="examples">
            <h2>Try a question</h2>
            <p>Each one searches the papers in your index.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, type="secondary", use_container_width=True, key=f"example-{index}"):
            st.session_state["pending_query"] = example
            st.session_state["auto_query"] = example
            st.rerun()


def render_footer() -> None:
    st.markdown(
        (
            f'<footer class="app-footer">ArxivLens &copy; {datetime.now().year}'
            '<p class="ack">Thank you to arXiv for use of its open access interoperability. '
            "This product was not reviewed or approved by, nor does it necessarily express "
            "or reflect the policies or opinions of, arXiv.</p>"
            "</footer>"
        ),
        unsafe_allow_html=True,
    )


def _store_query_outcome(result: dict | None, error: str | None) -> None:
    if result is not None:
        st.session_state[RESULT_KEY] = result
        st.session_state.pop(ERROR_KEY, None)
        return
    if error:
        st.session_state[ERROR_KEY] = error
        st.session_state.pop(RESULT_KEY, None)


def main() -> None:
    st.set_page_config(page_title="ArxivLens", page_icon=":books:", layout="centered")
    inject_styles()
    render_header()

    query, top_k, filters, ask_clicked, year_error = render_query_form()
    auto_query = st.session_state.pop("auto_query", None)
    submitted_query = auto_query if auto_query else query

    if ask_clicked or auto_query:
        error = None
        result = None
        if not str(submitted_query).strip():
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
                with st.status("Running pipeline...", expanded=False) as status:
                    status.update(label="Retrieving papers...", state="running", expanded=True)
                    result = query_rag_api(str(submitted_query).strip(), top_k, filters)
                    status.update(label="Completed", state="complete", expanded=False)
            except requests.exceptions.RequestException as exc:
                error = (
                    "Could not reach backend API. Make sure `python main.py` is running "
                    f"on `{API_BASE_URL}`. Details: {exc}"
                )
        _store_query_outcome(result, error)

    error = st.session_state.get(ERROR_KEY)
    result = st.session_state.get(RESULT_KEY)
    if error:
        st.markdown(f'<div class="error-box">Error: {_html(error)}</div>', unsafe_allow_html=True)

    if result:
        render_result(result)
    else:
        render_examples()
    render_footer()


if __name__ == "__main__":
    main()
