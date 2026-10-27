import base64
import html
from datetime import datetime

import requests
import streamlit as st

from src.settings import (
    API_BASE_URL,
    ARXIV_ABS_BASE_URL,
    ARXIV_HOME_URL,
    COMMON_ARXIV_CATEGORIES,
    DEFAULT_TOP_K,
    UI_CONFIDENCE_HIGH,
    UI_CONFIDENCE_MEDIUM,
    UI_MAX_TOP_K,
    UI_MIN_TOP_K,
    YEAR_MAX_BOUND,
    YEAR_MIN_BOUND,
)
from src.ui.answer_display import render_answer_html
from src.ui.examples import (
    EXAMPLE_PROMPT,
    EXAMPLE_QUESTIONS,
    example_chip_label,
    example_retrieval_filters,
)
from src.ui.paper_visuals import (
    arxiv_logo_path,
    clean_arxiv_id,
    load_figure_urls,
    uncached_figure_ids,
)
from src.ui.result_cards import (
    SORT_NEWEST,
    SORT_RELEVANCE,
    arxiv_pdf_url,
    build_bibtex,
    citation_quotes,
    order_papers,
    paper_card_html,
)
from src.ui.search_flow import request_answer, search_papers, search_then_answer
from src.ui.search_params import decode_search_params, encode_search_params, present_search_params
from src.ui.sidebar_filters import (
    AUTHOR_INPUT_HINT,
    active_filter_count,
    author_filter_name,
    clear_all_filters,
    clear_author_filter,
    filters_heading,
    results_empty_message,
)

QUERY_INPUT_KEY = "query_input"
TOP_K_KEY = "top_k_input"
CATEGORIES_KEY = "categories_input"
AUTHORS_KEY = "authors_input"
YEAR_RANGE_KEY = "year_range_input"
SORT_KEY = "sort_by_input"
CATEGORY_CUSTOM_KEY = "category_custom_input"
PENDING_CATEGORIES_KEY = "pending_categories"
CLEAR_CATEGORY_CUSTOM_KEY = "clear_category_custom"
SEARCH_HYDRATED_KEY = "search_params_hydrated"
SEARCH_RESULT_KEY = "search_result"
APPLIED_FILTERS_KEY = "applied_search_filters"
ANSWER_KEY = "answer_result"
ANSWER_ERROR_KEY = "answer_error"
ERROR_KEY = "query_error"
SUMMARY_TITLE = "AI summary of these papers"
GLOBAL_STYLES = """
<style>
header[data-testid="stHeader"] {
    background: transparent !important;
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
    min-width: 40px;
    min-height: 40px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
}
.arxiv-mark img {
    height: 42px;
    width: auto;
    display: block;
}
.example-prompt {
    margin: 0.2rem 0 0.45rem 0;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.4;
}
[data-testid="stVerticalBlock"].st-key-example_chips {
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: flex-start !important;
    gap: 0.45rem 0.5rem !important;
}
[data-testid="stVerticalBlock"].st-key-example_chips > [data-testid="stElementContainer"] {
    width: fit-content !important;
    max-width: calc(50% - 0.35rem) !important;
    flex: 0 1 auto !important;
}
[data-testid="stVerticalBlock"].st-key-example_chips button[kind="secondary"] {
    width: auto !important;
    max-width: 100% !important;
    height: auto !important;
    min-height: 40px !important;
    border-radius: 999px !important;
    padding: 0.35rem 0.75rem !important;
    margin-bottom: 0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    line-height: 1.3 !important;
    white-space: normal !important;
    text-align: left !important;
}
[data-testid="stVerticalBlock"].st-key-example_chips button[kind="secondary"] p {
    width: auto !important;
    font-size: 0.82rem !important;
    white-space: normal !important;
    text-align: left !important;
}
[data-testid="stVerticalBlock"].st-key-example_chips button em {
    font-style: normal;
    font-size: 0.85em;
    font-weight: 400;
    color: #94a3b8;
}
.query-label {
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.stTextInput input, .stTextArea textarea,
div[data-testid="stNumberInput"] input {
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    min-height: 40px !important;
    box-sizing: border-box !important;
}
div[data-baseweb="select"] > div {
    min-height: 40px !important;
}
button[data-testid="stBaseButton-headerNoPadding"],
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"],
[data-testid="stTooltipIcon"] {
    min-width: 40px !important;
    min-height: 40px !important;
}
div[data-testid="stRadio"] label {
    min-height: 40px !important;
    min-width: 40px !important;
    display: inline-flex !important;
    align-items: center !important;
}
label[data-testid="stWidgetLabel"] {
    min-height: 40px !important;
    align-items: center;
}
div[data-testid="stSlider"] [role="slider"] {
    width: 16px !important;
    height: 16px !important;
}
div[data-testid="stSliderThumbValue"] {
    color: #f8fafc !important;
    background: #0f172a !important;
    border-radius: 4px;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #cbd5e1 !important;
    opacity: 1 !important;
}
button[kind="secondaryFormSubmit"],
button[kind="primary"] {
    background: #2563eb !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    min-height: 40px !important;
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
    min-height: 40px !important;
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
div[data-testid="stLinkButton"] a,
div[data-testid="stPopover"] button {
    background: #111827 !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0.75rem !important;
    margin-bottom: 0 !important;
    min-height: 40px !important;
    box-sizing: border-box !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-decoration: none !important;
}
div[data-testid="stLinkButton"] a {
    color: #93c5fd !important;
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
.answer-box, .paper-card {
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
    box-sizing: border-box;
    max-width: min(280px, 100%);
    min-height: 40px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: middle;
    margin: 0 0.35rem 0.35rem 0;
    padding: 0.55rem 0.75rem;
    border-radius: 999px;
    border: 1px solid #334155;
    background: #1e293b;
    color: #93c5fd;
    font-size: 0.8rem;
    line-height: 1.3;
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
    scroll-margin-top: 5rem;
}
.paper-body {
    min-width: 0;
    flex: 1;
}
.paper-body h3 {
    margin-top: 0;
    padding: 0 !important;
    font-size: 18px;
    line-height: 1.35 !important;
    display: -webkit-box !important;
    -webkit-line-clamp: 3 !important;
    -webkit-box-orient: vertical !important;
    overflow: hidden !important;
    max-height: calc(1.35em * 3) !important;
    overflow-wrap: anywhere;
}
.paper-byline {
    margin-top: 0.2rem;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.45;
}
.category-chip {
    display: inline-block;
    padding: 0.02rem 0.45rem;
    border-radius: 999px;
    border: 1px solid #334155;
    font-size: 0.78rem;
    line-height: 1.4;
    vertical-align: baseline;
}
.band-label {
    display: inline-block;
    margin-left: 0.55rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    color: var(--band, #94a3b8);
}
.best-sentence {
    margin: 0.5rem 0 0 0;
    color: #e2e8f0;
    line-height: 1.45;
}
mark {
    background: rgba(250, 204, 21, 0.28);
    color: inherit;
    border-radius: 2px;
    padding: 0 0.12em;
}
.cited-badge {
    display: inline-block;
    margin-top: 0.55rem;
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
    background: rgba(52, 211, 153, 0.15);
    color: #6ee7b7;
    font-size: 0.75rem;
    font-weight: 700;
}
.cited-quote {
    margin: 0.4rem 0 0 0;
    padding-left: 0.7rem;
    border-left: 2px solid #334155;
    color: #e2e8f0;
    font-size: 0.92rem;
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
a.paper-media.has-figure, .paper-media.has-figure {
    flex: 0 0 132px;
    width: 132px;
    height: auto;
    overflow: hidden;
    background: #0b1220;
}
.paper-media.has-figure img {
    width: 100%;
    height: 120px;
    object-fit: contain;
    object-position: center top;
    background: #f8fafc;
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
    min-height: 40px;
    display: flex;
    align-items: center;
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
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    height: 3.75rem !important;
    padding: 0 0.75rem !important;
    box-sizing: border-box !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0 !important;
    padding-bottom: 0.5rem !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.45rem !important;
}
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
    min-height: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0.15rem !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.field-hint) {
    padding-bottom: 0.85rem;
}
section[data-testid="stSidebar"] div[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 1.65rem;
}
section[data-testid="stSidebar"] button[kind="secondary"] {
    padding: 0.2rem 0.65rem !important;
    margin-bottom: 0 !important;
}
section[data-testid="stSidebar"] button[kind="secondary"] p {
    width: auto !important;
    text-align: center !important;
    white-space: nowrap !important;
}
section[data-testid="stSidebar"] .filter-heading {
    margin: 0;
    min-height: 40px;
    display: flex;
    align-items: center;
    font-weight: 700;
    color: #f8fafc;
    font-size: 1.05rem;
    line-height: 1.2;
}
section[data-testid="stSidebar"] .field-hint {
    margin: 0;
    color: #94a3b8;
    font-size: 0.78rem;
    line-height: 1.3;
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
    .app-header, .brand-row, .brand-copy, .paper-card, .paper-body, .point-copy {
        min-width: 0;
        max-width: 100%;
    }
    .arxiv-mark {
        padding: 0.15rem;
        min-width: 40px;
        min-height: 40px;
    }
    .arxiv-mark img {
        height: 28px;
    }
    .lens-mark svg {
        width: 32px;
        height: 32px;
    }
    .paper-card a.paper-media,
    .paper-card .paper-media,
    a.paper-media.has-figure,
    .paper-media.has-figure {
        display: none !important;
    }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)


def _html(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def _arxiv_url(paper_id: object) -> str | None:
    clean = clean_arxiv_id(paper_id)
    if not clean:
        return None
    return f"{ARXIV_ABS_BASE_URL}/{clean}"


def _year_range_value(year_min: int | None, year_max: int | None) -> tuple[int, int]:
    start = YEAR_MIN_BOUND if year_min is None else int(year_min)
    end = YEAR_MAX_BOUND if year_max is None else int(year_max)
    start = min(max(start, YEAR_MIN_BOUND), YEAR_MAX_BOUND)
    end = min(max(end, YEAR_MIN_BOUND), YEAR_MAX_BOUND)
    if start > end:
        start, end = end, start
    return start, end


def _category_options() -> list[str]:
    selected = st.session_state.get(CATEGORIES_KEY) or []
    if isinstance(selected, str):
        selected = [part.strip() for part in selected.split(",") if part.strip()]
    options = list(COMMON_ARXIV_CATEGORIES)
    for value in selected:
        text = str(value).strip()
        if text and text not in options:
            options.append(text)
    return options


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
                    <p>Find the papers your research should be cited with.</p>
                </div>
            </div>
            {logo_html}
        </header>
        """,
        unsafe_allow_html=True,
    )


def _ensure_form_state() -> None:
    st.session_state.setdefault(QUERY_INPUT_KEY, "")
    st.session_state.setdefault(TOP_K_KEY, DEFAULT_TOP_K)
    st.session_state.setdefault(CATEGORIES_KEY, [])
    st.session_state.setdefault(AUTHORS_KEY, "")
    st.session_state.setdefault(YEAR_RANGE_KEY, (YEAR_MIN_BOUND, YEAR_MAX_BOUND))
    st.session_state.setdefault(SORT_KEY, SORT_RELEVANCE)
    st.session_state.setdefault(CATEGORY_CUSTOM_KEY, "")
    categories = st.session_state.get(CATEGORIES_KEY)
    if isinstance(categories, str):
        st.session_state[CATEGORIES_KEY] = [part.strip() for part in categories.split(",") if part.strip()]


def _typed_categories(raw: str) -> list[str]:
    found: list[str] = []
    for part in raw.split(","):
        text = part.strip()
        if text and text not in found:
            found.append(text)
    return found


def _consume_typed_categories() -> None:
    """Move a typed category into the multiselect before that widget is created."""
    if st.session_state.pop(CLEAR_CATEGORY_CUSTOM_KEY, False):
        st.session_state[CATEGORY_CUSTOM_KEY] = ""
    pending = st.session_state.pop(PENDING_CATEGORIES_KEY, None)
    if not pending:
        return
    selected = st.session_state.get(CATEGORIES_KEY) or []
    if isinstance(selected, str):
        selected = _typed_categories(selected)
    else:
        selected = [str(value).strip() for value in selected if str(value).strip()]
    for value in pending:
        if value not in selected:
            selected.append(value)
    st.session_state[CATEGORIES_KEY] = selected


def _hydrate_search_from_query_params() -> None:
    if st.session_state.get(SEARCH_HYDRATED_KEY):
        return
    query, top_k, filters = decode_search_params(st.query_params)
    st.session_state[QUERY_INPUT_KEY] = query
    st.session_state[TOP_K_KEY] = top_k
    st.session_state[CATEGORIES_KEY] = list(filters["categories"])
    st.session_state[AUTHORS_KEY] = ", ".join(filters["authors"])
    st.session_state[YEAR_RANGE_KEY] = _year_range_value(filters["year_min"], filters["year_max"])
    st.session_state[SEARCH_HYDRATED_KEY] = True
    if query:
        st.session_state["auto_query"] = query


def _publish_pending_example() -> None:
    """Write a chip's preset into the URL before the sidebar widgets render."""
    filters = st.session_state.pop("pending_example_filters", None)
    if filters is None:
        return
    query = str(st.session_state.get(QUERY_INPUT_KEY) or "")
    top_k = int(st.session_state.get(TOP_K_KEY, DEFAULT_TOP_K))
    _persist_search_params(query, top_k, filters)


def _persist_search_params(query: str, top_k: int, filters: dict) -> None:
    desired = encode_search_params(query, top_k, filters)
    if present_search_params(st.query_params) == desired:
        return
    st.query_params.from_dict(desired)


def _filters_from_sidebar(categories: list, authors_raw: str, year_range: tuple[int, int]) -> dict:
    selected_categories: list[str] = []
    for value in categories:
        selected_categories.extend(part.strip() for part in str(value).split(",") if part.strip())
    year_min, year_max = year_range
    # The slider's outer bounds mean "no year filter", so undated papers stay in the result.
    return {
        "categories": selected_categories,
        "authors": [part.strip() for part in authors_raw.split(",") if part.strip()],
        "year_min": None if int(year_min) <= YEAR_MIN_BOUND else int(year_min),
        "year_max": None if int(year_max) >= YEAR_MAX_BOUND else int(year_max),
    }


def _on_clear_all_filters() -> None:
    clear_all_filters(
        st.session_state,
        st.query_params,
        categories_key=CATEGORIES_KEY,
        authors_key=AUTHORS_KEY,
        year_key=YEAR_RANGE_KEY,
        extra_blank_keys=(CATEGORY_CUSTOM_KEY,),
        pop_keys=(PENDING_CATEGORIES_KEY, CLEAR_CATEGORY_CUSTOM_KEY),
    )


def _on_clear_author_filter() -> None:
    clear_author_filter(st.session_state, st.query_params, authors_key=AUTHORS_KEY)


def _sidebar_filter_state() -> dict:
    categories = st.session_state.get(CATEGORIES_KEY) or []
    if isinstance(categories, str):
        categories = _typed_categories(categories)
    else:
        categories = list(categories)
    year_range = st.session_state.get(YEAR_RANGE_KEY, (YEAR_MIN_BOUND, YEAR_MAX_BOUND))
    if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
        year_range = (YEAR_MIN_BOUND, YEAR_MAX_BOUND)
    return _filters_from_sidebar(
        categories,
        str(st.session_state.get(AUTHORS_KEY) or ""),
        (int(year_range[0]), int(year_range[1])),
    )


def render_sidebar() -> tuple[int, dict]:
    _ensure_form_state()
    _consume_typed_categories()
    with st.sidebar:
        heading = filters_heading(active_filter_count(_sidebar_filter_state()))
        title_col, clear_col = st.columns([1.7, 1], vertical_alignment="center", gap="small")
        title_col.markdown(
            f'<p class="filter-heading">{html.escape(heading)}</p>',
            unsafe_allow_html=True,
        )
        clear_col.button("Clear all", on_click=_on_clear_all_filters, key="clear_all_filters")
        categories = st.multiselect(
            "Categories",
            options=_category_options(),
            key=CATEGORIES_KEY,
            placeholder="Choose categories",
        )
        custom = st.text_input(
            "Add category",
            key=CATEGORY_CUSTOM_KEY,
            placeholder="cs.IR",
            help="Any arXiv category code. It is added to Categories.",
        )
        typed = [value for value in _typed_categories(custom) if value not in categories]
        if typed:
            st.session_state[PENDING_CATEGORIES_KEY] = typed
            st.session_state[CLEAR_CATEGORY_CUSTOM_KEY] = True
            st.rerun()
        year_range = st.slider(
            "Year",
            min_value=YEAR_MIN_BOUND,
            max_value=YEAR_MAX_BOUND,
            key=YEAR_RANGE_KEY,
        )
        authors_raw = st.text_input("Author", placeholder="Yann LeCun", key=AUTHORS_KEY)
        st.markdown(
            f'<p class="field-hint">{html.escape(AUTHOR_INPUT_HINT)}</p>',
            unsafe_allow_html=True,
        )
        st.radio(
            "Sort by",
            options=[SORT_RELEVANCE, SORT_NEWEST],
            horizontal=True,
            key=SORT_KEY,
        )
        top_k = st.number_input(
            "Results to show",
            min_value=UI_MIN_TOP_K,
            max_value=UI_MAX_TOP_K,
            step=1,
            key=TOP_K_KEY,
        )
    return int(top_k), _filters_from_sidebar(list(categories), authors_raw, tuple(year_range))


def render_query_form() -> tuple[str, bool]:
    _ensure_form_state()
    with st.form("query-form", clear_on_submit=False):
        st.markdown(
            '<div class="query-label">Describe your research topic or question</div>',
            unsafe_allow_html=True,
        )
        query = st.text_area(
            "Describe your research topic or question",
            placeholder="Enter your question...",
            label_visibility="collapsed",
            height=110,
            key=QUERY_INPUT_KEY,
        )
        ask_clicked = st.form_submit_button("Ask")
    return query, ask_clicked


def _confidence_band(chunk: dict) -> str | None:
    score = chunk.get("reranker_score")
    if score is None:
        score = chunk.get("vector_score")
    if score is None:
        return None
    score = float(score)
    score = min(max(score, 0.0), 1.0)
    if score >= UI_CONFIDENCE_HIGH:
        return "High"
    if score >= UI_CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def _paper_id_from_metadata(metadata: dict) -> str:
    return str(metadata.get("doc_id") or metadata.get("id") or metadata.get("paper_id") or "").strip()


def _load_result_figures(paper_ids: list[str]) -> dict[str, str | None]:
    if uncached_figure_ids(paper_ids):
        with st.spinner("Fetching figures from arXiv..."):
            return load_figure_urls(paper_ids)
    return load_figure_urls(paper_ids)


def render_summary(answer: dict) -> None:
    with st.expander(SUMMARY_TITLE, expanded=True):
        st.markdown(
            render_answer_html(answer),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="trust-note">AI-generated synthesis. Verify details with linked sources.</div>',
            unsafe_allow_html=True,
        )


def render_summary_error(message: str) -> None:
    with st.expander(SUMMARY_TITLE, expanded=True):
        st.markdown(f'<div class="error-box">{_html(message)}</div>', unsafe_allow_html=True)


def _render_card_actions(paper_id: str, metadata: dict) -> None:
    actions: list[tuple[str, str]] = []
    abs_url = _arxiv_url(paper_id)
    pdf_url = arxiv_pdf_url(paper_id)
    if abs_url:
        actions.append(("arXiv", abs_url))
    if pdf_url:
        actions.append(("PDF", pdf_url))
    copy_count = 2 if clean_arxiv_id(paper_id) else 1
    columns = st.columns(len(actions) + copy_count)
    for index, (label, url) in enumerate(actions):
        columns[index].link_button(label, url, use_container_width=True)
    bibtex_column = columns[len(actions)]
    with bibtex_column.popover("Copy BibTeX", use_container_width=True):
        st.code(build_bibtex(metadata, paper_id), language=None)
    clean_id = clean_arxiv_id(paper_id)
    if clean_id:
        with columns[len(actions) + 1].popover("Copy ID", use_container_width=True):
            st.code(clean_id, language=None)


def render_papers(
    result: dict,
    answer: dict | None = None,
    filters: dict | None = None,
    *,
    show_author_clear: bool = True,
) -> None:
    filters = filters or {}
    chunks = order_papers(list(result.get("context_chunks") or []), st.session_state.get(SORT_KEY, SORT_RELEVANCE))
    paper_ids = [_paper_id_from_metadata(chunk.get("metadata") or {}) for chunk in chunks if isinstance(chunk, dict)]
    figure_urls = _load_result_figures([paper_id for paper_id in paper_ids if paper_id])
    query = str(result.get("query") or "")

    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown("<h2>Papers</h2>", unsafe_allow_html=True)
    if not chunks:
        message = results_empty_message(0, filters) or ""
        st.markdown(f'<div class="source-meta">{html.escape(message)}</div>', unsafe_allow_html=True)
        if (
            show_author_clear
            and author_filter_name(filters)
            and str(st.session_state.get(AUTHORS_KEY) or "").strip()
        ):
            st.button("Clear author", on_click=_on_clear_author_filter, key="clear_author_filter")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata") or {}
        title = str(metadata.get("title") or "Untitled")
        paper_id = _paper_id_from_metadata(metadata)
        clean_id = clean_arxiv_id(paper_id)
        figure_url = figure_urls.get(clean_id) if clean_id else None
        st.markdown(
            paper_card_html(
                title=title,
                paper_id=paper_id,
                metadata=metadata,
                text=str(chunk.get("text") or ""),
                query=query,
                score_band=_confidence_band(chunk),
                figure_url=figure_url,
                quotes=citation_quotes(answer, paper_id),
                abs_url=_arxiv_url(paper_id),
            ),
            unsafe_allow_html=True,
        )
        _render_card_actions(paper_id, metadata)
    st.markdown("</div>", unsafe_allow_html=True)


def _apply_example(index: int) -> None:
    """Fill the query, preset the sidebar, and run that example on the next script run."""
    example = EXAMPLE_QUESTIONS[index]
    filters = example_retrieval_filters(example.get("filters") or {})
    st.session_state[QUERY_INPUT_KEY] = str(example["query"])
    st.session_state[CATEGORIES_KEY] = list(filters["categories"])
    st.session_state[AUTHORS_KEY] = ", ".join(filters["authors"])
    st.session_state[YEAR_RANGE_KEY] = _year_range_value(filters["year_min"], filters["year_max"])
    st.session_state[CATEGORY_CUSTOM_KEY] = ""
    st.session_state.pop(PENDING_CATEGORIES_KEY, None)
    st.session_state.pop(CLEAR_CATEGORY_CUSTOM_KEY, None)
    st.session_state["auto_query"] = str(example["query"])
    st.session_state["pending_example_filters"] = filters


def render_examples() -> None:
    st.markdown(
        f'<p class="example-prompt">{html.escape(EXAMPLE_PROMPT)}</p>',
        unsafe_allow_html=True,
    )
    with st.container(key="example_chips"):
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            st.button(
                example_chip_label(example),
                key=f"example-{index}",
                on_click=_apply_example,
                args=(index,),
            )


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


def _clear_results() -> None:
    st.session_state.pop(SEARCH_RESULT_KEY, None)
    st.session_state.pop(APPLIED_FILTERS_KEY, None)
    st.session_state.pop(ANSWER_KEY, None)
    st.session_state.pop(ANSWER_ERROR_KEY, None)


def main() -> None:
    st.set_page_config(page_title="ArxivLens", page_icon=":books:", layout="wide")
    inject_styles()
    _hydrate_search_from_query_params()
    _publish_pending_example()
    top_k, filters = render_sidebar()
    render_header()

    query, ask_clicked = render_query_form()
    auto_query = st.session_state.pop("auto_query", None)
    submitted_query = auto_query if auto_query else query
    searching = bool(ask_clicked or auto_query)
    has_results = bool(st.session_state.get(SEARCH_RESULT_KEY))
    summary_slot = st.empty() if searching or has_results else None
    papers_slot = st.empty() if searching or has_results else None

    if searching:
        error = None
        if not str(submitted_query).strip():
            error = "Please enter a question first."
        else:
            submitted_text = str(submitted_query).strip()

            def on_papers(papers: dict) -> None:
                st.session_state[SEARCH_RESULT_KEY] = papers
                st.session_state[APPLIED_FILTERS_KEY] = filters
                st.session_state.pop(ANSWER_KEY, None)
                st.session_state.pop(ANSWER_ERROR_KEY, None)
                st.session_state.pop(ERROR_KEY, None)
                with papers_slot.container():
                    render_papers(papers, filters=filters, show_author_clear=False)

            def search_with_spinner(text: str, size: int, selected_filters: dict) -> dict:
                with st.spinner("Retrieving papers..."):
                    return search_papers(text, size, selected_filters)

            def answer_with_spinner(text: str, chunks: list) -> dict:
                with st.spinner("Writing the summary..."):
                    return request_answer(text, chunks)

            try:
                _papers, answer, answer_error = search_then_answer(
                    submitted_text,
                    top_k,
                    filters,
                    on_papers,
                    search=search_with_spinner,
                    answer=answer_with_spinner,
                )
            except requests.exceptions.RequestException as exc:
                error = (
                    "Could not reach backend API. Make sure `python main.py` is running "
                    f"on `{API_BASE_URL}`. Details: {exc}"
                )
            else:
                if answer is not None:
                    st.session_state[ANSWER_KEY] = answer
                    st.session_state.pop(ANSWER_ERROR_KEY, None)
                else:
                    st.session_state.pop(ANSWER_KEY, None)
                    st.session_state[ANSWER_ERROR_KEY] = answer_error
            _persist_search_params(submitted_text, top_k, filters)
        if error:
            st.session_state[ERROR_KEY] = error
            _clear_results()

    error = st.session_state.get(ERROR_KEY)
    papers = st.session_state.get(SEARCH_RESULT_KEY)
    if error:
        st.markdown(f'<div class="error-box">Error: {_html(error)}</div>', unsafe_allow_html=True)

    if papers and summary_slot is not None and papers_slot is not None:
        with summary_slot:
            answer_error = st.session_state.get(ANSWER_ERROR_KEY)
            answer = st.session_state.get(ANSWER_KEY)
            if answer_error:
                render_summary_error(answer_error)
            elif answer:
                render_summary(answer)
        with papers_slot.container():
            applied_filters = st.session_state.get(APPLIED_FILTERS_KEY, filters)
            render_papers(papers, answer=st.session_state.get(ANSWER_KEY), filters=applied_filters)
    else:
        render_examples()
    render_footer()


if __name__ == "__main__":
    main()
