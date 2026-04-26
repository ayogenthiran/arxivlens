import os
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
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


def query_rag_api(query: str, top_k: int) -> dict:
    response = requests.post(
        QUERY_URL,
        json={"query": query, "top_k": top_k},
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


def render_query_form() -> tuple[str, bool]:
    with st.form("query-form", clear_on_submit=False):
        st.markdown('<div class="query-label">Ask a question</div>', unsafe_allow_html=True)
        query = st.text_input(
            "Enter your question...",
            placeholder="Enter your question...",
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("Search")
    return query, ask_clicked


def render_result(result: dict) -> None:
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.markdown("<h2>Answer</h2>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{result.get("answer", "No answer returned.")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<h2>Sources</h2>", unsafe_allow_html=True)
    for chunk in result.get("context_chunks", []):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown")
        text = chunk.get("text", "")
        st.markdown(
            (
                '<div class="source-card">'
                f"<h3>{title}</h3>"
                f'<div class="source-meta">Source: {source}</div>'
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

    query, ask_clicked = render_query_form()

    if ask_clicked:
        if not query.strip():
            error = "Please enter a question first."
        else:
            try:
                with st.spinner("Searching..."):
                    result = query_rag_api(query.strip(), 5)
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
