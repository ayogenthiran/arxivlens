from pathlib import Path

from src.ui.result_cards import (
    SORT_NEWEST,
    SORT_RELEVANCE,
    best_abstract_sentence,
    build_bibtex,
    figure_thumbnail_html,
    format_authors,
    highlight_query_terms,
    order_papers,
    paper_card_html,
)

ROOT = Path(__file__).resolve().parents[1]


def test_best_sentence_picks_highest_token_overlap():
    abstract = (
        "Transformers use attention for translation. "
        "Later work studies hallucination in retrieved evidence."
    )
    sentence = best_abstract_sentence(abstract, "hallucination evidence")

    assert sentence == "Later work studies hallucination in retrieved evidence."

    tied = "Alpha models retrieve documents. Beta models retrieve documents."
    assert best_abstract_sentence(tied, "retrieve documents").startswith("Alpha")
    assert best_abstract_sentence("just one line about retrieval", "") == "just one line about retrieval"

    marked = highlight_query_terms("Hallucination is <bad>.", "hallucination")
    assert "<mark>Hallucination</mark>" in marked
    assert "<bad>" not in marked
    assert "&lt;bad&gt;" in marked
    assert "<mark>cat</mark>" not in highlight_query_terms("category labels", "cat")
    assert "<mark>" not in highlight_query_terms("Models in language", "in the")


def test_bibtex_entry_includes_arxiv_fields():
    entry = build_bibtex(
        {
            "title": "Attention & More {draft}",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "categories": ["cs.CL", "cs.LG"],
            "doc_id": "1706.03762",
        },
        "1706.03762v5",
    )

    assert entry.startswith("@misc{170603762v5,")
    assert "eprint = {1706.03762v5}" in entry
    assert "archivePrefix = {arXiv}" in entry
    assert "primaryClass = {cs.CL}" in entry
    assert "author = {Ashish Vaswani and Noam Shazeer}" in entry
    assert r"title = {Attention \& More \{draft\}}" in entry
    assert "<" not in entry


def test_author_line_uses_et_al_after_three_names():
    four = ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Donald Knuth"]

    assert format_authors(four) == "Ada Lovelace, Alan Turing, Grace Hopper et al."
    assert format_authors(four[:3]) == "Ada Lovelace, Alan Turing, Grace Hopper"
    assert "et al." not in format_authors(["Ada Lovelace"])
    assert format_authors("Ada Lovelace and Alan Turing and Grace Hopper and Donald Knuth") == (
        "Ada Lovelace, Alan Turing, Grace Hopper et al."
    )
    assert format_authors([]) == ""


def test_newest_sort_orders_submission_date_descending():
    chunks = [
        {"id": "old", "metadata": {"submission_date": "2020-01-02"}},
        {"id": "new", "metadata": {"submission_date": "2024-06-01"}},
        {"id": "none", "metadata": {"title": "undated"}},
        {"id": "mid", "metadata": {"submission_date": "2024-01-01"}},
        {"id": "same", "metadata": {"submission_date": "2024-06-01"}},
        {"id": "bad", "metadata": {"submission_date": "June 2024"}},
    ]

    newest = [chunk["id"] for chunk in order_papers(chunks, SORT_NEWEST)]
    relevance = [chunk["id"] for chunk in order_papers(chunks, SORT_RELEVANCE)]

    assert newest == ["new", "same", "mid", "old", "none", "bad"]
    assert relevance == ["old", "new", "none", "mid", "same", "bad"]
    assert [chunk["id"] for chunk in chunks] == relevance


def test_card_escapes_metadata_and_skips_a_missing_figure():
    markup = paper_card_html(
        title="<script>",
        paper_id="1706.03762",
        metadata={
            "title": "<script>",
            "authors": ["A <b>", "B", "C", "D"],
            "year": 2017,
            "categories": ["cs.CL"],
        },
        text="<script>\n\nAlpha is safe. Hallucination shows up in the evidence.",
        query="hallucination evidence",
        score_band="High",
        figure_url=None,
        quotes=["He said <q>hallucination</q>"],
        abs_url="https://arxiv.org/abs/1706.03762",
    )

    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup
    assert "A &lt;b&gt;, B, C et al." in markup
    assert "Cited in summary" in markup
    assert 'id="paper-1706.03762"' in markup
    assert "<mark>Hallucination</mark>" in markup
    assert "paper-media" not in markup
    assert "High" in markup
    assert "Source:" not in markup
    assert "relevance-track" not in markup
    assert "&lt;q&gt;" in markup

    hidden = figure_thumbnail_html(
        paper_id="1706.03762",
        metadata={"categories": ["cs.CL"], "year": 2017},
        figure_url="https://evil.example/x.png",
        abs_url=None,
    )
    shown = figure_thumbnail_html(
        paper_id="1706.03762",
        metadata={"categories": ["cs.CL"], "year": 2017},
        figure_url="https://arxiv.org/html/1706.03762v1/x.png",
        abs_url="https://arxiv.org/abs/1706.03762",
    )
    assert hidden == ""
    assert "has-figure" in shown
    assert "evil.example" not in shown


def test_results_layout_keeps_filters_in_the_sidebar():
    source = (ROOT / "streamlit_app.py").read_text()
    mobile = source.split("@media (max-width: 640px)", 1)[1]

    assert "st.sidebar" in source
    assert "Advanced retrieval settings" not in source
    assert ">Citations<" not in source
    assert ">Facets<" not in source
    assert '"Sort by"' in source
    assert "Add category" in source
    assert "accept_new_options" not in source
    assert "display: none" in mobile
    assert ".paper-media" in mobile
