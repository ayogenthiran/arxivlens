from src.ui.answer_display import render_answer_html


def test_structured_answer_renders_rows_and_paper_links():
    html = render_answer_html(
        {
            "summary": "These papers describe two approaches.",
            "points": [
                {
                    "title": "Peer review",
                    "detail": "Models score each other's answers.",
                    "sources": [
                        {
                            "doc_id": 1,
                            "arxiv_id": "1706.03762",
                            "title": "Attention Is All You Need",
                        }
                    ],
                },
                {
                    "title": "Uncertainty",
                    "detail": "Report how sure the model is.",
                    "sources": [{"doc_id": 2, "arxiv_id": "not a paper", "title": "BERT"}],
                },
            ],
        },
    )
    assert "These papers describe two approaches." in html
    assert 'class="point-index">1<' in html
    assert 'class="point-index">2<' in html
    assert "Peer review" in html
    assert "Models score each other&#x27;s answers." in html
    assert 'href="#paper-1706.03762"' in html
    assert "https://arxiv.org/abs/1706.03762" not in html
    assert "Attention Is All You Need" in html
    assert 'class="source-chip"' in html
    assert "<ol" in html
    assert "not a paper" not in html
    assert ">BERT</span>" in html


def test_plain_answer_when_there_are_no_points():
    html = render_answer_html(
        {"answer": "I do not have enough information in the provided context.", "points": []},
    )
    assert "I do not have enough information" in html
    assert "<ol" not in html


def test_answer_escapes_markup():
    html = render_answer_html(
        {
            "summary": "<script>alert(1)</script>",
            "points": [{"title": "Safe", "detail": "ok", "sources": []}],
        },
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
