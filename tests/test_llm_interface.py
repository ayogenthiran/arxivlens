import json

from src.llm.llm_interface import LLMInterface


def _context():
    return [
        {
            "id": "1706.03762-0",
            "text": "Attention is all you need for sequence transduction.",
            "metadata": {
                "doc_id": "1706.03762",
                "title": "Attention Is All You Need",
                "chunk_id": "1706.03762-0",
            },
        },
        {
            "id": "1810.04805-0",
            "text": "BERT pretrains deep bidirectional representations.",
            "metadata": {"doc_id": "1810.04805", "title": "BERT"},
        },
    ]


def _llm():
    return LLMInterface()


def test_parse_keeps_citation_when_quote_is_in_chunk():
    raw = json.dumps(
        {
            "answer": "Transformers use attention.",
            "citations": [
                {"doc_id": 1, "quote": " Attention is all you need "},
                {"doc_id": 1, "quote": "this quote is not in the chunk"},
                {"doc_id": 9, "quote": "Attention is all you need"},
            ],
        }
    )
    parsed = _llm()._parse_and_validate_response(raw, _context())
    assert parsed == {
        "answer": "Transformers use attention.",
        "summary": "Transformers use attention.",
        "points": [],
        "citations": [
            {
                "doc_id": 1,
                "arxiv_id": "1706.03762",
                "chunk_id": "1706.03762-0",
                "title": "Attention Is All You Need",
                "quote": "Attention is all you need",
            }
        ],
    }


def test_parse_drops_quote_that_is_not_in_chunk():
    raw = json.dumps(
        {
            "answer": "Not supported.",
            "citations": [{"doc_id": 1, "quote": "A sentence the chunk never contains."}],
        }
    )
    parsed = _llm()._parse_and_validate_response(raw, _context())
    assert parsed == {
        "answer": "Not supported.",
        "summary": "Not supported.",
        "points": [],
        "citations": [],
    }


def test_parse_drops_out_of_range_doc_id():
    raw = json.dumps(
        {
            "answer": "Out of range.",
            "citations": [
                {"doc_id": 0, "quote": "Attention is all you need"},
                {"doc_id": 3, "quote": "BERT pretrains deep bidirectional representations."},
            ],
        }
    )
    parsed = _llm()._parse_and_validate_response(raw, _context())
    assert parsed == {
        "answer": "Out of range.",
        "summary": "Out of range.",
        "points": [],
        "citations": [],
    }


def test_parse_invalid_json_falls_back_to_raw_text():
    raw = "  The model replied in prose.  "
    parsed = _llm()._parse_and_validate_response(raw, _context())
    assert parsed == {
        "answer": "The model replied in prose.",
        "summary": "The model replied in prose.",
        "points": [],
        "citations": [],
    }


def test_parse_empty_response():
    parsed = _llm()._parse_and_validate_response("  \n", _context())
    assert parsed == {
        "answer": "No answer returned.",
        "summary": "No answer returned.",
        "points": [],
        "citations": [],
    }


def test_parse_structured_points_keeps_valid_sources_and_strips_markup():
    raw = json.dumps(
        {
            "summary": "Transformers use **attention** (Document 1).",
            "points": [
                {
                    "title": "1. Attention",
                    "detail": "Sequence models use attention (Document 1, Document 9).",
                    "doc_ids": [1, 1, 9, True],
                },
                {"title": "   ", "detail": "   ", "doc_ids": [2]},
                {
                    "title": "BERT",
                    "detail": "BERT pretrains deep bidirectional representations.",
                    "doc_ids": [2, 0],
                },
            ],
            "citations": [{"doc_id": 1, "quote": "Attention is all you need"}],
        }
    )
    parsed = _llm()._parse_and_validate_response(raw, _context())
    assert parsed["summary"] == "Transformers use attention."
    assert parsed["points"] == [
        {
            "title": "Attention",
            "detail": "Sequence models use attention.",
            "sources": [
                {
                    "doc_id": 1,
                    "arxiv_id": "1706.03762",
                    "title": "Attention Is All You Need",
                }
            ],
        },
        {
            "title": "BERT",
            "detail": "BERT pretrains deep bidirectional representations.",
            "sources": [
                {"doc_id": 2, "arxiv_id": "1810.04805", "title": "BERT"}
            ],
        },
    ]
    assert parsed["answer"] == (
        "Transformers use attention.\n"
        "1. Attention: Sequence models use attention.\n"
        "2. BERT: BERT pretrains deep bidirectional representations."
    )
    assert parsed["citations"][0]["arxiv_id"] == "1706.03762"


def test_parse_markdown_prose_strips_bold_and_document_labels():
    raw = "  **Hello** (Document 1).  "
    parsed = _llm()._parse_and_validate_response(raw, _context())
    assert parsed["answer"] == "Hello."
    assert parsed["points"] == []


def test_prompt_asks_for_summary_and_points():
    prompt = _llm()._create_prompt("What is attention?", "[Document 1] Attention")
    assert '"summary"' in prompt
    assert '"points"' in prompt
    assert "No markdown" in prompt
    assert "helpful" not in prompt.lower()


def test_query_response_accepts_points():
    from src.api.server import QueryResponse

    payload = QueryResponse(
        answer="Lead\n1. Attention: Models use attention.",
        summary="Lead",
        points=[
            {
                "title": "Attention",
                "detail": "Models use attention.",
                "sources": [{"doc_id": 1, "arxiv_id": "1706.03762", "title": "Attention Is All You Need"}],
            }
        ],
        citations=[],
        context_chunks=[],
        facets={},
        query="What is attention?",
        rewritten_query="What is attention?",
    )
    assert payload.summary == "Lead"
    assert payload.points[0]["title"] == "Attention"
