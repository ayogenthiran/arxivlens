import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

from src.settings import (
    ENABLE_QUERY_REWRITE,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_REQUEST_TIMEOUT_S,
    LLM_TEMPERATURE,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    QUERY_REWRITE_MAX_TOKENS,
    QUERY_REWRITE_MODEL,
    QUERY_REWRITE_TEMPERATURE,
)

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_ANSWER = "I do not have enough information in the provided context."
EMPTY_ANSWER = "No answer returned."
_DEFAULT_SYSTEM_MESSAGE = "You answer questions from the provided context."
_BOLD = re.compile(r"\*\*|__")
_DOC_REF = re.compile(
    r"\s*\(\s*Document\s+\d+(?:\s*,\s*Document\s+\d+)*\s*\)",
    re.IGNORECASE,
)
_LEADING_NUMBER = re.compile(r"^\d+[\.\)]\s*")
_ANSWER_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_answer_with_citations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "detail": {"type": "string"},
                            "doc_ids": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["title", "detail", "doc_ids"],
                        "additionalProperties": False,
                    },
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "integer"},
                            "quote": {"type": "string"},
                        },
                        "required": ["doc_id", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "points", "citations"],
            "additionalProperties": False,
        },
    },
}

class LLMInterface:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or LLM_MODEL_NAME
        self.rewrite_model_name = QUERY_REWRITE_MODEL or self.model_name
        self.api_key = OPENAI_API_KEY
        self.api_base = OPENAI_API_BASE

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. API calls will fail.")

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a structured answer whose citations were checked against the chunks."""
        formatted_context = self._format_context(context_chunks)
        prompt = self._create_prompt(query, formatted_context)
        raw_response = self._call_llm_api(
            prompt=prompt,
            system_message=(
                "You answer questions from the provided context. "
                "Write plain text in the summary and point fields. Do not use markdown."
            ),
            response_format=_ANSWER_RESPONSE_FORMAT,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return self._parse_and_validate_response(raw_response, context_chunks)

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite the user query into a retrieval-optimized version.
        Falls back to the original query if rewrite is disabled or fails.
        """
        if not query.strip():
            return query

        if not ENABLE_QUERY_REWRITE:
            return query

        rewrite_prompt = (
            "Rewrite the user query for retrieval in a scientific paper RAG system.\n"
            "Preserve intent, add key technical terms, and keep it concise.\n"
            "Return only the rewritten query on a single line.\n\n"
            f"User query: {query}\n"
            "Rewritten query:"
        )

        try:
            rewritten = self._call_llm_api(
                prompt=rewrite_prompt,
                system_message=(
                    "You rewrite search queries for document retrieval. "
                    "Do not answer the question."
                ),
                temperature=QUERY_REWRITE_TEMPERATURE,
                max_tokens=QUERY_REWRITE_MAX_TOKENS,
                model_name=self.rewrite_model_name,
            )
            rewritten = rewritten.strip().replace("\n", " ")
            return rewritten or query
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original query: {e}")
            return query

    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "No relevant context found."

        formatted_chunks = []
        for i, chunk in enumerate(context_chunks):
            title = chunk["metadata"].get("title", "Untitled")
            source = chunk["metadata"].get("source", "Unknown source")
            text = chunk["text"]
            formatted_chunk = f"[Document {i+1}] {title}\nSource: {source}\n\n{text}\n"
            formatted_chunks.append(formatted_chunk)

        return "\n".join(formatted_chunks)

    def _create_prompt(self, query: str, context: str) -> str:
        return f"""CONTEXT:
{context}

USER QUERY:
{query}

Answer the query from the provided context only.
If the context doesn't contain relevant information, set "summary" to "{INSUFFICIENT_CONTEXT_ANSWER}", "points" to [], and "citations" to [].
Return a JSON object with:
- "summary": one sentence that directly answers the query. Plain text. No markdown. Do not mention document numbers.
- "points": a list of distinct findings. Each point has "title" (a few words, plain text, no markdown), "detail" (one or two sentences, plain text, no markdown, no document numbers), and "doc_ids" (integers matching [Document N] that support the point).
- "citations": list of objects with "doc_id" (matching [Document N]) and "quote" (exact short supporting quote from that document)
Only cite documents that appear in the provided context. Keep each point tied to the documents that support it.
"""

    def _parse_and_validate_response(self, raw_response: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Keep citations whose quote actually appears in the cited chunk."""
        if not raw_response.strip():
            return _payload(EMPTY_ANSWER, EMPTY_ANSWER, [], [])

        try:
            parsed = json.loads(raw_response)
            citations = parsed.get("citations", [])
            if not isinstance(citations, list):
                citations = []
        except (json.JSONDecodeError, AttributeError, TypeError):
            prose = _clean_sentence(raw_response) or EMPTY_ANSWER
            return _payload(prose, prose, [], [])

        validated_citations = _validate_citations(citations, context_chunks)
        if "summary" in parsed or "points" in parsed:
            summary = _clean_sentence(parsed.get("summary", ""))
            points = _validate_points(parsed.get("points"), context_chunks)
            answer = _compose_answer(summary, points) or INSUFFICIENT_CONTEXT_ANSWER
            if not summary and not points:
                summary = answer
            return _payload(answer, summary, points, validated_citations)

        answer = _clean_sentence(parsed.get("answer", "")) or INSUFFICIENT_CONTEXT_ANSWER
        return _payload(answer, answer, [], validated_citations)

    def _call_llm_api(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("API key not configured. Please set the OPENAI_API_KEY environment variable.")

        if temperature is None:
            temperature = LLM_TEMPERATURE
        if max_tokens is None:
            max_tokens = LLM_MAX_TOKENS

        url = f"{self.api_base}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": model_name or self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_message or _DEFAULT_SYSTEM_MESSAGE,
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format is not None:
            data["response_format"] = response_format

        try:
            response = requests.post(url, headers=headers, json=data, timeout=LLM_REQUEST_TIMEOUT_S)
        except requests.RequestException as exc:
            logger.error("Error calling LLM API: %s", exc)
            raise RuntimeError(f"Error calling LLM API: {exc}") from exc

        if response.status_code != 200:
            logger.error("API call failed with status code %s: %s", response.status_code, response.text)
            raise RuntimeError(f"LLM API call failed: {response.text}")
        return response.json()["choices"][0]["message"]["content"]


def _payload(
    answer: str,
    summary: str,
    points: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "summary": summary,
        "points": points,
        "citations": citations,
    }


def _clean_sentence(value: object) -> str:
    text = _BOLD.sub("", str(value or ""))
    text = _DOC_REF.sub("", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    return text.strip()


def _clean_title(value: object) -> str:
    text = _LEADING_NUMBER.sub("", _clean_sentence(value))
    return text.strip(" :")


def _doc_id(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 1:
        return None
    return value


def _compose_answer(summary: str, points: List[Dict[str, Any]]) -> str:
    lines = [summary] if summary else []
    for index, point in enumerate(points, start=1):
        title = str(point.get("title") or "").strip()
        detail = str(point.get("detail") or "").strip()
        if title and detail:
            lines.append(f"{index}. {title}: {detail}")
        elif title or detail:
            lines.append(f"{index}. {title or detail}")
    return "\n".join(lines).strip()


def _source_for(doc_id: int, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk = context_chunks[doc_id - 1]
    metadata = chunk.get("metadata") or {}
    return {
        "doc_id": doc_id,
        "arxiv_id": str(metadata.get("doc_id") or "").strip(),
        "title": str(metadata.get("title") or "").strip(),
    }


def _validate_points(points: object, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(points, list):
        return []
    allowed = set(range(1, len(context_chunks) + 1))
    validated: List[Dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        title = _clean_title(point.get("title", ""))
        detail = _clean_sentence(point.get("detail", ""))
        if not title and not detail:
            continue
        doc_ids = point.get("doc_ids")
        if not isinstance(doc_ids, list):
            doc_ids = []
        sources: List[Dict[str, Any]] = []
        seen_docs: set[int] = set()
        seen_papers: set[str] = set()
        for raw_id in doc_ids:
            doc_id = _doc_id(raw_id)
            if doc_id is None or doc_id not in allowed or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            source = _source_for(doc_id, context_chunks)
            paper_key = source["arxiv_id"] or f"doc:{doc_id}"
            if paper_key in seen_papers:
                continue
            seen_papers.add(paper_key)
            sources.append(source)
        validated.append({"title": title, "detail": detail, "sources": sources})
    return validated


def _validate_citations(citations: List[Any], context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed_doc_ids = set(range(1, len(context_chunks) + 1))
    validated: List[Dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        doc_id = citation.get("doc_id")
        quote = citation.get("quote")
        if isinstance(doc_id, bool) or not isinstance(doc_id, int) or doc_id not in allowed_doc_ids:
            continue
        if not isinstance(quote, str) or not quote.strip():
            continue

        chunk = context_chunks[doc_id - 1]
        chunk_text = str(chunk.get("text", ""))
        if quote.strip().lower() not in chunk_text.lower():
            continue

        metadata = chunk.get("metadata") or {}
        validated.append(
            {
                "doc_id": doc_id,
                "arxiv_id": str(metadata.get("doc_id") or "").strip(),
                "chunk_id": str(chunk.get("id") or metadata.get("chunk_id") or "").strip(),
                "title": str(metadata.get("title") or "").strip(),
                "quote": quote.strip(),
            }
        )
    return validated
