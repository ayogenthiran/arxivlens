"""Paper-card text: authors, best abstract sentence, BibTeX, and display order.

Retrieval order is unchanged. Newest is a client-side reorder of papers already returned.
"""

from __future__ import annotations

import html
import re
from typing import Any

from src.settings import ARXIV_HOME_URL, UI_MAX_AUTHORS_SHOWN
from src.ui.paper_visuals import (
    category_color,
    clean_arxiv_id,
    paper_media_html,
    passage_body,
    passage_preview,
    primary_category,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
)

BAND_COLORS = {
    "High": "#34d399",
    "Medium": "#fbbf24",
    "Low": "#f87171",
}

SORT_RELEVANCE = "Relevance"
SORT_NEWEST = "Newest"


def author_names(authors: object) -> list[str]:
    """Names from a list or from an arXiv-style ``and`` / comma string."""
    if isinstance(authors, str):
        separator = " and " if " and " in authors else ","
        return [name.strip() for name in authors.split(separator) if name.strip()]
    if isinstance(authors, list):
        return [str(name).strip() for name in authors if str(name).strip()]
    return []


def format_authors(authors: object, limit: int = UI_MAX_AUTHORS_SHOWN) -> str:
    """First ``limit`` names. ``et al.`` is added only when some names are left out."""
    names = author_names(authors)
    if not names or limit <= 0:
        return ""
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        return f"{shown} et al."
    return shown


def split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if not collapsed:
        return []
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(collapsed) if part.strip()]
    return parts or [collapsed]


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 2 and token.lower() not in _STOPWORDS
    }


def best_abstract_sentence(abstract: str, query: str) -> str:
    """Sentence with the largest unique-token overlap with the query.

    Equal overlaps keep the earlier sentence. An empty query keeps the first sentence.
    """
    sentences = split_sentences(abstract)
    if not sentences:
        return ""
    query_tokens = _tokens(query)
    if not query_tokens:
        return sentences[0]

    def score(index: int, sentence: str) -> tuple[int, int]:
        return (len(_tokens(sentence) & query_tokens), -index)

    _index, sentence = max(enumerate(sentences), key=lambda pair: score(pair[0], pair[1]))
    return sentence


def highlight_query_terms(sentence: str, query: str) -> str:
    """Escape ``sentence``, then wrap query tokens in ``<mark>``."""
    escaped = html.escape(sentence)
    terms = sorted(_tokens(query), key=len, reverse=True)
    if not terms:
        return escaped
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(term) for term in terms) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", escaped)


def _bibtex_escape(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace("\\", r"\textbackslash{}")
    for char, replacement in (
        ("{", r"\{"),
        ("}", r"\}"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(char, replacement)
    return text


def _eprint(paper_id: object, metadata: dict) -> str:
    for raw in (paper_id, metadata.get("doc_id"), metadata.get("id"), metadata.get("paper_id")):
        clean = clean_arxiv_id(raw)
        if clean:
            return clean
    return ""


def build_bibtex(metadata: dict | None, paper_id: object) -> str:
    """``@misc`` entry with ``eprint``, ``archivePrefix = {arXiv}``, and ``primaryClass``."""
    meta = metadata or {}
    eprint = _eprint(paper_id, meta)
    key = re.sub(r"[^A-Za-z0-9]+", "", eprint) or "paper"
    fields: list[str] = []
    title = str(meta.get("title") or "").strip()
    if title:
        fields.append(f"  title = {{{_bibtex_escape(title)}}}")
    names = author_names(meta.get("authors"))
    if names:
        joined = " and ".join(_bibtex_escape(name) for name in names)
        fields.append(f"  author = {{{joined}}}")
    year = meta.get("year")
    if isinstance(year, bool):
        year = None
    if isinstance(year, int) or (isinstance(year, str) and year.strip().isdigit()):
        fields.append(f"  year = {{{int(year)}}}")
    if eprint:
        fields.append(f"  eprint = {{{_bibtex_escape(eprint)}}}")
    fields.append("  archivePrefix = {arXiv}")
    category = primary_category(meta.get("categories"))
    if category:
        fields.append(f"  primaryClass = {{{_bibtex_escape(category)}}}")
    return f"@misc{{{key},\n" + ",\n".join(fields) + "\n}\n"


def paper_dom_id(paper_id: object) -> str:
    """Anchor shared by a summary chip and that paper's card."""
    clean = clean_arxiv_id(paper_id)
    if not clean:
        return ""
    base = _VERSION_RE.sub("", clean)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-")
    if not slug:
        return ""
    return f"paper-{slug}"


def sort_by_newest(chunks: list) -> list:
    """Order by ``submission_date`` descending. Undated papers keep their relative order last."""

    def sort_key(item: tuple[int, Any]) -> tuple:
        index, chunk = item
        metadata = chunk.get("metadata") if isinstance(chunk, dict) else None
        raw = ""
        if isinstance(metadata, dict):
            raw = str(metadata.get("submission_date") or "").strip()
        if _DATE_RE.fullmatch(raw):
            year, month, day = (int(part) for part in raw.split("-"))
            return (0, -year, -month, -day, index)
        return (1, 0, 0, 0, index)

    return [chunk for _index, chunk in sorted(enumerate(chunks), key=sort_key)]


def order_papers(chunks: list, sort_by: str) -> list:
    """Relevance keeps the returned order. Newest reorders a copy of that list."""
    if sort_by == SORT_NEWEST:
        return sort_by_newest(chunks)
    return list(chunks)


def citation_quotes(answer: dict | None, paper_id: object) -> list[str]:
    if not isinstance(answer, dict):
        return []
    target = paper_dom_id(paper_id)
    if not target:
        return []
    citations = answer.get("citations")
    if not isinstance(citations, list):
        return []
    quotes: list[str] = []
    for item in citations:
        if not isinstance(item, dict):
            continue
        if paper_dom_id(item.get("arxiv_id")) != target:
            continue
        quote = str(item.get("quote") or "").strip()
        if quote:
            quotes.append(quote)
    return quotes


def arxiv_pdf_url(paper_id: object) -> str | None:
    clean = clean_arxiv_id(paper_id)
    if not clean:
        return None
    return f"{ARXIV_HOME_URL.rstrip('/')}/pdf/{clean}"


def _esc(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def paper_byline_html(metadata: dict, *, author_limit: int = UI_MAX_AUTHORS_SHOWN) -> str:
    authors = format_authors(metadata.get("authors"), limit=author_limit)
    year = metadata.get("year")
    category = primary_category(metadata.get("categories"))
    pieces: list[str] = []
    if authors:
        pieces.append(html.escape(authors))
    if year not in (None, ""):
        pieces.append(html.escape(str(year)))
    byline = " · ".join(pieces)
    if not category:
        return byline
    color = html.escape(category_color(category), quote=True)
    chip = f'<span class="category-chip" style="border-color:{color};color:{color}">{html.escape(category)}</span>'
    if byline:
        return f"{byline} · {chip}"
    return chip


def figure_thumbnail_html(
    *,
    paper_id: str,
    metadata: dict,
    figure_url: str | None,
    abs_url: str | None,
) -> str:
    """Thumbnail markup only when the figure URL is a safe arXiv image. Otherwise nothing."""
    if not figure_url:
        return ""
    markup = paper_media_html(
        paper_id=paper_id,
        category=primary_category(metadata.get("categories")),
        year=metadata.get("year"),
        figure_url=figure_url,
        abs_url=abs_url,
    )
    if "has-figure" not in markup:
        return ""
    return markup


def _passage_html(text: str, title: str) -> str:
    preview, truncated = passage_preview(text, title)
    if not preview:
        return ""
    if not truncated:
        return f'<div class="chunk-text">{_esc(preview)}</div>'
    full = passage_body(text, title)
    return (
        '<div class="passage-block">'
        f'<div class="chunk-text chunk-preview">{_esc(preview)}</div>'
        '<details class="passage">'
        "<summary>Show passage</summary>"
        f'<div class="chunk-text">{_esc(full)}</div>'
        "</details>"
        "</div>"
    )


def paper_card_html(
    *,
    title: str,
    paper_id: str,
    metadata: dict,
    text: str,
    query: str,
    score_band: str | None,
    figure_url: str | None,
    quotes: list[str],
    abs_url: str | None,
) -> str:
    """One paper card. The best abstract sentence is highlighted and shown before the preview."""
    abstract = passage_body(text, title)
    sentence = best_abstract_sentence(abstract, query)
    sentence_html = f'<p class="best-sentence">{highlight_query_terms(sentence, query)}</p>' if sentence else ""
    preview, truncated = passage_preview(text, title)
    if sentence and not truncated and preview.strip() == sentence.strip():
        passage = ""
    else:
        passage = _passage_html(text, title)
    byline = paper_byline_html(metadata)
    band = score_band if score_band in BAND_COLORS else ""
    band_html = f'<span class="band-label">{html.escape(band)}</span>' if band else ""
    meta = f'<div class="paper-byline">{byline}{band_html}</div>' if byline or band_html else ""
    quote_html = ""
    if quotes:
        blocks = "".join(f'<blockquote class="cited-quote">{_esc(quote)}</blockquote>' for quote in quotes)
        quote_html = f'<div class="cited-badge">Cited in summary</div>{blocks}'
    media = figure_thumbnail_html(
        paper_id=clean_arxiv_id(paper_id) or paper_id,
        metadata=metadata,
        figure_url=figure_url,
        abs_url=abs_url,
    )
    dom_id = paper_dom_id(paper_id)
    id_attr = f' id="{html.escape(dom_id, quote=True)}"' if dom_id else ""
    band_color = BAND_COLORS.get(band, "#334155")
    return (
        f'<article class="paper-card"{id_attr} style="--band:{band_color}">'
        f"{media}"
        '<div class="paper-body">'
        f"<h3>{_esc(title)}</h3>"
        f"{meta}"
        f"{sentence_html}"
        f"{quote_html}"
        f"{passage}"
        "</div></article>"
    )
