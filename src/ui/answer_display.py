"""Render a structured answer as HTML for the Streamlit app."""

import html
from typing import Any

from src.ui.paper_visuals import clean_arxiv_id


def render_answer_html(result: dict[str, Any], *, abs_base_url: str) -> str:
    """Lead sentence, then one row per finding with paper-title links."""
    summary = str(result.get("summary") or "").strip()
    points = _valid_points(result.get("points"))
    if not points:
        text = summary or str(result.get("answer") or "No answer returned.")
        return f'<div class="answer-box"><p class="answer-lead">{_esc(text)}</p></div>'

    lead = f'<p class="answer-lead">{_esc(summary)}</p>' if summary else ""
    items = "".join(
        _point_html(index, point, abs_base_url) for index, point in enumerate(points, start=1)
    )
    return f'<div class="answer-box">{lead}<ol class="answer-points">{items}</ol></div>'


def _valid_points(points: object) -> list[dict[str, Any]]:
    if not isinstance(points, list):
        return []
    valid: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        title = str(point.get("title") or "").strip()
        detail = str(point.get("detail") or "").strip()
        if title or detail:
            valid.append(point)
    return valid


def _point_html(index: int, point: dict[str, Any], abs_base_url: str) -> str:
    title = str(point.get("title") or "").strip()
    detail = str(point.get("detail") or "").strip()
    title_html = f'<div class="point-title">{_esc(title)}</div>' if title else ""
    detail_html = f'<div class="point-detail">{_esc(detail)}</div>' if detail else ""
    chips = _chips_html(point.get("sources"), abs_base_url)
    return (
        "<li>"
        f'<div class="point-index">{index}</div>'
        f'<div class="point-copy">{title_html}{detail_html}{chips}</div>'
        "</li>"
    )


def _chips_html(sources: object, abs_base_url: str) -> str:
    if not isinstance(sources, list):
        return ""
    chips: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        arxiv_id = clean_arxiv_id(source.get("arxiv_id"))
        title = str(source.get("title") or "").strip() or arxiv_id or "Untitled"
        key = arxiv_id or title
        if key in seen:
            continue
        seen.add(key)
        chips.append(_chip_html(title, arxiv_id, abs_base_url))
    if not chips:
        return ""
    return f'<div class="point-sources">{"".join(chips)}</div>'


def _chip_html(title: str, arxiv_id: str, abs_base_url: str) -> str:
    label = _esc(title)
    tooltip = html.escape(title, quote=True)
    if not arxiv_id:
        return f'<span class="source-chip" title="{tooltip}">{label}</span>'
    href = html.escape(f"{abs_base_url.rstrip('/')}/{arxiv_id}", quote=True)
    return (
        f'<a class="source-chip" title="{tooltip}" href="{href}" '
        f'target="_blank" rel="noopener noreferrer">{label}</a>'
    )


def _esc(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")
