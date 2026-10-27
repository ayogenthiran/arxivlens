"""Paper covers and first-figure lookups for the Streamlit UI.

The arXiv logo file under assets/ is the official single-color white mark.
Do not edit that SVG. Figure bytes stay on arXiv; this module only resolves a URL.
"""

from __future__ import annotations

import html
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from src.settings import (
    ARXIV_ABS_BASE_URL,
    ARXIV_HOME_URL,
    ARXIV_HTML_BASE_URL,
    ARXIV_USER_AGENT,
    UI_FIGURE_FAILURE_TTL_S,
    UI_FIGURE_SUCCESS_TTL_S,
    UI_FIGURE_TIMEOUT_S,
    UI_MAX_FIGURE_LOOKUPS,
    UI_MAX_HTML_CHARS,
    UI_PASSAGE_PREVIEW_CHARS,
)

REQUEST_HEADERS = {
    "User-Agent": ARXIV_USER_AGENT,
}

ALLOWED_FIGURE_HOSTS = frozenset({"arxiv.org", "export.arxiv.org", "ar5iv.labs.arxiv.org"})

# Archive prefixes mapped to stable hues. Subcategories such as cs.CL use the prefix.
_CATEGORY_COLORS = {
    "cs": "#60a5fa",
    "math": "#c4b5fd",
    "stat": "#34d399",
    "q-bio": "#f472b6",
    "q-fin": "#2dd4bf",
    "eess": "#fb923c",
    "econ": "#facc15",
    "astro-ph": "#fbbf24",
    "cond-mat": "#fbbf24",
    "gr-qc": "#fbbf24",
    "hep-ex": "#fbbf24",
    "hep-lat": "#fbbf24",
    "hep-ph": "#fbbf24",
    "hep-th": "#fbbf24",
    "math-ph": "#fbbf24",
    "nlin": "#fbbf24",
    "nucl-ex": "#fbbf24",
    "nucl-th": "#fbbf24",
    "physics": "#fbbf24",
    "quant-ph": "#a78bfa",
}
_DEFAULT_CATEGORY_COLOR = "#94a3b8"
# Subcategories in one archive, such as cs.CL and cs.AI, need their own hues.
_SUBCATEGORY_COLORS = {
    "cs.AI": "#a78bfa",
    "cs.CL": "#60a5fa",
    "cs.CV": "#fb923c",
    "cs.IR": "#2dd4bf",
    "cs.LG": "#34d399",
    "cs.LO": "#f472b6",
    "cs.NE": "#f87171",
    "cs.SE": "#facc15",
    "math.ST": "#c4b5fd",
    "stat.ML": "#34d399",
}
_SUBCATEGORY_PALETTE = (
    "#60a5fa",
    "#a78bfa",
    "#34d399",
    "#fb923c",
    "#2dd4bf",
    "#f472b6",
    "#facc15",
    "#f87171",
    "#38bdf8",
    "#c084fc",
)

_MODERN_ID = r"\d{4}\.\d{4,5}"
_LEGACY_ID = r"[a-z][a-z0-9-]*(?:\.[A-Za-z]{2})?/\d{7}"
_ARXIV_ID_RE = re.compile(rf"^(?:{_MODERN_ID}|{_LEGACY_ID})(?:v\d+)?$", re.IGNORECASE)
_LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "arxiv-logo-one-color-white.svg"

# paper id -> (expires_at, figure_url or None)
_FIGURE_CACHE: dict[str, tuple[float, str | None]] = {}


def category_color(category: str) -> str:
    """Return a palette color for an arXiv category code."""
    code = str(category or "").strip()
    if code in _SUBCATEGORY_COLORS:
        return _SUBCATEGORY_COLORS[code]
    if code in _CATEGORY_COLORS:
        return _CATEGORY_COLORS[code]
    if "." in code:
        digest = sum(ord(char) for char in code)
        return _SUBCATEGORY_PALETTE[digest % len(_SUBCATEGORY_PALETTE)]
    prefix = code.split(".", 1)[0]
    return _CATEGORY_COLORS.get(prefix, _DEFAULT_CATEGORY_COLOR)


def passage_preview(text: str, title: str, limit: int = UI_PASSAGE_PREVIEW_CHARS) -> tuple[str, bool]:
    """Return a short passage and whether the original was longer."""
    body = _without_leading_title(text, title)
    collapsed = re.sub(r"\s+", " ", body).strip()
    if len(collapsed) <= limit:
        return collapsed, False
    cut = collapsed[:limit].rsplit(" ", 1)[0].rstrip()
    if not cut:
        cut = collapsed[:limit].rstrip()
    return cut + "…", True


def passage_body(text: str, title: str) -> str:
    """Full passage with a repeated title line removed."""
    return _without_leading_title(text, title)


def _without_leading_title(text: str, title: str) -> str:
    raw = str(text or "").strip()
    title_clean = re.sub(r"\s+", " ", str(title or "")).strip()
    if not raw or not title_clean:
        return raw
    first_line, _, remainder = raw.partition("\n")
    if re.sub(r"\s+", " ", first_line).strip().lower() == title_clean.lower():
        return remainder.strip()
    collapsed = re.sub(r"\s+", " ", raw)
    if collapsed.lower().startswith(title_clean.lower()):
        return collapsed[len(title_clean):].strip(" :-")
    return raw


def primary_category(categories: object) -> str:
    """First category code from a list or a whitespace-separated string."""
    if isinstance(categories, str):
        values = categories.split()
    elif isinstance(categories, list):
        values = categories
    else:
        return ""
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def clean_arxiv_id(paper_id: object) -> str:
    """Return a canonical arXiv id, or an empty string when it is not one."""
    text = str(paper_id or "").strip()
    text = re.sub(r"(?i)^arxiv:", "", text).strip()
    if _ARXIV_ID_RE.fullmatch(text):
        return text
    return ""


def arxiv_logo_path() -> Path:
    return _LOGO_PATH


class _FigureParser(HTMLParser):
    """Collect figure images, preferring LaTeXML graphics inside a figure."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._figure_depth = 0
        self.candidates: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key.lower(): value or "" for key, value in attrs}
        if tag == "figure":
            self._figure_depth += 1
            return
        if tag != "img":
            return
        src = attr.get("src", "").strip()
        if not src or src.startswith("data:"):
            return
        lowered = src.lower()
        if "logo" in lowered or "favicon" in lowered or "icon" in lowered:
            return
        classes = attr.get("class", "")
        in_figure = self._figure_depth > 0
        is_graphic = "ltx_graphics" in classes
        if is_graphic and in_figure:
            priority = 0
        elif in_figure:
            priority = 1
        elif is_graphic:
            priority = 2
        else:
            return
        self.candidates.append((priority, src))

    def handle_endtag(self, tag: str) -> None:
        if tag == "figure" and self._figure_depth:
            self._figure_depth -= 1


def extract_figure_url(page_html: str, page_url: str) -> str | None:
    """Return the first paper figure URL on an arXiv HTML page."""
    parser = _FigureParser()
    parser.feed(page_html[:UI_MAX_HTML_CHARS])
    parser.close()
    if not parser.candidates:
        return None
    best_priority = min(priority for priority, _src in parser.candidates)
    for priority, src in parser.candidates:
        if priority != best_priority:
            continue
        return _https_arxiv_url(urljoin(page_url, src))
    return None


def _allowed_figure_hosts() -> frozenset[str]:
    """Known arXiv hosts, plus whatever host the configured arXiv base URLs use."""
    hosts = {host.lower() for host in ALLOWED_FIGURE_HOSTS}
    for base in (ARXIV_HTML_BASE_URL, ARXIV_ABS_BASE_URL, ARXIV_HOME_URL):
        hostname = urlparse(base).hostname
        if hostname:
            hosts.add(hostname.lower())
    return frozenset(hosts)


def _https_arxiv_url(url: str) -> str | None:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or hostname not in _allowed_figure_hosts():
        return None
    return url


def fetch_figure_url(paper_id: str, timeout: float = UI_FIGURE_TIMEOUT_S) -> str | None:
    """Fetch the arXiv HTML page and return its first figure URL."""
    clean_id = clean_arxiv_id(paper_id)
    if not clean_id:
        return None
    page_url = f"{ARXIV_HTML_BASE_URL}/{clean_id}"
    try:
        response = requests.get(page_url, headers=REQUEST_HEADERS, timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type:
        return None
    figure_url = extract_figure_url(response.text, response.url)
    if figure_url is None or not _url_is_image(figure_url, timeout):
        return None
    return figure_url


def _url_is_image(url: str, timeout: float) -> bool:
    try:
        with requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, stream=True) as response:
            content_type = response.headers.get("content-type", "").lower()
            return response.status_code == 200 and content_type.startswith("image/")
    except requests.RequestException:
        return False


def _cache_hit(paper_id: str) -> bool:
    cached = _FIGURE_CACHE.get(paper_id)
    return cached is not None and cached[0] > time.time()


def _remember_figure(paper_id: str, figure_url: str | None) -> None:
    ttl = UI_FIGURE_SUCCESS_TTL_S if figure_url else UI_FIGURE_FAILURE_TTL_S
    _FIGURE_CACHE[paper_id] = (time.time() + ttl, figure_url)


def _limited_ids(paper_ids: list[str], limit: int) -> list[str]:
    unique: list[str] = []
    for paper_id in paper_ids:
        clean_id = clean_arxiv_id(paper_id)
        if not clean_id or clean_id in unique:
            continue
        unique.append(clean_id)
        if len(unique) >= limit:
            break
    return unique


def uncached_figure_ids(paper_ids: list[str], limit: int = UI_MAX_FIGURE_LOOKUPS) -> list[str]:
    return [paper_id for paper_id in _limited_ids(paper_ids, limit) if not _cache_hit(paper_id)]


def load_figure_urls(paper_ids: list[str], limit: int = UI_MAX_FIGURE_LOOKUPS) -> dict[str, str | None]:
    """Resolve figure URLs for the first few papers, using a process-local cache."""
    selected = _limited_ids(paper_ids, limit)
    missing = [paper_id for paper_id in selected if not _cache_hit(paper_id)]
    if missing:
        workers = min(len(missing), UI_MAX_FIGURE_LOOKUPS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            found = list(pool.map(fetch_figure_url, missing))
        for paper_id, figure_url in zip(missing, found):
            _remember_figure(paper_id, figure_url)
    resolved: dict[str, str | None] = {}
    for paper_id in selected:
        cached = _FIGURE_CACHE.get(paper_id)
        resolved[paper_id] = cached[1] if cached else None
    return resolved


def paper_media_html(
    *,
    paper_id: str,
    category: str,
    year: object,
    figure_url: str | None,
    abs_url: str | None,
) -> str:
    """Cover card, or the paper's first figure when a safe URL is available."""
    href = html.escape(abs_url, quote=True) if abs_url else ""
    safe_figure = _https_arxiv_url(figure_url or "")
    media_class = "paper-media has-figure" if safe_figure else "paper-media"
    open_tag = (
        f'<a class="{media_class}" href="{href}" target="_blank" rel="noopener noreferrer">'
        if href
        else f'<div class="{media_class}">'
    )
    close_tag = "</a>" if href else "</div>"
    if safe_figure:
        src = html.escape(safe_figure, quote=True)
        label = html.escape(paper_id or "this paper")
        inner = (
            f'<img src="{src}" alt="First figure from {label}">'
            '<span class="media-caption">Figure from the paper on arXiv</span>'
        )
    else:
        inner = _cover_html(paper_id=paper_id, category=category, year=year)
    return f"{open_tag}{inner}{close_tag}"


def _cover_html(*, paper_id: str, category: str, year: object) -> str:
    color = category_color(category)
    label = category or "arXiv"
    year_text = str(year).strip() if year else ""
    year_html = f'<span class="cover-year">{html.escape(year_text)}</span>' if year_text else ""
    id_html = html.escape(paper_id) if paper_id else "arXiv"
    return (
        f'<span class="paper-cover" style="--cover:{color}">'
        f'<span class="cover-cat">{html.escape(label)}</span>'
        f"{year_html}"
        f'<span class="cover-id">{id_html}</span>'
        "</span>"
    )
