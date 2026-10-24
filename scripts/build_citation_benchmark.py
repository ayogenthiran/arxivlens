"""Build a citation-based retrieval benchmark from an arXiv snapshot.

The eval corpus is a capped slice of cs.CL and cs.LG papers whose first-submission
year is 2023 or later. It is separate from the live app index. Query papers are
the most recent six months of that slice. A reference is relevant when its arXiv
id is in the eval corpus and its first-submission date is earlier than the query
paper's. `--seed` shuffles those candidates. The benchmark keeps the first
`--num-queries` papers that have at least three relevant references, which is a
uniform sample of the qualifying papers. Fetching stops once that sample is full.

Semantic Scholar paper batch (verified against the Graph API docs and a live
response):

    POST https://api.semanticscholar.org/graph/v1/paper/batch
    query fields: externalIds,references.externalIds
    body: {"ids": ["ARXIV:<id>", ...]}   # at most 500 ids

`fields` is a query parameter. The response is a JSON list aligned with `ids`
(`null` when a paper is missing). Each reference is a paper object whose
`externalIds.ArXiv` is the arXiv id. The dedicated references endpoint wraps
that paper in `citedPaper`; this batch endpoint does not. Responses are also
capped at 10 MB and 9999 citations, so a rejected batch is split.

Raw response bodies are cached at data/cache/s2/<sha256>.json. A re-run of the
same batches reads the cache and does not call the network.
"""

import argparse
import calendar
import hashlib
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.document_processor import DocumentProcessor  # noqa: E402
from src.settings import DATA_DIR, S2_API_KEY  # noqa: E402

EVAL_CATEGORIES = frozenset({"cs.CL", "cs.LG"})
MIN_YEAR = 2023
RECENT_MONTHS = 6
MIN_RELEVANT = 3
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "externalIds,references.externalIds"
S2_MAX_IDS_PER_CALL = 500
DEFAULT_S2_BATCH_SIZE = 100
DEFAULT_MAX_DOCS = 50000
DEFAULT_NUM_QUERIES = 500
DEFAULT_SEED = 0
DEFAULT_CORPUS_IDS = PROJECT_ROOT / "evals" / "data" / "eval_corpus_ids.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "data" / "citation_benchmark.jsonl"
DEFAULT_STATS = PROJECT_ROOT / "evals" / "data" / "citation_benchmark_stats.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "s2"
# Authenticated calls are documented at 1 request per second. This builder
# shares the unauthenticated pool unless S2_API_KEY is set. Reference batches
# are large, and a short retry storm keeps the 429 window open, so the first
# wait is already tens of seconds and then doubles.
S2_MIN_INTERVAL_S = 5.0
S2_TIMEOUT_S = 120
S2_MAX_ATTEMPTS = 12
S2_BASE_DELAY_S = 20.0
S2_MAX_DELAY_S = 90.0

_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_URL = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(r"^arxiv:", re.IGNORECASE)

Clock = Callable[[], float]
Sleep = Callable[[float], None]
Poster = Callable[..., Any]


class S2BatchRejected(RuntimeError):
    """Semantic Scholar returned a status that splitting or retrying will not fix."""

    def __init__(self, status_code: int, body: str):
        snippet = " ".join(body.split())[:300]
        super().__init__(f"Semantic Scholar returned {status_code}: {snippet}")
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class ReferenceSummary:
    arxiv_ids: list[str]
    total: int
    on_arxiv: int
    in_corpus: int


@dataclass(frozen=True)
class BuiltQuery:
    record: dict[str, Any]
    n_references: int
    n_references_on_arxiv: int
    n_references_in_corpus: int


def months_before(value: date, months: int) -> date:
    """Return the calendar date `months` before `value`, clamping the day."""
    if months < 0:
        raise ValueError("months must be >= 0")
    month_index = value.year * 12 + (value.month - 1) - months
    year, month0 = divmod(month_index, 12)
    month = month0 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def select_corpus(documents: Iterable[dict[str, Any]], max_docs: int) -> list[dict[str, Any]]:
    """Keep papers in cs.CL or cs.LG whose fixed `year` is at least 2023."""
    if max_docs < 1:
        raise ValueError("max_docs must be >= 1")
    selected: list[dict[str, Any]] = []
    for doc in documents:
        if len(selected) >= max_docs:
            break
        categories = set(doc.get("categories") or [])
        if not categories.intersection(EVAL_CATEGORIES):
            continue
        year = doc.get("year")
        if not isinstance(year, int) or year < MIN_YEAR:
            continue
        paper_id = str(doc.get("id") or "").strip()
        if not paper_id:
            continue
        selected.append(doc)
        if len(selected) % 5000 == 0:
            print(f"[INFO] Eval corpus papers kept: {len(selected)}", flush=True)
    return selected


def load_eval_corpus(data_dir: Path, max_docs: int, processor: DocumentProcessor | None = None) -> list[dict[str, Any]]:
    """Stream the snapshot and stop once `max_docs` eval-corpus papers are kept."""
    processor = processor or DocumentProcessor()
    documents = processor.iter_documents(str(data_dir), min_year=MIN_YEAR)
    return select_corpus(documents, max_docs)


def query_candidates(papers: Sequence[dict[str, Any]], months: int = RECENT_MONTHS) -> list[dict[str, Any]]:
    """Papers in the most recent `months` of the eval corpus, in corpus order."""
    dated: list[tuple[dict[str, Any], date]] = []
    latest: date | None = None
    for paper in papers:
        parsed = as_date(paper.get("submission_date"))
        if parsed is None:
            continue
        dated.append((paper, parsed))
        if latest is None or parsed > latest:
            latest = parsed
    if latest is None:
        return []
    cutoff = months_before(latest, months)
    return [paper for paper, parsed in dated if parsed >= cutoff]


def normalize_arxiv_id(raw: str) -> str:
    value = raw.strip()
    value = _ARXIV_URL.sub("", value)
    value = _ARXIV_PREFIX.sub("", value)
    if value.lower().endswith(".pdf"):
        value = value[:-4]
    value = _ARXIV_VERSION.sub("", value).strip()
    return value


def arxiv_id_from_external_ids(external: Any) -> str | None:
    if not isinstance(external, dict):
        return None
    for key, value in external.items():
        if str(key).lower() != "arxiv" or value is None:
            continue
        normalized = normalize_arxiv_id(str(value))
        return normalized or None
    return None


def _reference_external_ids(reference: Any) -> dict[str, Any] | None:
    if not isinstance(reference, dict):
        return None
    external = reference.get("externalIds")
    if isinstance(external, dict):
        return external
    cited = reference.get("citedPaper")
    if isinstance(cited, dict) and isinstance(cited.get("externalIds"), dict):
        return cited["externalIds"]
    return None


def summarize_references(paper: Any, corpus_ids: set[str]) -> ReferenceSummary:
    """Count every reference entry and collect normalized arXiv ids in response order."""
    if not isinstance(paper, dict):
        return ReferenceSummary([], 0, 0, 0)
    references = paper.get("references")
    if not isinstance(references, list):
        return ReferenceSummary([], 0, 0, 0)
    arxiv_ids: list[str] = []
    on_arxiv = 0
    in_corpus = 0
    for reference in references:
        arxiv_id = arxiv_id_from_external_ids(_reference_external_ids(reference))
        if not arxiv_id:
            continue
        on_arxiv += 1
        arxiv_ids.append(arxiv_id)
        if arxiv_id in corpus_ids:
            in_corpus += 1
    return ReferenceSummary(arxiv_ids, len(references), on_arxiv, in_corpus)


def relevant_paper_ids(
    query_id: str,
    query_date: date,
    reference_ids: Sequence[str],
    corpus_ids: set[str],
    corpus_dates: dict[str, date | None],
) -> list[str]:
    """In-corpus references whose first-submission date is strictly earlier."""
    relevant: list[str] = []
    seen: set[str] = set()
    for ref_id in reference_ids:
        if ref_id in seen or ref_id == query_id:
            continue
        seen.add(ref_id)
        if ref_id not in corpus_ids:
            continue
        ref_date = corpus_dates.get(ref_id)
        if ref_date is None or ref_date >= query_date:
            continue
        relevant.append(ref_id)
    return relevant


def query_text(paper: dict[str, Any]) -> str:
    """Title, a blank line, and the abstract. Same shape as an embedded chunk."""
    title = re.sub(r"\s+", " ", str(paper.get("title") or "")).strip()
    abstract = re.sub(r"\s+", " ", str(paper.get("abstract") or "")).strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def make_query_record(
    paper: dict[str, Any],
    reference_ids: Sequence[str],
    corpus_ids: set[str],
    corpus_dates: dict[str, date | None],
    min_relevant: int = MIN_RELEVANT,
) -> dict[str, Any] | None:
    paper_id = str(paper.get("id") or "").strip()
    query_date = as_date(paper.get("submission_date"))
    text = query_text(paper)
    if not paper_id or query_date is None or not text:
        return None
    relevant = relevant_paper_ids(paper_id, query_date, reference_ids, corpus_ids, corpus_dates)
    if len(relevant) < min_relevant:
        return None
    return {
        "query_id": paper_id,
        "query_type": "paper",
        "query_text": text,
        "query_date": query_date.isoformat(),
        "relevant_paper_ids": relevant,
    }


def build_queries(
    candidates: Sequence[dict[str, Any]],
    papers_by_id: dict[str, Any],
    corpus_ids: set[str],
    corpus_dates: dict[str, date | None],
    min_relevant: int = MIN_RELEVANT,
) -> list[BuiltQuery]:
    built: list[BuiltQuery] = []
    for paper in candidates:
        paper_id = str(paper.get("id") or "")
        summary = summarize_references(papers_by_id.get(paper_id), corpus_ids)
        record = make_query_record(paper, summary.arxiv_ids, corpus_ids, corpus_dates, min_relevant)
        if record is None:
            continue
        built.append(
            BuiltQuery(
                record=record,
                n_references=summary.total,
                n_references_on_arxiv=summary.on_arxiv,
                n_references_in_corpus=summary.in_corpus,
            )
        )
    return built


def seeded_candidate_order(candidates: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Shuffle a copy. The same seed always yields the same order."""
    ordered = list(candidates)
    random.Random(seed).shuffle(ordered)
    return ordered


def collect_queries(
    ordered_candidates: Sequence[dict[str, Any]],
    fetch_ids: Callable[[Sequence[str]], dict[str, Any]],
    corpus_ids: set[str],
    corpus_dates: dict[str, date | None],
    num_queries: int,
    batch_size: int,
    min_relevant: int = MIN_RELEVANT,
) -> tuple[list[BuiltQuery], int]:
    """Fetch in seed order and stop once `num_queries` papers qualify.

    Walking a seeded shuffle until enough papers qualify is a simple random
    sample of every paper that would have qualified.
    """
    if num_queries < 1:
        raise ValueError("num_queries must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    chosen: list[BuiltQuery] = []
    examined = 0
    for start in range(0, len(ordered_candidates), batch_size):
        batch = list(ordered_candidates[start : start + batch_size])
        fetched = fetch_ids([str(paper["id"]) for paper in batch])
        chosen.extend(build_queries(batch, fetched, corpus_ids, corpus_dates, min_relevant))
        examined += len(batch)
        if len(chosen) >= num_queries:
            break
    return chosen[:num_queries], examined


def benchmark_stats(queries: Sequence[BuiltQuery]) -> dict[str, Any]:
    n_queries = len(queries)
    relevant_counts = [len(query.record["relevant_paper_ids"]) for query in queries]
    n_references = sum(query.n_references for query in queries)
    n_on_arxiv = sum(query.n_references_on_arxiv for query in queries)
    n_in_corpus = sum(query.n_references_in_corpus for query in queries)
    if n_queries == 0:
        mean_relevant = None
        median_relevant = None
    else:
        mean_relevant = round(float(statistics.mean(relevant_counts)), 4)
        median_relevant = round(float(statistics.median(relevant_counts)), 4)
    return {
        "n_queries": n_queries,
        "mean_relevant_per_query": mean_relevant,
        "median_relevant_per_query": median_relevant,
        "pct_references_on_arxiv": _percent(n_on_arxiv, n_references),
        "pct_references_in_corpus": _percent(n_in_corpus, n_references),
        "n_references": n_references,
        "n_references_on_arxiv": n_on_arxiv,
        "n_references_in_corpus": n_in_corpus,
    }


def _percent(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return round(100.0 * part / whole, 2)


def papers_from_cache(cache_dir: Path) -> dict[str, Any]:
    """Index cached batch bodies by normalized arXiv id."""
    found: dict[str, Any] = {}
    if not cache_dir.is_dir():
        return found
    for path in cache_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for paper in payload:
            if not isinstance(paper, dict):
                continue
            arxiv_id = arxiv_id_from_external_ids(paper.get("externalIds"))
            if arxiv_id and arxiv_id not in found:
                found[arxiv_id] = paper
    return found


def corpus_dates_for(papers: Sequence[dict[str, Any]]) -> dict[str, date | None]:
    return {str(paper["id"]): as_date(paper.get("submission_date")) for paper in papers}


def write_corpus_ids(path: Path, papers: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(str(paper["id"]))
            handle.write("\n")


def write_jsonl(path: Path, queries: Sequence[BuiltQuery]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query.record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _chunks(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _unique(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for paper_id in ids:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered.append(paper_id)
    return ordered


class S2Client:
    """Fetch paper batches, cache the raw body, and back off on 429 and 5xx."""

    def __init__(
        self,
        cache_dir: Path,
        api_key: str = "",
        *,
        batch_size: int = DEFAULT_S2_BATCH_SIZE,
        post: Poster | None = None,
        sleep: Sleep = time.sleep,
        clock: Clock = time.monotonic,
        min_interval: float = S2_MIN_INTERVAL_S,
        max_attempts: int = S2_MAX_ATTEMPTS,
        base_delay: float = S2_BASE_DELAY_S,
        max_delay: float = S2_MAX_DELAY_S,
        timeout: float = S2_TIMEOUT_S,
    ):
        if not 1 <= batch_size <= S2_MAX_IDS_PER_CALL:
            raise ValueError(f"batch_size must be from 1 to {S2_MAX_IDS_PER_CALL}")
        self.cache_dir = cache_dir
        self.api_key = api_key.strip()
        self.batch_size = batch_size
        self._post = post or requests.post
        self._sleep = sleep
        self._clock = clock
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.network_requests = 0
        self.cache_hits = 0
        self._last_request: float | None = None
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_papers(self, arxiv_ids: Sequence[str]) -> dict[str, Any]:
        ordered = _unique(arxiv_ids)
        found: dict[str, Any] = {}
        batches = list(_chunks(ordered, self.batch_size))
        for index, batch in enumerate(batches, start=1):
            print(
                f"[INFO] Semantic Scholar batch {index}/{len(batches)} ({len(batch)} papers)",
                flush=True,
            )
            payloads = self._load_batch(batch)
            for paper_id, payload in zip(batch, payloads, strict=True):
                found[paper_id] = payload
        return found

    def _load_batch(self, arxiv_ids: list[str]) -> list[Any]:
        cached = self._cached_payload(arxiv_ids)
        if cached is not None:
            self.cache_hits += 1
            return cached
        try:
            raw_text = self._post_with_backoff(arxiv_ids)
        except S2BatchRejected as exc:
            if exc.status_code == 400 and len(arxiv_ids) > 1:
                mid = len(arxiv_ids) // 2
                print(
                    f"[WARN] Semantic Scholar rejected a batch of {len(arxiv_ids)}. Splitting.",
                    flush=True,
                )
                return self._load_batch(arxiv_ids[:mid]) + self._load_batch(arxiv_ids[mid:])
            if exc.status_code == 400 and len(arxiv_ids) == 1:
                # A batch of one unrecognized id returns 400 instead of [null].
                # Store the null paper so a re-run does not request it again.
                print(f"[WARN] Semantic Scholar has no paper for ARXIV:{arxiv_ids[0]}.", flush=True)
                self._write_cache(self._cache_path(arxiv_ids), "[null]")
                return [None]
            raise
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list) or len(parsed) != len(arxiv_ids):
            length = len(parsed) if isinstance(parsed, list) else "n/a"
            raise RuntimeError(
                f"Semantic Scholar batch returned {type(parsed).__name__} of length {length}; "
                f"expected a list of {len(arxiv_ids)}."
            )
        self._write_cache(self._cache_path(arxiv_ids), raw_text)
        return parsed

    def _post_with_backoff(self, arxiv_ids: Sequence[str]) -> str:
        s2_ids = [f"ARXIV:{paper_id}" for paper_id in arxiv_ids]
        headers: dict[str, str] = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            self._wait_turn()
            self.network_requests += 1
            try:
                response = self._post(
                    S2_BATCH_URL,
                    params={"fields": S2_FIELDS},
                    json={"ids": s2_ids},
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    raise
                delay = min(self.max_delay, self.base_delay * (2**attempt))
                print(f"[WARN] Semantic Scholar request failed ({exc}). Retrying in {delay:.1f}s.", flush=True)
                self._sleep(delay)
                continue
            status = int(getattr(response, "status_code", 0))
            body = str(getattr(response, "text", "") or "")
            if status == 200:
                return body
            if status == 429 or status >= 500:
                last_error = S2BatchRejected(status, body)
                if attempt + 1 >= self.max_attempts:
                    raise S2BatchRejected(
                        status,
                        body + " Cached batches are reused if you rerun the builder.",
                    )
                delay = self._retry_delay(attempt, response)
                print(f"[WARN] Semantic Scholar returned {status}. Retrying in {delay:.1f}s.", flush=True)
                self._sleep(delay)
                continue
            raise S2BatchRejected(status, body)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Semantic Scholar request failed without a response.")

    def _retry_delay(self, attempt: int, response: Any) -> float:
        delay = min(self.max_delay, self.base_delay * (2**attempt))
        headers = getattr(response, "headers", {}) or {}
        raw = headers.get("Retry-After")
        if raw is None:
            return delay
        try:
            return max(delay, min(float(raw), self.max_delay))
        except (TypeError, ValueError):
            return delay

    def _wait_turn(self) -> None:
        now = self._clock()
        if self._last_request is None:
            self._last_request = now
            return
        elapsed = now - self._last_request
        if elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)
        self._last_request = self._clock()

    def _cache_path(self, arxiv_ids: Sequence[str]) -> Path:
        payload = json.dumps(
            {"fields": S2_FIELDS, "ids": [f"ARXIV:{paper_id}" for paper_id in arxiv_ids]},
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _cached_payload(self, arxiv_ids: Sequence[str]) -> list[Any] | None:
        """Return a cached batch, including one that was split into halves.

        A re-run then reads those files and does not repeat the request that
        Semantic Scholar rejected.
        """
        path = self._cache_path(arxiv_ids)
        if path.is_file():
            return self._read_cache(path, len(arxiv_ids))
        if len(arxiv_ids) <= 1:
            return None
        mid = len(arxiv_ids) // 2
        left = self._cached_payload(arxiv_ids[:mid])
        right = self._cached_payload(arxiv_ids[mid:])
        if left is None or right is None:
            return None
        return left + right

    def _read_cache(self, path: Path, expected: int) -> list[Any]:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cache file {path.name} is not valid JSON. Delete it and retry.") from exc
        if not isinstance(parsed, list) or len(parsed) != expected:
            raise RuntimeError(f"Cache file {path.name} does not match this batch. Delete it and retry.")
        return parsed

    def _write_cache(self, path: Path, raw_text: str) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(raw_text, encoding="utf-8")
        temporary.replace(path)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def build_benchmark(
    *,
    data_dir: Path,
    max_docs: int,
    num_queries: int,
    seed: int,
    corpus_ids_path: Path,
    output_path: Path,
    stats_path: Path,
    cache_dir: Path,
    batch_size: int,
    api_key: str,
    client: S2Client | None = None,
) -> dict[str, Any]:
    print(f"[INFO] Streaming eval corpus from {data_dir} (max_docs={max_docs})", flush=True)
    papers = load_eval_corpus(data_dir, max_docs)
    if not papers:
        raise SystemExit(f"No cs.CL/cs.LG papers with year >= {MIN_YEAR} under {data_dir}.")
    write_corpus_ids(corpus_ids_path, papers)
    print(f"[INFO] Wrote {len(papers)} corpus ids to {corpus_ids_path}", flush=True)

    candidates = query_candidates(papers)
    print(
        f"[INFO] Query candidates in the most recent {RECENT_MONTHS} months: {len(candidates)}",
        flush=True,
    )
    corpus_dates = corpus_dates_for(papers)
    corpus_ids = set(corpus_dates)
    s2 = client or S2Client(cache_dir, api_key=api_key, batch_size=batch_size)
    known = papers_from_cache(cache_dir)

    def fetch_ids(ids: Sequence[str]) -> dict[str, Any]:
        missing = [paper_id for paper_id in ids if paper_id not in known]
        if missing:
            known.update(s2.fetch_papers(missing))
        return {paper_id: known.get(paper_id) for paper_id in ids}

    ordered = seeded_candidate_order(candidates, seed)
    chosen, examined = collect_queries(
        ordered,
        fetch_ids,
        corpus_ids,
        corpus_dates,
        num_queries,
        batch_size,
    )
    print(
        f"[INFO] Examined {examined} of {len(candidates)} candidates. "
        f"Writing {len(chosen)} queries (seed={seed}).",
        flush=True,
    )
    write_jsonl(output_path, chosen)
    stats = benchmark_stats(chosen)
    write_stats(stats_path, stats)
    print(
        f"[INFO] Semantic Scholar network requests: {s2.network_requests}, cache hits: {s2.cache_hits}",
        flush=True,
    )
    print(f"[INFO] Wrote {output_path}", flush=True)
    print(f"[INFO] Wrote {stats_path}", flush=True)
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the citation-based retrieval benchmark.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory containing the arXiv snapshot.")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corpus-ids", default=str(DEFAULT_CORPUS_IDS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--s2-batch-size",
        type=int,
        default=DEFAULT_S2_BATCH_SIZE,
        help=f"Ids per Semantic Scholar call. At most {S2_MAX_IDS_PER_CALL}. "
        "A smaller batch stays under the 9999-citation and 10 MB response caps.",
    )
    args = parser.parse_args(argv)
    if args.max_docs < 1:
        parser.error("--max-docs must be >= 1")
    if args.num_queries < 1:
        parser.error("--num-queries must be >= 1")
    if not 1 <= args.s2_batch_size <= S2_MAX_IDS_PER_CALL:
        parser.error(f"--s2-batch-size must be from 1 to {S2_MAX_IDS_PER_CALL}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_benchmark(
        data_dir=_resolve(args.data_dir),
        max_docs=args.max_docs,
        num_queries=args.num_queries,
        seed=args.seed,
        corpus_ids_path=_resolve(args.corpus_ids),
        output_path=_resolve(args.output),
        stats_path=_resolve(args.stats),
        cache_dir=_resolve(args.cache_dir),
        batch_size=args.s2_batch_size,
        api_key=S2_API_KEY,
    )


if __name__ == "__main__":
    main()
