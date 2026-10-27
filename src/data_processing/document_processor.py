import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.settings import (
    INGEST_JSONL_STREAM_MIN_BYTES,
    INGEST_MODERN_SNAPSHOT_FRACTION,
    INGEST_YEAR_SEEK_WINDOW_BYTES,
)

logger = logging.getLogger(__name__)

_CATEGORY_CODE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*(?:\.[A-Za-z0-9\-]+)?")
_MODERN_ARXIV_ID = re.compile(r"^(\d{2})(\d{2})\.")


def _split_author_string(authors: str) -> List[str]:
    """Split an arXiv author string. ``and`` separates names; a comma can sit inside one name."""
    separator = " and " if " and " in authors else ","
    return [name.strip() for name in authors.split(separator) if name.strip()]


class DocumentProcessor:
    """Load arXiv metadata and build one chunk per paper."""

    def iter_documents(self, directory_path: str, min_year: int = 0) -> Iterator[Dict[str, Any]]:
        """
        Stream documents from a directory without loading the whole snapshot.

        min_year starts a large JSONL snapshot at the first modern arXiv id
        from that submission year, so a corpus of recent papers does not scan
        the older archive first.
        """
        directory = Path(directory_path)

        for file_path in directory.glob("**/*.json"):
            try:
                if file_path.stat().st_size > INGEST_JSONL_STREAM_MIN_BYTES:
                    yield from self._iter_jsonl(file_path, min_year)
                    continue

                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        yield self._build_document(data, file_path)
                    except json.JSONDecodeError:
                        f.seek(0)
                        yield from self._iter_jsonl_handle(f, file_path)
            except Exception as exc:
                logger.error("Error streaming document %s: %s", file_path, exc)

    def _iter_jsonl(self, file_path: Path, min_year: int) -> Iterator[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            if min_year > 0:
                f.seek(self._find_year_offset(f, min_year))
                f.readline()
            yield from self._iter_jsonl_handle(f, file_path)

    def _find_year_offset(self, handle, min_year: int) -> int:
        """Byte offset of the first modern arXiv id submitted in min_year or later."""
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        lo = 0
        hi = int(size * INGEST_MODERN_SNAPSHOT_FRACTION)
        while hi - lo > INGEST_YEAR_SEEK_WINDOW_BYTES:
            mid = (lo + hi) // 2
            record = self._read_jsonl_record_at(handle, mid)
            paper_id = str((record or {}).get("id") or "")
            if self._submitted_on_or_after(paper_id, min_year):
                hi = mid
            else:
                lo = mid
        return lo

    def _read_jsonl_record_at(self, handle, offset: int) -> Dict[str, Any] | None:
        handle.seek(offset)
        handle.readline()
        line = handle.readline()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data
        return None

    def _submitted_on_or_after(self, paper_id: str, min_year: int) -> bool:
        submission_year = self._submission_year(paper_id)
        if submission_year is None:
            # Legacy ids are stored after the modern section.
            return True
        return submission_year >= min_year

    def _submission_year(self, paper_id: str) -> int | None:
        match = _MODERN_ARXIV_ID.match(str(paper_id))
        if not match:
            return None
        return 2000 + int(match.group(1))

    def _iter_jsonl_handle(self, handle, file_path: Path) -> Iterator[Dict[str, Any]]:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield self._build_document(data, file_path)

    def _build_document(self, data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Normalize a raw source record into the internal document schema."""
        authors = self._normalize_authors(
            data.get("authors", []),
            data.get("authors_parsed"),
        )
        categories = self._normalize_categories(data.get("categories", []))
        year = self._extract_year(data)
        submission_date = self._earliest_submission_date(data.get("versions"))
        title = re.sub(r"\s+", " ", str(data.get("title") or "")).strip()
        abstract = re.sub(r"\s+", " ", str(data.get("abstract") or "")).strip()
        paper_id = str(data.get("id") or "")

        return {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "year": year,
            "submission_date": submission_date.isoformat() if submission_date else None,
            "updated_year": self._updated_year(data),
            "submission_year": self._submission_year(paper_id),
            "text": abstract,
            "source": str(file_path),
        }
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """One searchable record per paper: title, then the full abstract."""
        chunks = []
        
        for doc in documents:
            text = self._paper_text(doc)
            if not text.strip():
                continue

            paper_id = str(doc.get("id") or "")
            metadata = {
                "doc_id": paper_id,
                "title": doc.get("title", ""),
                "chunk_id": f"{paper_id}-0",
                "chunk_index": 0,
                "source": doc.get("source", ""),
                "authors": doc.get("authors", []),
                "categories": doc.get("categories", []),
                "year": doc.get("year"),
                "updated_year": doc.get("updated_year"),
            }
            submission_year = doc.get("submission_year")
            if isinstance(submission_year, int):
                metadata["submission_year"] = submission_year
            submission_date = doc.get("submission_date")
            if isinstance(submission_date, str) and submission_date.strip():
                metadata["submission_date"] = submission_date.strip()
            chunks.append({"text": text, "metadata": metadata})
        
        logger.info("Created %s chunks from %s documents", len(chunks), len(documents))
        return chunks

    def _paper_text(self, doc: Dict[str, Any]) -> str:
        """Title followed by the full abstract. This is the embedded chunk text."""
        title = str(doc.get("title") or "").strip()
        abstract = str(doc.get("abstract") or doc.get("text") or "").strip()
        if title and abstract:
            return f"{title}\n\n{abstract}"
        return title or abstract

    def _normalize_authors(self, authors: Any, authors_parsed: Any = None) -> List[str]:
        """Normalize authors into separate clean names."""
        if isinstance(authors_parsed, list) and authors_parsed:
            names: List[str] = []
            for parts in authors_parsed:
                if isinstance(parts, list):
                    last = str(parts[0]).strip() if len(parts) > 0 else ""
                    first = str(parts[1]).strip() if len(parts) > 1 else ""
                    suffix = str(parts[2]).strip() if len(parts) > 2 else ""
                    name = " ".join(piece for piece in (first, last, suffix) if piece)
                else:
                    name = str(parts).strip()
                if name:
                    names.append(name)
            if names:
                return names

        if isinstance(authors, list):
            return [str(author).strip() for author in authors if str(author).strip()]
        if isinstance(authors, str):
            return _split_author_string(authors)
        return []

    def _normalize_categories(self, categories: Any) -> List[str]:
        """Normalize categories into arXiv codes such as cs.CL."""
        if isinstance(categories, list):
            raw = " ".join(str(category) for category in categories)
        elif isinstance(categories, str):
            raw = categories
        else:
            return []

        codes: List[str] = []
        for match in _CATEGORY_CODE.findall(raw):
            if match not in codes:
                codes.append(match)
        return codes

    def _extract_year(self, data: Dict[str, Any]) -> int | None:
        """Year the paper was first submitted, not the latest revision.

        ``update_date`` changes when a paper is revised, so a 2016 paper
        updated in 2024 would pass a 2024 filter. Prefer the earliest
        ``versions`` created date, then the year encoded in a modern arXiv
        id, and use ``update_date`` only when both are missing.
        """
        earliest = self._earliest_version_year(data.get("versions"))
        if earliest is not None:
            return earliest

        submission_year = self._submission_year(str(data.get("id") or ""))
        if submission_year is not None:
            return submission_year

        return self._updated_year(data)

    def _earliest_version_year(self, versions: Any) -> int | None:
        """Year of the earliest parseable ``created`` date. List order is ignored."""
        if not isinstance(versions, list):
            return None
        years: List[int] = []
        for version in versions:
            if not isinstance(version, dict):
                continue
            created = version.get("created")
            if not isinstance(created, str):
                continue
            year = self._parse_year(created)
            if year is not None:
                years.append(year)
        if not years:
            return None
        return min(years)

    def _earliest_submission_date(self, versions: Any) -> date | None:
        """Calendar date of the earliest parseable version ``created`` value.

        This is the first-submission date. ``update_date`` is the latest
        revision and is not a fallback.
        """
        if not isinstance(versions, list):
            return None
        dates: List[date] = []
        for version in versions:
            if not isinstance(version, dict):
                continue
            created = version.get("created")
            if not isinstance(created, str):
                continue
            parsed = self._parse_date(created)
            if parsed is not None:
                dates.append(parsed)
        if not dates:
            return None
        return min(dates)

    def _parse_date(self, date_value: str) -> date | None:
        """Parse a calendar date from known arXiv date formats."""
        if not date_value:
            return None
        date_value = date_value.strip()
        for fmt in ("%Y-%m-%d", "%d %b %Y", "%a, %d %b %Y %H:%M:%S %Z"):
            try:
                return datetime.strptime(date_value, fmt).date()
            except ValueError:
                continue
        if len(date_value) >= 10 and date_value[4] == "-" and date_value[7] == "-":
            try:
                return datetime.strptime(date_value[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    def _updated_year(self, data: Dict[str, Any]) -> int | None:
        """Year of ``update_date``, the latest revision date."""
        update_date = data.get("update_date")
        if not isinstance(update_date, str):
            return None
        return self._parse_year(update_date)

    def _parse_year(self, date_value: str) -> int | None:
        """Parse a year from known arXiv date formats."""
        parsed = self._parse_date(date_value)
        if parsed is not None:
            return parsed.year
        if not date_value:
            return None
        # Fallback for timestamps like "2007-05-23T17:46:05Z" and bare years.
        stripped = date_value.strip()
        if len(stripped) >= 4 and stripped[:4].isdigit():
            return int(stripped[:4])
        return None
