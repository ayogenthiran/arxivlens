import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.settings import (
    INGEST_JSONL_STREAM_MIN_BYTES,
    INGEST_MODERN_SNAPSHOT_FRACTION,
    INGEST_YEAR_SEEK_WINDOW_BYTES,
)

_CATEGORY_CODE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*(?:\.[A-Za-z0-9\-]+)?")
_MODERN_ARXIV_ID = re.compile(r"^(\d{2})(\d{2})\.")

class DocumentProcessor:
    """
    Handles document loading, chunking, and preprocessing for the RAG pipeline.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: The size of text chunks in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of document dictionaries with text and metadata
        """
        documents = []
        directory = Path(directory_path)
        
        for file_path in directory.glob("**/*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        # Handle standard single-JSON-object files.
                        data = json.load(f)
                        documents.append(self._build_document(data, file_path))
                    except json.JSONDecodeError:
                        # Handle JSONL files like arxiv-metadata-oai-snapshot.json.
                        f.seek(0)
                        line_count = 0
                        for raw_line in f:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                documents.append(self._build_document(data, file_path))
                                line_count += 1
                            except json.JSONDecodeError:
                                continue
                        print(f"Loaded {line_count} JSONL records from {file_path}")
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents from {directory_path}")
        return documents

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
            except Exception as e:
                print(f"Error streaming document {file_path}: {e}")

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
            "submission_year": self._submission_year(paper_id),
            "text": abstract,
            "source": str(file_path),
        }
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        for doc in documents:
            text = self._paper_text(doc)

            # Skip empty documents
            if not text.strip():
                continue

            # A paper is one searchable record: title, then the full abstract.
            # Overlapping slices let one paper fill the result list.
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
            }
            submission_year = doc.get("submission_year")
            if isinstance(submission_year, int):
                metadata["submission_year"] = submission_year
            chunks.append({"text": text, "metadata": metadata})
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    def _paper_text(self, doc: Dict[str, Any]) -> str:
        """Title followed by the full abstract. This is the embedded chunk text."""
        title = str(doc.get("title") or "").strip()
        abstract = str(doc.get("abstract") or doc.get("text") or "").strip()
        if title and abstract:
            return f"{title}\n\n{abstract}"
        return title or abstract
    
    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Create overlapping chunks from text.
        
        Args:
            text: The text to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        end = chunk_size
        
        while start < len(text):
            # Adjust chunk end to not cut words
            if end < len(text):
                # Try to find a good breaking point
                while end > start and end < len(text) and text[end] not in ['.', '!', '?', '\n']:
                    end += 1
                
                # If we couldn't find a good breaking point, just use the original end
                if end - start >= chunk_size * 1.5:
                    end = start + chunk_size
            
            # Extract the chunk
            chunk = text[start:min(end + 1, len(text))].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = end - chunk_overlap
            end = start + chunk_size
        
        return chunks
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Placeholder for embedding chunks.
        This would be implemented by the embedding module.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with embeddings added
        """
        # This is a placeholder - actual embedding happens in the embedding module
        print(f"Prepared {len(chunks)} chunks for embedding")
        return chunks

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
            return [name.strip() for name in re.split(r"\s+and\s+|,\s*", authors) if name.strip()]
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
        """Extract publication year from arXiv metadata fields."""
        date_candidates: List[str] = []
        update_date = data.get("update_date")
        if isinstance(update_date, str):
            date_candidates.append(update_date)

        versions = data.get("versions", [])
        if isinstance(versions, list):
            for version in versions:
                if not isinstance(version, dict):
                    continue
                created = version.get("created")
                if isinstance(created, str):
                    date_candidates.append(created)

        for date_value in date_candidates:
            year = self._parse_year(date_value)
            if year is not None:
                return year
        return None

    def _parse_year(self, date_value: str) -> int | None:
        """Parse a year from known arXiv date formats."""
        if not date_value:
            return None
        date_value = date_value.strip()
        for fmt in ("%Y-%m-%d", "%d %b %Y", "%a, %d %b %Y %H:%M:%S %Z"):
            try:
                return datetime.strptime(date_value, fmt).year
            except ValueError:
                continue

        # Fallback for timestamps like "2007-05-23T17:46:05Z"
        if len(date_value) >= 4 and date_value[:4].isdigit():
            return int(date_value[:4])
        return None
    
    def save_processed_chunks(self, chunks: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save processed chunks to disk.
        
        Args:
            chunks: List of chunk dictionaries
            output_dir: Directory to save processed chunks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "processed_chunks.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)
            
        print(f"Saved {len(chunks)} processed chunks to {output_path}")
