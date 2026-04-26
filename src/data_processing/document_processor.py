import os
import json
from datetime import datetime
from typing import List, Dict, Any, Iterator
from pathlib import Path

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

    def iter_documents(self, directory_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream documents from a directory to avoid loading all files in memory.
        Supports both regular JSON and JSONL inputs.
        """
        directory = Path(directory_path)

        for file_path in directory.glob("**/*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        yield self._build_document(data, file_path)
                    except json.JSONDecodeError:
                        f.seek(0)
                        for raw_line in f:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                yield self._build_document(data, file_path)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                print(f"Error streaming document {file_path}: {e}")

    def _build_document(self, data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Normalize a raw source record into the internal document schema."""
        authors = self._normalize_authors(data.get("authors", []))
        categories = self._normalize_categories(data.get("categories", []))
        year = self._extract_year(data)

        return {
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "authors": authors,
            "categories": categories,
            "year": year,
            "text": data.get("abstract", ""),
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
            text = doc.get("text", "")
            
            # Skip empty documents
            if not text.strip():
                continue
                
            # Create chunks with overlap
            doc_chunks = self._create_chunks(text, self.chunk_size, self.chunk_overlap)
            
            # Create chunk objects with metadata
            for i, chunk_text in enumerate(doc_chunks):
                chunk = {
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "chunk_id": f"{doc.get('id', '')}-{i}",
                        "chunk_index": i,
                        "source": doc.get("source", ""),
                        "authors": doc.get("authors", []),
                        "categories": doc.get("categories", []),
                        "year": doc.get("year"),
                    }
                }
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
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

    def _normalize_authors(self, authors: Any) -> List[str]:
        """Normalize authors into a list of clean names."""
        if isinstance(authors, list):
            return [str(author).strip() for author in authors if str(author).strip()]
        if isinstance(authors, str):
            return [name.strip() for name in authors.split(" and ") if name.strip()]
        return []

    def _normalize_categories(self, categories: Any) -> List[str]:
        """Normalize categories into a list of category codes."""
        if isinstance(categories, list):
            return [str(category).strip() for category in categories if str(category).strip()]
        if isinstance(categories, str):
            return [category.strip() for category in categories.split() if category.strip()]
        return []

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
