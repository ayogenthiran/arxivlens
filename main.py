# main.py

import os
import sys
import json
import time
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from src.data_processing import DocumentProcessor
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.llm import LLMInterface
from src.api.server import APIServer  # Correct import (from server.py)


def _read_checkpoint(checkpoint_path: str) -> int:
    if not os.path.exists(checkpoint_path):
        return 0
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("processed_docs", 0))
    except Exception:
        return 0


def _write_checkpoint(checkpoint_path: str, processed_docs: int) -> None:
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump({"processed_docs": processed_docs}, f)


def _append_chunks_jsonl(output_path: str, chunks: list[dict]) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk))
            f.write("\n")


def _with_retry(operation_name: str, operation, max_attempts: int = 4, base_delay_s: float = 1.0):
    """
    Execute an operation with exponential backoff retries.
    Raises the final exception if all attempts fail.
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts:
                print(f"[ERROR] {operation_name} failed after {attempt} attempts: {exc}")
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            print(
                f"[WARN] {operation_name} failed on attempt {attempt}/{max_attempts}: {exc}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)


def main():
    """
    Main entry point for the ArxivLens application.
    """
    print("[INFO] Starting ArxivLens...")

    # Initialize components
    document_processor = DocumentProcessor()
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(persist_directory=os.getenv("VECTOR_STORE_DIR", "data/vector_store"))
    llm_interface = LLMInterface()

    # Handle command-line arguments
    args = sys.argv[1:]
    
    if "--process-data" in args:
        print("[INFO] Processing documents...")
        
        data_dir = os.getenv("DATA_DIR", "data/raw")
        output_dir = os.getenv("OUTPUT_DIR", "data/processed")
        os.makedirs(output_dir, exist_ok=True)

        batch_docs = int(os.getenv("INGEST_BATCH_DOCS", "250"))
        max_docs = int(os.getenv("MAX_DOCS", "0"))
        retry_max_attempts = int(os.getenv("INGEST_RETRY_MAX_ATTEMPTS", "4"))
        retry_base_delay_s = float(os.getenv("INGEST_RETRY_BASE_DELAY_S", "1.0"))
        checkpoint_path = os.path.join(output_dir, "ingest_checkpoint.json")
        chunks_jsonl_path = os.path.join(output_dir, "processed_chunks.jsonl")
        reset_ingest = os.getenv("RESET_INGEST", "false").lower() == "true"

        if reset_ingest:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if os.path.exists(chunks_jsonl_path):
                os.remove(chunks_jsonl_path)

        resume_from = _read_checkpoint(checkpoint_path)
        print(f"[INFO] Ingestion config -> batch_docs={batch_docs}, max_docs={max_docs or 'ALL'}, resume_from={resume_from}")

        embedding_model.load_model()
        vector_store.initialize()

        processed_docs = 0
        processed_chunks = 0
        current_batch_docs = []
        ingest_started_at = time.perf_counter()
        total_target_docs = max_docs if max_docs > 0 else None

        def _log_progress(prefix: str) -> None:
            elapsed_s = max(time.perf_counter() - ingest_started_at, 1e-6)
            docs_per_min = (processed_docs / elapsed_s) * 60
            chunks_per_min = (processed_chunks / elapsed_s) * 60
            eta = "N/A"
            if total_target_docs:
                remaining_docs = max(total_target_docs - processed_docs, 0)
                if docs_per_min > 0 and remaining_docs > 0:
                    eta_min = remaining_docs / docs_per_min
                    eta = f"{eta_min:.1f} min"
                elif remaining_docs == 0:
                    eta = "0.0 min"

            print(
                f"[INFO] {prefix} -> docs={processed_docs}, chunks={processed_chunks}, "
                f"docs/min={docs_per_min:.1f}, chunks/min={chunks_per_min:.1f}, eta={eta}"
            )

        for idx, doc in enumerate(document_processor.iter_documents(data_dir)):
            if idx < resume_from:
                continue

            if max_docs > 0 and processed_docs >= max_docs:
                break

            current_batch_docs.append(doc)

            if len(current_batch_docs) < batch_docs:
                continue

            chunks = document_processor.chunk_documents(current_batch_docs)
            chunks_with_embeddings = _with_retry(
                "embedding generation",
                lambda: embedding_model.embed_chunks(chunks),
                max_attempts=retry_max_attempts,
                base_delay_s=retry_base_delay_s,
            )
            if chunks_with_embeddings is None:
                raise RuntimeError("Embedding generation returned no chunks")
            embedded_chunks = chunks_with_embeddings
            _append_chunks_jsonl(chunks_jsonl_path, embedded_chunks)
            _with_retry(
                "vector store write",
                lambda: vector_store.add_embeddings(embedded_chunks),
                max_attempts=retry_max_attempts,
                base_delay_s=retry_base_delay_s,
            )

            processed_docs += len(current_batch_docs)
            processed_chunks += len(embedded_chunks)
            _write_checkpoint(checkpoint_path, resume_from + processed_docs)
            _log_progress("Batch done")
            current_batch_docs = []

        if current_batch_docs:
            chunks = document_processor.chunk_documents(current_batch_docs)
            chunks_with_embeddings = _with_retry(
                "embedding generation",
                lambda: embedding_model.embed_chunks(chunks),
                max_attempts=retry_max_attempts,
                base_delay_s=retry_base_delay_s,
            )
            if chunks_with_embeddings is None:
                raise RuntimeError("Embedding generation returned no chunks")
            embedded_chunks = chunks_with_embeddings
            _append_chunks_jsonl(chunks_jsonl_path, embedded_chunks)
            _with_retry(
                "vector store write",
                lambda: vector_store.add_embeddings(embedded_chunks),
                max_attempts=retry_max_attempts,
                base_delay_s=retry_base_delay_s,
            )
            processed_docs += len(current_batch_docs)
            processed_chunks += len(embedded_chunks)
            _write_checkpoint(checkpoint_path, resume_from + processed_docs)
            _log_progress("Final batch done")

        print(f"[INFO] Processed {processed_docs} documents into {processed_chunks} chunks in this run.")
        print(f"[INFO] Checkpoint saved at: {checkpoint_path}")
        print(f"[INFO] Chunk output saved at: {chunks_jsonl_path}")
        return

    # Start API server
    print("[INFO] Starting API server...")
    api_server = APIServer(
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_interface=llm_interface
    )
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    api_server.run(host=host, port=port)

if __name__ == "__main__":
    main()
