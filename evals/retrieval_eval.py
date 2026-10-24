import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

# Ensure project root is importable when running as a module/script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.settings import (
    CITATION_GROUNDEDNESS_MIN,
    DEFAULT_TOP_K,
    EVAL_SUPPORT_THRESHOLD,
    VECTOR_STORE_DIR,
)


@dataclass
class EvalExample:
    query: str
    relevant_chunk_ids: List[str]
    top_k: int = DEFAULT_TOP_K


def _normalize_chunk_id(value: Any) -> str:
    return str(value).strip()


def load_eval_set(path: Path) -> List[EvalExample]:
    examples: List[EvalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

            query = str(payload.get("query", "")).strip()
            relevant_ids = payload.get("relevant_chunk_ids", [])
            top_k = int(payload.get("top_k", DEFAULT_TOP_K))

            if not query:
                raise ValueError(f"Missing query at line {line_number}")
            if not isinstance(relevant_ids, list) or not relevant_ids:
                raise ValueError(f"relevant_chunk_ids must be a non-empty list at line {line_number}")

            examples.append(
                EvalExample(
                    query=query,
                    relevant_chunk_ids=[_normalize_chunk_id(item) for item in relevant_ids],
                    top_k=max(top_k, 1),
                )
            )

    if not examples:
        raise ValueError("Eval set is empty.")
    return examples


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    relevant = set(relevant_ids)
    for idx, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / idx
    return 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_top_k.intersection(relevant)) / float(len(relevant))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def answer_support_ratio(answer: str, context_chunks: Sequence[Dict[str, Any]]) -> float:
    """
    Heuristic groundedness score:
    fraction of answer content tokens that appear in retrieved context text.
    """
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_text = " ".join(str(chunk.get("text", "")) for chunk in context_chunks)
    context_tokens = set(_tokenize(context_text))
    if not context_tokens:
        return 0.0
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / float(len(answer_tokens))


def _cited_chunk_text(citation: Dict[str, Any], context_chunks: Sequence[Dict[str, Any]]) -> str:
    """
    Text of the chunk a citation points at.

    Prefer the chunk id, then the arXiv id. Fall back to the prompt's
    [Document N] position when those fields are absent.
    """
    chunk_id = str(citation.get("chunk_id") or "").strip()
    arxiv_id = str(citation.get("arxiv_id") or "").strip()
    if chunk_id or arxiv_id:
        for chunk in context_chunks:
            metadata = chunk.get("metadata") or {}
            candidate_chunk = str(chunk.get("id") or metadata.get("chunk_id") or "")
            candidate_paper = str(metadata.get("doc_id") or "")
            if chunk_id and candidate_chunk == chunk_id:
                return str(chunk.get("text", ""))
            if not chunk_id and candidate_paper == arxiv_id:
                return str(chunk.get("text", ""))
        return ""

    doc_id = citation.get("doc_id")
    if isinstance(doc_id, int) and 1 <= doc_id <= len(context_chunks):
        return str(context_chunks[doc_id - 1].get("text", ""))
    return ""


def citation_groundedness(citations: Sequence[Dict[str, Any]], context_chunks: Sequence[Dict[str, Any]]) -> float:
    """
    Fraction of citations whose quoted text appears in the cited document.
    """
    if not citations:
        return 0.0
    valid = 0
    total = 0
    for citation in citations:
        total += 1
        quote = str(citation.get("quote", "")).strip()
        if not quote:
            continue
        doc_text = _cited_chunk_text(citation, context_chunks)
        if doc_text and quote.lower() in doc_text.lower():
            valid += 1
    return valid / float(total) if total > 0 else 0.0


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def run_eval(
    eval_set_path: Path,
    vector_store_dir: str,
    default_top_k: int,
    run_groundedness: bool,
    rerank: bool,
    support_threshold: float,
    citation_threshold: float,
) -> Dict[str, Any]:
    from embeddings import EmbeddingModel
    from llm import LLMInterface
    from reranker import CrossEncoderReranker
    from vector_store import VectorStore

    examples = load_eval_set(eval_set_path)
    embedding_model = EmbeddingModel()
    embedding_model.load_model()
    vector_store = VectorStore(persist_directory=vector_store_dir)
    vector_store.initialize()
    reranker = CrossEncoderReranker() if rerank else None
    llm_interface: Optional[LLMInterface] = LLMInterface() if run_groundedness else None

    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    support_scores: List[float] = []
    citation_scores: List[float] = []
    grounded_passes = 0
    grounded_total = 0
    per_query_rows: List[Dict[str, Any]] = []

    for sample in examples:
        top_k = sample.top_k or default_top_k
        query_embedding = embedding_model.embed_query(sample.query)
        retrieved = vector_store.query(
            query_embedding=query_embedding,
            query_text=sample.query,
            n_results=top_k,
            filters=None,
        )
        if reranker is not None:
            retrieved = reranker.rerank(sample.query, retrieved, final_top_k=top_k)

        retrieved_ids = [_normalize_chunk_id(chunk.get("id")) for chunk in retrieved]
        recall = recall_at_k(retrieved_ids, sample.relevant_chunk_ids, top_k)
        rr = reciprocal_rank(retrieved_ids, sample.relevant_chunk_ids)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

        row: Dict[str, Any] = {
            "query": sample.query,
            "top_k": top_k,
            "recall_at_k": recall,
            "reciprocal_rank": rr,
            "retrieved_ids": retrieved_ids,
            "relevant_chunk_ids": sample.relevant_chunk_ids,
        }

        if run_groundedness and llm_interface is not None:
            grounded_total += 1
            response_payload = llm_interface.generate_response(sample.query, retrieved)
            answer = str(response_payload.get("answer", ""))
            citations = response_payload.get("citations", [])
            support = answer_support_ratio(answer, retrieved)
            citation_score = citation_groundedness(citations, retrieved)
            support_scores.append(support)
            citation_scores.append(citation_score)
            is_grounded = support >= support_threshold and citation_score >= citation_threshold
            if is_grounded:
                grounded_passes += 1
            row.update(
                {
                    "answer_support_ratio": support,
                    "citation_groundedness": citation_score,
                    "grounded_pass": is_grounded,
                }
            )

        per_query_rows.append(row)

    mean_recall = sum(recalls) / len(recalls)
    mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    summary: Dict[str, Any] = {
        "num_queries": len(examples),
        "metrics": {
            f"recall@{default_top_k}": mean_recall,
            "mrr": mean_mrr,
        },
        "per_query": per_query_rows,
    }

    if run_groundedness:
        support_avg = sum(support_scores) / len(support_scores) if support_scores else 0.0
        citation_avg = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
        pass_rate = grounded_passes / grounded_total if grounded_total > 0 else 0.0
        summary["metrics"].update(
            {
                "answer_support_ratio_avg": support_avg,
                "citation_groundedness_avg": citation_avg,
                "grounded_pass_rate": pass_rate,
            }
        )

    return summary


def print_summary(result: Dict[str, Any], default_top_k: int, run_groundedness: bool) -> None:
    metrics = result["metrics"]
    print("=== ArxivLens Retrieval Eval ===")
    print(f"Queries: {result['num_queries']}")
    recall_key = f"recall@{default_top_k}"
    print(f"{recall_key}: {_percent(float(metrics[recall_key]))}")
    print(f"MRR: {float(metrics['mrr']):.4f}")
    if run_groundedness:
        print(f"Answer support ratio avg: {_percent(float(metrics['answer_support_ratio_avg']))}")
        print(f"Citation groundedness avg: {_percent(float(metrics['citation_groundedness_avg']))}")
        print(f"Grounded pass rate: {_percent(float(metrics['grounded_pass_rate']))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and groundedness.")
    parser.add_argument("--eval-set", required=True, help="Path to JSONL eval set")
    parser.add_argument(
        "--vector-store-dir",
        default=VECTOR_STORE_DIR,
        help="Vector store directory",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Default top-k for metrics")
    parser.add_argument(
        "--run-groundedness",
        action="store_true",
        help="Run LLM groundedness checks (requires OPENAI_API_KEY).",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable reranker during eval.")
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=EVAL_SUPPORT_THRESHOLD,
        help="Minimum answer support ratio to pass groundedness.",
    )
    parser.add_argument(
        "--citation-threshold",
        type=float,
        default=CITATION_GROUNDEDNESS_MIN,
        help="Minimum citation groundedness to pass.",
    )
    parser.add_argument("--output", default="", help="Optional path to write full JSON results.")
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if not (0.0 <= args.support_threshold <= 1.0):
        raise ValueError("--support-threshold must be between 0 and 1")
    if not (0.0 <= args.citation_threshold <= 1.0):
        raise ValueError("--citation-threshold must be between 0 and 1")

    load_dotenv()

    try:
        result = run_eval(
            eval_set_path=Path(args.eval_set),
            vector_store_dir=args.vector_store_dir,
            default_top_k=args.top_k,
            run_groundedness=args.run_groundedness,
            rerank=args.rerank,
            support_threshold=args.support_threshold,
            citation_threshold=args.citation_threshold,
        )
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install project requirements before running evals."
        ) from exc
    print_summary(result, args.top_k, args.run_groundedness)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"Wrote eval details to {output_path}")


if __name__ == "__main__":
    main()
