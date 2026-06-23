"""
main.py — Paper Approach Entry Point
=====================================
Orchestrates the full threat modeling pipeline replicating the AISCC 2024 paper:
  "Facilitating Threat Modeling by Leveraging Large Language Models"
  (Elsharef, Zeng, Gu — University of Wisconsin-Milwaukee & IBM Research)

This file wires together four decoupled modules:
  1. document_processing.document_processor  → load + chunk PDF
  2. keyword_extraction.keyword_extractor    → BERT keyword extraction (Task 2)
  3. nvd_querier.nvd_querier                 → fetch CVEs from NVD API (Task 2)
  4. vector_db.vector_knowledge_base         → embed + store in ChromaDB
  5. rag_pipeline.rag_pipeline               → Llama 2 RAG query
  6. evaluation.binary_evaluator             → +/- human evaluation

Usage:
    python main.py --task 1 --input design.pdf --question "What components are involved?"
    python main.py --task 2 --input design.pdf --question "How can private keys be exposed?"

    # Use Llama 2 for full paper replication (requires HF token + GPU):
    python main.py --task 1 --input design.pdf --question "..." --model meta-llama/Llama-2-7b-chat-hf

Environment variables (set in .env):
    NVD_API_KEY      → elevated NVD rate limit (optional but recommended)
    HUGGINGFACE_TOKEN → required for Llama 2 gated model access
"""

import argparse
import json
import logging
import os
import sys
from typing import List

# Load .env file into os.environ so all modules see the variables
from dotenv import load_dotenv
load_dotenv()

# ── Import decoupled modules ───────────────────────────────────────────────────
from document_processing.document_processor import DocumentProcessor
from keyword_extraction.keyword_extractor   import KeywordExtractor
from nvd_querier.nvd_querier                import NVDQuerier
from vector_db.vector_knowledge_base        import VectorKnowledgeBase
from rag_pipeline.rag_pipeline              import ThreatModelRAG
from evaluation.binary_evaluator            import BinaryEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — System Understanding (MQ1)
# ═══════════════════════════════════════════════════════════════════════════════

def run_task1(
    pdf_path:   str,
    questions:  List[str],
    model_name: str = "facebook/opt-125m",
) -> List[dict]:
    """
    Execute the Task 1 (MQ1) pipeline end-to-end.

    Pipeline steps:
        PDF → chunk → embed → ChromaDB → RAG query → binary evaluation

    Args:
        pdf_path   : path to the design document PDF
        questions  : list of MQ1 questions (system understanding)
        model_name : HuggingFace model for generation

    Returns:
        List of result dicts (one per question) containing answer + evaluation
    """
    logger.info("=" * 60)
    logger.info("TASK 1: System Understanding (MQ1)")
    logger.info("=" * 60)

    # Step 1-1: Load PDF and split into overlapping text chunks
    processor = DocumentProcessor()
    docs      = processor.load_pdf(pdf_path)
    chunks    = processor.split_documents(docs)

    # Steps 1-2 to 1-4: Embed chunks and persist to ChromaDB
    kb = VectorKnowledgeBase(collection_name="task1_kb", persist_dir="./chroma_task1")
    kb.build_from_documents(chunks)
    retriever = kb.get_retriever(k=4)  # fetch top-4 chunks per query

    # Step 1-5: RAG query for each MQ1 question
    rag       = ThreatModelRAG(retriever, model_name=model_name)
    evaluator = BinaryEvaluator()
    results   = []

    for question in questions:
        response    = rag.answer_question(question, task=1)
        eval_result = evaluator.evaluate_response(question, response["answer"], task=1)
        response["evaluation"] = eval_result["rating"]
        results.append(response)

    evaluator.compute_satisfaction_rate()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Threat Identification (MQ2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_task2(
    pdf_path:   str,
    questions:  List[str],
    model_name: str = "facebook/opt-125m",
) -> List[dict]:
    """
    Execute the Task 2 (MQ2) pipeline end-to-end.

    Pipeline steps:
        PDF → full text → KeyBERT keywords → NVD CVE fetch → JSONL →
        embed CVE descriptions → ChromaDB → RAG query → binary evaluation

    Args:
        pdf_path   : path to the design document PDF
        questions  : list of MQ2 questions (threat identification)
        model_name : HuggingFace model for generation

    Returns:
        List of result dicts (one per question)
    """
    logger.info("=" * 60)
    logger.info("TASK 2: Threat Identification (MQ2)")
    logger.info("=" * 60)

    # Step 2-1a: Load and concatenate all PDF text for keyword extraction
    processor = DocumentProcessor()
    docs      = processor.load_pdf(pdf_path)
    full_text = "\n".join(doc.page_content for doc in docs)

    # Step 2-1b: Extract top-15 discriminative keywords with KeyBERT
    extractor = KeywordExtractor()
    keywords  = extractor.extract_keywords(full_text)

    # Step 2-2: Query NVD API for each keyword → collect CVEs
    querier = NVDQuerier()  # reads NVD_API_KEY from environment
    cves    = querier.build_vulnerability_dataset(keywords)

    # Step 2-3: Save CVEs to JSONL (stream-safe persistence)
    jsonl_path = pdf_path.replace(".pdf", "_cves.jsonl")
    NVDQuerier.save_as_jsonl(cves, jsonl_path)

    # Step 2-4: Embed CVE descriptions and store in ChromaDB
    kb = VectorKnowledgeBase(collection_name="task2_kb", persist_dir="./chroma_task2")
    kb.build_from_cves(cves)
    retriever = kb.get_retriever(k=5)  # fetch top-5 CVE chunks per query

    # RAG query + evaluation
    rag       = ThreatModelRAG(retriever, model_name=model_name)
    evaluator = BinaryEvaluator()
    results   = []

    for question in questions:
        response    = rag.answer_question(question, task=2)
        eval_result = evaluator.evaluate_response(question, response["answer"], task=2)
        response["evaluation"] = eval_result["rating"]
        results.append(response)

    evaluator.compute_satisfaction_rate()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Threat Modeling — Paper Replication (MQ1 & MQ2)"
    )
    parser.add_argument(
        "--task", type=int, choices=[1, 2], required=True,
        help="1 = MQ1 system understanding | 2 = MQ2 threat identification",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the design document PDF",
    )
    parser.add_argument(
        "--question", type=str, nargs="+", required=True,
        help="Threat modeling question(s) to answer",
    )
    parser.add_argument(
        "--model", type=str, default="facebook/opt-125m",
        help=(
            "HuggingFace model name. "
            "Default: facebook/opt-125m (CPU demo). "
            "Full paper: meta-llama/Llama-2-7b-chat-hf (GPU + HF token required)."
        ),
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Output JSON file path (optional; defaults to task<N>_results.json)",
    )
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.isfile(args.input):
        logger.error("File not found: %s", args.input)
        sys.exit(1)

    # Dispatch to the appropriate task pipeline
    if args.task == 1:
        results = run_task1(args.input, args.question, args.model)
    else:
        results = run_task2(args.input, args.question, args.model)

    # Save results to JSON
    out_path = args.output or f"task{args.task}_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results saved to: %s", out_path)


if __name__ == "__main__":
    main()
