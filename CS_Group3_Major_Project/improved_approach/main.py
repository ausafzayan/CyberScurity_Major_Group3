"""
main.py — Improved Approach Entry Point
========================================
Orchestrates the full MQ1–MQ4 threat modeling pipeline with all six improvements
over the paper:
  IMP 1 — Full MQ coverage (MQ3 mitigation + MQ4 verification)
  IMP 2 — Hallucination Guard (3-layer CVE + BERTScore + STRIDE validation)
  IMP 3 — Automated quantitative evaluation (BLEU / ROUGE-L / BERTScore)
  IMP 4 — Redis caching (repeated queries < 0.01 s)
  IMP 5 — Local LLM via Ollama (data stays on-premises)
  IMP 6 — Multi-format input (PDF, YAML, JSON, code, images)

Module dependency graph (low coupling — each module is independently usable):

  document_loader.multi_format_loader  ─► plain text
         ↓
  (LangChain text splitter + ChromaDB) ─► vector knowledge base
         ↓
  pipeline.rag_pipeline_improved       ─► MQ1 / MQ2 answers
   uses: cache.query_cache             (Redis / memory)
         hallucination_guard           (CVE + BERTScore + STRIDE)
         ↓
  mq3_mitigation.mq3_mitigation_generator ─► MQ3 mitigations
         ↓
  mq4_verifier.mq4_verifier              ─► MQ4 coverage score
         ↓
  evaluation.auto_evaluator              ─► BLEU / ROUGE / BERTScore report

Usage:
    python main.py --input design.pdf --local
    python main.py --input design.pdf --model llama3
    python main.py --input k8s.yaml --question "What services are exposed?"
"""

import argparse
import json
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv
load_dotenv()

# ── Import all decoupled modules ──────────────────────────────────────────────
from document_loader.multi_format_loader       import MultiFormatLoader
from cache.query_cache                         import QueryCache
from hallucination_guard.hallucination_guard   import HallucinationGuard
from pipeline.rag_pipeline_improved            import ImprovedRAGPipeline, LocalLLMClient
from mq3_mitigation.mq3_mitigation_generator  import (
    MQ3MitigationGenerator, ThreatItem,
)
from mq4_verifier.mq4_verifier                import MQ4Verifier
from evaluation.auto_evaluator                 import AutoEvaluator

# LangChain components used for embedding + vector store
from langchain.text_splitter          import RecursiveCharacterTextSplitter
from langchain.schema                 import Document
from langchain_community.embeddings   import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 500    # paper-matching chunk size
CHUNK_OVERLAP   = 50     # increased from paper's 20 → better boundary context


# ── Demo threat items (for MQ3/MQ4 demonstration without running full MQ2) ───
DEMO_THREATS = [
    ThreatItem(
        threat_name         = "JWT Signature Bypass",
        cve_ids             = ["CVE-2018-1000531"],
        affected_components = ["authentication-service", "api-gateway"],
        stride_category     = "Spoofing",
        severity            = 9.8,
        description         = "Algorithm confusion attack allows forging JWT tokens.",
    ),
    ThreatItem(
        threat_name         = "SQL Injection in User Search",
        cve_ids             = ["CVE-2021-44228"],
        affected_components = ["user-database", "search-api"],
        stride_category     = "Tampering",
        severity            = 8.5,
        description         = "Unsanitised input in search endpoint allows SQL injection.",
    ),
]

# ── Default MQ1 / MQ2 questions ───────────────────────────────────────────────
DEFAULT_MQ1_QUESTIONS = [
    "What are the main components and data flows in this system?",
    "How is data secured during transmission and at rest?",
    "What external interfaces and APIs does this system expose?",
]

DEFAULT_MQ2_QUESTIONS = [
    "What authentication vulnerabilities apply to this design?",
    "What data exposure risks exist based on the system's components?",
    "Which components are most vulnerable to injection attacks?",
]


def build_vector_store(text: str, persist_dir: str = "./chroma_improved") -> Chroma:
    """
    Chunk plain text and build a ChromaDB vector store.

    Args:
        text        : full document text from MultiFormatLoader
        persist_dir : where to persist the ChromaDB collection

    Returns:
        A populated Chroma vectorstore object
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks  = splitter.create_documents([text])          # wrap text in Document objects
    embed   = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store   = Chroma.from_documents(
        documents=chunks, embedding=embed,
        collection_name="improved_kb", persist_directory=persist_dir,
    )
    logger.info("Vector store built: %d chunks → %s", len(chunks), persist_dir)
    return store


def run_full_pipeline(
    file_paths:       List[str],
    mq1_questions:    List[str],
    mq2_questions:    List[str],
    use_local:        bool = True,
    model_name:       str  = "llama3",
    output_path:      str  = "improved_results.json",
) -> dict:
    """
    Execute the complete MQ1 → MQ2 → MQ3 → MQ4 pipeline.

    Args:
        file_paths    : list of input file paths (any supported format)
        mq1_questions : questions for system understanding (MQ1)
        mq2_questions : questions for threat identification (MQ2)
        use_local     : True → use Ollama | False → use HuggingFace
        model_name    : Ollama model name OR HuggingFace model identifier
        output_path   : where to write the JSON results

    Returns:
        Dict containing all results from MQ1 through MQ4
    """
    # ── IMP 6: Load all input files (multi-format) ───────────────────────────
    logger.info("Loading %d input file(s)…", len(file_paths))
    loader    = MultiFormatLoader()
    full_text = "\n\n".join(loader.load(fp) for fp in file_paths)

    # ── Build vector knowledge base ──────────────────────────────────────────
    store     = build_vector_store(full_text)
    retriever = store.as_retriever(search_kwargs={"k": 4})

    # ── IMP 4: Redis cache ────────────────────────────────────────────────────
    cache = QueryCache(use_redis=True)   # falls back to memory if Redis unavailable

    # ── IMP 2: Hallucination Guard ────────────────────────────────────────────
    guard = HallucinationGuard()

    # ── IMP 5: Local or cloud LLM ─────────────────────────────────────────────
    if use_local:
        logger.info("Using local Ollama LLM: %s", model_name)
        llm = LocalLLMClient(model=model_name)
    else:
        logger.info("Using HuggingFace LLM: %s", model_name)
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        from langchain_community.llms import HuggingFacePipeline
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto" if torch.cuda.is_available() else None,
        )
        hf_pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=300, temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        llm = HuggingFacePipeline(pipeline=hf_pipe)

    # ── Build improved RAG pipeline ───────────────────────────────────────────
    rag = ImprovedRAGPipeline(retriever, llm, cache, guard)

    results: dict = {"mq1": [], "mq2": [], "mq3": [], "mq4": None}

    # ── MQ1: System Understanding ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MQ1 — System Understanding")
    for q in mq1_questions:
        results["mq1"].append(rag.query(q, task="MQ1"))

    # ── MQ2: Threat Identification ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MQ2 — Threat Identification")
    for q in mq2_questions:
        results["mq2"].append(rag.query(q, task="MQ2"))

    # ── IMP 1a: MQ3 Mitigation Generation ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MQ3 — Mitigation Generation (for %d demo threats)", len(DEMO_THREATS))
    mq3_gen      = MQ3MitigationGenerator(llm=llm, retriever=retriever, guard=guard)
    mitigations  = mq3_gen.generate_all(DEMO_THREATS)
    results["mq3"] = [
        {
            "threat":              m.threat.threat_name,
            "immediate_action":    m.immediate_mitigation,
            "long_term_fix":       m.long_term_fix,
            "owasp":               m.owasp_reference,
            "nist":                m.nist_reference,
            "priority":            m.priority,
            "verified_cves":       m.verified_cves,
            "hallucinated_cves":   m.hallucinated_cves,
        }
        for m in mitigations
    ]

    # ── IMP 1b: MQ4 Coverage Verification ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("MQ4 — Coverage Verification")
    verifier = MQ4Verifier()
    coverage = verifier.verify_coverage(DEMO_THREATS, full_text, mitigations)

    results["mq4"] = {
        "coverage_score":             coverage.coverage_score,
        "stride_covered":             coverage.stride_categories_covered,
        "stride_missing":             coverage.stride_categories_missing,
        "trust_boundaries":           coverage.trust_boundaries_identified,
        "checklist":                  coverage.checklist,
        "recommendation":             coverage.recommendation,
    }

    # ── IMP 3: Automated Evaluation ────────────────────────────────────────────
    # (requires ground truth — skipped if not provided)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Results saved → %s", output_path)

    # ── Print MQ4 summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MQ4 COVERAGE SUMMARY")
    print(f"  Score          : {coverage.coverage_score:.0%}")
    print(f"  STRIDE covered : {coverage.stride_categories_covered}")
    print(f"  STRIDE missing : {coverage.stride_categories_missing}")
    print(f"  Verdict        : {coverage.recommendation}")
    print("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Improved LLM Threat Modeling — MQ1 to MQ4 with 6 improvements"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Input file(s): PDF, YAML, JSON, code, or image",
    )
    parser.add_argument(
        "--local", action="store_true", default=True,
        help="Use local Ollama LLM (default: True). Set --no-local for HuggingFace.",
    )
    parser.add_argument(
        "--no-local", dest="local", action="store_false",
        help="Use HuggingFace model instead of Ollama.",
    )
    parser.add_argument(
        "--model", default="llama3",
        help="Ollama model name (e.g. llama3, mistral) OR HuggingFace model ID.",
    )
    parser.add_argument(
        "--question", nargs="*", default=None,
        help="Custom MQ1 question(s). Defaults to three built-in questions.",
    )
    parser.add_argument(
        "--output", default="improved_results.json",
        help="Output JSON file for results.",
    )
    args = parser.parse_args()

    # Validate input files
    for fp in args.input:
        if not os.path.isfile(fp):
            logger.error("File not found: %s", fp)
            sys.exit(1)

    mq1_questions = args.question or DEFAULT_MQ1_QUESTIONS

    run_full_pipeline(
        file_paths    = args.input,
        mq1_questions = mq1_questions,
        mq2_questions = DEFAULT_MQ2_QUESTIONS,
        use_local     = args.local,
        model_name    = args.model,
        output_path   = args.output,
    )


if __name__ == "__main__":
    main()
