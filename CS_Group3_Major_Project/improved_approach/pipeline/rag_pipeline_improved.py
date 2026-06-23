"""
rag_pipeline_improved.py
========================
Improved RAG pipeline with Redis caching, local LLM support (Ollama),
and hallucination validation on every response.

Improvements over paper_approach/rag_pipeline/rag_pipeline.py:
  IMP 4 — Redis cache: identical queries return instantly from cache
  IMP 5 — Local LLM: Ollama runs Llama 3 / Mistral entirely on local hardware
           (no data leaves the machine — critical for sensitive design docs)

Key concepts:
  Ollama   : a tool that serves open-source LLMs (Llama 3, Mistral, CodeLlama)
             locally via a REST API at http://localhost:11434.
             Data stays on-premises — addresses GAP 5 (privacy).
  Caching  : the QueryCache checks for a stored result before calling the LLM.
             On a cache hit, response time drops from 25–50 s → < 0.01 s.

Libraries used:
  - requests (stdlib)  : HTTP calls to Ollama REST API
  - transformers       : HuggingFace pipeline fallback when Ollama unavailable
  - langchain          : RetrievalQA chain, HuggingFacePipeline
"""

import logging
import time
from typing import Dict, List, Optional

import requests  # used for Ollama REST API calls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── System prompts (same four MQs) ────────────────────────────────────────────
MQ_PROMPTS = {
    "MQ1": (
        "You are a security engineer analysing system design documents.\n"
        "Answer ONLY from the provided context. Be concise and specific.\n"
        "If the context is insufficient, say so."
    ),
    "MQ2": (
        "You are a cybersecurity expert identifying threats from vulnerability data.\n"
        "Reference specific CVE IDs. Base your answer on the context provided."
    ),
    "MQ3": (
        "You are a senior security engineer providing NIST/OWASP-aligned mitigations.\n"
        "Provide structured, actionable recommendations. Base them on the context."
    ),
    "MQ4": (
        "You are auditing a threat model for completeness.\n"
        "Evaluate STRIDE coverage, asset identification, and mitigation completeness."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL LLM CLIENT (Ollama — IMP 5: Privacy)
# ══════════════════════════════════════════════════════════════════════════════

class LocalLLMClient:
    """
    Sends prompts to a locally running Ollama server.

    Ollama exposes a REST API so no HuggingFace token or internet access is
    required at inference time. All data stays on the local machine.

    Start Ollama:
        ollama serve
        ollama pull llama3      # download the model once

    Usage:
        llm = LocalLLMClient(model="llama3")
        answer = llm.invoke("What is threat modeling?")
    """

    def __init__(
        self,
        model:    str = "llama3",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self._verify_server()  # warn if Ollama is not running

    def _verify_server(self) -> None:
        """Ping Ollama to check it is running and the requested model is available."""
        try:
            resp   = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s\n"
                    "Pull it with: ollama pull %s",
                    self.model, models, self.model,
                )
        except Exception as exc:
            logger.warning(
                "Ollama not reachable at %s: %s\n"
                "Start it with: ollama serve", self.base_url, exc
            )

    def invoke(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Send a prompt to Ollama and return the generated text.

        Args:
            prompt     : the full prompt string
            max_tokens : maximum tokens to generate

        Returns:
            Generated text string, or error message on failure
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": prompt,
                    "stream": False,    # get the full response at once
                    "options": {"num_predict": max_tokens},
                },
                timeout=120,  # local models can take up to 2 minutes on CPU
            )
            return resp.json().get("response", "")
        except Exception as exc:
            logger.error("Ollama generate failed: %s", exc)
            return f"[LLM error: {exc}]"

    def __call__(self, prompt: str) -> str:
        """Allow the client to be called like a function (LangChain compatibility)."""
        return self.invoke(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class ImprovedRAGPipeline:
    """
    RAG pipeline with caching, local-LLM support, and hallucination validation.

    Usage:
        from cache.query_cache import QueryCache
        from hallucination_guard.hallucination_guard import HallucinationGuard
        from vector_db.vector_knowledge_base import VectorKnowledgeBase

        kb        = VectorKnowledgeBase(); kb.build_from_documents(chunks)
        retriever = kb.get_retriever(k=4)
        cache     = QueryCache()
        guard     = HallucinationGuard()
        llm       = LocalLLMClient(model="llama3")

        pipeline = ImprovedRAGPipeline(retriever, llm, cache, guard)
        result   = pipeline.query("How is data encrypted at rest?", task="MQ1")
    """

    def __init__(
        self,
        retriever,
        llm,
        cache,
        guard,
        k: int = 4,
    ) -> None:
        """
        Args:
            retriever : LangChain retriever from VectorKnowledgeBase
            llm       : LocalLLMClient or any HuggingFacePipeline-like object
            cache     : QueryCache instance
            guard     : HallucinationGuard instance
            k         : number of chunks to retrieve per query
        """
        self.retriever = retriever
        self.llm       = llm
        self.cache     = cache
        self.guard     = guard
        self.k         = k

    def query(self, question: str, task: str = "MQ1") -> Dict:
        """
        Answer a threat modeling question with caching and validation.

        Flow:
          1. Check cache → return immediately on hit
          2. Retrieve top-k relevant chunks from vector DB
          3. Build prompt with system instruction + context + question
          4. Call LLM (local or HuggingFace)
          5. Run HallucinationGuard validation
          6. Store result in cache
          7. Return enriched result dict

        Args:
            question : natural-language threat modeling question
            task     : one of "MQ1" | "MQ2" | "MQ3" | "MQ4"

        Returns:
            Dict with answer, sources, validation report, and timing
        """
        # ── Step 1: Cache check ────────────────────────────────────────────────
        cached = self.cache.get(question)
        if cached:
            logger.info("Cache HIT for: %s…", question[:50])
            cached["from_cache"] = True
            return cached

        # ── Step 2: Retrieve relevant chunks ──────────────────────────────────
        start     = time.time()
        docs      = self.retriever.get_relevant_documents(question)
        context   = "\n\n".join(doc.page_content for doc in docs[: self.k])

        # ── Step 3: Build prompt ───────────────────────────────────────────────
        system    = MQ_PROMPTS.get(task, MQ_PROMPTS["MQ1"])
        prompt    = (
            f"{system}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        # ── Step 4: Call LLM ───────────────────────────────────────────────────
        if hasattr(self.llm, "invoke"):
            answer = self.llm.invoke(prompt)
        else:
            answer = str(self.llm(prompt))
        answer = answer.strip()

        elapsed = time.time() - start

        # ── Step 5: Hallucination validation ──────────────────────────────────
        validation = self.guard.validate_output(answer, context)

        # ── Step 6: Store in cache ────────────────────────────────────────────
        result = {
            "question":          question,
            "answer":            answer,
            "task":              task,
            "sources":           [doc.metadata for doc in docs],
            "word_count":        len(answer.split()),
            "response_time_s":   round(elapsed, 2),
            "from_cache":        False,
            "validation":        validation,
        }
        self.cache.set(question, result)

        logger.info(
            "Query answered in %.1f s | confidence=%s | hallucinated_cves=%s",
            elapsed,
            validation["overall_confidence"],
            validation["hallucinated_cves"],
        )
        return result
