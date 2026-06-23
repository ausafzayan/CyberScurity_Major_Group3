"""
rag_pipeline.py
===============
Implements the RAG (Retrieval-Augmented Generation) pipeline used in the paper
to answer threat modeling questions (MQ1 and MQ2).

RAG concept:
  Without RAG, a language model answers from its training data alone, which
  causes hallucinations and misses document-specific details.
  RAG fixes this by:
    1. Retriever: given a question, fetches the top-k most relevant text chunks
       from the vector database (ChromaDB / Pinecone).
    2. Generator: the LLM receives the retrieved chunks as context alongside
       the question, grounding its answer in actual document content.

This module implements Steps 1-5 (Task 1) and the RAG query step (Task 2).

Libraries used:
  - transformers (HuggingFace) : loads Llama 2 / OPT language models
  - langchain                  : RetrievalQA chain combining retriever + LLM
  - torch                      : tensor operations; enables GPU if available
"""

import logging
import time
from typing import Dict

# HuggingFace Transformers: library for pre-trained language models.
# AutoTokenizer       : loads the correct tokeniser for any HF model.
# AutoModelForCausalLM: loads a decoder-only LM (Llama 2, OPT, GPT-2, …).
# pipeline            : wraps tokeniser + model into a text-generation pipeline.
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# torch: PyTorch deep learning framework.
# cuda.is_available(): returns True if an NVIDIA GPU is present — the model
# will be loaded to GPU automatically via device_map="auto".
import torch

# LangChain components:
# HuggingFacePipeline : wraps a HuggingFace pipeline as a LangChain LLM.
# RetrievalQA         : built-in chain that combines retriever + LLM into one call.
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── System prompts (one per task) ─────────────────────────────────────────────
# These are prepended to the user question to guide the LLM's role and behaviour.

TASK1_SYSTEM_PROMPT = (
    "You are a security engineer assistant analysing system design documents.\n"
    "Answer questions about the system's architecture, components, data flows, "
    "and security measures.\n"
    "Base your answer ONLY on the provided context from the design documents.\n"
    "If the context does not contain enough information, say so explicitly.\n"
    "Keep answers concise and specific."
)

TASK2_SYSTEM_PROMPT = (
    "You are a cybersecurity expert analysing vulnerabilities relevant to a system design.\n"
    "Answer questions about potential security threats, CVE vulnerabilities, and security risks.\n"
    "Base your answer on the provided vulnerability context and design information.\n"
    "Reference specific CVE IDs when available. Be concise and specific."
)


class ThreatModelRAG:
    """
    Retrieval-Augmented Generation pipeline for threat modeling questions.

    The paper uses Llama 2 (7B / 13B chat variants) running on a Google Colab T4 GPU.
    For lightweight local demos, we default to facebook/opt-125m (no GPU required).

    Usage:
        rag = ThreatModelRAG(retriever, model_name="facebook/opt-125m")
        result = rag.answer_question("What components are involved?", task=1)
        print(result["answer"])
    """

    def __init__(self, retriever, model_name: str = "facebook/opt-125m") -> None:
        """
        Load the language model and build the RetrievalQA chain.

        Args:
            retriever  : LangChain retriever from VectorKnowledgeBase.get_retriever()
            model_name : HuggingFace model identifier.
                         Paper: "meta-llama/Llama-2-7b-chat-hf" (requires HF token + GPU)
                         Demo:  "facebook/opt-125m" (CPU-compatible, ~250 MB)
        """
        self.retriever  = retriever
        self.model_name = model_name
        self._build_chain()  # load model and assemble RetrievalQA chain

    def _build_chain(self) -> None:
        """
        Load the tokeniser and model from HuggingFace Hub, then wire everything
        into a LangChain RetrievalQA chain.

        Steps:
          1. AutoTokenizer.from_pretrained  → tokeniser
          2. AutoModelForCausalLM.from_pretrained → model (on GPU if available)
          3. pipeline("text-generation", …)  → HuggingFace pipeline
          4. HuggingFacePipeline(pipeline)   → LangChain-compatible LLM
          5. RetrievalQA.from_chain_type(…)  → full RAG chain
        """
        logger.info("Loading LLM: %s", self.model_name)

        # Load tokeniser: converts text ↔ token IDs for this specific model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model weights; use GPU automatically if torch detects one
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype="auto",  # fp16 on GPU, fp32 on CPU
        )

        # text-generation pipeline: given a prompt string, returns generated text
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,    # paper RAG average: ~74 words → 300 tokens is generous
            temperature=0.1,       # near-deterministic: lower = more focused
            do_sample=True,        # required when temperature < 1.0
            pad_token_id=tokenizer.eos_token_id,  # prevents padding warning
        )

        # Wrap the HuggingFace pipeline in LangChain's LLM interface
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # RetrievalQA chain:
        #   chain_type="stuff" → concatenates all retrieved chunks into one prompt
        #   return_source_documents=True → include chunk metadata in the result
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
        )
        logger.info("RAG chain ready.")

    def answer_question(self, question: str, task: int = 1) -> Dict:
        """
        Answer a threat modeling question using the RAG pipeline.

        The full prompt sent to the LLM is:
            [system prompt for task]
            \n\nQuestion: [question]

        The retrieved chunks are injected by RetrievalQA as additional context.

        Args:
            question : natural-language threat modeling question
            task     : 1 → MQ1 system-understanding prompt
                       2 → MQ2 threat-identification prompt

        Returns:
            Dict with keys:
              question       : the original question
              answer         : LLM-generated answer string
              task           : "MQ1" or "MQ2"
              sources        : list of source chunk metadata dicts
              word_count     : number of words in the answer
              response_time_s: seconds taken for the RAG query
        """
        # Select the appropriate system prompt for the task
        system_prompt = TASK1_SYSTEM_PROMPT if task == 1 else TASK2_SYSTEM_PROMPT
        full_query    = f"{system_prompt}\n\nQuestion: {question}"

        start  = time.time()
        result = self.qa_chain.invoke({"query": full_query})  # triggers retrieval + generation
        elapsed = time.time() - start

        # Extract the generated answer text
        answer     = result.get("result", "").strip()
        # Collect metadata from retrieved source documents
        sources    = [doc.metadata for doc in result.get("source_documents", [])]
        word_count = len(answer.split())

        logger.info("  Answered in %.1f s  (%d words)", elapsed, word_count)

        return {
            "question":        question,
            "answer":          answer,
            "task":            f"MQ{task}",
            "sources":         sources,
            "word_count":      word_count,
            "response_time_s": round(elapsed, 2),
        }
