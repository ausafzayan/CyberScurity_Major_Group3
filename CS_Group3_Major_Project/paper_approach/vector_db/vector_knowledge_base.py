"""
vector_knowledge_base.py
========================
Builds and queries a ChromaDB vector database from document chunks
or CVE descriptions. Provides a retriever interface for the RAG pipeline.

This module implements Steps 1-2 through 1-4 (Task 1) and Steps 2-3 to 2-4
(Task 2) of the paper's workflow:
  Task 1: embed document chunks → store in vector DB
  Task 2: embed CVE descriptions → store in a separate vector DB

Key concepts:
  Embedding  : a fixed-length numerical vector representing the semantic
                meaning of a piece of text. Similar texts have similar vectors.
  ChromaDB   : open-source local vector database (replaces cloud Pinecone
                in this demo). Persists embeddings to disk.
  Pinecone   : cloud-hosted vector database used in the paper's original
                implementation (requires API key and internet access).
  Cosine similarity : distance metric used to rank retrieved chunks by how
                      similar their embeddings are to the query embedding.

Libraries used:
  - langchain-community (Chroma, HuggingFaceEmbeddings) : vector DB wrapper
  - sentence-transformers : provides the embedding model
"""

import logging
from typing import Dict, List, Optional

# HuggingFaceEmbeddings: wraps sentence-transformers models as LangChain embedders.
# We use all-MiniLM-L6-v2 — a fast, accurate model that runs on CPU.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma: LangChain wrapper around ChromaDB (local persistent vector store).
# The paper uses Pinecone; ChromaDB is a fully local, open-source alternative.
from langchain_community.vectorstores import Chroma

# LangChain Document schema — wraps text + metadata into a single object.
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Embedding model identifier ────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# This 80 MB model converts text to 384-dimensional vectors.
# It balances speed (CPU-friendly) and accuracy (trained on 1B sentence pairs).


class VectorKnowledgeBase:
    """
    Manages a ChromaDB vector store for semantic document retrieval.

    Supports two modes:
      1. Document mode (Task 1): stores design-document text chunks
      2. CVE mode    (Task 2): stores CVE descriptions for threat identification

    Usage:
        kb = VectorKnowledgeBase(collection_name="task1_kb")
        kb.build_from_documents(chunks)           # embed & store doc chunks
        retriever = kb.get_retriever(k=4)         # top-4 similarity search

        kb2 = VectorKnowledgeBase(collection_name="task2_kb")
        kb2.build_from_cves(cves)                 # embed & store CVE records
        retriever2 = kb2.get_retriever(k=5)
    """

    def __init__(
        self,
        collection_name: str = "threat_model_kb",
        persist_dir: str    = "./chroma_db",
    ) -> None:
        """
        Initialise the embedding model and prepare references.

        Args:
            collection_name : ChromaDB collection identifier (like a table name)
            persist_dir     : filesystem path where ChromaDB stores its data
        """
        self.collection_name = collection_name
        self.persist_dir     = persist_dir
        self.vectorstore: Optional[Chroma] = None  # populated by build_* methods

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        # HuggingFaceEmbeddings downloads and caches the model the first time.
        # Subsequent calls use the local cache (~/.cache/huggingface/).
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ── Task 1: build from document chunks ────────────────────────────────────

    def build_from_documents(self, chunks: List) -> None:
        """
        Embed document chunks and store them in ChromaDB (Task 1 pipeline).

        Process for each chunk:
          text → embedding model → 384-dim float vector → ChromaDB collection

        Args:
            chunks : list of LangChain Document objects from DocumentProcessor
        """
        logger.info("Building vector DB from %d document chunks…", len(chunks))

        # Chroma.from_documents:
        #   1. Calls self.embeddings.embed_documents() on every chunk
        #   2. Inserts (vector, metadata, text) tuples into ChromaDB
        #   3. Persists the collection to disk at persist_dir
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        logger.info("  Vector DB persisted at: %s", self.persist_dir)

    # ── Task 2: build from CVE records ────────────────────────────────────────

    def build_from_cves(self, cves: List[Dict]) -> None:
        """
        Embed CVE descriptions and store them in a separate ChromaDB collection
        (Task 2 pipeline).

        Each CVE description is treated as a document; CVE metadata (ID, score,
        keyword) is stored alongside so it can be returned with retrieved chunks.

        Args:
            cves : list of CVE dicts from NVDQuerier.build_vulnerability_dataset
        """
        # Convert each CVE dict into a LangChain Document
        documents = [
            Document(
                page_content=cve["description"],   # text to embed
                metadata={
                    "cve_id":     cve["cve_id"],
                    "cvss_score": str(cve.get("cvss_score", "")),
                    "keyword":    cve.get("keyword", ""),
                    "weaknesses": ", ".join(cve.get("weaknesses", [])),
                },
            )
            for cve in cves
            if cve.get("description")   # skip CVEs with empty descriptions
        ]

        logger.info("Building CVE vector DB from %d CVE descriptions…", len(documents))

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            # Use a distinct name so Task 1 and Task 2 collections don't collide
            collection_name=self.collection_name + "_cve",
            persist_directory=self.persist_dir + "_cve",
        )

    # ── Retriever interface ────────────────────────────────────────────────────

    def get_retriever(self, k: int = 4):
        """
        Return a LangChain retriever that fetches the top-k most relevant chunks.

        The retriever uses cosine similarity between the query embedding and all
        stored embeddings to rank chunks. The top-k are fed into the LLM as
        context (RAG = Retrieval-Augmented Generation).

        Args:
            k : number of chunks to retrieve per query

        Returns:
            A LangChain BaseRetriever object compatible with RetrievalQA

        Raises:
            RuntimeError : if called before build_from_documents / build_from_cves
        """
        if self.vectorstore is None:
            raise RuntimeError(
                "Vector store not built. "
                "Call build_from_documents() or build_from_cves() first."
            )
        # as_retriever wraps ChromaDB's similarity_search in the LangChain interface
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
