"""
document_processor.py
=====================
Handles loading PDF design documents and splitting them into chunks
suitable for embedding and vector storage.

This corresponds to Steps 1-1 and 2-1 in the paper's workflow diagram:
  - Step 1-1: Load & Extract Text from PDF
  - Step 2-1: Prepare document text for keyword extraction (Task 2)

Libraries used:
  - LangChain PyPDFLoader  : loads multi-page PDF files into Document objects
  - RecursiveCharacterTextSplitter : splits large documents into overlapping chunks
"""

import logging
from typing import List

# LangChain — framework for building LLM applications.
# PyPDFLoader: reads each PDF page and returns a list of LangChain Document objects.
from langchain_community.document_loaders import PyPDFLoader

# RecursiveCharacterTextSplitter: splits text by trying separators in order
# ("\n\n", "\n", " ", "") until chunks fit within chunk_size.
# Overlap ensures context is not lost at chunk boundaries.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Module-level logger — prints timestamped INFO messages to stdout.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paper-matching constants (Table I values from the paper) ──────────────────
CHUNK_SIZE    = 500   # Maximum characters per chunk (paper: 500)
CHUNK_OVERLAP = 20    # Characters of overlap between consecutive chunks (paper: 20)


class DocumentProcessor:
    """
    Loads a PDF design document and splits it into text chunks.

    The paper uses LangChain's RecursiveCharacterTextSplitter with
    chunk_size=500 and chunk_overlap=20 to prepare document text for
    embedding into Pinecone (or ChromaDB in our local demo).

    Usage:
        processor = DocumentProcessor()
        docs   = processor.load_pdf("design.pdf")     # load pages
        chunks = processor.split_documents(docs)       # split into chunks
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        """
        Initialise the splitter with given chunk parameters.

        Args:
            chunk_size    : maximum number of characters per chunk
            chunk_overlap : number of characters shared between adjacent chunks
        """
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries each separator in order:
        # paragraph break → line break → space → single character
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # preference order
        )

    def load_pdf(self, pdf_path: str) -> List:
        """
        Load a PDF file and return a list of LangChain Document objects.

        Each Document corresponds to one PDF page and carries:
          - page_content : the extracted plain text of that page
          - metadata     : {"source": pdf_path, "page": page_number}

        Args:
            pdf_path : filesystem path to the PDF file

        Returns:
            List of LangChain Document objects (one per page)
        """
        logger.info("Loading PDF: %s", pdf_path)

        # PyPDFLoader opens the PDF, reads each page with pypdf,
        # and returns one Document per page.
        loader    = PyPDFLoader(pdf_path)
        documents = loader.load()

        logger.info("  Loaded %d pages from %s", len(documents), pdf_path)
        return documents

    def split_documents(self, documents: List) -> List:
        """
        Split a list of Documents into smaller overlapping chunks.

        Chunks are required because embedding models (and LLMs) have a
        maximum token/character limit. Splitting ensures each piece fits
        within that limit while keeping the overlap for context continuity.

        Args:
            documents : list of LangChain Document objects (from load_pdf)

        Returns:
            List of smaller LangChain Document chunks ready for embedding
        """
        # split_documents respects existing metadata from parent documents
        chunks = self.splitter.split_documents(documents)

        logger.info(
            "  Split into %d chunks (size=%d, overlap=%d)",
            len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks

    def load_and_split(self, pdf_path: str) -> List:
        """
        Convenience method: load a PDF and immediately split it into chunks.

        Args:
            pdf_path : filesystem path to the PDF file

        Returns:
            List of overlapping text chunks ready for embedding
        """
        documents = self.load_pdf(pdf_path)   # Step 1: load pages
        return self.split_documents(documents) # Step 2: split into chunks
