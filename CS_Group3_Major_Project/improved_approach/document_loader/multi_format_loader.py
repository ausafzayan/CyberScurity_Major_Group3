"""
multi_format_loader.py
======================
Multi-format document loader — Improvement 6 over the paper's PDF-only input.

The paper's system only accepted PDF files. This module extends input support to:
  PDF, YAML, JSON, Python/Go/Java/JS code files, Markdown, plain text, and
  architecture diagram images (via OCR).

Key concept:
  OCR (Optical Character Recognition): software that extracts text from
  images/scanned documents. We use pytesseract (Python wrapper for Tesseract,
  an open-source OCR engine by Google).

Libraries used:
  - langchain PyPDFLoader   : PDF text extraction
  - PyYAML (yaml)           : YAML configuration parsing
  - json (stdlib)           : JSON file parsing
  - pytesseract + Pillow    : OCR for architecture diagram images (optional)
  - pathlib (stdlib)        : file extension detection
"""

import json
import logging
from pathlib import Path
from typing import Dict

import yaml  # PyYAML — parses YAML configuration files (k8s, docker-compose, etc.)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Supported file extensions → internal format tags ─────────────────────────
EXTENSION_MAP: Dict[str, str] = {
    ".pdf":  "pdf",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".json": "json",
    ".py":   "code",
    ".js":   "code",
    ".go":   "code",
    ".java": "code",
    ".tf":   "code",   # Terraform infrastructure-as-code
    ".md":   "text",
    ".txt":  "text",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}


class MultiFormatLoader:
    """
    Routes file loading to the correct sub-loader based on file extension.

    Usage:
        loader = MultiFormatLoader()
        text   = loader.load("design.pdf")
        text   = loader.load("k8s-config.yaml")
        text   = loader.load("architecture.png")   # OCR
    """

    def load(self, file_path: str) -> str:
        """
        Load any supported file and return its content as plain text.

        Args:
            file_path : absolute or relative path to the file

        Returns:
            Extracted text suitable for chunking and embedding.
            Images return OCR-extracted text; YAML/JSON return pretty-printed strings.
        """
        path   = Path(file_path)
        suffix = path.suffix.lower()

        logger.info("Loading file: %s  (format: %s)", file_path, suffix)

        # Route to the appropriate loader
        if suffix in IMAGE_EXTENSIONS:
            return self._load_image_ocr(file_path)

        format_tag = EXTENSION_MAP.get(suffix, "text")

        dispatch = {
            "pdf":  self._load_pdf,
            "yaml": self._load_yaml,
            "json": self._load_json,
            "code": self._load_code,
            "text": self._load_text,
        }
        return dispatch[format_tag](file_path)

    # ── Sub-loaders ────────────────────────────────────────────────────────────

    def _load_pdf(self, path: str) -> str:
        """
        Load a PDF using LangChain's PyPDFLoader.
        Returns concatenated plain text of all pages.
        """
        from langchain_community.document_loaders import PyPDFLoader
        docs = PyPDFLoader(path).load()
        return "\n".join(doc.page_content for doc in docs)

    def _load_yaml(self, path: str) -> str:
        """
        Parse a YAML file (Kubernetes manifests, docker-compose, Terraform vars, etc.)
        and convert to an indented JSON string so the LLM can read it naturally.
        """
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)  # safe_load prevents arbitrary code execution
        # json.dumps produces a readable, indented text representation
        return f"YAML Configuration ({path}):\n{json.dumps(data, indent=2)}"

    def _load_json(self, path: str) -> str:
        """
        Parse a JSON specification file and return indented text.
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return f"JSON Configuration ({path}):\n{json.dumps(data, indent=2)}"

    def _load_code(self, path: str) -> str:
        """
        Return source code as plain text, with a file-type label prepended.
        Preserves indentation and comments.
        """
        suffix = Path(path).suffix
        with open(path, encoding="utf-8") as fh:
            code = fh.read()
        return f"Source Code [{suffix}] — {path}:\n{code}"

    def _load_text(self, path: str) -> str:
        """
        Return plain text files (Markdown, .txt) as-is.
        """
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def _load_image_ocr(self, path: str) -> str:
        """
        Extract text from architecture diagram images using Tesseract OCR.

        OCR pipeline:
          Image file → Pillow.Image.open() → pytesseract.image_to_string()
          → plain text

        Falls back gracefully if pytesseract or Pillow is not installed.
        Install: pip install pytesseract Pillow
                 And Tesseract binary: https://tesseract-ocr.github.io/tessdoc/Installation.html

        Args:
            path : path to a PNG, JPG, or SVG image file

        Returns:
            OCR-extracted text, or a placeholder if OCR is unavailable
        """
        try:
            from PIL import Image        # Pillow: opens image files
            import pytesseract          # Python wrapper for Tesseract OCR engine

            img  = Image.open(path)
            text = pytesseract.image_to_string(img)  # runs Tesseract on the image
            logger.info("  OCR extracted %d chars from %s", len(text), path)
            return f"[OCR from image: {path}]\n{text}"

        except ImportError:
            logger.warning(
                "pytesseract or Pillow not installed. "
                "Install with: pip install pytesseract Pillow\n"
                "Also install Tesseract binary: https://tesseract-ocr.github.io/tessdoc/Installation.html"
            )
            return f"[Image file: {path} — install pytesseract for OCR text extraction]"
