"""
keyword_extractor.py
====================
Extracts relevant keywords from a design document using KeyBERT,
then post-processes them with NLTK stopword removal and Porter stemming.

This module implements Step 2-1 (Keyword Extraction) of the paper's
Task 2 (MQ2 — Threat Identification) pipeline.

Libraries used:
  - KeyBERT           : BERT-based keyword extraction (cosine similarity)
  - NLTK              : English stopword list + PorterStemmer
  - sentence-transformers : underlying BERT encoder used by KeyBERT
"""

import logging
from typing import List, Set

import nltk
# stopwords: a pre-built set of common English words ("the", "is", etc.)
# PorterStemmer: reduces words to their root form ("encrypts" → "encrypt")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# KeyBERT: extracts key phrases by comparing phrase embeddings to the
# full-document embedding using cosine similarity.
# Internally uses a SentenceTransformer model (all-MiniLM-L6-v2 by default).
from keybert import KeyBERT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paper parameters ──────────────────────────────────────────────────────────
TOP_N_KEYWORDS      = 15         # number of final keywords to return (paper: 15)
KEYPHRASE_NGRAM_RANGE = (1, 2)  # allow single words AND two-word phrases

# Expert-defined security stopwords (Table I in the paper).
# These terms appear so often in security docs that KeyBERT always ranks them
# highly, even though they add no discriminative value for NVD searches.
EXPERT_STOPWORDS: Set[str] = {
    "allow", "attack", "hacker", "security", "cyberattack", "cyber",
    "cybersecurity", "intelligence", "architecture", "secure", "threat",
    "vulnerability", "exploit", "malicious", "adversary", "risk",
    "protection", "defense", "mitigation", "patch", "incident",
}


class KeywordExtractor:
    """
    Extracts discriminative keywords from design document text using KeyBERT.

    Pipeline for each document:
      1. KeyBERT computes BERT embeddings for the document and for candidate
         phrases, then ranks phrases by cosine similarity to the document vector.
      2. NLTK English stopwords + expert security stopwords are removed.
      3. PorterStemmer deduplicates semantically similar words.
      4. Top-N clean keywords are returned for NVD querying.

    Usage:
        extractor = KeywordExtractor()
        keywords  = extractor.extract_keywords(full_text)
    """

    def __init__(self) -> None:
        """
        Download required NLTK data and initialise KeyBERT.
        KeyBERT automatically loads the sentence-transformers/all-MiniLM-L6-v2
        model on first instantiation (downloads ~80 MB if not cached).
        """
        # Download NLTK corpora if not already present on this machine
        nltk.download("stopwords", quiet=True)  # English stopword list
        nltk.download("punkt",     quiet=True)  # tokeniser data

        # Combine NLTK stopwords with expert-defined security stopwords
        self.all_stopwords: Set[str] = (
            set(stopwords.words("english")).union(EXPERT_STOPWORDS)
        )

        # PorterStemmer: a rule-based stemmer that strips common suffixes
        self.stemmer = PorterStemmer()

        logger.info("Initialising KeyBERT (loads sentence-transformers/all-MiniLM-L6-v2)…")
        # KeyBERT() with no arguments uses all-MiniLM-L6-v2 which is fast
        # and accurate for keyword extraction. Runs on CPU.
        self.kw_model = KeyBERT()

    def extract_keywords(
        self,
        text: str,
        top_n: int = TOP_N_KEYWORDS,
    ) -> List[str]:
        """
        Extract the top-N most representative keywords from the given text.

        Args:
            text  : full document text (concatenated pages)
            top_n : maximum number of keywords to return (default: 15)

        Returns:
            List of clean, deduplicated keyword strings
        """
        # Extract 2×top_n candidates so we have room to filter stopwords
        raw_keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,  # uni + bigrams
            stop_words="english",  # KeyBERT's built-in English filter
            top_n=top_n * 2,       # over-extract then prune
        )

        final_keywords: List[str] = []
        seen_stems:     Set[str]  = set()  # prevent duplicate stems

        for keyword, _score in raw_keywords:
            clean = keyword.lower().strip()

            # Filter: skip expert-defined security stopwords
            if clean in self.all_stopwords:
                continue

            # Deduplicate by stem ("encrypting" and "encrypt" → same stem)
            stem = self.stemmer.stem(clean)
            if stem in seen_stems:
                continue  # already have a keyword with this root

            # Accept this keyword
            seen_stems.add(stem)
            final_keywords.append(clean)

            if len(final_keywords) >= top_n:
                break  # reached the requested limit

        logger.info(
            "  Extracted %d keywords: %s …",
            len(final_keywords),
            final_keywords[:5],
        )
        return final_keywords
