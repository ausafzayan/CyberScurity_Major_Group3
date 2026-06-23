"""
nvd_querier.py
==============
Queries the NIST National Vulnerability Database (NVD) REST API to fetch
CVE (Common Vulnerability and Exposure) records relevant to extracted keywords.

This module implements Step 2-2 of the paper's Task 2 pipeline:
  "Query the NVD API with extracted keywords to build a vulnerability dataset."

Key concepts:
  NVD  : National Vulnerability Database — U.S. government repository of
         vulnerability management data, maintained by NIST.
  CVE  : Common Vulnerabilities and Exposures — unique identifiers for
         specific vulnerabilities (format: CVE-YYYY-NNNNN).
  CVSS : Common Vulnerability Scoring System — numeric severity score 0–10.
  CWE  : Common Weakness Enumeration — weakness category code.
  JSONL: JSON Lines format — one JSON object per line. Enables stream
         processing and isolates parse errors to individual records.

Libraries used:
  - requests : HTTP client for REST API calls
  - json     : serialise/deserialise CVE records to JSONL
  - time     : respect NVD rate-limit sleep intervals
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import requests  # HTTP client — sends GET requests to the NVD REST endpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── NVD REST API endpoint ─────────────────────────────────────────────────────
NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# Rate limits (requests per 30 seconds):
#   Without API key: 5 requests / 30 s  → sleep 6 s between calls
#   With NVD_API_KEY env var: 50 requests / 30 s → sleep 0.6 s
SLEEP_WITH_KEY    = 0.6   # seconds between requests when API key present
SLEEP_WITHOUT_KEY = 6.0   # seconds between requests without API key


class NVDQuerier:
    """
    Fetches CVE records from the NVD API for a list of keywords.

    Each keyword is used as a full-text search term against NVD.
    Results are deduplicated by CVE ID across all keywords.

    Usage:
        querier = NVDQuerier()                              # reads NVD_API_KEY from env
        cves    = querier.build_vulnerability_dataset(keywords)
        NVDQuerier.save_as_jsonl(cves, "cves.jsonl")
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialise the querier.

        Args:
            api_key : NVD API key (optional). If None, reads from the
                      NVD_API_KEY environment variable. Without a key,
                      requests are rate-limited to 5 per 30 seconds.
        """
        # Prefer explicit argument; fall back to environment variable
        self.api_key = api_key or os.getenv("NVD_API_KEY")

        # Include the API key in every request header if available
        self.headers: Dict[str, str] = {}
        if self.api_key:
            self.headers["apiKey"] = self.api_key
            logger.info("NVD API key found — using elevated rate limit (50 req/30 s)")
        else:
            logger.warning(
                "No NVD_API_KEY found — rate limited to 5 req/30 s. "
                "Get a free key at: https://nvd.nist.gov/developers/request-an-api-key"
            )

    # ── Single-keyword query ──────────────────────────────────────────────────

    def query_cves_for_keyword(
        self,
        keyword: str,
        max_results: int = 100,
    ) -> List[Dict]:
        """
        Fetch CVE records matching a single keyword from the NVD API.

        The NVD /cves/2.0 endpoint performs full-text search across CVE
        descriptions, titles, and reference data.

        Args:
            keyword     : search term (e.g. "diffie-hellman")
            max_results : maximum CVEs to return per keyword (cap: 2000)

        Returns:
            List of dicts, each containing:
              cve_id      : "CVE-YYYY-NNNNN"
              description : English description of the vulnerability
              cvss_score  : CVSS base score (float 0–10) or None
              weaknesses  : list of CWE identifiers
              keyword     : the search term that found this CVE
        """
        # Build query parameters for the NVD REST API
        params = {
            "keywordSearch":  keyword,
            "resultsPerPage": min(max_results, 2000),  # NVD hard limit: 2000
            "startIndex":     0,
        }

        try:
            # Send GET request to NVD API with a 30-second timeout
            response = requests.get(
                NVD_BASE_URL,
                params=params,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()  # raise HTTPError for 4xx/5xx codes
            data = response.json()

        except requests.RequestException as exc:
            # Log the failure but continue — one failed keyword is not fatal
            logger.warning("NVD query failed for '%s': %s", keyword, exc)
            return []

        cves: List[Dict] = []
        for item in data.get("vulnerabilities", []):
            cve_obj = item.get("cve", {})

            # Extract CVE identifier (e.g. "CVE-2021-44228")
            cve_id = cve_obj.get("id", "")

            # Extract the English-language description
            desc = next(
                (d["value"] for d in cve_obj.get("descriptions", [])
                 if d.get("lang") == "en"),
                "",
            )

            # Extract CVSS base score — prefer v3.1 → v3.0 → v2
            cvss_score: Optional[float] = None
            for version_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                metrics = cve_obj.get("metrics", {}).get(version_key)
                if metrics:
                    cvss_score = metrics[0].get("cvssData", {}).get("baseScore")
                    break  # use the highest available CVSS version

            # Extract CWE weakness identifiers (e.g. "CWE-287")
            weaknesses = [
                w.get("description", [{}])[0].get("value", "")
                for w in cve_obj.get("weaknesses", [])
            ]

            cves.append({
                "cve_id":      cve_id,
                "description": desc,
                "cvss_score":  cvss_score,
                "weaknesses":  weaknesses,
                "keyword":     keyword,   # track which keyword found this CVE
            })

        logger.info("  Keyword '%s': found %d CVEs", keyword, len(cves))

        # Respect NVD rate limit to avoid HTTP 429 (Too Many Requests)
        sleep_time = SLEEP_WITH_KEY if self.api_key else SLEEP_WITHOUT_KEY
        time.sleep(sleep_time)

        return cves

    # ── Multi-keyword dataset builder ─────────────────────────────────────────

    def build_vulnerability_dataset(self, keywords: List[str]) -> List[Dict]:
        """
        Query NVD for every keyword and return a combined, deduplicated CVE list.

        Deduplication is by CVE ID — the same vulnerability may appear under
        multiple keywords but is only stored once.

        Args:
            keywords : list of keyword strings from KeywordExtractor

        Returns:
            Deduplicated list of CVE dicts ready for vector embedding
        """
        all_cves: List[Dict] = []
        seen_ids: set        = set()  # track CVE IDs already added

        for kw in keywords:
            for cve in self.query_cves_for_keyword(kw):
                if cve["cve_id"] not in seen_ids:
                    seen_ids.add(cve["cve_id"])
                    all_cves.append(cve)

        logger.info("Total unique CVEs collected: %d", len(all_cves))
        return all_cves

    # ── JSONL persistence (static — no instance state needed) ────────────────

    @staticmethod
    def save_as_jsonl(cves: List[Dict], output_path: str) -> None:
        """
        Persist CVE records to disk in JSONL (JSON Lines) format.

        JSONL advantages over a single JSON array:
          - Stream processing: read one CVE at a time without loading the file
          - Error isolation: a corrupt line doesn't break the entire file
          - Append-friendly: new CVEs can be appended without rewriting

        Args:
            cves        : list of CVE dicts from build_vulnerability_dataset
            output_path : destination file path (e.g. "design_cves.jsonl")
        """
        with open(output_path, "w", encoding="utf-8") as fh:
            for cve in cves:
                # Each line is a complete, self-contained JSON object
                fh.write(json.dumps(cve) + "\n")

        logger.info("Saved %d CVEs to %s", len(cves), output_path)
