"""
mq3_mitigation_generator.py
===========================
MQ3 Mitigation Generator — Improvement 1 (part A) over the paper.

The paper only answers MQ1 (What are we working on?) and MQ2 (What can go wrong?).
This module adds MQ3: "What are we going to do about it?"

For each identified threat (ThreatItem from MQ2), the generator:
  1. Builds a structured JSON-forcing LLM prompt using NIST/OWASP references
  2. Retrieves relevant standards context from the vector DB via RAG
  3. Calls the LLM to produce a structured mitigation plan
  4. Parses the JSON output robustly (strips markdown fences)
  5. Passes CVE IDs through the HallucinationGuard for verification

Key concepts:
  CVSS   : Common Vulnerability Scoring System (0–10 severity score)
  NIST   : National Institute of Standards & Technology (SP 800-53 controls)
  OWASP  : Open Web Application Security Project (Top-10, ASVS, WSTG)
  MitigationItem: structured dataclass holding the full mitigation plan

Libraries used:
  - dataclasses (stdlib) : ThreatItem and MitigationItem data containers
  - re (stdlib)          : strip markdown fences from LLM JSON output
  - json (stdlib)        : parse LLM JSON output
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class ThreatItem:
    """
    Represents a single threat identified in MQ2.

    Fields:
      threat_name         : human-readable name (e.g. "JWT Signature Bypass")
      cve_ids             : related CVE identifiers
      affected_components : which system parts are at risk
      stride_category     : STRIDE classification (Spoofing / Tampering / …)
      severity            : CVSS score 0–10 (None if unknown)
      description         : full threat description from MQ2 output
    """
    threat_name:          str
    cve_ids:              List[str]
    affected_components:  List[str]
    stride_category:      str
    severity:             Optional[float]
    description:          str


@dataclass
class MitigationItem:
    """
    Structured mitigation recommendation produced by MQ3.

    Fields:
      threat               : the originating ThreatItem
      immediate_mitigation : action to take right now
      long_term_fix        : architectural / process change
      owasp_reference      : e.g. "A07:2021 — Identification and Authentication Failures"
      nist_reference       : e.g. "NIST SP 800-53 IA-2"
      priority             : CRITICAL | HIGH | MEDIUM | LOW (derived from CVSS)
      verified_cves        : CVE IDs confirmed in NVD
      hallucinated_cves    : CVE IDs fabricated by the LLM
    """
    threat:               ThreatItem
    immediate_mitigation: str
    long_term_fix:        str
    owasp_reference:      str
    nist_reference:       str
    priority:             str
    verified_cves:        List[str] = field(default_factory=list)
    hallucinated_cves:    List[str] = field(default_factory=list)


# ── Prompt template ────────────────────────────────────────────────────────────

MQ3_PROMPT = """\
You are a senior cybersecurity engineer providing mitigation recommendations.

THREAT IDENTIFIED:
- Threat       : {threat_name}
- CVE(s)       : {cve_ids}
- Component(s) : {affected_components}
- STRIDE       : {stride_category}
- Severity     : {severity}/10

RELEVANT STANDARDS CONTEXT:
{context}

Provide a structured mitigation plan as a JSON object with exactly these keys:
{{
  "immediate_mitigation": "specific action to take now (patch, config, rotate keys, etc.)",
  "long_term_fix":        "architectural or process change for permanent remediation",
  "owasp_reference":      "e.g. A07:2021 Identification and Authentication Failures",
  "nist_reference":       "e.g. NIST SP 800-53 IA-5 Authenticator Management",
  "priority":             "CRITICAL | HIGH | MEDIUM | LOW  (based on CVSS score)"
}}
Respond ONLY with valid JSON. No preamble, no markdown fences, no explanation.
"""


class MQ3MitigationGenerator:
    """
    Generates structured MQ3 mitigation plans for each ThreatItem.

    Usage:
        from hallucination_guard.hallucination_guard import HallucinationGuard
        guard = HallucinationGuard()
        gen   = MQ3MitigationGenerator(llm=my_llm, retriever=my_retriever, guard=guard)
        items = gen.generate_all(threats)
    """

    def __init__(self, llm, retriever, guard) -> None:
        """
        Args:
            llm       : any callable LLM object — supports llm.invoke(prompt) or llm(prompt)
            retriever : LangChain retriever from VectorKnowledgeBase.get_retriever()
            guard     : HallucinationGuard instance for CVE verification
        """
        self.llm       = llm
        self.retriever = retriever
        self.guard     = guard

    def generate_mitigation(self, threat: ThreatItem) -> MitigationItem:
        """
        Generate a single MQ3 mitigation plan for one threat.

        Steps:
          1. Retrieve NIST/OWASP context from the vector DB
          2. Build the JSON-forcing prompt
          3. Call the LLM
          4. Parse JSON output (strip markdown fences)
          5. Verify CVE IDs via HallucinationGuard

        Args:
            threat : a ThreatItem from MQ2 output

        Returns:
            MitigationItem with all fields populated
        """
        # Step 1: retrieve relevant standards / CVE context
        retrieved_docs = self.retriever.get_relevant_documents(threat.threat_name)
        context        = "\n".join(doc.page_content for doc in retrieved_docs[:2])

        # Step 2: build prompt with threat details substituted
        prompt = MQ3_PROMPT.format(
            threat_name         = threat.threat_name,
            cve_ids             = ", ".join(threat.cve_ids) if threat.cve_ids else "N/A",
            affected_components = ", ".join(threat.affected_components),
            stride_category     = threat.stride_category,
            severity            = threat.severity if threat.severity is not None else "Unknown",
            context             = context or "No specific context retrieved.",
        )

        # Step 3: call the LLM
        raw_output = self._call_llm(prompt)

        # Step 4: parse JSON (robust — handles markdown fences, extra text)
        mitigation_data = self._parse_json(raw_output)

        # Step 5: verify CVE IDs from both the threat definition and LLM output
        combined_text             = " ".join(threat.cve_ids) + " " + raw_output
        verified, hallucinated    = self.guard.verify_cve_ids(combined_text)

        logger.info(
            "  MQ3 generated for '%s' | priority=%s | hallucinated_cves=%s",
            threat.threat_name,
            mitigation_data.get("priority", "?"),
            hallucinated,
        )

        return MitigationItem(
            threat               = threat,
            immediate_mitigation = mitigation_data.get("immediate_mitigation", "Manual review required."),
            long_term_fix        = mitigation_data.get("long_term_fix",        "Consult security architect."),
            owasp_reference      = mitigation_data.get("owasp_reference",      ""),
            nist_reference       = mitigation_data.get("nist_reference",       ""),
            priority             = mitigation_data.get("priority",             "MEDIUM"),
            verified_cves        = verified,
            hallucinated_cves    = hallucinated,
        )

    def generate_all(self, threats: List[ThreatItem]) -> List[MitigationItem]:
        """
        Generate mitigations for every threat in the list.

        Args:
            threats : list of ThreatItems from MQ2 output

        Returns:
            List of MitigationItems in the same order
        """
        mitigations = []
        for i, threat in enumerate(threats, start=1):
            logger.info("Generating MQ3 mitigation %d/%d: %s", i, len(threats), threat.threat_name)
            mitigations.append(self.generate_mitigation(threat))
        return mitigations

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM and return its raw text response.
        Supports both llm.invoke(prompt) (LangChain) and llm(prompt) (callable).
        """
        try:
            return self.llm.invoke(prompt) if hasattr(self.llm, "invoke") else str(self.llm(prompt))
        except Exception as exc:
            logger.error("LLM call failed: %s — returning empty dict", exc)
            return "{}"

    def _parse_json(self, raw: str) -> dict:
        """
        Robustly parse a JSON object from the LLM's raw output.

        Handles:
          - Markdown code fences (```json ... ```)
          - Preamble text before the JSON object
          - Trailing text after the closing brace
        """
        # Remove markdown fences
        cleaned = re.sub(r"```json|```", "", raw).strip()
        # Find the first {...} block in the output
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse error: %s — returning empty dict", exc)
        return {}
