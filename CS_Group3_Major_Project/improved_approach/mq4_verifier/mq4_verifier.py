"""
mq4_verifier.py
===============
MQ4 Coverage Verifier — Improvement 1 (part B) over the paper.

The paper only answers MQ1 and MQ2. This module implements MQ4:
"Did we do a good enough job?"

The verifier produces a scored checklist that automatically measures how
comprehensive the threat model is across eight weighted criteria.

Coverage score calculation:
    score = Σ(weight_i × pass_i) / Σ(weight_i)

Criteria and weights:
  Weight 2 (critical): All assets identified | All STRIDE categories addressed
                        All threats have mitigations | All CVEs NVD-verified
  Weight 1 (standard): Trust boundaries documented | Severity scores assigned
                        Priority levels assigned | OWASP references included

A score ≥ 0.8 is considered "comprehensive and ready for review."

Libraries used:
  - dataclasses (stdlib) : MQ4Coverage structured output
  - re (stdlib)          : regex for trust-boundary detection
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── STRIDE reference set ───────────────────────────────────────────────────────
STRIDE_ALL = [
    "Spoofing",
    "Tampering",
    "Repudiation",
    "Information Disclosure",
    "Denial of Service",
    "Elevation of Privilege",
]

# Regex to detect trust-boundary mentions in design documents
TRUST_BOUNDARY_PATTERN = re.compile(
    r"trust\s*boundary|api\s*gateway|firewall|dmz|network\s*segment|auth\s*service",
    re.IGNORECASE,
)

# Heuristic patterns for counting system assets
ASSET_PATTERNS = [
    r"\bservice\b", r"\bdatabase\b", r"\bapi\b", r"\bserver\b",
    r"\bclient\b",  r"\bcomponent\b", r"\bmodule\b", r"\bendpoint\b",
]

# Checklist weights: 2 = critical check, 1 = standard check
CHECKLIST_WEIGHTS = {
    "All assets identified":             2,
    "All STRIDE categories addressed":   2,
    "All threats have mitigations":      2,
    "All CVEs NVD-verified":             2,
    "Trust boundaries documented":       1,
    "Severity scores assigned":          1,
    "Priority levels assigned":          1,
    "OWASP references included":         1,
}

COVERAGE_PASS_THRESHOLD = 0.8   # ≥ 80% weighted score → "comprehensive"


@dataclass
class MQ4Coverage:
    """
    Coverage verification result produced by MQ4Verifier.

    Fields:
      total_assets               : heuristic count of system assets in design text
      covered_assets             : assets that have at least one associated threat
      stride_categories_covered  : STRIDE categories addressed by identified threats
      stride_categories_missing  : STRIDE categories with no coverage
      trust_boundaries_identified: count of trust-boundary mentions in design text
      coverage_score             : weighted score 0.0 – 1.0
      checklist                  : Dict[criterion, bool]
      recommendation             : human-readable verdict
    """
    total_assets:               int
    covered_assets:             int
    stride_categories_covered:  List[str]
    stride_categories_missing:  List[str]
    trust_boundaries_identified: int
    coverage_score:             float
    checklist:                  Dict[str, bool] = field(default_factory=dict)
    recommendation:             str = ""


class MQ4Verifier:
    """
    Automatically generates and scores an MQ4 completeness checklist.

    Usage:
        verifier = MQ4Verifier()
        coverage = verifier.verify_coverage(threats, design_text, mitigations)
        print(f"Coverage score: {coverage.coverage_score:.0%}")
        print(coverage.recommendation)
    """

    def verify_coverage(
        self,
        identified_threats: list,  # List[ThreatItem]
        design_text:        str,
        mitigations:        list,  # List[MitigationItem]
    ) -> MQ4Coverage:
        """
        Compute the MQ4 coverage score for a completed threat model.

        Args:
            identified_threats : threats from MQ2 (list of ThreatItem dataclasses)
            design_text        : full text of the design document (for asset / boundary counting)
            mitigations        : mitigations from MQ3 (list of MitigationItem dataclasses)

        Returns:
            MQ4Coverage with score, checklist, and recommendation
        """
        # ── STRIDE coverage ──────────────────────────────────────────────────
        covered_stride = list({
            t.stride_category for t in identified_threats
            if getattr(t, "stride_category", "") not in ("", "Unknown")
        })
        missing_stride = [
            s for s in STRIDE_ALL
            if not any(s.lower() in c.lower() for c in covered_stride)
        ]

        # ── Asset count (heuristic) ──────────────────────────────────────────
        # Count keyword mentions, divide by 3 to normalise, cap at 20
        raw_count   = sum(
            len(re.findall(p, design_text, re.IGNORECASE))
            for p in ASSET_PATTERNS
        )
        total_assets  = min(raw_count // 3, 20)
        covered_assets = min(total_assets, len(identified_threats))

        # ── Trust boundary detection ─────────────────────────────────────────
        trust_boundaries = len(TRUST_BOUNDARY_PATTERN.findall(design_text))

        # ── CVE verification status ──────────────────────────────────────────
        # A mitigation "passes" CVE verification if it has no hallucinated CVEs
        all_cves_verified = all(
            not getattr(m, "hallucinated_cves", []) for m in mitigations
        )

        # ── Build checklist ──────────────────────────────────────────────────
        checklist: Dict[str, bool] = {
            "All assets identified":           total_assets > 0,
            "All STRIDE categories addressed": len(missing_stride) == 0,
            "All threats have mitigations":    len(mitigations) >= len(identified_threats),
            "All CVEs NVD-verified":           all_cves_verified,
            "Trust boundaries documented":     trust_boundaries > 0,
            "Severity scores assigned":        all(
                getattr(t, "severity", None) is not None for t in identified_threats
            ),
            "Priority levels assigned":        all(
                getattr(m, "priority", "") in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
                for m in mitigations
            ),
            "OWASP references included":       all(
                getattr(m, "owasp_reference", "") for m in mitigations
            ),
        }

        # ── Weighted coverage score ──────────────────────────────────────────
        weighted_pass = sum(
            CHECKLIST_WEIGHTS[criterion]
            for criterion, passed in checklist.items()
            if passed
        )
        max_weight    = sum(CHECKLIST_WEIGHTS.values())
        score         = weighted_pass / max_weight

        # ── Recommendation ───────────────────────────────────────────────────
        if score >= COVERAGE_PASS_THRESHOLD:
            recommendation = (
                "✅ Threat model is comprehensive and ready for stakeholder review."
            )
        else:
            gaps = ", ".join(missing_stride) if missing_stride else "see checklist"
            recommendation = (
                f"⚠️  Coverage score {score:.0%}. "
                f"Missing STRIDE categories: {gaps}. "
                f"Review checklist items marked False."
            )

        logger.info(
            "MQ4 coverage: %.0f%%  STRIDE missing: %s",
            score * 100, missing_stride or "none",
        )

        return MQ4Coverage(
            total_assets                = total_assets,
            covered_assets              = covered_assets,
            stride_categories_covered   = covered_stride,
            stride_categories_missing   = missing_stride,
            trust_boundaries_identified = trust_boundaries,
            coverage_score              = round(score, 3),
            checklist                   = checklist,
            recommendation              = recommendation,
        )
