"""
auto_evaluator.py
=================
Automated Quantitative Evaluation — Improvement 3 over the paper.

The paper used binary +/- human evaluation (subjective, not reproducible).
This module replaces it with three standard NLP evaluation metrics:

BLEU (Bilingual Evaluation Understudy):
  Measures n-gram precision between generated and reference text.
  Originally designed for machine translation evaluation.
  Range: 0.0 (no overlap) – 1.0 (perfect match).
  Library: sacrebleu (standardised, corpus-level BLEU)

ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation):
  Measures the longest common subsequence (LCS) F1 between output and reference.
  Focuses on sentence-level structure and fluency.
  Range: 0.0 – 1.0 (F1 score).
  Library: rouge-score

BERTScore:
  Uses contextual BERT embeddings to compute token-level cosine similarity.
  Unlike BLEU/ROUGE, it captures semantic meaning — paraphrases score highly.
  Range: ~0.0 – 1.0 (F1 score; rescaled with baseline).
  Library: bert-score

Composite score = mean(BLEU, ROUGE-L, BERTScore F1).
Grade = PASS if composite ≥ 0.4.

Libraries used:
  - sacrebleu   : BLEU scoring (pip install sacrebleu)
  - rouge_score : ROUGE scoring (pip install rouge-score)
  - bert_score  : BERTScore (pip install bert-score)
"""

import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMPOSITE_PASS_THRESHOLD = 0.4   # minimum composite score for PASS grade


class AutoEvaluator:
    """
    Evaluates LLM-generated answers against expert ground-truth references
    using BLEU, ROUGE-L, and BERTScore.

    Usage:
        evaluator = AutoEvaluator()
        scores    = evaluator.evaluate_answer(generated_answer, reference_answer)
        print(scores["composite_score"], scores["grade"])

        batch = evaluator.evaluate_batch(responses, ground_truth_dict)
        print(batch["avg_bleu"], batch["pass_rate"])
    """

    def evaluate_answer(self, generated: str, reference: str) -> Dict:
        """
        Compute BLEU, ROUGE-L, and BERTScore for a single answer pair.

        Args:
            generated : the LLM-produced answer string
            reference : the expert ground-truth answer string

        Returns:
            Dict with keys:
              generated_words : word count of generated answer
              reference_words : word count of reference answer
              bleu            : BLEU score (float or None if library missing)
              rouge_l         : ROUGE-L F1 (float or None)
              bert_score_f1   : BERTScore F1 (float or None)
              composite_score : mean of available scores
              grade           : "PASS" or "FAIL"
        """
        results = {
            "generated_words": len(generated.split()),
            "reference_words": len(reference.split()),
        }

        # ── BLEU Score ────────────────────────────────────────────────────────
        # sacrebleu.BLEU.sentence_score returns a score on 0–100 scale; we /100
        try:
            from sacrebleu.metrics import BLEU
            bleu_scorer      = BLEU(effective_order=True)  # handles short sentences
            results["bleu"]  = round(
                bleu_scorer.sentence_score(generated, [reference]).score / 100, 4
            )
        except ImportError:
            logger.warning("sacrebleu not installed — skipping BLEU. pip install sacrebleu")
            results["bleu"] = None

        # ── ROUGE-L ──────────────────────────────────────────────────────────
        # rouge_scorer.score(reference, hypothesis) — note argument order
        try:
            from rouge_score import rouge_scorer
            scorer           = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_scores     = scorer.score(reference, generated)
            results["rouge_l"] = round(rouge_scores["rougeL"].fmeasure, 4)
        except ImportError:
            logger.warning("rouge-score not installed — skipping ROUGE-L. pip install rouge-score")
            results["rouge_l"] = None

        # ── BERTScore ─────────────────────────────────────────────────────────
        # bert_score.score returns (Precision, Recall, F1) tensors
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                [generated], [reference],
                lang="en",
                rescale_with_baseline=True,   # shifts scores to ~0–1 range
            )
            results["bert_score_f1"] = round(float(F1[0]), 4)
        except ImportError:
            logger.warning("bert-score not installed — skipping BERTScore. pip install bert-score")
            results["bert_score_f1"] = None

        # ── Composite score ───────────────────────────────────────────────────
        valid_scores = [
            v for v in (results["bleu"], results["rouge_l"], results["bert_score_f1"])
            if v is not None
        ]
        composite             = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        results["composite_score"] = round(composite, 4)
        results["grade"]           = "PASS" if composite >= COMPOSITE_PASS_THRESHOLD else "FAIL"

        return results

    def evaluate_batch(
        self,
        responses:    List[Dict],
        ground_truth: Dict[str, str],
    ) -> Dict:
        """
        Evaluate a batch of LLM responses against expert ground-truth answers.

        Args:
            responses    : list of dicts, each with "question" and "answer" keys
            ground_truth : dict mapping question strings to expert reference answers

        Returns:
            Summary dict with average metrics and per-question details
        """
        all_scores = []

        for resp in responses:
            question  = resp.get("question", "")
            generated = resp.get("answer",   "")
            reference = ground_truth.get(question, "")

            if not reference:
                logger.warning("No ground truth found for: %s…", question[:50])
                continue

            scores             = self.evaluate_answer(generated, reference)
            scores["question"] = question
            all_scores.append(scores)

        if not all_scores:
            return {"error": "No ground-truth matches found — check question strings."}

        # ── Aggregate averages ────────────────────────────────────────────────
        def safe_avg(key: str) -> Optional[float]:
            vals = [s[key] for s in all_scores if s.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary = {
            "total_evaluated":       len(all_scores),
            "avg_bleu":              safe_avg("bleu"),
            "avg_rouge_l":           safe_avg("rouge_l"),
            "avg_bert_score_f1":     safe_avg("bert_score_f1"),
            "avg_composite":         safe_avg("composite_score"),
            "pass_rate":             round(
                sum(1 for s in all_scores if s["grade"] == "PASS") / len(all_scores) * 100, 1
            ),
            "paper_binary_equiv":    "75%+ (for reference)",
            "detailed":              all_scores,
        }

        # ── Print summary table ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("AUTOMATED EVALUATION RESULTS  (Our Improved Method)")
        print("=" * 60)
        for label, key in [
            ("BLEU Score",     "avg_bleu"),
            ("ROUGE-L",        "avg_rouge_l"),
            ("BERTScore F1",   "avg_bert_score_f1"),
            ("Composite",      "avg_composite"),
        ]:
            val = summary[key]
            print(f"  {label:<20}: {f'{val:.4f}' if val else 'N/A'}")
        print(f"  {'Pass Rate':<20}: {summary['pass_rate']}%")
        print(f"  {'Paper baseline':<20}: {summary['paper_binary_equiv']}")
        print("=" * 60)

        return summary
