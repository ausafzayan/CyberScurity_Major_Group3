"""
binary_evaluator.py
===================
Replicates the paper's binary human evaluation method.

The paper asked two human coders to rate each LLM response as:
  "+" (meets expectations) or "-" (does not meet expectations)
and reported a 75%+ satisfaction rate across 72 evaluations.

This module supports:
  1. Interactive evaluation: prompts a human evaluator in the terminal
  2. Automated fallback: applies a word-count heuristic when no terminal is available

Limitation (acknowledged in the paper):
  Binary +/- evaluation is subjective, not reproducible, and cannot be
  compared against other systems. Our improved_approach replaces this
  with automated BLEU/ROUGE/BERTScore metrics (see improved_approach/evaluation/).

Libraries used:
  - sys : detect if running in an interactive terminal
  - json : save evaluation results to disk
"""

import json
import logging
import sys
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Heuristic threshold for automated evaluation when no terminal is attached.
# Responses with fewer words than this are considered non-informative.
MIN_WORDS_FOR_PASS = 20


class BinaryEvaluator:
    """
    Simulates the paper's binary human evaluation.

    The paper used 2 human coders; this class supports:
      - Interactive mode (sys.stdin.isatty() == True): prompts a real user
      - Automated mode (CI / script): applies a simple heuristic

    Usage:
        evaluator = BinaryEvaluator()
        result    = evaluator.evaluate_response(question, answer, task=1)
        rate      = evaluator.compute_satisfaction_rate()
    """

    def __init__(self) -> None:
        # Accumulates one result dict per evaluated response
        self.results: List[Dict] = []

    def evaluate_response(
        self,
        question: str,
        answer:   str,
        task:     int,
    ) -> Dict:
        """
        Evaluate a single LLM response.

        In interactive mode, prints the Q&A and prompts the user for + or -.
        In automated mode, rates as "+" if the answer has >= MIN_WORDS_FOR_PASS
        words and does not contain a refusal phrase.

        Args:
            question : the threat modeling question
            answer   : the LLM-generated answer
            task     : 1 (MQ1) or 2 (MQ2)

        Returns:
            Dict with keys: question, task, rating ("+"/"-"), word_count
        """
        word_count = len(answer.split())

        # ── Display the Q&A to the evaluator ──────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Task MQ{task} — Question:\n  {question}")
        print(f"\nAnswer ({word_count} words):\n{answer}")
        print(f"{'='*60}")

        # ── Interactive vs automated rating ───────────────────────────────────
        if sys.stdin.isatty():
            # Real terminal: wait for human input
            while True:
                rating = input("Evaluation (+ = meets expectations, - = does not): ").strip()
                if rating in ("+", "-"):
                    break
                print("Please enter '+' or '-'.")
        else:
            # No terminal (e.g. running in a pipeline or subprocess):
            # apply a simple heuristic that approximates human judgment
            is_informative = (
                word_count >= MIN_WORDS_FOR_PASS
                and "i don't know" not in answer.lower()
                and "i cannot"     not in answer.lower()
            )
            rating = "+" if is_informative else "-"
            logger.info("  Auto-eval: %s  (heuristic, %d words)", rating, word_count)

        result = {
            "question":  question,
            "task":      task,
            "rating":    rating,
            "word_count": word_count,
        }
        self.results.append(result)
        return result

    def compute_satisfaction_rate(self) -> float:
        """
        Compute the overall satisfaction rate (% of '+' ratings).

        Returns:
            Float in [0, 100] representing the satisfaction percentage
        """
        if not self.results:
            logger.warning("No evaluations recorded.")
            return 0.0

        positive = sum(1 for r in self.results if r["rating"] == "+")
        rate     = positive / len(self.results) * 100

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY  (Paper's Binary Method)")
        print(f"  Total evaluations : {len(self.results)}")
        print(f"  Positive (+)      : {positive}")
        print(f"  Satisfaction rate : {rate:.1f}%")
        print(f"  (Paper reported   : 75%+)")
        print(f"{'='*60}")

        return rate

    def save_results(self, output_path: str) -> None:
        """
        Persist evaluation results to a JSON file.

        Args:
            output_path : destination file path (e.g. "eval_results.json")
        """
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "satisfaction_rate": self.compute_satisfaction_rate(),
                    "results":           self.results,
                },
                fh,
                indent=2,
            )
        logger.info("Evaluation results saved to: %s", output_path)
