"""
MATH benchmark suite.

The MATH dataset consists of 12,500 problems from mathematics competitions
(AMC, AIME, etc.) covering topics like algebra, counting, geometry, number
theory, and more.  Difficulty levels 1–5.

Paper: https://arxiv.org/abs/2103.03874
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (15 representative problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "math-001",
        "problem": "What is the value of $2^{10}$?",
        "answer": "1024",
        "subject": "algebra",
        "level": 1,
    },
    {
        "id": "math-002",
        "problem": "What is $15 \\% $ of $200$?",
        "answer": "30",
        "subject": "algebra",
        "level": 1,
    },
    {
        "id": "math-003",
        "problem": "Solve for $x$: $3x + 7 = 22$.",
        "answer": "5",
        "subject": "algebra",
        "level": 1,
    },
    {
        "id": "math-004",
        "problem": "What is the sum of the first 10 positive integers?",
        "answer": "55",
        "subject": "algebra",
        "level": 1,
    },
    {
        "id": "math-005",
        "problem": "What is $\\gcd(12, 18)$?",
        "answer": "6",
        "subject": "number_theory",
        "level": 1,
    },
    {
        "id": "math-006",
        "problem": "How many ways can you choose 3 items from a set of 5 items?",
        "answer": "10",
        "subject": "counting",
        "level": 2,
    },
    {
        "id": "math-007",
        "problem": "What is the area of a triangle with base 10 and height 6?",
        "answer": "30",
        "subject": "geometry",
        "level": 1,
    },
    {
        "id": "math-008",
        "problem": "Evaluate $\\sum_{k=1}^{5} k^2$.",
        "answer": "55",
        "subject": "algebra",
        "level": 2,
    },
    {
        "id": "math-009",
        "problem": "Find the remainder when $2^{20}$ is divided by $7$.",
        "answer": "4",
        "subject": "number_theory",
        "level": 3,
    },
    {
        "id": "math-010",
        "problem": "What is the least common multiple of 12 and 18?",
        "answer": "36",
        "subject": "number_theory",
        "level": 2,
    },
    {
        "id": "math-011",
        "problem": "A bag contains 3 red and 5 blue marbles. What is the probability of drawing a red marble? Express as a simplified fraction.",
        "answer": "3/8",
        "subject": "counting",
        "level": 2,
    },
    {
        "id": "math-012",
        "problem": "What is the value of $\\sqrt{144}$?",
        "answer": "12",
        "subject": "algebra",
        "level": 1,
    },
    {
        "id": "math-013",
        "problem": "If $f(x) = 2x^2 - 3x + 1$, what is $f(3)$?",
        "answer": "10",
        "subject": "algebra",
        "level": 2,
    },
    {
        "id": "math-014",
        "problem": "How many prime numbers are there between 1 and 20?",
        "answer": "8",
        "subject": "number_theory",
        "level": 2,
    },
    {
        "id": "math-015",
        "problem": "What is the value of $3! + 4!$?",
        "answer": "30",
        "subject": "counting",
        "level": 1,
    },
]


def _normalize_answer(text: str) -> str:
    """Normalize a math answer for comparison.

    Strips whitespace, dollar signs, \\boxed{}, etc.
    """
    text = text.strip()
    # Remove \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        text = m.group(1)
    # Remove dollar signs and whitespace
    text = text.replace("$", "").replace(",", "").strip()
    # Try to normalize fractions
    text = text.replace("\\frac{", "").replace("}{", "/").replace("}", "")
    return text.strip()


class MATHBenchmark(BenchmarkSuite):
    """MATH: competition mathematics problems."""

    name = "MATH"
    description = "Competition mathematics problems (algebra, geometry, number theory, etc.)"
    category = "math"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "math.jsonl"
        if jsonl_path.exists():
            problems: list[dict[str, Any]] = []
            with open(jsonl_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        problems.append(json.loads(line))
            if self.verbose:
                print(f"  Loaded {len(problems)} problems from {jsonl_path}")
            return problems

        if self.verbose:
            print(f"  {jsonl_path} not found — using built-in 15-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        prob = problem["problem"]
        return (
            f"Solve the following math problem. Write ONLY the final numerical "
            f"answer (a single number or simple expression) to a file called answer.txt.\n\n"
            f"Problem: {prob}\n\n"
            f"Save just the answer to answer.txt — no explanation, no work, just the answer."
        )

    def recover_output_files(
        self,
        problem: dict[str, Any],
        workspace: str,
        agent_output: str,
        metadata: dict[str, Any],
    ) -> None:
        del problem
        answer_path = Path(workspace) / "answer.txt"
        if answer_path.exists():
            return
        answer = _normalize_answer(agent_output)
        if answer:
            answer_path.write_text(answer + "\n", encoding="utf-8")
            metadata["recovered_answer_from_output"] = True

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", str(problem.get("task_id", "unknown")))
        expected_raw = str(problem["answer"])
        answer_file = os.path.join(workspace, "answer.txt")

        if not os.path.exists(answer_file):
            return BenchmarkResult(
                problem_id=pid, passed=False, expected=expected_raw,
                error="answer.txt not found",
            )

        with open(answer_file) as fh:
            actual_raw = fh.read().strip()

        expected_norm = _normalize_answer(expected_raw)
        actual_norm = _normalize_answer(actual_raw)

        # Try numeric comparison
        try:
            passed = abs(float(actual_norm) - float(expected_norm)) < 1e-6
        except (ValueError, ZeroDivisionError):
            passed = actual_norm == expected_norm

        return BenchmarkResult(
            problem_id=pid, passed=passed,
            expected=expected_raw, actual=actual_raw,
            error="" if passed else f"expected={expected_raw}, got={actual_raw}",
        )
