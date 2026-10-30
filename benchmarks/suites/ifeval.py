"""
IFEval benchmark suite.

IFEval (Instruction Following Evaluation) tests whether a model can
follow verifiable natural-language instructions — e.g., "write at least
200 words", "include exactly 3 bullet points", "respond in all lowercase".

Paper: https://arxiv.org/abs/2311.07911
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset — verifiable instruction-following tasks
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "ifeval-001",
        "instruction": "Write a short paragraph about the benefits of exercise. Your response must be entirely in lowercase. Save it to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "all_lowercase", "file": "output.txt"},
            {"type": "min_words", "file": "output.txt", "value": 20},
        ],
    },
    {
        "id": "ifeval-002",
        "instruction": "Write a poem about the ocean. It must have exactly 4 lines. Save it to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "exact_line_count", "file": "output.txt", "value": 4},
        ],
    },
    {
        "id": "ifeval-003",
        "instruction": "List 5 programming languages. Format each as a numbered item (1. Language). Save to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "min_lines", "file": "output.txt", "value": 5},
            {"type": "contains_pattern", "file": "output.txt", "pattern": r"^\d+\.", "min_count": 5},
        ],
    },
    {
        "id": "ifeval-004",
        "instruction": "Write a summary of what Python is in EXACTLY 3 sentences. Each sentence must end with a period. Save to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "exact_sentence_count", "file": "output.txt", "value": 3},
        ],
    },
    {
        "id": "ifeval-005",
        "instruction": "Write a response that contains the word 'innovation' at least 3 times. The response should be about technology. Save to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "word_frequency", "file": "output.txt", "word": "innovation", "min_count": 3},
        ],
    },
    {
        "id": "ifeval-006",
        "instruction": "Write a short paragraph about space exploration. It must contain exactly 2 bullet points, each starting with '- '. Save to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "contains_pattern", "file": "output.txt", "pattern": r"^- ", "min_count": 2, "max_count": 2},
        ],
    },
    {
        "id": "ifeval-007",
        "instruction": "Write the numbers from 1 to 10, each on a separate line. Save to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "exact_line_count", "file": "output.txt", "value": 10},
            {"type": "contains", "file": "output.txt", "text": "1"},
            {"type": "contains", "file": "output.txt", "text": "10"},
        ],
    },
    {
        "id": "ifeval-008",
        "instruction": "Write a haiku (3 lines: 5 syllables, 7 syllables, 5 syllables) about nature. Save to output.txt. The response must be exactly 3 lines.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "exact_line_count", "file": "output.txt", "value": 3},
        ],
    },
    {
        "id": "ifeval-009",
        "instruction": "Write a paragraph about artificial intelligence. Every sentence must start with the word 'AI'. Save to output.txt.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "sentences_start_with", "file": "output.txt", "prefix": "AI"},
        ],
    },
    {
        "id": "ifeval-010",
        "instruction": "Create a JSON file called output.txt containing an object with exactly 3 keys: 'name', 'age', and 'city'. Values can be anything.",
        "checks": [
            {"type": "file_exists", "file": "output.txt"},
            {"type": "valid_json", "file": "output.txt"},
            {"type": "json_has_keys", "file": "output.txt", "keys": ["name", "age", "city"]},
        ],
    },
]


def _run_check(check: dict[str, Any], workspace: str) -> bool:
    """Run a single verification check. Returns True if passed."""
    check_type = check["type"]
    filepath = os.path.join(workspace, check.get("file", "output.txt"))

    if check_type == "file_exists":
        return os.path.exists(filepath)

    if not os.path.exists(filepath):
        return False

    with open(filepath) as fh:
        content = fh.read()

    lines = [l for l in content.strip().split("\n") if l.strip()]

    if check_type == "all_lowercase":
        # Check that alphabetic characters are lowercase
        alpha_chars = [c for c in content if c.isalpha()]
        return all(c.islower() for c in alpha_chars) if alpha_chars else True

    if check_type == "min_words":
        words = content.split()
        return len(words) >= check["value"]

    if check_type == "exact_line_count":
        return len(lines) == check["value"]

    if check_type == "min_lines":
        return len(lines) >= check["value"]

    if check_type == "contains_pattern":
        pattern = check["pattern"]
        matches = sum(1 for line in lines if re.search(pattern, line))
        if "min_count" in check and matches < check["min_count"]:
            return False
        if "max_count" in check and matches > check["max_count"]:
            return False
        return True

    if check_type == "exact_sentence_count":
        # Count sentences ending with . ! or ?
        sentences = re.split(r"[.!?]+", content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences) == check["value"]

    if check_type == "word_frequency":
        word = check["word"].lower()
        count = content.lower().count(word)
        return count >= check["min_count"]

    if check_type == "contains":
        return check["text"] in content

    if check_type == "sentences_start_with":
        prefix = check["prefix"]
        sentences = re.split(r"[.!?]\s+", content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        return all(s.startswith(prefix) for s in sentences) if sentences else False

    if check_type == "valid_json":
        try:
            json.loads(content)
            return True
        except json.JSONDecodeError:
            return False

    if check_type == "json_has_keys":
        try:
            data = json.loads(content)
            return all(k in data for k in check["keys"])
        except (json.JSONDecodeError, TypeError):
            return False

    return False


class IFEvalBenchmark(BenchmarkSuite):
    """IFEval: instruction following evaluation."""

    name = "IFEval"
    description = "Verifiable instruction-following evaluation"
    category = "instruction-following"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "ifeval.jsonl"
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
            print(f"  {jsonl_path} not found — using built-in 10-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        return problem["instruction"]

    def recover_output_files(
        self,
        problem: dict[str, Any],
        workspace: str,
        agent_output: str,
        metadata: dict[str, Any],
    ) -> None:
        del problem
        output_path = Path(workspace) / "output.txt"
        if output_path.exists():
            return
        text = agent_output.strip()
        if text:
            output_path.write_text(text + "\n", encoding="utf-8")
            metadata["recovered_output_from_agent_text"] = True

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        checks = problem.get("checks", [])

        failed_checks: list[str] = []
        for check in checks:
            if not _run_check(check, workspace):
                failed_checks.append(check["type"])

        passed = len(failed_checks) == 0
        return BenchmarkResult(
            problem_id=pid, passed=passed,
            error=f"Failed checks: {', '.join(failed_checks)}" if failed_checks else "",
            metadata={"total_checks": len(checks), "failed_checks": failed_checks},
        )
