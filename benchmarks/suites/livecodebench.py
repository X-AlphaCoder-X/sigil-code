"""
LiveCodeBench benchmark suite.

LiveCodeBench evaluates coding agents on competitive-programming-style
problems that were published *after* the model's training cutoff, preventing
data contamination.

Reference: https://livecodebench.github.io/
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset — competitive-programming style problems
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "lcb-001",
        "title": "Two Sum Sorted",
        "description": (
            "Given a sorted array of integers `nums` and a target integer `target`, "
            "return the indices (0-based) of the two numbers that add up to target. "
            "You may assume each input has exactly one solution, and you may not use "
            "the same element twice. Return them as a list [i, j] where i < j."
        ),
        "test_cases": [
            {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
            {"input": {"nums": [1, 3, 5, 7], "target": 8}, "expected": [0, 3]},
            {"input": {"nums": [-1, 0, 3, 5], "target": 4}, "expected": [0, 3]},
            {"input": {"nums": [1, 2], "target": 3}, "expected": [0, 1]},
        ],
        "function_name": "two_sum_sorted",
    },
    {
        "id": "lcb-002",
        "title": "Maximum Subarray",
        "description": (
            "Given an array of integers `nums`, find the contiguous subarray "
            "(containing at least one number) which has the largest sum and return its sum."
        ),
        "test_cases": [
            {"input": {"nums": [-2, 1, -3, 4, -1, 2, 1, -5, 4]}, "expected": 6},
            {"input": {"nums": [1]}, "expected": 1},
            {"input": {"nums": [-1, -2, -3]}, "expected": -1},
            {"input": {"nums": [5, 4, -1, 7, 8]}, "expected": 23},
        ],
        "function_name": "max_subarray",
    },
    {
        "id": "lcb-003",
        "title": "Valid Parentheses",
        "description": (
            "Given a string `s` containing just the characters '(', ')', '{', '}', "
            "'[' and ']', determine if the input string is valid. An input string is "
            "valid if open brackets are closed by the same type of brackets in the "
            "correct order."
        ),
        "test_cases": [
            {"input": {"s": "()"}, "expected": True},
            {"input": {"s": "()[]{}"}, "expected": True},
            {"input": {"s": "(]"}, "expected": False},
            {"input": {"s": "([)]"}, "expected": False},
            {"input": {"s": "{[]}"}, "expected": True},
            {"input": {"s": ""}, "expected": True},
        ],
        "function_name": "is_valid",
    },
    {
        "id": "lcb-004",
        "title": "Longest Common Prefix",
        "description": (
            "Write a function that finds the longest common prefix string "
            "amongst a list of strings. If there is no common prefix, return "
            "an empty string."
        ),
        "test_cases": [
            {"input": {"strs": ["flower", "flow", "flight"]}, "expected": "fl"},
            {"input": {"strs": ["dog", "racecar", "car"]}, "expected": ""},
            {"input": {"strs": ["abc"]}, "expected": "abc"},
            {"input": {"strs": ["", "b"]}, "expected": ""},
            {"input": {"strs": ["ab", "ab", "ab"]}, "expected": "ab"},
        ],
        "function_name": "longest_common_prefix",
    },
    {
        "id": "lcb-005",
        "title": "Count Inversions",
        "description": (
            "Given an array of integers, count the number of inversions. "
            "An inversion is a pair (i, j) where i < j but nums[i] > nums[j]. "
            "Return the total number of inversions."
        ),
        "test_cases": [
            {"input": {"nums": [2, 4, 1, 3, 5]}, "expected": 3},
            {"input": {"nums": [1, 2, 3, 4]}, "expected": 0},
            {"input": {"nums": [4, 3, 2, 1]}, "expected": 6},
            {"input": {"nums": [1]}, "expected": 0},
        ],
        "function_name": "count_inversions",
    },
]


class LiveCodeBenchBenchmark(BenchmarkSuite):
    """LiveCodeBench: competitive programming problems."""

    name = "LiveCodeBench"
    description = "Competitive programming problems (post-training-cutoff)"
    category = "coding"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "livecodebench.jsonl"
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
            print(f"  {jsonl_path} not found — using built-in 5-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        desc = problem["description"]
        fname = problem["function_name"]
        # Build example from first test case
        tc = problem["test_cases"][0]
        args = ", ".join(f"{k}={repr(v)}" for k, v in tc["input"].items())
        expected = repr(tc["expected"])
        return (
            f"Solve the following competitive programming problem.\n\n"
            f"## {problem.get('title', 'Problem')}\n\n"
            f"{desc}\n\n"
            f"Example: {fname}({args}) => {expected}\n\n"
            f"Write a Python function called `{fname}` and save it to solution.py."
        )

    def setup_workspace(self, problem: dict[str, Any], workspace: str) -> None:
        fname = problem["function_name"]
        test_lines = [
            "import sys",
            'sys.path.insert(0, ".")',
            f"from solution import {fname}",
            "",
        ]
        for i, tc in enumerate(problem["test_cases"]):
            args = ", ".join(f"{k}={repr(v)}" for k, v in tc["input"].items())
            test_lines.append(
                f"assert {fname}({args}) == {repr(tc['expected'])}, "
                f"'Test case {i+1} failed'"
            )
        test_lines.append("")
        test_lines.append('print("ALL_TESTS_PASSED")')
        with open(os.path.join(workspace, "test_harness.py"), "w") as fh:
            fh.write("\n".join(test_lines))

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        sol = os.path.join(workspace, "solution.py")
        if not os.path.exists(sol):
            return BenchmarkResult(
                problem_id=pid, passed=False, error="solution.py not found"
            )
        code, output = self._run_shell(
            f"{sys.executable} test_harness.py", cwd=workspace, timeout=30.0
        )
        passed = code == 0 and "ALL_TESTS_PASSED" in output
        return BenchmarkResult(
            problem_id=pid, passed=passed, actual=output[:500],
            error="" if passed else output[:500],
        )
