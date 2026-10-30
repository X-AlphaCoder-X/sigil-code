"""
SWE-Bench benchmark suite.

SWE-Bench tests whether an agent can resolve real GitHub issues.
Each problem provides a repository, an issue description, and a patch
that should make the failing tests pass.

Paper: https://arxiv.org/abs/2310.06770
Dataset: https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite
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
# Built-in mini dataset — simplified SWE-Bench-style problems
# These are self-contained (no actual repo cloning needed).
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "instance_id": "swe-mini-001",
        "problem_statement": (
            "The `StringFormatter.format_title` method should capitalize the first letter "
            "of each word in the input string. Currently it lowercases everything instead.\n\n"
            "Expected: format_title('hello world') => 'Hello World'\n"
            "Actual:   format_title('hello world') => 'hello world'"
        ),
        "setup_code": textwrap.dedent("""\
            cat > formatter.py << 'PYEOF'
            class StringFormatter:
                def format_title(self, text):
                    return text.lower()
            PYEOF
            cat > test_formatter.py << 'PYEOF'
            from formatter import StringFormatter
            f = StringFormatter()
            assert f.format_title('hello world') == 'Hello World'
            assert f.format_title('PYTHON IS FUN') == 'Python Is Fun'
            assert f.format_title('already Good') == 'Already Good'
            print("ALL_TESTS_PASSED")
            PYEOF
        """),
        "test_cmd": "python3 test_formatter.py",
    },
    {
        "instance_id": "swe-mini-002",
        "problem_statement": (
            "The `DataValidator.validate_email` function returns True for emails "
            "that don't have a domain extension (e.g., 'user@domain' without .com). "
            "It should require at least one dot after the @ sign.\n\n"
            "Expected: validate_email('user@domain') => False\n"
            "Actual:   validate_email('user@domain') => True"
        ),
        "setup_code": textwrap.dedent("""\
            cat > validator.py << 'PYEOF'
            class DataValidator:
                def validate_email(self, email):
                    return '@' in email
            PYEOF
            cat > test_validator.py << 'PYEOF'
            from validator import DataValidator
            v = DataValidator()
            assert v.validate_email('user@example.com') == True
            assert v.validate_email('bad') == False
            assert v.validate_email('user@domain') == False
            assert v.validate_email('a@b.c') == True
            print("ALL_TESTS_PASSED")
            PYEOF
        """),
        "test_cmd": "python3 test_validator.py",
    },
    {
        "instance_id": "swe-mini-003",
        "problem_statement": (
            "The `FileProcessor.count_lines` method crashes with an unhandled "
            "exception when the file doesn't exist. It should return -1 for missing files.\n\n"
            "Expected: count_lines('nonexistent.txt') => -1\n"
            "Actual:   FileNotFoundError is raised"
        ),
        "setup_code": textwrap.dedent("""\
            cat > processor.py << 'PYEOF'
            class FileProcessor:
                def count_lines(self, filepath):
                    with open(filepath) as f:
                        return len(f.readlines())
            PYEOF
            echo -e "line1\\nline2\\nline3" > sample.txt
            cat > test_processor.py << 'PYEOF'
            from processor import FileProcessor
            p = FileProcessor()
            assert p.count_lines('sample.txt') == 3
            assert p.count_lines('nonexistent.txt') == -1
            print("ALL_TESTS_PASSED")
            PYEOF
        """),
        "test_cmd": "python3 test_processor.py",
    },
    {
        "instance_id": "swe-mini-004",
        "problem_statement": (
            "The `MathHelper.factorial` function gives wrong results for n=0. "
            "factorial(0) should return 1, but it currently returns 0.\n\n"
            "Expected: factorial(0) => 1\n"
            "Actual:   factorial(0) => 0"
        ),
        "setup_code": textwrap.dedent("""\
            cat > mathhelper.py << 'PYEOF'
            class MathHelper:
                def factorial(self, n):
                    result = 0
                    for i in range(1, n + 1):
                        result *= i
                    return result
            PYEOF
            cat > test_math.py << 'PYEOF'
            from mathhelper import MathHelper
            m = MathHelper()
            assert m.factorial(0) == 1
            assert m.factorial(1) == 1
            assert m.factorial(5) == 120
            assert m.factorial(10) == 3628800
            print("ALL_TESTS_PASSED")
            PYEOF
        """),
        "test_cmd": "python3 test_math.py",
    },
    {
        "instance_id": "swe-mini-005",
        "problem_statement": (
            "The `ListUtils.unique` method should return unique elements in the "
            "order they first appear, but it currently sorts them alphabetically.\n\n"
            "Expected: unique(['b', 'a', 'b', 'c', 'a']) => ['b', 'a', 'c']\n"
            "Actual:   unique(['b', 'a', 'b', 'c', 'a']) => ['a', 'b', 'c']"
        ),
        "setup_code": textwrap.dedent("""\
            cat > listutils.py << 'PYEOF'
            class ListUtils:
                def unique(self, items):
                    return sorted(set(items))
            PYEOF
            cat > test_list.py << 'PYEOF'
            from listutils import ListUtils
            u = ListUtils()
            assert u.unique(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']
            assert u.unique([3, 1, 2, 1, 3]) == [3, 1, 2]
            assert u.unique([]) == []
            assert u.unique([1]) == [1]
            print("ALL_TESTS_PASSED")
            PYEOF
        """),
        "test_cmd": "python3 test_list.py",
    },
]


class SWEBenchBenchmark(BenchmarkSuite):
    """SWE-Bench: resolve real-world GitHub issues."""

    name = "SWE-Bench"
    description = "Resolve GitHub issues by editing source code"
    category = "coding"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "swe_bench.jsonl"
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
        stmt = problem["problem_statement"]
        return (
            f"You are a software engineer fixing a bug. Here is the issue:\n\n"
            f"{stmt}\n\n"
            f"Read the code in the workspace, find the bug, and fix it. "
            f"Do not modify the test files."
        )

    def setup_workspace(self, problem: dict[str, Any], workspace: str) -> None:
        setup = problem.get("setup_code", "")
        if setup:
            self._run_shell(setup, cwd=workspace, timeout=30.0)

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("instance_id", "unknown")
        test_cmd = problem.get("test_cmd", "echo FAIL && exit 1")
        code, output = self._run_shell(test_cmd, cwd=workspace, timeout=30.0)
        passed = code == 0 and "ALL_TESTS_PASSED" in output
        return BenchmarkResult(
            problem_id=pid, passed=passed, actual=output[:500],
            error="" if passed else output[:500],
        )
