"""
MBPP (Mostly Basic Python Problems) benchmark suite.

974 crowd-sourced Python programming problems designed to be solvable by
entry-level programmers.  Each problem has a task description, code solution
and 3 automated test cases.

Paper: https://arxiv.org/abs/2108.07732
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

from .base import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Built-in mini dataset (15 representative MBPP problems)
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "task_id": 1,
        "text": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix.",
        "code": "R = 3\r\nC = 3\r\ndef min_cost(cost, m, n):\r\n\ttc = [[0 for x in range(C)] for x in range(R)]\r\n\ttc[0][0] = cost[0][0]\r\n\tfor i in range(1, m+1):\r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0]\r\n\tfor j in range(1, n+1):\r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j]\r\n\tfor i in range(1, m+1):\r\n\t\tfor j in range(1, n+1):\r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]\r\n\treturn tc[m][n]",
        "test_list": [
            "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
            "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12",
            "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16",
        ],
    },
    {
        "task_id": 2,
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
    },
    {
        "task_id": 3,
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
    },
    {
        "task_id": 4,
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
    },
    {
        "task_id": 5,
        "text": "Write a function to find the number of ways to fill it with 2 x 1 dominoes for the given 3 x n board.",
        "code": "def count_ways(n):\n\tA = [0] * (n + 1)\n\tB = [0] * (n + 1)\n\tA[0] = 1\n\tA[1] = 0\n\tB[0] = 0\n\tB[1] = 1\n\tfor i in range(2, n+1):\n\t\tA[i] = A[i - 2] + 2 * B[i - 1]\n\t\tB[i] = A[i - 1] + B[i - 2]\n\treturn A[n]",
        "test_list": [
            "assert count_ways(2) == 3",
            "assert count_ways(8) == 153",
            "assert count_ways(12) == 2131",
        ],
    },
    {
        "task_id": 6,
        "text": "Write a python function to check whether the two numbers differ at one bit position only or not.",
        "code": "def differ_At_One_Bit_Pos(a,b):\n    return ((a ^ b) & (a ^ b - 1)) == 0 and a ^ b != 0",
        "test_list": [
            "assert differ_At_One_Bit_Pos(13,9) == True",
            "assert differ_At_One_Bit_Pos(15,8) == False",
            "assert differ_At_One_Bit_Pos(2,4) == False",
        ],
    },
    {
        "task_id": 7,
        "text": "Write a function to find all words which are at least 4 characters long in a string.",
        "code": 'import re\ndef find_char_long(text):\n  return (re.findall(r"\\b\\w{4,}\\b", text))',
        "test_list": [
            'assert find_char_long(\'Please move back to stream\') == [\'Please\', \'move\', \'back\', \'stream\']',
            'assert find_char_long(\'Joveacvber gfnbb vcdf\') == [\'Jove\', \'acvber\', \'gfnbb\', \'vcdf\']',
            'assert find_char_long(\'Davvede aridge\') == [\'Davvede\', \'aridge\']',
        ],
    },
    {
        "task_id": 8,
        "text": "Write a function to find squares of individual elements in a list.",
        "code": "def square_nums(nums):\n return list(map(lambda x: x ** 2, nums))",
        "test_list": [
            "assert square_nums([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])==[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]",
            "assert square_nums([10,20,30])==[100,400,900]",
            "assert square_nums([12,15])==[144,225]",
        ],
    },
    {
        "task_id": 9,
        "text": "Write a python function to find the minimum number of rotations required to get the same string.",
        "code": "def find_Rotations(s):\n    n = len(s)\n    s2 = s + s\n    for i in range(1, n + 1):\n        if s2[i:i+n] == s:\n            return i\n    return n",
        "test_list": [
            'assert find_Rotations("aaaa") == 1',
            'assert find_Rotations("ab") == 2',
            'assert find_Rotations("abc") == 3',
        ],
    },
    {
        "task_id": 10,
        "text": "Write a function to get the n smallest items from a dataset.",
        "code": "import heapq\ndef small_nnum(list1,n):\n  smallest=heapq.nsmallest(n,list1)\n  return smallest",
        "test_list": [
            "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],2)==[10,20]",
            "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],5)==[10,20,20,40,50]",
            "assert small_nnum([10, 20, 50, 70, 90, 20, 50, 40, 60, 80, 100],3)==[10,20,20]",
        ],
    },
    {
        "task_id": 11,
        "text": "Write a python function to remove first and last occurrence of a given character from the string.",
        "code": "def remove_Occ(s,ch):\n    # Remove first occurrence\n    for i in range(len(s)):\n        if (s[i] == ch):\n            s = s[0 : i] + s[i + 1:]\n            break\n    # Remove last occurrence\n    for i in range(len(s) - 1,-1,-1):\n        if (s[i] == ch):\n            s = s[0 : i] + s[i + 1:]\n            break\n    return s",
        "test_list": [
            'assert remove_Occ("hello","l") == "heo"',
            'assert remove_Occ("abcda","a") == "bcd"',
            'assert remove_Occ("PHP","P") == "H"',
        ],
    },
    {
        "task_id": 12,
        "text": "Write a function to sort a given matrix in ascending order according to the sum of its rows.",
        "code": "def sort_matrix(M):\n    result = sorted(M, key=sum)\n    return result",
        "test_list": [
            "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
            "assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
            "assert sort_matrix([[5,8,9],[6,4,3],[2,1,4]])==[[2, 1, 4], [6, 4, 3], [5, 8, 9]]",
        ],
    },
    {
        "task_id": 13,
        "text": "Write a function to count the most common words in a dictionary.",
        "code": "from collections import Counter\ndef count_common(words):\n  word_counts = Counter(words)\n  top_four = word_counts.most_common(4)\n  return top_four",
        "test_list": [
            "assert count_common(['red','green','black','pink','black','white','black','eyes','white','black','orange','pink','pink','red','red','white','orange','white','black','pink','green','green','pink','green','pink','white','orange','orange','red']) == [('pink', 6), ('black', 5), ('white', 5), ('red', 4)]",
            "assert count_common(['one', 'two', 'three', 'four', 'five', 'one', 'two', 'one', 'three', 'one']) == [('one', 4), ('two', 2), ('three', 2), ('four', 1)]",
            "assert count_common(['Facebook', 'Apple', 'Amazon', 'Netflix', 'Google', 'Apple', 'Netflix', 'Amazon']) == [('Apple', 2), ('Amazon', 2), ('Netflix', 2), ('Facebook', 1)]",
        ],
    },
    {
        "task_id": 14,
        "text": "Write a python function to find the volume of a triangular prism.",
        "code": "def find_Volume(l,b,h):\n    return (l * b * h) / 2",
        "test_list": [
            "assert find_Volume(10,8,6) == 240",
            "assert find_Volume(3,2,2) == 6",
            "assert find_Volume(1,2,1) == 1",
        ],
    },
    {
        "task_id": 15,
        "text": "Write a function to split a string at uppercase letters.",
        "code": "import re\ndef split_upperstring(text):\n return re.findall('[A-Z][^A-Z]*', text)",
        "test_list": [
            'assert split_upperstring("PythonProgramLanguage")==[\'Python\',\'Program\',\'Language\']',
            'assert split_upperstring("PythonProgram")==[\'Python\',\'Program\']',
            'assert split_upperstring("ProgramLanguage")==[\'Program\',\'Language\']',
        ],
    },
]


class MBPPBenchmark(BenchmarkSuite):
    """MBPP: Mostly Basic Python Problems (974 problems)."""

    name = "MBPP"
    description = "Crowd-sourced basic Python programming problems"
    category = "coding"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "mbpp.jsonl"
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
        desc = problem["text"]
        tests = "\n".join(problem.get("test_list", []))
        return (
            f"{desc}\n\n"
            f"Write the Python code and save it to solution.py.\n"
            f"The following test assertions must pass:\n```python\n{tests}\n```"
        )

    def setup_workspace(self, problem: dict[str, Any], workspace: str) -> None:
        tests = problem.get("test_list", [])
        test_code = "\n".join(tests)
        harness = (
            "import sys\n"
            'sys.path.insert(0, ".")\n'
            "from solution import *\n\n"
            f"{test_code}\n\n"
            'print("ALL_TESTS_PASSED")\n'
        )
        with open(os.path.join(workspace, "test_harness.py"), "w") as fh:
            fh.write(harness)

    def recover_output_files(
        self,
        problem: dict[str, Any],
        workspace: str,
        agent_output: str,
        metadata: dict[str, Any],
    ) -> None:
        del problem
        solution_path = Path(workspace) / "solution.py"
        if solution_path.exists():
            return
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", agent_output, flags=re.DOTALL | re.IGNORECASE)
        for block in blocks:
            candidate = block.strip()
            if "def " in candidate or "import " in candidate or "class " in candidate:
                solution_path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
                metadata["recovered_solution_from_output"] = True
                return

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = str(problem.get("task_id", "unknown"))
        sol = os.path.join(workspace, "solution.py")
        if not os.path.exists(sol):
            return BenchmarkResult(problem_id=pid, passed=False, error="solution.py not found")

        code, output = self._run_shell(
            f"{sys.executable} test_harness.py", cwd=workspace, timeout=30.0
        )
        passed = code == 0 and "ALL_TESTS_PASSED" in output
        return BenchmarkResult(
            problem_id=pid, passed=passed, actual=output[:500],
            error="" if passed else output[:500],
        )
