"""
BFCL (Berkeley Function Calling Leaderboard) benchmark suite.

BFCL tests whether an agent can correctly identify and call functions/tools
based on natural language instructions. This is directly relevant for
coding agents that use tool calling.

Reference: https://gorilla.cs.berkeley.edu/leaderboard.html
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
# Built-in mini dataset — function-calling tasks
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "bfcl-001",
        "instruction": (
            "You have access to a function `get_weather(city: str, unit: str = 'celsius') -> dict` "
            "that returns weather info. Write a Python script called solution.py that calls "
            "get_weather for 'San Francisco' with unit 'fahrenheit' and prints the result."
        ),
        "expected_function": "get_weather",
        "expected_args": {"city": "San Francisco", "unit": "fahrenheit"},
        "setup_code": (
            "cat > weather_api.py << 'PYEOF'\n"
            "def get_weather(city: str, unit: str = 'celsius') -> dict:\n"
            "    return {'city': city, 'unit': unit, 'temp': 72 if unit == 'fahrenheit' else 22}\n"
            "PYEOF"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "# Check that solution.py exists and calls get_weather correctly\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'get_weather' in code, 'Must call get_weather'\n"
            "assert 'San Francisco' in code, 'Must use San Francisco'\n"
            "assert 'fahrenheit' in code, 'Must use fahrenheit'\n"
            "exec(open('solution.py').read())\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
    {
        "id": "bfcl-002",
        "instruction": (
            "You have access to these functions:\n"
            "- `search_files(directory: str, pattern: str) -> list[str]`\n"
            "- `read_file(path: str) -> str`\n"
            "- `write_file(path: str, content: str) -> None`\n\n"
            "Write a Python script called solution.py that:\n"
            "1. Calls search_files('.', '*.txt') to find all text files\n"
            "2. Reads each file using read_file\n"
            "3. Writes a combined output using write_file to 'combined.txt'"
        ),
        "expected_function": "search_files",
        "expected_args": {"directory": ".", "pattern": "*.txt"},
        "setup_code": (
            "cat > file_api.py << 'PYEOF'\n"
            "import glob\n"
            "def search_files(directory: str, pattern: str) -> list:\n"
            "    import os\n"
            "    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt') and f != 'combined.txt']\n"
            "def read_file(path: str) -> str:\n"
            "    with open(path) as f:\n"
            "        return f.read()\n"
            "def write_file(path: str, content: str) -> None:\n"
            "    with open(path, 'w') as f:\n"
            "        f.write(content)\n"
            "PYEOF\n"
            "echo 'hello' > a.txt\n"
            "echo 'world' > b.txt\n"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'search_files' in code, 'Must call search_files'\n"
            "assert 'read_file' in code, 'Must call read_file'\n"
            "assert 'write_file' in code, 'Must call write_file'\n"
            "exec(open('solution.py').read())\n"
            "import os\n"
            "assert os.path.exists('combined.txt'), 'combined.txt must exist'\n"
            "content = open('combined.txt').read()\n"
            "assert 'hello' in content and 'world' in content\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
    {
        "id": "bfcl-003",
        "instruction": (
            "You have a function `calculate(expression: str) -> float` that evaluates "
            "a mathematical expression string. Write solution.py that:\n"
            "1. Calculates '(15 + 25) * 3'\n"
            "2. Calculates '100 / 4 - 5'\n"
            "3. Prints both results"
        ),
        "expected_function": "calculate",
        "expected_args": {"expression": "(15 + 25) * 3"},
        "setup_code": (
            "cat > calc_api.py << 'PYEOF'\n"
            "def calculate(expression: str) -> float:\n"
            "    return float(eval(expression))\n"
            "PYEOF"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'calculate' in code, 'Must call calculate'\n"
            "exec(open('solution.py').read())\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
    {
        "id": "bfcl-004",
        "instruction": (
            "You have these functions:\n"
            "- `create_user(name: str, email: str, role: str = 'user') -> dict`\n"
            "- `delete_user(user_id: int) -> bool`\n"
            "- `list_users() -> list[dict]`\n\n"
            "Write solution.py that creates a user named 'Alice' with email "
            "'alice@example.com' and role 'admin', then lists all users."
        ),
        "expected_function": "create_user",
        "expected_args": {"name": "Alice", "email": "alice@example.com", "role": "admin"},
        "setup_code": (
            "cat > user_api.py << 'PYEOF'\n"
            "_users = []\n"
            "_next_id = 1\n"
            "def create_user(name: str, email: str, role: str = 'user') -> dict:\n"
            "    global _next_id\n"
            "    user = {'id': _next_id, 'name': name, 'email': email, 'role': role}\n"
            "    _users.append(user)\n"
            "    _next_id += 1\n"
            "    return user\n"
            "def delete_user(user_id: int) -> bool:\n"
            "    global _users\n"
            "    _users = [u for u in _users if u['id'] != user_id]\n"
            "    return True\n"
            "def list_users() -> list:\n"
            "    return list(_users)\n"
            "PYEOF"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'create_user' in code, 'Must call create_user'\n"
            "assert 'Alice' in code, 'Must use Alice'\n"
            "assert 'alice@example.com' in code, 'Must use correct email'\n"
            "assert 'admin' in code, 'Must use admin role'\n"
            "assert 'list_users' in code, 'Must call list_users'\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
    {
        "id": "bfcl-005",
        "instruction": (
            "You have a function `send_email(to: str, subject: str, body: str, "
            "cc: list[str] = None) -> bool`. Write solution.py that sends an email "
            "to 'boss@company.com' with subject 'Weekly Report', body 'Please see attached report.', "
            "and cc=['team@company.com']."
        ),
        "expected_function": "send_email",
        "expected_args": {
            "to": "boss@company.com",
            "subject": "Weekly Report",
            "body": "Please see attached report.",
            "cc": ["team@company.com"],
        },
        "setup_code": (
            "cat > email_api.py << 'PYEOF'\n"
            "def send_email(to: str, subject: str, body: str, cc: list = None) -> bool:\n"
            "    print(f'Email sent to {to}, subject: {subject}')\n"
            "    return True\n"
            "PYEOF"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'send_email' in code, 'Must call send_email'\n"
            "assert 'boss@company.com' in code, 'Must use correct recipient'\n"
            "assert 'Weekly Report' in code, 'Must use correct subject'\n"
            "assert 'team@company.com' in code, 'Must include cc'\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
    {
        "id": "bfcl-006",
        "instruction": (
            "You have these functions:\n"
            "- `connect_db(host: str, port: int, database: str) -> object`\n"
            "- `execute_query(connection: object, query: str) -> list`\n"
            "- `close_db(connection: object) -> None`\n\n"
            "Write solution.py that connects to host='localhost', port=5432, database='mydb', "
            "executes 'SELECT * FROM users', and closes the connection."
        ),
        "expected_function": "connect_db",
        "expected_args": {"host": "localhost", "port": 5432, "database": "mydb"},
        "setup_code": (
            "cat > db_api.py << 'PYEOF'\n"
            "class FakeConnection:\n"
            "    def __init__(self, host, port, db):\n"
            "        self.host = host\n"
            "        self.port = port\n"
            "        self.db = db\n"
            "        self.closed = False\n"
            "def connect_db(host: str, port: int, database: str) -> object:\n"
            "    return FakeConnection(host, port, database)\n"
            "def execute_query(connection: object, query: str) -> list:\n"
            "    return [{'id': 1, 'name': 'test'}]\n"
            "def close_db(connection: object) -> None:\n"
            "    connection.closed = True\n"
            "PYEOF"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'connect_db' in code, 'Must call connect_db'\n"
            "assert 'execute_query' in code, 'Must call execute_query'\n"
            "assert 'close_db' in code, 'Must call close_db'\n"
            "assert 'localhost' in code, 'Must use localhost'\n"
            "assert '5432' in code, 'Must use port 5432'\n"
            "assert 'mydb' in code, 'Must use mydb database'\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
    {
        "id": "bfcl-007",
        "instruction": (
            "You have a function `sort_data(data: list, key: str, reverse: bool = False) -> list`. "
            "Write solution.py that sorts the list [{'name': 'Charlie', 'age': 30}, "
            "{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 35}] by 'age' in "
            "descending order (reverse=True) and prints the result."
        ),
        "expected_function": "sort_data",
        "expected_args": {"key": "age", "reverse": True},
        "setup_code": (
            "cat > sort_api.py << 'PYEOF'\n"
            "def sort_data(data: list, key: str, reverse: bool = False) -> list:\n"
            "    return sorted(data, key=lambda x: x[key], reverse=reverse)\n"
            "PYEOF"
        ),
        "test_code": (
            "import sys\n"
            "sys.path.insert(0, '.')\n"
            "with open('solution.py') as f:\n"
            "    code = f.read()\n"
            "assert 'sort_data' in code, 'Must call sort_data'\n"
            "assert 'age' in code, 'Must sort by age'\n"
            "assert 'reverse' in code or 'True' in code, 'Must use reverse order'\n"
            "print('ALL_TESTS_PASSED')\n"
        ),
    },
]


class BFCLBenchmark(BenchmarkSuite):
    """BFCL: Berkeley Function Calling Leaderboard."""

    name = "BFCL"
    description = "Function/tool calling evaluation"
    category = "instruction-following"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "bfcl.jsonl"
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
            print(f"  {jsonl_path} not found — using built-in 7-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        return problem["instruction"]

    def setup_workspace(self, problem: dict[str, Any], workspace: str) -> None:
        setup = problem.get("setup_code", "")
        if setup:
            self._run_shell(setup, cwd=workspace, timeout=30.0)

        test_code = problem.get("test_code", "")
        if test_code:
            with open(os.path.join(workspace, "test_harness.py"), "w") as fh:
                fh.write(test_code)

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")

        sol = os.path.join(workspace, "solution.py")
        if not os.path.exists(sol):
            return BenchmarkResult(
                problem_id=pid, passed=False, error="solution.py not found"
            )

        output = ""
        test_harness = os.path.join(workspace, "test_harness.py")
        if os.path.exists(test_harness):
            code, output = self._run_shell(
                f"{sys.executable} test_harness.py", cwd=workspace, timeout=30.0
            )
            passed = code == 0 and "ALL_TESTS_PASSED" in output
        else:
            # Basic check: verify the expected function is called
            with open(sol) as fh:
                code_content = fh.read()
            expected_fn = problem.get("expected_function", "")
            passed = expected_fn in code_content if expected_fn else True

        return BenchmarkResult(
            problem_id=pid, passed=passed,
            actual=output[:500],
            error="" if passed else (output[:500] if output else "function call not found"),
        )
