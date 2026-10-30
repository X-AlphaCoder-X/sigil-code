"""
Aider benchmark suite.

Aider's code editing benchmark measures how well an agent can apply
specified edits to existing codebases — refactoring, adding features,
and fixing bugs based on natural language instructions.

Reference: https://aider.chat/docs/leaderboards/
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
# Built-in mini dataset — Aider-style edit tasks
# ---------------------------------------------------------------------------

_BUILTIN_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "aider-001",
        "instruction": "Add a `__len__` method to the `TaskList` class that returns the number of tasks.",
        "setup_code": textwrap.dedent("""\
            cat > tasks.py << 'PYEOF'
            class TaskList:
                def __init__(self):
                    self._tasks = []

                def add(self, task):
                    self._tasks.append(task)

                def get_all(self):
                    return list(self._tasks)
            PYEOF
        """),
        "test_code": textwrap.dedent("""\
            from tasks import TaskList
            t = TaskList()
            assert len(t) == 0
            t.add("task1")
            t.add("task2")
            assert len(t) == 2
            # Ensure original methods still work
            assert t.get_all() == ["task1", "task2"]
            print("ALL_TESTS_PASSED")
        """),
    },
    {
        "id": "aider-002",
        "instruction": (
            "Refactor the `process_data` function to use a list comprehension "
            "instead of the for loop. The behavior must remain identical."
        ),
        "setup_code": textwrap.dedent("""\
            cat > processor.py << 'PYEOF'
            def process_data(items):
                result = []
                for item in items:
                    if item > 0:
                        result.append(item * 2)
                return result
            PYEOF
        """),
        "test_code": textwrap.dedent("""\
            import inspect
            from processor import process_data
            assert process_data([1, -2, 3, -4, 5]) == [2, 6, 10]
            assert process_data([]) == []
            assert process_data([-1, -2]) == []
            # Check that a list comprehension is used
            src = inspect.getsource(process_data)
            assert 'for' in src and '[' in src, "Should use list comprehension"
            assert src.count('for') <= 2, "Should not have a separate for loop"
            print("ALL_TESTS_PASSED")
        """),
    },
    {
        "id": "aider-003",
        "instruction": (
            "Add error handling to the `divide` function so it raises a "
            "`ValueError` with message 'Division by zero' when b is 0, "
            "instead of letting ZeroDivisionError propagate."
        ),
        "setup_code": textwrap.dedent("""\
            cat > mathops.py << 'PYEOF'
            def divide(a, b):
                return a / b
            PYEOF
        """),
        "test_code": textwrap.dedent("""\
            from mathops import divide
            assert divide(10, 2) == 5.0
            assert divide(7, 2) == 3.5
            try:
                divide(1, 0)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert str(e) == 'Division by zero'
            except ZeroDivisionError:
                assert False, "Should raise ValueError, not ZeroDivisionError"
            print("ALL_TESTS_PASSED")
        """),
    },
    {
        "id": "aider-004",
        "instruction": (
            "Add a `to_dict` method to the `Config` class that returns a dictionary "
            "of all configuration key-value pairs."
        ),
        "setup_code": textwrap.dedent("""\
            cat > config.py << 'PYEOF'
            class Config:
                def __init__(self):
                    self._data = {}

                def set(self, key, value):
                    self._data[key] = value

                def get(self, key, default=None):
                    return self._data.get(key, default)
            PYEOF
        """),
        "test_code": textwrap.dedent("""\
            from config import Config
            c = Config()
            assert c.to_dict() == {}
            c.set('host', 'localhost')
            c.set('port', 8080)
            d = c.to_dict()
            assert d == {'host': 'localhost', 'port': 8080}
            # Ensure it's a copy, not a reference
            d['new'] = 'value'
            assert 'new' not in c.to_dict()
            print("ALL_TESTS_PASSED")
        """),
    },
    {
        "id": "aider-005",
        "instruction": (
            "Add a `reverse` method to the `LinkedList` class that reverses the "
            "list in-place."
        ),
        "setup_code": textwrap.dedent("""\
            cat > linkedlist.py << 'PYEOF'
            class Node:
                def __init__(self, val, next=None):
                    self.val = val
                    self.next = next

            class LinkedList:
                def __init__(self):
                    self.head = None

                def append(self, val):
                    if not self.head:
                        self.head = Node(val)
                        return
                    curr = self.head
                    while curr.next:
                        curr = curr.next
                    curr.next = Node(val)

                def to_list(self):
                    result = []
                    curr = self.head
                    while curr:
                        result.append(curr.val)
                        curr = curr.next
                    return result
            PYEOF
        """),
        "test_code": textwrap.dedent("""\
            from linkedlist import LinkedList
            ll = LinkedList()
            ll.append(1)
            ll.append(2)
            ll.append(3)
            ll.reverse()
            assert ll.to_list() == [3, 2, 1]
            # Test single element
            ll2 = LinkedList()
            ll2.append(42)
            ll2.reverse()
            assert ll2.to_list() == [42]
            # Test empty
            ll3 = LinkedList()
            ll3.reverse()
            assert ll3.to_list() == []
            print("ALL_TESTS_PASSED")
        """),
    },
    {
        "id": "aider-006",
        "instruction": (
            "Convert the `UserStore` class to use a context manager pattern. "
            "Add `__enter__` and `__exit__` methods. `__enter__` should return self, "
            "and `__exit__` should call the existing `close` method."
        ),
        "setup_code": textwrap.dedent("""\
            cat > userstore.py << 'PYEOF'
            class UserStore:
                def __init__(self):
                    self._users = {}
                    self._closed = False

                def add(self, name, email):
                    self._users[name] = email

                def get(self, name):
                    return self._users.get(name)

                def close(self):
                    self._closed = True

                @property
                def is_closed(self):
                    return self._closed
            PYEOF
        """),
        "test_code": textwrap.dedent("""\
            from userstore import UserStore
            with UserStore() as store:
                store.add("alice", "alice@example.com")
                assert store.get("alice") == "alice@example.com"
                assert not store.is_closed
            assert store.is_closed
            # Test exception in with block still calls close
            try:
                with UserStore() as store2:
                    store2.add("bob", "bob@example.com")
                    raise ValueError("test error")
            except ValueError:
                pass
            assert store2.is_closed
            print("ALL_TESTS_PASSED")
        """),
    },
]


class AiderBenchmark(BenchmarkSuite):
    """Aider: code editing benchmark."""

    name = "Aider"
    description = "Code editing and refactoring tasks"
    category = "coding"

    def load_dataset(self) -> list[dict[str, Any]]:
        jsonl_path = Path(self.data_dir) / "aider.jsonl"
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
            print(f"  {jsonl_path} not found — using built-in 6-problem subset")
        return list(_BUILTIN_PROBLEMS)

    def build_prompt(self, problem: dict[str, Any]) -> str:
        return (
            f"Edit the code in the workspace to satisfy this requirement:\n\n"
            f"{problem['instruction']}\n\n"
            f"Read the existing file(s), make the minimal changes needed, "
            f"and save the updated file(s)."
        )

    def setup_workspace(self, problem: dict[str, Any], workspace: str) -> None:
        setup = problem.get("setup_code", "")
        if setup:
            self._run_shell(setup, cwd=workspace, timeout=30.0)

        # Write test file
        test_code = problem.get("test_code", "")
        if test_code:
            with open(os.path.join(workspace, "test_harness.py"), "w") as fh:
                fh.write(test_code)

    def evaluate(self, problem: dict[str, Any], workspace: str) -> BenchmarkResult:
        pid = problem.get("id", "unknown")
        code, output = self._run_shell(
            f"{sys.executable} test_harness.py", cwd=workspace, timeout=30.0
        )
        passed = code == 0 and "ALL_TESTS_PASSED" in output
        return BenchmarkResult(
            problem_id=pid, passed=passed, actual=output[:500],
            error="" if passed else output[:500],
        )
