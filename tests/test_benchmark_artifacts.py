from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.suites.base import BenchmarkResult, BenchmarkSuite
from benchmarks.suites.gsm8k import GSM8KBenchmark
from benchmarks.suites.humaneval import HumanEvalBenchmark


class _DummyBenchmark(BenchmarkSuite):
    name = "DummySuite"
    description = "dummy"
    category = "coding"

    def __init__(self, *, pass_result: bool, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._pass_result = pass_result

    def load_dataset(self) -> list[dict[str, object]]:
        return [{"id": "dummy/0", "value": 1}]

    def build_prompt(self, problem: dict[str, object]) -> str:
        del problem
        return "write solution.py"

    def setup_workspace(self, problem: dict[str, object], workspace: str) -> None:
        del problem
        Path(workspace, "input.txt").write_text("fixture", encoding="utf-8")

    def run_agent(self, instruction: str, workspace: str) -> tuple[int, str, float]:
        del instruction
        Path(workspace, "solution.py").write_text("print('hello')\n", encoding="utf-8")
        return 0, "agent completed", 0.1

    def evaluate(self, problem: dict[str, object], workspace: str) -> BenchmarkResult:
        del problem, workspace
        if self._pass_result:
            return BenchmarkResult(problem_id="dummy/0", passed=True)
        return BenchmarkResult(problem_id="dummy/0", passed=False, error="boom")


class BenchmarkArtifactTests(unittest.TestCase):
    def test_failed_problem_saves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = _DummyBenchmark(
                pass_result=False,
                artifacts_dir=tmp_dir,
            )
            report = suite.run_all()
            result = report.results[0]
            artifact_path = result.metadata.get("artifact_path")
            self.assertIsInstance(artifact_path, str)
            artifact_root = Path(artifact_path)
            self.assertTrue((artifact_root / "problem.json").exists())
            self.assertTrue((artifact_root / "prompt.txt").exists())
            self.assertTrue((artifact_root / "agent_output.txt").exists())
            self.assertTrue((artifact_root / "result.json").exists())
            self.assertTrue((artifact_root / "workspace" / "solution.py").exists())
            payload = json.loads((artifact_root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["agent_exit_code"], 0)
            self.assertFalse(payload["passed"])

    def test_passing_problem_not_saved_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = _DummyBenchmark(
                pass_result=True,
                artifacts_dir=tmp_dir,
            )
            report = suite.run_all()
            result = report.results[0]
            self.assertNotIn("artifact_path", result.metadata)

    def test_passing_problem_saved_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = _DummyBenchmark(
                pass_result=True,
                artifacts_dir=tmp_dir,
                save_passing_artifacts=True,
            )
            report = suite.run_all()
            result = report.results[0]
            artifact_path = result.metadata.get("artifact_path")
            self.assertIsInstance(artifact_path, str)
            self.assertTrue(Path(str(artifact_path)).exists())

    def test_humaneval_recovers_solution_from_chat_code_block(self) -> None:
        class _RecoveringHumanEval(HumanEvalBenchmark):
            def run_agent(self, instruction: str, workspace: str) -> tuple[int, str, float]:
                del instruction, workspace
                output = """Here is the implementation:

```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```
"""
                return 0, output, 0.1

        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = _RecoveringHumanEval(
                data_dir=str(Path(tmp_dir) / "missing"),
                limit=1,
            )
            report = suite.run_all()
            result = report.results[0]
            self.assertTrue(result.passed)
            self.assertTrue(result.metadata.get("recovered_solution_from_output"))

    def test_gsm8k_recovers_answer_from_chat_output(self) -> None:
        class _RecoveringGSM8K(GSM8KBenchmark):
            def run_agent(self, instruction: str, workspace: str) -> tuple[int, str, float]:
                del instruction, workspace
                return 0, "The answer is 18.", 0.1

        with tempfile.TemporaryDirectory() as tmp_dir:
            suite = _RecoveringGSM8K(
                data_dir=str(Path(tmp_dir) / "missing"),
                limit=1,
            )
            report = suite.run_all()
            result = report.results[0]
            self.assertTrue(result.passed)
            self.assertTrue(result.metadata.get("recovered_answer_from_output"))


if __name__ == "__main__":
    unittest.main()
