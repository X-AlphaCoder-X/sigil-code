from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.download_datasets import (
    _download_gsm8k,
    _download_humaneval,
    _extract_gsm8k_answer,
    _extract_math_answer,
    _fetch_hf_rows,
    _write_manifest,
    prepare_suite,
)


class BenchmarkDatasetDownloadTests(unittest.TestCase):
    def test_extract_gsm8k_answer_uses_hash_marker(self) -> None:
        self.assertEqual(_extract_gsm8k_answer("work #### 42"), "42")
        self.assertEqual(_extract_gsm8k_answer("Total is 1,234 dollars"), "1234")

    def test_extract_math_answer_prefers_boxed_content(self) -> None:
        solution = "We compute everything and get \\boxed{\\frac{3}{8}} as the result."
        self.assertEqual(_extract_math_answer(solution), "3/8")

    def test_fetch_hf_rows_paginates(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_fetcher(
            endpoint: str,
            params: dict[str, object],
            headers: dict[str, str] | None,
            timeout: float,
        ) -> object:
            del headers, timeout
            calls.append((endpoint, dict(params)))
            if endpoint == "splits":
                return {
                    "splits": [
                        {"dataset": "demo", "config": "main", "split": "test"},
                    ]
                }
            if params["offset"] == 0:
                return {
                    "rows": [{"row": {"value": 1}}, {"row": {"value": 2}}],
                    "num_rows_total": 3,
                }
            return {
                "rows": [{"row": {"value": 3}}],
                "num_rows_total": 3,
            }

        rows = _fetch_hf_rows(
            "demo",
            config_preference=("main",),
            split_preference=("test",),
            json_fetcher=fake_fetcher,
        )
        self.assertEqual(rows, [{"value": 1}, {"value": 2}, {"value": 3}])
        self.assertEqual([call[0] for call in calls], ["splits", "rows", "rows"])

    def test_download_gsm8k_normalizes_answers(self) -> None:
        def fake_fetcher(
            endpoint: str,
            params: dict[str, object],
            headers: dict[str, str] | None,
            timeout: float,
        ) -> object:
            del headers, timeout
            if endpoint == "splits":
                return {
                    "splits": [
                        {"dataset": "openai/gsm8k", "config": "main", "split": "test"},
                    ]
                }
            self.assertEqual(params["dataset"], "openai/gsm8k")
            return {
                "rows": [
                    {"row": {"question": "q1", "answer": "reasoning #### 12"}},
                    {"row": {"question": "q2", "answer": "Total = 1,004"}},
                ],
                "num_rows_total": 2,
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "gsm8k.jsonl"
            result = _download_gsm8k(
                output_path,
                timeout=1.0,
                json_fetcher=fake_fetcher,
            )
            self.assertEqual(result.rows, 2)
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(rows[0]["answer"], "12")
            self.assertEqual(rows[1]["answer"], "1004")

    def test_download_humaneval_decompresses_gzip_payload(self) -> None:
        payload = gzip.compress(
            (
                json.dumps(
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "def f():\n    pass\n",
                        "canonical_solution": "    return 1\n",
                        "test": "def check(candidate):\n    assert True\n",
                        "entry_point": "f",
                    }
                )
                + "\n"
            ).encode("utf-8")
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "humaneval.jsonl"
            with patch(
                "benchmarks.download_datasets.fetch_bytes",
                return_value=payload,
            ):
                result = _download_humaneval(output_path, timeout=1.0)
            self.assertEqual(result.rows, 1)
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(rows[0]["task_id"], "HumanEval/0")

    def test_prepare_suite_exports_builtin_only_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = prepare_suite(
                "ifeval",
                data_dir=Path(tmp_dir),
                force=True,
                builtin_only=False,
                official_only=False,
                timeout=1.0,
            )
            output_path = Path(tmp_dir) / "ifeval.jsonl"
            self.assertEqual(result.source, "builtin")
            self.assertTrue(output_path.exists())
            self.assertGreater(result.rows, 0)

    def test_prepare_suite_falls_back_to_builtin_when_official_download_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "benchmarks.download_datasets._download_mbpp",
                side_effect=RuntimeError("network down"),
            ):
                result = prepare_suite(
                    "mbpp",
                    data_dir=Path(tmp_dir),
                    force=True,
                    builtin_only=False,
                    official_only=False,
                    timeout=1.0,
                )
            self.assertEqual(result.source, "builtin-fallback")
            self.assertIn("official download failed", result.note)
            self.assertTrue((Path(tmp_dir) / "mbpp.jsonl").exists())

    def test_write_manifest_writes_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = prepare_suite(
                "aime",
                data_dir=Path(tmp_dir),
                force=True,
                builtin_only=False,
                official_only=False,
                timeout=1.0,
            )
            manifest_path = _write_manifest(Path(tmp_dir), [result])
            payload = json.loads(manifest_path.read_text())
            self.assertEqual(payload["results"][0]["suite"], "aime")


if __name__ == "__main__":
    unittest.main()
