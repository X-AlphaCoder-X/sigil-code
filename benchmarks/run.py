#!/usr/bin/env python3
"""
Local benchmark runner for Sigil Code.

Runs the REAL agent binary against a suite of coding tasks and scores
pass/fail automatically. Tasks run in temporary folders on this machine, so no containers are needed.

Usage:
    # Run all tasks
    python3 -m benchmarks.run

    # Run a single task
    python3 -m benchmarks.run --task file-create-basic

    # Run a category
    python3 -m benchmarks.run --category bugfix

    # Run a difficulty level
    python3 -m benchmarks.run --difficulty easy

    # List available tasks
    python3 -m benchmarks.run --list

    # Verbose output
    python3 -m benchmarks.run --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.tasks.definitions import TASKS, BenchmarkTask, get_task
from benchmarks.suites.base import make_temp_workspace


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    category: str
    difficulty: str
    passed: bool
    duration_sec: float
    agent_exit_code: int
    verify_exit_code: int
    error: str = ""


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def _run_shell(cmd: str, cwd: str, timeout: float = 30.0) -> tuple[int, str]:
    """Run a shell command, return (exit_code, combined_output)."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr).strip()
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        return 1, f"[TIMEOUT after {timeout}s]"
    except Exception as exc:
        return 1, str(exc)


def run_task(
    task: BenchmarkTask,
    *,
    project_root: str,
    agent_timeout: float = 300.0,
    verbose: bool = False,
) -> TaskResult:
    """Run a single benchmark task end-to-end."""

    # Create isolated temp workspace
    workspace = make_temp_workspace("sigil_bench", task.category, task.id)

    if verbose:
        print(f"  workspace: {workspace}")

    try:
        # --- SETUP ---
        if task.setup:
            code, out = _run_shell(task.setup, cwd=workspace)
            if code != 0:
                return TaskResult(
                    task_id=task.id,
                    category=task.category,
                    difficulty=task.difficulty,
                    passed=False,
                    duration_sec=0.0,
                    agent_exit_code=-1,
                    verify_exit_code=-1,
                    error=f"Setup failed: {out}",
                )

        # --- RUN AGENT ---
        agent_cmd = (
            f"{sys.executable} -m src.main agent "
            f"{_shell_quote(task.instruction)} "
            f"--cwd {_shell_quote(workspace)} "
            f"--allow-write "
            f"--allow-shell"
        )

        if verbose:
            print(f"  agent cmd: {agent_cmd[:120]}...")

        start = time.time()
        agent_code, agent_out = _run_shell(
            agent_cmd,
            cwd=project_root,
            timeout=agent_timeout,
        )
        duration = time.time() - start

        if verbose:
            print(f"  agent exit={agent_code}  duration={duration:.1f}s")
            if agent_out:
                # Print last few lines of agent output
                lines = agent_out.split("\n")
                for line in lines[-5:]:
                    print(f"    > {line}")

        # --- VERIFY ---
        verify_code, verify_out = _run_shell(task.verify, cwd=workspace, timeout=30.0)

        if verbose:
            status = "PASS" if verify_code == 0 else "FAIL"
            print(f"  verify exit={verify_code}  -> {status}")
            if verify_code != 0 and verify_out:
                print(f"    verify output: {verify_out[:200]}")

        return TaskResult(
            task_id=task.id,
            category=task.category,
            difficulty=task.difficulty,
            passed=(verify_code == 0),
            duration_sec=duration,
            agent_exit_code=agent_code,
            verify_exit_code=verify_code,
            error=verify_out if verify_code != 0 else "",
        )

    finally:
        # Clean up workspace
        shutil.rmtree(workspace, ignore_errors=True)


def _shell_quote(s: str) -> str:
    """Quote a string for shell use."""
    import shlex
    return shlex.quote(s)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: list[TaskResult]) -> None:
    """Print a formatted results table."""

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print()
    print("=" * 72)
    print("  SIGIL CODE — BENCHMARK RESULTS")
    print("=" * 72)
    print()
    print(f"  {'Task ID':<30} {'Category':<12} {'Diff':<8} {'Result':<8} {'Time':>6}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 6}")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "  ✅" if r.passed else "  ❌"
        time_str = f"{r.duration_sec:.1f}s"
        print(f"{icon} {r.task_id:<30} {r.category:<12} {r.difficulty:<8} {status:<8} {time_str:>6}")

    print()
    print("─" * 72)
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Score: {passed}/{total} ({100*passed/total:.0f}%)")
    print("─" * 72)

    # Breakdown by category
    categories: dict[str, list[TaskResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print()
    print("  Category Breakdown:")
    for cat, cat_results in sorted(categories.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_total = len(cat_results)
        bar = "█" * cat_passed + "░" * (cat_total - cat_passed)
        print(f"    {cat:<14} {bar}  {cat_passed}/{cat_total}")

    # Breakdown by difficulty
    difficulties: dict[str, list[TaskResult]] = {}
    for r in results:
        difficulties.setdefault(r.difficulty, []).append(r)

    print()
    print("  Difficulty Breakdown:")
    for diff in ("easy", "medium", "hard"):
        if diff in difficulties:
            diff_results = difficulties[diff]
            diff_passed = sum(1 for r in diff_results if r.passed)
            diff_total = len(diff_results)
            print(f"    {diff:<14} {diff_passed}/{diff_total} ({100*diff_passed/diff_total:.0f}%)")

    total_time = sum(r.duration_sec for r in results)
    print()
    print(f"  Total time: {total_time:.1f}s")
    print()


def save_results(results: list[TaskResult], output_path: str) -> None:
    """Save results to JSON."""
    data = {
        "benchmark": "sigilcode-local",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": os.environ.get("OPENAI_MODEL", "unknown"),
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "score_pct": round(100 * sum(1 for r in results if r.passed) / len(results), 1) if results else 0,
        "results": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "duration_sec": round(r.duration_sec, 2),
                "agent_exit_code": r.agent_exit_code,
                "verify_exit_code": r.verify_exit_code,
                "error": r.error,
            }
            for r in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sigil Code local benchmark")
    parser.add_argument("--task", help="Run a single task by ID")
    parser.add_argument("--category", help="Run tasks in a category")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Run tasks by difficulty")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--timeout", type=float, default=300.0, help="Agent timeout per task in seconds (default: 300)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    args = parser.parse_args()

    if args.list:
        print(f"\n  {'ID':<30} {'Category':<12} {'Difficulty':<10}")
        print(f"  {'─' * 30} {'─' * 12} {'─' * 10}")
        for t in TASKS:
            print(f"  {t.id:<30} {t.category:<12} {t.difficulty:<10}")
        print(f"\n  Total: {len(TASKS)} tasks\n")
        return

    # Select tasks
    tasks_to_run: list[BenchmarkTask] = []

    if args.task:
        t = get_task(args.task)
        if t is None:
            print(f"Unknown task: {args.task}")
            print("Use --list to see available tasks")
            sys.exit(1)
        tasks_to_run = [t]
    else:
        tasks_to_run = list(TASKS)
        if args.category:
            tasks_to_run = [t for t in tasks_to_run if t.category == args.category]
        if args.difficulty:
            tasks_to_run = [t for t in tasks_to_run if t.difficulty == args.difficulty]

    if not tasks_to_run:
        print("No tasks matched the filters.")
        sys.exit(1)

    # Find project root
    project_root = str(Path(__file__).resolve().parent.parent)

    # Check environment
    model = os.environ.get("OPENAI_MODEL", "not set")
    base_url = os.environ.get("OPENAI_BASE_URL", "not set")

    print()
    print("=" * 72)
    print("  SIGIL CODE — LOCAL BENCHMARK")
    print("=" * 72)
    print(f"  Model:    {model}")
    print(f"  Base URL: {base_url}")
    print(f"  Tasks:    {len(tasks_to_run)}")
    print(f"  Timeout:  {args.timeout}s per task")
    print("=" * 72)
    print()

    # Run tasks
    results: list[TaskResult] = []

    for i, task in enumerate(tasks_to_run, 1):
        print(f"[{i}/{len(tasks_to_run)}] {task.id} ({task.category}, {task.difficulty})")

        result = run_task(
            task,
            project_root=project_root,
            agent_timeout=args.timeout,
            verbose=args.verbose,
        )
        results.append(result)

        status = "PASS ✅" if result.passed else "FAIL ❌"
        print(f"  -> {status}  ({result.duration_sec:.1f}s)")
        print()

    # Report
    print_results(results)

    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
