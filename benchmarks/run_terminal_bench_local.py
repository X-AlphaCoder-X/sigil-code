#!/usr/bin/env python3
"""
Run Terminal-Bench tasks locally with Apptainer and Sigil Code.

Each task's Docker image (named in its task.toml) is pulled and run with Apptainer, so no Docker
daemon is needed. The model server must be reachable from the machine that runs the containers.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_TASKS_DIR = Path.home() / ".cache/harbor/tasks/packages/terminal-bench"
DEFAULT_JOBS_DIR = Path("jobs/terminal_bench_local")


@dataclass
class TerminalBenchTask:
    task_dir: Path
    name: str
    short_name: str
    instruction: str
    docker_image: str | None
    agent_timeout_sec: float | None
    verifier_timeout_sec: float
    workdir: str
    has_docker_compose: bool
    raw_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalTrialResult:
    task_name: str
    short_name: str
    passed: bool
    reward: float | None
    duration_sec: float
    status: str
    image: str | None
    workdir: str
    trial_dir: str
    error: str = ""
    agent_return_code: int | None = None
    verifier_return_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def strip_canary(text: str) -> str:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("<!--") and "canary" in stripped.lower():
            index += 1
            continue
        if stripped.startswith("#") and "canary" in stripped.lower():
            index += 1
            continue
        break
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


def parse_dockerfile_workdir(dockerfile_path: Path) -> str:
    if not dockerfile_path.exists():
        return "/workspace"
    workdir = "/workspace"
    pattern = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE)
    for raw_line in dockerfile_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if value.startswith("["):
            continue
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        workdir = value
    return workdir


def load_task(task_dir: Path) -> TerminalBenchTask:
    config = _load_toml(task_dir / "task.toml")
    task_config = config.get("task") or {}
    environment = config.get("environment") or {}
    agent = config.get("agent") or {}
    verifier = config.get("verifier") or {}
    name = task_config.get("name", task_dir.name)
    short_name = name.split("/", 1)[-1]
    instruction = strip_canary((task_dir / "instruction.md").read_text(encoding="utf-8"))
    environment_dir = task_dir / "environment"
    return TerminalBenchTask(
        task_dir=task_dir,
        name=name,
        short_name=short_name,
        instruction=instruction,
        docker_image=environment.get("docker_image"),
        agent_timeout_sec=agent.get("timeout_sec"),
        verifier_timeout_sec=float(verifier.get("timeout_sec", 600.0)),
        workdir=parse_dockerfile_workdir(environment_dir / "Dockerfile"),
        has_docker_compose=(environment_dir / "docker-compose.yaml").exists(),
        raw_config=config,
    )


def discover_tasks(root: Path) -> list[TerminalBenchTask]:
    if (root / "task.toml").exists():
        return [load_task(root)]
    tasks: list[TerminalBenchTask] = []
    for config_path in sorted(root.rglob("task.toml")):
        try:
            tasks.append(load_task(config_path.parent))
        except Exception:
            continue
    return tasks


def filter_tasks(
    tasks: list[TerminalBenchTask],
    *,
    include_patterns: list[str],
    exclude_patterns: list[str],
    limit: int | None,
) -> list[TerminalBenchTask]:
    selected: list[TerminalBenchTask] = []
    for task in tasks:
        candidates = {task.name, task.short_name, task.task_dir.name}
        if include_patterns and not any(
            fnmatch.fnmatchcase(candidate, pattern)
            for pattern in include_patterns
            for candidate in candidates
        ):
            continue
        if any(
            fnmatch.fnmatchcase(candidate, pattern)
            for pattern in exclude_patterns
            for candidate in candidates
        ):
            continue
        selected.append(task)
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_shell(
    cmd: str,
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


def ensure_apptainer_image(task: TerminalBenchTask, images_dir: Path, force_pull: bool) -> Path:
    if not task.docker_image:
        raise ValueError("task has no docker_image")
    images_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"{safe_name(task.name)}.sif"
    if image_path.exists() and not force_pull:
        return image_path
    cmd = f"apptainer pull --force {shlex.quote(str(image_path))} docker://{shlex.quote(task.docker_image)}"
    result = run_shell(cmd, timeout=3600.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"apptainer pull failed for {task.docker_image}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return image_path


def seed_workspace(task: TerminalBenchTask, image_path: Path, workspace_dir: Path) -> None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    marker = workspace_dir / ".seed_complete"
    if marker.exists():
        return

    copy_cmd = (
        "set -euo pipefail; "
        f"if [ -d {shlex.quote(task.workdir)} ]; then "
        f"mkdir -p /mnt/workspace && cp -a {shlex.quote(task.workdir)}/. /mnt/workspace/; "
        "fi"
    )
    cmd = (
        f"apptainer exec --bind {shlex.quote(str(workspace_dir))}:/mnt/workspace:rw "
        f"{shlex.quote(str(image_path))} "
        f"bash -lc {shlex.quote(copy_cmd)}"
    )
    result = run_shell(cmd, timeout=1800.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to seed workspace from {task.workdir}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    marker.write_text("ok\n", encoding="utf-8")


def read_reward(verifier_dir: Path) -> tuple[float | None, dict[str, Any]]:
    reward_txt = verifier_dir / "reward.txt"
    reward_json = verifier_dir / "reward.json"
    if reward_txt.exists():
        raw = reward_txt.read_text(encoding="utf-8").strip()
        return float(raw), {"reward_source": "reward.txt"}
    if reward_json.exists():
        payload = json.loads(reward_json.read_text(encoding="utf-8"))
        reward = payload.get("reward")
        if reward is None:
            numeric = [value for value in payload.values() if isinstance(value, (int, float))]
            reward = float(numeric[0]) if numeric else None
        return (float(reward) if reward is not None else None), {"reward_source": "reward.json", "reward_payload": payload}
    return None, {}


def build_host_agent_command(
    *,
    task: TerminalBenchTask,
    workspace_dir: Path,
    repo_dir: Path,
    agent_logs_dir: Path,
) -> str:
    instruction_file = agent_logs_dir / "instruction.txt"
    instruction_file.write_text(task.instruction, encoding="utf-8")
    stdout_path = agent_logs_dir / "stdout.txt"
    stderr_path = agent_logs_dir / "stderr.txt"
    return (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(repo_dir))}; "
        f"instruction=$(cat {shlex.quote(str(instruction_file))}); "
        f"{shlex.quote(sys.executable)} -m src.main agent \"$instruction\" "
        f"--cwd {shlex.quote(str(workspace_dir))} --allow-write --allow-shell "
        f"> {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}"
    )


def build_verifier_exec_command(
    *,
    task: TerminalBenchTask,
    image_path: Path,
    workspace_dir: Path,
    task_dir: Path,
    verifier_logs_dir: Path,
    env: dict[str, str],
    fakeroot: bool = False,
) -> str:
    binds = [
        f"{workspace_dir}:{task.workdir}:rw",
        f"{task_dir / 'tests'}:/tests:ro",
        f"{verifier_logs_dir}:/logs/verifier:rw",
    ]
    bind_flags = " ".join(f"--bind {shlex.quote(spec)}" for spec in binds)

    # When using fakeroot (no-sudo HPC), inject env vars needed for apt-get
    # and SSL inside the container.
    if fakeroot:
        fakeroot_env = {
            "TMPDIR": "/tmp",
            "DEBIAN_FRONTEND": "noninteractive",
            "CURL_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
            "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
        }
        env = {**fakeroot_env, **env}

    env_flags = " ".join(
        f"--env {shlex.quote(key)}={shlex.quote(value)}"
        for key, value in sorted(env.items())
    )
    inner = (
        "set -euo pipefail; "
        "chmod +x /tests/test.sh 2>/dev/null || true; "
        f"cd {shlex.quote(task.workdir)}; "
        "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2> /logs/verifier/test-stderr.txt"
    )
    # --fakeroot: simulate root inside the container (for apt-get etc.)
    # --writable-tmpfs: in-memory overlay so packages can be installed
    # --contain: prevent host dirs from leaking into container
    fakeroot_flags = "--fakeroot --writable-tmpfs --contain " if fakeroot else ""
    return (
        f"apptainer exec --cleanenv {fakeroot_flags}{bind_flags} {env_flags} "
        f"--cwd {shlex.quote(task.workdir)} {shlex.quote(str(image_path))} "
        f"bash -lc {shlex.quote(inner)}"
    )


def run_trial(
    task: TerminalBenchTask,
    *,
    repo_dir: Path,
    jobs_dir: Path,
    images_dir: Path,
    force_pull: bool,
    keep_images: bool,
    timeout_multiplier: float,
    fakeroot: bool = False,
) -> LocalTrialResult:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    trial_dir = jobs_dir / f"{timestamp}_{safe_name(task.short_name)}"
    workspace_dir = trial_dir / "workspace"
    agent_logs_dir = trial_dir / "agent"
    verifier_logs_dir = trial_dir / "verifier"
    for path in (trial_dir, workspace_dir, agent_logs_dir, verifier_logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = LocalTrialResult(
        task_name=task.name,
        short_name=task.short_name,
        passed=False,
        reward=None,
        duration_sec=0.0,
        status="pending",
        image=task.docker_image,
        workdir=task.workdir,
        trial_dir=str(trial_dir),
    )

    if task.has_docker_compose:
        result.status = "skipped"
        result.error = "docker-compose task is not supported by the local Apptainer runner"
        result.duration_sec = time.time() - start
        return result
    if not task.docker_image:
        result.status = "skipped"
        result.error = "task has no prebuilt docker_image in task.toml"
        result.duration_sec = time.time() - start
        return result

    image_path = ensure_apptainer_image(task, images_dir, force_pull)
    if not keep_images:
        result.metadata["image_path"] = str(image_path)
    seed_workspace(task, image_path, workspace_dir)

    env = {
        key: os.environ[key]
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
        if os.environ.get(key)
    }
    env.update({str(k): str(v) for k, v in (task.raw_config.get("environment") or {}).get("env", {}).items()})
    verifier_env = {str(k): str(v) for k, v in (task.raw_config.get("verifier") or {}).get("env", {}).items()}

    agent_cmd = build_host_agent_command(
        task=task,
        workspace_dir=workspace_dir,
        repo_dir=repo_dir,
        agent_logs_dir=agent_logs_dir,
    )
    verifier_cmd = build_verifier_exec_command(
        task=task,
        image_path=image_path,
        workspace_dir=workspace_dir,
        task_dir=task.task_dir,
        verifier_logs_dir=verifier_logs_dir,
        env=verifier_env,
        fakeroot=fakeroot,
    )

    agent_timeout = (task.agent_timeout_sec or 1800.0) * timeout_multiplier
    verifier_timeout = task.verifier_timeout_sec * timeout_multiplier

    try:
        agent_proc = run_shell(agent_cmd, timeout=agent_timeout, env={**os.environ, **env})
        result.agent_return_code = agent_proc.returncode
        if agent_proc.returncode != 0:
            result.status = "agent_failed"
            stdout_path = agent_logs_dir / "stdout.txt"
            stderr_path = agent_logs_dir / "stderr.txt"
            stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore") if stdout_path.exists() else agent_proc.stdout
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore") if stderr_path.exists() else agent_proc.stderr
            result.error = (stdout_text + "\n" + stderr_text).strip()[:4000]
            result.duration_sec = time.time() - start
            return result

        verifier_proc = run_shell(verifier_cmd, timeout=verifier_timeout)
        result.verifier_return_code = verifier_proc.returncode
        reward, reward_metadata = read_reward(verifier_logs_dir)
        result.reward = reward
        result.metadata.update(reward_metadata)
        result.passed = verifier_proc.returncode == 0 and reward is not None and reward > 0.0
        result.status = "passed" if result.passed else "failed"
        if not result.passed:
            stdout_text = (verifier_logs_dir / "test-stdout.txt").read_text(encoding="utf-8", errors="ignore") if (verifier_logs_dir / "test-stdout.txt").exists() else ""
            stderr_text = (verifier_logs_dir / "test-stderr.txt").read_text(encoding="utf-8", errors="ignore") if (verifier_logs_dir / "test-stderr.txt").exists() else ""
            result.error = (stdout_text + "\n" + stderr_text).strip()[:4000]
        return result
    except subprocess.TimeoutExpired as exc:
        result.status = "timeout"
        result.error = f"timeout after {exc.timeout}s"
        return result
    except Exception as exc:
        result.status = "error"
        result.error = str(exc)
        return result
    finally:
        result.duration_sec = time.time() - start


def print_report(results: list[LocalTrialResult]) -> None:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    failed = sum(1 for item in results if item.status == "failed")
    skipped = sum(1 for item in results if item.status == "skipped")
    print()
    print("=" * 80)
    print("  TERMINAL-BENCH LOCAL REPORT")
    print("=" * 80)
    for item in results:
        icon = "PASS ✅" if item.passed else ("SKIP ⚪" if item.status == "skipped" else "FAIL ❌")
        print(f"  {icon:<8} {item.short_name:<32} {item.duration_sec:>7.1f}s  {item.status}")
    print("─" * 80)
    print(f"  Total: {total}  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    score = (100.0 * passed / total) if total else 0.0
    print(f"  Score: {score:.1f}%")
    print("─" * 80)


def save_results(path: Path, results: list[LocalTrialResult]) -> None:
    payload = {
        "benchmark": "terminal-bench-local-apptainer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": os.environ.get("OPENAI_MODEL", "unknown"),
        "results": [asdict(item) for item in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Terminal-Bench tasks locally with Apptainer and Sigil Code.")
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR, help="Directory containing downloaded Harbor task packages.")
    parser.add_argument("--include-task-name", "-i", action="append", default=[], help="Include only task names matching this glob pattern.")
    parser.add_argument("--exclude-task-name", "-x", action="append", default=[], help="Exclude task names matching this glob pattern.")
    parser.add_argument("--n-tasks", "-l", type=int, default=None, help="Maximum number of tasks to run after filtering.")
    parser.add_argument("--jobs-dir", "-o", type=Path, default=DEFAULT_JOBS_DIR, help="Directory where trial outputs will be written.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_JOBS_DIR / "_images", help="Directory where pulled Apptainer images will be cached.")
    parser.add_argument("--force-pull", action="store_true", help="Re-pull Apptainer images even if cached locally.")
    parser.add_argument("--keep-images", action="store_true", help="Keep pulled SIF images in the image cache directory.")
    parser.add_argument("--timeout-multiplier", type=float, default=1.0, help="Multiplier applied to agent and verifier timeouts.")
    parser.add_argument("--output", type=Path, help="Optional JSON file for the run summary.")
    parser.add_argument("--list", action="store_true", help="List discovered tasks and exit.")
    parser.add_argument(
        "--fakeroot",
        action="store_true",
        help="Use Apptainer --fakeroot + --writable-tmpfs for the verifier container. "
        "Required on HPC systems without sudo where test.sh scripts need apt-get.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.tasks_dir.exists():
        parser.error(f"tasks directory not found: {args.tasks_dir}")

    tasks = discover_tasks(args.tasks_dir)
    tasks = filter_tasks(
        tasks,
        include_patterns=args.include_task_name,
        exclude_patterns=args.exclude_task_name,
        limit=args.n_tasks,
    )

    if args.list:
        for task in tasks:
            print(task.short_name)
        return

    if not tasks:
        parser.error("no tasks matched")

    repo_dir = Path(__file__).resolve().parent.parent
    results: list[LocalTrialResult] = []
    print()
    print("=" * 80)
    print("  TERMINAL-BENCH LOCAL")
    print("=" * 80)
    print(f"  Tasks:    {len(tasks)}")
    print(f"  Jobs dir: {args.jobs_dir}")
    if args.fakeroot:
        print("  Fakeroot: enabled (no-sudo HPC mode)")
    print("=" * 80)
    print()

    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] {task.short_name}")
        result = run_trial(
            task,
            repo_dir=repo_dir,
            jobs_dir=args.jobs_dir,
            images_dir=args.images_dir,
            force_pull=args.force_pull,
            keep_images=args.keep_images,
            timeout_multiplier=args.timeout_multiplier,
            fakeroot=args.fakeroot,
        )
        results.append(result)
        icon = "PASS ✅" if result.passed else ("SKIP ⚪" if result.status == "skipped" else "FAIL ❌")
        print(f"  -> {icon}  ({result.duration_sec:.1f}s)")
        print(f"     trial_dir: {result.trial_dir}")
        if result.error:
            print(f"     error: {result.error[:200]}")
        print()

    print_report(results)
    if args.output:
        save_results(args.output, results)


if __name__ == "__main__":
    main()
