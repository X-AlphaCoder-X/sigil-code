from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


DEFAULT_BACKGROUND_DIR = Path('.port_sessions') / 'background'
_DETACHED_PROCESSES: dict[int, subprocess.Popen[Any]] = {}


@dataclass(frozen=True)
class BackgroundSessionRecord:
    background_id: str
    pid: int
    prompt: str
    workspace_cwd: str
    model: str
    mode: str
    status: str
    log_path: str
    record_path: str
    started_at: str
    command: tuple[str, ...]
    finished_at: str | None = None
    exit_code: int | None = None
    stop_reason: str | None = None
    session_id: str | None = None
    session_path: str | None = None
    # Live note from the worker, e.g. 'blocked: needs approval (Bash(git push *))'.
    note: str | None = None
    # OS creation time of the worker process (epoch seconds). A live PID with another creation
    # time is a different program that reused the number: it is never signalled.
    pid_started: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'BackgroundSessionRecord':
        return cls(
            background_id=str(payload.get('background_id') or ''),
            pid=int(payload.get('pid') or 0),
            prompt=str(payload.get('prompt') or ''),
            workspace_cwd=str(payload.get('workspace_cwd') or ''),
            model=str(payload.get('model') or ''),
            mode=str(payload.get('mode') or 'agent'),
            status=str(payload.get('status') or 'unknown'),
            log_path=str(payload.get('log_path') or ''),
            record_path=str(payload.get('record_path') or ''),
            started_at=str(payload.get('started_at') or ''),
            command=tuple(
                str(item)
                for item in payload.get('command', [])
                if isinstance(item, (str, int, float))
            ),
            finished_at=(
                str(payload.get('finished_at'))
                if isinstance(payload.get('finished_at'), str) and payload.get('finished_at')
                else None
            ),
            exit_code=(
                int(payload.get('exit_code'))
                if isinstance(payload.get('exit_code'), int)
                else None
            ),
            stop_reason=(
                str(payload.get('stop_reason'))
                if isinstance(payload.get('stop_reason'), str) and payload.get('stop_reason')
                else None
            ),
            session_id=(
                str(payload.get('session_id'))
                if isinstance(payload.get('session_id'), str) and payload.get('session_id')
                else None
            ),
            session_path=(
                str(payload.get('session_path'))
                if isinstance(payload.get('session_path'), str) and payload.get('session_path')
                else None
            ),
            note=(str(payload['note']) if isinstance(payload.get('note'), str) and payload.get('note') else None),
            pid_started=(float(payload['pid_started'])
                         if isinstance(payload.get('pid_started'), (int, float)) else None),
        )


class BackgroundSessionRuntime:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or DEFAULT_BACKGROUND_DIR).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_id(self) -> str:
        return f'bg_{uuid4().hex[:12]}'

    def record_path(self, background_id: str) -> Path:
        return self.root / f'{background_id}.json'

    def log_path(self, background_id: str) -> Path:
        return self.root / f'{background_id}.log'

    def result_path(self, background_id: str) -> Path:
        return self.root / f'{background_id}.result'

    def write_result(self, background_id: str, payload: dict[str, Any]) -> None:
        """The worker's final answer, session id, cost and stop reason (read by status)."""
        path = self.result_path(background_id)
        from .settings import write_text_atomic

        write_text_atomic(path, json.dumps(payload, ensure_ascii=True, indent=2))  # read by status

    def read_result(self, background_id: str) -> dict[str, Any] | None:
        try:
            from .settings import read_text_retry

            data = json.loads(read_text_retry(self.result_path(background_id)))
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            return None
        return data if isinstance(data, dict) else None

    def running_count(self) -> int:
        return sum(1 for record in self.list_records() if record.status == 'running')

    def check_stalled(self, record: 'BackgroundSessionRecord') -> 'BackgroundSessionRecord':
        """Mark a running agent 'stalled' when its log has been quiet too long.

        SIGIL_BACKGROUND_STALL_MINUTES (default 15, 0 = off) sets the quiet time;
        SIGIL_BACKGROUND_STALL_KILL=1 also stops it.
        """
        import os
        import time as _time

        if record.status != 'running':
            return record
        try:
            minutes = float(os.environ.get('SIGIL_BACKGROUND_STALL_MINUTES', '') or 15)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            minutes = 15.0
        if minutes <= 0:
            return record
        try:
            quiet = _time.time() - Path(record.log_path).stat().st_mtime
        except OSError as _ignored:
            _log_ignored(_ignored)
            return record
        if quiet < minutes * 60:
            return record
        note = f'stalled: no output for {int(quiet // 60)} min'
        if os.environ.get('SIGIL_BACKGROUND_STALL_KILL', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
            killed = self.kill(record.background_id)
            self.set_note(record.background_id, note + ' (stopped)')
            return self.load_record(killed.background_id)
        if record.note != note:
            self.set_note(record.background_id, note)
        return self.load_record(record.background_id)

    def launch(self, command: list[str], **kwargs: Any) -> BackgroundSessionRecord:
        """Start a worker. The SIGIL_MAX_BACKGROUND_AGENTS check and the start happen under one lock,
        so parallel starts in this process cannot pass the limit together (separate Sigil processes
        still check on their own)."""
        with _LAUNCH_LOCK:
            message = background_limit_message(self)
            if message:
                raise RuntimeError(message)
            return self._launch(command, **kwargs)

    def _launch(
        self,
        command: list[str],
        *,
        prompt: str,
        workspace_cwd: Path,
        model: str,
        mode: str = 'agent',
        background_id: str | None = None,
        process_cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> BackgroundSessionRecord:
        background_id = background_id or self.create_id()
        log_path = self.log_path(background_id)
        record_path = self.record_path(background_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == 'win32':
            # start_new_session is ignored on Windows: detach from our console so
            # console control events (Ctrl+C) can never cross between the processes.
            platform_kwargs: dict[str, Any] = {
                'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
            }
        else:
            platform_kwargs = {'start_new_session': True}
        with log_path.open('a', encoding='utf-8') as handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=str(process_cwd or Path.cwd()),
                env=env,
                **platform_kwargs,
            )
        _DETACHED_PROCESSES[process.pid] = process
        record = BackgroundSessionRecord(
            background_id=background_id,
            pid=process.pid,
            prompt=prompt,
            workspace_cwd=str(workspace_cwd),
            model=model,
            mode=mode,
            status='running',
            log_path=str(log_path),
            record_path=str(record_path),
            started_at=_utc_now(),
            command=tuple(command),
            pid_started=_process_start_time(process.pid),
        )
        self.save_record(record)
        return record

    def save_record(self, record: BackgroundSessionRecord) -> Path:
        path = Path(record.record_path)
        from .settings import write_text_atomic

        # why atomic: agent-ps, the chat and the worker read and write it at the same time
        write_text_atomic(path, json.dumps(asdict(record), ensure_ascii=True, indent=2))
        return path

    def load_record(self, background_id: str) -> BackgroundSessionRecord:
        from .settings import read_text_retry

        data = json.loads(read_text_retry(self.record_path(background_id)))  # the worker may be saving it
        record = BackgroundSessionRecord.from_dict(data)
        return self.refresh_record(record)

    def list_records(self) -> tuple[BackgroundSessionRecord, ...]:
        records: list[BackgroundSessionRecord] = []
        for path in sorted(self.root.glob('bg_*.json')):
            try:
                from .settings import read_text_retry

                payload = json.loads(read_text_retry(path))
            except (OSError, json.JSONDecodeError) as _ignored:
                _log_ignored(_ignored)
                continue
            records.append(self.refresh_record(BackgroundSessionRecord.from_dict(payload)))
        return tuple(sorted(records, key=lambda item: item.started_at, reverse=True))

    def refresh_record(self, record: BackgroundSessionRecord) -> BackgroundSessionRecord:
        if record.status != 'running':
            return record
        alive = _is_process_running(record.pid)
        if alive and _same_process(record):
            return record
        updated = BackgroundSessionRecord(
            background_id=record.background_id,
            pid=record.pid,
            prompt=record.prompt,
            workspace_cwd=record.workspace_cwd,
            model=record.model,
            mode=record.mode,
            status='exited',
            log_path=record.log_path,
            record_path=record.record_path,
            started_at=record.started_at,
            command=record.command,
            finished_at=record.finished_at or _utc_now(),
            exit_code=record.exit_code,
            stop_reason=record.stop_reason or (
                'process gone (its PID now belongs to another program)' if alive else None),
            session_id=record.session_id,
            session_path=record.session_path,
            note=record.note,  # why: refresh and kill used to drop why a worker stopped
            pid_started=record.pid_started,
        )
        self.save_record(updated)
        process = _DETACHED_PROCESSES.pop(record.pid, None)
        if process is not None and process.returncode is None:
            process.returncode = updated.exit_code
        return updated

    def mark_finished(
        self,
        background_id: str,
        *,
        exit_code: int,
        stop_reason: str | None = None,
        session_id: str | None = None,
        session_path: str | None = None,
        status: str | None = None,
    ) -> BackgroundSessionRecord:
        record = self.load_record(background_id)
        final_status = status or ('completed' if exit_code == 0 else 'failed')
        updated = BackgroundSessionRecord(
            background_id=record.background_id,
            pid=record.pid,
            prompt=record.prompt,
            workspace_cwd=record.workspace_cwd,
            model=record.model,
            mode=record.mode,
            status=final_status,
            log_path=record.log_path,
            record_path=record.record_path,
            started_at=record.started_at,
            command=record.command,
            finished_at=_utc_now(),
            exit_code=exit_code,
            stop_reason=stop_reason,
            session_id=session_id,
            session_path=session_path,
            note=record.note,  # why: refresh and kill used to drop why a worker stopped
            pid_started=record.pid_started,
        )
        self.save_record(updated)
        process = _DETACHED_PROCESSES.pop(record.pid, None)
        if process is not None and process.returncode is None:
            process.returncode = updated.exit_code
        return updated

    def kill(self, background_id: str) -> BackgroundSessionRecord:
        record = self.load_record(background_id)
        if record.status != 'running':
            return record
        from .process_group import safe_kill_pid, safe_killpg

        # the pid comes from a record file in the workspace: never our own process, our parent or a
        # system pid (taskkill /T or a group signal there would end Sigil itself)
        foreign = record.pid > 4 and record.pid not in (os.getpid(), os.getppid())
        if not foreign:
            from .debug_log import get_logger

            get_logger('background').warning('not stopping %s: its pid %s is ours or a system pid',
                                              record.background_id, record.pid)
        elif sys.platform == 'win32':
            # os.killpg/os.WNOHANG do not exist on Windows; terminate the whole tree.
            subprocess.run(
                ['taskkill', '/PID', str(record.pid), '/T', '/F'],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            # the worker leads its own group (start_new_session): signal that group, or the pid alone
            if not safe_killpg(record.pid, signal.SIGTERM):
                safe_kill_pid(record.pid, signal.SIGTERM)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if sys.platform != 'win32':
                try:
                    waited_pid, _ = os.waitpid(record.pid, os.WNOHANG)
                    if waited_pid == record.pid:
                        break
                except ChildProcessError as _ignored:
                    _log_ignored(_ignored)
                    break
                except OSError as _ignored:
                    _log_ignored(_ignored)
                    break
            if not _is_process_running(record.pid):
                break
            time.sleep(0.05)
        else:
            if sys.platform != 'win32' and foreign:
                # Still alive after SIGTERM: kill the whole process group (its shells and MCP servers).
                if not safe_killpg(record.pid, signal.SIGKILL):
                    safe_kill_pid(record.pid, signal.SIGKILL)
        process = _DETACHED_PROCESSES.get(record.pid)
        if process is not None:
            try:
                process.wait(timeout=0.5)  # reap our own child so it is not left as a zombie handle
            except subprocess.TimeoutExpired as _ignored:
                _log_ignored(_ignored)
                pass
        updated = BackgroundSessionRecord(
            background_id=record.background_id,
            pid=record.pid,
            prompt=record.prompt,
            workspace_cwd=record.workspace_cwd,
            model=record.model,
            mode=record.mode,
            status='killed',
            log_path=record.log_path,
            record_path=record.record_path,
            started_at=record.started_at,
            command=record.command,
            finished_at=_utc_now(),
            exit_code=-signal.SIGTERM,
            stop_reason='killed',
            session_id=record.session_id,
            session_path=record.session_path,
            note=record.note,  # why: refresh and kill used to drop why a worker stopped
            pid_started=record.pid_started,
        )
        self.save_record(updated)
        process = _DETACHED_PROCESSES.pop(record.pid, None)
        if process is not None and process.returncode is None:
            process.returncode = updated.exit_code
        return updated

    def read_logs(self, background_id: str, *, tail: int | None = None) -> str:
        record = self.load_record(background_id)
        path = Path(record.log_path)
        if not path.exists():
            return ''
        text = path.read_text(encoding='utf-8', errors='replace')
        if tail is None or tail <= 0:
            return text
        lines = text.splitlines()
        return '\n'.join(lines[-tail:])

    def restart(
        self,
        background_id: str,
        *,
        prompt: str | None = None,
        env: dict[str, str] | None = None,
        process_cwd: Path | None = None,
        limits: 'RestartLimits | None' = None,
    ) -> 'BackgroundSessionRecord':
        """Start a stopped background agent again with the same id and settings.

        The worker continues the agent's saved session (its session id is the
        background id) or, when none was saved, runs the prompt from the start.
        The command is rebuilt as Sigil's own worker with the record's worker options;
        `limits` (the model's restart tool) caps them to the restarting session.
        """
        record = self.load_record(background_id)
        if record.background_id != background_id:
            raise RuntimeError(f'{background_id}: its record file names another agent ({record.background_id}); '
                               'not restarted')
        if record.status == 'running':
            raise RuntimeError(f'{background_id} is still running')
        limit_message = background_limit_message(self)
        if limit_message:
            raise RuntimeError(limit_message)
        # why (S05): the record file lives in the workspace, so a cloned repository can carry one. The
        # program is always Sigil's own worker, and only options the worker accepts are kept.
        options = _record_worker_options(record)
        if limits is not None and not _env_on('SIGIL_BACKGROUND_RESTART_UNLIMITED'):
            options = limits.apply(options, background_id)
            env = limits.worker_env(options, env)
        forwarded: list[str] = []
        for name, value in options:
            if name not in ('--background-root', '--resume-session'):
                forwarded.extend([name] if value is None else [name, value])
        forwarded.extend(['--background-root', str(self.root), '--resume-session'])
        new_prompt = prompt or record.prompt
        command = build_background_worker_command(background_id=background_id, prompt=new_prompt,
                                                  forwarded_args=forwarded)
        workspace = next((value for name, value in options if name == '--cwd' and value), record.workspace_cwd)
        restarted = self.launch(
            command,
            prompt=new_prompt,
            workspace_cwd=Path(workspace),
            model=record.model,
            mode=record.mode,
            background_id=background_id,
            process_cwd=process_cwd or Path(__file__).resolve().parent.parent,
            env=env,
        )
        self.set_note(background_id, 'restarted')
        return self.load_record(restarted.background_id)

    def set_note(self, background_id: str, note: str | None) -> None:
        """Record a live note (shown by agent-ps and background_agent_status)."""
        from dataclasses import replace as _replace

        try:
            record = self.load_record(background_id)
        except (OSError, ValueError, KeyError) as _ignored:
            _log_ignored(_ignored)
            return
        self.save_record(_replace(record, note=note))

    def render_ps(self) -> str:
        records = tuple(self.check_stalled(record) for record in self.list_records())
        lines = ['# Background Sessions', '']
        if not records:
            lines.append('No local background sessions are currently recorded.')
            return '\n'.join(lines)
        for record in records:
            parts = [
                record.background_id,
                f'status={record.status}',
                f'pid={record.pid}',
                f'model={record.model}',
                f'cwd={record.workspace_cwd}',
            ]
            if record.exit_code is not None:
                parts.append(f'exit_code={record.exit_code}')
            if record.note:
                parts.append(record.note)
            lines.append('- ' + '; '.join(parts))
            lines.append(f'  prompt: {_snapshot_text(record.prompt)}')
        return '\n'.join(lines)

    def render_logs(self, background_id: str, *, tail: int | None = None) -> str:
        record = self.load_record(background_id)
        log_text = self.read_logs(background_id, tail=tail)
        lines = [
            '# Background Logs',
            '',
            f'- Background session: {record.background_id}',
            f'- Status: {record.status}' + (f' ({record.note})' if record.note else ''),
            f'- PID: {record.pid}',
            f'- Log path: {record.log_path}',
            '',
            log_text.rstrip() or '(empty log)',
        ]
        return '\n'.join(lines)

    def render_attach(self, background_id: str, *, tail: int | None = None) -> str:
        record = self.load_record(background_id)
        lines = [
            '# Background Attach',
            '',
            f'- Background session: {record.background_id}',
            f'- Status: {record.status}',
            f'- Workspace cwd: {record.workspace_cwd}',
        ]
        if record.session_id:
            lines.append(f'- Agent session id: {record.session_id}')
        if record.session_path:
            lines.append(f'- Agent session path: {record.session_path}')
        lines.extend(['', self.read_logs(background_id, tail=tail).rstrip() or '(empty log)'])
        return '\n'.join(lines)


_LAUNCH_LOCK = threading.Lock()


def _env_on(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


_WORKER_OPTIONS: dict[str, bool] = {}


def _worker_option_arity() -> dict[str, bool]:
    """Every option the agent-bg-worker command accepts -> whether it takes a value (from the parser
    itself, so the list cannot drift from main.py)."""
    if not _WORKER_OPTIONS:
        import argparse

        from .main import build_parser

        for action in build_parser()._actions:
            if isinstance(action, argparse._SubParsersAction):
                worker = action.choices['agent-bg-worker']
                for option in worker._actions:
                    for name in option.option_strings:
                        if name.startswith('--'):
                            _WORKER_OPTIONS[name] = option.nargs != 0
    return _WORKER_OPTIONS


def _record_worker_options(record: 'BackgroundSessionRecord') -> list[tuple[str, str | None]]:
    """The worker options saved in a record's command, as (name, value or None) pairs. Anything the
    worker would not accept (another program, a stray argument, an unknown option) is refused."""
    command = list(record.command)
    try:
        start = command.index('agent-bg-worker')
    except ValueError:
        raise RuntimeError(f'{record.background_id} has no worker command to restart') from None
    arity = _worker_option_arity()
    tokens = command[start + 3:]
    options: list[tuple[str, str | None]] = []
    index = 0
    while index < len(tokens):
        token = str(tokens[index])
        name, has_inline, inline = token.partition('=')
        if name not in arity:
            raise RuntimeError(f'{record.background_id}: its record holds {token!r}, which is not a worker '
                               'option; not restarted')
        if not arity[name]:
            if has_inline:
                raise RuntimeError(f'{record.background_id}: {name} takes no value; not restarted')
            options.append((name, None))
            index += 1
            continue
        if has_inline:
            options.append((name, inline))
            index += 1
            continue
        if index + 1 >= len(tokens):
            raise RuntimeError(f'{record.background_id}: {name} has no value; not restarted')
        options.append((name, str(tokens[index + 1])))
        index += 2
    return options


# Options that load code or configuration, or widen what the worker may touch: the model's restart
# tool drops them (a record file can be planted; the user's own `sigil agent-restart` keeps them).
_LOADING_OPTIONS = frozenset({
    '--mcp-config', '--plugin-dir', '--settings', '--setting-sources', '--profile', '--debug-file',
    '--scratchpad-root', '--system-prompt', '--append-system-prompt', '--override-system-prompt',
    '--response-schema-file',
})


@dataclass(frozen=True)
class RestartLimits:
    """What the session that asks for a restart may hand on (same rule as background_agent_start:
    a worker never gets more than its parent has)."""

    allow_write: bool
    allow_shell: bool
    allow_destructive: bool
    permission_mode: str
    allowed_rules: tuple[str, ...]
    roots: tuple[Path, ...]
    base_url: str

    def _inside(self, value: str) -> bool:
        try:
            path = Path(value).expanduser().resolve()
        except (OSError, RuntimeError) as _ignored:
            _log_ignored(_ignored)
            return False
        return any(path == root or root in path.parents for root in self.roots)

    def apply(self, options: list[tuple[str, str | None]], background_id: str) -> list[tuple[str, str | None]]:
        kept: list[tuple[str, str | None]] = []
        for name, value in options:
            if name in _LOADING_OPTIONS:
                continue
            if name == '--allow-write' and not self.allow_write:
                continue
            if name == '--allow-shell' and not self.allow_shell:
                continue
            if name == '--unsafe' and not self.allow_destructive:
                continue
            if name == '--allowed-tools' and value not in self.allowed_rules:
                continue
            if name == '--add-dir' and not self._inside(str(value)):
                continue
            if name == '--permission-mode' and not self._mode_allowed(str(value)):
                continue
            if name == '--permission-prompts' and value != 'deny':  # 'allow' approves ask rules unasked
                continue
            if name == '--cwd' and not self._inside(str(value)):
                raise RuntimeError(f'{background_id} works in {value}, outside this session\'s folders; restart '
                                   'it yourself with `sigil agent-restart` if that is intended')
            kept.append((name, value))
        if not any(name == '--cwd' for name, _ in kept):
            kept.insert(0, ('--cwd', str(self.roots[0])))
        return kept

    def _mode_allowed(self, mode: str) -> bool:
        if mode in ('default', 'plan', 'dontAsk') or mode == self.permission_mode:
            return True
        return mode == 'acceptEdits' and self.allow_write

    def worker_env(self, options: list[tuple[str, str | None]], env: dict[str, str] | None) -> dict[str, str]:
        """The main API key goes only to the session's own server (same rule as fallback models)."""
        from .profiles import _host, host_key_env

        env = dict(os.environ if env is None else env)
        target = next((value for name, value in options if name == '--base-url' and value), None)
        if target and _host(str(target)) != _host(self.base_url):
            for name in ('OPENAI_API_KEY', 'SIGIL_API_KEY', host_key_env(self.base_url)):
                env.pop(name, None)
        return env


def background_limit_message(runtime: 'BackgroundSessionRuntime') -> str | None:
    """SIGIL_MAX_BACKGROUND_AGENTS (default 4, 0 = no limit) running at once."""
    import os

    try:
        limit = int(os.environ.get('SIGIL_MAX_BACKGROUND_AGENTS', '') or 4)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        limit = 4
    if limit > 0 and runtime.running_count() >= limit:
        return (f'{limit} background agents are already running. Wait for one, stop one, '
                'or raise SIGIL_MAX_BACKGROUND_AGENTS.')
    return None


def build_background_worker_command(
    *,
    background_id: str,
    prompt: str,
    forwarded_args: list[str],
) -> list[str]:
    return [
        sys.executable,
        '-m',
        'src.main',
        'agent-bg-worker',
        background_id,
        prompt,
        *forwarded_args,
    ]


def _process_start_time(pid: int) -> float | None:
    """When the OS created process `pid` (epoch seconds), or None if it cannot be read."""
    if pid <= 0:
        return None
    try:
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                times = [wintypes.FILETIME() for _ in range(4)]
                if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                    return None
                ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                return (ticks - 116444736000000000) / 10_000_000
            finally:
                kernel32.CloseHandle(handle)
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            start_ticks = int(fields[19])  # field 22: starttime, clock ticks after boot
            boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                        if line.startswith('btime '))
            return boot + start_ticks / os.sysconf('SC_CLK_TCK')
        out = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True,
                             timeout=5, env={**os.environ, 'LC_ALL': 'C'}).stdout.strip()
        return time.mktime(time.strptime(out, '%a %b %d %H:%M:%S %Y')) if out else None
    except (OSError, ValueError, IndexError, StopIteration, subprocess.SubprocessError) as exc:
        from .debug_log import get_logger

        get_logger('background').debug('start time of pid %s: %s', pid, exc)
        return None


def _same_process(record: 'BackgroundSessionRecord') -> bool:
    """False when the record's PID is alive but belongs to a process created at another time
    (the worker ended and the number was reused). An unreadable start time counts as the same
    (logged as a warning)."""
    if record.pid in _DETACHED_PROCESSES:
        return True
    started = _process_start_time(record.pid)
    if started is None:
        # Cannot tell (no /proc, no ps, access denied): treat it as the worker, as before, so a
        # worker is never left running because its start time could not be read. Logged, not silent.
        from .debug_log import get_logger

        get_logger('background').warning('cannot read the start time of pid %s (%s); treating it as '
                                         'background agent %s', record.pid, sys.platform, record.background_id)
        return True
    expected, tolerance = record.pid_started, 2.0
    if expected is None:  # record from before pid_started existed: started_at is written just after launch
        try:
            expected, tolerance = datetime.fromisoformat(record.started_at).timestamp(), 10.0
        except (TypeError, ValueError) as _ignored:
            _log_ignored(_ignored)
            return True
    return abs(started - expected) <= tolerance


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    process = _DETACHED_PROCESSES.get(pid)
    if process is not None:
        return process.poll() is None
    if sys.platform == 'win32':
        # NEVER use os.kill(pid, 0) on Windows: signal 0 == CTRL_C_EVENT, which sends
        # Ctrl+C to the shared console and interrupts the calling agent.
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError as _ignored:
        _log_ignored(_ignored)
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_text(text: str, limit: int = 140) -> str:
    normalized = ' '.join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + '...'
