"""Background shells: bash commands that keep running while the agent works.

``bash`` with ``run_in_background: true`` starts the command and returns an id
at once. Output goes to a file in the scratchpad (the model can also read_file
it); ``bash_output`` returns the lines added since the last read and
``bash_kill`` stops the job and everything it started. ``/bashes`` (and
``/tasks``) list jobs. Jobs are stopped when the session ends.

PC-safety limits (each can be changed; 0 turns it off):
- SIGIL_MAX_BACKGROUND_SHELLS: jobs running at once (default 8).
- SIGIL_BACKGROUND_SHELL_TIMEOUT: seconds before a job is stopped (default 0 = no limit).
- SIGIL_BACKGROUND_SHELL_MAX_BYTES: output file size at which a job is stopped (default 1 GiB).
A watchdog thread checks the last two every SIGIL_BACKGROUND_SHELL_CHECK_SECONDS (default 1),
also when nobody reads the output. Each job keeps the limits that were set when it started.
Reads take at most the last max_chars of new output.
"""
from __future__ import annotations

import atexit
import itertools
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_COUNTER = itertools.count(1)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, '') or default)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return default


@dataclass
class ShellJob:
    job_id: str
    command: str
    cwd: str
    output_path: Path
    process: subprocess.Popen[Any]
    started_at: float = field(default_factory=time.time)
    read_offset: int = 0
    killed: bool = False
    ended_at: float | None = None
    win_job: Any = None  # Windows Job Object holding the shell and all it starts
    stop_reason: str | None = None  # why the watchdog stopped it (timeout / size cap)
    # Limits read once when the job starts (SIGIL_BACKGROUND_SHELL_TIMEOUT / _MAX_BYTES /
    # _CHECK_SECONDS): a later change of the environment does not change a running job's limits.
    timeout_seconds: int = 0
    max_bytes: int = 0
    check_seconds: float = 1.0
    # Why a lock: the watchdog, bash_kill, kill_all and atexit can stop one job at the same time.
    # Each must take the Job Object handle and clear it in one step; a handle closed twice can be
    # a handle Windows has already given to another process or pipe (closing that breaks it).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def take_win_job(self) -> Any:
        """The Job Object handle, now owned by the caller (None if another caller took it)."""
        with self._lock:
            job, self.win_job = self.win_job, None
            return job

    @property
    def exit_code(self) -> int | None:
        code = self.process.poll()
        if code is not None and self.ended_at is None:
            self.ended_at = time.time()
        return code

    @property
    def runtime_seconds(self) -> int:
        end = self.ended_at if self.exit_code is not None and self.ended_at else time.time()
        return int(end - self.started_at)

    @property
    def status(self) -> str:
        code = self.exit_code
        if code is None:
            return 'running'
        if self.killed:
            return 'killed'
        return 'completed' if code == 0 else 'failed'


_CREATE_SUSPENDED = 0x00000004
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


def _win_contain(process: subprocess.Popen[Any], limits: dict[str, Any] | None = None) -> Any:
    """Put a (suspended) Windows process in a Job Object, then resume it.

    Git Bash forks: a command that isn't the last in a script runs in a new
    Windows process whose parent exits at once, so taskkill /T cannot find it.
    Every process started inside a job stays in the job, so TerminateJobObject
    stops them all. KILL_ON_JOB_CLOSE also stops them if Sigil itself exits.
    Returns the job handle, or None (the process is resumed either way).
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    ntdll = ctypes.WinDLL('ntdll')
    handle = int(process._handle)  # type: ignore[attr-defined]
    job = None
    try:
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                'ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class BASIC_LIMITS(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class EXTENDED_LIMITS(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', BASIC_LIMITS),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        if job:
            info = EXTENDED_LIMITS()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if limits:  # opt-in Windows sandbox (sandbox.py): memory, process count, UI
                if limits.get('memory_bytes'):
                    info.BasicLimitInformation.LimitFlags |= 0x200  # JOB_OBJECT_LIMIT_JOB_MEMORY
                    info.JobMemoryLimit = int(limits['memory_bytes'])
                if limits.get('max_processes'):
                    info.BasicLimitInformation.LimitFlags |= 0x8  # JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                    info.BasicLimitInformation.ActiveProcessLimit = int(limits['max_processes'])
            kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
            if limits and limits.get('ui_restrictions'):
                ui = wintypes.DWORD(int(limits['ui_restrictions']))
                kernel32.SetInformationJobObject(job, 4, ctypes.byref(ui), ctypes.sizeof(ui))
            if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(handle)):
                kernel32.CloseHandle(wintypes.HANDLE(job))
                job = None
    except Exception as _ignored:
        _log_ignored(_ignored)
        job = None
    finally:
        ntdll.NtResumeProcess(wintypes.HANDLE(handle))
    if job is None:
        from .debug_log import get_logger

        level = 30 if limits else 10  # WARNING when limits were asked for, else DEBUG
        get_logger('sandbox').log(level, 'Windows Job Object not created for pid %s: the command runs '
                                  'without its limits and its children may outlive a kill (error %s)',
                                  process.pid, ctypes.get_last_error())
    return job


def _win_terminate_job(job: Any) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.TerminateJobObject(wintypes.HANDLE(job), 1)
    kernel32.CloseHandle(wintypes.HANDLE(job))


def _win_release_job(job: Any) -> None:
    """Let a Job Object go without ending its processes (the command finished normally; what it
    started on purpose keeps running): clear KILL_ON_JOB_CLOSE, then close the handle."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    class _Basic(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                    ('SchedulingClass', wintypes.DWORD)]

    class _Extended(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', _Basic), ('IoInfo', ctypes.c_ulonglong * 6),
                    ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                    ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

    info = _Extended()  # all limits off, KILL_ON_JOB_CLOSE included
    kernel32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    kernel32.CloseHandle(wintypes.HANDLE(job))


def run_command_tree(args: Any, *, timeout: float | None, cwd: Any = None, env: Any = None, shell: bool = False,
                     executable: str | None = None, input_text: str | None = None,
                     creationflags: int = 0) -> subprocess.CompletedProcess[str]:
    """subprocess.run(capture_output=True, text=True) for a command that may start children:
    when it times out, the whole process tree ends, not only the shell (/test, /lint, autoTest,
    command hooks). Windows: a Job Object; POSIX: its own process group. On a normal exit the
    children it started on purpose keep running, as before. Raises subprocess.TimeoutExpired."""
    kwargs: dict[str, Any] = {
        'cwd': cwd, 'env': env, 'shell': shell, 'executable': executable, 'text': True, 'encoding': 'utf-8',
        'errors': 'replace', 'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE,
        'stdin': subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
    }
    if sys.platform == 'win32':
        kwargs['creationflags'] = creationflags | _CREATE_SUSPENDED
    else:
        kwargs['start_new_session'] = True
    process = subprocess.Popen(args, **kwargs)
    job = _win_contain(process) if sys.platform == 'win32' else None
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(process, job)
        job = None
        try:
            process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
            from .debug_log import log_ignored_error

            log_ignored_error(exc)  # pipes held open by a child that was already killed
        raise
    finally:
        if job is not None:
            try:
                _win_release_job(job)
            except OSError as exc:
                from .debug_log import log_ignored_error

                log_ignored_error(exc)
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def kill_process_tree(process: subprocess.Popen[Any], win_job: Any = None) -> None:
    """Stop a shell and every process it started. The caller owns `win_job` (it is closed here)."""
    if win_job is not None:
        try:
            _win_terminate_job(win_job)
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
        try:
            # TerminateJobObject ends the processes asynchronously: wait for the shell briefly
            # before falling back to taskkill (this used to take about a second every stop)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as _ignored:
            _log_ignored(_ignored)
    if process.poll() is not None:
        return
    if sys.platform == 'win32':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        from .process_group import safe_killpg

        # only the shell's own group (it was started in its own session); never ours (CI runner killed)
        if safe_killpg(process.pid, signal.SIGTERM):
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as _ignored:
                _log_ignored(_ignored)
                safe_killpg(process.pid, signal.SIGKILL)
    try:
        process.kill()
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as _ignored:
        _log_ignored(_ignored)
        pass


_DEFAULT_MAX_BYTES = 1024 ** 3


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, '') or default)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return default


class BackgroundShellManager:
    def __init__(self) -> None:
        self.jobs: dict[str, ShellJob] = {}
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()  # limit check + registration in one step
        self._watchdog: threading.Thread | None = None

    def _ensure_watchdog(self) -> None:
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(target=self._watch, name='sigil-shell-watchdog', daemon=True)
            self._watchdog.start()

    def _watch(self) -> None:
        """Stop jobs past their time or size limit; ends when no job is running.

        Why every step is guarded: an exception here used to end the thread silently, and from
        then on no job of the session was stopped at its limit (the watchdog still looked alive
        to _ensure_watchdog until the next start)."""
        interval = max(0.05, _env_float('SIGIL_BACKGROUND_SHELL_CHECK_SECONDS', 1.0))
        while True:
            time.sleep(interval)
            try:
                with self._lock:  # same lock as _ensure_watchdog: a job started now finds no watchdog
                    running = self.running()
                    if not running:
                        self._watchdog = None
                        return
                for job in running:
                    try:
                        self._enforce_limits(job)
                    except Exception:  # keep watching the other jobs
                        from .debug_log import get_logger

                        get_logger('shells').warning('watchdog check failed for %s', job.job_id, exc_info=True)
                interval = max(0.05, min(job.check_seconds for job in running))
            except Exception:
                from .debug_log import get_logger

                get_logger('shells').warning('background shell watchdog step failed', exc_info=True)

    def running(self) -> list[ShellJob]:
        found = []
        for job in list(self.jobs.values()):  # a copy: others insert
            try:
                if job.exit_code is None:
                    found.append(job)
            except OSError as exc:  # its process handle is unusable: it cannot be watched or counted
                from .debug_log import get_logger

                get_logger('shells').warning('background shell %s: process handle unusable (%s)', job.job_id, exc)
        return found

    def start(self, command: str, *, cwd: Path, env: dict[str, str] | None, bash: str,
              output_dir: Path | None, argv: list[str] | None = None,
              limits: dict[str, Any] | None = None) -> ShellJob:
        with self._start_lock:
            return self._start_locked(command, cwd=cwd, env=env, bash=bash, output_dir=output_dir, argv=argv,
                                      limits=limits)

    def _start_locked(self, command: str, *, cwd: Path, env: dict[str, str] | None, bash: str,
                      output_dir: Path | None, argv: list[str] | None = None,
                      limits: dict[str, Any] | None = None) -> ShellJob:
        limit = _env_int('SIGIL_MAX_BACKGROUND_SHELLS', 8)
        if limit > 0 and len(self.running()) >= limit:
            raise RuntimeError(
                f'{limit} background shells are already running. Stop one with bash_kill, '
                'or raise SIGIL_MAX_BACKGROUND_SHELLS.'
            )
        folder = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / 'sigil-background-shells'
        folder.mkdir(parents=True, exist_ok=True)
        job_id = f'bash_{next(_COUNTER)}'
        output_path = folder / f'{job_id}_{os.getpid()}.log'
        handle = output_path.open('wb')
        win_job = None
        try:
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    argv or [bash, '-c', command], cwd=str(cwd), env=env,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                                   | _CREATE_SUSPENDED),
                )
                win_job = _win_contain(process, limits)
            else:
                process = subprocess.Popen(
                    argv or [bash, '-c', command], cwd=str(cwd), env=env,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        finally:
            handle.close()
        job = ShellJob(job_id=job_id, command=command, cwd=str(cwd), output_path=output_path,
                       process=process, win_job=win_job,
                       timeout_seconds=_env_int('SIGIL_BACKGROUND_SHELL_TIMEOUT', 0),
                       max_bytes=_env_int('SIGIL_BACKGROUND_SHELL_MAX_BYTES', _DEFAULT_MAX_BYTES),
                       check_seconds=max(0.05, _env_float('SIGIL_BACKGROUND_SHELL_CHECK_SECONDS', 1.0)))
        with self._lock:
            self.jobs[job_id] = job
        self._ensure_watchdog()
        return job

    def get(self, job_id: str) -> ShellJob:
        job = self.jobs.get(job_id.strip())
        if job is None:
            known = ', '.join(self.jobs) or 'none'
            raise KeyError(f'No background shell {job_id!r} (known: {known})')
        return job

    def _enforce_timeout(self, job: ShellJob) -> None:
        limit = job.timeout_seconds
        if limit > 0 and job.exit_code is None and time.time() - job.started_at > limit:
            job.stop_reason = f'ran longer than {limit} s (SIGIL_BACKGROUND_SHELL_TIMEOUT)'
            self._stop(job)

    def _enforce_limits(self, job: ShellJob) -> None:
        self._enforce_timeout(job)
        cap = job.max_bytes
        if cap > 0 and job.exit_code is None:
            try:
                size = job.output_path.stat().st_size
            except OSError as _ignored:
                _log_ignored(_ignored)
                return
            if size > cap:
                job.stop_reason = (f'output reached {size:,} bytes, over {cap:,} '
                                   '(SIGIL_BACKGROUND_SHELL_MAX_BYTES)')
                self._stop(job)

    def _stop(self, job: ShellJob) -> None:
        if job.exit_code is None:
            job.killed = True
        # take-and-clear first: a second caller (watchdog, bash_kill, kill_all, atexit) finds no
        # handle, so the Job Object is terminated and closed exactly once
        kill_process_tree(job.process, job.take_win_job())

    def read_new(self, job_id: str, max_chars: int = 30000) -> tuple[ShellJob, str]:
        job = self.get(job_id)
        self._enforce_limits(job)
        skipped = 0
        try:
            with job.output_path.open('rb') as handle:
                end = handle.seek(0, os.SEEK_END)
                # read at most the last max_chars*4 bytes (UTF-8 is up to 4 bytes a character)
                start = max(job.read_offset, end - max(1, max_chars) * 4)
                skipped = start - job.read_offset
                handle.seek(start)
                data = handle.read(end - start)
        except OSError as _ignored:
            _log_ignored(_ignored)
            data = b''
        job.read_offset += skipped + len(data)
        text = data.decode('utf-8', errors='replace')
        if len(text) > max_chars:
            skipped += len(text[:-max_chars].encode('utf-8', errors='replace'))
            text = text[-max_chars:]
        if skipped:
            text = f'[... {skipped:,} earlier bytes skipped; full log: {job.output_path}]\n' + text
        return job, text

    def kill(self, job_id: str) -> ShellJob:
        job = self.get(job_id)
        if job.exit_code is None or job.win_job is not None:
            self._stop(job)
        return job

    def kill_all(self) -> int:
        stopped = 0
        for job in list(self.jobs.values()):
            try:
                if job.exit_code is None or job.win_job is not None:
                    stopped += job.exit_code is None
                    self._stop(job)
            except OSError as exc:  # one unusable job must not leave the others running (atexit)
                from .debug_log import get_logger

                get_logger('shells').warning('could not stop background shell %s: %s', job.job_id, exc)
        return stopped

    def render(self) -> str:
        if not self.jobs:
            return 'No background shells.'
        lines = ['# Background shells', '']
        for job in self.jobs.values():
            code = job.exit_code
            runtime = job.runtime_seconds
            exit_text = '' if code is None else f', exit {code}'
            lines.append(f'- {job.job_id} [{job.status}{exit_text}, {runtime}s] {job.command[:100]}')
            if job.stop_reason:
                lines.append(f'  stopped: {job.stop_reason}')
            lines.append(f'  output: {job.output_path}')
        return '\n'.join(lines)


_MANAGER = BackgroundShellManager()


def shell_manager() -> BackgroundShellManager:
    return _MANAGER


atexit.register(_MANAGER.kill_all)
