"""Custom status line: a command from settings whose first output line is shown under
the prompt (the `statusLine` setting).

    "statusLine": {"type": "command", "command": "python ~/.sigil/statusline.py"}

The command gets session JSON on stdin in the usual shape (model.id and
model.display_name, workspace.current_dir and project_dir, cost.total_cost_usd, session_id,
session_name, cwd, permission_mode). It runs in a background thread at most every 2 seconds
with a 2 second timeout, so typing never waits on it (only the very first render waits, up to
2 seconds, so the line shows at once). A command that runs too long is ended together with the
processes it started. `~/` in the command is expanded to the home folder. The command runs in the system shell (cmd.exe on Windows).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_INTERVAL = 2.0
_TIMEOUT = 2.0


def _log_debug(message: str, *args: object) -> None:
    from .debug_log import get_logger

    get_logger('status_line').debug(message, *args)


def expand_home(command: str) -> str:
    """`~/` (or `~\\`) at the start of a word becomes the home folder; a lone `~` is left alone."""
    home = str(Path.home())
    return re.sub(r'(?<![^\s"\'=])~(?=[/\\])', lambda _m: home, command)


def status_payload(agent: Any) -> dict[str, Any]:
    """Session JSON for the status line command, (model.id / model.display_name shape)."""
    cwd = str(agent.runtime_config.cwd)
    model = str(getattr(agent.model_config, 'model', '') or '')
    return {
        'session_id': getattr(agent, 'active_session_id', None),
        'session_name': getattr(agent, '_session_name', None),
        'cwd': cwd,
        'model': {'id': model, 'display_name': model},
        'workspace': {'current_dir': cwd, 'project_dir': cwd},
        'cost': {'total_cost_usd': float(getattr(agent, 'cumulative_cost_usd', 0.0) or 0.0)},
        'permission_mode': getattr(agent, 'permission_mode', 'default'),
    }


class StatusLine:
    def __init__(self, command: str, cwd: Path) -> None:
        self.command = command
        self.cwd = cwd
        self._cached = ''
        self._at = float('-inf')
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._has_run = False

    def render(self, payload: dict[str, Any]) -> str:
        now = time.monotonic()
        with self._lock:
            busy = self._worker is not None and self._worker.is_alive()
            due = not busy and now - self._at >= _INTERVAL
            first = not self._has_run
            worker = None
            if due:
                self._at = now
                worker = threading.Thread(target=self._run, args=(dict(payload),), name='status-line', daemon=True)
                self._worker = worker
                worker.start()
        if worker is not None and first:
            worker.join(timeout=_TIMEOUT + 0.5)
        return self._cached

    def _run(self, payload: dict[str, Any]) -> None:
        kwargs: dict[str, Any] = {}
        if sys.platform == 'win32':
            kwargs['creationflags'] = (getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                                       | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
        else:
            kwargs['start_new_session'] = True
        output = b''
        from .child_env import child_env

        try:
            process = subprocess.Popen(self.command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, cwd=str(self.cwd), env=child_env(), **kwargs)
        except OSError as _ignored:
            _log_ignored(_ignored)
            process = None
        if process is not None:
            try:
                output, _ = process.communicate(json.dumps(payload).encode('utf-8'), timeout=_TIMEOUT)
            except subprocess.TimeoutExpired as _ignored:
                _log_ignored(_ignored)
                from .agent_tools import _kill_process_tree

                if sys.platform != 'win32':
                    import signal

                    from .process_group import safe_killpg

                    safe_killpg(process.pid, signal.SIGKILL)  # its own session's group, never ours
                _kill_process_tree(process)  # the shell and everything it started
                try:
                    process.communicate(timeout=5)
                except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    _log_debug('status line drain after kill: %s', exc)
                output = b''
            except (OSError, ValueError) as _ignored:
                _log_ignored(_ignored)
                output = b''
        text = output.decode('utf-8', errors='replace').strip()
        self._cached = text.splitlines()[0] if text else ''
        self._has_run = True


def status_line_for(cwd: Path) -> StatusLine | None:
    from .settings import load_settings

    try:
        config = load_settings(cwd).get('statusLine')
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None
    command = config.get('command') if isinstance(config, dict) else config if isinstance(config, str) else None
    if not isinstance(command, str) or not command.strip():
        return None
    if not _from_user_settings(command):
        from .workspace_trust import allowed, report_skipped

        # Why: a project's statusLine command runs on every prompt; it needs the folder's trust
        if not allowed(cwd, 'statusLine', 'statusLine', command.strip()):
            report_skipped(cwd)
            return None
    return StatusLine(expand_home(os.path.expanduser(command)), cwd)


def _from_user_settings(command: str) -> bool:
    from .settings import _home, read_settings_file

    for path in (_home() / '.sigil' / 'settings.json', _home() / '.sigil' / 'settings.json'):
        config = read_settings_file(path).get('statusLine')
        user = config.get('command') if isinstance(config, dict) else config
        if isinstance(user, str) and user.strip() == command.strip():
            return True
    return False
