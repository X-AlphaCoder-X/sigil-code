"""Signals to child process groups on POSIX, guarded so they can never reach our own group.

Sigil starts shells, MCP servers, status-line commands and background agents with
start_new_session=True, so each child leads its own process group (pgid == pid) and the whole tree
can be stopped with os.killpg. A pid that is ours, 0/1, or no longer a group leader (an exited child
whose pid was reused, a stale or planted record) would resolve to another group, possibly our own:
os.killpg(os.getpgid(pid)) on a child without its own session signalled the caller's whole group
(on CI it took down the test runner). Every group signal goes through safe_killpg, which refuses
and logs instead.
"""
from __future__ import annotations

import os
import sys

from .debug_log import get_logger, log_ignored_error as _log_ignored

_log = get_logger('process')


def _posix() -> bool:
    return sys.platform != 'win32' and all(hasattr(os, name) for name in ('killpg', 'getpgid', 'getpgrp'))


def _refuse(reason: str, pid: object) -> bool:
    _log.warning('not signalling process group of pid %s: %s', pid, reason)
    return False


def safe_killpg(pid: int, sig: int) -> bool:
    """Signal the process group that `pid` leads (a child started in its own session). Returns
    whether a signal was sent. Refused (logged): pid <= 1, our own pid or group, or a pid whose
    group is not its own (pgid != pid)."""
    if not _posix():
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError) as exc:
        return _refuse(f'not a pid ({exc})', pid)
    if pid <= 1 or pid == os.getpid():
        return _refuse('our own process or a system pid', pid)
    own_group = os.getpgrp()
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError as gone:
        _log_ignored(gone)
        pgid = pid  # the leader exited; its group keeps the id while children (orphans) still run
    except OSError as exc:
        return _refuse(f'getpgid failed: {exc}', pid)
    if pgid != pid:
        return _refuse(f'it is not a group leader (pgid {pgid})', pid)
    if pgid == own_group or pgid <= 1:
        return _refuse('that group is our own', pid)
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError) as exc:  # already gone (normal)
        _log_ignored(exc)
        return False


def safe_kill_pid(pid: int, sig: int) -> bool:
    """os.kill for a pid read from a record, refusing our own pid, our parent and pid <= 1."""
    if not _posix():
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError) as exc:
        return _refuse(f'not a pid ({exc})', pid)
    if pid <= 1 or pid in (os.getpid(), os.getppid()):
        return _refuse('our own process, our parent or a system pid', pid)
    try:
        os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError) as exc:
        _log_ignored(exc)
        return False
