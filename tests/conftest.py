"""Test-run setup: the suite never writes to the repo or the user's own folders.

- Logs go to a temporary folder, never ~/.sigil/logs.
- The home folder is a throw-away temp folder, so ~/.sigil and ~/.sigil (settings, hooks,
  SIGIL.md, agents, skills, memory) are never read or written. SIGIL_TEST_REAL_HOME=1 keeps the
  real one.
- Tests run from a temp working folder, so code that falls back to ./.port_sessions writes there,
  not into the repo. The repo is put on PYTHONPATH for `python -m src.main` subprocesses.
- Temp folders (tempfile, TEMP/TMP) live inside that home. When the system temp folder is under
  the real home (the Windows default), the fake home goes to the drive root instead
  (C:\\sigil-test-home-*), so SIGIL.md / .sigil walks up from a workspace never pass the real
  home. Everything is removed when the run ends.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CREATED: list[str] = []


def _temp_dir(prefix: str) -> str:
    path = tempfile.mkdtemp(prefix=prefix)
    _CREATED.append(path)
    return path


def _remove_tree(path: str, seconds: float = 10.0) -> bool:
    """Remove a folder, retrying: on Windows a process killed a moment ago (a background shell,
    a test's child) still holds its working folder or a file in it, and rmtree gives up."""
    deadline = time.monotonic() + seconds
    while True:
        shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _remove_created() -> None:
    # close Sigil's log file first: atexit runs this before logging's own shutdown, and on
    # Windows an open log file keeps its folder from being removed (one folder leaked per run)
    import logging

    logging.shutdown()
    left = [path for path in _CREATED if not _remove_tree(path)]
    if left:  # never silent: say which folder stays (the next run removes it once it is old)
        print('tests/conftest.py: could not remove ' + ', '.join(left) + ' (still in use)', file=sys.stderr)


atexit.register(_remove_created)

_STALE_SECONDS = 6 * 3600  # a whole suite run takes minutes; older test folders are leftovers


def _remove_stale_leftovers() -> None:
    """Test folders of earlier runs that were killed (timeout, Ctrl+C) before their cleanup ran."""
    roots = {Path(tempfile.gettempdir())}
    try:
        roots.add(Path(Path(tempfile.gettempdir()).resolve().anchor))
    except OSError:
        pass
    now = time.time()
    for root in roots:
        try:
            candidates = [p for p in root.iterdir()
                          if p.is_dir() and p.name.startswith(('sigil-test-home-', 'sigil-test-logs-'))]
        except OSError:
            continue
        for path in candidates:
            try:
                if now - path.stat().st_mtime > _STALE_SECONDS:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                continue


_remove_stale_leftovers()


def _test_home() -> str:
    base = Path(tempfile.gettempdir()).resolve()
    try:
        real_home = Path.home().resolve()
    except (OSError, RuntimeError):
        real_home = None
    if real_home is not None and base.is_relative_to(real_home):
        try:
            path = tempfile.mkdtemp(prefix='sigil-test-home-', dir=base.anchor)
        except OSError:
            path = None
        if path:
            _CREATED.append(path)
            return path
    return _temp_dir('sigil-test-home-')

if 'SIGIL_LOG_DIR' not in os.environ:
    os.environ['SIGIL_LOG_DIR'] = _temp_dir('sigil-test-logs-')
if os.environ.get('SIGIL_TEST_REAL_HOME', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
    _home = _test_home()
    for _name in ('USERPROFILE', 'HOME'):
        os.environ[_name] = _home
    # Temp folders go inside the throw-away home: lookups that walk up from a workspace
    # (parent .sigil/skills, SIGIL.md) then stop at this home and never reach the real one.
    _tmp = os.path.join(_home, 'tmp')
    os.makedirs(_tmp, exist_ok=True)
    for _name in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[_name] = _tmp
    tempfile.tempdir = _tmp
os.environ['PYTHONPATH'] = os.pathsep.join(
    part for part in (str(_REPO), os.environ.get('PYTHONPATH', '')) if part)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# Project hooks / MCP / env in test workspaces run as before; the trust tests clear this.
os.environ.setdefault('SIGIL_TRUST_ALL', '1')
# Personal ~/.sigil/skills would change skill lists in tests; tests that need them enable it.
os.environ.setdefault('SIGIL_DISABLE_USER_SKILLS', '1')
# Tests never touch the real clipboard or ring the bell.
os.environ.setdefault('SIGIL_CLIPBOARD', 'off')
os.environ.setdefault('SIGIL_NOTIFY', 'off')
# No /models lookup at start-up: CLI tests feed fixed replies to a fake model server.
os.environ.setdefault('SIGIL_DETECT_MODEL', '0')


@pytest.fixture(autouse=True, scope='session')
def _run_outside_the_repo():
    previous = os.getcwd()
    os.chdir(_temp_dir('sigil-test-cwd-'))
    try:
        yield
    finally:
        os.chdir(previous)
