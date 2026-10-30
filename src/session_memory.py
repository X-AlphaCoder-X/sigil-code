"""Session memory (opt-in): a running notes file that survives compaction.

Turn it on with SIGIL_SESSION_MEMORY=1 or "sessionMemory": true in settings.
Each session gets <session folder>/notes/<session id>.md. The session_note tool
adds a line (a decision, a file, an error and its fix) at any time. On every
compaction the notes so far are put back right after the new summary, and then
the new summary is added to the notes, so details from earlier compactions are
kept as they were written instead of being summarized again and again. Only the
newest SIGIL_SESSION_MEMORY_CHARS (12000) characters are put back.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from .settings import write_text_atomic  # replaces a file in one step: readers never see half of it
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

DEFAULT_CHARS = 12000


def enabled(cwd: Path) -> bool:
    raw = os.environ.get('SIGIL_SESSION_MEMORY', '').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    try:
        from .settings import load_settings

        return load_settings(cwd).get('sessionMemory') is True
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def notes_path(agent: Any) -> Path | None:
    session_id = getattr(agent, 'active_session_id', None)
    directory = getattr(agent.runtime_config, 'session_directory', None)
    if not session_id or directory is None:
        return None
    return Path(directory) / 'notes' / f'{session_id}.md'


def append(agent: Any, text: str, heading: str | None = None) -> Path | None:
    path = notes_path(agent)
    body = str(text).strip()
    if path is None or not body:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding='utf-8') if path.exists() else '# Session notes\n'
    block = f'\n## {heading}\n{body}\n' if heading else f'- {" ".join(body.split())}\n'
    if not existing.endswith('\n'):
        existing += '\n'
    write_text_atomic(path, existing + block, encoding='utf-8')
    return path


def load(agent: Any) -> str:
    path = notes_path(agent)
    if path is None or not path.is_file():
        return ''
    text = path.read_text(encoding='utf-8', errors='replace').strip()
    if text == '# Session notes':
        return ''
    try:
        limit = max(500, int(os.environ.get('SIGIL_SESSION_MEMORY_CHARS', DEFAULT_CHARS)))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        limit = DEFAULT_CHARS
    if len(text) > limit:
        text = '... (older notes cut; the full file is ' + str(path) + ')\n' + text[-limit:]
    return text


def after_compaction(agent: Any, summary_text: str) -> str:
    """Notes to show after this compaction; then the new summary joins the notes."""
    notes = load(agent)
    append(agent, summary_text, heading=f'Compaction summary ({time.strftime("%Y-%m-%d %H:%M")})')
    return notes
