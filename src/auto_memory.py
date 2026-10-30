"""Auto memory (opt-in): short notes the agent keeps per project.

Turn it on with SIGIL_AUTO_MEMORY=1 or "autoMemoryEnabled": true in settings
(SIGIL_AUTO_MEMORY=0 turns it off again). Notes live in
~/.sigil/projects/<repo>/memory/MEMORY.md (or the "autoMemoryDirectory"
setting); <repo> is the git root of the workspace, so every folder of one repo
shares the same notes. The first SIGIL_AUTO_MEMORY_LINES (200) lines are loaded
with the other memory files, and the memory_save tool adds a note. The file is
plain Markdown: read, edit or delete it any time (/memory shows the path).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

INDEX_NAME = 'MEMORY.md'
DEFAULT_LINES = 200
MAX_NOTE_CHARS = 500


def _settings(cwd: Path) -> dict:
    try:
        from .settings import load_settings

        return load_settings(cwd)
    except Exception as _ignored:
        _log_ignored(_ignored)
        return {}


def enabled(cwd: Path) -> bool:
    raw = os.environ.get('SIGIL_AUTO_MEMORY', '').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off') or os.environ.get('SIGIL_SAFE_MODE') == '1':
        return False
    return _settings(cwd).get('autoMemoryEnabled') is True


def _repo_root(cwd: Path) -> Path:
    current = Path(cwd).resolve()
    for folder in (current, *current.parents):
        if (folder / '.git').exists():
            return folder
    return current


def memory_dir(cwd: Path) -> Path:
    custom = _settings(cwd).get('autoMemoryDirectory')
    if isinstance(custom, str) and custom.strip():
        return Path(custom).expanduser().resolve()
    from .settings import _home

    slug = re.sub(r'[^A-Za-z0-9]+', '-', str(_repo_root(cwd))).strip('-') or 'workspace'
    return _home() / '.sigil' / 'projects' / slug / 'memory'


def index_path(cwd: Path) -> Path:
    return memory_dir(cwd) / INDEX_NAME


def _line_limit() -> int:
    try:
        return max(1, int(os.environ.get('SIGIL_AUTO_MEMORY_LINES', DEFAULT_LINES)))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return DEFAULT_LINES


def load_block(cwd: Path) -> str | None:
    """The memory block for the prompt, or None when auto memory is off."""
    if not enabled(cwd):
        return None
    path = index_path(cwd)
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError as _ignored:
        _log_ignored(_ignored)
        lines = []
    limit = _line_limit()
    body = '\n'.join(lines[:limit]).strip()
    if len(lines) > limit:
        body += f'\n... ({len(lines) - limit} more line(s) in {path})'
    intro = (f'## Auto memory ({path})\n'
             'Your notes from earlier sessions in this project. When the user corrects you or tells you '
             'something that will matter in later sessions (a preference, a command, a project fact), save '
             'a one-line note with the memory_save tool.')
    return f'{intro}\n\n{body}' if body else f'{intro}\n\n(no notes yet)'


def save_note(cwd: Path, note: str) -> tuple[Path, bool]:
    """Append a one-line note. Returns (file, added); an existing identical note is not added twice."""
    text = ' '.join(str(note).split())[:MAX_NOTE_CHARS]
    if not text:
        raise ValueError('note is empty')
    path = index_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding='utf-8') if path.exists() else '# Memory\n\n'
    line = f'- {text}'
    if line in existing.splitlines():
        return path, False
    prefix = '' if existing.endswith('\n') else '\n'
    path.write_text(existing + prefix + line + '\n', encoding='utf-8')
    return path, True
