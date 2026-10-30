"""Microcompaction: offload bulky old tool results to disk.

Microcompact layer. A small "hot tail" of the most
recent tool results stays fully inline; older tool results above a size
threshold are written to ``<scratchpad>/tool-results/`` and replaced in the
conversation by a short reference (path + preview). Nothing is lost: the
model can ``read_file`` the stored path when it needs the full output again.

Runs before every model call, needs no model call itself, and is idempotent
(already-offloaded messages are skipped).

Env knobs:
- ``SIGIL_DISABLE_MICROCOMPACT=1``      turn it off
- ``SIGIL_MICROCOMPACT_KEEP_RECENT``    tool results kept inline (default 5)
- ``SIGIL_MICROCOMPACT_MIN_CHARS``      only offload results larger than this (default 8000)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .agent_session import AgentSessionState
from .settings import write_text_atomic  # replaces a file in one step: readers never see half of it
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

MICROCOMPACT_KIND = 'microcompacted_tool_result'
DEFAULT_KEEP_RECENT = 5
DEFAULT_MIN_CHARS = 8000
_PREVIEW_CHARS = 600


@dataclass(frozen=True)
class MicrocompactResult:
    offloaded_count: int = 0
    chars_removed: int = 0
    paths: tuple[str, ...] = ()


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return default
    return value if value >= 0 else default


def microcompact_session(
    session: AgentSessionState,
    storage_dir: Path,
    *,
    keep_recent: int | None = None,
    min_chars: int | None = None,
) -> MicrocompactResult:
    if os.environ.get('SIGIL_DISABLE_MICROCOMPACT', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        return MicrocompactResult()
    keep_recent = _env_int('SIGIL_MICROCOMPACT_KEEP_RECENT', DEFAULT_KEEP_RECENT) if keep_recent is None else keep_recent
    min_chars = _env_int('SIGIL_MICROCOMPACT_MIN_CHARS', DEFAULT_MIN_CHARS) if min_chars is None else min_chars

    tool_indexes = [i for i, m in enumerate(session.messages) if m.role == 'tool']
    cold = tool_indexes[: max(len(tool_indexes) - keep_recent, 0)]
    offloaded = 0
    removed = 0
    paths: list[str] = []
    for index in cold:
        message = session.messages[index]
        if message.metadata.get('kind') in {MICROCOMPACT_KIND, 'snipped_message'}:
            continue
        if message.state == 'tombstoned' or len(message.content) <= min_chars:
            continue
        storage_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r'[^A-Za-z0-9_.-]', '_', message.tool_call_id or message.message_id or str(index))
        path = (storage_dir / f'{safe_id}.txt').resolve()
        try:
            write_text_atomic(path, message.content, encoding='utf-8')
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
        preview = message.content[:_PREVIEW_CHARS].rstrip()
        replacement = (
            f'[Older tool result ({message.name or "tool"}, {len(message.content):,} chars) '
            f'was moved out of context to save space. Full output is stored at: {path}\n'
            f'Use read_file on that path if you need it again.]\n'
            f'Preview:\n{preview}\n...'
        )
        session.update_message(
            index,
            content=replacement,
            metadata={
                'kind': MICROCOMPACT_KIND,
                'stored_path': str(path),
                'original_chars': len(message.content),
            },
            mutation_kind='microcompact',
        )
        offloaded += 1
        removed += len(message.content) - len(replacement)
        paths.append(str(path))
    return MicrocompactResult(offloaded_count=offloaded, chars_removed=removed, paths=tuple(paths))
