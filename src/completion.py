"""Tab completion for the chat prompt: /commands and @paths.

``SigilCompleter`` is a prompt_toolkit completer. It completes:
    /name      built-in slash commands, chat commands and custom commands/skills
    @path      files and folders under the workspace (and --add-dir folders)
The matching logic lives in plain functions so it works (and is tested)
without a terminal.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

try:
    from prompt_toolkit.completion import Completer, Completion
except ImportError:  # prompt_toolkit is a core dependency, but stay importable without it
    Completer = object  # type: ignore[assignment,misc]
    Completion = None  # type: ignore[assignment]

CHAT_COMMANDS = ('exit', 'quit', 'mode', 'rewind', 'compact', 'tokens')
MAX_PATH_MATCHES = 200


def slash_command_names(agent: Any = None) -> list[str]:
    from .agent_slash_commands import get_slash_command_specs

    names: set[str] = set(CHAT_COMMANDS)
    for spec in get_slash_command_specs():
        names.update(spec.names)
    if agent is not None:
        try:
            from .custom_commands import discover_commands

            names.update(discover_commands(agent.runtime_config.cwd, getattr(agent, 'skills_runtime', None)))
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
    return sorted(names)


def complete_slash(text: str, names: Iterable[str]) -> list[str]:
    """'/he' -> ['/help', ...]. Only the first word is completed."""
    if not text.startswith('/') or ' ' in text:
        return []
    prefix = text[1:].lower()
    return ['/' + name for name in names if name.lower().startswith(prefix)]


def complete_path(fragment: str, roots: Iterable[Path]) -> list[str]:
    """Paths for an @mention fragment, relative to the first root that has matches.

    Folders end with '/'. Hidden entries show only when the fragment's last part
    starts with '.'.
    """
    raw = fragment.strip('"')
    cut = max(raw.rfind('/'), raw.rfind('\\')) + 1
    typed_head, tail = raw[:cut], raw[cut:]
    head = os.path.expanduser(typed_head) if typed_head else ''
    if head and os.path.isabs(head):
        candidates = [Path(head)]
    else:
        candidates = [Path(root) / head for root in roots]
    results: list[str] = []
    for folder in candidates:
        try:
            entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
        for entry in entries:
            name = entry.name
            if name.startswith('.') and not tail.startswith('.'):
                continue
            if not name.lower().startswith(tail.lower()):
                continue
            shown = typed_head + name + ('/' if entry.is_dir() else '')
            if shown not in results:
                results.append(shown)
            if len(results) >= MAX_PATH_MATCHES:
                return results
        if results:
            break
    return results


def _mention_fragment(text_before_cursor: str) -> str | None:
    """The text after the last '@' when the cursor is inside an @mention."""
    index = text_before_cursor.rfind('@')
    if index < 0:
        return None
    if index > 0 and not text_before_cursor[index - 1].isspace():
        return None  # e-mail address or similar
    fragment = text_before_cursor[index + 1:]
    if any(ch.isspace() for ch in fragment) and not fragment.startswith('"'):
        return None
    return fragment


class SigilCompleter(Completer):  # type: ignore[misc]
    def __init__(self, roots: Callable[[], list[Path]], names: Callable[[], list[str]]) -> None:
        self._roots = roots
        self._names = names

    def get_completions(self, document: Any, complete_event: Any):  # noqa: ANN001
        text = document.text_before_cursor
        if text.startswith('/') and ' ' not in text:
            for match in complete_slash(text, self._names()):
                yield Completion(match, start_position=-len(text))
            return
        fragment = _mention_fragment(text)
        if fragment is None:
            return
        for match in complete_path(fragment, self._roots()):
            yield Completion(match, start_position=-len(fragment), display=match)


def build_completer(agent: Any) -> Any:
    """A completer for this agent's workspace, or None without prompt_toolkit."""
    if Completion is None:
        return None
    config = agent.runtime_config

    def _roots() -> list[Path]:
        return [Path(config.cwd), *[Path(p) for p in config.additional_working_directories]]

    cache: dict[str, list[str]] = {}

    def _names() -> list[str]:
        if 'names' not in cache:
            cache['names'] = slash_command_names(agent)
        return cache['names']

    return SigilCompleter(_roots, _names)
