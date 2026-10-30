"""Plain output mode and the terminal title.

Plain mode (--plain or SIGIL_PLAIN=1) is for screen readers and log files: no
colours or cursor codes, box lines and symbols become plain ASCII, and the chat
reads input with a simple prompt (no toolbar, no full-screen view).

The terminal title shows the session name (or folder) and what the agent is
doing: "Sigil: <name> - working", "- ready", "- needs approval". It is set
only on a real terminal, and not in plain mode; SIGIL_TERMINAL_TITLE=0 turns it off.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_ANSI = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-Z\\-_]')
_ASCII = str.maketrans({
    '╭': '+', '╮': '+', '╰': '+', '╯': '+', '┌': '+', '┐': '+', '└': '+', '┘': '+', '│': '|', '─': '-',
    '━': '-', '┃': '|', '├': '+', '┤': '+', '✓': 'ok', '✔': 'ok', '✗': 'x', '✘': 'x', '↳': '->', '→': '->',
    '←': '<-', '↶': '<-', '⏹': '[stop]', '⏵': '>', '•': '*', '…': '...', '○': 'o', '●': '*', '⚠': '!',
})
_EMOJI = re.compile('[\U0001F000-\U0001FAFF☀-➿️]')


def plain_enabled() -> bool:
    return os.environ.get('SIGIL_PLAIN', '').strip().lower() in ('1', 'true', 'yes', 'on')


def to_plain(text: str) -> str:
    return _EMOJI.sub('', _ANSI.sub('', text).translate(_ASCII))


class PlainStream:
    """Wraps stdout: everything written comes out as plain text."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, text: str) -> int:
        return self._stream.write(to_plain(text))

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def enable_plain() -> None:
    """Switch the process to plain output (idempotent)."""
    os.environ['NO_COLOR'] = '1'
    if not isinstance(sys.stdout, PlainStream):
        sys.stdout = PlainStream(sys.stdout)  # type: ignore[assignment]


THEMES = ('dark', 'light', 'auto')
# On light backgrounds, white and bright colours are hard to read: use dark ones.
_LIGHT = {'97': '30', '37': '30', '93': '33', '96': '36', '92': '32', '95': '35', '94': '34', '91': '31', '90': '2'}
_SGR = re.compile(r'\x1b\[([0-9;]*)m')


def current_theme() -> str:
    raw = os.environ.get('SIGIL_THEME', '').strip().lower()
    if raw in THEMES:
        return raw
    try:
        from pathlib import Path

        from .settings import load_settings

        value = str(load_settings(Path.cwd()).get('theme') or '').lower()
    except Exception as _ignored:
        _log_ignored(_ignored)
        value = ''
    return value if value in THEMES else 'auto'


def to_light(text: str) -> str:
    return _SGR.sub(lambda m: '\x1b[' + ';'.join(_LIGHT.get(code, code) for code in m.group(1).split(';')) + 'm',
                    text)


class LightStream(PlainStream):
    def write(self, text: str) -> int:
        return self._stream.write(to_light(text))

    def isatty(self) -> bool:
        return bool(getattr(self._stream, 'isatty', lambda: False)())


def apply_theme(theme: str | None = None) -> str:
    """light: rewrite bright and white colours to dark ones; dark / auto: as they are."""
    theme = theme or current_theme()
    if theme == 'light' and not isinstance(sys.stdout, PlainStream):
        sys.stdout = LightStream(sys.stdout)  # type: ignore[assignment]
    elif theme != 'light' and isinstance(sys.stdout, LightStream):
        sys.stdout = sys.stdout._stream  # type: ignore[assignment]
    return theme


def title_enabled() -> bool:
    if os.environ.get('SIGIL_TERMINAL_TITLE', '1').strip().lower() in ('0', 'false', 'no', 'off'):
        return False
    if plain_enabled():
        return False
    stream = sys.__stdout__
    return bool(stream is not None and stream.isatty())


# Escape sequences and control characters in text from the model, a tool, a file or a web page:
# OSC (title, clipboard write: OSC 52, hyperlinks), CSI (cursor moves that hide or rewrite lines),
# other ESC sequences, C0 controls except tab/newline, DEL, and C1 controls.
_TERMINAL_CONTROLS = re.compile(
    r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?'   # OSC ... BEL / ST
    r'|\x1b\[[0-?]*[ -/]*[@-~]'               # CSI
    r'|\x1b[@-Z\\-_]'                        # other two-byte ESC sequences
    r'|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]')


def strip_terminal_controls(text: Any) -> Any:
    """Text that came from outside (model, tools, files, pages) made safe to print: no escape
    sequences or control characters, a lone carriage return becomes a newline (it could hide what
    precedes it on the line). SIGIL_RAW_TERMINAL_OUTPUT=1 prints text unchanged."""
    if not isinstance(text, str) or not text:
        return text
    if os.environ.get('SIGIL_RAW_TERMINAL_OUTPUT', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return text
    return _TERMINAL_CONTROLS.sub('', text.replace('\r\n', '\n').replace('\r', '\n'))


def clean_event(value: Any, depth: int = 0) -> Any:
    """strip_terminal_controls applied to every string in an event (dicts and lists, bounded depth)."""
    if isinstance(value, str):
        return strip_terminal_controls(value)
    if depth > 6:
        return value
    if isinstance(value, dict):
        return {key: clean_event(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clean_event(item, depth + 1) for item in value)
    return value


def set_title(text: str) -> bool:
    """Set the terminal title (OSC 0). Returns True when it was written."""
    if not title_enabled():
        return False
    try:
        sys.__stdout__.write(f'\x1b]0;{strip_terminal_controls(str(text))[:200]}\x07')
        sys.__stdout__.flush()
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return False
    return True


def session_title(agent: Any, state: str) -> str:
    name = getattr(agent, '_session_name', None) or agent.runtime_config.cwd.name or str(agent.runtime_config.cwd)
    return f'Sigil: {name} - {state}'
