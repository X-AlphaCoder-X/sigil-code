"""Reading .env lines the way python-dotenv does (github.com/theskumar/python-dotenv, src/dotenv/parser.py).

- Blank lines and lines starting with # are skipped. `export KEY=value` is the same as `KEY=value`.
- Unquoted value: a # starts a comment only after whitespace (`re.sub(r"\\s+#.*", "", value)`), so
  `KEY=1  # note` is `1` and `KEY=abc#def` keeps its #. Surrounding whitespace is removed. One deliberate
  difference: `KEY=   # note` is empty here (python-dotenv would read "# note").
- Quoted value ('...' or "..."): everything inside the quotes is kept, # included. \\" (or \\') and \\\\
  inside are an escaped quote and backslash. Anything after the closing quote is a comment.
Why: the three .env readers used to keep an inline comment as part of the value, so
`SIGIL_TRUST_ALL=1  # note` was read as "1  # note" and the setting did nothing.
"""
from __future__ import annotations

import re

_INLINE_COMMENT = re.compile(r'\s+#.*')


def parse_env_line(line: str) -> tuple[str, str] | None:
    """(key, value) for one .env line, or None for a blank line, a comment or a line without a key."""
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        return None
    key, value = line.split('=', 1)
    key = key.strip()
    if key.startswith('export '):
        key = key[len('export '):].strip()
    if not key:
        return None
    raw = value
    value = value.strip()
    if value[:1] in ('"', "'"):
        quote = value[0]
        chars: list[str] = []
        index = 1
        while index < len(value):
            char = value[index]
            if char == '\\' and index + 1 < len(value) and value[index + 1] in (quote, '\\'):
                chars.append(value[index + 1])
                index += 2
                continue
            if char == quote:
                return key, ''.join(chars)       # what follows the closing quote is a comment
            chars.append(char)
            index += 1
        # No closing quote: read as before (quotes stripped), and let the caller log it.
        raise UnterminatedQuote(key, value.strip(quote))
    # The comment rule runs before the spaces after '=' are trimmed, so `KEY=   # note` is empty
    # (python-dotenv trims first and would read "# note"). `KEY=#x` has no space and keeps "#x".
    return key, _INLINE_COMMENT.sub('', raw).strip()


class UnterminatedQuote(ValueError):
    """A quoted value without its closing quote. `value` is the old reading (quotes stripped)."""

    def __init__(self, key: str, value: str) -> None:
        super().__init__(f'{key}: the value starts with a quote that is never closed')
        self.key = key
        self.value = value


def read_env_lines(text: str, source: str = '.env') -> list[tuple[str, str]]:
    """Every (key, value) of a .env file's text. A value whose quote is never closed is still read the
    old way (quotes stripped) and logged by key name and line number, never by value."""
    pairs: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        try:
            parsed = parse_env_line(line)
        except UnterminatedQuote as exc:
            from .debug_log import get_logger

            get_logger('settings').warning('%s line %d: %s', source, number, exc)
            parsed = (exc.key, exc.value)
        if parsed is not None:
            pairs.append(parsed)
    return pairs
