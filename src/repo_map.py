"""Repo map (opt-in): the project's files and their main symbols, sent at the start.

Turn it on with SIGIL_REPO_MAP=1 or "repoMap": true in settings; /map shows it.
Files come from `git ls-files` (or a folder walk that skips build and cache
folders). Python symbols are read with the standard ast module, C and C++ with
the shredder when tree-sitter is installed, and other languages with simple
patterns. Files whose symbols are used by the most other files come first (a
cheap form of Aider's ranking: names defined in many files and plain words
count little, test files count less). The map is cut at SIGIL_REPO_MAP_CHARS
(6000) characters; paths without symbols fill what is left. Reading stops after
SIGIL_REPO_MAP_SECONDS (5) so a huge repo cannot hold up the start.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

DEFAULT_CHARS = 6000
MAX_FILES = 3000
MAX_FILE_BYTES = 200_000
SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env', 'dist', 'build', 'target',
             '.mypy_cache', '.pytest_cache', '.tox', '.idea', '.vscode', '.port_sessions', '.next', 'vendor'}
CODE_SUFFIXES = {'.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.go', '.rs', '.java', '.kt', '.cs', '.rb', '.php',
                 '.c', '.h', '.cc', '.cpp', '.hpp', '.cxx', '.swift', '.scala', '.lua', '.mq4', '.mq5', '.mqh'}
_PATTERNS = {
    'js': re.compile(r'^\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\*?\s+([A-Za-z_$][\w$]*)|'
                     r'class\s+([A-Za-z_$][\w$]*)|(?:const|let)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(|'
                     r'(?:interface|type|enum)\s+([A-Za-z_$][\w$]*))', re.M),
    'go': re.compile(r'^(?:func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)|type\s+([A-Za-z_]\w*))', re.M),
    'rs': re.compile(r'^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_]\w*)', re.M),
    'java': re.compile(r'^\s*(?:public|private|protected|internal|static|final|abstract|sealed|\s)*'
                       r'(?:class|interface|enum|record|struct)\s+([A-Za-z_]\w*)', re.M),
    'rb': re.compile(r'^\s*(?:def|class|module)\s+([A-Za-z_][\w.?!]*)', re.M),
    'c': re.compile(r'^[A-Za-z_][\w\s\*&:<>,]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*(?:const\s*)?\{', re.M),
}
_KIND = {'.js': 'js', '.jsx': 'js', '.ts': 'js', '.tsx': 'js', '.mjs': 'js', '.go': 'go', '.rs': 'rs',
         '.java': 'java', '.kt': 'java', '.cs': 'java', '.scala': 'java', '.swift': 'java', '.rb': 'rb',
         '.c': 'c', '.h': 'c', '.cc': 'c', '.cpp': 'c', '.hpp': 'c', '.cxx': 'c', '.mq4': 'c', '.mq5': 'c',
         '.mqh': 'c', '.php': 'java', '.lua': 'rb'}
_WORD = re.compile(r'[A-Za-z_]\w{2,}')


def enabled(cwd: Path) -> bool:
    raw = os.environ.get('SIGIL_REPO_MAP', '').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    try:
        from .settings import load_settings

        return load_settings(cwd).get('repoMap') is True
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def list_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=root,
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return sorted(f for f in files if not (set(Path(f).parts) & SKIP_DIRS))[:MAX_FILES]
    except (OSError, subprocess.SubprocessError) as _ignored:
        _log_ignored(_ignored)
        pass
    found: list[str] = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        for name in sorted(names):
            found.append(Path(folder, name).relative_to(root).as_posix())
            if len(found) >= MAX_FILES:
                return found
    return found


def _python_symbols(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return []
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and not n.name.startswith('_')]
            symbols.append(f'class {node.name}' + (f': {", ".join(methods[:8])}' if methods else ''))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
            symbols.append(f'def {node.name}()')
    return symbols


def _c_symbols(text: str, suffix: str) -> list[str]:
    try:
        from . import shredder

        if shredder.CPP_LANGUAGE is not None and suffix in ('.c', '.h', '.cc', '.cpp', '.hpp', '.cxx'):
            outline = shredder.get_file_outline(text, suffix)
            return [line.strip() for line in outline.splitlines() if line.strip() and not line.startswith(' ')][:30]
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    return [m for m in (match.group(1) for match in _PATTERNS['c'].finditer(text))
            if m not in ('if', 'for', 'while', 'switch', 'return')]


def _read_code(path: Path) -> str | None:
    if path.suffix.lower() not in CODE_SUFFIXES:
        return None
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError as _ignored:
        _log_ignored(_ignored)
        return None


def file_symbols(path: Path) -> list[str]:
    text = _read_code(path)
    return [] if text is None else symbols_from_text(text, path.suffix.lower())


def symbols_from_text(text: str, suffix: str) -> list[str]:
    if suffix == '.py':
        return _python_symbols(text)
    kind = _KIND.get(suffix)
    if kind == 'c':
        return _c_symbols(text, suffix)
    if kind is None:
        return []
    names = []
    for match in _PATTERNS[kind].finditer(text):
        name = next((group for group in match.groups() if group), None)
        if name and name not in names:
            names.append(name)
    return names


def _weight(name: str) -> float:
    """Specific names (CamelCase, snake_case) say more than plain words like load or run."""
    return 1.0 if '_' in name.strip('_') or any(c.isupper() for c in name) else 0.2


def _is_test(rel: str) -> bool:
    parts = Path(rel).parts
    return any(part in ('tests', 'test', '__tests__', 'spec') for part in parts[:-1]) or \
        Path(rel).name.startswith('test_') or '.test.' in rel or '.spec.' in rel


def _names(symbols: list[str]) -> list[str]:
    out = []
    for symbol in symbols:
        word = re.sub(r'^(class|def|struct|namespace)\s+', '', symbol).split('(')[0].split(':')[0].split()[0:1]
        if word:
            out.append(word[0])
    return out


def build_repo_map(root: Path, max_chars: int | None = None) -> str:
    root = Path(root).resolve()
    if max_chars is None:
        try:
            max_chars = max(500, int(os.environ.get('SIGIL_REPO_MAP_CHARS', DEFAULT_CHARS)))
        except ValueError as _ignored:
            _log_ignored(_ignored)
            max_chars = DEFAULT_CHARS
    try:
        budget = float(os.environ.get('SIGIL_REPO_MAP_SECONDS', '5'))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        budget = 5.0
    deadline = time.monotonic() + budget
    files = list_files(root)
    symbols: dict[str, list[str]] = {}
    files_using: Counter[str] = Counter()  # name -> number of files that mention it
    defined_in: Counter[str] = Counter()  # name -> number of files that define it
    own: dict[str, set[str]] = {}
    for rel in files:
        if time.monotonic() > deadline:
            break
        text = _read_code(root / rel)
        if text is None:
            continue
        found = symbols_from_text(text, Path(rel).suffix.lower())
        if found:
            symbols[rel] = found
            defined_in.update(set(_names(found)))
        words = set(_WORD.findall(text))
        files_using.update(words)
        own[rel] = words

    def score(rel: str) -> float:
        total = 0.0
        for name in set(_names(symbols[rel])):
            if defined_in[name] > 2:
                continue  # run, main, setUp...: defined everywhere, says nothing
            others = files_using[name] - (1 if name in own.get(rel, ()) else 0)
            total += max(others, 0) * _weight(name)
        return total * (0.25 if _is_test(rel) else 1.0)

    ranked = sorted(symbols, key=lambda rel: (-score(rel), rel))
    lines = [f'Repo map of {root} ({len(files)} files; most used first):']
    used = len(lines[0])
    shown: set[str] = set()
    for rel in ranked:
        block = f'{rel}\n' + '\n'.join(f'  {symbol}' for symbol in symbols[rel][:25])
        if used + len(block) + 1 > max_chars:
            continue
        lines.append(block)
        used += len(block) + 1
        shown.add(rel)
    rest = [rel for rel in files if rel not in shown]
    listed = 0
    for rel in rest:
        if used + len(rel) + 1 > max_chars - 40:
            break
        lines.append(rel)
        used += len(rel) + 1
        listed += 1
    if listed < len(rest):
        lines.append(f'... {len(rest) - listed} more file(s)')
    return '\n'.join(lines)
