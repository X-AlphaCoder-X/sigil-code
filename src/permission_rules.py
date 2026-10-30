"""Permission rules: ``permissions.allow / ask / deny``.

Rule syntax:

    Bash                     every bash command
    Bash(git diff *)         commands matching the pattern ('*' matches anything;
                             a trailing ':*' is the older prefix form)
    Read(./.env)             file paths; Read also covers glob, grep and list_dir
    Edit(src/**)             Edit covers edit_file, write_file and notebook_edit
    WebFetch(domain:x.com)   a host and its subdomains
    mcp__server / mcp__server__tool

Path patterns: ``//abs/path`` is absolute, ``~/x`` is under the home folder,
``/x`` is relative to the project root, ``./x`` and ``x`` are relative to the
working folder. ``**`` matches any number of folders and ``*`` one name. A
pattern without a ``/`` matches a file name anywhere.

Sigil tool names work too (``bash``, ``read_file(...)``).
Deny wins over ask, and ask wins over allow.
"""
from __future__ import annotations

import fnmatch
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .command_hooks import HOOK_TOOL_ALIASES
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

# A rule for the key tool also covers these rule tool names.
_RULE_COVERS = {
    'Read': {'Read', 'Glob', 'Grep', 'LS'},
    'Edit': {'Edit', 'Write', 'NotebookEdit'},
}
_PATH_TOOLS = {'Read', 'Glob', 'Grep', 'LS', 'Edit', 'Write', 'NotebookEdit'}
_RULE_RE = re.compile(r'^\s*([A-Za-z0-9_\-]+)\s*(?:\((.*)\))?\s*$', re.S)
# Command separators: && || ; | newline, and a lone & (background) that is not part of a
# redirection such as 2>&1 or &>.
_COMMAND_SPLIT_RE = re.compile(r'\s*(?:&&|\|\||;|\||\r?\n|(?<![<>&])&(?![&>]))\s*')
# Text that runs another command inside the line; an allow rule never covers it.
# Why both: `Bash(git diff *)` used to approve `git diff x<newline>rm -rf ~`, `git diff & rm -rf ~` and
# `git diff $(curl ...|sh)`, and an allow match also skipped the shell guard's question.
_SUBSTITUTION_RE = re.compile(r'\$\(|`|<\(|>\(')


@dataclass(frozen=True)
class PermissionRule:
    tool: str
    specifier: str | None
    raw: str


def parse_rule(text: str) -> PermissionRule | None:
    match = _RULE_RE.match(str(text))
    if match is None:
        return None
    specifier = match.group(2)
    if specifier is not None:
        specifier = specifier.strip()
        if specifier in ('', '*'):
            specifier = None
    return PermissionRule(tool=match.group(1), specifier=specifier, raw=str(text).strip())


def rule_tool_name(tool_name: str) -> str:
    from .plugin_runtime import base_tool_name

    tool_name = base_tool_name(tool_name)  # a plugin alias is matched as the tool it runs
    return HOOK_TOOL_ALIASES.get(tool_name, tool_name)


def _tool_matches(rule: PermissionRule, tool_name: str) -> bool:
    rule_name = rule_tool_name(tool_name)
    from .plugin_runtime import base_tool_name

    if rule.tool in (tool_name, rule_name, base_tool_name(tool_name)):
        return True
    # A Sigil-name rule ('edit_file') maps to its rule tool name too.
    rule_tool = rule_tool_name(rule.tool)
    if rule_name in _RULE_COVERS.get(rule_tool, ()):
        return True
    if rule.tool.startswith('mcp__') and tool_name.startswith('mcp__'):
        prefix = rule.tool[:-1] if rule.tool.endswith('*') else rule.tool
        return tool_name == prefix or tool_name.startswith(prefix.rstrip('_') + '__')
    return False


def _command_pattern_matches(pattern: str, command: str) -> bool:
    command = command.strip()
    if pattern.endswith(':*'):
        prefix = pattern[:-2]
        return command == prefix or command.startswith(prefix)
    regex = '^' + re.escape(pattern).replace(r'\*', '.*') + '$'
    if re.match(regex, command, re.S):
        return True
    # 'ls *' also matches a bare 'ls'.
    return pattern.endswith(' *') and command == pattern[:-2]


def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith('**/', i):
            out.append('(?:.*/)?')
            i += 3
        elif pattern.startswith('**', i):
            out.append('.*')
            i += 2
        elif pattern[i] == '*':
            out.append('[^/]*')
            i += 1
        elif pattern[i] == '?':
            out.append('[^/]')
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    flags = re.I if os.name == 'nt' else 0
    return re.compile('^' + ''.join(out) + '$', flags)


def _path_pattern_matches(pattern: str, raw_path: str, cwd: Path, project_root: Path) -> bool:
    target = Path(raw_path).expanduser()
    if not target.is_absolute():
        target = cwd / target
    try:
        target = target.resolve()
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass
    target_posix = target.as_posix()
    if '/' not in pattern:
        return bool(_glob_regex(pattern).match(target.name)) or fnmatch.fnmatch(target.name, pattern)
    if pattern.startswith('//'):
        rest = pattern[2:]
        # '//C:/x' on Windows, '//etc/x' elsewhere.
        base = Path(rest) if re.match(r'^[A-Za-z]:', rest) else Path('/' + rest)
    elif pattern.startswith('~/'):
        base = Path.home() / pattern[2:]
    elif pattern.startswith('/'):
        base = project_root / pattern[1:]
    else:
        base = cwd / (pattern[2:] if pattern.startswith('./') else pattern)
    full = base.as_posix()
    # Resolve the fixed part so 'C:/x' and '/c/x' style differences don't matter.
    fixed = re.split(r'[*?\[]', full, maxsplit=1)[0]
    if fixed and fixed != full:
        try:
            resolved_fixed = Path(fixed).resolve().as_posix()
            if fixed.endswith('/') and not resolved_fixed.endswith('/'):
                resolved_fixed += '/'
            full = resolved_fixed + full[len(fixed):]
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass
    elif fixed == full:
        try:
            full = Path(full).resolve().as_posix()
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass
    if _glob_regex(full).match(target_posix):
        return True
    # 'src/' or 'src/**' style folder patterns match everything inside.
    return full.endswith('/**') and _glob_regex(full[:-3]).match(target_posix) is not None


def _path_argument(arguments: Mapping[str, Any]) -> str | None:
    for key in ('path', 'file_path', 'notebook_path'):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def rule_matches(
    rule: PermissionRule,
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    cwd: Path,
    project_root: Path | None = None,
    for_allow: bool = False,
) -> bool:
    if not _tool_matches(rule, tool_name):
        return False
    if rule.specifier is None:
        return True
    rule_name = rule_tool_name(tool_name)
    spec = rule.specifier
    if rule_name in ('Bash', 'PowerShell'):  # both read arguments['command'] (S11)
        command = str(arguments.get('command', ''))
        if for_allow and _SUBSTITUTION_RE.search(command):
            return False
        parts = [part for part in _COMMAND_SPLIT_RE.split(command) if part.strip()] or [command]
        if len(parts) == 1:
            return _command_pattern_matches(spec, command)
        # Compound commands: an allow rule must cover every part; deny/ask any part
        # (or the whole line, e.g. a deny rule written for the full command).
        hits = [_command_pattern_matches(spec, part) for part in parts]
        if for_allow:
            return all(hits)
        return any(hits) or _command_pattern_matches(spec, command)
    if rule_name == 'WebFetch' and spec.startswith('domain:'):
        host = (urllib.parse.urlparse(str(arguments.get('url', ''))).hostname or '').lower()
        domain = spec[len('domain:'):].strip().lower()
        return host == domain or host.endswith('.' + domain)
    if rule_name in _PATH_TOOLS:
        raw = _path_argument(arguments)
        if raw is None:
            raw = '.' if rule_name in ('Glob', 'Grep', 'LS') else None
        if raw is None:
            return False
        return _path_pattern_matches(spec, raw, Path(cwd), Path(project_root or cwd))
    return False


@dataclass(frozen=True)
class PermissionDecision:
    behavior: str  # 'allow' | 'ask' | 'deny'
    rule: PermissionRule


@dataclass(frozen=True)
class PermissionRules:
    allow: tuple[PermissionRule, ...] = ()
    ask: tuple[PermissionRule, ...] = ()
    deny: tuple[PermissionRule, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> 'PermissionRules':
        permissions = (settings or {}).get('permissions') if settings else None
        if not isinstance(permissions, dict):
            return cls()
        lists: dict[str, list[PermissionRule]] = {'allow': [], 'ask': [], 'deny': []}
        errors: list[str] = []
        for kind in lists:
            raw_rules = permissions.get(kind)
            if not isinstance(raw_rules, list):
                continue
            for raw in raw_rules:
                rule = parse_rule(raw) if isinstance(raw, str) else None
                if rule is None:
                    errors.append(f'permissions.{kind}: cannot parse {raw!r}')
                    continue
                lists[kind].append(rule)
        return cls(tuple(lists['allow']), tuple(lists['ask']), tuple(lists['deny']), tuple(errors))

    @property
    def empty(self) -> bool:
        return not (self.allow or self.ask or self.deny)

    def decide(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        cwd: Path,
        project_root: Path | None = None,
    ) -> PermissionDecision | None:
        for behavior, rules in (('deny', self.deny), ('ask', self.ask), ('allow', self.allow)):
            for rule in rules:
                if rule_matches(rule, tool_name, arguments, cwd=cwd, project_root=project_root,
                                for_allow=(behavior == 'allow')):
                    return PermissionDecision(behavior, rule)
        return None


def suggest_rule(tool_name: str, arguments: Mapping[str, Any], cwd: Path) -> str:
    """The rule saved by an 'always allow' answer: narrow, so it allows only this."""
    rule_name = rule_tool_name(tool_name)
    if rule_name in ('Bash', 'PowerShell'):
        return f'{rule_name}({str(arguments.get("command", "")).strip()})'
    if rule_name == 'WebFetch':
        host = urllib.parse.urlparse(str(arguments.get('url', ''))).hostname
        return f'WebFetch(domain:{host})' if host else 'WebFetch'
    raw = _path_argument(arguments)
    if rule_name in _PATH_TOOLS and raw:
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = Path(cwd) / target
        try:
            rel = target.resolve().relative_to(Path(cwd).resolve()).as_posix()
            return f'{rule_name}(./{rel})'
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            return f'{rule_name}(//{target.as_posix().lstrip("/")})'
    return rule_name
