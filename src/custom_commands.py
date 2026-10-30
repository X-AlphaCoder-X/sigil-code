"""Custom slash commands and skills as /name.

Commands are Markdown files:
    .sigil/commands/*.md        project commands (subfolders too)
    ~/.sigil/commands/*.md      personal commands
The file name (without .md) is the command name; a project command wins over a
personal one with the same name. Every discovered skill also runs as
/skill-name.

Frontmatter (optional): description, argument-hint, model, allowed-tools.
Body placeholders:
    $ARGUMENTS            everything after the command name
    $ARGUMENTS[N] / $N    one argument (0-based; quotes group words)
    !`cmd`                replaced by the command's output (bash, through the
                          normal shell permission checks)
    @path                 replaced by the file's contents (through path access)
If the body has no $ARGUMENTS placeholder, the arguments are appended.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

try:
    import yaml
except ImportError:  # PyYAML is optional
    yaml = None

MAX_MENTION_CHARS = 100_000
_SHELL_RE = re.compile(r'!`([^`\n]+)`')
_MENTION_RE = re.compile(r'(?<![\w`])@((?:"[^"]+")|(?:[^\s`,;:!?()\[\]{}<>"\']+))')
_ARG_INDEX_RE = re.compile(r'\$ARGUMENTS\[(\d+)\]|\$(\d+)(?!\d)')


def _fallback_frontmatter(raw: str) -> dict[str, Any]:
    """The YAML that frontmatter uses, without PyYAML.

    Handles ``key: value``, quoted values, ``[a, b]`` lists, ``- item`` lists on
    the following lines, and ``|`` / ``>`` block text.
    """
    data: dict[str, Any] = {}
    lines = raw.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith('#') or line.startswith((' ', '\t')):
            continue
        key, sep, value = line.partition(':')
        if not sep or not key.strip():
            continue
        key, value = key.strip(), value.strip()
        if value in ('|', '|-', '|+', '>', '>-', '>+'):
            block: list[str] = []
            while index < len(lines) and (lines[index].startswith((' ', '\t')) or not lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            joiner = '\n' if value.startswith('|') else ' '
            data[key] = joiner.join(block).strip()
        elif value == '':
            items: list[str] = []
            while index < len(lines) and (lines[index].lstrip().startswith('-') or not lines[index].strip()):
                item = lines[index].strip()
                if item:
                    items.append(item[1:].strip().strip('"\''))
                index += 1
            data[key] = items if items else ''
        elif value.startswith('[') and value.endswith(']'):
            data[key] = [v.strip().strip('"\'') for v in value[1:-1].split(',') if v.strip()]
        else:
            data[key] = value.strip('"\'')
    return data


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """YAML frontmatter between --- lines (PyYAML when installed, else a built-in parser)."""
    if not text.startswith('---'):
        return {}, text
    match = re.search(r'\n---[ \t]*(?:\n|$)', text[3:])
    if not match:
        return {}, text
    raw = text[3:match.start() + 3]
    body = text[match.end() + 3:]
    data: dict[str, Any] = {}
    if yaml is not None:
        try:
            loaded = yaml.safe_load(raw) or {}
            if isinstance(loaded, dict):
                data = loaded
        except Exception as _ignored:
            _log_ignored(_ignored)
            data = {}
    if not data:
        data = _fallback_frontmatter(raw)
    return data, body


def _split_outside_parens(text: str, separators: str) -> list[str]:
    parts, depth, current = [], 0, ''
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch in separators and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += ch
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def tool_list(value: Any) -> tuple[str, ...]:
    """tools / allowed-tools as a tuple: a YAML list, or a comma- or space-separated string."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    text = str(value)
    parts = _split_outside_parens(text, ',')
    if len(parts) <= 1:
        parts = _split_outside_parens(text, ' \t')
    return tuple(parts)


@dataclass(frozen=True)
class CustomCommand:
    name: str
    body: str
    source: str  # 'project', 'user' or 'skill'
    path: Path | None = None
    description: str = ''
    argument_hint: str = ''
    model: str | None = None
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    fork_agent: str | None = None  # context: fork -> run in this subagent type


BUILTIN_COMMANDS = {
    'init': CustomCommand(
        name='init', source='built-in', description='Write a starter SIGIL.md for this repository',
        body=(
            'Look through this repository and write a SIGIL.md in its root (update it if it exists). '
            'Cover: what the project is, how to install, build, run the tests and lint (exact '
            'commands), the folder layout and main modules, code style and conventions you can see '
            'in the code, and anything a new contributor would trip over. Keep it short and factual; '
            'only write what you checked in the files. $ARGUMENTS'
        ),
    ),
    'review': CustomCommand(
        name='review', source='built-in', argument_hint='[files or PR]',
        description='Review the current changes',
        body=(
            'Review the current changes: run `git status` and `git diff` (staged and unstaged), or '
            'review $ARGUMENTS if given. Report real problems first (bugs, missing error handling, '
            'broken edge cases, missing tests), each with file:line and a suggested fix, then smaller '
            'style notes. Do not edit files.'
        ),
    ),
    'security-review': CustomCommand(
        name='security-review', source='built-in', argument_hint='[files]',
        description='Security review of the current changes',
        body=(
            'Do a security review of the current changes (`git diff`, staged and unstaged), or of '
            '$ARGUMENTS if given. Look for injection, unsafe deserialization, path traversal, secrets '
            'in code, weak auth checks and unsafe defaults. For each finding give file:line, the risk, '
            'how it could be triggered, and a fix. Do not edit files.'
        ),
    ),
}


def _home() -> Path:
    return Path.home()


def _load_dir(folder: Path, source: str) -> dict[str, CustomCommand]:
    found: dict[str, CustomCommand] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.rglob('*.md')):
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as _ignored:
            _log_ignored(_ignored)
            continue
        meta, body = parse_frontmatter(text)
        name = path.stem
        found.setdefault(name, CustomCommand(
            name=name,
            body=body.strip(),
            source=source,
            path=path,
            description=str(meta.get('description', '') or ''),
            argument_hint=str(meta.get('argument-hint', '') or ''),
            model=(str(meta['model']) if meta.get('model') else None),
            allowed_tools=tool_list(meta.get('allowed-tools')),
        ))
    return found


def _from_repository(path: Path | None) -> bool:
    """A command or skill file the repository supplies (not ~/.sigil, not a --plugin-dir plugin)."""
    if path is None:
        return False
    from .sigil_plugins import loaded_plugins

    try:
        resolved = Path(path).resolve()
        mine = [(_home() / '.sigil').resolve(), *(Path(p.root).resolve() for p in loaded_plugins())]
    except OSError as _ignored:
        _log_ignored(_ignored)
        return True
    return not any(resolved.is_relative_to(folder) for folder in mine)


def _grants_allowed(cwd: Path) -> bool:
    from .workspace_trust import is_trusted

    return is_trusted(cwd)


def discover_commands(cwd: Path, skills_runtime: Any = None) -> dict[str, CustomCommand]:
    """All custom commands for a workspace: project, then user, then skills.

    `allowed-tools` of a project command or skill pre-approve tools (and its !`cmd` lines) only
    when the folder is trusted (workspace_trust); until then they run with the session's normal
    permissions."""
    from dataclasses import replace as _replace

    commands: dict[str, CustomCommand] = {}
    trusted = _grants_allowed(cwd)
    for name, command in _load_dir(Path(cwd) / '.sigil' / 'commands', 'project').items():
        commands[name] = command if trusted else _replace(command, allowed_tools=())
    for name, command in _load_dir(_home() / '.sigil' / 'commands', 'user').items():
        commands.setdefault(name, command)
    from .sigil_plugins import expand_root, loaded_plugins

    for plugin in loaded_plugins():  # --plugin-dir: /plugin:name, and /name when it is free
        for folder in plugin.command_dirs:
            for name, command in _load_dir(folder, f'plugin:{plugin.name}').items():
                command = _replace(command, body=expand_root(command.body, plugin.root))
                commands.setdefault(f'{plugin.name}:{name}', _replace(command, name=f'{plugin.name}:{name}'))
                commands.setdefault(name, command)
    for name, command in BUILTIN_COMMANDS.items():
        commands.setdefault(name, command)
    if skills_runtime is not None:
        try:
            names = skills_runtime.list_skills()
        except Exception as _ignored:
            _log_ignored(_ignored)
            names = []
        for name in names:
            if name in commands:
                continue
            entry = skills_runtime.get_skill(name)
            if entry is None:
                continue
            metadata = entry.metadata
            if getattr(metadata, 'user_invocable', True) is False:
                continue  # the model may load it, but it is not a /command
            allowed = tuple(getattr(metadata, 'allowed_tools', None) or ()) or \
                tool_list(getattr(metadata, 'compatible_tools', None))
            if not trusted and _from_repository(entry.file_path):
                allowed = ()
            fork = None
            if str(getattr(metadata, 'context', '')).strip().lower() == 'fork':
                fork = getattr(metadata, 'agent', '') or 'general-purpose'
            commands[name] = CustomCommand(
                name=name,
                body=entry.content,
                source='skill',
                path=entry.file_path,
                description=metadata.description,
                allowed_tools=allowed,
                fork_agent=fork,
            )
    return commands


def split_arguments(args: str) -> list[str]:
    try:
        return shlex.split(args, posix=True)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return args.split()


def substitute_arguments(body: str, args: str) -> str:
    positional = split_arguments(args)

    def _index(match: re.Match[str]) -> str:
        index = int(match.group(1) if match.group(1) is not None else match.group(2))
        return positional[index] if index < len(positional) else ''

    has_placeholder = '$ARGUMENTS' in body or _ARG_INDEX_RE.search(body) is not None
    text = _ARG_INDEX_RE.sub(_index, body)
    text = text.replace('$ARGUMENTS', args)
    if not has_placeholder and args.strip():
        text = f'{text.rstrip()}\n\nARGUMENTS: {args}'
    return text


def expand_file_mentions(text: str, agent: Any, *, max_chars: int | None = None) -> tuple[str, list[str]]:
    """Attach the contents of each @path that names a readable file.

    Paths go through the agent's path access (workspace, --add-dir, other
    directories). Returns (new text, attached paths). Unknown names are left as
    typed, so e-mail addresses and @decorators stay intact.
    """
    from .agent_tools import ToolExecutionError, _resolve_path

    context = getattr(agent, 'tool_context', None)
    if context is None:
        return text, []
    if max_chars is None:
        max_chars = MAX_MENTION_CHARS
    attached: list[str] = []
    blocks: list[str] = []
    for match in _MENTION_RE.finditer(text):
        raw = match.group(1).strip('"')
        raw = raw.rstrip('.')
        if not raw or raw in attached:
            continue
        try:
            target = _resolve_path(raw, context)
        except (ToolExecutionError, OSError) as _ignored:
            _log_ignored(_ignored)
            continue
        if not target.is_file():
            continue
        try:
            content = target.read_text(encoding='utf-8', errors='replace')
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
        if len(content) > max_chars:
            content = content[:max_chars] + f'\n... [truncated: file is {len(content):,} chars]'
        attached.append(raw)
        blocks.append(f'<file path="{raw}">\n{content}\n</file>')
    if not blocks:
        return text, []
    return text + '\n\n' + '\n\n'.join(blocks), attached


def _shell_injection_disabled(agent: Any) -> bool:
    """disableSkillShellExecution (or SIGIL_DISABLE_SKILL_SHELL=1): !`cmd` in skills and commands is not run."""
    import os

    if os.environ.get('SIGIL_DISABLE_SKILL_SHELL', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return True
    try:
        from .settings import load_settings

        return load_settings(agent.runtime_config.cwd).get('disableSkillShellExecution') is True
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def run_shell_injections(text: str, agent: Any, allowed_tools: tuple[str, ...] = ()) -> str:
    """Replace each !`cmd` with its output, through the bash permission checks.

    A Bash(...) entry in allowed-tools approves matching commands the way an
    allow rule does; anything else follows the session's shell permissions.
    """

    from .agent_tools import execute_tool
    from .permission_rules import parse_rule, rule_matches

    if '!`' not in text:
        return text
    if _shell_injection_disabled(agent):
        return _SHELL_RE.sub(lambda m: '[shell command execution disabled by disableSkillShellExecution]', text)
    rules = [rule for rule in (parse_rule(t) for t in allowed_tools) if rule is not None]

    def _run(match: re.Match[str]) -> str:
        command = match.group(1).strip()
        context = agent.tool_context
        if any(rule_matches(rule, 'bash', {'command': command}, cwd=context.root, for_allow=True)
               for rule in rules):
            context = agent._approved_context('bash')
        result = execute_tool(agent.tool_registry, 'bash', {'command': command}, context)
        if not result.ok:
            return f'(command not run: {result.content.strip()})'
        output = result.content
        # bash results look like "exit_code=0\n[stdout]\n...\n[stderr]\n..."
        stdout = output.split('[stdout]', 1)[-1].split('[stderr]', 1)[0].strip()
        return stdout

    return _SHELL_RE.sub(_run, text)


def render_command_list(commands: dict[str, CustomCommand]) -> list[str]:
    lines = []
    for name in sorted(commands):
        command = commands[name]
        hint = f' {command.argument_hint}' if command.argument_hint else ''
        description = f' - {command.description}' if command.description else ''
        lines.append(f'/{name}{hint}{description} ({command.source})')
    return lines
