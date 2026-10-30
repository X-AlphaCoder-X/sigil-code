"""Subagent types for delegate_agent's ``agent`` argument.

Built in:
    general-purpose   every tool the parent has (except spawning more agents)
    Explore           read-only tools: search and read, no writes, no shell
    Plan              read-only tools, for researching and writing a plan

Custom agents are Markdown files with frontmatter (name, description, tools,
disallowedTools, model), from ``.sigil/agents/**/*.md`` (project) and
``~/.sigil/agents/**/*.md`` (personal), plus any skills-repo agents folder
(see skills_runtime). The body becomes the child's extra system prompt.
``tools`` may use rule tool names (Read, Grep, Glob, Bash, Edit, Write,
WebFetch, ...) or Sigil names; ``model`` may be ``inherit`` or a Claude alias
(mapped through SIGIL_MODEL_<ALIAS>, see providers.resolve_model_alias).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .command_hooks import HOOK_TOOL_ALIASES
from .custom_commands import parse_frontmatter, tool_list
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

READ_ONLY_TOOLS = frozenset({
    'read_file', 'glob_search', 'grep_search', 'list_dir', 'LSP',
    'web_fetch', 'web_search', 'tool_search', 'todo_write', 'sleep',
    'skill_get', 'skill_list', 'skill_search', 'agent_get', 'agent_list', 'agent_search',
    'plan_get', 'task_get', 'task_list', 'config_get', 'config_list',
    'mcp_list_tools', 'mcp_list_resources', 'mcp_read_resource',
    'worktree_status', 'search_status', 'search_list_providers', 'background_agent_status',
})
# rule tool name -> Sigil tool names
_RULE_TO_SIGIL: dict[str, set[str]] = {}
for _sigil, _rule in HOOK_TOOL_ALIASES.items():
    _RULE_TO_SIGIL.setdefault(_rule, set()).add(_sigil)
_RULE_TO_SIGIL.setdefault('Edit', set()).update({'edit_file', 'notebook_edit'})


@dataclass(frozen=True)
class SubagentDefinition:
    name: str
    description: str
    tools: tuple[str, ...] | None = None  # None = all of the parent's tools
    disallowed_tools: tuple[str, ...] = ()
    model: str | None = None
    prompt: str = ''
    source: str = 'built-in'
    read_only: bool = False
    path: Path | None = field(default=None, compare=False)
    max_turns: int | None = None
    permission_mode: str | None = None  # default, plan, acceptEdits, bypassPermissions
    isolation: str | None = None  # 'worktree': run in a fresh git worktree
    skills: tuple[str, ...] = ()  # skills loaded into the agent's prompt
    mcp_servers: tuple[str, ...] = ()  # only these MCP servers' tools (empty = all)
    hooks: dict | None = field(default=None, compare=False)  # hooks active while it runs
    memory: str | None = None  # user, project or local: a MEMORY.md of its own
    effort: str | None = None  # low, medium, high, max
    max_budget_usd: float | None = None  # cost cap for one run

    def filter_tools(self, registry: dict[str, Any]) -> dict[str, Any]:
        allowed = None if self.tools is None else expand_tool_names(self.tools, registry)
        denied = expand_tool_names(self.disallowed_tools, registry)
        return {
            name: tool for name, tool in registry.items()
            if (allowed is None or name in allowed) and name not in denied
        }


BUILT_IN_AGENTS = {
    'general-purpose': SubagentDefinition(
        name='general-purpose',
        description='General agent for multi-step tasks; has every tool the parent has.',
    ),
    'Explore': SubagentDefinition(
        name='Explore',
        description='Fast read-only agent for searching and reading code. Cannot edit or run shell.',
        tools=tuple(sorted(READ_ONLY_TOOLS)),
        read_only=True,
        prompt=('You are an Explore subagent. Search and read only, then report what you found '
                'concisely, with file paths and line numbers. You cannot edit files or run commands.'),
    ),
    'Plan': SubagentDefinition(
        name='Plan',
        description='Read-only agent that researches the code and writes an implementation plan.',
        tools=tuple(sorted(READ_ONLY_TOOLS)),
        read_only=True,
        prompt=('You are a Plan subagent. Research with read-only tools, then return a concrete, '
                'step-by-step plan with the files to change. You cannot edit files or run commands.'),
    ),
}


def expand_tool_names(names: tuple[str, ...] | list[str], registry: dict[str, Any]) -> set[str]:
    """Map rule or Sigil tool names (and Bash(...) style rules) to registry names."""
    result: set[str] = set()
    for raw in names:
        name = str(raw).split('(', 1)[0].strip()
        if not name:
            continue
        if name in registry:
            result.add(name)
        for sigil in _RULE_TO_SIGIL.get(name, ()):
            if sigil in registry:
                result.add(sigil)
        if name.endswith('*'):
            result.update(n for n in registry if n.startswith(name[:-1]))
    return result


def _positive_int(value: Any) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None
    return number if number > 0 else None


def _extra_fields(meta: dict) -> dict[str, Any]:
    """Frontmatter keys skills, mcpServers, hooks, memory, effort and maxBudgetUsd."""
    servers = meta.get('mcpServers')
    if isinstance(servers, dict):
        servers = list(servers)
    memory = str(meta.get('memory') or '').strip().lower()
    effort = str(meta.get('effort') or '').strip().lower()
    try:
        budget = float(meta['maxBudgetUsd']) if meta.get('maxBudgetUsd') not in (None, '') else None
    except (TypeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        budget = None
    return {
        'skills': tool_list(meta.get('skills')),
        'mcp_servers': tool_list(servers) if not isinstance(servers, list) else tuple(str(s) for s in servers),
        'hooks': meta.get('hooks') if isinstance(meta.get('hooks'), dict) else None,
        'memory': memory if memory in ('user', 'project', 'local') else None,
        'effort': effort if effort in ('low', 'medium', 'high', 'xhigh', 'max') else None,
        'max_budget_usd': budget if budget and budget > 0 else None,
    }


def agent_memory_dir(cwd: Path, name: str, scope: str) -> Path:
    """memory: user -> ~/.sigil/agent-memory/<name>, project -> .sigil/agent-memory/<name>,
    local -> .sigil/agent-memory-local/<name>."""
    import re as _re

    safe = _re.sub(r'[^A-Za-z0-9_.-]+', '-', name)
    if scope == 'user':
        return _home() / '.sigil' / 'agent-memory' / safe
    folder = 'agent-memory' if scope == 'project' else 'agent-memory-local'
    return Path(cwd) / '.sigil' / folder / safe


def apply_definition_extras(parent: Any, definition: SubagentDefinition, tools: dict[str, Any]) -> dict[str, Any]:
    """What skills, mcpServers, memory and hooks add to a child agent: its tools, extra prompt
    text and hook runtime."""
    from dataclasses import replace as _replace

    parts: list[str] = []
    runtime = getattr(parent, 'skills_runtime', None)
    for name in definition.skills:
        entry = runtime.get_skill(name) if runtime is not None else None
        if entry is not None:
            parts.append(f'# Skill: {name}\n{entry.content}')
    if definition.mcp_servers:
        from .agent_runtime import native_mcp_tool_name

        prefixes = tuple(native_mcp_tool_name(server, 'x')[:-1] for server in definition.mcp_servers)
        tools = {name: tool for name, tool in tools.items()
                 if not name.startswith('mcp__') or name.startswith(prefixes)}
    if definition.memory:
        folder = agent_memory_dir(parent.runtime_config.cwd, definition.name, definition.memory)
        index = folder / 'MEMORY.md'
        try:
            lines = index.read_text(encoding='utf-8').splitlines()[:200]
        except OSError as _ignored:
            _log_ignored(_ignored)
            lines = []
        parts.append(f'# Your memory ({index})\n' + ('\n'.join(lines) or '(empty)')
                     + '\nSave lasting lessons with the memory_save tool.')
        from .agent_tools import SigilTool, ToolExecutionError

        def _save(arguments: dict[str, Any], context: Any, _index: Path = index) -> Any:
            note = ' '.join(str(arguments.get('note', '')).split())[:500]
            if not note:
                raise ToolExecutionError('memory_save: note is empty')
            _index.parent.mkdir(parents=True, exist_ok=True)
            existing = _index.read_text(encoding='utf-8') if _index.exists() else '# Memory\n\n'
            if f'- {note}' not in existing.splitlines():
                _index.write_text(existing + ('' if existing.endswith('\n') else '\n') + f'- {note}\n',
                                  encoding='utf-8')
            return f'Saved to {_index}'

        tools = {**tools, 'memory_save': SigilTool(
            name='memory_save', description='Save a one-line note to your own memory for later runs.',
            parameters={'type': 'object', 'properties': {'note': {'type': 'string'}}, 'required': ['note']},
            handler=_save)}
    hooks = getattr(parent, 'command_hook_runtime', None)
    if definition.hooks:
        from .command_hooks import CommandHookRuntime, _parse_hooks

        errors: list[str] = []
        parsed = _parse_hooks(definition.hooks, f'agent:{definition.name}', errors)
        base = hooks if hooks is not None else CommandHookRuntime(cwd=parent.runtime_config.cwd)
        hooks = _replace(base, hooks=(*base.hooks, *parsed), load_errors=(*base.load_errors, *errors))
    return {'tools': tools, 'prompt': '\n\n'.join(parts), 'hooks': hooks}


def _session_agents() -> dict[str, SubagentDefinition]:
    """--agents '<json>' (SIGIL_AGENTS_JSON): agents defined for this session only."""
    import json
    import os

    raw = os.environ.get('SIGIL_AGENTS_JSON', '').strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return {}
    found: dict[str, SubagentDefinition] = {}
    for name, spec in (data.items() if isinstance(data, dict) else []):
        if not isinstance(spec, dict):
            continue
        found[str(name)] = SubagentDefinition(
            name=str(name),
            description=str(spec.get('description') or name),
            tools=tool_list(spec.get('tools')) or None,
            disallowed_tools=tool_list(spec.get('disallowedTools')),
            model=str(spec['model']) if spec.get('model') else None,
            prompt=str(spec.get('prompt') or ''),
            source='session',
            max_turns=_positive_int(spec.get('maxTurns')),
            permission_mode=str(spec['permissionMode']) if spec.get('permissionMode') else None,
            isolation=str(spec['isolation']).lower() if spec.get('isolation') else None,
            **_extra_fields(spec),
        )
    return found


def _home() -> Path:
    return Path.home()


def _load_agent_dir(folder: Path, source: str) -> dict[str, SubagentDefinition]:
    found: dict[str, SubagentDefinition] = {}
    if not folder.is_dir():
        return found
    for path in sorted(folder.rglob('*.md')):
        try:
            meta, body = parse_frontmatter(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError) as _ignored:
            _log_ignored(_ignored)
            continue
        name = str(meta.get('name') or path.stem).strip()
        description = str(meta.get('description') or '').strip()
        if not name or not description:
            continue
        tools = tool_list(meta.get('tools'))
        found.setdefault(name, SubagentDefinition(
            name=name,
            description=description,
            tools=tools or None,
            disallowed_tools=tool_list(meta.get('disallowedTools') or meta.get('disallowed-tools')),
            model=(str(meta['model']).strip() if meta.get('model') else None),
            prompt=body.strip(),
            source=source,
            path=path,
            max_turns=_positive_int(meta.get('maxTurns')),
            permission_mode=(str(meta['permissionMode']).strip() if meta.get('permissionMode') else None),
            isolation=(str(meta['isolation']).strip().lower() if meta.get('isolation') else None),
            **_extra_fields(meta),
        ))
    return found


def discover_subagents(cwd: Path, skills_runtime: Any = None) -> dict[str, SubagentDefinition]:
    """Built-in types, then project agents, personal agents and skills-repo agents."""
    agents: dict[str, SubagentDefinition] = dict(BUILT_IN_AGENTS)
    agents.update(_session_agents())
    from dataclasses import replace as _replace

    project_trusted: bool | None = None
    for name, definition in _load_agent_dir(Path(cwd) / '.sigil' / 'agents', 'project').items():
        if definition.hooks:
            # frontmatter hooks of a repository's agent run commands: only in a trusted folder
            if project_trusted is None:
                from .workspace_trust import is_trusted

                project_trusted = is_trusted(cwd)
            if not project_trusted:
                definition = _replace(definition, hooks=None)
        agents.setdefault(name, definition)
    for name, definition in _load_agent_dir(_home() / '.sigil' / 'agents', 'user').items():
        agents.setdefault(name, definition)

    from .sigil_plugins import expand_root, loaded_plugins

    for plugin in loaded_plugins():  # --plugin-dir: plugin:name, and name when it is free
        for folder in plugin.agent_dirs:
            for name, definition in _load_agent_dir(folder, f'plugin:{plugin.name}').items():
                # plugin agents ignore permissionMode, mcpServers and hooks
                definition = _replace(definition, prompt=expand_root(definition.prompt, plugin.root),
                                      permission_mode=None, mcp_servers=(), hooks=None)
                agents.setdefault(f'{plugin.name}:{name}', _replace(definition, name=f'{plugin.name}:{name}'))
                agents.setdefault(name, definition)
    if skills_runtime is not None:
        try:
            names = skills_runtime.list_agents()
        except Exception as _ignored:
            _log_ignored(_ignored)
            names = []
        for name in names:
            if name in agents:
                continue
            entry = skills_runtime.get_agent(name)
            if entry is None:
                continue
            metadata = entry.metadata
            tools = tool_list(getattr(metadata, 'tools', None))
            agents[name] = SubagentDefinition(
                name=name,
                description=metadata.description,
                tools=tools or None,
                model=getattr(metadata, 'model', None),
                prompt=entry.content,
                source='skills-repo',
                path=entry.file_path,
            )
    return agents


def find_subagent(name: str, cwd: Path, skills_runtime: Any = None) -> SubagentDefinition | None:
    agents = discover_subagents(cwd, skills_runtime)
    if name in agents:
        return agents[name]
    lowered = name.lower()
    for key, definition in agents.items():
        if key.lower() == lowered:
            return definition
    return None


def render_agent_list(agents: dict[str, SubagentDefinition], limit: int = 30) -> str:
    lines = []
    for name in list(agents)[:limit]:
        lines.append(f'- {name}: {agents[name].description[:120]}')
    if len(agents) > limit:
        lines.append(f'- ... plus {len(agents) - limit} more')
    return '\n'.join(lines)
