from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .models import PortingBacklog, PortingModule

COMMAND_SOURCE = 'src/agent_slash_commands.py'


@dataclass(frozen=True)
class CommandExecution:
    name: str
    source_hint: str
    prompt: str
    handled: bool
    message: str


# Commands the chat loop handles itself (src/main.py), not through agent_slash_commands.
CHAT_LOOP_COMMANDS = {
    'compact': 'Compact the conversation now',
    'goal': 'Keep working towards a goal until it is met',
    'loop': 'Run a prompt again on an interval or cron schedule',
    'mode': 'Switch the permission mode',
    'resume': 'Switch to another saved session',
    'rewind': 'Roll back the conversation and/or files to an earlier prompt',
}


@lru_cache(maxsize=1)
def load_built_in_commands() -> tuple[PortingModule, ...]:
    """Sigil's built-in slash commands: agent_slash_commands (first name of each), the built-in
    prompt commands (custom_commands.BUILTIN_COMMANDS) and the chat loop's own commands."""
    from .agent_slash_commands import get_slash_command_specs
    from .custom_commands import BUILTIN_COMMANDS

    modules: dict[str, PortingModule] = {}
    for spec in get_slash_command_specs():
        if spec.names:
            modules.setdefault(spec.names[0], PortingModule(
                name=spec.names[0], responsibility=spec.description, source_hint=COMMAND_SOURCE, status='built-in'))
    for name, command in BUILTIN_COMMANDS.items():
        modules.setdefault(name, PortingModule(
            name=name, responsibility=command.description, source_hint='src/custom_commands.py', status='built-in'))
    for name, description in CHAT_LOOP_COMMANDS.items():
        modules.setdefault(name, PortingModule(
            name=name, responsibility=description, source_hint='src/main.py', status='built-in'))
    return tuple(modules.values())


PORTED_COMMANDS = load_built_in_commands()


@lru_cache(maxsize=1)
def built_in_command_names() -> frozenset[str]:
    return frozenset(module.name for module in PORTED_COMMANDS)


def build_command_backlog() -> PortingBacklog:
    return PortingBacklog(title='Command surface', modules=list(PORTED_COMMANDS))


def command_names() -> list[str]:
    return [module.name for module in PORTED_COMMANDS]


def get_command(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in PORTED_COMMANDS:
        if module.name.lower() == needle:
            return module
    return None


def get_commands(cwd: str | None = None, include_plugin_commands: bool = True, include_skill_commands: bool = True) -> tuple[PortingModule, ...]:
    commands = list(PORTED_COMMANDS)
    if not include_plugin_commands:
        commands = [module for module in commands if 'plugin' not in module.source_hint.lower()]
    if not include_skill_commands:
        commands = [module for module in commands if 'skills' not in module.source_hint.lower()]
    return tuple(commands)


def find_commands(query: str, limit: int = 20) -> list[PortingModule]:
    needle = query.lower()
    matches = [module for module in PORTED_COMMANDS if needle in module.name.lower() or needle in module.source_hint.lower()]
    return matches[:limit]


def execute_command(name: str, prompt: str = '') -> CommandExecution:
    module = get_command(name)
    if module is None:
        return CommandExecution(name=name, source_hint='', prompt=prompt, handled=False, message=f'Unknown command: {name}')
    action = f"Built-in command '/{module.name}' ({module.source_hint}) would handle prompt {prompt!r}."
    return CommandExecution(name=module.name, source_hint=module.source_hint, prompt=prompt, handled=True, message=action)


def render_command_index(limit: int = 20, query: str | None = None) -> str:
    modules = find_commands(query, limit) if query else list(PORTED_COMMANDS[:limit])
    lines = [f'Command entries: {len(PORTED_COMMANDS)}', '']
    if query:
        lines.append(f'Filtered by: {query}')
        lines.append('')
    lines.extend(f'- {module.name} — {module.source_hint}' for module in modules)
    return '\n'.join(lines)
