from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .models import PortingBacklog, PortingModule
from .permissions import ToolPermissionContext

TOOL_SOURCE = 'src/agent_tools.py'
SIMPLE_MODE_TOOLS = frozenset({'bash', 'read_file', 'edit_file'})


@dataclass(frozen=True)
class ToolExecution:
    name: str
    source_hint: str
    payload: str
    handled: bool
    message: str


@lru_cache(maxsize=1)
def load_built_in_tools() -> tuple[PortingModule, ...]:
    """Sigil's built-in tools, from agent_tools.default_tool_registry."""
    from .agent_tools import default_tool_registry

    return tuple(
        PortingModule(name=tool.name, responsibility=tool.description, source_hint=TOOL_SOURCE, status='built-in')
        for tool in default_tool_registry().values()
    )


PORTED_TOOLS = load_built_in_tools()


def build_tool_backlog() -> PortingBacklog:
    return PortingBacklog(title='Tool surface', modules=list(PORTED_TOOLS))


def tool_names() -> list[str]:
    return [module.name for module in PORTED_TOOLS]


def get_tool(name: str) -> PortingModule | None:
    needle = name.lower()
    for module in PORTED_TOOLS:
        if module.name.lower() == needle:
            return module
    return None


def filter_tools_by_permission_context(tools: tuple[PortingModule, ...], permission_context: ToolPermissionContext | None = None) -> tuple[PortingModule, ...]:
    if permission_context is None:
        return tools
    return tuple(module for module in tools if not permission_context.blocks(module.name))


def get_tools(
    simple_mode: bool = False,
    include_mcp: bool = True,
    permission_context: ToolPermissionContext | None = None,
) -> tuple[PortingModule, ...]:
    tools = list(PORTED_TOOLS)
    if simple_mode:
        tools = [module for module in tools if module.name in SIMPLE_MODE_TOOLS]
    if not include_mcp:
        tools = [module for module in tools if 'mcp' not in module.name.lower() and 'mcp' not in module.source_hint.lower()]
    return filter_tools_by_permission_context(tuple(tools), permission_context)


def find_tools(query: str, limit: int = 20) -> list[PortingModule]:
    needle = query.lower()
    matches = [module for module in PORTED_TOOLS if needle in module.name.lower() or needle in module.source_hint.lower()]
    return matches[:limit]


def execute_tool(name: str, payload: str = '') -> ToolExecution:
    module = get_tool(name)
    if module is None:
        return ToolExecution(name=name, source_hint='', payload=payload, handled=False, message=f'Unknown tool: {name}')
    action = f"Built-in tool '{module.name}' ({module.source_hint}) would handle payload {payload!r}."
    return ToolExecution(name=module.name, source_hint=module.source_hint, payload=payload, handled=True, message=action)


def render_tool_index(limit: int = 20, query: str | None = None) -> str:
    modules = find_tools(query, limit) if query else list(PORTED_TOOLS[:limit])
    lines = [f'Tool entries: {len(PORTED_TOOLS)}', '']
    if query:
        lines.append(f'Filtered by: {query}')
        lines.append('')
    lines.extend(f'- {module.name} — {module.source_hint}' for module in modules)
    return '\n'.join(lines)
