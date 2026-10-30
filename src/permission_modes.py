"""Permission modes, cycled with Shift+Tab or set with /mode.

  default      - permissions exactly as the agent was launched (--allow-write etc.)
  acceptEdits  - file edits are allowed without extra flags
  plan         - read-only: research and propose a plan, no edits, no
                 state-changing shell commands, until the user approves
  bypass       - everything allowed, including destructive shell commands
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent_types import AgentPermissions

MODES = ('default', 'acceptEdits', 'plan', 'bypass', 'dontAsk')
CYCLE = ('default', 'acceptEdits', 'plan')  # Shift+Tab order (bypass only via /mode)

MODE_LABELS = {
    'default': 'default',
    'acceptEdits': 'accept edits on',
    'plan': 'plan mode on',
    'bypass': 'bypass permissions on',
    'dontAsk': "don't ask: refuse anything that needs approval",
}

# Tools that change files or the outside world. Blocked in plan mode.
WRITE_TOOLS = frozenset({
    'write_file', 'edit_file', 'notebook_edit', 'create_new_tool', 'rollback_workspace',
    'config_set', 'worktree_enter', 'worktree_exit', 'remote_trigger', 'workflow_run',
    'mcp_call_tool', 'background_agent_start', 'background_agent_kill', 'background_agent_restart', 'send_message',
    'team_create', 'team_delete', 'account_login', 'account_logout', 'remote_connect',
    'remote_disconnect', 'search_activate_provider', 'cron_create',
})

PLAN_MODE_REMINDER = (
    '<system-reminder>\n'
    'Plan mode is active. Do NOT edit files, run state-changing commands, or make any '
    'other changes yet. Research using read-only tools (read_file, grep_search, '
    'glob_search, list_dir, read-only bash, web_search, web_fetch), then present a '
    'concise step-by-step plan and stop so the user can approve it. The user leaves '
    'plan mode with Shift+Tab or /mode default.\n'
    '</system-reminder>'
)


def normalize_mode(name: str) -> str | None:
    key = name.strip().lower().replace('-', '').replace('_', '').replace(' ', '')
    aliases = {
        'default': 'default', 'normal': 'default',
        'acceptedits': 'acceptEdits', 'accept': 'acceptEdits', 'edits': 'acceptEdits', 'auto': 'acceptEdits',
        'plan': 'plan', 'planmode': 'plan',
        'bypass': 'bypass', 'bypasspermissions': 'bypass', 'yolo': 'bypass',
        'dontask': 'dontAsk', 'noask': 'dontAsk',
    }
    return aliases.get(key)


def next_mode(current: str) -> str:
    if current not in CYCLE:
        return CYCLE[0]
    return CYCLE[(CYCLE.index(current) + 1) % len(CYCLE)]


def plan_mode_block_message(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Why a tool call is refused in plan mode, or None if it is allowed."""
    from .plugin_runtime import base_tool_name

    tool_name = base_tool_name(tool_name)  # a plugin alias is checked as the tool it runs
    if tool_name in WRITE_TOOLS:
        return (
            f'Plan mode is active: {tool_name} is not allowed. Finish researching with '
            'read-only tools, present your plan, and wait for the user to approve it.'
        )
    if tool_name in ('bash', 'powershell', 'pwsh'):
        from .bash_security import is_command_read_only, is_powershell_read_only

        command = str(arguments.get('command', ''))
        read_only = is_powershell_read_only if tool_name in ('powershell', 'pwsh') else is_command_read_only
        if not read_only(command):
            return (
                'Plan mode is active: only read-only shell commands are allowed '
                f'(refused: {command[:120]}). Present your plan and wait for approval.'
            )
    if tool_name == 'delegate_agent' and (arguments.get('allow_write') or arguments.get('allow_shell')):
        return 'Plan mode is active: subagents may not get write or shell access.'
    return None


def apply_mode(agent: Any, mode: str) -> None:
    """Switch an agent's permission mode (keeps the launch permissions for 'default')."""
    if mode not in MODES:
        raise ValueError(f'Unknown permission mode: {mode}')
    base = getattr(agent, 'base_permissions', None)
    if base is None:
        base = agent.runtime_config.permissions
        agent.base_permissions = base
    if mode == 'acceptEdits':
        perms = replace(base, allow_file_write=True)
    elif mode == 'bypass':
        perms = AgentPermissions(
            allow_file_write=True,
            allow_shell_commands=True,
            allow_destructive_shell_commands=True,
        )
    else:  # default and plan (plan is enforced separately, per tool call)
        perms = base
    agent.permission_mode = mode
    agent.runtime_config = replace(agent.runtime_config, permissions=perms)
    if getattr(agent, 'tool_context', None) is not None:
        # Bypass mode: shell-guard blocks become warnings (SIGIL_SHELL_GUARD still wins).
        agent.tool_context = replace(
            agent.tool_context, permissions=perms, shell_guard='warn' if mode == 'bypass' else None,
            permission_mode=mode)
