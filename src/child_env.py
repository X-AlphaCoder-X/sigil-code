"""Environment for child processes: what happens to the SSH agent connection.

SSH_AUTH_SOCK points at the ssh-agent socket; any process of the same user that can reach it can use
the loaded keys (ssh-agent(1): the socket "is easily abused by root or another instance of the same
user"). So:

- MCP servers (third-party programs) do not get SSH_AUTH_SOCK / SSH_AGENT_PID by default.
  SIGIL_MCP_SSH_AGENT=1 passes them; a server's own `env` block in its MCP config can still set them.
- bash, background shells, hooks, the status line, subagents and background agents keep them, so git
  over SSH works in bash.
- SIGIL_SSH_AGENT=off removes them everywhere.
"""
from __future__ import annotations

import os
from typing import Mapping

SSH_AGENT_VARS = ('SSH_AUTH_SOCK', 'SSH_AGENT_PID')
_OFF = ('0', 'off', 'false', 'no')
_ON = ('1', 'on', 'true', 'yes')


def ssh_agent_off() -> bool:
    return os.environ.get('SIGIL_SSH_AGENT', '').strip().lower() in _OFF


def mcp_gets_ssh_agent() -> bool:
    return not ssh_agent_off() and os.environ.get('SIGIL_MCP_SSH_AGENT', '').strip().lower() in _ON


def is_ssh_agent_var(name: str) -> bool:
    return name.upper() in SSH_AGENT_VARS


def strip_ssh_agent(env: dict[str, str], *, mcp: bool = False) -> dict[str, str]:
    """Remove the SSH agent variables from `env` (in place) when the switches say so."""
    if ssh_agent_off() or (mcp and not mcp_gets_ssh_agent()):
        for name in [key for key in env if is_ssh_agent_var(key)]:
            env.pop(name, None)
    return env


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of `base` (default: this process's environment) for a non-MCP child process."""
    return strip_ssh_agent(dict(os.environ if base is None else base))
