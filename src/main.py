from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import replace
import json
from typing import Callable, Any
try:
    from prompt_toolkit import PromptSession, ANSI
    from prompt_toolkit.patch_stdout import patch_stdout
    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False
    PromptSession = None
    ANSI = None
    patch_stdout = None


from .background_runtime import BackgroundSessionRuntime, build_background_worker_command
from .agent_bar import AgentBar, build_prompt_kwargs as build_agent_bar_kwargs
from .account_runtime import AccountRuntime
from .ask_user_runtime import AskUserRuntime
from .agent_runtime import LocalCodingAgent
from .message_queue import ConsoleInputReader, MessageQueue

# Simple .env loader for systems without python-dotenv
def _load_env_file():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

_load_env_file()

from .token_router import TokenRouter, limits_from_env
TokenRouter.configure(*limits_from_env())  # off unless SIGIL_RPM_LIMIT / SIGIL_TPM_LIMIT are set

from .agent_types import (
    AgentPermissions,
    AgentRuntimeConfig,
    BudgetConfig,
    ModelConfig,
    ModelPricing,
    OutputSchemaConfig,
)
from .bootstrap_graph import build_bootstrap_graph
from .command_graph import build_command_graph
from .commands import execute_command, get_command, get_commands, render_command_index
from .config_runtime import ConfigRuntime
from .lsp_runtime import LSPRuntime
from .mcp_runtime import MCPRuntime
from .parity_audit import run_parity_audit
from .permissions import ToolPermissionContext
from .port_manifest import build_port_manifest
from .query_engine import QueryEnginePort
from .remote_runtime import (
    RemoteRuntime,
    run_deep_link_mode,
    run_direct_connect_mode,
    run_remote_mode,
    run_ssh_mode,
    run_teleport_mode,
)
from .remote_trigger_runtime import RemoteTriggerRuntime
from .search_runtime import SearchRuntime
from .team_runtime import TeamRuntime
from .workflow_runtime import WorkflowRuntime
from .worktree_runtime import WorktreeRuntime
from .runtime import PortRuntime
from .session_store import (
    StoredAgentSession,
    deserialize_model_config,
    deserialize_runtime_config,
    load_agent_session,
    load_session,
)
from .setup import run_setup
from .tool_pool import assemble_tool_pool
from .tools import execute_tool, get_tool, get_tools, render_tool_index
from .tui_display import SigilTUIDisplay
from .settings import write_text_atomic  # replaces a file in one step: readers never see half of it
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


def _session_root_for(cwd: str | Path) -> Path:
    """Where a workspace keeps sessions and scratchpads.

    SIGIL_SESSION_STORE=project (default): <cwd>/.port_sessions (with a .gitignore so
    git leaves it alone). SIGIL_SESSION_STORE=home: ~/.sigil/projects/<path-hash>,
    outside the repo; existing project sessions are copied there once.
    """
    import hashlib
    import shutil

    workspace = Path(cwd).resolve()
    project_root = workspace / '.port_sessions'
    if os.environ.get('SIGIL_SESSION_STORE', 'project').strip().lower() != 'home':
        _ensure_gitignore(project_root)
        return project_root
    from .settings import _home

    digest = hashlib.sha256(str(workspace).lower().encode('utf-8')).hexdigest()[:16]
    home_root = _home() / '.sigil' / 'projects' / f'{workspace.name}-{digest}'
    old_sessions = project_root / 'agent'
    if old_sessions.is_dir() and not (home_root / 'agent').exists():
        try:
            shutil.copytree(old_sessions, home_root / 'agent')  # one-time copy; the old folder stays
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass
    return home_root


def _ensure_gitignore(folder: Path) -> None:
    try:
        if folder.is_dir() and not (folder / '.gitignore').exists():
            (folder / '.gitignore').write_text('# Sigil Code session data\n*\n', encoding='utf-8')
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass


def _background_runtime_for_args(args: argparse.Namespace) -> 'BackgroundSessionRuntime':
    """Background sessions of --cwd (<cwd>/.port_sessions/background, where agent-bg, the agent
    tool and the chat put them); without --cwd, the process folder."""
    cwd = getattr(args, 'background_cwd', None)  # only the --cwd of agent-ps/logs/attach/kill/approve/restart
    if cwd:
        return BackgroundSessionRuntime(Path(cwd).expanduser().resolve() / '.port_sessions' / 'background')
    return BackgroundSessionRuntime()


def _background_dir_for(agent: Any) -> Path:
    """Where the agent tool starts background agents (agent_tools: <cwd>/.port_sessions/background),
    read at call time so /cd and --cwd are followed. The relative default is taken from the
    agent's folder, not the process folder; an absolute DEFAULT_BACKGROUND_DIR is used as is."""
    from .background_runtime import DEFAULT_BACKGROUND_DIR

    root = Path(DEFAULT_BACKGROUND_DIR)
    if root.is_absolute():
        return root
    return Path(agent.runtime_config.cwd).resolve() / root


def _session_directory_for(cwd: str | Path) -> Path:
    """Sessions live under the workspace given by --cwd, not the process folder."""
    return _session_root_for(cwd) / 'agent'


def _lookup_latest_session_id(
    cwd: str,
    is_backup: bool = False,
    session_dir: Path | None = None,
) -> str:
    workspace = Path(cwd).resolve()

    if is_backup:
        # the session folder first; <cwd>/session_backup.json is where older versions wrote it
        for backup_path in ((session_dir or _session_directory_for(workspace)).parent / 'session_backup.json',
                            workspace / 'session_backup.json'):
            if backup_path.exists():
                with open(backup_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data['session_id']

    session_dir = session_dir or _session_directory_for(workspace)
    if not session_dir.exists():
        raise OSError(f"No session directory found at {session_dir}")
    
    # subagent (<id>--sub--N) and background (bg_*) sessions are not what -c / -r mean
    sessions = [p for p in session_dir.glob('*.json')
                if '--sub--' not in p.stem and not p.stem.startswith('bg_') and p.stem != 'session_backup']
    if not sessions:
        raise OSError(f"No saved sessions found in {session_dir}")
    
    # Sort by modification time
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0].stem


def _add_agent_common_args(parser: argparse.ArgumentParser, *, include_backend: bool) -> None:
    parser.add_argument('--model', default=os.environ.get('OPENAI_MODEL', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'))
    if include_backend:
        parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))
        parser.add_argument('--api-key', default=None,
                            help='key for the model server (else SIGIL_API_KEY_<HOST>, SIGIL_API_KEY, OPENAI_API_KEY)')
        # Unset by default: Claude 4.7+ rejects temperature values below 1, and
        # most servers already pick a sensible default when it is left out.
        parser.add_argument('--temperature', type=float, default=None)
        parser.add_argument('--timeout-seconds', type=float, default=120.0)
        parser.add_argument('--input-cost-per-million', type=float, default=0.0)
        parser.add_argument('--output-cost-per-million', type=float, default=0.0)
    parser.add_argument('--cwd', default='.')
    parser.add_argument('--add-dir', action='append', default=[])
    parser.add_argument('--image', action='append', default=[],
                        help='attach an image (png, jpg, gif, webp) to the first prompt; repeatable')
    parser.add_argument('--settings', dest='settings_override', default=None,
                        help='extra settings: a JSON string or a file path (applied last)')
    parser.add_argument('--setting-sources', default=None,
                        help='which settings files to read: comma list of user, project, local')
    parser.add_argument('--mcp-config', action='append', default=[],
                        help='extra MCP config file (repeatable)')
    parser.add_argument('--strict-mcp-config', action='store_true', default=False,
                        help='use only the --mcp-config files')
    parser.add_argument('--plugin-dir', action='append', default=[],
                        help='load a plugin folder (repeatable)')
    parser.add_argument('--session-id', default=None, help='use this id for a new session')
    parser.add_argument('--plain', action='store_true', default=False,
                        help='plain output for screen readers and logs: no colours, boxes or animation')
    parser.add_argument('--fork-session', action='store_true', default=False,
                        help='when resuming (-c, -r, --resume-session-id), continue in a copy with a new id')
    parser.add_argument('--no-session-persistence', action='store_true', default=False,
                        help='do not keep the session on disk (it lives in a temp folder until exit)')
    parser.add_argument('--system-prompt-file', default=None, help='read --system-prompt from a file')
    parser.add_argument('--append-system-prompt-file', default=None,
                        help='read --append-system-prompt from a file')
    parser.add_argument('--disable-slash-commands', action='store_true', default=False,
                        help='treat /text as a plain prompt: no slash commands, custom commands or skills')
    parser.add_argument('--include-partial-messages', action='store_true', default=False,
                        help='stream-json output: include partial text and tool deltas')
    parser.add_argument('--replay-user-messages', action='store_true', default=False,
                        help='stream-json input and output: echo each user message back')
    parser.add_argument('--permission-prompt-tool', default=None,
                        help='MCP tool (mcp__server__tool) that answers approval questions in print mode')
    parser.add_argument('--init-only', action='store_true', default=False,
                        help='run the Setup hooks and exit')
    parser.add_argument('--exclude-dynamic-system-prompt-sections', dest='exclude_dynamic_sections',
                        action='store_true', default=False,
                        help='move the environment and session guidance into the first user message')
    parser.add_argument('--architect', action='store_true', default=False,
                        help='architect mode: this model plans, the editor model makes the edits')
    parser.add_argument('--editor-model', default=None,
                        help='model for edits in architect mode: "model" or "model@base_url"')
    parser.add_argument('--safe-mode', '--bare', dest='safe_mode', action='store_true', default=False,
                        help='start without hooks, skills, plugins, MCP servers or SIGIL.md')
    parser.add_argument('--agent', dest='session_agent', default=None,
                        help='run the whole session as this agent (built-in or .sigil/agents)')
    parser.add_argument('--agents', dest='agents_json', default=None,
                        help='JSON object defining agents for this session')
    parser.add_argument('-w', '--worktree', default=None, metavar='NAME',
                        help='work in a git worktree with this name (created if needed)')
    parser.add_argument('-n', '--name', dest='session_name', default=None,
                        help='name this session; resume it later with --resume-session-id NAME')
    parser.add_argument('--max-tokens', type=int, default=None, help='max_tokens for each model request')
    parser.add_argument('--profile', default=None, help='provider profile from settings (profiles)')
    parser.add_argument('--fallback-model', default=None,
                        help='comma list of models (or model@base_url) to try when the main one fails')
    parser.add_argument('--effort', choices=['off', 'none', 'low', 'medium', 'high', 'xhigh', 'max', 'auto'],
                        default=None,
                        help='reasoning effort, sent in the field this provider reads')
    parser.add_argument('--disable-sigil-md', action='store_true')
    parser.add_argument('--allow-write', action='store_true')
    parser.add_argument('--allow-shell', action='store_true')
    parser.add_argument('--unsafe', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--auto-snip-threshold', type=int)
    parser.add_argument('--auto-compact-threshold', type=int)
    parser.add_argument('--compact-preserve-messages', type=int, default=4)
    parser.add_argument('--max-total-tokens', type=int)
    parser.add_argument('--max-input-tokens', type=int)
    parser.add_argument('--max-output-tokens', type=int)
    parser.add_argument('--max-reasoning-tokens', type=int)
    parser.add_argument('--max-budget-usd', type=float)
    parser.add_argument('--max-tool-calls', type=int)
    parser.add_argument('--max-delegated-tasks', type=int)
    parser.add_argument('--max-model-calls', type=int)
    parser.add_argument('--max-session-turns', type=int)
    parser.add_argument('--response-schema-file')
    parser.add_argument('--response-schema-name')
    parser.add_argument('--response-schema-strict', action='store_true')
    parser.add_argument('--scratchpad-root')
    parser.add_argument('--nudge-strategy', choices=['off', 'structural', 'heuristic'], default=None)
    parser.add_argument('--permission-prompts', choices=['allow', 'deny'], default=None,
                        help='with no terminal to ask in, allow or deny calls that need approval')
    parser.add_argument('--permission-mode', default=None,
                        help='start in default, acceptEdits, plan or bypass (or permissions.defaultMode)')
    parser.add_argument('--allowed-tools', action='append', default=[],
                        help='allow rules, e.g. "Bash(git *) Read" (no approval needed)')
    parser.add_argument('--disallowed-tools', action='append', default=[],
                        help='deny rules; a bare tool name also hides that tool')
    parser.add_argument('--tools', default=None, help='only these tools, e.g. "Read,Grep,Glob"')
    parser.add_argument('--debug', action='store_true', default=None,
                        help='also log raw model requests and responses (see the log file in /status)')
    parser.add_argument('--debug-file', default=None, help='write the log to this file')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print mode: show each tool call and result on stderr; also logs at debug level')
    parser.add_argument('--system-prompt')
    parser.add_argument('--append-system-prompt')
    parser.add_argument('--override-system-prompt')


def _build_runtime_config(args: argparse.Namespace) -> AgentRuntimeConfig:
    cwd = Path(args.cwd).resolve()
    config_runtime = ConfigRuntime.from_workspace(cwd)
    
    # Precedence: CLI argument > Config file > Default
    nudge_strategy = getattr(args, 'nudge_strategy', None)
    if nudge_strategy is None:
        try:
            nudge_strategy = config_runtime.get_value('nudge_strategy')
        except KeyError as _ignored:
            _log_ignored(_ignored)
            nudge_strategy = 'structural'

    return AgentRuntimeConfig(
        cwd=cwd,
        max_turns=getattr(args, 'max_turns', 12),
        permissions=AgentPermissions(
            allow_file_write=args.allow_write,
            allow_shell_commands=args.allow_shell,
            allow_destructive_shell_commands=args.unsafe,
        ),
        stream_model_responses=bool(getattr(args, 'stream', False)),
        auto_snip_threshold_tokens=getattr(args, 'auto_snip_threshold', None),
        auto_compact_threshold_tokens=getattr(args, 'auto_compact_threshold', None),
        compact_preserve_messages=max(0, int(getattr(args, 'compact_preserve_messages', 4))),
        additional_working_directories=_added_directories(cwd, args.add_dir),
        disable_sigil_md_discovery=args.disable_sigil_md or bool(getattr(args, 'safe_mode', False)),
        budget_config=BudgetConfig(
            max_total_tokens=getattr(args, 'max_total_tokens', None),
            max_input_tokens=getattr(args, 'max_input_tokens', None),
            max_output_tokens=getattr(args, 'max_output_tokens', None),
            max_reasoning_tokens=getattr(args, 'max_reasoning_tokens', None),
            max_total_cost_usd=getattr(args, 'max_budget_usd', None),
            max_tool_call_count=getattr(args, 'max_tool_calls', None),
            max_delegated_tasks=getattr(args, 'max_delegated_tasks', None),
            max_model_calls=getattr(args, 'max_model_calls', None),
            max_session_turns=getattr(args, 'max_session_turns', None),
        ),
        output_schema=_load_output_schema_config(args),
        session_directory=_session_directory_for(cwd),
        scratchpad_root=(
            Path(getattr(args, 'scratchpad_root')).resolve()
            if getattr(args, 'scratchpad_root', None)
            else _session_root_for(cwd) / 'scratchpad'
        ),
        nudge_strategy=nudge_strategy,
        permission_prompts=getattr(args, 'permission_prompts', None),
        cli_allowed_tools=_rule_args(getattr(args, 'allowed_tools', None)),
        cli_disallowed_tools=_rule_args(getattr(args, 'disallowed_tools', None)),
        cli_tools=(_rule_args([args.tools]) if getattr(args, 'tools', None) is not None else None),
    )


def _added_directories(cwd: Path, flags: list[str]) -> tuple[Path, ...]:
    """--add-dir folders plus permissions.additionalDirectories from settings."""
    folders = [Path(path).resolve() for path in flags]
    try:
        from .settings import load_settings

        extra = (load_settings(cwd).get('permissions') or {}).get('additionalDirectories')
    except Exception as _ignored:
        _log_ignored(_ignored)
        extra = None
    for raw in extra if isinstance(extra, list) else []:
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        candidate = candidate.resolve()
        if candidate not in folders:
            folders.append(candidate)
    return tuple(folders)


def _rule_args(values: Any) -> tuple[str, ...]:
    """Rule flags may repeat and hold several rules (commas or spaces)."""
    from .custom_commands import tool_list

    rules: list[str] = []
    for value in values or ():
        rules.extend(tool_list(value))
    return tuple(rules)


def _start_permission_mode(agent: Any, args: argparse.Namespace) -> None:
    """--permission-mode, else permissions.defaultMode from settings.json."""
    from .permission_modes import apply_mode, normalize_mode

    raw = getattr(args, 'permission_mode', None)
    if not raw:
        try:
            from .settings import load_settings

            raw = (load_settings(agent.runtime_config.cwd).get('permissions') or {}).get('defaultMode')
        except Exception as _ignored:
            _log_ignored(_ignored)
            raw = None
    if not raw:
        return
    mode = normalize_mode(str(raw))
    if mode is None:
        raise SystemExit(f'unknown permission mode: {raw}')
    apply_mode(agent, mode)


def _forward_key(env: dict[str, str], base_url: str, key: str) -> None:
    """Give a worker process the key through its environment (never argv): under the server's own
    SIGIL_API_KEY_<HOST>, which comes first in resolve_api_key, so an inherited SIGIL_API_KEY does
    not win, and as OPENAI_API_KEY for tools that read only that."""
    from .profiles import host_key_env

    env['OPENAI_API_KEY'] = key
    if base_url:
        env[host_key_env(base_url)] = key


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    from .profiles import resolve_api_key

    base_url = getattr(args, 'base_url', os.environ.get('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1'))
    return ModelConfig(
        model=args.model,
        base_url=base_url,
        api_key=resolve_api_key(base_url, getattr(args, 'api_key', None)),
        temperature=getattr(args, 'temperature', None),
        timeout_seconds=getattr(args, 'timeout_seconds', 120.0),
        effort=(None if getattr(args, 'effort', None) in (None, 'auto') else args.effort),
        fallback_models=tuple(
            item.strip() for item in (getattr(args, 'fallback_model', None) or '').split(',') if item.strip()),
        max_tokens=getattr(args, 'max_tokens', None),
        pricing=ModelPricing(
            input_cost_per_million_tokens_usd=float(
                getattr(args, 'input_cost_per_million', 0.0) or 0.0
            ),
            output_cost_per_million_tokens_usd=float(
                getattr(args, 'output_cost_per_million', 0.0) or 0.0
            ),
        ),
    )


def _load_output_schema_config(args: argparse.Namespace) -> OutputSchemaConfig | None:
    schema_file = getattr(args, 'response_schema_file', None)
    if not schema_file:
        return None
    payload = json.loads(Path(schema_file).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('response schema file must contain a top-level JSON object')
    name = getattr(args, 'response_schema_name', None) or Path(schema_file).stem
    return OutputSchemaConfig(
        name=name,
        schema=payload,
        strict=bool(getattr(args, 'response_schema_strict', False)),
    )


def _apply_flag_environment(args: argparse.Namespace) -> None:
    """Flags that every part of the program (and child agents) must see go to the env."""
    if getattr(args, 'settings_override', None):
        os.environ['SIGIL_SETTINGS'] = args.settings_override
    if getattr(args, 'setting_sources', None):
        os.environ['SIGIL_SETTING_SOURCES'] = args.setting_sources
    if getattr(args, 'mcp_config', None):
        os.environ['SIGIL_MCP_CONFIG'] = os.pathsep.join(str(Path(p).resolve()) for p in args.mcp_config)
    if getattr(args, 'strict_mcp_config', False):
        os.environ['SIGIL_STRICT_MCP_CONFIG'] = '1'
    if getattr(args, 'exclude_dynamic_sections', False):
        os.environ['SIGIL_EXCLUDE_DYNAMIC_SECTIONS'] = '1'
    if getattr(args, 'disable_slash_commands', False):
        os.environ['SIGIL_DISABLE_SLASH_COMMANDS'] = '1'
    if getattr(args, 'plain', False):
        os.environ['SIGIL_PLAIN'] = '1'
    for name, target in (('system_prompt_file', 'system_prompt'), ('append_system_prompt_file', 'append_system_prompt')):
        value = getattr(args, name, None)
        if value:
            try:
                setattr(args, target, Path(value).expanduser().read_text(encoding='utf-8'))
            except OSError as exc:
                raise SystemExit(f'--{name.replace("_", "-")}: cannot read {value}: {exc}')
    if getattr(args, 'architect', False):
        os.environ['SIGIL_ARCHITECT'] = '1'
    if getattr(args, 'editor_model', None):
        os.environ['SIGIL_EDITOR_MODEL'] = args.editor_model
    if getattr(args, 'plugin_dir', None):
        current = [p for p in os.environ.get('SIGIL_PLUGIN_DIRS', '').split(os.pathsep) if p.strip()]
        added = [str(Path(p).expanduser().resolve()) for p in args.plugin_dir]
        os.environ['SIGIL_PLUGIN_DIRS'] = os.pathsep.join(dict.fromkeys(current + added))
    if getattr(args, 'safe_mode', False):
        os.environ['SIGIL_SAFE_MODE'] = '1'
    if getattr(args, 'agents_json', None):
        os.environ['SIGIL_AGENTS_JSON'] = args.agents_json


def _apply_session_agent(agent: Any, name: str | None, *, user_mode: bool = False) -> None:
    """--agent NAME: the whole session uses that agent's prompt, tools, model and mode.

    The mode follows the rule for a project settings file (S15): a repository's .sigil/agents file
    never starts bypass mode, and acceptEdits from it needs a trusted folder; the user's own
    --permission-mode always wins."""
    if not name:
        return
    from .permission_modes import apply_mode, normalize_mode
    from .providers import resolve_model_alias
    from .subagents import discover_subagents, find_subagent

    definition = find_subagent(name, agent.runtime_config.cwd, agent.skills_runtime)
    if definition is None:
        raise SystemExit(f'unknown agent {name!r}; known: {", ".join(discover_subagents(agent.runtime_config.cwd))}')
    agent.tool_registry = definition.filter_tools(agent.tool_registry)
    if definition.prompt:
        agent.append_system_prompt = '\n\n'.join(p for p in (agent.append_system_prompt, definition.prompt) if p)
    model = resolve_model_alias(definition.model)
    if model:
        agent.set_model(model)
    if definition.read_only:
        apply_mode(agent, 'plan')
    elif definition.permission_mode and not user_mode:
        mode = normalize_mode(definition.permission_mode)
        if mode and definition.source == 'project' and mode not in ('default', 'plan', 'dontAsk'):
            from .workspace_trust import is_trusted

            if mode == 'bypass' or not is_trusted(agent.runtime_config.cwd):
                why = ('a project agent file cannot start bypass mode' if mode == 'bypass'
                       else 'the folder is not trusted (sigil trust grant)')
                print(f'agent {name}: permissionMode {definition.permission_mode} ignored: {why}; '
                      'use --permission-mode to choose the mode yourself.', file=sys.stderr)
                mode = None
        if mode:
            apply_mode(agent, mode)


def _warn_schema_on_compat(agent: Any) -> None:
    """Anthropic's OpenAI-compatible endpoint ignores response_format."""
    from .anthropic_client import use_native_anthropic
    from .providers import detect_provider

    config = agent.model_config
    if agent.runtime_config.output_schema is not None and \
            detect_provider(config.base_url, config.model) == 'anthropic' and not use_native_anthropic(config):
        sys.stderr.write('warning: --response-schema-file is ignored by the Anthropic OpenAI-compatible endpoint; '
                         'set SIGIL_ANTHROPIC_NATIVE=1 to send the schema as an instruction.\n')


def _mcp_config_file(scope: str, cwd: str) -> Path:
    from .mcp_runtime import user_mcp_config_path

    return user_mcp_config_path() if scope == 'user' else Path(cwd).resolve() / '.mcp.json'


def _read_mcp_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get('mcpServers'), dict):
        data['mcpServers'] = {}
    return data


def _run_mcp_command(args: argparse.Namespace) -> int:
    """sigil mcp add|list|remove|get (project: .mcp.json, user: ~/.sigil/mcp.json)."""
    scopes = [args.scope] if getattr(args, 'scope', None) else ['project', 'user']
    if args.mcp_command in ('login', 'logout'):
        from .mcp_oauth import mcp_auth_command
        from .mcp_runtime import MCPRuntime

        message = mcp_auth_command(MCPRuntime.from_workspace(Path(args.cwd).resolve()), args.mcp_command, args.name)
        print(message)
        return 0 if message.startswith(('Logged in', 'Logged out')) else 1
    if args.mcp_command == 'add':
        path = _mcp_config_file(args.scope, args.cwd)
        data = _read_mcp_file(path)
        transport = args.transport or ('http' if args.url else 'stdio')
        if transport == 'stdio':
            if not args.server_command:
                print('mcp add: give a command (or --url for http/sse servers)')
                return 2
            entry: dict[str, Any] = {'command': args.server_command, 'args': list(args.server_args)}
        else:
            if not args.url:
                print(f'mcp add: --url is needed for {transport} servers')
                return 2
            entry = {'type': transport, 'url': args.url}
            if args.header:
                entry['headers'] = dict(item.split(':', 1) for item in args.header if ':' in item)
                entry['headers'] = {k.strip(): v.strip() for k, v in entry['headers'].items()}
        if args.env:
            entry['env'] = dict(item.split('=', 1) for item in args.env if '=' in item)
        data['mcpServers'][args.name] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(data, indent=2) + '\n', encoding='utf-8')
        if args.scope != 'user':
            from .workspace_trust import mark_user_item

            # the user added it: it runs without the folder's trust question
            mark_user_item(args.cwd, 'mcp', args.name,
                           entry.get('url') or ' '.join([str(entry.get('command') or ''),
                                                         *(str(a) for a in entry.get('args', []))]).strip())
        print(f'Added MCP server {args.name} to {path}')
        return 0
    found = False
    for scope in scopes:
        path = _mcp_config_file(scope, args.cwd)
        data = _read_mcp_file(path)
        servers = data['mcpServers']
        if args.mcp_command == 'list':
            for name, entry in servers.items():
                target = entry.get('url') or ' '.join([str(entry.get('command', ''))] + [str(a) for a in entry.get('args', [])])
                print(f'{name} ({scope}): {target}')
                found = True
        elif args.name in servers:
            found = True
            if args.mcp_command == 'get':
                print(json.dumps({args.name: servers[args.name], 'scope': scope, 'file': str(path)}, indent=2))
                return 0
            del servers[args.name]
            write_text_atomic(path, json.dumps(data, indent=2) + '\n', encoding='utf-8')
            print(f'Removed MCP server {args.name} from {path}')
            return 0
    if not found:
        print('No MCP servers configured.' if args.mcp_command == 'list' else f'No MCP server named {args.name}')
        return 0 if args.mcp_command == 'list' else 1
    return 0


def _build_agent(args: argparse.Namespace) -> LocalCodingAgent:
    _apply_flag_environment(args)
    _enter_worktree(args)
    agent = LocalCodingAgent(
        model_config=_build_model_config(args),
        runtime_config=_build_runtime_config(args),
        custom_system_prompt=args.system_prompt,
        append_system_prompt=args.append_system_prompt,
        override_system_prompt=args.override_system_prompt,
    )
    _start_permission_mode(agent, args)
    if getattr(args, 'profile', None):
        from .profiles import apply_profile

        try:
            apply_profile(agent, args.profile)
        except KeyError as exc:
            raise SystemExit(str(exc.args[0]))
    _apply_session_agent(agent, getattr(args, 'session_agent', None),
                         user_mode=bool(getattr(args, 'permission_mode', None)))
    _warn_schema_on_compat(agent)
    _apply_session_flags(agent, args)
    _auto_cleanup_sessions(agent)
    try:
        from .model_info import apply_detected

        # Context window and vision from the server (SIGIL_DETECT_MODEL=0 turns it off).
        apply_detected(agent, threshold_given=getattr(args, 'auto_compact_threshold', None) is not None)
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    if getattr(args, 'session_name', None):
        agent._session_name = args.session_name
    problems = agent.attach_images(getattr(args, 'image', None) or [])
    for problem in problems:
        sys.stderr.write(f'image not attached: {problem}\n')
    return agent


def _apply_session_flags(agent: LocalCodingAgent, args: argparse.Namespace) -> None:
    """--session-id, --fork-session, --no-session-persistence, --permission-prompt-tool."""
    if getattr(args, 'no_session_persistence', False):
        import atexit
        import shutil
        import tempfile

        folder = Path(tempfile.mkdtemp(prefix='sigil-nosave-'))
        atexit.register(shutil.rmtree, folder, True)
        agent.runtime_config = replace(agent.runtime_config, session_directory=folder)
    requested = getattr(args, 'session_id', None)
    if requested:
        from .session_store import _validate_session_id

        try:
            _validate_session_id(requested)
        except ValueError as exc:
            raise SystemExit(f'--session-id: {exc}')
        if (agent.runtime_config.session_directory / f'{requested}.json').exists():
            raise SystemExit(f'--session-id: session {requested} already exists (use --resume-session-id)')
        agent.requested_session_id = requested
    agent.fork_session = bool(getattr(args, 'fork_session', False))
    tool_name = getattr(args, 'permission_prompt_tool', None)
    if tool_name:
        agent.permission_prompt = _permission_tool_prompt(agent, tool_name)


def _permission_tool_prompt(agent: LocalCodingAgent, tool_name: str) -> Callable[[dict], str]:
    """--permission-prompt-tool: an MCP tool decides. It gets {"tool_name", "input"} and answers JSON
    {"behavior": "allow" | "deny", "message": ...}; anything else counts as deny."""
    def _ask(request: dict) -> str:
        tool = agent.tool_registry.get(tool_name)
        if tool is None:
            sys.stderr.write(f'[permission] {tool_name} is not an available tool; denying\n')
            return 'deny'
        try:
            reply = tool.handler({'tool_name': request.get('tool'), 'input': request.get('arguments') or {}},
                                 agent.tool_context)
            text = reply[0] if isinstance(reply, tuple) else str(reply)
            decision = json.loads(text)
        except Exception as exc:
            sys.stderr.write(f'[permission] {tool_name} failed: {exc}; denying\n')
            return 'deny'
        return 'once' if isinstance(decision, dict) and decision.get('behavior') == 'allow' else 'deny'

    return _ask


def _auto_cleanup_sessions(agent: Any) -> None:
    """cleanupPeriodDays: delete this workspace's sessions older than that many days."""
    from .session_store import cleanup_period_days, purge_agent_sessions

    days = cleanup_period_days(agent.runtime_config.cwd)
    if days is None:
        return
    try:
        removed = purge_agent_sessions(agent.runtime_config.session_directory, older_than_days=days)
    except OSError as _ignored:
        _log_ignored(_ignored)
        return
    if removed:
        from .debug_log import get_logger

        get_logger('sessions').info('cleanupPeriodDays=%s: removed %d old session(s)', days, len(removed))


def _run_purge(args: argparse.Namespace) -> int:
    """sigil purge: delete old sessions of a workspace (--days N, default cleanupPeriodDays or 30; --all)."""
    from .session_store import cleanup_period_days, purge_agent_sessions

    cwd = Path(args.cwd).resolve()
    days = None if args.all else (args.days if args.days is not None else cleanup_period_days(cwd) or 30.0)
    removed = purge_agent_sessions(_session_directory_for(cwd), older_than_days=days, dry_run=args.dry_run)
    what = 'all sessions' if days is None else f'sessions older than {days:g} day(s)'
    verb = 'Would delete' if args.dry_run else 'Deleted'
    print(f'{verb} {len(removed)} of the {what}' + (':' if removed else '.'))
    for session_id in removed:
        print(f'  {session_id}')
    return 0


def _bg_mark(session_id: str) -> str:
    return '  [bg]' if session_id.startswith('bg_') else ''


def _print_turn_done(output_func: Callable[[str], None], elapsed: float) -> None:
    """After a turn of at least SIGIL_COMPLETION_MIN_SECONDS (1): "✻ Baked for 12s"."""
    from .prompt_constants import TEARDROP_ASTERISK, random_completion_verb

    try:
        minimum = float(os.environ.get('SIGIL_COMPLETION_MIN_SECONDS', '1'))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        minimum = 1.0
    if elapsed >= minimum:
        output_func(f'\033[2m{TEARDROP_ASTERISK} {random_completion_verb()} for {elapsed:.0f}s\033[0m')


def _take_session_id(agent: Any) -> str | None:
    """The --session-id for the first new session (used once)."""
    requested = getattr(agent, 'requested_session_id', None)
    agent.requested_session_id = None
    return requested


def _maybe_fork(agent: Any, session_id: str | None) -> str | None:
    """--fork-session: continue a resumed session in a copy with a new id."""
    if not session_id or not getattr(agent, 'fork_session', False):
        return session_id
    from .session_store import fork_agent_session

    return fork_agent_session(agent.runtime_config.session_directory, session_id,
                              _take_session_id(agent))


def _append_agent_forwarded_args(
    command: list[str],
    args: argparse.Namespace,
    *,
    include_backend: bool,
) -> None:
    command.extend(['--cwd', str(args.cwd)])
    command.extend(['--max-turns', str(getattr(args, 'max_turns', 12))])
    if include_backend:
        command.extend(['--model', str(args.model)])
        command.extend(['--base-url', str(args.base_url)])
        # The API key is never put on the command line (any process list would
        # show it); _launch_background_agent passes it in the child env instead.
        for name, value in (
            ('--temperature', getattr(args, 'temperature', None)),
            ('--timeout-seconds', getattr(args, 'timeout_seconds', None)),
            ('--input-cost-per-million', getattr(args, 'input_cost_per_million', None)),
            ('--output-cost-per-million', getattr(args, 'output_cost_per_million', None)),
        ):
            if value is not None:
                command.extend([name, str(value)])
    else:
        command.extend(['--model', str(args.model)])
    for path in getattr(args, 'add_dir', []):
        command.extend(['--add-dir', str(path)])
    for name, values in (('--mcp-config', getattr(args, 'mcp_config', None)),
                         ('--plugin-dir', getattr(args, 'plugin_dir', None)),
                         ('--allowed-tools', getattr(args, 'allowed_tools', None)),
                         ('--disallowed-tools', getattr(args, 'disallowed_tools', None))):
        for value in values or ():
            command.extend([name, str(value)])
    for flag in (
        ('--disable-sigil-md', getattr(args, 'disable_sigil_md', False)),
        ('--allow-write', getattr(args, 'allow_write', False)),
        ('--allow-shell', getattr(args, 'allow_shell', False)),
        ('--unsafe', getattr(args, 'unsafe', False)),
        ('--stream', getattr(args, 'stream', False)),
        ('--debug', getattr(args, 'debug', False)),
        ('--strict-mcp-config', getattr(args, 'strict_mcp_config', False)),
        ('--architect', getattr(args, 'architect', False)),
        ('--exclude-dynamic-system-prompt-sections', getattr(args, 'exclude_dynamic_sections', False)),
        ('--safe-mode', getattr(args, 'safe_mode', False)),
        ('--verbose', getattr(args, 'verbose', False)),
        ('--show-transcript', getattr(args, 'show_transcript', False)),
        ('--tui', getattr(args, 'tui', False)),
        (
            '--response-schema-strict',
            getattr(args, 'response_schema_strict', False),
        ),
    ):
        if flag[1]:
            command.append(flag[0])
    for name, value in (
        ('--auto-snip-threshold', getattr(args, 'auto_snip_threshold', None)),
        ('--auto-compact-threshold', getattr(args, 'auto_compact_threshold', None)),
        ('--compact-preserve-messages', getattr(args, 'compact_preserve_messages', None)),
        ('--max-total-tokens', getattr(args, 'max_total_tokens', None)),
        ('--max-input-tokens', getattr(args, 'max_input_tokens', None)),
        ('--max-output-tokens', getattr(args, 'max_output_tokens', None)),
        ('--max-reasoning-tokens', getattr(args, 'max_reasoning_tokens', None)),
        ('--max-budget-usd', getattr(args, 'max_budget_usd', None)),
        ('--max-tool-calls', getattr(args, 'max_tool_calls', None)),
        ('--max-delegated-tasks', getattr(args, 'max_delegated_tasks', None)),
        ('--max-model-calls', getattr(args, 'max_model_calls', None)),
        ('--max-session-turns', getattr(args, 'max_session_turns', None)),
        ('--response-schema-file', getattr(args, 'response_schema_file', None)),
        ('--response-schema-name', getattr(args, 'response_schema_name', None)),
        ('--scratchpad-root', getattr(args, 'scratchpad_root', None)),
        ('--nudge-strategy', getattr(args, 'nudge_strategy', None)),
        ('--permission-prompts', getattr(args, 'permission_prompts', None)),
        ('--permission-mode', getattr(args, 'permission_mode', None)),
        ('--effort', getattr(args, 'effort', None)),
        ('--fallback-model', getattr(args, 'fallback_model', None)),
        ('--max-tokens', getattr(args, 'max_tokens', None)),
        ('--profile', getattr(args, 'profile', None)),
        ('--settings', getattr(args, 'settings_override', None)),
        ('--setting-sources', getattr(args, 'setting_sources', None)),
        ('--tools', getattr(args, 'tools', None)),
        ('--debug-file', getattr(args, 'debug_file', None)),
        ('--system-prompt', getattr(args, 'system_prompt', None)),
        ('--append-system-prompt', getattr(args, 'append_system_prompt', None)),
        ('--override-system-prompt', getattr(args, 'override_system_prompt', None)),
    ):
        if value is not None:
            command.extend([name, str(value)])


def _add_agent_resume_args(parser: argparse.ArgumentParser) -> None:
    _add_agent_common_args(parser, include_backend=True)
    parser.add_argument('session_id', nargs='?', default='latest', help='Session ID or "latest" / "backup"')
    parser.add_argument('prompt')
    parser.add_argument('--show-transcript', action='store_true')
def _launch_background_agent(args: argparse.Namespace) -> int:
    # records go under the agent's workspace (<--cwd>/.port_sessions/background), the same folder the
    # agent tool, the chat and `agent-ps --cwd` use; without --cwd that is the current folder
    background_runtime = BackgroundSessionRuntime(
        Path(getattr(args, 'cwd', None) or '.').expanduser().resolve() / '.port_sessions' / 'background')
    limit_message = _check_background_limit(background_runtime)
    if limit_message:
        sys.stderr.write(limit_message + '\n')
        return 1
    background_id = background_runtime.create_id()
    forwarded_args: list[str] = []
    _append_agent_forwarded_args(forwarded_args, args, include_backend=True)
    forwarded_args.extend(['--background-root', str(background_runtime.root)])
    command = build_background_worker_command(
        background_id=background_id,
        prompt=args.prompt,
        forwarded_args=forwarded_args,
    )
    # API key goes through the environment (read by --api-key's default), never argv.
    from .child_env import child_env

    env = child_env()  # N21: SIGIL_SSH_AGENT=off reaches background agents too
    if getattr(args, 'api_key', None):
        _forward_key(env, getattr(args, 'base_url', None) or os.environ.get('OPENAI_BASE_URL', ''), str(args.api_key))
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SIGIL_BACKGROUND'] = '1'
    record = background_runtime.launch(
        command,
        prompt=args.prompt,
        workspace_cwd=Path(args.cwd).resolve(),
        model=args.model,
        background_id=background_id,
        process_cwd=Path(__file__).resolve().parent.parent,
        env=env,
    )
    print('# Background Session')
    print(f'background_id={record.background_id}')
    print(f'pid={record.pid}')
    print(f'log_path={record.log_path}')
    print(f'record_path={record.record_path}')
    return 0


def _restart_background_agent(args: argparse.Namespace) -> int:
    runtime = _background_runtime_for_args(args)
    from .child_env import child_env

    env = child_env()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SIGIL_BACKGROUND'] = '1'
    try:
        record = runtime.restart(args.background_id, prompt=getattr(args, 'prompt', None), env=env)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'Cannot restart {args.background_id}: {exc}')
        return 1
    print('# Background Session')
    print(f'background_id={record.background_id}')
    print(f'status={record.status}')
    print(f'pid={record.pid}')
    return 0


def _run_background_worker(args: argparse.Namespace) -> int:
    background_runtime = BackgroundSessionRuntime(Path(args.background_root))
    exit_code = 1
    stop_reason = 'worker_failed'
    session_id = None
    session_path = None
    _cap_worker_output()
    os.environ['SIGIL_BACKGROUND'] = '1'  # hooks can tell they run inside a background agent
    _exit_cleanly_on_sigterm()
    try:
        agent = _build_agent(args)
        agent.on_permission_blocked = lambda message: background_runtime.set_note(args.background_id, message)
        from .background_approvals import ask as _ask_main_chat

        # Approval questions go to the main chat (or `sigil agent-approve`), see background_approvals.py.
        agent.background_ask = lambda request: _ask_main_chat(
            background_runtime.root, args.background_id, request,
            on_wait=lambda note: background_runtime.set_note(args.background_id, note))
        # The session id is the background id, so a restarted agent can continue it.
        stored = None
        if getattr(args, 'resume_session', False):
            try:
                stored = load_agent_session(args.background_id, directory=agent.runtime_config.session_directory)
            except (OSError, ValueError) as _ignored:
                _log_ignored(_ignored)
                stored = None  # stopped before saving: start the prompt again
        result = (agent.resume(args.prompt, stored) if stored is not None
                  else agent.run(args.prompt, session_id=args.background_id))
        _print_agent_result(result, show_transcript=args.show_transcript)
        from .command_hooks import fire_session_end

        fire_session_end(agent.command_hook_runtime, result.session_id, 'other')
        exit_code = 0
        stop_reason = result.stop_reason or 'completed'
        session_id = result.session_id
        session_path = result.session_path
        background_runtime.write_result(args.background_id, {
            'final_output': result.final_output,
            'session_id': result.session_id,
            'stop_reason': result.stop_reason,
            'total_cost_usd': result.total_cost_usd,
            'usage': result.usage.to_dict(),
            'turns': result.turns,
        })
        return 0
    finally:
        from .background_shells import shell_manager
        from .mcp_runtime import connection_manager

        shell_manager().kill_all()  # shells the worker started end with it
        connection_manager().close_all()
        background_runtime.mark_finished(
            args.background_id,
            exit_code=exit_code,
            stop_reason=stop_reason,
            session_id=session_id,
            session_path=session_path,
        )


def _exit_cleanly_on_sigterm() -> None:
    """POSIX: SIGTERM (agent-kill) raises SystemExit, so the worker's finally block still stops
    the background shells and MCP servers it started (they run in their own sessions)."""
    import signal

    if sys.platform == 'win32' or threading_main() is False:
        return

    def _handler(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError) as _ignored:
        _log_ignored(_ignored)
        pass


def threading_main() -> bool:
    import threading

    return threading.current_thread() is threading.main_thread()


class _CappedWriter:
    """Stops writing to a background log after SIGIL_BACKGROUND_LOG_MAX_BYTES (20 MB)."""

    def __init__(self, stream: Any, state: dict[str, int], limit: int) -> None:
        self._stream = stream
        self._state = state
        self._limit = limit

    def write(self, text: str) -> int:
        size = len(text.encode('utf-8', errors='replace'))
        if self._state['written'] + size > self._limit:
            if not self._state.get('capped'):
                self._state['capped'] = 1
                self._stream.write('\n[log size cap reached; further output is dropped]\n')
                self._stream.flush()
            return len(text)
        self._state['written'] += size
        return self._stream.write(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _cap_worker_output() -> None:
    try:
        limit = int(os.environ.get('SIGIL_BACKGROUND_LOG_MAX_BYTES', '') or 20 * 1024 * 1024)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        limit = 20 * 1024 * 1024
    if limit <= 0:
        return
    state = {'written': 0}
    sys.stdout = _CappedWriter(sys.stdout, state, limit)  # type: ignore[assignment]
    sys.stderr = _CappedWriter(sys.stderr, state, limit)  # type: ignore[assignment]


def _check_background_limit(runtime: Any) -> str | None:
    from .background_runtime import background_limit_message

    return background_limit_message(runtime)


def _build_resumed_agent(args: argparse.Namespace) -> tuple[LocalCodingAgent, StoredAgentSession]:
    session_id = args.session_id
    if session_id in ('latest', 'backup'):
        session_id = _lookup_latest_session_id(args.cwd, is_backup=(session_id == 'backup'))
    
    stored_session = load_agent_session(session_id, directory=_session_directory_for(args.cwd))
    model_config = deserialize_model_config(stored_session.model_config)
    runtime_config = deserialize_runtime_config(stored_session.runtime_config)
    if os.environ.get('SIGIL_RESUME_STORED_PERMISSIONS', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        # why (S14): the session file lives in the workspace and can be edited or planted, so the
        # folders and write/shell/unsafe permissions come from this command line, as for a new session
        cwd = Path(args.cwd).resolve()
        runtime_config = replace(
            runtime_config,
            cwd=cwd,
            permissions=AgentPermissions(),
            additional_working_directories=_added_directories(cwd, args.add_dir),
            session_directory=_session_directory_for(cwd),
            scratchpad_root=_session_root_for(cwd) / 'scratchpad',
        )

    if args.model:
        model_config = replace(model_config, model=args.model)
    if args.base_url:
        model_config = replace(model_config, base_url=args.base_url)
    if args.api_key:
        model_config = replace(model_config, api_key=args.api_key)
    if args.temperature is not None:
        model_config = replace(model_config, temperature=args.temperature)
    if args.timeout_seconds is not None:
        model_config = replace(model_config, timeout_seconds=args.timeout_seconds)
    if args.input_cost_per_million is not None or args.output_cost_per_million is not None:
        model_config = replace(
            model_config,
            pricing=replace(
                model_config.pricing,
                input_cost_per_million_tokens_usd=(
                    args.input_cost_per_million
                    if args.input_cost_per_million is not None
                    else model_config.pricing.input_cost_per_million_tokens_usd
                ),
                output_cost_per_million_tokens_usd=(
                    args.output_cost_per_million
                    if args.output_cost_per_million is not None
                    else model_config.pricing.output_cost_per_million_tokens_usd
                ),
            ),
        )

    if getattr(args, "max_turns", None) is not None:
        runtime_config = replace(runtime_config, max_turns=args.max_turns)
    if getattr(args, "allow_write", False) or getattr(args, "allow_shell", False) or getattr(args, "unsafe", False):
        runtime_config = replace(
            runtime_config,
            permissions=AgentPermissions(
                allow_file_write=runtime_config.permissions.allow_file_write or args.allow_write,
                allow_shell_commands=runtime_config.permissions.allow_shell_commands or args.allow_shell,
                allow_destructive_shell_commands=runtime_config.permissions.allow_destructive_shell_commands or args.unsafe,
            ),
        )
    if getattr(args, "stream", False):
        runtime_config = replace(runtime_config, stream_model_responses=True)
    if (
        getattr(args, "auto_snip_threshold", None) is not None
        or getattr(args, "auto_compact_threshold", None) is not None
        or getattr(args, "compact_preserve_messages", None) is not None
    ):
        runtime_config = replace(
            runtime_config,
            auto_snip_threshold_tokens=(
                getattr(args, "auto_snip_threshold", None)
                if getattr(args, "auto_snip_threshold", None) is not None
                else runtime_config.auto_snip_threshold_tokens
            ),
            auto_compact_threshold_tokens=(
                getattr(args, "auto_compact_threshold", None)
                if getattr(args, "auto_compact_threshold", None) is not None
                else runtime_config.auto_compact_threshold_tokens
            ),
            compact_preserve_messages=(
                max(0, getattr(args, "compact_preserve_messages", None))
                if getattr(args, "compact_preserve_messages", None) is not None
                else runtime_config.compact_preserve_messages
            ),
        )
    
    runtime_config = replace(runtime_config, tui=getattr(args, 'tui', False))

    if (
        getattr(args, "max_total_tokens", None) is not None
        or getattr(args, "max_input_tokens", None) is not None
        or getattr(args, "max_output_tokens", None) is not None
        or getattr(args, "max_reasoning_tokens", None) is not None
        or getattr(args, "max_budget_usd", None) is not None
        or getattr(args, "max_tool_calls", None) is not None
        or getattr(args, "max_delegated_tasks", None) is not None
        or getattr(args, "max_model_calls", None) is not None
        or getattr(args, "max_session_turns", None) is not None
    ):
        runtime_config = replace(
            runtime_config,
            budget_config=BudgetConfig(
                max_total_tokens=(
                    getattr(args, "max_total_tokens", None)
                    if getattr(args, "max_total_tokens", None) is not None
                    else runtime_config.budget_config.max_total_tokens
                ),
                max_input_tokens=(
                    getattr(args, "max_input_tokens", None)
                    if getattr(args, "max_input_tokens", None) is not None
                    else runtime_config.budget_config.max_input_tokens
                ),
                max_output_tokens=(
                    getattr(args, "max_output_tokens", None)
                    if getattr(args, "max_output_tokens", None) is not None
                    else runtime_config.budget_config.max_output_tokens
                ),
                max_reasoning_tokens=(
                    getattr(args, "max_reasoning_tokens", None)
                    if getattr(args, "max_reasoning_tokens", None) is not None
                    else runtime_config.budget_config.max_reasoning_tokens
                ),
                max_total_cost_usd=(
                    getattr(args, "max_budget_usd", None)
                    if getattr(args, "max_budget_usd", None) is not None
                    else runtime_config.budget_config.max_total_cost_usd
                ),
                max_tool_call_count=(
                    getattr(args, "max_tool_calls", None)
                    if getattr(args, "max_tool_calls", None) is not None
                    else runtime_config.budget_config.max_tool_call_count
                ),
                max_delegated_tasks=(
                    getattr(args, "max_delegated_tasks", None)
                    if getattr(args, "max_delegated_tasks", None) is not None
                    else runtime_config.budget_config.max_delegated_tasks
                ),
                max_model_calls=(
                    getattr(args, "max_model_calls", None)
                    if getattr(args, "max_model_calls", None) is not None
                    else runtime_config.budget_config.max_model_calls
                ),
                max_session_turns=(
                    getattr(args, "max_session_turns", None)
                    if getattr(args, "max_session_turns", None) is not None
                    else runtime_config.budget_config.max_session_turns
                ),
            ),
        )
    
    runtime_config = replace(runtime_config, tui=getattr(args, 'tui', False))

    output_schema = _load_output_schema_config(args)
    if output_schema is not None:
        runtime_config = replace(runtime_config, output_schema=output_schema)
    if getattr(args, "scratchpad_root", None):
        runtime_config = replace(
            runtime_config,
            scratchpad_root=Path(args.scratchpad_root).resolve(),
        )
    # Priority: CLI argument > Config file > Session state
    nudge_override = getattr(args, "nudge_strategy", None)
    if nudge_override:
        runtime_config = replace(runtime_config, nudge_strategy=nudge_override)
    else:
        config_runtime = ConfigRuntime.from_workspace(runtime_config.cwd)
        try:
            persisted_nudge = config_runtime.get_value('nudge_strategy')
            runtime_config = replace(runtime_config, nudge_strategy=persisted_nudge)
        except KeyError as _ignored:
            _log_ignored(_ignored)
            pass # Keep session's existing setting

    agent = LocalCodingAgent(
        model_config=model_config,
        runtime_config=runtime_config,
    )
    return agent, stored_session


def _print_agent_result(result, *, show_transcript: bool) -> None:
    print(result.final_output)
    print('\n# Usage')
    print(f'total_tokens={result.usage.total_tokens}')
    print(f'input_tokens_cumulative={result.usage.input_tokens}')
    print(f'last_context_tokens={result.last_turn_input_tokens}')
    print(f'output_tokens={result.usage.output_tokens}')
    print(f'total_cost_usd={result.total_cost_usd:.6f}')
    if result.stop_reason:
        print(f'stop_reason={result.stop_reason}')
    if result.session_id:
        print('\n# Session')
        print(f'session_id={result.session_id}')
        if result.session_path:
            print(f'session_path={result.session_path}')
    if result.scratchpad_directory:
        print(f'scratchpad_directory={result.scratchpad_directory}')
    if show_transcript:
        print('\n# Transcript')
        for message in result.transcript:
            role = message.get('role', 'unknown')
            print(f'[{role}]')
            print(message.get('content', ''))


def build_permission_prompt(
    ask: Callable[[str], str],
    output_func: Callable[[str], None] = print,
) -> Callable[[dict[str, Any]], str]:
    """An approval prompt: allow once, always allow in this project, or deny."""
    def _prompt(request: dict[str, Any]) -> str:
        from .terminal import set_title

        set_title(f'Sigil: needs approval ({request.get("tool")})')
        arguments = request.get('arguments') or {}
        from .terminal import strip_terminal_controls as _clean

        detail = (arguments.get('command') or arguments.get('path') or arguments.get('url')
                  or str(arguments.get('plan') or '')[:600] or '')
        # what the user approves is shown as plain text: an escape sequence in a model-written
        # command could otherwise hide or rewrite part of it on screen
        output_func(f'\033[33m? {_clean(str(request.get("tool")))} needs approval: {_clean(str(detail))}\033[0m')
        output_func(f'  {_clean(str(request.get("reason", "")))}')
        output_func(f'  [y] allow once   [s] allow `{_clean(str(request.get("rule")))}` for this session   '
                    f'[a] always allow it here   [n] deny')
        answer = ask('allow? [y/s/a/N] ').strip().lower()
        if answer in ('y', 'yes', '1', 'once'):
            return 'once'
        if answer in ('s', 'session'):
            return 'session'
        if answer in ('a', 'always', '2'):
            return 'always'
        return 'deny'

    return _prompt


def _prompt_history() -> Any:
    """Prompt history saved in ~/.sigil/prompt_history (SIGIL_HISTORY_FILE overrides;
    an empty value keeps history in memory only)."""
    try:
        from prompt_toolkit.history import FileHistory, InMemoryHistory
    except ImportError:
        return None
    raw = os.environ.get('SIGIL_HISTORY_FILE')
    if raw is not None and not raw.strip():
        return InMemoryHistory()
    path = Path(raw).expanduser() if raw else Path.home() / '.sigil' / 'prompt_history'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileHistory(str(path))
    except OSError as _ignored:
        _log_ignored(_ignored)
        return InMemoryHistory()


def _run_shell_escape(agent: Any, command: str, notes: list[str]) -> str:
    """`!cmd` in chat: run through the bash permission checks and keep the output."""
    from .agent_tools import execute_tool
    from .agent_types import ToolCall

    if not command:
        return 'Usage: !<command>'
    call = ToolCall(id='bang_command', name='bash', arguments={'command': command})
    blocked, context = agent._check_permission_rules(call, [])
    if blocked is not None:
        return blocked.content
    result = execute_tool(agent.tool_registry, 'bash', {'command': command}, context or agent.tool_context)
    if result.ok:
        from .prompt_constants import BASH_INPUT_TAG, BASH_STDOUT_TAG

        notes.append(f'<{BASH_INPUT_TAG}>{command}</{BASH_INPUT_TAG}>\n'
                     f'<{BASH_STDOUT_TAG}>\n{result.content}\n</{BASH_STDOUT_TAG}>')
    return result.content


def _pick_session_in_chat(agent: Any, arg: str, output_func: Callable[[str], None],
                          ask: Callable[[str], str]) -> str | None:
    from .session_store import list_agent_sessions, session_first_prompt

    directory = agent.runtime_config.session_directory
    if not arg:
        sessions = list_agent_sessions(directory)
        if not sessions:
            output_func('No saved sessions in this workspace.')
            return None
        output_func('Saved sessions (newest first):')
        for index, (session_id, mtime, count) in enumerate(sessions[:20]):
            stamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            first = session_first_prompt(directory, session_id)
            output_func(f'  [{index}] {session_id}{_bg_mark(session_id)}  {stamp}  {count} msgs  {first}')
        try:
            choice = ask('resume # (Enter to cancel)> ').strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice:
            output_func('Resume cancelled.')
            return None
        arg = sessions[int(choice)][0] if choice.isdigit() and int(choice) < len(sessions) else choice
    elif not (directory / f'{arg}.json').exists():
        from .session_store import resolve_session_name, search_agent_sessions

        if resolve_session_name(directory, arg) == arg:
            matches = search_agent_sessions(directory, arg)
            if not matches:
                output_func(f'No session id, name or text matches {arg!r}.')
                return None
            output_func(f'Sessions mentioning {arg!r} (newest first):')
            for index, (session_id, mtime, count, snippet) in enumerate(matches[:20]):
                stamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                output_func(f'  [{index}] {session_id}{_bg_mark(session_id)}  {stamp}  {count} msgs  {snippet}')
            try:
                choice = ask('resume # (Enter to cancel)> ').strip()
            except (EOFError, KeyboardInterrupt):
                return None
            if not choice.isdigit() or int(choice) >= len(matches):
                output_func('Resume cancelled.')
                return None
            arg = matches[int(choice)][0]
        else:
            arg = resolve_session_name(directory, arg)
    try:
        load_agent_session(arg, directory=directory)
    except (OSError, ValueError) as exc:
        output_func(f'Cannot resume {arg}: {exc}')
        return None
    return arg


def _argv_value(argv: list[str], flag: str) -> str | None:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(flag + '='):
            return token.split('=', 1)[1]
    return None


def _apply_user_config(cwd: str | Path) -> None:
    """Fill in env vars the user has not set: <cwd>/.env, ~/.sigil/.env, settings `env`,
    and ~/.sigil/settings.json `model` (as OPENAI_MODEL).
    """
    from .settings import _home, load_settings, read_settings_file, settings_paths

    root = Path(cwd).expanduser()
    from .workspace_trust import is_trusted, report_skipped

    # Why: a cloned repository's .env / settings env (OPENAI_BASE_URL, ...) used to apply unasked
    project_trusted = is_trusted(root)
    if not project_trusted:
        report_skipped(root)
    from_project: list[str] = []  # names a project file filled in (shown on stderr, never values)
    for env_path in ((root / '.env',) if project_trusted else ()) + (_home() / '.sigil' / '.env',):
        filled = _load_env_values(env_path)
        if env_path == root / '.env':
            from_project += filled
    try:
        merged_env = load_settings(root).get('env') if project_trusted else \
            read_settings_file(_home() / '.sigil' / 'settings.json').get('env')
    except Exception as _ignored:
        _log_ignored(_ignored)
        merged_env = None
    project_env_keys: set[str] = set()
    for label, path in settings_paths(root, include_user=False):
        env_block = read_settings_file(path).get('env')
        if isinstance(env_block, dict):
            project_env_keys.update(str(key) for key in env_block)
    sigil_settings = read_settings_file(_home() / '.sigil' / 'settings.json')
    for source in (merged_env, sigil_settings.get('env')):
        if isinstance(source, dict):
            for key, value in source.items():
                if isinstance(key, str) and isinstance(value, (str, int, float)) and key not in os.environ:
                    os.environ[key] = str(value)
                    if source is merged_env and key in project_env_keys:
                        from_project.append(key)
    model = sigil_settings.get('model')
    if isinstance(model, str) and model.strip():
        os.environ.setdefault('OPENAI_MODEL', model.strip())
    if from_project and os.environ.get('SIGIL_QUIET_PROJECT_ENV', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        names = sorted(set(from_project))
        print(f'Sigil: this project set {len(names)} environment value(s) from .env / .sigil settings: '
              f'{", ".join(names)} (SIGIL_QUIET_PROJECT_ENV=1 hides this line)', file=sys.stderr)


def _load_env_values(path: Path) -> list[str]:
    """Set the variables a .env file names that are not set yet; returns the names it set."""
    try:
        text = path.read_text(encoding='utf-8-sig')
    except OSError as _ignored:
        _log_ignored(_ignored)
        return []
    filled: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            if key.startswith('export '):
                key = key[len('export '):].strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
                filled.append(key)
    return filled


def _auto_test_enabled(agent: Any) -> bool:
    """autoTest: true in settings (or SIGIL_AUTO_TEST=1): run the tests after edits, once per prompt."""
    if os.environ.get('SIGIL_AUTO_TEST', '').strip() == '1':
        return True
    try:
        from .settings import load_settings

        return load_settings(agent.runtime_config.cwd).get('autoTest') is True
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def _vim_mode(cwd: Path) -> bool:
    """editorMode: "vim" in settings (or SIGIL_EDITOR_MODE=vim) turns on vi key bindings."""
    mode = os.environ.get('SIGIL_EDITOR_MODE')
    if not mode:
        try:
            from .settings import load_settings

            mode = load_settings(cwd).get('editorMode')
        except Exception as _ignored:
            _log_ignored(_ignored)
            mode = None
    return str(mode or '').strip().lower() == 'vim'


def _status_line(agent: Any) -> Any:
    try:
        from .status_line import status_line_for

        return status_line_for(agent.runtime_config.cwd)
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None


def _message_display(agent: Any, result: Any) -> str | None:
    """The MessageDisplay hook for a finished answer (one final batch). Returns the hook's
    displayContent, shown on screen instead of the text (the transcript keeps the original), or None."""
    from uuid import uuid4

    text = (result.final_output or '')[:20000]
    outcome = agent._run_hook_event('MessageDisplay', {
        'session_id': result.session_id, 'turn_id': str(uuid4()), 'message_id': str(uuid4()),
        'message': text, 'delta': text, 'index': 0, 'final': True})
    shown = getattr(outcome, 'display_content', None)
    return shown if isinstance(shown, str) else None


def _worktree_runtime(cwd: Path) -> Any:
    """WorktreeRuntime for the CLI routes, with the workspace's WorktreeCreate / WorktreeRemove hooks."""
    from .command_hooks import CommandHookRuntime, wire_worktree_hooks
    from .worktree_runtime import WorktreeRuntime

    return wire_worktree_hooks(WorktreeRuntime.from_workspace(cwd.resolve()),
                               CommandHookRuntime.from_workspace(cwd.resolve()).run_if_configured)


def _enter_worktree(args: argparse.Namespace) -> None:
    """-w/--worktree NAME: work in a git worktree (created if needed) for this session."""
    name = getattr(args, 'worktree', None)
    if not name:
        return

    runtime = _worktree_runtime(Path(args.cwd))
    current = runtime.current_report()
    if current.active and current.worktree_path:
        args.cwd = current.worktree_path  # reuse the worktree this workspace is already in
        sys.stderr.write(f'Working in worktree {current.worktree_path}\n')
        return
    try:
        report = runtime.enter(name)
        path = report.worktree_path
    except (RuntimeError, ValueError) as exc:
        text = str(exc)
        if 'already exists:' not in text:
            raise SystemExit(f'cannot create worktree {name!r}: {text}')
        path = text.split('already exists:', 1)[1].strip()
    if not path:
        raise SystemExit(f'cannot create worktree {name!r}')
    args.cwd = path
    sys.stderr.write(f'Working in worktree {path}\n')


def _attach_mentions(agent: Any, prompt: str, output_func: Callable[[str], None] | None = None) -> str:
    """Add the contents of @path mentions (through path access and size limits)."""
    if '@' not in prompt:
        return prompt
    try:
        from .custom_commands import expand_file_mentions

        expanded, attached = expand_file_mentions(prompt, agent)
    except Exception as _ignored:
        _log_ignored(_ignored)
        return prompt
    if attached and output_func is not None:
        output_func('\033[2m  \u21b3 attached: ' + ', '.join(attached) + '\033[0m')
    return expanded


def _other_directories_line(agent: Any) -> str | None:
    """'Other directories: ...' when the switch is on, else None."""
    try:
        from .agent_tools import _path_access

        access = _path_access(agent.tool_context)
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None
    return access.describe() if access.enabled else None


def _run_agent_chat_loop(
    agent: LocalCodingAgent,
    *,
    initial_prompt: str | None,
    resume_session_id: str | None,
    show_transcript: bool,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    result_printer: Callable[..., None] = _print_agent_result,
) -> int:
    active_session_id = resume_session_id
    if active_session_id and active_session_id not in ('latest', 'backup'):
        from .session_store import resolve_session_name

        active_session_id = resolve_session_name(agent.runtime_config.session_directory, active_session_id)
    if active_session_id in ('latest', 'backup'):
        # -r / --resume: pick the newest session in this workspace's session folder.
        try:
            active_session_id = _lookup_latest_session_id(
                str(agent.runtime_config.cwd),
                is_backup=(active_session_id == 'backup'),
                session_dir=agent.runtime_config.session_directory,
            )
        except OSError as exc:
            output_func(f'No session to resume ({exc}); starting a new one.')
            active_session_id = None
    if active_session_id:
        try:
            forked = _maybe_fork(agent, active_session_id)
        except (OSError, ValueError) as exc:
            output_func(f'Cannot fork {active_session_id}: {exc}')
            forked = active_session_id
        if forked != active_session_id:
            output_func(f'forked_session_id={forked} (from {active_session_id})')
            active_session_id = forked
    first_prompt = initial_prompt
    end_reason = 'other'

    # Launchers can enter here without going through main(), so VT processing
    # must be enabled here as well.
    _enable_windows_vt()

    from .terminal import enable_plain, plain_enabled, session_title, set_title

    if plain_enabled():
        enable_plain()  # no colours, boxes or animation (screen readers, log files)
    else:
        from .terminal import apply_theme

        apply_theme()  # /theme light rewrites hard-to-read colours
    set_title(session_title(agent, 'ready'))
    tui = None
    if getattr(agent.runtime_config, 'tui', False) and not plain_enabled():
        tui = SigilTUIDisplay()
        agent.event_callback = tui.on_event
        tui.start()

    try:
        if not tui:
            output_func('# Agent Chat')
            output_func("Enter a prompt. Use '/exit' or '/quit' to stop.")
            access_line = _other_directories_line(agent)
            if access_line:
                output_func(access_line)

        # Initialize prompt_toolkit session if available
        pt_session: Any = None
        agent_bar = AgentBar(agent.runtime_config.cwd)
        agent_bar.mode = getattr(agent, 'permission_mode', 'default') or 'default'
        status_line = _status_line(agent)
        if status_line is not None:
            from .status_line import status_payload

            agent_bar.status_provider = lambda: status_line.render(status_payload(agent))
        bar_kwargs: dict[str, Any] = {}
        # Only use prompt_toolkit for real interactive terminals with the default reader;
        # a custom input_func (tests, embedding) or piped stdin must keep working.
        if HAS_PROMPT_TOOLKIT and input_func is input and sys.stdin.isatty() and not plain_enabled():
            try:
                from .completion import build_completer

                # Tab completes /commands and @paths; history is kept across sessions
                # (Up / Ctrl+R search it).
                pt_session = PromptSession(
                    completer=build_completer(agent),
                    complete_while_typing=False,
                    history=_prompt_history(),
                    vi_mode=_vim_mode(agent.runtime_config.cwd),
                )
            except Exception as _ignored:
                _log_ignored(_ignored)
                try:
                    pt_session = PromptSession()
                except Exception as _ignored:
                    _log_ignored(_ignored)
                    pt_session = None
        if pt_session is not None:
            try:
                bar_kwargs = build_agent_bar_kwargs(agent_bar)
            except Exception as _ignored:
                _log_ignored(_ignored)
                bar_kwargs = {}

        # [ASK USER] ask_user_question reads the answer here. The queued-input
        # reader (set per prompt below) is paused while the question is open.
        reader_holder: dict[str, Any] = {'reader': None}
        shell_notes: list[str] = []  # !cmd output waiting to go with the next prompt
        next_is_auto_test = False  # the queued prompt came from autoTest (no second round)

        def _ask_user_input(prompt_text: str) -> str:
            reader = reader_holder['reader']
            if reader is not None:
                reader.pause()
            try:
                if pt_session is not None and patch_stdout is not None:
                    with patch_stdout():
                        return pt_session.prompt(prompt_text)
                return input_func(prompt_text)
            finally:
                if reader is not None:
                    reader.resume()

        # [MCP] Elicitation: servers may ask the user for input during a tool call.
        try:
            from .mcp_runtime import set_elicitation_handler

            set_elicitation_handler(lambda server, params: agent.answer_elicitation(
                server, params, ask=_ask_user_input, output=output_func))
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)

        # [MCP] Start configured servers in the background; /mcp shows their status.
        mcp_runtime = getattr(agent, 'mcp_runtime', None)
        if mcp_runtime is not None and getattr(mcp_runtime, 'servers', ()) and                 os.environ.get('SIGIL_MCP_EAGER', '1').strip().lower() not in {'0', 'false', 'no', 'off'}:
            try:
                mcp_runtime.start_background()
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)
        ask_runtime = getattr(agent, 'ask_user_runtime', None)
        if ask_runtime is not None and getattr(ask_runtime, 'input_func', None) is None:
            ask_runtime.input_func = _ask_user_input
            ask_runtime.interactive = True
        # [APPROVALS] With a real terminal, calls that need approval are asked here.
        if pt_session is not None and getattr(agent, 'permission_prompt', None) is None:
            agent.permission_prompt = build_permission_prompt(_ask_user_input, output_func)
        if active_session_id:
            output_func(f'resuming_session_id={active_session_id}')
            try:
                historical_session = load_agent_session(
                    active_session_id,
                    directory=agent.runtime_config.session_directory,
                )
                if historical_session.messages:
                    # [TUI REPLAY]
                    if tui:
                        # Filter and replay last 10 relevant messages
                        # We convert AgentMessage to TUI events
                        history = historical_session.messages[-15:] # Buffer for context
                        for msg in history:
                            role = msg.get('role')
                            content = msg.get('content')
                            
                            event = None
                            if role == 'user':
                                event = {"type": "assistant_response", "content": f"user> {content}"}
                            elif role == 'assistant':
                                # Simplify: convert assistant message to response + tool calls
                                if content:
                                    event = {"type": "assistant_response", "content": f"agent> {content}"}
                                    tui.on_event(event, replay=True)
                                
                                for tc in msg.get('tool_calls', []):
                                    # Handle both tool_calls and legacy structure if needed, but assuming standard format
                                    name = 'tool'
                                    if isinstance(tc, dict):
                                        name = tc.get('function', {}).get('name', 'tool')
                                    tui.on_event({"type": "tool_start", "tool_name": name}, replay=True)
                                    tui.on_event({"type": "tool_result", "tool_name": name, "ok": True}, replay=True)
                                continue # Already handled internal TUI events
                            
                            if event:
                                tui.on_event(event, replay=True)
                        
                        # Add gold divider
                        tui.on_event({"type": "resumption_marker"})
                    
                    elif show_transcript:
                        output_func('\n--- Previous Conversation History ---')
                        for msg in historical_session.messages:
                            role = msg.get('role')
                            content = msg.get('content')
                            if role == 'user':
                                output_func(f'user> {content}')
                            elif role == 'assistant' and content:
                                output_func(f'agent> {content}')
                            elif role == 'system' and content and not content.startswith('<system-reminder>'):
                                output_func(f'system> {content}')
                        output_func('--------------------------------------\n')
            except Exception as e:
                output_func(f'Warning: Could not load history transcript: {e}')

        if pt_session is not None:
            try:
                from .background_approvals import start_watcher

                start_watcher(_background_dir_for(agent), output_func)
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)

        # [SCHEDULE] /loop, /goal and the cron_* tools (scheduler.py): prompts on a timer.
        from .scheduler import GOAL_DONE, Scheduler, goal_prompt, goal_rounds
        from .scheduler import register_tools as _register_cron_tools

        scheduler = Scheduler()
        agent.scheduler = scheduler
        _register_cron_tools(agent, scheduler)
        goal_state: dict[str, Any] = {}
        import threading as _threading

        scheduler_stop = _threading.Event()
        if pt_session is not None:
            def _wake_for_due_jobs() -> None:
                # A job that comes due while the prompt waits ends the prompt (empty input).
                while not scheduler_stop.wait(1.0):
                    wait = scheduler.seconds_until_next()
                    app = getattr(pt_session, 'app', None)
                    if wait is None or wait > 0 or app is None or not getattr(app, 'is_running', False):
                        continue
                    try:
                        loop = getattr(app, 'loop', None)
                        if loop is not None:
                            loop.call_soon_threadsafe(lambda: app.exit(result=''))
                    except Exception as _ignored:
                        from .debug_log import log_ignored_error

                        log_ignored_error(_ignored)

            _threading.Thread(target=_wake_for_due_jobs, name='scheduled-prompts', daemon=True).start()

        while True:
            auto_test_prompt = next_is_auto_test
            next_is_auto_test = False
            if first_prompt is None:
                due = scheduler.pop_due()
                if due is not None:
                    from .scheduler import prompt_text
                    from .terminal import strip_terminal_controls

                    output_func(f'\033[2m  ↳ scheduled prompt {due.job_id}: {strip_terminal_controls(due.prompt)}\033[0m')
                    first_prompt = prompt_text(due)
            if first_prompt is not None:
                prompt = first_prompt
                first_prompt = None
            else:
                nudge_mode = agent.runtime_config.nudge_strategy
                has_nudge = nudge_mode != 'off'
                
                # ANSI-formatted label for prompt_toolkit
                pt_label = f'user [\033[36m{nudge_mode}\033[0m]> ' if has_nudge else 'user> '
                # Plain-text label for standard input() to avoid raw escape codes on legacy terminals
                plain_label = f'user [{nudge_mode}]> ' if has_nudge else 'user> '
                
                try:
                    from .background_approvals import answer_pending

                    # Approval questions from background agents (background_approvals.py).
                    answer_pending(_background_dir_for(agent), _ask_user_input, output_func)
                except Exception as _ignored:
                    from .debug_log import log_ignored_error

                    log_ignored_error(_ignored)
                try:
                    if HAS_PROMPT_TOOLKIT and pt_session and ANSI:
                        with patch_stdout():
                            agent_bar.reset()
                            prompt = pt_session.prompt(ANSI(pt_label), **bar_kwargs)
                    else:
                        prompt = input_func(plain_label)
                except EOFError:
                    output_func('chat_ended=eof')
                    end_reason = 'prompt_input_exit'
                    return 0
                except KeyboardInterrupt:
                    output_func('\nchat_ended=interrupt')
                    return 130

            # Show current context usage status if a session is active
            if active_session_id and agent.last_session:
                cur_tokens = agent._calculate_total_context_tokens(agent.last_session)
                limit = agent.runtime_config.auto_compact_threshold_tokens or 0
                if limit > 0:
                    perc = (cur_tokens / limit) * 100
                    output_func(f'  [Context: {cur_tokens:,} / {limit:,} | {perc:.1f}% used]')
                else:
                    output_func(f'  [Context: {cur_tokens:,} used (no limit)]')

            def _sync_mode() -> None:
                from .permission_modes import MODE_LABELS, apply_mode

                if agent_bar.mode != getattr(agent, 'permission_mode', 'default'):
                    apply_mode(agent, agent_bar.mode)
                    output_func(f'\033[35m\u23f5\u23f5 Permission mode: {MODE_LABELS[agent_bar.mode]}\033[0m')

            _sync_mode()  # Shift+Tab may have changed it at the prompt
            agent.show_todos = agent_bar.show_todos or os.environ.get('SIGIL_SHOW_TODOS', '') == '1'
            normalized = prompt.strip()
            if not normalized:
                continue
            if normalized in {'/exit', '/quit'}:
                output_func('chat_ended=user_exit')
                end_reason = 'exit'
                return 0

            bar_output = agent_bar.handle_command(normalized)
            if bar_output is not None:
                output_func(bar_output)
                continue

            # [SCHEDULE] /loop <interval|cron> <prompt>, /loop stop [id|all]; /goal <goal>, /goal stop
            if normalized == '/loop' or normalized.startswith('/loop '):
                rest = normalized[len('/loop'):].strip()
                if not rest:
                    output_func(scheduler.render())
                elif rest == 'stop' or rest.startswith('stop '):
                    target = rest[len('stop'):].strip() or 'all'
                    output_func(f'Stopped {target}.' if scheduler.delete(target) else f'Nothing to stop: {target}')
                else:
                    import re as _re

                    cron_fields = rest.split()
                    if len(cron_fields) > 5 and all(_re.fullmatch(r'[\d*/,-]+', f) for f in cron_fields[:5]):
                        schedule, loop_prompt = ' '.join(cron_fields[:5]), ' '.join(cron_fields[5:])
                    else:
                        schedule, _, loop_prompt = rest.partition(' ')
                    try:
                        job = scheduler.add(schedule, loop_prompt)
                        output_func(f'Scheduled {job.job_id}: every {job.schedule}: {job.prompt} (/loop stop {job.job_id})')
                    except ValueError as exc:
                        output_func(f'/loop: {exc}. Usage: /loop 5m <prompt>  or  /loop */15 * * * * <prompt>')
                continue
            if normalized == '/goal' or normalized.startswith('/goal '):
                goal = normalized[len('/goal'):].strip()
                if not goal:
                    output_func(f'Goal: {goal_state["text"]} ({goal_state["rounds"]} round(s) left)' if goal_state
                                else 'No goal. Usage: /goal <goal>  (/goal stop ends it)')
                    continue
                if goal == 'stop':
                    goal_state.clear()
                    output_func('Goal stopped.')
                    continue
                goal_state.update(text=goal, rounds=goal_rounds())
                prompt = goal_prompt(goal, True)
                normalized = prompt

            # [MODES] Shift+Tab changes agent_bar.mode; /mode <name> sets it.
            from .permission_modes import MODES, normalize_mode

            if normalized == '/mode' or normalized.startswith('/mode '):
                arg = normalized[len('/mode'):].strip()
                if not arg:
                    output_func(
                        f'Permission mode: {agent_bar.mode}. Options: {", ".join(MODES)}. '
                        'Shift+Tab cycles default / acceptEdits / plan.'
                    )
                    continue
                chosen = normalize_mode(arg)
                if chosen is None:
                    output_func(f'Unknown mode {arg!r}. Options: {", ".join(MODES)}')
                    continue
                agent_bar.mode = chosen
                _sync_mode()
                continue

            # [REWIND] /rewind lists prompts; /rewind <n> [both|conversation|code] goes back.
            if normalized == '/rewind' or normalized.startswith('/rewind '):
                from .rewind import list_prompt_points, render_prompt_list, rewind_session

                if not active_session_id:
                    output_func('Nothing to rewind yet.')
                    continue
                args_list = normalized.split()[1:]
                try:
                    stored_for_rewind = load_agent_session(
                        active_session_id, directory=agent.runtime_config.session_directory,
                    )
                    if not args_list:
                        points = list_prompt_points(stored_for_rewind.messages)
                        output_func(render_prompt_list(points, agent._checkpoints_for(active_session_id)))
                        if points and pt_session is not None:
                            with patch_stdout():
                                choice = pt_session.prompt('rewind to # (Enter to cancel)> ').strip()
                            if not choice:
                                output_func('Rewind cancelled.')
                                continue
                            args_list = choice.split()
                        else:
                            continue
                    number = int(args_list[0])
                    mode_arg = args_list[1] if len(args_list) > 1 else 'both'
                    report, original_text = rewind_session(agent, active_session_id, number, mode_arg)
                    output_func(f'\033[32m\u21b6 {report}\033[0m')
                    if mode_arg in ('both', 'conversation'):
                        output_func(f'\033[2mYour prompt was:\033[0m {original_text}')
                except ValueError as exc:
                    output_func(f'\033[33m{exc}\033[0m')
                except Exception as exc:
                    output_func(f'\033[31mRewind failed: {exc}\033[0m')
                continue
            
            if normalized == '/compact' or normalized.startswith('/compact '):
                if active_session_id:
                    output_func('\033[2m🗜️ Manually compacting session...\033[0m')
                    try:
                        stored_session = load_agent_session(
                            active_session_id,
                            directory=agent.runtime_config.session_directory,
                        )
                        before_count = len(stored_session.messages)
                        instructions = normalized[len('/compact'):].strip() or None
                        compact_result = agent.compact(stored_session, instructions)
                        after_count = len(agent.last_session.messages) if agent.last_session else before_count
                        if compact_result is not None and not compact_result.error:
                            output_func(
                                f'\033[32m✓ Session compacted with an LLM summary: '
                                f'{before_count} → {after_count} messages '
                                f'(~{compact_result.pre_compact_token_count:,} → '
                                f'~{compact_result.post_compact_token_count:,} tokens).\033[0m'
                            )
                        else:
                            reason = compact_result.error if compact_result is not None else 'unknown'
                            output_func(
                                f'\033[33m! LLM summary failed ({reason}). '
                                f'Fell back to trimming old tool output: {before_count} → {after_count} messages.\033[0m'
                            )
                    except Exception as e:
                        output_func(f'\033[31mError during manual compaction: {e}\033[0m')
                else:
                    output_func('\033[33mNo active session to compact.\033[0m')
                continue

            if normalized in {'/tokens', '/token'}:
                if active_session_id:
                    output_func('\033[2m📊 Detailed Context Breakdown:\033[0m')
                    output_func(agent.render_context_report())
                else:
                    output_func('\033[33mNo active session to report tokens for.\033[0m')
                continue

            # [!CMD] Run a shell command now; its output goes along with the next prompt.
            if normalized.startswith('!') and len(normalized) > 1:
                output_func(_run_shell_escape(agent, normalized[1:].strip(), shell_notes))
                continue

            # [RESUME] /resume [id]: switch to a saved session in this workspace.
            if normalized == '/resume' or normalized.startswith('/resume '):
                picked = _pick_session_in_chat(agent, normalized[len('/resume'):].strip(), output_func,
                                               (lambda label: pt_session.prompt(label)) if pt_session is not None
                                               else input_func)
                if picked:
                    active_session_id = picked
                    output_func(f'resuming_session_id={picked}')
                continue

            if shell_notes:
                prompt = '\n\n'.join(shell_notes) + '\n\n' + prompt
                shell_notes.clear()
            if agent_bar.pasted_images:
                for problem in agent.attach_images(agent_bar.pasted_images):
                    output_func(f'\033[33mimage not attached: {problem}\033[0m')
                agent_bar.pasted_images = []

            # [@MENTIONS] @path attaches that file's contents to the prompt.
            if not normalized.startswith('/'):
                prompt = _attach_mentions(agent, prompt, output_func)

            # [QUEUED INPUT] Let the user type to the agent while it works.
            input_reader = None
            if pt_session is not None and ConsoleInputReader.supported():
                queue = MessageQueue(
                    notify=lambda text: output_func(f'\033[2m  ↳ queued for agent: {text}\033[0m')
                )
                agent.message_queue = queue

                real_stdout = sys.stdout

                def _echo(text: str) -> None:
                    real_stdout.write(text)
                    real_stdout.flush()

                def _esc_interrupt() -> None:
                    # Esc on an empty line = interrupt, same path as Ctrl+C.
                    import _thread
                    _echo('\n\033[33m⏹ Esc pressed: interrupting...\033[0m\n')
                    _thread.interrupt_main()

                def _send_now() -> None:
                    import _thread
                    _echo('\n\033[33m\u23f5 Ctrl+Enter: sending the queued message(s) now...\033[0m\n')
                    _thread.interrupt_main()

                input_reader = ConsoleInputReader(queue, echo=_echo, on_escape=_esc_interrupt, on_send_now=_send_now)
                input_reader.redirect_output(agent)
                input_reader.start()
            reader_holder['reader'] = input_reader
            try:
                try:
                    run_started = time.monotonic()
                    set_title(session_title(agent, 'working'))
                    files_before_run = 0
                    if active_session_id:
                        try:
                            files_before_run = len(load_agent_session(
                                active_session_id, directory=agent.runtime_config.session_directory).file_history)
                        except Exception as _ignored:
                            _log_ignored(_ignored)
                            files_before_run = 0
                        stored_session = load_agent_session(
                            active_session_id,
                            directory=agent.runtime_config.session_directory,
                        )
                        result = agent.resume(prompt, stored_session)
                    else:
                        result = agent.run(prompt, session_id=_take_session_id(agent))
                    shown = _message_display(agent, result)
                    from .desktop import notify, notify_after_seconds

                    if pt_session is not None and time.monotonic() - run_started >= notify_after_seconds():
                        notify('Sigil Code', 'finished: ' + (result.stop_reason or 'done'))
                finally:
                    reader_holder['reader'] = None
                    set_title(session_title(agent, 'ready'))
                    agent_bar.mode = getattr(agent, 'permission_mode', agent_bar.mode)  # /plan <task>
                    if input_reader is not None:
                        input_reader.stop()
                        input_reader.restore_output()
                        leftover = agent.message_queue.drain() if agent.message_queue else []
                        agent.message_queue = None
                        if leftover:
                            first_prompt = '\n\n'.join(leftover)
                result_printer(result if shown is None else replace(result, final_output=shown),
                               show_transcript=show_transcript or agent_bar.verbose)  # Ctrl+O
                _print_turn_done(output_func, time.monotonic() - run_started)
                agent_bar.mode = getattr(agent, 'permission_mode', agent_bar.mode) or agent_bar.mode
                edited_now = len(result.file_history or ()) > files_before_run
                if first_prompt is None and edited_now and not auto_test_prompt and _auto_test_enabled(agent):
                    from .agent_slash_commands import run_project_check

                    passed, command, output = run_project_check(agent, 'test')
                    if command and not passed:
                        output_func(f'\033[33m\u21b3 autoTest: {command} failed; sending the output to the agent\033[0m')
                        first_prompt = (f'`{command}` failed after your changes. Fix it.\n\n'
                                        f'<test-output>\n{output[-12000:]}\n</test-output>')
                        next_is_auto_test = True
                if goal_state and first_prompt is None:
                    if GOAL_DONE in (result.final_output or ''):
                        output_func(f'\033[32m\u2713 Goal met: {goal_state["text"]}\033[0m')
                        goal_state.clear()
                    elif goal_state['rounds'] <= 1 or result.stop_reason == 'interrupted':
                        output_func(f'\033[33mGoal not met after the last round; /goal {goal_state["text"]} to go on.\033[0m')
                        goal_state.clear()
                    else:
                        goal_state['rounds'] -= 1
                        first_prompt = goal_prompt(goal_state['text'], False)
                if getattr(agent, '_session_name', None) and result.session_id:
                    from .session_store import save_session_name

                    save_session_name(agent.runtime_config.session_directory, agent._session_name, result.session_id)
                if first_prompt is not None:
                    output_func(f'\033[2m  ↳ sending queued message: {first_prompt}\033[0m')
                active_session_id = result.session_id

                if result.stop_reason == 'interrupted':
                    output_func('\n[!] Query halted. (Press Ctrl+C at the prompt to exit session)')
                    continue

                # [AUTO-COMPACT] Check if current context exceeds threshold (not cumulative)
                threshold = agent.runtime_config.auto_compact_threshold_tokens
                context_tokens = result.last_turn_input_tokens
                if threshold is not None and threshold > 0 and context_tokens > 0 and context_tokens > threshold:
                    output_func(f'\033[2m🗜️ Auto-compacting session (context {context_tokens:,} > limit {threshold:,})...\033[0m')
                    try:
                        compact_session = load_agent_session(
                            active_session_id,
                            directory=agent.runtime_config.session_directory,
                        )
                        agent.compact(compact_session)
                        output_func('\033[32m✓ Auto-compaction complete.\033[0m')
                    except Exception as e:
                        output_func(f'\033[31mError during auto-compaction: {e}\033[0m')
            except KeyboardInterrupt:
                output_func('\nchat_ended=interrupt')
                return 130
            except Exception:
                from .debug_log import get_logger

                get_logger('chat').exception('error in agent session %s', active_session_id)
                output_func(f'\n\033[1;31m[!] CRITICAL ERROR in agent session\033[0m')
                output_func(f'\033[91m{traceback.format_exc()}\033[0m')
                output_func(f'\nSession: \033[33m{active_session_id}\033[0m')
                output_func('The PowerShell window was kept open so you can see this error. Please report it.')
                # We don't return here so the loop continues (or user can /exit)
                continue
    finally:
        from .background_shells import shell_manager
        from .command_hooks import fire_session_end

        shell_manager().kill_all()  # background shells stop with the session
        try:
            from .mcp_runtime import connection_manager

            connection_manager().close_all()  # and so do MCP server processes
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
        try:
            from .mcp_runtime import set_elicitation_handler

            set_elicitation_handler(None)  # back to declining once the chat ends
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
        try:
            scheduler_stop.set()
        except NameError as _ignored:
            _log_ignored(_ignored)
            pass
        fire_session_end(getattr(agent, 'command_hook_runtime', None), active_session_id, end_reason)
        if tui:
            tui.stop()
        # Cleanup handled manually via /cleanup now to allow tool persistence
        # agent.cleanup_custom_tools()


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version('sigilcode')
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    try:
        text = (Path(__file__).resolve().parent.parent / 'pyproject.toml').read_text(encoding='utf-8')
        import re

        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        return match.group(1) if match else 'unknown'
    except OSError as _ignored:
        _log_ignored(_ignored)
        return 'unknown'


_CLI_NAME = 'sigil'


def _completion_script(parser: argparse.ArgumentParser, shell: str) -> str:
    """`sigil completion bash|zsh|powershell`: subcommands and top-level flags."""
    words: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            words += [name for name in action.choices if name != 'agent-bg-worker']
        else:
            words += [opt for opt in action.option_strings if opt.startswith('--')]
    joined = ' '.join(sorted(set(words)))
    if shell == 'bash':
        return f'complete -W "{joined}" {_CLI_NAME}\n'
    if shell == 'zsh':
        return f'#compdef {_CLI_NAME}\ncompadd {joined}\n'
    items = ', '.join(f"'{word}'" for word in sorted(set(words)))
    return (f'Register-ArgumentCompleter -Native -CommandName {_CLI_NAME} -ScriptBlock {{\n'
            '    param($wordToComplete)\n'
            f'    @({items}) | Where-Object {{ $_ -like "$wordToComplete*" }} |\n'
            '        ForEach-Object { [System.Management.Automation.CompletionResult]::new($_) }\n'
            '}\n')


def _run_init_only(args: argparse.Namespace) -> int:
    """--init-only: run the Setup hooks (trigger "init") and exit."""
    agent = _build_agent(args)
    hooks = agent.command_hook_runtime
    if hooks is None or not hooks.has('Setup'):
        print('No Setup hooks configured.')
        return 0
    outcome = hooks.run('Setup', {'trigger': 'init', 'cwd': str(agent.runtime_config.cwd)})
    text = outcome.context_text() if hasattr(outcome, 'context_text') else ''
    if text:
        print(text)
    if getattr(outcome, 'blocked', False) or getattr(outcome, 'stop', False):
        print(f'Setup hook stopped: {outcome.reason or outcome.stop_reason}')
        return 1
    print('Setup hooks done.')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sigil Code: an open-source Python coding agent')
    parser.add_argument('--version', action='version', version=f'Sigil Code {_package_version()}')
    # Shortcuts: bare `sigil` starts a chat, `sigil -r`
    # resumes the latest session. Distinct dests so the agent-chat
    # subparser's own --resume/--resume-session-id never clobber these.
    parser.add_argument('-r', '--resume', dest='top_resume', action='store_true',
                        help='resume the latest session (shortcut for agent-chat --resume)')
    parser.add_argument('--resume-session-id', dest='top_resume_session_id', default=None,
                        help='resume a specific session by ID (shortcut for agent-chat --resume-session-id)')
    parser.add_argument('-c', '--continue', dest='top_continue', action='store_true',
                        help='continue the latest session in this folder (also with -p)')
    parser.add_argument('-p', '--print', dest='top_print', nargs='?', const='', default=None,
                        metavar='PROMPT',
                        help='print mode: answer once and exit (reads piped stdin); same flags as agent')
    parser.add_argument('--output-format', dest='top_output_format', default=None,
                        choices=['text', 'json', 'stream-json'],
                        help='print mode output: text (default), json or stream-json')
    subparsers = parser.add_subparsers(dest='command', required=False)
    subparsers.add_parser('summary', help='render a Markdown summary of the Sigil Code workspace')
    subparsers.add_parser('manifest', help='print the current Python workspace manifest')
    subparsers.add_parser('parity-audit', help='print which documented commands, hook events and tools Sigil covers (to stdout)')
    subparsers.add_parser('setup-report', help='render the startup/prefetch setup report')
    subparsers.add_parser('command-graph', help='show command graph segmentation')
    subparsers.add_parser('tool-pool', help='show assembled tool pool with default settings')
    subparsers.add_parser('bootstrap-graph', help='show the mirrored bootstrap/runtime graph stages')

    list_parser = subparsers.add_parser('subsystems', help='list the current Python modules in the workspace')
    list_parser.add_argument('--limit', type=int, default=32)

    commands_parser = subparsers.add_parser('commands', help='list mirrored command entries from the archived snapshot')
    commands_parser.add_argument('--limit', type=int, default=20)
    commands_parser.add_argument('--query')
    commands_parser.add_argument('--no-plugin-commands', action='store_true')
    commands_parser.add_argument('--no-skill-commands', action='store_true')

    tools_parser = subparsers.add_parser('tools', help='list mirrored tool entries from the archived snapshot')
    tools_parser.add_argument('--limit', type=int, default=20)
    tools_parser.add_argument('--query')
    tools_parser.add_argument('--simple-mode', action='store_true')
    tools_parser.add_argument('--no-mcp', action='store_true')
    tools_parser.add_argument('--deny-tool', action='append', default=[])
    tools_parser.add_argument('--deny-prefix', action='append', default=[])

    route_parser = subparsers.add_parser('route', help='route a prompt across mirrored command/tool inventories')
    route_parser.add_argument('prompt')
    route_parser.add_argument('--limit', type=int, default=5)

    bootstrap_parser = subparsers.add_parser('bootstrap', help='build a runtime-style session report from the mirrored inventories')
    bootstrap_parser.add_argument('prompt')
    bootstrap_parser.add_argument('--limit', type=int, default=5)

    loop_parser = subparsers.add_parser('turn-loop', help='run a small stateful turn loop for the mirrored runtime')
    loop_parser.add_argument('prompt')
    loop_parser.add_argument('--limit', type=int, default=5)
    loop_parser.add_argument('--max-turns', type=int, default=3)
    loop_parser.add_argument('--structured-output', action='store_true')

    flush_parser = subparsers.add_parser('flush-transcript', help='persist and flush a temporary session transcript')
    flush_parser.add_argument('prompt')

    load_session_parser = subparsers.add_parser('load-session', help='load a previously persisted session')
    load_session_parser.add_argument('session_id')

    remote_parser = subparsers.add_parser('remote-mode', help='simulate remote-control runtime branching')
    remote_parser.add_argument('target')
    remote_parser.add_argument('--cwd', default='.')
    ssh_parser = subparsers.add_parser('ssh-mode', help='simulate SSH runtime branching')
    ssh_parser.add_argument('target')
    ssh_parser.add_argument('--cwd', default='.')
    teleport_parser = subparsers.add_parser('teleport-mode', help='simulate teleport runtime branching')
    teleport_parser.add_argument('target')
    teleport_parser.add_argument('--cwd', default='.')
    direct_parser = subparsers.add_parser('direct-connect-mode', help='simulate direct-connect runtime branching')
    direct_parser.add_argument('target')
    direct_parser.add_argument('--cwd', default='.')
    deep_link_parser = subparsers.add_parser('deep-link-mode', help='simulate deep-link runtime branching')
    deep_link_parser.add_argument('target')
    deep_link_parser.add_argument('--cwd', default='.')
    remote_status_parser = subparsers.add_parser('remote-status', help='show local remote runtime status')
    remote_status_parser.add_argument('--cwd', default='.')
    remote_profiles_parser = subparsers.add_parser('remote-profiles', help='list configured local remote profiles')
    remote_profiles_parser.add_argument('--cwd', default='.')
    remote_profiles_parser.add_argument('--query')
    remote_disconnect_parser = subparsers.add_parser('remote-disconnect', help='disconnect the active local remote target')
    remote_disconnect_parser.add_argument('--cwd', default='.')
    worktree_status_parser = subparsers.add_parser('worktree-status', help='show local managed git worktree status')
    worktree_status_parser.add_argument('--cwd', default='.')
    worktree_enter_parser = subparsers.add_parser('worktree-enter', help='create and enter a managed git worktree')
    worktree_enter_parser.add_argument('name', nargs='?')
    worktree_enter_parser.add_argument('--cwd', default='.')
    worktree_exit_parser = subparsers.add_parser('worktree-exit', help='exit the active managed git worktree')
    worktree_exit_parser.add_argument('--action', default='keep')
    worktree_exit_parser.add_argument('--discard-changes', action='store_true')
    worktree_exit_parser.add_argument('--cwd', default='.')
    account_status_parser = subparsers.add_parser('account-status', help='show local account runtime status')
    account_status_parser.add_argument('--cwd', default='.')
    account_profiles_parser = subparsers.add_parser('account-profiles', help='list configured local account profiles')
    account_profiles_parser.add_argument('--cwd', default='.')
    account_profiles_parser.add_argument('--query')
    account_login_parser = subparsers.add_parser('account-login', help='activate a local account profile or ephemeral identity')
    account_login_parser.add_argument('target')
    account_login_parser.add_argument('--provider')
    account_login_parser.add_argument('--auth-mode')
    account_login_parser.add_argument('--cwd', default='.')
    account_logout_parser = subparsers.add_parser('account-logout', help='clear the active local account session')
    account_logout_parser.add_argument('--cwd', default='.')
    ask_status_parser = subparsers.add_parser('ask-status', help='show local ask-user runtime status')
    ask_status_parser.add_argument('--cwd', default='.')
    ask_history_parser = subparsers.add_parser('ask-history', help='show local ask-user interaction history')
    ask_history_parser.add_argument('--cwd', default='.')
    search_status_parser = subparsers.add_parser('search-status', help='show local search runtime status')
    search_status_parser.add_argument('--cwd', default='.')
    search_status_parser.add_argument('--provider')
    search_providers_parser = subparsers.add_parser('search-providers', help='list configured local search providers')
    search_providers_parser.add_argument('--cwd', default='.')
    search_providers_parser.add_argument('--query')
    search_activate_parser = subparsers.add_parser('search-activate', help='set the active local search provider')
    search_activate_parser.add_argument('provider')
    search_activate_parser.add_argument('--cwd', default='.')
    search_parser = subparsers.add_parser('search', help='run a real web search against the configured local search runtime')
    search_parser.add_argument('query')
    search_parser.add_argument('--cwd', default='.')
    search_parser.add_argument('--provider')
    search_parser.add_argument('--max-results', type=int, default=5)
    search_parser.add_argument('--domain', action='append', default=[])
    mcp_parser = subparsers.add_parser('mcp', help='manage MCP servers: add, list, remove, get, login, logout')
    mcp_sub = mcp_parser.add_subparsers(dest='mcp_command', required=True)
    mcp_add = mcp_sub.add_parser('add', help='add a server: mcp add <name> <command> [args...] or --url')
    mcp_add.add_argument('name')
    mcp_add.add_argument('server_command', nargs='?', default=None)
    mcp_add.add_argument('server_args', nargs='*', default=[])
    mcp_add.add_argument('--transport', choices=['stdio', 'http', 'sse'], default=None)
    mcp_add.add_argument('--url', default=None)
    mcp_add.add_argument('--env', action='append', default=[], help='KEY=VALUE (repeatable)')
    mcp_add.add_argument('--header', action='append', default=[], help='Name: value (repeatable)')
    mcp_add.add_argument('--scope', choices=['project', 'user'], default='project')
    mcp_add.add_argument('--cwd', default='.')
    for verb in ('list', 'get', 'remove', 'login', 'logout'):
        sub = mcp_sub.add_parser(verb)
        if verb != 'list':
            sub.add_argument('name')
        sub.add_argument('--scope', choices=['project', 'user'], default=None)
        sub.add_argument('--cwd', default='.')

    mcp_status_parser = subparsers.add_parser('mcp-status', help='show local MCP runtime status')
    mcp_status_parser.add_argument('--cwd', default='.')
    mcp_resources_parser = subparsers.add_parser('mcp-resources', help='list MCP resources discovered through local manifests and transport-backed servers')
    mcp_resources_parser.add_argument('--cwd', default='.')
    mcp_resources_parser.add_argument('--query')
    mcp_resource_parser = subparsers.add_parser('mcp-resource', help='read an MCP resource by URI')
    mcp_resource_parser.add_argument('uri')
    mcp_resource_parser.add_argument('--cwd', default='.')
    mcp_tools_parser = subparsers.add_parser('mcp-tools', help='list MCP tools exposed by configured MCP servers')
    mcp_tools_parser.add_argument('--cwd', default='.')
    mcp_tools_parser.add_argument('--query')
    mcp_tools_parser.add_argument('--server')
    mcp_call_tool_parser = subparsers.add_parser('mcp-call-tool', help='call an MCP tool exposed by a configured MCP server')
    mcp_call_tool_parser.add_argument('tool_name')
    mcp_call_tool_parser.add_argument('--arguments-json', default='{}')
    mcp_call_tool_parser.add_argument('--server')
    mcp_call_tool_parser.add_argument('--cwd', default='.')
    config_status_parser = subparsers.add_parser('config-status', help='show local workspace config runtime summary')
    config_status_parser.add_argument('--cwd', default='.')
    config_effective_parser = subparsers.add_parser('config-effective', help='render the merged effective local workspace config')
    config_effective_parser.add_argument('--cwd', default='.')
    config_source_parser = subparsers.add_parser('config-source', help='render a specific local config source')
    config_source_parser.add_argument('source')
    config_source_parser.add_argument('--cwd', default='.')
    config_get_parser = subparsers.add_parser('config-get', help='read a local config value by dotted key path')
    config_get_parser.add_argument('key_path')
    config_get_parser.add_argument('--source')
    config_get_parser.add_argument('--cwd', default='.')
    config_set_parser = subparsers.add_parser('config-set', help='write a local config value by dotted key path')
    config_set_parser.add_argument('key_path')
    config_set_parser.add_argument('value_json')
    config_set_parser.add_argument('--source', default='local')
    config_set_parser.add_argument('--cwd', default='.')
    lsp_status_parser = subparsers.add_parser('lsp-status', help='show local LSP runtime summary')
    lsp_status_parser.add_argument('--cwd', default='.')
    lsp_symbols_parser = subparsers.add_parser('lsp-symbols', help='show local LSP document symbols for one file')
    lsp_symbols_parser.add_argument('file_path')
    lsp_symbols_parser.add_argument('--cwd', default='.')
    lsp_workspace_parser = subparsers.add_parser('lsp-workspace-symbols', help='search workspace symbols through the local LSP runtime')
    lsp_workspace_parser.add_argument('query')
    lsp_workspace_parser.add_argument('--max-results', type=int, default=50)
    lsp_workspace_parser.add_argument('--cwd', default='.')
    lsp_definition_parser = subparsers.add_parser('lsp-definition', help='run a local LSP definition query')
    lsp_definition_parser.add_argument('file_path')
    lsp_definition_parser.add_argument('line', type=int)
    lsp_definition_parser.add_argument('character', type=int)
    lsp_definition_parser.add_argument('--max-results', type=int, default=20)
    lsp_definition_parser.add_argument('--cwd', default='.')
    lsp_references_parser = subparsers.add_parser('lsp-references', help='run a local LSP references query')
    lsp_references_parser.add_argument('file_path')
    lsp_references_parser.add_argument('line', type=int)
    lsp_references_parser.add_argument('character', type=int)
    lsp_references_parser.add_argument('--max-results', type=int, default=50)
    lsp_references_parser.add_argument('--cwd', default='.')
    lsp_hover_parser = subparsers.add_parser('lsp-hover', help='run a local LSP hover query')
    lsp_hover_parser.add_argument('file_path')
    lsp_hover_parser.add_argument('line', type=int)
    lsp_hover_parser.add_argument('character', type=int)
    lsp_hover_parser.add_argument('--cwd', default='.')
    lsp_diagnostics_parser = subparsers.add_parser('lsp-diagnostics', help='show local LSP diagnostics')
    lsp_diagnostics_parser.add_argument('--file-path')
    lsp_diagnostics_parser.add_argument('--cwd', default='.')
    lsp_hierarchy_parser = subparsers.add_parser('lsp-call-hierarchy', help='show local LSP call hierarchy at a position')
    lsp_hierarchy_parser.add_argument('file_path')
    lsp_hierarchy_parser.add_argument('line', type=int)
    lsp_hierarchy_parser.add_argument('character', type=int)
    lsp_hierarchy_parser.add_argument('--cwd', default='.')
    lsp_incoming_parser = subparsers.add_parser('lsp-incoming-calls', help='show local LSP incoming calls at a position')
    lsp_incoming_parser.add_argument('file_path')
    lsp_incoming_parser.add_argument('line', type=int)
    lsp_incoming_parser.add_argument('character', type=int)
    lsp_incoming_parser.add_argument('--max-results', type=int, default=50)
    lsp_incoming_parser.add_argument('--cwd', default='.')
    lsp_outgoing_parser = subparsers.add_parser('lsp-outgoing-calls', help='show local LSP outgoing calls at a position')
    lsp_outgoing_parser.add_argument('file_path')
    lsp_outgoing_parser.add_argument('line', type=int)
    lsp_outgoing_parser.add_argument('character', type=int)
    lsp_outgoing_parser.add_argument('--max-results', type=int, default=50)
    lsp_outgoing_parser.add_argument('--cwd', default='.')
    workflow_list_parser = subparsers.add_parser('workflow-list', help='list local workflow definitions')
    workflow_list_parser.add_argument('--cwd', default='.')
    workflow_list_parser.add_argument('--query')
    workflow_get_parser = subparsers.add_parser('workflow-get', help='show one local workflow definition')
    workflow_get_parser.add_argument('workflow_name')
    workflow_get_parser.add_argument('--cwd', default='.')
    workflow_run_parser = subparsers.add_parser('workflow-run', help='record and render a local workflow run')
    workflow_run_parser.add_argument('workflow_name')
    workflow_run_parser.add_argument('--arguments-json', default='{}')
    workflow_run_parser.add_argument('--cwd', default='.')
    trigger_list_parser = subparsers.add_parser('trigger-list', help='list local remote triggers')
    trigger_list_parser.add_argument('--cwd', default='.')
    trigger_list_parser.add_argument('--query')
    trigger_get_parser = subparsers.add_parser('trigger-get', help='show one local remote trigger')
    trigger_get_parser.add_argument('trigger_id')
    trigger_get_parser.add_argument('--cwd', default='.')
    trigger_create_parser = subparsers.add_parser('trigger-create', help='create a local remote trigger')
    trigger_create_parser.add_argument('--body-json', required=True)
    trigger_create_parser.add_argument('--cwd', default='.')
    trigger_update_parser = subparsers.add_parser('trigger-update', help='update a local remote trigger')
    trigger_update_parser.add_argument('trigger_id')
    trigger_update_parser.add_argument('--body-json', required=True)
    trigger_update_parser.add_argument('--cwd', default='.')
    trigger_run_parser = subparsers.add_parser('trigger-run', help='run a local remote trigger')
    trigger_run_parser.add_argument('trigger_id')
    trigger_run_parser.add_argument('--body-json', default='{}')
    trigger_run_parser.add_argument('--cwd', default='.')
    teams_status_parser = subparsers.add_parser('team-status', help='show local collaboration team runtime summary')
    teams_status_parser.add_argument('--cwd', default='.')
    teams_list_parser = subparsers.add_parser('team-list', help='list local collaboration teams')
    teams_list_parser.add_argument('--cwd', default='.')
    teams_list_parser.add_argument('--query')
    team_get_parser = subparsers.add_parser('team-get', help='show one local collaboration team')
    team_get_parser.add_argument('team_name')
    team_get_parser.add_argument('--cwd', default='.')
    team_create_parser = subparsers.add_parser('team-create', help='create a local collaboration team')
    team_create_parser.add_argument('team_name')
    team_create_parser.add_argument('--description')
    team_create_parser.add_argument('--member', action='append', default=[])
    team_create_parser.add_argument('--cwd', default='.')
    team_delete_parser = subparsers.add_parser('team-delete', help='delete a local collaboration team')
    team_delete_parser.add_argument('team_name')
    team_delete_parser.add_argument('--cwd', default='.')
    team_messages_parser = subparsers.add_parser('team-messages', help='show local team messages')
    team_messages_parser.add_argument('--team-name')
    team_messages_parser.add_argument('--cwd', default='.')

    show_command = subparsers.add_parser('show-command', help='show one mirrored command entry by exact name')
    show_command.add_argument('name')
    show_tool = subparsers.add_parser('show-tool', help='show one mirrored tool entry by exact name')
    show_tool.add_argument('name')

    exec_command_parser = subparsers.add_parser('exec-command', help='execute a mirrored command shim by exact name')
    exec_command_parser.add_argument('name')
    exec_command_parser.add_argument('prompt')

    exec_tool_parser = subparsers.add_parser('exec-tool', help='execute a mirrored tool shim by exact name')
    exec_tool_parser.add_argument('name')
    exec_tool_parser.add_argument('payload')

    agent_parser = subparsers.add_parser('agent', help='run the real Python local-model agent')
    agent_parser.add_argument('prompt')
    agent_parser.add_argument('--max-turns', type=int, default=12)
    agent_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(agent_parser, include_backend=True)

    background_parser = subparsers.add_parser('agent-bg', help='run the Python local-model agent as a local background session')
    background_parser.add_argument('prompt')
    background_parser.add_argument('--max-turns', type=int, default=12)
    background_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(background_parser, include_backend=True)

    background_worker_parser = subparsers.add_parser('agent-bg-worker', help=argparse.SUPPRESS)
    background_worker_parser.add_argument('background_id')
    background_worker_parser.add_argument('prompt')
    background_worker_parser.add_argument('--background-root', required=True)
    background_worker_parser.add_argument('--resume-session', action='store_true', default=False)
    background_worker_parser.add_argument('--max-turns', type=int, default=12)
    background_worker_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(background_worker_parser, include_backend=True)

    ps_parser = subparsers.add_parser('agent-ps', help='list local background agent sessions')
    ps_parser.add_argument('--cwd', dest='background_cwd', default=None, help='workspace whose background agents to use (default: this folder)')
    ps_parser.add_argument('--tail', type=int, default=None)

    logs_parser = subparsers.add_parser('agent-logs', help='show logs for a local background agent session')
    logs_parser.add_argument('--cwd', dest='background_cwd', default=None, help='workspace whose background agents to use (default: this folder)')
    logs_parser.add_argument('background_id')
    logs_parser.add_argument('--tail', type=int, default=None)

    attach_parser = subparsers.add_parser('agent-attach', help='show the current output snapshot for a local background agent session')
    attach_parser.add_argument('--cwd', dest='background_cwd', default=None, help='workspace whose background agents to use (default: this folder)')
    attach_parser.add_argument('background_id')
    attach_parser.add_argument('--tail', type=int, default=None)

    kill_parser = subparsers.add_parser('agent-kill', help='stop a local background agent session')
    kill_parser.add_argument('--cwd', dest='background_cwd', default=None, help='workspace whose background agents to use (default: this folder)')
    kill_parser.add_argument('background_id')

    approve_parser = subparsers.add_parser('agent-approve',
                                           help="answer a background agent's approval question")
    approve_parser.add_argument('--cwd', dest='background_cwd', default=None, help='workspace whose background agents to use (default: this folder)')
    approve_parser.add_argument('background_id', nargs='?', default=None)
    approve_parser.add_argument('answer', nargs='?', default=None, help='once, session, always or deny')
    restart_parser = subparsers.add_parser('agent-restart', help='start a stopped background agent again')
    restart_parser.add_argument('--cwd', dest='background_cwd', default=None, help='workspace whose background agents to use (default: this folder)')
    restart_parser.add_argument('background_id')
    restart_parser.add_argument('--prompt', default=None, help='new instructions (default: continue)')

    daemon_parser = subparsers.add_parser('daemon', help='manage local daemon-style background agent sessions')
    daemon_subparsers = daemon_parser.add_subparsers(dest='daemon_command')
    daemon_subparsers.required = True

    daemon_start_parser = daemon_subparsers.add_parser('start', help='launch a local daemon-style background agent session')
    daemon_start_parser.add_argument('prompt')
    daemon_start_parser.add_argument('--max-turns', type=int, default=12)
    daemon_start_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(daemon_start_parser, include_backend=True)

    daemon_worker_parser = daemon_subparsers.add_parser('worker', help=argparse.SUPPRESS)
    daemon_worker_parser.add_argument('background_id')
    daemon_worker_parser.add_argument('prompt')
    daemon_worker_parser.add_argument('--background-root', required=True)
    daemon_worker_parser.add_argument('--resume-session', action='store_true', default=False)
    daemon_worker_parser.add_argument('--max-turns', type=int, default=12)
    daemon_worker_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(daemon_worker_parser, include_backend=True)

    daemon_ps_parser = daemon_subparsers.add_parser('ps', help='list local daemon-style background sessions')
    daemon_ps_parser.add_argument('--tail', type=int, default=None)

    daemon_logs_parser = daemon_subparsers.add_parser('logs', help='show logs for a local daemon-style background session')
    daemon_logs_parser.add_argument('background_id')
    daemon_logs_parser.add_argument('--tail', type=int, default=None)

    daemon_attach_parser = daemon_subparsers.add_parser('attach', help='show the current output snapshot for a local daemon-style background session')
    daemon_attach_parser.add_argument('background_id')
    daemon_attach_parser.add_argument('--tail', type=int, default=None)

    daemon_kill_parser = daemon_subparsers.add_parser('kill', help='stop a local daemon-style background session')
    daemon_kill_parser.add_argument('background_id')

    daemon_restart_parser = daemon_subparsers.add_parser('restart', help='start a stopped background session again')
    daemon_restart_parser.add_argument('background_id')
    daemon_restart_parser.add_argument('--prompt', default=None)

    chat_parser = subparsers.add_parser('agent-chat', help='run an interactive Python local-model chat loop')
    chat_parser.add_argument('prompt', nargs='?')
    chat_parser.add_argument('--resume', action='store_true', help='Resume the latest session')
    chat_parser.add_argument('--resume-session-id')
    chat_parser.add_argument('--tui', action='store_true', help='Use Command Center TUI')
    # 0 = no turn cap in chat; the budget flags still apply.
    chat_parser.add_argument('--max-turns', type=int, default=0)
    chat_parser.add_argument('--show-transcript', action='store_true')
    _add_agent_common_args(chat_parser, include_backend=True)

    doctor_parser = subparsers.add_parser('doctor', help='check the installation, model and settings')
    _add_agent_common_args(doctor_parser, include_backend=True)
    trust_parser = subparsers.add_parser(
        'trust', help="show, grant, skip or revoke trust for this folder's project hooks, MCP servers and env")
    trust_parser.add_argument('action', nargs='?', default='status', choices=['status', 'grant', 'skip', 'revoke'])
    trust_parser.add_argument('--cwd', default='.')
    completion_parser = subparsers.add_parser('completion', help='print a shell completion script')
    completion_parser.add_argument('shell', choices=['bash', 'zsh', 'powershell'])

    print_parser = subparsers.add_parser('print', help='print mode (same as -p): answer once and exit')
    print_parser.add_argument('prompt', nargs='?', default=None)
    print_parser.add_argument('--output-format', choices=['text', 'json', 'stream-json'], default='text')
    print_parser.add_argument('--input-format', choices=['text', 'stream-json'], default='text',
                              help='stream-json: read one {"type": "user", ...} message per stdin line')
    print_parser.add_argument('--max-turns', type=int, default=0)
    print_parser.add_argument('--show-transcript', action='store_true')
    print_parser.add_argument('-c', '--continue', dest='continue_session', action='store_true',
                              help='continue the latest session in --cwd')
    _add_agent_common_args(print_parser, include_backend=True)

    resume_parser = subparsers.add_parser('agent-resume', help='resume a saved Python local-model agent session')
    _add_agent_resume_args(resume_parser)

    sessions_parser = subparsers.add_parser('sessions', help='list saved sessions and pick one to resume')
    sessions_parser.add_argument('--max-turns', type=int, default=0)
    sessions_parser.add_argument('--search', default=None, help='only sessions whose messages contain this text')
    plugin_parser = subparsers.add_parser('plugin', help='plugin marketplaces: marketplace add|list, install, list')
    plugin_parser.add_argument('words', nargs='*')
    purge_parser = subparsers.add_parser('purge', help='delete old saved sessions of a workspace')
    purge_parser.add_argument('--days', type=float, default=None,
                              help='older than N days (default: cleanupPeriodDays or 30)')
    purge_parser.add_argument('--all', action='store_true', help='delete every saved session')
    purge_parser.add_argument('--dry-run', action='store_true', help='only list what would be deleted')
    purge_parser.add_argument('--cwd', default='.')
    _add_agent_common_args(sessions_parser, include_backend=True)

    prompt_parser = subparsers.add_parser('agent-prompt', help='render the Python agent system prompt')
    _add_agent_common_args(prompt_parser, include_backend=False)

    context_parser = subparsers.add_parser('agent-context', help='render Python /context-style usage accounting')
    _add_agent_common_args(context_parser, include_backend=False)

    context_raw_parser = subparsers.add_parser('agent-context-raw', help='render the raw Python agent context snapshot')
    _add_agent_common_args(context_raw_parser, include_backend=False)

    token_budget_parser = subparsers.add_parser('token-budget', help='render the current token budget and prompt-length limits')
    _add_agent_common_args(token_budget_parser, include_backend=False)
    return parser


def _enable_windows_vt() -> None:
    """Turn on VT processing so ANSI escape codes (colors, cursor hide/show)
    actually work on Windows consoles. Without this, ESC[?25l and friends are
    printed literally or ignored on conhost, which leaves the cursor visible
    while the typed line is cleared and redrawn (reads as the line bouncing)."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return  # not a console (piped output): nothing to enable
        if mode.value & 0x4:  # ENABLE_VIRTUAL_TERMINAL_PROCESSING already on
            return
        kernel32.SetConsoleMode(handle, mode.value | 0x4)
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)  # best-effort: colors/cursor codes simply stay inert


def _run_session_picker(args: argparse.Namespace) -> int:
    """List saved sessions (newest first) and resume the one the user picks."""
    from .session_store import list_agent_sessions

    session_dir = _session_directory_for(args.cwd)
    sessions = list_agent_sessions(session_dir)
    if getattr(args, 'search', None):
        from .session_store import search_agent_sessions

        sessions = [(sid, mtime, count) for sid, mtime, count, _ in search_agent_sessions(session_dir, args.search)]
        if not sessions:
            print(f'No saved session mentions {args.search!r}.')
            return 0
    if not sessions:
        print('No saved sessions found.')
        return 0
    print('Saved sessions (newest first):')
    for index, (session_id, mtime, message_count) in enumerate(sessions):
        stamp = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        from .session_store import session_first_prompt

        first = session_first_prompt(session_dir, session_id)
        print(f'  [{index}] {session_id}{_bg_mark(session_id)}  ({stamp}, {message_count} messages)  {first}')
    print()
    print('  [N] Start a NEW session')
    choice = input('Select a session index or paste a Session ID (default 0): ').strip()
    if not choice:
        choice = '0'
    if choice.lower() == 'n':
        resume_id = None
    elif choice.isdigit() and int(choice) < len(sessions):
        resume_id = sessions[int(choice)][0]
    else:
        resume_id = choice
    agent = _build_agent(args)
    return _run_agent_chat_loop(
        agent,
        initial_prompt=None,
        resume_session_id=resume_id,
        show_transcript=False,
    )


_PRINT_ERROR_STOPS = frozenset({
    'budget_exceeded', 'max_turns', 'prompt_too_long', 'internal_error', 'interrupted',
    'hook_blocked', 'hook_stopped', 'content_filter',
})


def _subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _translate_print_argv(argv: list[str], subcommands: set[str]) -> list[str]:
    """`sigil -p "q" [flags]` -> `sigil print "q" [flags]` (so every agent flag works)."""
    if not argv or argv[0] in subcommands:
        return argv
    for index, token in enumerate(argv):
        if token in subcommands:
            return argv  # a subcommand comes first; leave it alone
        if token in ('-p', '--print') or token.startswith('--print='):
            rest = argv[:index] + argv[index + 1:]
            prompt: list[str] = []
            if token.startswith('--print='):
                prompt = [token.split('=', 1)[1]]
            elif index + 1 < len(argv) and not argv[index + 1].startswith('-'):
                prompt = [argv[index + 1]]
                rest = argv[:index] + argv[index + 2:]
            return ['print', *prompt, *rest]
    return argv


_TOP_LEVEL_RESUME = {'-r', '--resume', '-c', '--continue'}


def _translate_chat_argv(argv: list[str], subcommands: set[str]) -> list[str]:
    """`sigil -w feature-x -n feature-x --model m ["prompt"]` -> `sigil agent-chat ...`, so every
    chat flag works without the subcommand. -r / -c / --continue become --resume. Left alone: an
    explicit subcommand, --help / --version, only the top-level resume flags (handled by main), and
    a first word with no flags (a mistyped subcommand stays an error)."""
    if not argv or any(token in subcommands for token in argv):
        return argv
    if any(token in ('-h', '--help', '--version') for token in argv):
        return argv
    if not argv[0].startswith('-') and ' ' not in argv[0]:
        return argv
    rest: list[str] = []
    resume = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _TOP_LEVEL_RESUME:
            resume = True
        elif token == '--resume-session-id' and index + 1 < len(argv):
            rest += [token, argv[index + 1]]
            index += 1
        elif token.startswith('--resume-session-id='):
            rest.append(token)
        else:
            rest.append(token)
        index += 1
    if all(token == '--resume-session-id' or token.startswith('--resume-session-id=') or
           (i > 0 and rest[i - 1] == '--resume-session-id') for i, token in enumerate(rest)):
        return argv  # only -r / -c / --resume-session-id: main() handles these
    return ['agent-chat', *(['--resume'] if resume else []), *rest]


def _print_mode_prompt(prompt: str | None, stdin: Any = None) -> str:
    """Piped stdin goes before the prompt (like `cat log | sigil -p "explain"`)."""
    stdin = sys.stdin if stdin is None else stdin
    piped = ''
    try:
        if stdin is not None and not stdin.isatty():
            piped = stdin.read()
    except (OSError, ValueError, AttributeError) as _ignored:
        _log_ignored(_ignored)
        piped = ''
    parts = [part for part in (piped.strip('\n'), (prompt or '').strip()) if part]
    return '\n\n'.join(parts)


def _verbose_event_printer(stream: Any) -> Callable[[dict[str, Any]], None]:
    """--verbose: one line per turn, tool call and tool result (stdout stays the answer)."""
    def _print(event: dict[str, Any]) -> None:
        kind = event.get('type')
        if kind == 'turn_start':
            line = f"[turn {event.get('turn_index', '?')}]"
        elif kind == 'tool_start':
            args_text = json.dumps(event.get('arguments', {}), default=str)[:200]
            line = f"[tool] {event.get('tool_name')} {args_text}"
        elif kind == 'tool_result':
            status = 'ok' if event.get('ok', True) else 'failed'
            line = f"[tool] {event.get('tool_name')} -> {status}"
        elif kind == 'permission_denied':
            line = f"[permission] {event.get('tool_name')}: {event.get('message')}"
        else:
            return
        try:
            stream.write(line + '\n')
            stream.flush()
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)

    return _print


def _stream_message_text(message: Any) -> str:
    """Text of a stream-json input message: {"type": "user", "message": {"content": ...}}."""
    if not isinstance(message, dict):
        return ''
    body = message.get('message', message)
    content = body.get('content') if isinstance(body, dict) else body
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
    return ''


def _run_stream_json_input(args: argparse.Namespace, stdin: Any, write: Callable[[str], None]) -> int:
    """Multi-turn over stdin: each user message line continues the same session."""
    import time
    from uuid import uuid4

    agent = _build_agent(args)
    session_id = _take_session_id(agent) or uuid4().hex
    if args.output_format == 'stream-json':
        agent.event_callback = _stream_json_events(agent, args, write)
    stored = None
    exit_code = 0
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            write(json.dumps({'type': 'error', 'error': f'not JSON: {line[:200]}'}))
            exit_code = 1
            continue
        if isinstance(message, dict) and message.get('type') not in (None, 'user'):
            continue
        text = _stream_message_text(message)
        if not text:
            continue
        if getattr(args, 'replay_user_messages', False) and args.output_format == 'stream-json':
            body = message.get('message', message) if isinstance(message, dict) else {'content': text}
            write(json.dumps({'type': 'user', 'message': body, 'session_id': session_id}, default=str))
        started = time.monotonic()
        try:
            result = agent.resume(text, stored) if stored is not None else agent.run(text, session_id=session_id)
        except Exception as exc:
            write(json.dumps({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True,
                              'result': str(exc), 'session_id': session_id}))
            return 1
        payload = _result_payload(result, int((time.monotonic() - started) * 1000))
        write(json.dumps(payload, default=str))
        if payload['is_error']:
            exit_code = 1
        stored = load_agent_session(result.session_id, directory=agent.runtime_config.session_directory)
    return exit_code


PARTIAL_EVENTS = ('content_delta', 'tool_delta', 'reasoning_delta')


def _stream_json_events(agent: Any, args: argparse.Namespace, write: Callable[[str], None]) -> Callable[[dict], None]:
    """stream-json events; partial deltas only with --include-partial-messages (which also streams)."""
    partial = bool(getattr(args, 'include_partial_messages', False))
    if partial:
        agent.runtime_config = replace(agent.runtime_config, stream_model_responses=True)

    def _emit(event: dict) -> None:
        if not partial and event.get('type') in PARTIAL_EVENTS:
            return
        write(json.dumps(event, default=str))

    return _emit


def _result_payload(result: Any, duration_ms: int) -> dict[str, Any]:
    stop = result.stop_reason or ''
    is_error = stop in _PRINT_ERROR_STOPS or stop.startswith('backend_error')
    return {
        'type': 'result',
        'subtype': 'success' if not is_error else (
            'error_max_turns' if stop == 'max_turns' else 'error_during_execution'),
        'is_error': is_error,
        'result': result.final_output,
        'session_id': result.session_id,
        'stop_reason': result.stop_reason,
        'num_turns': result.turns,
        'duration_ms': duration_ms,
        'total_cost_usd': result.total_cost_usd,
        'usage': result.usage.to_dict() if hasattr(result.usage, 'to_dict') else {},
    }


def _run_print_mode(args: argparse.Namespace, *, stdin: Any = None,
                    out: Callable[[str], None] | None = None) -> int:
    """Run one prompt and print only the answer (or JSON). Exit 0 on success."""
    import time
    from uuid import uuid4

    from .command_hooks import fire_session_end

    write = out or (lambda text: (sys.stdout.write(text + '\n'), sys.stdout.flush()))
    if getattr(args, 'input_format', 'text') == 'stream-json':
        return _run_stream_json_input(args, stdin if stdin is not None else sys.stdin, write)
    prompt = _print_mode_prompt(args.prompt, stdin)
    if not prompt:
        sys.stderr.write('print mode needs a prompt (an argument or piped stdin)\n')
        return 2
    fmt = args.output_format
    agent = _build_agent(args)
    session_id = _take_session_id(agent) or uuid4().hex
    if fmt == 'stream-json':
        write(json.dumps({
            'type': 'system', 'subtype': 'init', 'session_id': session_id,
            'model': agent.model_config.model, 'cwd': str(agent.runtime_config.cwd),
            'tools': sorted(agent.tool_registry or {}),
        }, default=str))
        agent.event_callback = _stream_json_events(agent, args, write)
    elif getattr(args, 'verbose', False):
        agent.event_callback = _verbose_event_printer(sys.stderr)
    if not prompt.lstrip().startswith('/'):
        prompt = _attach_mentions(agent, prompt)
    started = time.monotonic()
    stored = None
    if getattr(args, 'continue_session', False):
        try:
            latest = _lookup_latest_session_id(str(agent.runtime_config.cwd),
                                               session_dir=agent.runtime_config.session_directory)
            stored = load_agent_session(_maybe_fork(agent, latest), directory=agent.runtime_config.session_directory)
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            stored = None  # nothing to continue: start a new session
    try:
        result = agent.resume(prompt, stored) if stored is not None else agent.run(prompt, session_id=session_id)
    except Exception as exc:  # backend down, bad config: report and exit non-zero
        from .debug_log import get_logger

        get_logger('print').exception('print mode failed')
        if fmt == 'text':
            sys.stderr.write(f'error: {exc}\n')
        else:
            write(json.dumps({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True,
                              'result': str(exc), 'session_id': session_id}))
        return 1
    payload = _result_payload(result, int((time.monotonic() - started) * 1000))
    shown = _message_display(agent, result)
    fire_session_end(agent.command_hook_runtime, result.session_id, 'other')
    if fmt == 'text':
        write(result.final_output if shown is None else shown)
    else:
        write(json.dumps(payload, default=str))
    return 1 if payload['is_error'] else 0


def _is_interactive_chat(raw_argv: list[str]) -> bool:
    """An interactive chat on a terminal (bare `sigil` or `sigil agent-chat`, not -p / --print)."""
    if any(arg in ('-p', '--print') for arg in raw_argv):
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except (AttributeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return False
    names = _subcommand_names(build_parser())
    command = next((arg for arg in raw_argv if arg in names), None)
    return command in (None, 'agent-chat')


def main(argv: list[str] | None = None) -> int:
    _enable_windows_vt()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if _is_interactive_chat(raw_argv):
        from .workspace_trust import ensure_trust

        # asked once per folder, before any project env, hook or MCP server is used
        ensure_trust(Path(_argv_value(raw_argv, '--cwd') or '.'), ask=input, output=print)
    _apply_user_config(_argv_value(raw_argv, '--cwd') or '.')  # before the parser reads env defaults
    parser = build_parser()
    argv = _translate_print_argv(raw_argv, _subcommand_names(parser))
    argv = _translate_chat_argv(argv, _subcommand_names(parser))
    mcp_tail: list[str] | None = None
    if argv[:2] == ['mcp', 'add'] and '--' in argv:
        # Form: mcp add <name> [options] -- <command> [args...]
        cut = argv.index('--')
        mcp_tail, argv = argv[cut + 1:], argv[:cut]
    args = parser.parse_args(argv)
    if mcp_tail is not None:
        args.server_command = mcp_tail[0] if mcp_tail else None
        args.server_args = mcp_tail[1:]
    from .debug_log import get_logger, setup_logging

    setup_logging(debug=(getattr(args, 'debug', None) or getattr(args, 'verbose', False) or None),
                  debug_file=getattr(args, 'debug_file', None))
    get_logger('cli').info('start: %s', args.command or 'chat')
    manifest = build_port_manifest()

    # Entry: a bare `sigil` (optionally with -r or
    # --resume-session-id) is translated into the agent-chat subcommand so
    # every chat flag keeps working exactly as before.
    if args.command is None:
        chat_argv = ['agent-chat']
        if args.top_resume or getattr(args, 'top_continue', False):
            chat_argv.append('--resume')
        if args.top_resume_session_id:
            chat_argv += ['--resume-session-id', args.top_resume_session_id]
        args = parser.parse_args(chat_argv)

    if getattr(args, 'init_only', False):
        return _run_init_only(args)
    if args.command == 'summary':
        print(QueryEnginePort(manifest).render_summary())
        return 0
    if args.command == 'manifest':
        print(manifest.to_markdown())
        return 0
    if args.command == 'parity-audit':
        print(run_parity_audit().to_markdown())
        return 0
    if args.command == 'setup-report':
        print(run_setup().as_markdown())
        return 0
    if args.command == 'command-graph':
        print(build_command_graph().as_markdown())
        return 0
    if args.command == 'tool-pool':
        print(assemble_tool_pool().as_markdown())
        return 0
    if args.command == 'bootstrap-graph':
        print(build_bootstrap_graph().as_markdown())
        return 0
    if args.command == 'subsystems':
        for subsystem in manifest.top_level_modules[: args.limit]:
            print(f'{subsystem.name}\t{subsystem.file_count}\t{subsystem.notes}')
        return 0
    if args.command == 'commands':
        if args.query:
            print(render_command_index(limit=args.limit, query=args.query))
        else:
            commands = get_commands(
                include_plugin_commands=not args.no_plugin_commands,
                include_skill_commands=not args.no_skill_commands,
            )
            output_lines = [f'Command entries: {len(commands)}', '']
            output_lines.extend(f'- {module.name} — {module.source_hint}' for module in commands[: args.limit])
            print('\n'.join(output_lines))
        return 0
    if args.command == 'tools':
        if args.query:
            print(render_tool_index(limit=args.limit, query=args.query))
        else:
            permission_context = ToolPermissionContext.from_iterables(args.deny_tool, args.deny_prefix)
            tools = get_tools(
                simple_mode=args.simple_mode,
                include_mcp=not args.no_mcp,
                permission_context=permission_context,
            )
            output_lines = [f'Tool entries: {len(tools)}', '']
            output_lines.extend(f'- {module.name} — {module.source_hint}' for module in tools[: args.limit])
            print('\n'.join(output_lines))
        return 0
    if args.command == 'route':
        matches = PortRuntime().route_prompt(args.prompt, limit=args.limit)
        if not matches:
            print('No mirrored command/tool matches found.')
            return 0
        for match in matches:
            print(f'{match.kind}\t{match.name}\t{match.score}\t{match.source_hint}')
        return 0
    if args.command == 'bootstrap':
        print(PortRuntime().bootstrap_session(args.prompt, limit=args.limit).as_markdown())
        return 0
    if args.command == 'turn-loop':
        results = PortRuntime().run_turn_loop(
            args.prompt,
            limit=args.limit,
            max_turns=args.max_turns,
            structured_output=args.structured_output,
        )
        for idx, result in enumerate(results, start=1):
            print(f'## Turn {idx}')
            print(result.output)
            print(f'stop_reason={result.stop_reason}')
        return 0
    if args.command == 'flush-transcript':
        engine = QueryEnginePort.from_workspace()
        engine.submit_message(args.prompt)
        path = engine.persist_session()
        print(path)
        print(f'flushed={engine.transcript_store.flushed}')
        return 0
    if args.command == 'load-session':
        session = load_session(args.session_id)
        print(f'{session.session_id}\n{len(session.messages)} messages\nin={session.input_tokens} out={session.output_tokens}')
        return 0
    if args.command == 'remote-mode':
        print(run_remote_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'ssh-mode':
        print(run_ssh_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'teleport-mode':
        print(run_teleport_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'direct-connect-mode':
        print(run_direct_connect_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'deep-link-mode':
        print(run_deep_link_mode(args.target, cwd=Path(args.cwd).resolve()).as_text())
        return 0
    if args.command == 'remote-status':
        runtime = RemoteRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Remote')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'remote-profiles':
        runtime = RemoteRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_profiles_index(query=args.query))
        return 0
    if args.command == 'remote-disconnect':
        runtime = RemoteRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.disconnect().as_text())
        return 0
    if args.command == 'worktree-status':
        runtime = WorktreeRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Worktree')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'worktree-enter':
        runtime = _worktree_runtime(Path(args.cwd))
        try:
            print(runtime.enter(name=args.name).as_text())
        except (RuntimeError, ValueError) as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'worktree-exit':
        runtime = _worktree_runtime(Path(args.cwd))
        try:
            print(
                runtime.exit(
                    action=args.action,
                    discard_changes=args.discard_changes,
                ).as_text()
            )
        except (RuntimeError, ValueError) as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'account-status':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Account')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'account-profiles':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_profiles_index(query=args.query))
        return 0
    if args.command == 'account-login':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print(
            runtime.login(
                args.target,
                provider=args.provider,
                auth_mode=args.auth_mode,
            ).as_text()
        )
        return 0
    if args.command == 'account-logout':
        runtime = AccountRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.logout().as_text())
        return 0
    if args.command == 'ask-status':
        runtime = AskUserRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Ask User')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'ask-history':
        runtime = AskUserRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_history())
        return 0
    if args.command == 'search-status':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        if args.provider:
            print(runtime.render_provider(args.provider))
        else:
            print('# Search')
            print()
            print(runtime.render_summary())
        return 0
    if args.command == 'search-providers':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_providers_index(query=args.query))
        return 0
    if args.command == 'search-activate':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            report = runtime.activate_provider(args.provider)
        except KeyError:
            print(f'Unknown search provider: {args.provider}')
            return 1
        print(report.as_text())
        return 0
    if args.command == 'search':
        runtime = SearchRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            output = runtime.render_search_results(
                args.query,
                provider_name=args.provider,
                max_results=args.max_results,
                domains=tuple(args.domain),
            )
        except (KeyError, LookupError, OSError, ValueError) as exc:
            print(f'Search failed: {exc}')
            return 1
        print(output)
        return 0
    if args.command == 'mcp':
        return _run_mcp_command(args)
    if args.command == 'mcp-status':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print('# MCP')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'mcp-resources':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_resource_index(query=args.query))
        return 0
    if args.command == 'mcp-resource':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_resource(args.uri))
        return 0
    if args.command == 'mcp-tools':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_tool_index(query=args.query, server_name=args.server))
        return 0
    if args.command == 'mcp-call-tool':
        runtime = MCPRuntime.from_workspace(Path(args.cwd).resolve())
        arguments = json.loads(args.arguments_json)
        if not isinstance(arguments, dict):
            print('arguments-json must decode to a JSON object')
            return 1
        print(
            runtime.render_tool_call(
                args.tool_name,
                arguments=arguments,
                server_name=args.server,
            )
        )
        return 0
    if args.command == 'config-status':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Config')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'config-effective':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_effective_config())
        return 0
    if args.command == 'config-source':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_source(args.source))
        return 0
    if args.command == 'config-get':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_value(args.key_path, source=args.source))
        return 0
    if args.command == 'config-set':
        runtime = ConfigRuntime.from_workspace(Path(args.cwd).resolve())
        value = json.loads(args.value_json)
        mutation = runtime.set_value(args.key_path, value, source=args.source)
        print('# Config')
        print()
        print(f'source={mutation.source_name}')
        print(f'key_path={mutation.key_path}')
        print(f'store_path={mutation.store_path}')
        print(f'effective_key_count={mutation.effective_key_count}')
        print(runtime.render_value(args.key_path))
        return 0
    if args.command == 'lsp-status':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        print('# LSP')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'lsp-symbols':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_document_symbols(args.file_path))
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-workspace-symbols':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_workspace_symbols(args.query, max_results=args.max_results))
        return 0
    if args.command == 'lsp-definition':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(
                runtime.render_definition(
                    args.file_path,
                    args.line,
                    args.character,
                    max_results=args.max_results,
                )
            )
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-references':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(
                runtime.render_references(
                    args.file_path,
                    args.line,
                    args.character,
                    max_results=args.max_results,
                )
            )
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-hover':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_hover(args.file_path, args.line, args.character))
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-diagnostics':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_diagnostics(args.file_path))
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-call-hierarchy':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(
                runtime.render_prepare_call_hierarchy(
                    args.file_path,
                    args.line,
                    args.character,
                )
            )
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-incoming-calls':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(
                runtime.render_incoming_calls(
                    args.file_path,
                    args.line,
                    args.character,
                    max_results=args.max_results,
                )
            )
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'lsp-outgoing-calls':
        runtime = LSPRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(
                runtime.render_outgoing_calls(
                    args.file_path,
                    args.line,
                    args.character,
                    max_results=args.max_results,
                )
            )
        except KeyError as exc:
            print(exc)
            return 1
        return 0
    if args.command == 'workflow-list':
        runtime = WorkflowRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_workflows_index(query=args.query))
        return 0
    if args.command == 'workflow-get':
        runtime = WorkflowRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_workflow(args.workflow_name))
        except KeyError:
            print(f'Unknown workflow: {args.workflow_name}')
            return 1
        return 0
    if args.command == 'workflow-run':
        runtime = WorkflowRuntime.from_workspace(Path(args.cwd).resolve())
        arguments = json.loads(args.arguments_json)
        if not isinstance(arguments, dict):
            print('arguments-json must decode to a JSON object')
            return 1
        try:
            print(runtime.render_run_report(args.workflow_name, arguments=arguments))
        except KeyError:
            print(f'Unknown workflow: {args.workflow_name}')
            return 1
        return 0
    if args.command == 'trigger-list':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_trigger_index(query=args.query))
        return 0
    if args.command == 'trigger-get':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_trigger(args.trigger_id))
        except KeyError:
            print(f'Unknown remote trigger: {args.trigger_id}')
            return 1
        return 0
    if args.command == 'trigger-create':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        body = json.loads(args.body_json)
        if not isinstance(body, dict):
            print('body-json must decode to a JSON object')
            return 1
        try:
            trigger = runtime.create_trigger(body)
        except (KeyError, TypeError, ValueError) as exc:
            print(exc)
            return 1
        print(runtime.render_trigger(trigger.trigger_id))
        return 0
    if args.command == 'trigger-update':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        body = json.loads(args.body_json)
        if not isinstance(body, dict):
            print('body-json must decode to a JSON object')
            return 1
        try:
            trigger = runtime.update_trigger(args.trigger_id, body)
        except (KeyError, TypeError, ValueError) as exc:
            print(exc)
            return 1
        print(runtime.render_trigger(trigger.trigger_id))
        return 0
    if args.command == 'trigger-run':
        runtime = RemoteTriggerRuntime.from_workspace(Path(args.cwd).resolve())
        body = json.loads(args.body_json)
        if not isinstance(body, dict):
            print('body-json must decode to a JSON object')
            return 1
        try:
            print(runtime.render_run_report(args.trigger_id, body=body))
        except KeyError:
            print(f'Unknown remote trigger: {args.trigger_id}')
            return 1
        return 0
    if args.command == 'team-status':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        print('# Teams')
        print()
        print(runtime.render_summary())
        return 0
    if args.command == 'team-list':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        print(runtime.render_teams_index(query=args.query))
        return 0
    if args.command == 'team-get':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_team(args.team_name))
        except KeyError:
            print(f'Unknown team: {args.team_name}')
            return 1
        return 0
    if args.command == 'team-create':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            team = runtime.create_team(
                args.team_name,
                description=args.description,
                members=args.member,
            )
        except KeyError:
            print(f'Team already exists: {args.team_name}')
            return 1
        print(f'created team {team.name}')
        return 0
    if args.command == 'team-delete':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            team = runtime.delete_team(args.team_name)
        except KeyError:
            print(f'Unknown team: {args.team_name}')
            return 1
        print(f'deleted team {team.name}')
        return 0
    if args.command == 'team-messages':
        runtime = TeamRuntime.from_workspace(Path(args.cwd).resolve())
        try:
            print(runtime.render_messages(team_name=args.team_name))
        except KeyError:
            print(f'Unknown team: {args.team_name}')
            return 1
        return 0
    if args.command == 'show-command':
        module = get_command(args.name)
        if module is None:
            print(f'Command not found: {args.name}')
            return 1
        print('\n'.join([module.name, module.source_hint, module.responsibility]))
        return 0
    if args.command == 'show-tool':
        module = get_tool(args.name)
        if module is None:
            print(f'Tool not found: {args.name}')
            return 1
        print('\n'.join([module.name, module.source_hint, module.responsibility]))
        return 0
    if args.command == 'exec-command':
        result = execute_command(args.name, args.prompt)
        print(result.message)
        return 0 if result.handled else 1
    if args.command == 'exec-tool':
        result = execute_tool(args.name, args.payload)
        print(result.message)
        return 0 if result.handled else 1
    if args.command == 'agent':
        from .command_hooks import fire_session_end

        agent = _build_agent(args)
        result = agent.run(args.prompt)
        _print_agent_result(result, show_transcript=args.show_transcript)
        fire_session_end(agent.command_hook_runtime, result.session_id, 'other')
        return 0
    if args.command == 'agent-bg':
        return _launch_background_agent(args)
    if args.command == 'agent-bg-worker':
        return _run_background_worker(args)
    if args.command == 'agent-ps':
        print(_background_runtime_for_args(args).render_ps())
        return 0
    if args.command == 'agent-logs':
        print(
            _background_runtime_for_args(args).render_logs(
                args.background_id,
                tail=args.tail,
            )
        )
        return 0
    if args.command == 'agent-attach':
        print(
            _background_runtime_for_args(args).render_attach(
                args.background_id,
                tail=args.tail,
            )
        )
        return 0
    if args.command == 'agent-approve':
        from .background_approvals import answer, pending, render

        root = _background_runtime_for_args(args).root
        if not args.background_id or not args.answer:
            requests = pending(root)
            print('\n'.join(render(r) for r in requests) if requests else 'No background agent is waiting.')
            return 0
        if answer(root, args.background_id, args.answer):
            print(f'Answered {args.background_id}: {args.answer}')
            return 0
        print(f'{args.background_id} is not waiting for approval.')
        return 1
    if args.command == 'agent-restart' or (args.command == 'daemon' and args.daemon_command == 'restart'):
        return _restart_background_agent(args)
    if args.command == 'agent-kill':
        record = _background_runtime_for_args(args).kill(args.background_id)
        print('# Background Session')
        print(f'background_id={record.background_id}')
        print(f'status={record.status}')
        print(f'pid={record.pid}')
        if record.exit_code is not None:
            print(f'exit_code={record.exit_code}')
        return 0
    if args.command == 'daemon':
        if args.daemon_command == 'start':
            return _launch_background_agent(args)
        if args.daemon_command == 'worker':
            return _run_background_worker(args)
        if args.daemon_command == 'ps':
            print(_background_runtime_for_args(args).render_ps())
            return 0
        if args.daemon_command == 'logs':
            print(
                _background_runtime_for_args(args).render_logs(
                    args.background_id,
                    tail=args.tail,
                )
            )
            return 0
        if args.daemon_command == 'attach':
            print(
                _background_runtime_for_args(args).render_attach(
                    args.background_id,
                    tail=args.tail,
                )
            )
            return 0
        if args.daemon_command == 'kill':
            record = _background_runtime_for_args(args).kill(args.background_id)
            print('# Background Session')
            print(f'background_id={record.background_id}')
            print(f'status={record.status}')
            print(f'pid={record.pid}')
            if record.exit_code is not None:
                print(f'exit_code={record.exit_code}')
            return 0
    if args.command == 'purge':
        return _run_purge(args)
    if args.command == 'plugin':
        from .plugin_marketplace import command as plugin_command

        text = plugin_command(' '.join(args.words))
        print(text)
        return 1 if text.startswith('/plugin: ') else 0
    if args.command == 'doctor':
        from .agent_slash_commands import _handle_doctor

        print(_handle_doctor(_build_agent(args), '', '/doctor').output)
        return 0
    if args.command == 'trust':
        from .workspace_trust import trust_command

        print(trust_command(args.cwd, args.action))
        return 0
    if args.command == 'completion':
        print(_completion_script(parser, args.shell), end='')
        return 0
    if args.command == 'agent-chat':
        agent = _build_agent(args)
        resume_id = args.resume_session_id
        if args.resume:
            resume_id = 'latest'
        return _run_agent_chat_loop(
            agent,
            initial_prompt=args.prompt,
            resume_session_id=resume_id,
            show_transcript=args.show_transcript,
        )
    if args.command == 'sessions':
        return _run_session_picker(args)
    if args.command == 'print':
        return _run_print_mode(args)
    if args.command == 'agent-resume':
        agent, stored_session = _build_resumed_agent(args)
        result = agent.resume(args.prompt, stored_session)
        _print_agent_result(result, show_transcript=args.show_transcript)
        return 0
    if args.command == 'agent-prompt':
        agent = _build_agent(args)
        print(agent.render_system_prompt())
        return 0
    if args.command == 'agent-context':
        agent = _build_agent(args)
        print(agent.render_context_report())
        return 0
    if args.command == 'agent-context-raw':
        agent = _build_agent(args)
        print(agent.render_context_snapshot_report())
        return 0
    if args.command == 'token-budget':
        agent = _build_agent(args)
        agent.last_session = agent.build_session()
        print(agent.render_token_budget_report())
        return 0

    parser.error(f'unknown command: {args.command}')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
