from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
from .settings import write_text_atomic  # replaces a file in one step: readers never see half of it
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

if TYPE_CHECKING:
    from .agent_runtime import LocalCodingAgent


@dataclass(frozen=True)
class ParsedSlashCommand:
    command_name: str
    args: str
    is_mcp: bool


@dataclass(frozen=True)
class SlashCommandResult:
    handled: bool
    should_query: bool
    prompt: str | None = None
    output: str = ''
    transcript: tuple[dict[str, Any], ...] = ()


SlashCommandHandler = Callable[['LocalCodingAgent', str, str], SlashCommandResult]


@dataclass(frozen=True)
class SlashCommandSpec:
    names: tuple[str, ...]
    description: str
    handler: SlashCommandHandler


def parse_slash_command(input_text: str) -> ParsedSlashCommand | None:
    trimmed = input_text.strip()
    if not trimmed.startswith('/'):
        return None

    without_slash = trimmed[1:]
    words = without_slash.split(' ')
    if not words or not words[0]:
        return None

    command_name = words[0]
    is_mcp = False
    args_start_index = 1
    if len(words) > 1 and words[1] == '(MCP)':
        command_name = f'{command_name} (MCP)'
        is_mcp = True
        args_start_index = 2

    return ParsedSlashCommand(
        command_name=command_name,
        args=' '.join(words[args_start_index:]),
        is_mcp=is_mcp,
    )


def looks_like_command(command_name: str) -> bool:
    return re.search(r'[^a-zA-Z0-9:\-_]', command_name) is None


def preprocess_slash_command(
    agent: 'LocalCodingAgent',
    input_text: str,
) -> SlashCommandResult:
    import os

    if not input_text.strip().startswith('/') or os.environ.get('SIGIL_DISABLE_SLASH_COMMANDS') == '1':
        return SlashCommandResult(handled=False, should_query=True, prompt=input_text)

    parsed = parse_slash_command(input_text)
    if parsed is None:
        return _local_result(
            input_text,
            'Commands are in the form `/command [args]`.',
        )

    normalized_name = (
        parsed.command_name[:-6]
        if parsed.is_mcp and parsed.command_name.endswith(' (MCP)')
        else parsed.command_name
    )
    spec = find_slash_command(normalized_name)
    if spec is None:
        custom = _find_custom_command(agent, normalized_name)
        if custom is not None:
            return _run_custom_command(agent, custom, parsed.args.strip())
        if normalized_name.startswith('mcp__'):
            mcp_prompt = _run_mcp_prompt(agent, normalized_name, parsed.args.strip(), input_text)
            if mcp_prompt is not None:
                return mcp_prompt
        if looks_like_command(parsed.command_name):
            label = normalized_name if parsed.is_mcp else parsed.command_name
            return _local_result(input_text, f'Unknown skill: {label}')
        return SlashCommandResult(handled=False, should_query=True, prompt=input_text)

    return spec.handler(agent, parsed.args.strip(), input_text)


def _run_mcp_prompt(agent: 'LocalCodingAgent', name: str, args: str, input_text: str) -> SlashCommandResult | None:
    """/mcp__<server>__<prompt> [args]: an MCP server's prompt becomes this turn's prompt.
    Arguments are positional (in the prompt's order) or name=value."""
    import shlex

    from .agent_runtime import native_mcp_tool_name

    runtime = getattr(agent, 'mcp_runtime', None)
    if runtime is None or not getattr(runtime, 'servers', ()):
        return None
    match = next((p for p in runtime.list_prompts() if native_mcp_tool_name(p['server'], p['name']) == name), None)
    if match is None:
        return None
    try:
        words = shlex.split(args)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        words = args.split()
    names = [str(a.get('name')) for a in match['arguments']]
    values: dict[str, str] = {}
    positional = [w for w in words if '=' not in w]
    for word in words:
        if '=' in word and word.split('=', 1)[0] in names:
            key, value = word.split('=', 1)
            values[key] = value
    for key in names:
        if key not in values and positional:
            values[key] = positional.pop(0)
    missing = [str(a.get('name')) for a in match['arguments'] if a.get('required') and str(a.get('name')) not in values]
    if missing:
        return _local_result(input_text, f'/{name} needs: {", ".join(missing)}')
    try:
        text = runtime.get_prompt(match['server'], match['name'], values)
    except Exception as exc:
        return _local_result(input_text, f'MCP prompt failed: {exc}')
    return SlashCommandResult(handled=True, should_query=True, prompt=text or args)


def _find_custom_command(agent: 'LocalCodingAgent', name: str):
    """.sigil/commands, ~/.sigil/commands, or a skill with this name."""
    from .custom_commands import discover_commands

    try:
        commands = discover_commands(agent.runtime_config.cwd, getattr(agent, 'skills_runtime', None))
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None
    return commands.get(name)


def _run_custom_command(agent: 'LocalCodingAgent', command: Any, args: str) -> SlashCommandResult:
    """Expand a custom command into the prompt for this turn."""
    from .custom_commands import expand_file_mentions, run_shell_injections, substitute_arguments

    text = substitute_arguments(command.body, args)
    text = run_shell_injections(text, agent, command.allowed_tools)
    text, _attached = expand_file_mentions(text, agent)
    hook = agent._run_hook_event('UserPromptExpansion', {
        'session_id': agent.active_session_id, 'command_name': command.name, 'arguments': args,
        'expanded_prompt': text[:20000]})
    if hook is not None and (hook.blocked or hook.stop):
        return _local_result(f'/{command.name} {args}'.strip(),
                             f'Blocked by UserPromptExpansion hook: {hook.reason or hook.stop_reason}')
    if hook is not None and hook.context_text():
        text = f'{text}\n\n{hook.context_text()}'
    if getattr(command, 'fork_agent', None):
        # context: fork - run the skill in a subagent; only its summary comes back.
        result = agent._execute_delegate_agent({
            'prompt': text, 'agent': command.fork_agent, 'include_parent_context': False,
            'allow_write': True, 'allow_shell': True,
        })
        return _local_result(f'/{command.name} {args}'.strip(), result.content)
    agent.apply_turn_overrides(model=command.model, allowed_tools=command.allowed_tools)
    return SlashCommandResult(handled=True, should_query=True, prompt=text)


def get_slash_command_specs() -> tuple[SlashCommandSpec, ...]:
    return (
        SlashCommandSpec(
            names=('help', 'commands'),
            description='Show the built-in Python slash commands.',
            handler=_handle_help,
        ),
        SlashCommandSpec(
            names=('context', 'usage', 'tokens', 'token'),
            description='Show estimated session context usage .',
            handler=_handle_context,
        ),
        SlashCommandSpec(
            names=('context-raw', 'env'),
            description='Show the raw environment, user context, and system context snapshot.',
            handler=_handle_context_raw,
        ),
        SlashCommandSpec(
            names=('token-budget', 'budget'),
            description='Show the current token-budget window, reserves, and prompt-length limits.',
            handler=_handle_token_budget,
        ),
        SlashCommandSpec(
            names=('mcp',),
            description='Show discovered local MCP manifests and resource counts.',
            handler=_handle_mcp,
        ),
        SlashCommandSpec(
            names=('plugin',),
            description='Plugin marketplaces: /plugin marketplace add <git URL>, /plugin install <name>, /plugin list.',
            handler=_handle_plugin,
        ),
        SlashCommandSpec(
            names=('thinking',),
            description='Turn thinking off or on for this session (Alt+T): /thinking on|off|toggle.',
            handler=_handle_thinking,
        ),
        SlashCommandSpec(
            names=('terminal-setup',),
            description='Make Shift+Enter insert a new line in VS Code or Windows Terminal.',
            handler=_handle_terminal_setup,
        ),
        SlashCommandSpec(
            names=('subtask',),
            description='Run a task in a subagent that sees this whole conversation: /subtask <task>.',
            handler=_handle_subtask,
        ),
        SlashCommandSpec(
            names=('recap',),
            description='Ask the model for a short recap of this session.',
            handler=lambda agent, args, input_text: SlashCommandResult(handled=True, should_query=True, prompt=(
                'Give a short recap of this session: what I asked for, what was done (files changed, '
                'commands run, results), and what is still open. Use a few bullet points.'
                + (f' Focus on: {args.strip()}' if args.strip() else ''))),
        ),
        SlashCommandSpec(
            names=('cd',),
            description='Move the session to another folder: /cd <path>.',
            handler=_handle_cd,
        ),
        SlashCommandSpec(
            names=('theme',),
            description='Colours for dark or light terminals: /theme dark|light|auto.',
            handler=_handle_theme,
        ),
        SlashCommandSpec(
            names=('keybindings', 'keys'),
            description='Show the keyboard shortcuts.',
            handler=lambda agent, _args, input_text: _local_result(input_text, KEY_HELP),
        ),
        SlashCommandSpec(
            names=('bug',),
            description='Open a prefilled GitHub issue for a bug report: /bug <what went wrong>.',
            handler=_handle_bug,
        ),
        SlashCommandSpec(
            names=('focus',),
            description='Show only the last prompt, a summary of its tool calls and the answer.',
            handler=_handle_focus,
        ),
        SlashCommandSpec(
            names=('read-only',),
            description='Add a file as reference the agent may read but not edit: /read-only <file>.',
            handler=_handle_read_only,
        ),
        SlashCommandSpec(
            names=('release-notes',),
            description='Show what changed in the newest release (CHANGELOG.md); /release-notes all.',
            handler=_handle_release_notes,
        ),
        SlashCommandSpec(
            names=('reload-skills',),
            description='Read the skill folders again without a restart.',
            handler=lambda agent, _args, input_text: _local_result(
                input_text, f'Reloaded {agent.reload_skills()} skill(s).'),
        ),
        SlashCommandSpec(
            names=('reload-plugins',),
            description='Reload plugins, plugin folders, hooks, MCP servers and skills without a restart.',
            handler=lambda agent, _args, input_text: _local_result(input_text, agent.reload_plugins()),
        ),
        SlashCommandSpec(
            names=('architect',),
            description='Show whether architect mode is on and which model makes the edits.',
            handler=_handle_architect,
        ),
        SlashCommandSpec(
            names=('map',),
            description='Show the repo map: files and their main symbols, most used first.',
            handler=_handle_map,
        ),
        SlashCommandSpec(
            names=('plugins',),
            description='List the plugin folders loaded with --plugin-dir.',
            handler=_handle_plugins,
        ),
        SlashCommandSpec(
            names=('search',),
            description='Show search runtime status, list or activate providers, or run a real web search query.',
            handler=_handle_search,
        ),
        SlashCommandSpec(
            names=('remote',),
            description='Show local remote runtime status or activate a remote target/profile.',
            handler=_handle_remote,
        ),
        SlashCommandSpec(
            names=('worktree',),
            description='Show managed git worktree status or enter/exit the current managed worktree session.',
            handler=_handle_worktree,
        ),
        SlashCommandSpec(
            names=('account',),
            description='Show local account runtime status or configured account profiles.',
            handler=_handle_account,
        ),
        SlashCommandSpec(
            names=('ask',),
            description='Show local ask-user runtime status or ask-user history.',
            handler=_handle_ask,
        ),
        SlashCommandSpec(
            names=('login',),
            description='Activate a local account profile or ephemeral identity.',
            handler=_handle_login,
        ),
        SlashCommandSpec(
            names=('logout',),
            description='Clear the active local account session.',
            handler=_handle_logout,
        ),
        SlashCommandSpec(
            names=('config', 'settings'),
            description='Show local config runtime state, effective config, config sources, or a config value.',
            handler=_handle_config,
        ),
        SlashCommandSpec(
            names=('lsp',),
            description='Show local LSP runtime status or run document symbols, definition, references, hover, call hierarchy, and diagnostics queries.',
            handler=_handle_lsp,
        ),
        SlashCommandSpec(
            names=('remotes',),
            description='List configured local remote profiles.',
            handler=_handle_remotes,
        ),
        SlashCommandSpec(
            names=('ssh',),
            description='Activate a local SSH remote target/profile.',
            handler=_handle_ssh,
        ),
        SlashCommandSpec(
            names=('teleport',),
            description='Activate a local teleport remote target/profile.',
            handler=_handle_teleport,
        ),
        SlashCommandSpec(
            names=('direct-connect',),
            description='Activate a local direct-connect remote target/profile.',
            handler=_handle_direct_connect,
        ),
        SlashCommandSpec(
            names=('deep-link',),
            description='Activate a local deep-link remote target/profile.',
            handler=_handle_deep_link,
        ),
        SlashCommandSpec(
            names=('disconnect', 'remote-disconnect'),
            description='Disconnect the active local remote runtime target.',
            handler=_handle_remote_disconnect,
        ),
        SlashCommandSpec(
            names=('resources',),
            description='List local MCP resources, optionally filtered by a query string.',
            handler=_handle_resources,
        ),
        SlashCommandSpec(
            names=('resource',),
            description='Render a local MCP resource by URI.',
            handler=_handle_resource,
        ),
        SlashCommandSpec(
            names=('tasks', 'todo', 'todos'),
            description='Show the local runtime task list, optionally filtered by status.',
            handler=_handle_tasks,
        ),
        SlashCommandSpec(
            names=('test',),
            description='Run the project tests (testCommand or a guess); failures go to the agent.',
            handler=_make_check_handler('test'),
        ),
        SlashCommandSpec(
            names=('lint',),
            description='Run the linter (lintCommand or a guess); problems go to the agent.',
            handler=_make_check_handler('lint'),
        ),
        SlashCommandSpec(
            names=('profile',),
            description='Switch to a provider profile from settings: /profile <name>.',
            handler=_handle_profile,
        ),
        SlashCommandSpec(
            names=('statusline',),
            description='Show or set the status line command: /statusline <command>.',
            handler=_handle_statusline,
        ),
        SlashCommandSpec(
            names=('output-style',),
            description='Show or pick the output style: /output-style [name].',
            handler=_handle_output_style,
        ),
        SlashCommandSpec(
            names=('add-dir',),
            description='Let the file tools use another folder for this session: /add-dir <path>.',
            handler=_handle_add_dir,
        ),
        SlashCommandSpec(
            names=('btw',),
            description='Ask a side question; it is not added to the conversation.',
            handler=_handle_btw,
        ),
        SlashCommandSpec(
            names=('bashes', 'shells'),
            description='List background shells (bash run_in_background); /bashes kill <id> stops one.',
            handler=_handle_bashes,
        ),
        SlashCommandSpec(
            names=('workflows',),
            description='List local workflows discovered from workflow manifests.',
            handler=_handle_workflows,
        ),
        SlashCommandSpec(
            names=('workflow',),
            description='Show or run one local workflow by name.',
            handler=_handle_workflow,
        ),
        SlashCommandSpec(
            names=('triggers',),
            description='List local remote triggers discovered from remote trigger manifests.',
            handler=_handle_triggers,
        ),
        SlashCommandSpec(
            names=('trigger',),
            description='Show or run one local remote trigger by id.',
            handler=_handle_trigger,
        ),
        SlashCommandSpec(
            names=('teams',),
            description='List the locally configured collaboration teams.',
            handler=_handle_teams,
        ),
        SlashCommandSpec(
            names=('team',),
            description='Show one local collaboration team by name.',
            handler=_handle_team,
        ),
        SlashCommandSpec(
            names=('messages',),
            description='Show recorded collaboration messages for all teams or one team.',
            handler=_handle_messages,
        ),
        SlashCommandSpec(
            names=('task-next', 'next-task'),
            description='Show the next actionable tasks from the local runtime task list.',
            handler=_handle_task_next,
        ),
        SlashCommandSpec(
            names=('plan', 'planner'),
            description='Show the current local runtime plan.',
            handler=_handle_plan,
        ),
        SlashCommandSpec(
            names=('task',),
            description='Show a local runtime task by id.',
            handler=_handle_task,
        ),
        SlashCommandSpec(
            names=('prompt', 'system-prompt'),
            description='Render the effective Python system prompt.',
            handler=_handle_prompt,
        ),
        SlashCommandSpec(
            names=('permissions',),
            description='Show the active tool permission mode.',
            handler=_handle_permissions,
        ),
        SlashCommandSpec(
            names=('hooks', 'policy'),
            description='Show discovered local hook and policy manifests.',
            handler=_handle_hooks,
        ),
        SlashCommandSpec(
            names=('trust',),
            description=('Show workspace trust (policy, managed settings, safe env) and folder trust for '
                         "project hooks, MCP servers and env; /trust grant|skip|revoke changes folder trust."),
            handler=_handle_trust,
        ),
        SlashCommandSpec(
            names=('model',),
            description='Show or update the active model for the current agent instance.',
            handler=_handle_model,
        ),
        SlashCommandSpec(
            names=('memory',),
            description='Show the currently loaded SIGIL.md memory bundle and discovered files.',
            handler=_handle_memory,
        ),
        SlashCommandSpec(
            names=('status', 'session'),
            description='Show a short runtime/session status summary.',
            handler=_handle_status,
        ),
        SlashCommandSpec(
            names=('clear',),
            description='Clear ephemeral Python runtime state for this process.',
            handler=_handle_clear,
        ),
        SlashCommandSpec(
            names=('compact',),
            description='Summarise and compact the conversation to free context space.',
            handler=_handle_compact,
        ),
        SlashCommandSpec(
            names=('cost',),
            description='Show the total cost and duration of the current session.',
            handler=_handle_cost,
        ),
        SlashCommandSpec(
            names=('exit', 'quit'),
            description='Exit the REPL.',
            handler=_handle_exit,
        ),
        SlashCommandSpec(
            names=('diff',),
            description='View uncommitted changes (git diff) in the working directory.',
            handler=_handle_diff,
        ),
        SlashCommandSpec(
            names=('files',),
            description='List files currently loaded in the session context.',
            handler=_handle_files,
        ),
        SlashCommandSpec(
            names=('copy',),
            description='Copy the last assistant response to a temp file.',
            handler=_handle_copy,
        ),
        SlashCommandSpec(
            names=('export',),
            description='Export the conversation to a text file.',
            handler=_handle_export,
        ),
        SlashCommandSpec(
            names=('stats',),
            description='Show session usage statistics.',
            handler=_handle_stats,
        ),
        SlashCommandSpec(
            names=('tag',),
            description='Add or remove a searchable tag on the current session.',
            handler=_handle_tag,
        ),
        SlashCommandSpec(
            names=('rename',),
            description='Rename the current conversation.',
            handler=_handle_rename,
        ),
        SlashCommandSpec(
            names=('branch',),
            description='Create a fork/branch of the current conversation.',
            handler=_handle_branch,
        ),
        SlashCommandSpec(
            names=('effort',),
            description='Show or set the model effort level (low, medium, high, max, auto).',
            handler=_handle_effort,
        ),
        SlashCommandSpec(
            names=('doctor',),
            description='Diagnose and verify the Sigil Code installation and settings.',
            handler=_handle_doctor,
        ),
        SlashCommandSpec(
            names=('tools',),
            description='List registered tools and highlight custom learned tools.',
            handler=_handle_tools_v2,
        ),
        SlashCommandSpec(
            names=('delete_tool', 'prune_tool'),
            description='Delete a persistent custom tool from the filesystem and memory.',
            handler=_handle_delete_tool,
        ),
        SlashCommandSpec(
            names=('cleanup_tools',),
            description='Purge all dynamically generated tools from src/custom_tools.',
            handler=_handle_cleanup_tools,
        ),
        SlashCommandSpec(
            names=('nudge',),
            description='Show or set the auto-nudge strategy (off, structural, heuristic).',
            handler=_handle_nudge,
        ),
    )


def find_slash_command(command_name: str) -> SlashCommandSpec | None:
    lowered = command_name.lower()
    for spec in get_slash_command_specs():
        if lowered in spec.names:
            return spec
    return None


def _handle_help(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    lines = ['# Slash Commands', '']
    for spec in get_slash_command_specs():
        primary = f'/{spec.names[0]}'
        aliases = ', '.join(f'/{name}' for name in spec.names[1:])
        label = f'{primary} ({aliases})' if aliases else primary
        lines.append(f'- `{label}`: {spec.description}')
    lines.extend(
        [
            '',
            'These commands are handled locally before the model loop.',
        ]
    )
    return _local_result(input_text, '\n'.join(lines))


def _handle_context(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    prompt = args or None
    return _local_result(input_text, agent.render_context_report(prompt))


def _handle_context_raw(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_context_snapshot_report())


def _handle_token_budget(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_token_budget_report())


KEY_HELP = '\n'.join([
    'Keys',
    '  Enter              send the prompt',
    '  Alt+Enter, Ctrl+J  new line',
    '  Shift+Tab          cycle permission mode (default / acceptEdits / plan)',
    '  Esc                interrupt the agent while it works',
    '  Esc Esc            open /rewind (empty prompt)',
    '  Up / Down, Ctrl+R  prompt history and search',
    '  Tab                complete /commands and @paths',
    '  Ctrl+C             interrupt; Ctrl+D on an empty prompt exits',
    '  !command           run a shell command and send its output with the next prompt',
    '  @path              attach a file',
    '  ?                  this help (empty prompt)',
    '  Ctrl+S             stash the draft; on an empty prompt, put it back',
    '  Ctrl+G, Ctrl+X Ctrl+E  edit the prompt in $VISUAL / $EDITOR',
    '  Ctrl+O             verbose transcript after each answer (on/off)',
    '  Ctrl+T             show or hide the todo list',
    '  Alt+P              list models (then /model <name>)',
    '  Alt+T              thinking on/off',
    '  Alt+V              paste an image from the clipboard',
    '  Ctrl+Enter         send queued messages now (while the agent works)',
])


def _handle_terminal_setup(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .terminal_setup import setup

    return _local_result(input_text, setup(args))


def _handle_plugin(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .plugin_marketplace import command

    return _local_result(input_text, command(args))


def _handle_thinking(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    choice = args.strip().lower() or 'toggle'
    off_now = agent.model_config.effort == 'off'
    if choice == 'toggle':
        choice = 'on' if off_now else 'off'
    if choice == 'off':
        if not off_now:
            agent._effort_before_off = agent.model_config.effort
        agent.set_effort('off')
        return _local_result(input_text, 'Thinking off for this session (/thinking on or Alt+T to turn it back on).')
    if choice == 'on':
        agent.set_effort(getattr(agent, '_effort_before_off', None) or 'auto')
        return _local_result(input_text, f'Thinking on (effort {agent.model_config.effort or "auto"}).')
    return _local_result(input_text, 'Usage: /thinking on|off|toggle')


def _handle_subtask(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/subtask <task>: a forked subagent gets the conversation so far (the newest
    SIGIL_SUBTASK_CONTEXT_CHARS, 60000, characters); its answer goes with the next prompt."""
    import os

    from .compact import _render_transcript

    task = args.strip()
    if not task:
        return _local_result(input_text, 'Usage: /subtask <task>')
    session = agent.last_session
    conversation = _render_transcript([m for m in session.messages if m.role != 'system']) if session else ''
    try:
        limit = int(os.environ.get('SIGIL_SUBTASK_CONTEXT_CHARS', '60000'))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        limit = 60000
    if len(conversation) > limit:
        conversation = '... (older part cut)\n' + conversation[-limit:]
    prompt = (f'<conversation-so-far>\n{conversation or "(empty)"}\n</conversation-so-far>\n\n'
              f'You are a fork of the agent in this conversation. Do this task and report back: {task}')
    result = agent._execute_delegate_agent({'prompt': prompt, 'label': 'subtask'})
    agent.pending_notes.append(f'<system-reminder>\nResult of /subtask "{task[:80]}":\n'
                               f'{result.content[:8000]}\n</system-reminder>')
    return _local_result(input_text, result.content)


def _handle_cd(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args.strip():
        return _local_result(input_text, f'Working folder: {agent.runtime_config.cwd}')
    try:
        target = agent.change_directory(args.strip())
    except ValueError as exc:
        return _local_result(input_text, str(exc))
    return _local_result(input_text, f'Working folder is now {target}')


def _handle_theme(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .terminal import THEMES, apply_theme, current_theme

    choice = args.strip().lower()
    if not choice:
        return _local_result(input_text, f'Theme: {current_theme()} (options: {", ".join(THEMES)})')
    if choice not in THEMES:
        return _local_result(input_text, f'Unknown theme {choice!r}; options: {", ".join(THEMES)}')
    import json as _json

    from .settings import LOCAL_SETTINGS, read_settings_file

    path = agent.runtime_config.cwd / LOCAL_SETTINGS
    payload = read_settings_file(path)
    payload['theme'] = choice
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, _json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    apply_theme(choice)
    return _local_result(input_text, f'Theme set to {choice} (saved in {path}).')


def _handle_bug(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    import platform
    import urllib.parse
    import webbrowser

    from .main import _package_version

    from .prompt_constants import PRODUCT_URL

    repo = PRODUCT_URL
    if not repo:
        return _local_result(input_text, 'No repository URL found in pyproject.toml.')
    body = '\n'.join([
        '**What happened**', args.strip() or '(describe the problem)', '',
        '**Environment**',
        f'- version: {_package_version()}',
        f'- Python: {platform.python_version()} on {platform.platform()}',
        f'- model: {agent.model_config.model}',
        f'- last stop reason: {getattr(agent.last_run_result, "stop_reason", None)}',
    ])
    url = f'{repo.rstrip("/")}/issues/new?' + urllib.parse.urlencode(
        {'title': (args.strip() or 'Bug report')[:80], 'body': body})
    try:
        webbrowser.open(url)
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    return _local_result(input_text, f'Opening a new issue (nothing is sent until you submit it):\n{url}')


def _handle_focus(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    result = agent.last_run_result
    transcript = list(getattr(result, 'transcript', ()) or ())
    if not transcript:
        return _local_result(input_text, 'Nothing to show yet.')
    last_user = max((i for i, m in enumerate(transcript) if m.get('role') == 'user'
                     and not (m.get('metadata') or {}).get('kind')
                     and not str(m.get('content', '')).startswith('<system-reminder>')), default=None)
    if last_user is None:
        return _local_result(input_text, 'Nothing to show yet.')
    calls, failed = [], 0
    for message in transcript[last_user + 1:]:
        for call in message.get('tool_calls') or ():
            function = call.get('function', {}) if isinstance(call, dict) else {}
            calls.append(str(function.get('name', '?')))
        if message.get('role') == 'tool' and str(message.get('content', '')).lower().startswith(
                ('error', 'denied', 'blocked', 'the user denied')):
            failed += 1
    from collections import Counter

    summary = ', '.join(f'{name} x{count}' if count > 1 else name for name, count in Counter(calls).items())
    lines = ['> ' + str(transcript[last_user].get('content', '')).strip(), '',
             f'Tools: {summary or "none"}' + (f' ({failed} failed)' if failed else ''), '',
             (getattr(result, 'final_output', '') or '').strip()]
    return _local_result(input_text, '\n'.join(lines))


def _handle_read_only(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/read-only <file>: its text goes with the next prompt, and edits to it are refused."""
    from pathlib import Path as _Path

    from .permission_rules import parse_rule

    if not args.strip():
        current = [rule.raw for rule in agent.session_deny_rules]
        return _local_result(input_text, 'Read-only files: ' + (', '.join(current) or 'none'))
    added = []
    for raw in args.split():
        path = _Path(raw).expanduser()
        if not path.is_absolute():
            path = agent.runtime_config.cwd / path
        path = path.resolve()
        if not path.is_file():
            return _local_result(input_text, f'Not a file: {path}')
        rule = parse_rule(f'Edit({path.as_posix()})')
        if rule is not None and rule.raw not in [r.raw for r in agent.session_deny_rules]:
            agent.session_deny_rules = (*agent.session_deny_rules, rule)
        from .prompt_constants import DEFAULT_MAX_RESULT_SIZE_CHARS

        text = path.read_text(encoding='utf-8', errors='replace')[:DEFAULT_MAX_RESULT_SIZE_CHARS]
        agent.pending_notes.append(f'<system-reminder>\nRead-only reference file {path} (do not edit it):\n'
                                   f'{text}\n</system-reminder>')
        added.append(str(path))
    return _local_result(input_text, 'Read-only for this session (sent with the next prompt): ' + ', '.join(added))


CHANGELOG_URL = 'https://github.com/X-AlphaCoder-X/sigil-code/blob/main/CHANGELOG.md'


def _changelog_candidates() -> list:
    from pathlib import Path as _Path

    return [_Path(__file__).resolve().parent.parent / 'CHANGELOG.md']


def _handle_release_notes(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    text = None
    for path in _changelog_candidates():
        try:
            text = path.read_text(encoding='utf-8')
            break
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
    if text is None:
        # an installed copy has no CHANGELOG.md next to the package
        return _local_result(input_text, f'No local CHANGELOG.md. Release notes: {CHANGELOG_URL}')
    if args.strip() != 'all':
        sections = text.split('\n## ')
        text = '## ' + sections[1].strip() if len(sections) > 1 else text
    return _local_result(input_text, text.strip())


def _handle_architect(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    from .architect import editor_model_config

    on = 'editor_edit' in agent.tool_registry
    editor = editor_model_config(agent.model_config, agent.runtime_config.cwd).model
    if on:
        text = f'Architect mode is on: {agent.model_config.model} plans, {editor} makes the edits (editor_edit).'
    else:
        text = ('Architect mode is off. Start with --architect (or SIGIL_ARCHITECT=1, "architect": true); '
                f'set the editor with --editor-model / SIGIL_EDITOR_MODEL (now: {editor}).')
    return _local_result(input_text, text)


def _handle_map(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .repo_map import build_repo_map, enabled

    limit = int(args) if args.strip().isdigit() else None
    note = '' if enabled(agent.runtime_config.cwd) else \
        '\n\n(Not sent to the model: SIGIL_REPO_MAP=1 or "repoMap": true sends it at the start.)'
    return _local_result(input_text, build_repo_map(agent.runtime_config.cwd, limit) + note)


def _handle_plugins(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    from .sigil_plugins import render_plugin_list

    return _local_result(input_text, render_plugin_list())


def _handle_mcp(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_mcp_report())
    verb, _, server = command.partition(' ')
    if verb in ('reconnect', 'enable', 'disable', 'login', 'logout'):
        return _local_result(input_text, _mcp_server_command(agent, verb, server.strip()))
    if command == 'tools':
        return _local_result(input_text, agent.render_mcp_tools_report())
    if command.startswith('tools '):
        query = command.split(' ', 1)[1].strip()
        return _local_result(input_text, agent.render_mcp_tools_report(query or None))
    if command.startswith('tool '):
        tool_name = command.split(' ', 1)[1].strip()
        if not tool_name:
            return _local_result(input_text, 'Usage: /mcp tool <tool-name>')
        return _local_result(input_text, agent.render_mcp_call_tool_report(tool_name))
    return _local_result(input_text, agent.render_mcp_report(command))


def _handle_search(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_search_report())
    if command == 'providers':
        return _local_result(input_text, agent.render_search_providers_report())
    if command.startswith('providers '):
        query = command.split(' ', 1)[1].strip()
        return _local_result(input_text, agent.render_search_providers_report(query or None))
    if command.startswith('provider '):
        provider = command.split(' ', 1)[1].strip()
        if not provider:
            return _local_result(input_text, 'Usage: /search provider <name>')
        return _local_result(input_text, agent.render_search_report(provider=provider))
    if command.startswith('use '):
        provider = command.split(' ', 1)[1].strip()
        if not provider:
            return _local_result(input_text, 'Usage: /search use <name>')
        return _local_result(input_text, agent.render_search_activate_report(provider))
    return _local_result(input_text, agent.render_search_report(command))


def _handle_remote(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    target = args or None
    return _local_result(input_text, agent.render_remote_report(target))


def _handle_worktree(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_worktree_report())
    if command == 'history':
        return _local_result(input_text, agent.render_worktree_history_report())
    if command.startswith('enter'):
        name = command.split(' ', 1)[1].strip() if ' ' in command else None
        return _local_result(input_text, agent.render_worktree_enter_report(name or None))
    if command.startswith('exit'):
        parts = command.split()
        action = parts[1] if len(parts) > 1 else 'keep'
        discard_changes = any(part in {'discard', 'discard_changes=true'} for part in parts[2:])
        return _local_result(
            input_text,
            agent.render_worktree_exit_report(
                action=action,
                discard_changes=discard_changes,
            ),
        )
    return _local_result(
        input_text,
        'Usage: /worktree [history|enter <name>|exit <keep|remove> [discard]]',
    )


def _handle_account(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_account_report())
    if command == 'profiles':
        return _local_result(input_text, agent.render_account_profiles_report())
    if command.startswith('profile '):
        profile = command.split(' ', 1)[1].strip()
        if not profile:
            return _local_result(input_text, 'Usage: /account profile <name>')
        return _local_result(input_text, agent.render_account_report(profile))
    return _local_result(input_text, 'Usage: /account [profiles|profile <name>]')


def _handle_ask(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_ask_user_report())
    if command == 'history':
        return _local_result(input_text, agent.render_ask_user_history_report())
    return _local_result(input_text, 'Usage: /ask [history]')


def _handle_login(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    target = args.strip()
    if not target:
        return _local_result(input_text, 'Usage: /login <profile-or-identity>')
    return _local_result(input_text, agent.render_account_login_report(target))


def _handle_logout(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_account_logout_report())


def _handle_config(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_config_report())
    if command == 'effective':
        return _local_result(input_text, agent.render_config_effective_report())
    if command.startswith('source '):
        source = command.split(' ', 1)[1].strip()
        if not source:
            return _local_result(input_text, 'Usage: /config source <source-name>')
        return _local_result(input_text, agent.render_config_source_report(source))
    if command.startswith('get '):
        key_path = command.split(' ', 1)[1].strip()
        if not key_path:
            return _local_result(input_text, 'Usage: /config get <key-path>')
        return _local_result(input_text, agent.render_config_value_report(key_path))
    return _local_result(
        input_text,
        'Usage: /config [effective|source <name>|get <key-path>]',
    )


def _handle_lsp(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, agent.render_lsp_report())
    if command == 'diagnostics':
        return _local_result(input_text, agent.render_lsp_diagnostics_report())
    if command.startswith('diagnostics '):
        file_path = command.split(' ', 1)[1].strip()
        if not file_path:
            return _local_result(input_text, 'Usage: /lsp diagnostics [file-path]')
        return _local_result(input_text, agent.render_lsp_diagnostics_report(file_path))
    if command.startswith('symbols '):
        file_path = command.split(' ', 1)[1].strip()
        if not file_path:
            return _local_result(input_text, 'Usage: /lsp symbols <file-path>')
        return _local_result(input_text, agent.render_lsp_document_symbols_report(file_path))
    if command.startswith('workspace '):
        query = command.split(' ', 1)[1].strip()
        if not query:
            return _local_result(input_text, 'Usage: /lsp workspace <query>')
        return _local_result(input_text, agent.render_lsp_workspace_symbols_report(query))
    parts = command.split()
    if len(parts) == 4 and parts[0] in {
        'definition',
        'references',
        'hover',
        'hierarchy',
        'incoming',
        'outgoing',
    }:
        subcommand, file_path, line_text, character_text = parts
        try:
            line = int(line_text)
            character = int(character_text)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            return _local_result(
                input_text,
                f'Usage: /lsp {subcommand} <file-path> <line> <character>',
            )
        if subcommand == 'definition':
            return _local_result(
                input_text,
                agent.render_lsp_definition_report(file_path, line, character),
            )
        if subcommand == 'references':
            return _local_result(
                input_text,
                agent.render_lsp_references_report(file_path, line, character),
            )
        if subcommand == 'hover':
            return _local_result(
                input_text,
                agent.render_lsp_hover_report(file_path, line, character),
            )
        if subcommand == 'hierarchy':
            return _local_result(
                input_text,
                agent.render_lsp_prepare_call_hierarchy_report(file_path, line, character),
            )
        if subcommand == 'incoming':
            return _local_result(
                input_text,
                agent.render_lsp_incoming_calls_report(file_path, line, character),
            )
        if subcommand == 'outgoing':
            return _local_result(
                input_text,
                agent.render_lsp_outgoing_calls_report(file_path, line, character),
            )
    return _local_result(
        input_text,
        'Usage: /lsp [symbols <file>|workspace <query>|definition <file> <line> <character>|references <file> <line> <character>|hover <file> <line> <character>|hierarchy <file> <line> <character>|incoming <file> <line> <character>|outgoing <file> <line> <character>|diagnostics [file]]',
    )


def _handle_remotes(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_remote_profiles_report(query))


def _handle_ssh(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /ssh <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='ssh'))


def _handle_teleport(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /teleport <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='teleport'))


def _handle_direct_connect(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /direct-connect <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='direct-connect'))


def _handle_deep_link(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /deep-link <target-or-profile>')
    return _local_result(input_text, agent.render_remote_mode_report(args, mode='deep-link'))


def _handle_remote_disconnect(
    agent: 'LocalCodingAgent',
    _args: str,
    input_text: str,
) -> SlashCommandResult:
    return _local_result(input_text, agent.render_remote_disconnect_report())


def _handle_resources(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_mcp_resources_report(query))


def _handle_resource(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /resource <mcp-resource-uri>')
    return _local_result(input_text, agent.render_mcp_resource_report(args))


def _handle_tasks(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .background_shells import shell_manager

    status = args or None
    report = agent.render_tasks_report(status)
    if shell_manager().jobs:
        report = report.rstrip() + '\n\n' + shell_manager().render()
    try:
        from .background_runtime import BackgroundSessionRuntime

        background = BackgroundSessionRuntime(agent.runtime_config.cwd / '.port_sessions' / 'background')
        if background.list_records():
            report = report.rstrip() + '\n\n' + background.render_ps()
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    return _local_result(input_text, report)


def _handle_bashes(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """List background shells; `/bashes kill <id>` stops one."""
    from .background_shells import shell_manager

    manager = shell_manager()
    parts = args.split()
    if len(parts) == 2 and parts[0] == 'kill':
        try:
            job = manager.kill(parts[1])
        except KeyError as exc:
            return _local_result(input_text, str(exc.args[0]))
        return _local_result(input_text, f'Background shell {job.job_id}: {job.status}')
    return _local_result(input_text, manager.render())


def _handle_workflows(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_workflows_report(query))


def _handle_workflow(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, 'Usage: /workflow <name> | /workflow run <name>')
    if command.startswith('run '):
        workflow_name = command.split(' ', 1)[1].strip()
        if not workflow_name:
            return _local_result(input_text, 'Usage: /workflow run <name>')
        return _local_result(input_text, agent.render_workflow_run_report(workflow_name))
    return _local_result(input_text, agent.render_workflow_report(command))


def _handle_triggers(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_remote_triggers_report(query))


def _handle_trigger(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    command = args.strip()
    if not command:
        return _local_result(input_text, 'Usage: /trigger <id> | /trigger run <id>')
    if command.startswith('run '):
        trigger_id = command.split(' ', 1)[1].strip()
        if not trigger_id:
            return _local_result(input_text, 'Usage: /trigger run <id>')
        return _local_result(
            input_text,
            agent.render_remote_trigger_action_report('run', trigger_id=trigger_id),
        )
    return _local_result(input_text, agent.render_remote_trigger_report(command))


def _handle_teams(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    query = args or None
    return _local_result(input_text, agent.render_teams_report(query))


def _handle_team(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    team_name = args.strip()
    if not team_name:
        return _local_result(input_text, 'Usage: /team <team-name>')
    return _local_result(input_text, agent.render_team_report(team_name))


def _handle_messages(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    team_name = args.strip() or None
    return _local_result(input_text, agent.render_team_messages_report(team_name))


def _handle_task_next(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_next_tasks_report())


def _handle_plan(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/plan shows the plan; /plan <task> switches to plan mode and starts the task."""
    if not args.strip():
        return _local_result(input_text, agent.render_plan_report())
    from .permission_modes import apply_mode

    apply_mode(agent, 'plan')
    return SlashCommandResult(handled=True, should_query=True, prompt=args.strip())


def _handle_task(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        return _local_result(input_text, 'Usage: /task <task-id>')
    return _local_result(input_text, agent.render_task_report(args))


def _handle_prompt(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_system_prompt())


def _handle_permissions(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/permissions [allow|ask|deny <rule> | remove <rule>]"""
    from .permission_rules import parse_rule
    from .settings import add_local_permission_rule, remove_local_permission_rule

    parts = args.strip().split(None, 1)
    if len(parts) == 2 and parts[0] in ('allow', 'ask', 'deny', 'remove'):
        rule = parts[1].strip()
        if parse_rule(rule) is None:
            return _local_result(input_text, f'Cannot parse rule: {rule}')
        cwd = agent.runtime_config.cwd
        if parts[0] == 'remove':
            removed = remove_local_permission_rule(cwd, rule)
            agent.load_permission_rules()
            if not removed:
                return _local_result(input_text, f'{rule} is not in .sigil/settings.local.json')
            return _local_result(input_text, f'Removed {rule} from permissions.{", ".join(removed)}')
        path = add_local_permission_rule(cwd, parts[0], rule)
        agent.load_permission_rules()
        agent.fire_config_change('local_settings', f'permissions.{parts[0]} += {rule}')
        return _local_result(input_text, f'Added {rule} to permissions.{parts[0]} in {path}')
    report = agent.render_permissions_report()
    rules = agent.permission_rules if agent.permission_rules is not None else agent.load_permission_rules()
    lines = [report, '', '## Permission rules (settings.json)']
    for kind in ('allow', 'ask', 'deny'):
        items = getattr(rules, kind, ())
        lines.append(f'- {kind}: ' + (', '.join(rule.raw for rule in items) if items else '(none)'))
    lines.append('Change them with /permissions allow|ask|deny <rule> or /permissions remove <rule>.')
    return _local_result(input_text, '\n'.join(lines))


def _handle_hooks(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_hook_policy_report())


def _handle_trust(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/trust: the policy trust report and folder trust; /trust grant|skip|revoke changes folder trust
    (workspace_trust.py)."""
    from .workspace_trust import trust_command

    action = args.strip().lower()
    if action in ('grant', 'skip', 'revoke', 'yes', 'no', 'trust', 'deny', 'forget', 'reset'):
        return _local_result(input_text, trust_command(agent.runtime_config.cwd, action))
    return _local_result(input_text, agent.render_trust_report() + '\n\n## Folder trust\n'
                         + trust_command(agent.runtime_config.cwd, 'status'))


def _handle_model(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    if not args:
        lines = [f'Current model: {agent.model_config.model}']
        models = _list_server_models(agent.model_config)
        if models:
            lines.append('Models on the server (/model <name> to switch):')
            lines.extend(f'  {"*" if name == agent.model_config.model else "-"} {name}' for name in models[:60])
        return _local_result(input_text, '\n'.join(lines))
    agent.set_model(args)
    return _local_result(input_text, f'Set model to {agent.model_config.model}')


def _handle_memory(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/memory, /memory add <text> (to ./SIGIL.md), /memory edit (opens $EDITOR)."""
    import os
    import subprocess

    command, _, rest = args.strip().partition(' ')
    path = agent.runtime_config.cwd / 'SIGIL.md'
    if command == 'refresh':
        return _local_result(input_text, _memory_refresh(agent))
    if command == 'add':
        note = rest.strip()
        if not note:
            return _local_result(input_text, 'Usage: /memory add <text>')
        existing = path.read_text(encoding='utf-8') if path.exists() else ''
        prefix = '' if not existing or existing.endswith('\n') else '\n'
        path.write_text(existing + prefix + f'- {note}\n', encoding='utf-8')
        _refresh_memory(agent)
        return _local_result(input_text, f'Added to {path}: {note}')
    if command == 'edit':
        editor = os.environ.get('VISUAL') or os.environ.get('EDITOR') or ('notepad' if os.name == 'nt' else 'vi')
        if not path.exists():
            path.write_text('', encoding='utf-8')
        try:
            import shlex

            parts = [part.strip('"') for part in shlex.split(editor, posix=os.name != 'nt')]
            subprocess.run([*parts, str(path)])
        except OSError as exc:
            return _local_result(input_text, f'Could not start the editor {editor!r}: {exc}')
        _refresh_memory(agent)
        return _local_result(input_text, f'Edited {path}')
    return _local_result(input_text, agent.render_memory_report())


def _memory_refresh(agent: 'LocalCodingAgent') -> str:
    """/memory refresh: read the memory files again; the new text goes with the next prompt."""
    from .agent_context import memory_file_paths

    _refresh_memory(agent)
    dirs = tuple(str(p) for p in agent.runtime_config.additional_working_directories)
    paths = memory_file_paths(agent.runtime_config.cwd, dirs)
    bundle = agent.build_prompt_context().user_context.get('sigilMd')
    if bundle:
        agent.pending_memory_reminder = (
            '<system-reminder>\nThe memory files were reloaded. They replace the earlier memory:\n\n'
            f'{bundle}\n</system-reminder>')
    listing = '\n'.join(f'  {p}' for p in paths) or '  (none)'
    return f'Reloaded {len(paths)} memory file(s); the next prompt sends them:\n{listing}'


def _refresh_memory(agent: 'LocalCodingAgent') -> None:
    try:
        from .agent_context import clear_context_caches

        clear_context_caches()
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)


def _list_server_models(model_config: Any) -> list[str]:
    """Model ids from the server's /models list (OpenAI, Ollama, LM Studio, vLLM...)."""
    import json as _json
    from urllib import request as _request

    url = model_config.base_url.rstrip('/') + '/models'
    req = _request.Request(url, headers={'Authorization': f'Bearer {model_config.api_key}'})
    try:
        with _request.urlopen(req, timeout=min(float(model_config.timeout_seconds or 10), 10.0)) as response:
            data = _json.loads(response.read().decode('utf-8'))
    except Exception as _ignored:
        _log_ignored(_ignored)
        return []
    items = data.get('data') if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return sorted(str(item.get('id')) for item in items if isinstance(item, dict) and item.get('id'))


def _handle_profile(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .profiles import apply_profile, load_profiles

    profiles = load_profiles(agent.runtime_config.cwd)
    name = args.strip()
    if not name:
        listing = '\n'.join(f'- {key}: {value.get("model")} @ {value.get("base_url") or value.get("baseUrl")}'
                             for key, value in profiles.items()) or '(none; add "profiles" to settings.json)'
        return _local_result(input_text, 'Profiles:\n' + listing)
    try:
        apply_profile(agent, name, profiles)
    except KeyError as exc:
        return _local_result(input_text, str(exc.args[0]))
    return _local_result(input_text, f'Using profile {name}: {agent.model_config.model} @ {agent.model_config.base_url}')


def _handle_statusline(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/statusline [command]: show it, or save it to .sigil/settings.local.json."""
    import json as _json

    from .settings import LOCAL_SETTINGS, load_settings, read_settings_file

    command = args.strip()
    if not command:
        current = load_settings(agent.runtime_config.cwd).get('statusLine')
        return _local_result(input_text, (
            f'Status line: {_json.dumps(current) if current else "(none)"}\n'
            'Set one with /statusline <command>. The command gets session JSON on stdin '
            '(session_id, model, cwd, permission_mode, cost) and its first output line is shown '
            'under the prompt, e.g. /statusline python -c "import json,sys;d=json.load(sys.stdin);'
            'print(d[\'model\'], d[\'permission_mode\'])"'))
    path = agent.runtime_config.cwd / LOCAL_SETTINGS
    payload = read_settings_file(path)
    payload['statusLine'] = {'type': 'command', 'command': command}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, _json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    from .workspace_trust import mark_user_item

    mark_user_item(agent.runtime_config.cwd, 'statusLine', 'statusLine', command)  # the user set it
    return _local_result(input_text, f'Status line set (shows from the next prompt): {command}')


def _mcp_server_command(agent: 'LocalCodingAgent', verb: str, name: str) -> str:
    """/mcp reconnect|enable|disable|login|logout <server>."""
    import json as _json
    from dataclasses import replace as _replace

    from .mcp_runtime import MCPRuntime, connection_manager
    from .settings import LOCAL_SETTINGS, read_settings_file

    if not name:
        return f'Usage: /mcp {verb} <server>'
    if verb in ('login', 'logout'):
        from .mcp_oauth import mcp_auth_command

        message = mcp_auth_command(agent.mcp_runtime, verb, name)
        if message.startswith('Logged in to') and 'still failed' not in message:
            agent.tool_registry = agent._register_mcp_tools(agent.tool_registry)  # its tools work from now on
        return message
    if verb == 'reconnect':
        server = agent.mcp_runtime.get_server(name) if agent.mcp_runtime is not None else None
        if server is None:
            return f'Unknown MCP server: {name}'
        manager = connection_manager()
        manager._drop(manager._key(server))
        try:
            manager.connection(server, 30.0)
        except Exception as exc:
            return f'Could not reconnect {name}: {exc}'
        return f'Reconnected {name}'
    path = agent.runtime_config.cwd / LOCAL_SETTINGS
    payload = read_settings_file(path)
    disabled = [str(item) for item in payload.get('disabledMcpServers', []) if isinstance(item, str)]
    if verb == 'disable' and name not in disabled:
        disabled.append(name)
    if verb == 'enable':
        disabled = [item for item in disabled if item != name]
    payload['disabledMcpServers'] = disabled
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, _json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    agent.mcp_runtime = MCPRuntime.from_workspace(
        agent.runtime_config.cwd, tuple(str(p) for p in agent.runtime_config.additional_working_directories))
    agent.tool_context = _replace(agent.tool_context, mcp_runtime=agent.mcp_runtime)
    agent.fire_config_change('local_settings', f'disabledMcpServers = {disabled}')
    return f'{"Disabled" if verb == "disable" else "Enabled"} MCP server {name}'


def _project_command(agent: 'LocalCodingAgent', kind: str) -> str | None:
    """The test or lint command: settings testCommand / lintCommand, else a guess."""
    from .settings import load_settings

    configured = load_settings(agent.runtime_config.cwd).get(f'{kind}Command')
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    root = agent.runtime_config.cwd
    if kind == 'test':
        if (root / 'package.json').is_file():
            return 'npm test'
        if (root / 'Cargo.toml').is_file():
            return 'cargo test'
        if (root / 'go.mod').is_file():
            return 'go test ./...'
        if (root / 'tests').is_dir() or (root / 'pyproject.toml').is_file():
            return 'python -m pytest -q'
    else:
        if (root / 'package.json').is_file():
            return 'npm run lint'
        if (root / 'Cargo.toml').is_file():
            return 'cargo clippy'
        if (root / 'pyproject.toml').is_file() or (root / 'setup.cfg').is_file():
            return 'ruff check .'
    return None


def run_project_check(agent: 'LocalCodingAgent', kind: str, command: str | None = None) -> tuple[bool, str, str]:
    """Run the test/lint command through the shell permission checks: (passed, command, output)."""
    from .agent_tools import execute_tool
    from .agent_types import ToolCall

    command = command or _project_command(agent, kind)
    if not command:
        return True, '', f'No {kind} command found. Set {kind}Command in .sigil/settings.json.'
    call = ToolCall(id=f'{kind}_check', name='bash', arguments={'command': command, 'timeout_seconds': 600})
    blocked, context = agent._check_permission_rules(call, [])
    if blocked is not None:
        return False, command, blocked.content
    result = execute_tool(agent.tool_registry, 'bash', call.arguments, context or agent.tool_context)
    passed = result.ok and 'exit_code=0' in result.content.split('\n', 1)[0]
    return passed, command, result.content


def _make_check_handler(kind: str):
    def _handler(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
        passed, command, output = run_project_check(agent, kind, args.strip() or None)
        if not command:
            return _local_result(input_text, output)
        if passed:
            return _local_result(input_text, f'{kind.capitalize()} passed: {command}')
        prompt = (f'`{command}` failed. Fix the problems it reports, then run it again.\n\n'
                  f'<{kind}-output>\n{output[-12000:]}\n</{kind}-output>')
        return SlashCommandResult(handled=True, should_query=True, prompt=prompt)

    return _handler


def _handle_output_style(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/output-style [name]: list styles, or pick one (saved to .sigil/settings.local.json)."""
    import json as _json

    from .output_styles import discover_styles
    from .settings import LOCAL_SETTINGS, read_settings_file

    styles = discover_styles(agent.runtime_config.cwd)
    choice = args.strip()
    if not choice:
        current = agent.output_style or '(setting or default)'
        lines = [f'Output style: {current}', 'Styles:']
        lines.extend(f'- {name}: {description}' for name, (description, _p) in styles.items())
        return _local_result(input_text, '\n'.join(lines))
    match = next((name for name in styles if name.lower() == choice.lower()), None)
    if match is None:
        return _local_result(input_text, f'Unknown style {choice!r}. Styles: {", ".join(styles)}')
    agent.output_style = match
    agent.fire_config_change('local_settings', f'outputStyle = {match}')
    path = agent.runtime_config.cwd / LOCAL_SETTINGS
    payload = read_settings_file(path)
    payload['outputStyle'] = match
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, _json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return _local_result(input_text, f'Output style set to {match} (applies from the next prompt).')


def _handle_add_dir(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/add-dir <path>: let the file tools use another folder for this session."""
    from dataclasses import replace as _replace
    from pathlib import Path as _Path

    raw = args.strip().strip('"')
    if not raw:
        added = agent.runtime_config.additional_working_directories
        return _local_result(input_text, 'Added folders: ' + (', '.join(map(str, added)) if added else '(none)'))
    folder = _Path(raw).expanduser()
    if not folder.is_absolute():
        folder = agent.runtime_config.cwd / folder
    folder = folder.resolve()
    if not folder.is_dir():
        return _local_result(input_text, f'Not a folder: {folder}')
    dirs = tuple(agent.runtime_config.additional_working_directories)
    if folder not in dirs:
        dirs = (*dirs, folder)
    agent.runtime_config = _replace(agent.runtime_config, additional_working_directories=dirs)
    agent.tool_context = _replace(agent.tool_context, additional_working_directories=dirs, path_access=None)
    agent._run_hook_event('DirectoryAdded', {'session_id': agent.active_session_id, 'directory': str(folder),
                                             'source': 'slash_command'})
    return _local_result(input_text, f'Added folder for this session: {folder}')


def _handle_btw(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """/btw <question>: ask on the side; neither question nor answer joins the conversation."""
    question = args.strip()
    if not question:
        return _local_result(input_text, 'Usage: /btw <question>')
    messages = agent.last_session.to_openai_messages() if agent.last_session is not None else []
    messages = list(messages) + [{'role': 'user', 'content': (
        f'{question}\n\n(Side question: answer briefly from what you already know; do not call tools.)')}]
    try:
        turn = agent.client.complete(messages, [])
    except Exception as exc:
        return _local_result(input_text, f'Side question failed: {exc}')
    from .reasoning import split_reasoning

    answer, _reasoning = split_reasoning(turn.content)
    return _local_result(input_text, answer or '(no answer)')


def _handle_status(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    return _local_result(input_text, agent.render_status_report())


def _handle_clear(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    from .command_hooks import fire_session_end

    fire_session_end(getattr(agent, 'command_hook_runtime', None), agent.active_session_id, 'clear')
    agent.clear_runtime_state()
    return _local_result(
        input_text,
        'Cleared ephemeral Python agent state for this process.',
    )


def _handle_compact(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    from .compact import compact_conversation

    custom_instructions = args.strip() if args.strip() else None
    result = compact_conversation(agent, custom_instructions, trigger='manual')

    if result.error:
        return _local_result(input_text, f'Compact failed: {result.error}')

    lines = ['Conversation compacted.']
    if result.pre_compact_token_count:
        lines.append(
            f'  Tokens before: ~{result.pre_compact_token_count:,}  '
            f'→  after: ~{result.post_compact_token_count:,}'
        )
    return _local_result(input_text, '\n'.join(lines))


def _handle_cost(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    usage = agent.cumulative_usage
    cost = agent.cumulative_cost_usd

    def _fmt_cost(usd: float) -> str:
        if usd < 0.01:
            return f'${usd:.4f}'
        return f'${usd:.2f}'

    lines = [
        f'Total cost:            {_fmt_cost(cost)}',
        f'Total input tokens:    {usage.input_tokens:,}',
        f'Total output tokens:   {usage.output_tokens:,}',
    ]
    if usage.cache_read_input_tokens:
        lines.append(f'Cache read tokens:     {usage.cache_read_input_tokens:,}')
    if usage.cache_creation_input_tokens:
        lines.append(f'Cache creation tokens:  {usage.cache_creation_input_tokens:,}')
    if usage.reasoning_tokens:
        lines.append(f'Reasoning tokens:      {usage.reasoning_tokens:,}')
    lines.append(f'Total tokens:          {usage.total_tokens:,}')
    return _local_result(input_text, '\n'.join(lines))


def _handle_exit(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    import random
    import sys

    messages = ['Goodbye!', 'See ya!', 'Bye!', 'Catch you later!']
    output = random.choice(messages)
    # Build the result first so the transcript is recorded, then exit.
    result = _local_result(input_text, output)
    print(output)
    sys.exit(0)
    return result  # unreachable, but satisfies the type checker


def _handle_diff(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    import subprocess

    cwd = str(agent.runtime_config.cwd)
    try:
        proc = subprocess.run(
            ['git', 'diff'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        diff_output = proc.stdout.strip()
        if not diff_output:
            # Also check staged changes
            proc_staged = subprocess.run(
                ['git', 'diff', '--staged'],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            diff_output = proc_staged.stdout.strip()
            if not diff_output:
                return _local_result(input_text, 'No uncommitted changes.')
            return _local_result(input_text, f'Staged changes:\n{diff_output}')
        return _local_result(input_text, diff_output)
    except FileNotFoundError as _ignored:
        _log_ignored(_ignored)
        return _local_result(input_text, 'git is not available.')
    except subprocess.TimeoutExpired as _ignored:
        _log_ignored(_ignored)
        return _local_result(input_text, 'git diff timed out.')
    except Exception as exc:
        return _local_result(input_text, f'Error running git diff: {exc}')


def _handle_files(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """List files loaded in the session context (from readFileState)."""
    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session.')

    # Collect file paths mentioned in tool results
    file_paths: list[str] = []
    for msg in session.messages:
        if msg.role == 'tool' and msg.name in ('Read', 'read_file', 'ReadFile'):
            # Extract path from content or metadata
            path = msg.metadata.get('path')
            if isinstance(path, str):
                file_paths.append(path)
            elif msg.content and msg.content.startswith('/'):
                # First line might be the path
                first_line = msg.content.split('\n', 1)[0].strip()
                if '/' in first_line and len(first_line) < 256:
                    file_paths.append(first_line)

    # Also look at tool_calls in assistant messages
    for msg in session.messages:
        if msg.role == 'assistant' and msg.tool_calls:
            for tc in msg.tool_calls:
                func = tc.get('function', {}) if isinstance(tc, dict) else {}
                if func.get('name') in ('Read', 'read_file', 'ReadFile', 'View'):
                    import json as _json
                    try:
                        args = _json.loads(func.get('arguments', '{}'))
                        path = args.get('file_path') or args.get('path')
                        if isinstance(path, str):
                            file_paths.append(path)
                    except (ValueError, TypeError) as _ignored:
                        _log_ignored(_ignored)
                        pass

    # Deduplicate preserving order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for p in file_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if not unique_paths:
        return _local_result(input_text, 'No files loaded in context.')

    cwd = str(agent.runtime_config.cwd)
    relative_paths = []
    for p in unique_paths:
        if p.startswith(cwd):
            relative_paths.append(p[len(cwd):].lstrip('/'))
        else:
            relative_paths.append(p)

    lines = [f'Files in context ({len(relative_paths)}):']
    for p in relative_paths:
        lines.append(f'  {p}')
    return _local_result(input_text, '\n'.join(lines))


def _handle_copy(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Copy the last assistant response to a temp file."""
    import tempfile as _tempfile

    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session.')

    # Find the Nth most recent assistant message (default N=0 = latest)
    n = 0
    if args.strip().isdigit():
        n = min(int(args.strip()), 20)

    assistant_messages = [
        msg for msg in session.messages
        if msg.role == 'assistant' and msg.content.strip()
    ]
    if not assistant_messages:
        return _local_result(input_text, 'No assistant responses to copy.')

    index = len(assistant_messages) - 1 - n
    if index < 0:
        return _local_result(
            input_text,
            f'Only {len(assistant_messages)} assistant responses available.',
        )

    content = assistant_messages[index].content

    # Write to temp file
    from pathlib import Path as _Path
    tmp_dir = _Path(_tempfile.gettempdir()) / 'sigilcode'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / 'response.md'
    out_path.write_text(content, encoding='utf-8')
    from .desktop import copy_to_clipboard

    method = copy_to_clipboard(content)
    char_count = len(content)
    line_count = content.count('\n') + 1
    where = f'the clipboard ({method}) and {out_path}' if method else str(out_path)
    return _local_result(
        input_text,
        f'Copied {char_count:,} chars ({line_count} lines) to {where}',
    )


def _handle_export(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Export the conversation transcript to a text file."""
    from pathlib import Path as _Path
    import time as _time

    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session to export.')

    markdown = any(token.endswith('.md') for token in args.split())
    lines: list[str] = ['# Conversation', ''] if markdown else []
    for msg in session.messages:
        label = msg.role.upper()
        if msg.role == 'tool' and msg.name:
            label = f'TOOL:{msg.name}'
        if markdown:
            if msg.role == 'system':
                continue
            lines.append(f'## {label.title() if msg.role != "tool" else "Tool: " + (msg.name or "")}')
            lines.append('```\n' + msg.content + '\n```' if msg.role == 'tool' else msg.content)
        else:
            lines.append(f'--- {label} ---')
            lines.append(msg.content)
        lines.append('')
    text = '\n'.join(lines)

    # /export [--redact] [file]: --redact hides keys, tokens and passwords (see redact.py)
    tokens = args.split()
    redact = '--redact' in tokens
    if redact:
        from .redact import redact_text

        text = redact_text(text)
    filename = ' '.join(token for token in tokens if token != '--redact')
    if not filename:
        timestamp = _time.strftime('%Y%m%d_%H%M%S')
        filename = f'conversation_{timestamp}.txt'
    if not filename.endswith(('.txt', '.md')):
        filename += '.txt'

    out_path = _Path(str(agent.runtime_config.cwd)) / filename
    out_path.write_text(text, encoding='utf-8')
    return _local_result(
        input_text,
        f'Exported {len(session.messages)} messages to {out_path}'
        + (' (secrets redacted; read it before sharing)' if redact else ''),
    )


def _handle_stats(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """Show session usage statistics."""
    usage = agent.cumulative_usage
    cost = agent.cumulative_cost_usd

    session = agent.last_session
    msg_count = len(session.messages) if session else 0
    user_msgs = sum(1 for m in (session.messages if session else []) if m.role == 'user')
    assistant_msgs = sum(1 for m in (session.messages if session else []) if m.role == 'assistant')
    tool_msgs = sum(1 for m in (session.messages if session else []) if m.role == 'tool')

    last_result = agent.last_run_result
    context_tokens = last_result.last_turn_input_tokens if last_result else 0

    lines = [
        '## Session Statistics',
        '',
        f'Messages:       {msg_count} total ({user_msgs} user, {assistant_msgs} assistant, {tool_msgs} tool)',
        f'Context (now):   {context_tokens:,} tokens',
        f'Input (cumul.):  {usage.input_tokens:,} tokens (sum across all turns)',
        f'Output tokens:   {usage.output_tokens:,}',
        f'Cost:            ${cost:.4f}',
        f'Model:           {agent.model_config.model}',
    ]
    if agent.active_session_id:
        lines.append(f'Session ID:    {agent.active_session_id}')
    return _local_result(input_text, '\n'.join(lines))


def _handle_tag(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Add or remove a tag on the current session."""
    tag = args.strip()
    if not tag:
        # Show current tags
        tags = getattr(agent, '_session_tags', set())
        if tags:
            return _local_result(input_text, f'Session tags: {", ".join(sorted(tags))}')
        return _local_result(input_text, 'No tags set. Usage: /tag <tag-name>')

    # Toggle tag
    if not hasattr(agent, '_session_tags'):
        agent._session_tags = set()

    if tag in agent._session_tags:
        agent._session_tags.discard(tag)
        return _local_result(input_text, f'Removed tag: {tag}')
    agent._session_tags.add(tag)
    return _local_result(input_text, f'Added tag: {tag}')


def _handle_rename(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Rename the current conversation."""
    name = args.strip()
    if not name:
        return _local_result(input_text, 'Usage: /rename <name>')

    if not hasattr(agent, '_session_name'):
        agent._session_name = None
    agent._session_name = name
    if agent.active_session_id:
        from .session_store import save_session_name

        save_session_name(agent.runtime_config.session_directory, name, agent.active_session_id)
    return _local_result(input_text, f'Session renamed to: {name}')


def _handle_branch(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Create a fork/branch of the current conversation."""
    import json as _json
    from uuid import uuid4

    session = agent.last_session
    if session is None:
        return _local_result(input_text, 'No active session to branch.')

    branch_name = args.strip() or f'branch-{uuid4().hex[:8]}'
    new_session_id = uuid4().hex

    # Save a copy of the current transcript as a new session file
    session_dir = agent.runtime_config.session_directory
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f'{new_session_id}.json'

    transcript = [msg.to_transcript_entry() for msg in session.messages]
    branch_data = {
        'session_id': new_session_id,
        'branch_name': branch_name,
        'branched_from': agent.active_session_id,
        'messages': transcript,
        'model': agent.model_config.model,
    }

    try:
        write_text_atomic(session_path, _json.dumps(branch_data, indent=2), encoding='utf-8')
        return _local_result(
            input_text,
            f'Created branch "{branch_name}" (session: {new_session_id})\n'
            f'Saved to: {session_path}',
        )
    except Exception as exc:
        return _local_result(input_text, f'Error creating branch: {exc}')


def _handle_effort(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Show or set the model effort level."""
    import os

    valid_levels = ('off', 'low', 'medium', 'high', 'xhigh', 'max', 'auto')
    current = getattr(getattr(agent, 'model_config', None), 'effort', None)
    env_override = os.environ.get('SIGIL_EFFORT_LEVEL')

    if not args.strip():
        level = current or env_override or 'auto'
        msg = f'Current effort level: {level}'
        if env_override:
            msg += f' (from SIGIL_EFFORT_LEVEL env var)'
        return _local_result(input_text, msg)

    level = args.strip().lower()
    if level == 'none':
        level = 'off'  # OpenAI's word for off
    if level not in valid_levels:
        return _local_result(
            input_text,
            f'Invalid effort level: {level}\nValid levels: {", ".join(valid_levels)}',
        )

    if env_override:
        return _local_result(
            input_text,
            f'Cannot change effort level — overridden by '
            f'SIGIL_EFFORT_LEVEL={env_override}',
        )

    # Sent with every request from now on, in the field this provider reads.
    agent.set_effort(level)
    from .providers import get_quirks

    quirks = get_quirks(agent.model_config.base_url, agent.model_config.model)
    note = (
        f' (sent as {quirks.effort_field} to {quirks.name})'
        if quirks.effort_field != 'none' and level != 'auto'
        else ''
    )
    if quirks.effort_field == 'none' and level != 'auto':
        note = (
            f' (note: no effort field is known for {quirks.name}; '
            'set SIGIL_EFFORT_FIELD to choose one)'
        )
    from .prompt_constants import EFFORT_FIGURES

    figure = EFFORT_FIGURES.get(level)
    return _local_result(input_text, f'Set effort level to: {level}{" " + figure if figure else ""}{note}')


def _handle_tools_v2(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    lines = ['## Registered Tools', '']
    for name in sorted(agent.tool_registry.keys()):
        is_custom = bool(getattr(agent.tool_registry[name], 'custom_source', ''))
        status = '[L]' if is_custom else '[Core]' # L for Learned
        lines.append(f"- `{name}` {status}")
    
    lines.append('\nLegend: `[Core]` = Core System Tool, `[L]` = Learned (Custom) Tool')
    return _local_result(input_text, '\n'.join(lines))
    
def _handle_nudge(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    """Show or set the auto-nudge strategy (off, structural, heuristic)."""
    from dataclasses import replace
    valid_strategies = ('off', 'structural', 'heuristic')
    current = agent.runtime_config.nudge_strategy

    if not args.strip():
        return _local_result(input_text, f'Current nudge strategy: {current}\nAvailable strategies: {", ".join(valid_strategies)}')

    strategy = args.strip().lower()
    if strategy not in valid_strategies:
        return _local_result(
            input_text,
            f'Invalid nudge strategy: {strategy}\nValid strategies: {", ".join(valid_strategies)}',
        )

    # Update the runtime config
    agent.runtime_config = replace(agent.runtime_config, nudge_strategy=strategy)

    if agent.config_runtime:
        try:
            agent.config_runtime.set_value('nudge_strategy', strategy)
            return _local_result(input_text, f'\033[1;32m✓ Nudge strategy set to:\033[0m \033[1;36m{strategy}\033[0m (saved persistently)')
        except Exception as e:
            return _local_result(input_text, f'\033[1;31m✗ Set nudge strategy to: {strategy} (Error saving: {e})\033[0m')

    return _local_result(input_text, f'\033[1;33m! Nudge strategy set to: {strategy} (In-memory only)\033[0m')

def _handle_delete_tool(agent: 'LocalCodingAgent', args: str, input_text: str) -> SlashCommandResult:
    tool_name = args.strip()
    if not tool_name:
        return _local_result(input_text, 'Usage: /delete_tool <tool_name>')
    
    custom_file = agent.runtime_config.cwd / 'src' / 'custom_tools' / f"{tool_name}.py"
    if not custom_file.exists():
        return _local_result(input_text, f'Error: Tool `{tool_name}` not found in custom_tools/.')
    
    try:
        custom_file.unlink()
        if tool_name in agent.tool_registry:
            del agent.tool_registry[tool_name]
        return _local_result(input_text, f'Successfully deleted learned tool: `{tool_name}`')
    except Exception as e:
        return _local_result(input_text, f'Error deleting tool: {e}')

def _handle_cleanup_tools(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    count = agent.cleanup_custom_tools(force=True)
    # Reset registry to core tools
    from .agent_tools import default_tool_registry
    agent.tool_registry = default_tool_registry()
    
    return _local_result(input_text, f'Purged {count} custom tools and reset registry to core tools.')


def _handle_doctor(agent: 'LocalCodingAgent', _args: str, input_text: str) -> SlashCommandResult:
    """Diagnose and verify the Sigil Code installation."""
    import shutil
    import sys

    checks: list[str] = []

    # Python version
    py_ver = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
    ok = sys.version_info >= (3, 10)
    checks.append(f'{"✓" if ok else "✗"} Python version: {py_ver} (need ≥3.10)')

    # Git available
    git_ok = shutil.which('git') is not None
    checks.append(f'{"✓" if git_ok else "✗"} git: {"found" if git_ok else "NOT FOUND"}')

    # Model config
    checks.append(f'✓ Model: {agent.model_config.model}')
    checks.append(f'✓ Base URL: {agent.model_config.base_url}')

    # Working directory
    cwd = agent.runtime_config.cwd
    checks.append(f'✓ Working directory: {cwd}')
    checks.append(f'{"✓" if cwd.exists() else "✗"} Working directory exists: {cwd.exists()}')

    # Session directory
    sess_dir = agent.runtime_config.session_directory
    checks.append(f'✓ Session directory: {sess_dir}')
    checks.append(f'{"✓" if sess_dir.exists() else "○"} Session directory exists: {sess_dir.exists()}')

    # API key
    has_key = bool(agent.model_config.api_key)
    checks.append(f'{"✓" if has_key else "✗"} API key: {"set" if has_key else "NOT SET"}')

    # Tools
    tool_count = len(agent.tool_registry) if agent.tool_registry else 0
    checks.append(f'✓ Registered tools: {tool_count}')

    # Memory files (SIGIL.md)
    sigil_md = cwd / 'SIGIL.md'
    checks.append(
        f'{"✓" if sigil_md.exists() else "○"} SIGIL.md: '
        f'{"found" if sigil_md.exists() else "not found (optional)"}'
    )

    output = '## Doctor Report\n\n' + '\n'.join(checks)
    return _local_result(input_text, output)


def _local_result(input_text: str, output: str) -> SlashCommandResult:
    transcript = (
        {'role': 'user', 'content': input_text},
        {'role': 'assistant', 'content': output},
    )
    return SlashCommandResult(
        handled=True,
        should_query=False,
        output=output,
        transcript=transcript,
    )
