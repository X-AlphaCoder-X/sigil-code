"""Command hooks (the settings.json ``hooks`` block).

Reads the ``hooks`` block from the settings files:

    ~/.sigil/settings.json            (user)
    <cwd>/.sigil/settings.json        (project, checked in)
    <cwd>/.sigil/settings.local.json  (local, not checked in)

Format::

    {
      "hooks": {
        "PreToolUse": [
          {"matcher": "Bash|Edit", "hooks": [{"type": "command", "command": "./check.sh", "timeout": 30}]}
        ]
      }
    }

Supported events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop,
SubagentStop, PreCompact, Notification, SessionEnd.

Hook types:
    command  ``"command"``: shell command, payload on stdin.
    http     ``"url"`` (+ optional ``"headers"``, $VARS expanded): payload POSTed as JSON;
             a 2xx JSON body is parsed like command stdout (that is how it blocks); any
             non-2xx status or connection error is a non-blocking error unless the hook has
             "onFailure": "block".
    prompt   ``"prompt"`` ($ARGUMENTS is replaced with the payload JSON): the model answers
             ``{"ok": true}`` or ``{"ok": false, "reason": ...}``; ok=false blocks.

Each hook command gets a JSON payload on stdin (session_id, cwd, hook_event_name,
plus tool_name/tool_input/tool_response or prompt) and ``SIGIL_PROJECT_DIR`` in env.

Exit codes:
    0  success. stdout may be JSON (see below). For SessionStart / UserPromptSubmit,
       plain stdout is added as context.
    2  blocking error. stderr is fed back to the model (blocks the tool call / prompt,
       or keeps the agent working for Stop).
    *  non-blocking error, unless the hook sets ``"onFailure": "block"``.

JSON stdout fields honoured: ``decision: "block"`` + ``reason``, ``continue: false`` +
``stopReason``, ``hookSpecificOutput.permissionDecision`` ("deny" blocks; "allow"/"ask"
fall through to the normal permission checks), ``permissionDecisionReason``,
``updatedInput`` (PreToolUse), ``additionalContext``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

SUPPORTED_EVENTS = (
    'SessionStart',
    'UserPromptSubmit',
    'PreToolUse',
    'PostToolUse',
    'Stop',
    'SubagentStop',
    'PreCompact',
    'Notification',
    'SessionEnd',
    'PostToolUseFailure',
    'SubagentStart',
    'PostCompact',
    'StopFailure',
    'PermissionRequest',
    'CwdChanged',
    'ConfigChange',
    'InstructionsLoaded',
    'PreModelSwitch',
    'PostModelSwitch',
    'TaskCreated',
    'TaskCompleted',
    'PostToolBatch',
    'UserPromptExpansion',
    'Setup',
    'FileChanged',
    'DirectoryAdded',
    'WorktreeCreate',
    'WorktreeRemove',
    'MessageDisplay',
    'Elicitation',
    'ElicitationResult',
)
# mcp_tool: "tool": "mcp__server__tool" gets the hook input (plus "input") as arguments.
# agent: "prompt" runs a read-only subagent that answers {"ok": ...} like a prompt hook.
SUPPORTED_TYPES = ('command', 'http', 'prompt', 'mcp_tool', 'agent')
# Events whose plain (non-JSON) stdout is injected as model context.
_PLAIN_STDOUT_CONTEXT_EVENTS = ('SessionStart', 'UserPromptSubmit', 'PreCompact')
DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_OUTPUT_CHARS = 10_000

# Sigil tool name -> rule tool name, so matchers can use either name.
HOOK_TOOL_ALIASES = {
    'bash': 'Bash',
    'bash_output': 'BashOutput',
    'bash_kill': 'KillShell',
    'powershell': 'PowerShell',
    'background_agent_restart': 'BackgroundAgentRestart',
    'skill': 'Skill',
    'edit_file': 'Edit',
    'write_file': 'Write',
    'read_file': 'Read',
    'glob_search': 'Glob',
    'grep_search': 'Grep',
    'list_dir': 'LS',
    'web_fetch': 'WebFetch',
    'web_search': 'WebSearch',
    'notebook_edit': 'NotebookEdit',
    'delegate_agent': 'Task',
    'todo_write': 'TodoWrite',
}


@dataclass(frozen=True)
class CommandHook:
    event: str
    matcher: str
    command: str
    timeout: float
    block_on_failure: bool
    source: str
    type: str = 'command'  # 'command' (shell), 'http' (command holds the URL), 'prompt' (command holds the prompt)
    headers: dict[str, str] = field(default_factory=dict)
    # "allowedEnvVars": the only variables $VAR / ${VAR} in http headers may read (docs: others
    # become empty). Why: every variable used to be expanded, so a project hook could send the
    # API key (or any secret in the environment) to its URL.
    allowed_env_vars: tuple[str, ...] = ()
    condition: str | None = None  # "if": a permission rule the tool call must match
    run_async: bool = False  # "async": start it and do not wait
    shell: str | None = None  # "shell": "powershell" | "bash" | "cmd" | "sh"
    args: tuple[str, ...] | None = None  # "args": run the command directly with these arguments
    status_message: str | None = None  # "statusMessage": shown while it runs


_HEADER_VAR_RE = re.compile(r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))')


def _expand_allowed(value: str, allowed: tuple[str, ...]) -> str:
    """$VAR / ${VAR} from the environment only for names in allowedEnvVars; others become ''."""
    def sub(match: 're.Match[str]') -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, '') if name in allowed else ''

    return _HEADER_VAR_RE.sub(sub, value)


@dataclass
class HookOutcome:
    blocked: bool = False
    reason: str = ''
    stop: bool = False  # "continue": false
    stop_reason: str = ''
    additional_context: list[str] = field(default_factory=list)
    updated_input: dict[str, Any] | None = None
    updated_tool_output: str | None = None  # PostToolUse "updatedToolOutput"
    system_messages: list[str] = field(default_factory=list)  # "systemMessage": shown to the user
    # Elicitation hooks: hookSpecificOutput {"action": accept|decline|cancel, "content": {...}} answers it.
    elicitation: dict[str, Any] | None = None
    worktree_path: str | None = None  # WorktreeCreate: the path the hook made
    display_content: str | None = None  # MessageDisplay: shown instead of the text (display only)
    ran: int = 0
    errors: list[str] = field(default_factory=list)

    def context_text(self) -> str:
        return '\n'.join(part for part in self.additional_context if part.strip())


def worktree_from_hook(run_event: Callable[[str, dict[str, Any]], Any], name: str,
                       session_id: str | None = None) -> str | None:
    """WorktreeCreate hooks replace git's worktree creation: a command hook prints
    the path on stdout, an HTTP hook returns hookSpecificOutput.worktreePath. None = no hook configured;
    a failed hook or a missing path fails the creation."""
    outcome = run_event('WorktreeCreate', {'session_id': session_id, 'name': name})
    if outcome is None or not outcome.ran:
        return None
    if outcome.blocked or outcome.errors or not outcome.worktree_path:
        raise RuntimeError('WorktreeCreate hook failed: '
                           + (outcome.reason or '; '.join(outcome.errors) or 'it gave no worktree path'))
    folder = Path(outcome.worktree_path).expanduser()
    if not folder.is_dir():
        raise RuntimeError(f'The WorktreeCreate hook path is not a folder: {folder}')
    return str(folder.resolve())


def wire_worktree_hooks(runtime: Any, run_event: Callable[[str, dict[str, Any]], Any],
                        session_id: Callable[[], str | None] = lambda: None) -> Any:
    """Give a WorktreeRuntime the WorktreeCreate / WorktreeRemove hooks (used on every route)."""
    runtime.create_hook = lambda slug: worktree_from_hook(run_event, slug, session_id())
    runtime.remove_hook = lambda path: remove_worktree_with_hook(run_event, path, session_id())
    return runtime


def remove_worktree_with_hook(run_event: Callable[[str, dict[str, Any]], Any], path: str,
                              session_id: str | None = None) -> bool | None:
    """WorktreeRemove for a hook-made worktree. None = no hook configured; True = removed (exit 0, or
    the folder is gone anyway)."""
    outcome = run_event('WorktreeRemove', {'session_id': session_id, 'worktree_path': path})
    if outcome is None or not outcome.ran:
        return None
    return not (outcome.blocked or outcome.errors) or not Path(path).exists()


def _matches(matcher: str, tool_name: str | None) -> bool:
    if tool_name is None or matcher in ('', '*'):
        return True
    names = {tool_name, HOOK_TOOL_ALIASES.get(tool_name, tool_name)}
    try:
        pattern = re.compile(rf'^(?:{matcher})$')
    except re.error as _ignored:
        _log_ignored(_ignored)
        return matcher in names
    return any(pattern.match(name) for name in names)


def _condition_matches(condition: str | None, tool_name: str | None, tool_input: Any, cwd: Path) -> bool:
    """A hook's "if" is a permission rule such as "Bash(git commit *)"."""
    if not condition:
        return True
    if tool_name is None:
        return False
    from .permission_rules import parse_rule, rule_matches

    rule = parse_rule(condition)
    if rule is None:
        return False
    return rule_matches(rule, tool_name, tool_input if isinstance(tool_input, dict) else {}, cwd=cwd)


def _hook_command_line(hook: 'CommandHook') -> list[str]:
    if hook.args is not None:
        return [hook.command, *hook.args]  # no shell
    shell = (hook.shell or '').lower()
    if shell in ('powershell', 'pwsh'):
        import shutil

        exe = shutil.which('pwsh') or shutil.which('powershell') or 'powershell'
        return [exe, '-NoProfile', '-NonInteractive', '-Command', hook.command]
    if shell == 'cmd':
        return ['cmd.exe', '/c', hook.command]
    if shell == 'sh':
        return ['/bin/sh', '-c', hook.command]
    return [*_resolve_shell(), hook.command]


def _resolve_shell() -> list[str]:
    if sys.platform == 'win32':
        from .agent_tools import _BASH_EXECUTABLE

        if os.path.exists(_BASH_EXECUTABLE):
            return [_BASH_EXECUTABLE, '-c']
        return ['cmd.exe', '/c']
    return ['/bin/sh', '-c']


@dataclass
class CommandHookRuntime:
    cwd: Path
    hooks: tuple[CommandHook, ...] = ()
    load_errors: tuple[str, ...] = ()
    # (system, prompt, timeout_seconds) -> reply text. Set by the agent so 'prompt' hooks can call the model.
    model_complete: Callable[[str, str, float], str] | None = None
    # (tool name, arguments) -> result text. Set by the agent for 'mcp_tool' hooks.
    mcp_call: Callable[[str, dict], str] | None = None
    # (system, prompt, timeout_seconds) -> reply text from a read-only subagent, for 'agent' hooks.
    agent_run: Callable[[str, str, float], str] | None = None
    # (event payload) -> fields every hook input gets (session_id, transcript_path, permission_mode).
    # Set by the agent.
    common_fields: Callable[[dict], dict] | None = None

    @classmethod
    def from_workspace(cls, cwd: Path, *, include_project: bool = True) -> 'CommandHookRuntime':
        cwd = cwd.resolve()
        if os.environ.get('SIGIL_SAFE_MODE') == '1':
            return cls(cwd=cwd)  # --safe-mode: no hooks
        from .settings import extra_settings, setting_sources

        allowed = setting_sources()
        sources: list[tuple[str, Path]] = []
        if not os.environ.get('SIGIL_DISABLE_USER_HOOKS') and 'user' in allowed:
            sources.append(('user', Path.home() / '.sigil' / 'settings.json'))
        if include_project:
            from .workspace_trust import is_trusted, report_skipped

            # Why: a cloned repository's hooks used to run with no question asked (workspace_trust.py)
            if is_trusted(cwd):
                if 'project' in allowed:
                    sources.append(('project', cwd / '.sigil' / 'settings.json'))
                if 'local' in allowed:
                    sources.append(('local', cwd / '.sigil' / 'settings.local.json'))
            else:
                report_skipped(cwd)
        hooks: list[CommandHook] = []
        errors: list[str] = []
        disabled = False
        seen: set[Path] = set()
        for label, path in sources:
            try:
                resolved = path.resolve()
            except OSError as _ignored:
                _log_ignored(_ignored)
                continue
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            try:
                payload = json.loads(resolved.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f'{resolved}: {exc}')
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get('disableAllHooks') is True:
                disabled = True
            hooks.extend(_parse_hooks(payload.get('hooks'), f'{label}:{resolved}', errors))
        extra = extra_settings()
        if extra:
            if extra.get('disableAllHooks') is True:
                disabled = True
            hooks.extend(_parse_hooks(extra.get('hooks'), 'settings-flag', errors))
        from .sigil_plugins import loaded_plugins

        for plugin in loaded_plugins():  # --plugin-dir: hooks/hooks.json
            if plugin.hooks:
                hooks.extend(_parse_hooks(plugin.hooks, f'plugin:{plugin.name}', errors))
        if disabled:
            hooks = []
        return cls(cwd=cwd, hooks=tuple(hooks), load_errors=tuple(errors))

    def run_if_configured(self, event: str, payload: dict[str, Any]) -> 'HookOutcome | None':
        return self.run(event, payload) if self.has(event) else None

    def has(self, event: str) -> bool:
        return any(hook.event == event for hook in self.hooks)

    def run(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        tool_name: str | None = None,
    ) -> HookOutcome:
        outcome = HookOutcome()
        selected = [h for h in self.hooks if h.event == event and _matches(h.matcher, tool_name)
                    and _condition_matches(h.condition, tool_name, payload.get('tool_input'), self.cwd)]
        if not selected:
            return outcome
        common: dict[str, Any] = {}
        if self.common_fields is not None:
            try:
                common = dict(self.common_fields(payload) or {})
            except Exception:
                from .debug_log import get_logger

                get_logger('hooks').debug('hook common fields failed', exc_info=True)
                common = {}
        full_payload = {'cwd': str(self.cwd), 'hook_event_name': event, **common, **payload}
        if tool_name is not None:
            full_payload.setdefault('tool_name', HOOK_TOOL_ALIASES.get(tool_name, tool_name))
            full_payload.setdefault('sigil_tool_name', tool_name)
        stdin_text = json.dumps(full_payload, ensure_ascii=False, default=str)
        from .child_env import child_env

        env = child_env()  # N21: SIGIL_SSH_AGENT=off removes the SSH agent socket here too
        env['SIGIL_PROJECT_DIR'] = str(self.cwd)
        for hook in selected:
            outcome.ran += 1
            if hook.status_message:
                sys.stderr.write(f'\033[2m{hook.status_message}\033[0m\n')
            if hook.run_async:
                # Fire and forget: the result does not change this event.
                threading.Thread(target=self._run_one, args=(hook, stdin_text, env, HookOutcome()),
                                 daemon=True).start()
                continue
            self._run_one(hook, stdin_text, env, outcome)
            if outcome.blocked or outcome.stop:
                break
        for message in outcome.system_messages:
            sys.stderr.write(f'\033[33m{message}\033[0m\n')
        return outcome

    def _run_one(self, hook: CommandHook, stdin_text: str, env: dict[str, str], outcome: HookOutcome) -> None:
        def failure(message: str) -> None:
            outcome.errors.append(message)
            if hook.block_on_failure:
                outcome.blocked = True
                outcome.reason = f'Hook failed and is configured with onFailure=block: {message}'

        if hook.type == 'http':
            self._run_http(hook, stdin_text, outcome, failure)
            return
        if hook.type == 'prompt':
            self._run_prompt(hook, stdin_text, outcome, failure)
            return
        if hook.type == 'agent':
            self._run_prompt(hook, stdin_text, outcome, failure, runner=self.agent_run)
            return
        if hook.type == 'mcp_tool':
            self._run_mcp_tool(hook, stdin_text, outcome, failure)
            return

        try:
            from .background_shells import run_command_tree

            # a hook that times out is ended with everything it started (Job Object / process group)
            completed = run_command_tree(_hook_command_line(hook), input_text=stdin_text, cwd=str(self.cwd),
                                         env=env, timeout=hook.timeout,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired:
            failure(f'{hook.event} hook timed out after {hook.timeout:g}s: {hook.command}')
            return
        except OSError as exc:
            failure(f'{hook.event} hook could not start ({exc}): {hook.command}')
            return

        stdout = (completed.stdout or '')[:_MAX_OUTPUT_CHARS].strip()
        stderr = (completed.stderr or '')[:_MAX_OUTPUT_CHARS].strip()
        if completed.returncode == 2:
            outcome.blocked = True
            outcome.reason = stderr or f'Blocked by {hook.event} hook: {hook.command}'
            return
        if completed.returncode != 0:
            failure(f'{hook.event} hook exited {completed.returncode}: {stderr or hook.command}')
            return

        self._apply_output(hook, stdout, outcome)

    def _apply_output(self, hook: CommandHook, stdout: str, outcome: HookOutcome) -> None:
        data: Any = None
        if stdout.startswith('{'):
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError as _ignored:
                _log_ignored(_ignored)
                data = None
        if not isinstance(data, dict):
            if stdout and hook.event in _PLAIN_STDOUT_CONTEXT_EVENTS:
                outcome.additional_context.append(stdout)
            if stdout and hook.event == 'WorktreeCreate':  # a command hook prints the worktree path
                outcome.worktree_path = stdout.splitlines()[-1].strip()
            return

        # prompt hooks: {"ok": false, "reason": "..."} (prompt-hook schema)
        if data.get('ok') is False:
            outcome.blocked = True
            outcome.reason = str(data.get('reason') or f'Blocked by {hook.event} prompt hook')

        specific = data.get('hookSpecificOutput')
        specific = specific if isinstance(specific, dict) else {}
        if data.get('continue') is False:
            outcome.stop = True
            outcome.stop_reason = str(data.get('stopReason') or 'Stopped by hook')
        if data.get('decision') == 'block':
            outcome.blocked = True
            outcome.reason = str(data.get('reason') or f'Blocked by {hook.event} hook')
        decision = specific.get('permissionDecision')
        if decision == 'deny':
            outcome.blocked = True
            outcome.reason = str(
                specific.get('permissionDecisionReason') or data.get('reason') or 'Denied by PreToolUse hook'
            )
        action = specific.get('action')
        if hook.event in ('Elicitation', 'ElicitationResult') and action in ('accept', 'decline', 'cancel'):
            outcome.elicitation = {'action': action}
            if action == 'accept' and isinstance(specific.get('content'), dict):
                outcome.elicitation['content'] = specific['content']
        if hook.event == 'WorktreeCreate' and isinstance(specific.get('worktreePath'), str):
            outcome.worktree_path = specific['worktreePath']  # HTTP hooks
        if hook.event == 'MessageDisplay' and isinstance(specific.get('displayContent'), str):
            outcome.display_content = specific['displayContent']
        updated = specific.get('updatedInput')
        if isinstance(updated, dict):
            outcome.updated_input = updated
        new_output = specific.get('updatedToolOutput', data.get('updatedToolOutput'))
        if isinstance(new_output, str):
            outcome.updated_tool_output = new_output
        system_message = data.get('systemMessage')
        if isinstance(system_message, str) and system_message.strip():
            outcome.system_messages.append(system_message.strip())
        context = specific.get('additionalContext')
        if isinstance(context, str) and context.strip():
            outcome.additional_context.append(context.strip())

    def _run_http(self, hook: CommandHook, body: str, outcome: HookOutcome, failure) -> None:
        import urllib.error
        import urllib.request

        url = hook.command
        if not url.lower().startswith(('http://', 'https://')):
            failure(f'{hook.event} http hook has a non-http(s) url: {url}')
            return
        headers = {'Content-Type': 'application/json'}
        headers.update({k: _expand_allowed(v, hook.allowed_env_vars) for k, v in hook.headers.items()})
        request = urllib.request.Request(url, data=body.encode('utf-8'), headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=hook.timeout) as response:
                text = response.read(_MAX_OUTPUT_CHARS).decode('utf-8', errors='replace').strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode('utf-8', errors='replace').strip() if exc.fp else ''
            # Non-2xx is a non-blocking error (blocks only with onFailure: "block"); a hook server
            # blocks with a 2xx JSON decision body.
            failure(f'{hook.event} http hook returned HTTP {exc.code}: {detail or url}')
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failure(f'{hook.event} http hook failed ({exc}): {url}')
            return
        self._apply_output(hook, text, outcome)

    def _run_mcp_tool(self, hook: CommandHook, payload_json: str, outcome: HookOutcome, failure) -> None:
        if self.mcp_call is None:
            failure(f'{hook.event} mcp_tool hook skipped: no MCP runtime available')
            return
        try:
            payload = json.loads(payload_json)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            payload = {}
        arguments = {**(hook.headers or {}), **(payload if isinstance(payload, dict) else {})}
        try:
            text = self.mcp_call(hook.command, arguments)
        except Exception as exc:
            failure(f'{hook.event} mcp_tool hook {hook.command} failed: {exc}')
            return
        self._apply_output(hook, text, outcome)

    def _run_prompt(self, hook: CommandHook, payload_json: str, outcome: HookOutcome, failure,
                    runner: Callable[[str, str, float], str] | None = None) -> None:
        complete = runner if hook.type == 'agent' else self.model_complete
        if complete is None:
            failure(f'{hook.event} {hook.type} hook skipped: no model client available')
            return
        instructions = hook.command.replace('$ARGUMENTS', payload_json)
        if '$ARGUMENTS' not in hook.command:
            instructions += f'\n\nHook input JSON:\n{payload_json}'
        system = (
            'You are a hook that evaluates an AI coding agent action. Reply with ONLY a JSON object: '
            '{"ok": true} to allow, or {"ok": false, "reason": "<why>"} to block. No other text.'
        )
        try:
            reply = complete(system, instructions, hook.timeout).strip()
        except Exception as exc:  # model/backend errors are hook failures, not crashes
            failure(f'{hook.event} prompt hook model call failed: {exc}')
            return
        match = re.search(r'\{.*\}', reply, re.DOTALL)
        if match is None:
            failure(f'{hook.event} prompt hook returned no JSON: {reply[:200]}')
            return
        self._apply_output(hook, match.group(0), outcome)

    def render_summary(self) -> str:
        if not self.hooks and not self.load_errors:
            return 'No command hooks configured (add a "hooks" block to .sigil/settings.json).'
        lines = [f'Command hooks: {len(self.hooks)}']
        for hook in self.hooks:
            matcher = f' [{hook.matcher}]' if hook.matcher else ''
            flags = ' onFailure=block' if hook.block_on_failure else ''
            target = hook.command if len(hook.command) <= 80 else hook.command[:77] + '...'
            lines.append(
                f'- {hook.event}{matcher} [{hook.type}]: {target} '
                f'(timeout {hook.timeout:g}s{flags}; {hook.source})'
            )
        for error in self.load_errors:
            lines.append(f'- load error: {error}')
        return '\n'.join(lines)


def fire_session_end(runtime: 'CommandHookRuntime | None', session_id: str | None, reason: str) -> None:
    """Run SessionEnd hooks (reason: 'exit', 'clear', 'logout', 'prompt_input_exit', 'other').

    SessionEnd cannot block anything; failures are swallowed so exit always succeeds.
    """
    if runtime is None or not runtime.has('SessionEnd'):
        return
    try:
        runtime.run('SessionEnd', {'session_id': session_id, 'reason': reason})
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)


def _parse_hooks(raw: Any, source: str, errors: list[str]) -> list[CommandHook]:
    if not isinstance(raw, dict):
        return []
    parsed: list[CommandHook] = []
    for event, groups in raw.items():
        if event not in SUPPORTED_EVENTS:
            errors.append(f'{source}: unsupported hook event {event!r} (supported: {", ".join(SUPPORTED_EVENTS)})')
            continue
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            matcher = group.get('matcher', '')
            matcher = matcher if isinstance(matcher, str) else ''
            entries = group.get('hooks')
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                hook_type = entry.get('type', 'command')
                if hook_type not in SUPPORTED_TYPES:
                    errors.append(f'{source}: unsupported hook type {hook_type!r} (supported: {", ".join(SUPPORTED_TYPES)})')
                    continue
                # command hooks: "command"; http hooks: "url"; prompt and agent hooks: "prompt";
                # mcp_tool hooks: "tool" (fixed extra arguments go in "input")
                target_key = {'command': 'command', 'http': 'url', 'prompt': 'prompt', 'agent': 'prompt',
                              'mcp_tool': 'tool'}[hook_type]
                command = entry.get(target_key)
                if not isinstance(command, str) or not command.strip():
                    errors.append(f'{source}: {event} {hook_type} hook is missing "{target_key}"')
                    continue
                default_timeout = {'prompt': 30.0, 'agent': 120.0}.get(hook_type, DEFAULT_TIMEOUT_SECONDS)
                timeout = entry.get('timeout', default_timeout)
                if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
                    timeout = default_timeout
                raw_headers = entry.get('input') if hook_type == 'mcp_tool' else entry.get('headers')
                headers = (
                    {str(k): str(v) for k, v in raw_headers.items()}
                    if isinstance(raw_headers, dict)
                    else {}
                )
                raw_args = entry.get('args')
                parsed.append(
                    CommandHook(
                        event=event,
                        matcher=matcher.strip(),
                        command=command,
                        timeout=float(timeout),
                        block_on_failure=entry.get('onFailure') == 'block',
                        source=source,
                        type=hook_type,
                        headers=headers,
                        allowed_env_vars=tuple(str(name) for name in entry.get('allowedEnvVars') or ()
                                               if isinstance(name, str)),
                        condition=str(entry['if']) if isinstance(entry.get('if'), str) and entry['if'].strip() else None,
                        run_async=entry.get('async') is True,
                        shell=str(entry['shell']).lower() if isinstance(entry.get('shell'), str) else None,
                        args=tuple(str(a) for a in raw_args) if isinstance(raw_args, list) else None,
                        status_message=(str(entry['statusMessage'])
                                        if isinstance(entry.get('statusMessage'), str) else None),
                    )
                )
    return parsed
