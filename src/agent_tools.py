from __future__ import annotations
import ast

import hashlib
import json
import os
import re
import collections
import selectors
import shutil
import subprocess

from .prompt_constants import DEFAULT_MAX_RESULT_SIZE_CHARS
import sys
import time


def _resolve_bash_executable(env: Any = None, which: Any = None, exists: Any = None,
                             platform: str | None = None) -> str:
    """bash for the bash tool. On Windows, in order: SIGIL_GIT_BASH_PATH,
    bash on PATH unless it is the WSL launcher (System32 / WindowsApps),
    [why: it used to try a path that only existed on one machine first, and could pick WSL bash]
    Git for Windows next to git.exe on PATH, then the standard Git install folders. When none is
    found the PowerShell tool is used instead (see _powershell_enabled)."""
    env = os.environ if env is None else env
    which = shutil.which if which is None else which
    exists = os.path.exists if exists is None else exists
    if (platform or sys.platform) != 'win32':
        return '/bin/bash'
    value = str(env.get('SIGIL_GIT_BASH_PATH') or '').strip().strip('"')
    if value and exists(value):
        return value
    import ntpath
    from pathlib import PureWindowsPath  # Windows path rules on any host (this branch is the Windows lookup)

    def _norm(value: Any) -> str:  # case-insensitive, either slash: C:/Windows == c:\WINDOWS
        return ntpath.normcase(ntpath.normpath(str(value)))

    system32 = _norm(ntpath.join(str(env.get('SystemRoot') or env.get('SYSTEMROOT') or r'C:\Windows'), 'System32'))
    on_path = which('bash')
    if on_path:
        lowered = _norm(on_path)
        # skip the WSL launcher (System32 / WindowsApps); the comparison used backslash text only, so
        # a path written with forward slashes picked WSL bash
        if not (lowered.startswith(system32 + '\\') or '\\windowsapps\\' in lowered):
            if exists(on_path):
                return str(on_path)

    candidates: list[PureWindowsPath] = []
    git = which('git')
    if git:
        for parent in list(PureWindowsPath(git).parents)[:3]:  # <Git>\cmd\git.exe, <Git>\mingw64\bin\git.exe
            candidates += [parent / 'bin' / 'bash.exe', parent / 'usr' / 'bin' / 'bash.exe']
    for base in (env.get('ProgramFiles'), env.get('ProgramW6432'), env.get('ProgramFiles(x86)')):
        if base:
            candidates += [PureWindowsPath(base) / 'Git' / 'bin' / 'bash.exe',
                           PureWindowsPath(base) / 'Git' / 'usr' / 'bin' / 'bash.exe']
    if env.get('LOCALAPPDATA'):
        candidates.append(PureWindowsPath(env['LOCALAPPDATA']) / 'Programs' / 'Git' / 'bin' / 'bash.exe')
    for candidate in candidates:
        if exists(str(candidate)):
            return str(candidate)
    return '/bin/bash'


_BASH_EXECUTABLE = _resolve_bash_executable()
_WIN_CHILD_FLAGS = (
    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    if sys.platform == 'win32'
    else 0
)


def _kill_process_tree(process: 'subprocess.Popen[Any]') -> None:
    """Kill a shell and everything it started (on Windows, process.kill() only
    ends bash.exe and leaves its children - e.g. pip - running)."""
    if sys.platform == 'win32':
        subprocess.run(
            ['taskkill', '/PID', str(process.pid), '/T', '/F'],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    try:
        process.kill()
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass


_WIN_BACKSLASH_PATH_RE = re.compile(r'(?<![\w\\])([A-Za-z]):\\((?:[^\s"\'\\;&|<>()`$]+\\?)+)')


def _normalize_windows_paths(command: str) -> str:
    """On Windows, bash treats backslashes as escapes, so C:\\Users\\x becomes
    C:Usersx. Rewrite unquoted/double-quoted Windows drive paths to forward slashes,
    which both bash and Windows programs accept. Single-quoted text is left alone."""
    if sys.platform != 'win32' or ':\\' not in command:
        return command
    parts = re.split(r"('[^']*')", command)
    for index, part in enumerate(parts):
        if part.startswith("'") and part.endswith("'") and len(part) >= 2:
            continue
        parts[index] = _WIN_BACKSLASH_PATH_RE.sub(
            lambda m: f'{m.group(1)}:/' + m.group(2).replace('\\', '/'),
            part,
        )
    return ''.join(parts)


import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Union

from .agent_types import AgentPermissions, AgentRuntimeConfig, ToolExecutionResult
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

if TYPE_CHECKING:
    from .account_runtime import AccountRuntime
    from .ask_user_runtime import AskUserRuntime
    from .config_runtime import ConfigRuntime
    from .lsp_runtime import LSPRuntime
    from .mcp_runtime import MCPRuntime
    from .plan_runtime import PlanRuntime
    from .remote_runtime import RemoteRuntime
    from .remote_trigger_runtime import RemoteTriggerRuntime
    from .search_runtime import SearchRuntime
    from .skills_runtime import SkillsRuntime
    from .task_runtime import TaskRuntime
    from .team_runtime import TeamRuntime
    from .workflow_runtime import WorkflowRuntime
    from .worktree_runtime import WorktreeRuntime


class ToolPermissionError(RuntimeError):
    """Raised when the runtime configuration does not allow a tool action."""


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete because of invalid input or state."""


@dataclass
class ToolExecutionContext:
    root: Path
    command_timeout_seconds: float
    max_output_chars: int
    permissions: AgentPermissions
    extra_env: dict[str, str] = field(default_factory=dict)
    tool_registry: dict[str, 'SigilTool'] | None = None
    search_runtime: 'SearchRuntime | None' = None
    account_runtime: 'AccountRuntime | None' = None
    ask_user_runtime: 'AskUserRuntime | None' = None
    config_runtime: 'ConfigRuntime | None' = None
    lsp_runtime: 'LSPRuntime | None' = None
    mcp_runtime: 'MCPRuntime | None' = None
    remote_runtime: 'RemoteRuntime | None' = None
    remote_trigger_runtime: 'RemoteTriggerRuntime | None' = None
    plan_runtime: 'PlanRuntime | None' = None
    task_runtime: 'TaskRuntime | None' = None
    team_runtime: 'TeamRuntime | None' = None
    workflow_runtime: 'WorkflowRuntime | None' = None
    worktree_runtime: 'WorktreeRuntime | None' = None
    skills_runtime: 'SkillsRuntime | None' = None
    additional_working_directories: tuple[Path, ...] = ()
    session: Any | None = None  # Using Any to avoid circular dependency with AgentSessionState
    scratchpad_root: Path | None = None  # shell output redirection is allowed only inside this dir
    model_config: Any | None = None  # used to launch background agents with the same backend
    path_access: Any | None = None  # PathAccess, built on first use (see _path_access)
    approved: bool = False  # this call was approved by the user or an allow rule
    shell_guard: str | None = None  # 'warn' in bypass mode; SIGIL_SHELL_GUARD overrides
    permission_mode: str = 'default'  # kept in step by permission_modes.apply_mode
    session_allow_rules: tuple[str, ...] = ()  # "allow for this session" answers
    # the session's --allowed-tools / --disallowed-tools / --tools, handed on to background agents
    cli_allowed_tools: tuple[str, ...] = ()
    cli_disallowed_tools: tuple[str, ...] = ()
    cli_tools: tuple[str, ...] | None = None


ToolHandler = Callable[
    [dict[str, Any], ToolExecutionContext],
    Union[str, tuple[str, dict[str, Any]]],
]


@dataclass(frozen=True)
class SigilTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    # '' for built-in, MCP and plugin tools; 'user:<file>' or 'project:<file>' for a custom tool
    # (load_custom_tools, create_new_tool): a custom tool may never take a built-in tool's name.
    custom_source: str = ''

    @staticmethod
    def _normalized_parameters(parameters: dict[str, Any] | Any) -> dict[str, Any]:
        """Return a valid JSON schema for tool parameters.

        Some providers (e.g. Anthropic's OpenAI compat layer) reject tools whose
        parameters schema is empty or missing the required "type" field.
        """
        if not isinstance(parameters, dict) or parameters.get('type') != 'object':
            return {'type': 'object', 'properties': {}}
        return parameters

    def to_openai_tool(self) -> dict[str, object]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self._normalized_parameters(self.parameters),
            },
        }

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
        try:
            result = self.handler(arguments, context)
            if isinstance(result, ToolExecutionResult):
                return result
            if isinstance(result, tuple):
                content, metadata = result
            else:
                content, metadata = result, {}
            return ToolExecutionResult(name=self.name, ok=True, content=content, metadata=metadata)
        except ToolPermissionError as exc:
            return ToolExecutionResult(
                name=self.name,
                ok=False,
                content=str(exc),
                metadata={'error_kind': 'permission_denied'},
            )
        except (ToolExecutionError, OSError, subprocess.SubprocessError) as exc:
            return ToolExecutionResult(
                name=self.name,
                ok=False,
                content=str(exc),
                metadata={'error_kind': 'tool_execution_error'},
            )
        except Exception as exc:  # a tool bug or an MCP server error must not end the run
            from .debug_log import get_logger

            get_logger('tools').exception('tool %s failed', self.name)  # full trace to the log
            return ToolExecutionResult(
                name=self.name,
                ok=False,
                content=f'{type(exc).__name__}: {exc}',
                metadata={'error_kind': 'tool_execution_error'},
            )


@dataclass(frozen=True)
class ToolStreamUpdate:
    kind: str
    content: str = ''
    stream: str | None = None
    result: ToolExecutionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_tool_context(
    config: AgentRuntimeConfig,
    *,
    extra_env: dict[str, str] | None = None,
    tool_registry: dict[str, SigilTool] | None = None,
    search_runtime: 'SearchRuntime | None' = None,
    account_runtime: 'AccountRuntime | None' = None,
    ask_user_runtime: 'AskUserRuntime | None' = None,
    config_runtime: 'ConfigRuntime | None' = None,
    lsp_runtime: 'LSPRuntime | None' = None,
    mcp_runtime: 'MCPRuntime | None' = None,
    remote_runtime: 'RemoteRuntime | None' = None,
    remote_trigger_runtime: 'RemoteTriggerRuntime | None' = None,
    plan_runtime: 'PlanRuntime | None' = None,
    task_runtime: 'TaskRuntime | None' = None,
    team_runtime: 'TeamRuntime | None' = None,
    workflow_runtime: 'WorkflowRuntime | None' = None,
    worktree_runtime: 'WorktreeRuntime | None' = None,
    skills_runtime: 'SkillsRuntime | None' = None,
    session: Any | None = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        root=config.cwd.resolve(),
        command_timeout_seconds=config.command_timeout_seconds,
        max_output_chars=config.max_output_chars,
        permissions=config.permissions,
        extra_env=dict(extra_env or {}),
        tool_registry=tool_registry,
        search_runtime=search_runtime,
        account_runtime=account_runtime,
        ask_user_runtime=ask_user_runtime,
        config_runtime=config_runtime,
        lsp_runtime=lsp_runtime,
        mcp_runtime=mcp_runtime,
        remote_runtime=remote_runtime,
        remote_trigger_runtime=remote_trigger_runtime,
        plan_runtime=plan_runtime,
        task_runtime=task_runtime,
        team_runtime=team_runtime,
        workflow_runtime=workflow_runtime,
        additional_working_directories=config.additional_working_directories,
        worktree_runtime=worktree_runtime,
        skills_runtime=skills_runtime,
        session=session,
        scratchpad_root=config.scratchpad_root,
        cli_allowed_tools=tuple(getattr(config, 'cli_allowed_tools', ()) or ()),
        cli_disallowed_tools=tuple(getattr(config, 'cli_disallowed_tools', ()) or ()),
        cli_tools=getattr(config, 'cli_tools', None),
    )


# Bash timeouts: 2 min default, 10 min ceiling, both overridable
# via BASH_DEFAULT_TIMEOUT_MS / BASH_MAX_TIMEOUT_MS. The ceiling is never below the default.
def _env_timeout_seconds(name: str, fallback: float) -> float:
    try:
        value = float(os.environ.get(name, '')) / 1000.0
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return fallback
    return value if value >= 1.0 else fallback


DEFAULT_BASH_TIMEOUT_SECONDS = _env_timeout_seconds('BASH_DEFAULT_TIMEOUT_MS', 120.0)
MAX_BASH_TIMEOUT_SECONDS = max(
    _env_timeout_seconds('BASH_MAX_TIMEOUT_MS', 600.0), DEFAULT_BASH_TIMEOUT_SECONDS
)


def _bash_timeout(arguments: dict[str, Any], context: ToolExecutionContext) -> float:
    raw = arguments.get('timeout_seconds')
    if raw is None:
        return max(context.command_timeout_seconds, DEFAULT_BASH_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ToolExecutionError('timeout_seconds must be a number')
    return max(1.0, min(value, MAX_BASH_TIMEOUT_SECONDS))


def _create_new_tool(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    """Handler for the create_new_tool tool."""
    _ensure_write_allowed(context)
    
    name = _require_string(arguments, 'name')
    description = _require_string(arguments, 'description')
    parameters_json = _require_string(arguments, 'parameters_json')
    code = _require_string(arguments, 'code')
    # Why: the name became a file path (`../x` or an absolute path wrote outside custom_tools) and
    # could replace a built-in tool (read_file, bash) for the rest of the session.
    problem = custom_tool_name_problem(name, context.tool_registry or {}, may_replace_project=True)
    if problem is not None:
        raise ToolExecutionError(f'create_new_tool: {problem}. Pick another name.')

    # [AST SECURITY CHECK] Validate syntax and block dangerous patterns
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (
            f"Error: Invalid Python syntax in tool code.\nTraceback:\n{e.msg} at line {e.lineno}, col {e.offset}\nCode snippet:\n{e.text}\n\nPlease fix the code and try again.",
            {'status': 'syntax_error', 'line': e.lineno}
        )
    except Exception as e:
        return (f"Error validating code structure: {e}", {'status': 'validation_error'})

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return (
                f"Error: Access to dunder attributes is forbidden in custom tools: "
                f"'{node.attr}' at line {node.lineno}. This is a security restriction.",
                {'status': 'security_error', 'attr': node.attr}
            )
        if isinstance(node, ast.Name) and node.id in (
            'exec', 'eval', 'compile', '__import__', 'breakpoint',
            'getattr', 'setattr', 'delattr', 'globals', 'locals',
            'vars', 'dir', 'type', 'super', 'open',
        ):
            return (
                f"Error: Use of '{node.id}' is forbidden in custom tools "
                f"at line {node.lineno}.",
                {'status': 'security_error', 'name': node.id}
            )
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module = getattr(node, 'module', None) or (
                node.names[0].name if node.names else '')
            if module in ('os', 'sys', 'subprocess', 'shutil', 'ctypes',
                          'importlib', 'code', 'codeop', 'pty', 'socket'):
                return (
                    f"Error: Import of '{module}' is forbidden in custom tools "
                    f"at line {node.lineno}.",
                    {'status': 'security_error', 'module': module}
                )

    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError as e:
        raise ToolExecutionError(f"Invalid parameters_json: {e}")

    # Ensure the code is somewhat valid and contains a handler
    if 'def handler' not in code and 'def ' not in code:
        raise ToolExecutionError("Code must define a tool handler function.")

    # Save to custom_tools directory
    custom_tools_dir = context.root / 'src' / 'custom_tools'
    custom_tools_dir.mkdir(parents=True, exist_ok=True)
    
    tool_file = custom_tools_dir / f"{name}.py"
    tool_file.write_text(code, encoding='utf-8')

    # Dynamically load and register (in-memory)
    def dynamic_handler(args: dict[str, Any], ctx: ToolExecutionContext) -> Any:
        _safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in ('exec', 'eval', 'compile', '__import__', 'open',
                         'breakpoint', 'exit', 'quit', 'input', 'memoryview')
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) for k in dir(__builtins__)
            if k not in ('exec', 'eval', 'compile', '__import__', 'open',
                         'breakpoint', 'exit', 'quit', 'input', 'memoryview')
            and not k.startswith('_')
        }
        namespace = {
            '__builtins__': _safe_builtins,
            'ToolExecutionContext': ToolExecutionContext,
            'ToolExecutionError': ToolExecutionError,
        }
        try:
            exec(code, namespace)
            handler_func = None
            for key, val in namespace.items():
                if callable(val) and key not in ('ToolExecutionContext', 'ToolExecutionError'):
                    handler_func = val
                    break
            
            if not handler_func:
                raise ToolExecutionError("No handler function found in the provided code.")
                
            return handler_func(args, ctx)
        except Exception as e:
            raise ToolExecutionError(f"Error executing custom tool {name}: {e}")

    new_tool = SigilTool(
        name=name,
        description=description,
        parameters=parameters,
        handler=dynamic_handler,
        custom_source=f'project:{tool_file}',
    )
    try:
        from .workspace_trust import custom_tool_detail, mark_user_item

        # made in this session (writes were allowed): it loads in the next session without a trust
        # question, as long as the file is unchanged (the content hash is part of the item)
        mark_user_item(context.root, 'custom-tool', name, custom_tool_detail(context.root, tool_file))
    except OSError as _ignored:
        _log_ignored(_ignored)

    if context.tool_registry is not None:
        context.tool_registry[name] = new_tool
        
    return (
        f"Successfully created and registered tool: {name}. You can now call it directly.",
        {
            'action': 'create_new_tool',
            'tool_name': name,
            'path': str(tool_file.relative_to(context.root) if tool_file.is_relative_to(context.root) else tool_file),
        }
    )


def _register_custom_tool(registry: dict[str, SigilTool], file_path: Path) -> None:
    """Loads a custom tool from a Python file and adds it to the registry."""
    import importlib.util
    
    name = file_path.stem
    try:
        spec = importlib.util.spec_from_file_location(f"custom_tool_{name}", str(file_path))
        if spec is None or spec.loader is None:
            return
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        tool_name = getattr(module, 'name', name)
        description = getattr(module, 'description', f"Custom tool {name}")
        parameters = getattr(module, 'parameters', {'type': 'object', 'properties': {}})
        handler = getattr(module, 'handler', None)
        
        if handler is None:
            for attr in dir(module):
                val = getattr(module, attr)
                if callable(val) and not attr.startswith('_'):
                    handler = val
                    break
                    
        if handler:
            registry[tool_name] = SigilTool(
                name=tool_name,
                description=description,
                parameters=parameters,
                handler=handler,
            )
    except Exception as e:
        print(f"Warning: Failed to load custom tool {name}: {e}")


_CUSTOM_TOOL_NAME_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}')
_BUILTIN_NAMES: frozenset[str] | None = None


def _builtin_tool_names() -> frozenset[str]:
    global _BUILTIN_NAMES
    if _BUILTIN_NAMES is None:
        _BUILTIN_NAMES = frozenset(default_tool_registry())
    return _BUILTIN_NAMES


def user_custom_tools_dir() -> Path:
    """The user's own custom tools: ~/.sigil/custom_tools/*.py (always loaded)."""
    from .settings import _home

    return _home() / '.sigil' / 'custom_tools'


def project_custom_tools_dir(cwd: Path) -> Path:
    """A folder's custom tools: <cwd>/src/custom_tools/*.py (loaded once the folder is trusted)."""
    return Path(cwd) / 'src' / 'custom_tools'


def custom_tool_files(folder: Path) -> list[Path]:
    try:
        if not folder.is_dir():
            return []
        return sorted(item for item in folder.iterdir()
                      if item.is_file() and item.suffix == '.py' and item.name != '__init__.py')
    except OSError as _ignored:
        _log_ignored(_ignored)
        return []


def custom_tool_name_problem(name: Any, registry: dict[str, SigilTool], *, may_replace_project: bool = False) -> str | None:
    """Why a custom tool may not use this name (None: it may). A custom tool never shadows or
    replaces a built-in tool (or an MCP/plugin tool) or one of the user's own custom tools."""
    if not isinstance(name, str) or not _CUSTOM_TOOL_NAME_RE.fullmatch(name):
        return (f'{name!r} is not a valid tool name (letters, digits and _, starting with a letter or _, '
                'at most 64 characters)')
    if name in _builtin_tool_names():  # also when --tools / --disallowed-tools hid the built-in
        return f'{name!r} is the name of a built-in tool'
    existing = registry.get(name)
    if existing is None:
        return None
    source = getattr(existing, 'custom_source', '')
    if not source:
        return f'{name!r} is the name of a built-in tool'
    if source.startswith('user:'):
        return f"{name!r} is already one of your own custom tools ({source[5:]})"
    return None if may_replace_project else f"{name!r} is already a custom tool ({source.split(':', 1)[-1]})"


def _refuse_custom_tool(path: Path, problem: str) -> None:
    from .debug_log import get_logger, sanitize_log

    print(f'Sigil: custom tool {path} not loaded: {problem}. Rename the file (or its `name`).', file=sys.stderr)
    get_logger('custom_tools').warning('custom tool %s not loaded: %s', path, sanitize_log(problem))


def _load_custom_tool_file(registry: dict[str, SigilTool], path: Path, origin: str) -> None:
    """Import one custom tool file (the exec mechanism of _register_custom_tool, unchanged) and
    register it under its name, unless that name is taken (see custom_tool_name_problem)."""
    problem = custom_tool_name_problem(path.stem, registry)
    if problem is not None:  # checked before the file runs
        _refuse_custom_tool(path, problem)
        return
    loaded: dict[str, SigilTool] = {}
    _register_custom_tool(loaded, path)
    for tool in loaded.values():  # the module may set its own `name`: check that one too
        problem = custom_tool_name_problem(tool.name, registry)
        if problem is not None:
            _refuse_custom_tool(path, problem)
            continue
        from dataclasses import replace as _with

        registry[tool.name] = _with(tool, custom_source=f'{origin}:{path}')


def load_custom_tools(registry: dict[str, SigilTool], cwd: Path) -> None:
    """Custom tools at agent start: the user's own (~/.sigil/custom_tools) always; the folder's
    src/custom_tools only when the folder is trusted (workspace_trust lists them as
    "custom tools: name (file)") or the user made the tool there with create_new_tool.
    Why: a cloned repository's src/custom_tools/*.py used to run as soon as Sigil started in it."""
    for path in custom_tool_files(user_custom_tools_dir()):
        _load_custom_tool_file(registry, path, 'user')
    project_files = custom_tool_files(project_custom_tools_dir(cwd))
    if not project_files:
        return
    from .workspace_trust import allowed, custom_tool_detail, report_skipped

    held_back = False
    for path in project_files:
        if allowed(cwd, 'custom-tool', path.stem, custom_tool_detail(cwd, path)):
            _load_custom_tool_file(registry, path, 'project')
        else:
            held_back = True
    if held_back:
        report_skipped(cwd)


def execute_tool(
    tool_registry: dict[str, SigilTool],
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    tool = tool_registry.get(name)
    if tool is None:
        return ToolExecutionResult(
            name=name,
            ok=False,
            content=f'Unknown tool: {name}',
        )
    return tool.execute(arguments, context)


def execute_tool_streaming(
    tool_registry: dict[str, SigilTool],
    name: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> Iterator[ToolStreamUpdate]:
    tool = tool_registry.get(name)
    if tool is None:
        yield ToolStreamUpdate(
            kind='result',
            result=ToolExecutionResult(
                name=name,
                ok=False,
                content=f'Unknown tool: {name}',
            ),
        )
        return

    if name == 'bash' and arguments.get('run_in_background') is not True:
        yield from _stream_bash(arguments, context)
        return

    result = tool.execute(arguments, context)
    if result.ok and result.content and name != 'delegate_agent':
        yield from _stream_static_text_result(result)
        return
    yield ToolStreamUpdate(kind='result', result=result)


# Skills/Agents tool handlers
def _skill_search(arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
    """Search skills by name or description."""
    query = arguments.get('query', '')
    max_results = arguments.get('max_results', 20)
    
    if not context.skills_runtime:
        return ToolExecutionResult(
            name='skill_search',
            ok=False,
            content='Skills runtime not initialized.',
        )
    
    matched_names = context.skills_runtime.match_skills(query)[:max_results]
    
    if not matched_names:
        return ToolExecutionResult(
            name='skill_search',
            ok=True,
            content=f'No skills found matching: {query}',
        )
    
    lines = [f'Found {len(matched_names)} skill(s) matching "{query}":\n']
    matches_meta = []
    for name in matched_names:
        skill = context.skills_runtime.get_skill(name)
        if skill:
            origin = skill.metadata.origin or 'unknown'
            desc = skill.metadata.description
            short_desc = desc[:100] + '...' if len(desc) > 100 else desc
            lines.append(f'- **{skill.metadata.name}** ({origin}): {short_desc}')
            matches_meta.append({'name': skill.metadata.name, 'origin': skill.metadata.origin})
    
    return ToolExecutionResult(
        name='skill_search',
        ok=True,
        content='\n'.join(lines),
        metadata={'matches': matches_meta},
    )


def _skill_get(arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
    """Get full content of a skill by name."""
    name = arguments.get('name', '')
    
    if not context.skills_runtime:
        return ToolExecutionResult(
            name='skill_get',
            ok=False,
            content='Skills runtime not initialized.',
        )
    
    skill = context.skills_runtime.get_skill(name)
    if not skill:
        return ToolExecutionResult(
            name='skill_get',
            ok=False,
            content=f'Skill not found: {name}',
        )
    
    content = f'# Skill: {skill.metadata.name}\n\n'
    content += f'**Origin:** {skill.metadata.origin or "unknown"}\n'
    content += f'**Description:** {skill.metadata.description}\n\n'
    content += '---\n\n'
    content += skill.content
    
    return ToolExecutionResult(
        name='skill_get',
        ok=True,
        content=content,
        metadata={'name': skill.metadata.name, 'origin': skill.metadata.origin},
    )


def _skill_list(arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
    """List all available skills."""
    max_results = arguments.get('max_results', 100)
    
    if not context.skills_runtime:
        return ToolExecutionResult(
            name='skill_list',
            ok=False,
            content='Skills runtime not initialized.',
        )
    
    skill_names = context.skills_runtime.list_skills()[:max_results]
    
    if not skill_names:
        return ToolExecutionResult(
            name='skill_list',
            ok=True,
            content='No skills registered.',
        )
    
    lines = [f'Available skills ({len(skill_names)}):\n']
    for name in skill_names:
        skill = context.skills_runtime.get_skill(name)
        if skill:
            origin = skill.metadata.origin or 'unknown'
            lines.append(f'- **{name}** ({origin})')
        else:
            lines.append(f'- **{name}**')
    
    return ToolExecutionResult(
        name='skill_list',
        ok=True,
        content='\n'.join(lines),
        metadata={'count': len(skill_names)},
    )


def _agent_search(arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
    """Search agents by name or description."""
    query = arguments.get('query', '')
    max_results = arguments.get('max_results', 20)
    
    if not context.skills_runtime:
        return ToolExecutionResult(
            name='agent_search',
            ok=False,
            content='Skills runtime not initialized.',
        )
    
    matched_names = context.skills_runtime.match_agents(query)[:max_results]
    
    if not matched_names:
        return ToolExecutionResult(
            name='agent_search',
            ok=True,
            content=f'No agents found matching: {query}',
        )
    
    lines = [f'Found {len(matched_names)} agent(s) matching "{query}":\n']
    matches_meta = []
    for name in matched_names:
        agent = context.skills_runtime.get_agent(name)
        if agent:
            model = agent.metadata.model or 'default'
            tools = ', '.join(agent.metadata.tools) if agent.metadata.tools else 'none'
            desc = agent.metadata.description
            short_desc = desc[:80] + '...' if len(desc) > 80 else desc
            lines.append(f'- **{agent.metadata.name}** (model: {model}, tools: {tools}): {short_desc}')
            matches_meta.append({'name': agent.metadata.name, 'model': agent.metadata.model})
    
    return ToolExecutionResult(
        name='agent_search',
        ok=True,
        content='\n'.join(lines),
        metadata={'matches': matches_meta},
    )


def _agent_get(arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
    """Get full content of an agent persona by name."""
    name = arguments.get('name', '')
    
    if not context.skills_runtime:
        return ToolExecutionResult(
            name='agent_get',
            ok=False,
            content='Skills runtime not initialized.',
        )
    
    agent = context.skills_runtime.get_agent(name)
    if not agent:
        return ToolExecutionResult(
            name='agent_get',
            ok=False,
            content=f'Agent not found: {name}',
        )
    
    content = f'# Agent: {agent.metadata.name}\n\n'
    content += f'**Description:** {agent.metadata.description}\n'
    content += f'**Model:** {agent.metadata.model or "default"}\n'
    content += f'**Tools:** {", ".join(agent.metadata.tools) if agent.metadata.tools else "all"}\n\n'
    content += '---\n\n'
    content += agent.content
    
    return ToolExecutionResult(
        name='agent_get',
        ok=True,
        content=content,
        metadata={'name': agent.metadata.name, 'model': agent.metadata.model, 'tools': agent.metadata.tools},
    )


def _agent_list(arguments: dict[str, Any], context: ToolExecutionContext) -> ToolExecutionResult:
    """List all available agents."""
    max_results = arguments.get('max_results', 100)
    
    if not context.skills_runtime:
        return ToolExecutionResult(
            name='agent_list',
            ok=False,
            content='Skills runtime not initialized.',
        )
    
    agent_names = context.skills_runtime.list_agents()[:max_results]
    
    if not agent_names:
        return ToolExecutionResult(
            name='agent_list',
            ok=True,
            content='No agents registered.',
        )
    
    lines = [f'Available agents ({len(agent_names)}):\n']
    for name in agent_names:
        agent = context.skills_runtime.get_agent(name)
        if agent:
            model = agent.metadata.model or 'default'
            lines.append(f'- **{name}** (model: {model})')
        else:
            lines.append(f'- **{name}**')
    
    return ToolExecutionResult(
        name='agent_list',
        ok=True,
        content='\n'.join(lines),
        metadata={'count': len(agent_names)},
    )


def default_tool_registry() -> dict[str, SigilTool]:
    tools = [
        SigilTool(
            name='create_new_tool',
            description='Create a new custom Python tool and register it for the current session. The tool will be saved to src/custom_tools/ and exists for the duration of the session.',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Name of the tool (snake_case)'},
                    'description': {'type': 'string', 'description': 'High-level description of what the tool does'},
                    'parameters_json': {'type': 'string', 'description': 'JSON string of the tool\'s parameters schema (OpenAI format)'},
                    'code': {'type': 'string', 'description': 'Python source code for the tool handler function. It must take (arguments, context) as input.'},
                },
                'required': ['name', 'description', 'parameters_json', 'code'],
            },
            handler=_create_new_tool,
        ),
        SigilTool(
            name='list_dir',
            description='List files and directories under a workspace path.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Relative path from workspace root.'},
                    'max_entries': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                },
            },
            handler=_list_dir,
        ),
        SigilTool(
            name='read_file',
            description=('Read a file inside the workspace: UTF-8 text (optionally a line range), '
                         'an image (png, jpg, gif, webp; shown to vision models) or a PDF (text by '
                         'page; offset and limit pick the pages).'),
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string', 'description': 'Relative file path from workspace root.'},
                    'start_line': {'type': 'integer', 'minimum': 1},
                    'end_line': {'type': 'integer', 'minimum': 1},
                    'offset': {'type': 'integer', 'minimum': 1, 'description': 'PDF only: first page (1-based).'},
                    'limit': {'type': 'integer', 'minimum': 1, 'description': 'PDF only: number of pages.'},
                    'allow_large': {
                        'type': 'boolean',
                        'description': (
                            'Read past the usual output size limit in one call (hard cap '
                            f'{_READ_ALLOW_LARGE_MAX_CHARS} chars). Use only when the whole '
                            'file is needed and the context has room.'
                        ),
                    },
                },
                'required': ['path'],
            },
            handler=_read_file,
        ),
        SigilTool(
            name='write_file',
            description='Write a complete file inside the workspace. Creates parent directories when needed.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['path', 'content'],
            },
            handler=_write_file,
        ),
        SigilTool(
            name='edit_file',
            description='Replace text inside a workspace file using exact string matching.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'old_text': {'type': 'string'},
                    'new_text': {'type': 'string'},
                    'replace_all': {'type': 'boolean'},
                },
                'required': ['path', 'old_text', 'new_text'],
            },
            handler=_edit_file,
        ),
        SigilTool(
            name='notebook_edit',
            description='Edit a Jupyter notebook cell by replacing or appending source in a .ipynb file.',
            parameters={
                'type': 'object',
                'properties': {
                    'path': {'type': 'string'},
                    'cell_index': {'type': 'integer', 'minimum': 0},
                    'source': {'type': 'string'},
                    'cell_type': {'type': 'string'},
                    'create_cell': {'type': 'boolean'},
                },
                'required': ['path', 'cell_index', 'source'],
            },
            handler=_notebook_edit,
        ),
        SigilTool(
            name='glob_search',
            description='Find files matching a glob pattern inside the workspace.',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                },
                'required': ['pattern'],
            },
            handler=_glob_search,
        ),
        SigilTool(
            name='grep_search',
            description='Search for a string or regular expression inside workspace files.',
            parameters={
                'type': 'object',
                'properties': {
                    'pattern': {'type': 'string'},
                    'path': {'type': 'string'},
                    'literal': {'type': 'boolean'},
                    'max_matches': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                },
                'required': ['pattern'],
            },
            handler=_grep_search,
        ),
        SigilTool(
            name='bash',
            description=(
                'Run a shell command in the workspace. Use sparingly and prefer dedicated file tools for edits. '
                'On Windows this is Git-Bash: use forward slashes (C:/Users/... or /c/Users/...), never backslashes. '
                'Output redirection (>, >>) is allowed only into the session scratchpad directory. '
                f'Default timeout is {DEFAULT_BASH_TIMEOUT_SECONDS:.0f}s; set timeout_seconds '
                f'(max {MAX_BASH_TIMEOUT_SECONDS:.0f}) for long jobs such as installs or test suites. '
                'Set run_in_background for dev servers, watchers and long runs: it returns a shell id '
                'at once; read new output with bash_output and stop it with bash_kill.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'command': {'type': 'string'},
                    'timeout_seconds': {'type': 'number', 'minimum': 1, 'maximum': MAX_BASH_TIMEOUT_SECONDS},
                    'run_in_background': {'type': 'boolean'},
                },
                'required': ['command'],
            },
            handler=_run_bash,
        ),
        SigilTool(
            name='exit_plan_mode',
            description=(
                'In plan mode: present your finished plan and ask the user to approve it. '
                'If they approve, plan mode ends and you carry out the plan.'
            ),
            parameters={
                'type': 'object',
                'properties': {'plan': {'type': 'string', 'description': 'The plan, in Markdown.'}},
                'required': ['plan'],
            },
            handler=_delegate_agent_placeholder,
        ),
        SigilTool(
            name='enter_plan_mode',
            description='Switch to plan mode: read-only research, then a plan for the user to approve.',
            parameters={'type': 'object', 'properties': {}},
            handler=_delegate_agent_placeholder,
        ),
        SigilTool(
            name='wait_for_mcp_servers',
            description=(
                'Wait until MCP servers that are still starting are connected (or failed), '
                'then report the status of each server.'
            ),
            parameters={
                'type': 'object',
                'properties': {'timeout_seconds': {'type': 'number', 'minimum': 0, 'maximum': 120}},
            },
            handler=_wait_for_mcp_servers,
        ),
        SigilTool(
            name='bash_output',
            description=(
                'Read new output from a background shell started with bash run_in_background, '
                'and its status (running, completed, failed, killed).'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'id': {'type': 'string', 'description': 'Shell id, e.g. bash_1'},
                    'filter': {'type': 'string', 'description': 'Regex: only return matching lines'},
                    'wait_seconds': {'type': 'number', 'minimum': 0, 'maximum': 60,
                                     'description': 'Wait up to this long for new (matching) output'},
                },
                'required': ['id'],
            },
            handler=_bash_output,
        ),
        SigilTool(
            name='skill',
            description=(
                'Run a custom command or skill by name (the same as the user typing /name args): '
                'returns its instructions to follow now.'
            ),
            parameters={
                'type': 'object',
                'properties': {'name': {'type': 'string'}, 'arguments': {'type': 'string'}},
                'required': ['name'],
            },
            handler=_skill_tool,
        ),
        SigilTool(
            name='bash_kill',
            description='Stop a background shell and every process it started.',
            parameters={
                'type': 'object',
                'properties': {'id': {'type': 'string', 'description': 'Shell id, e.g. bash_1'}},
                'required': ['id'],
            },
            handler=_bash_kill,
        ),
        SigilTool(
            name='LSP',
            description='Use local LSP-style code intelligence for definitions, references, hover, symbols, and call hierarchy.',
            parameters={
                'type': 'object',
                'properties': {
                    'operation': {
                        'type': 'string',
                        'enum': [
                            'goToDefinition',
                            'findReferences',
                            'hover',
                            'documentSymbol',
                            'workspaceSymbol',
                            'goToImplementation',
                            'prepareCallHierarchy',
                            'incomingCalls',
                            'outgoingCalls',
                        ],
                    },
                    'file_path': {'type': 'string'},
                    'line': {'type': 'integer', 'minimum': 1},
                    'character': {'type': 'integer', 'minimum': 1},
                    'query': {'type': 'string'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
                'required': ['operation', 'file_path', 'line', 'character'],
            },
            handler=_lsp_query,
        ),
        SigilTool(
            name='web_fetch',
            description=(
                'Fetch a URL and return its content converted to markdown. HTTP is upgraded to '
                'HTTPS; results are cached for 15 minutes; redirects to another host are reported '
                'instead of followed. Pass `prompt` to have the page read by a model and get back '
                'only the answer to your question (keeps large pages out of context).'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'url': {'type': 'string', 'description': 'The URL to fetch.'},
                    'prompt': {
                        'type': 'string',
                        'description': 'Optional: what to extract from the page. When set, the answer is returned instead of the raw page.',
                    },
                    'max_chars': {
                        'type': 'integer',
                        'description': 'Maximum number of characters to fetch.',
                        'default': 10000,
                    },
                },
                'required': ['url'],
            },
            handler=_web_fetch,
        ),
        SigilTool(
            name='rollback_workspace',
            description='Revert the workspace to the last git checkpoint. Use this if a previous edit caused a build failure or a regression.',
            parameters={'type': 'object', 'properties': {}},
            handler=_rollback_workspace_tool,
        ),
        SigilTool(
            name='search_status',
            description='Show the local search runtime summary or a specific configured search provider.',
            parameters={
                'type': 'object',
                'properties': {
                    'provider': {'type': 'string'},
                },
            },
            handler=_search_status,
        ),
        SigilTool(
            name='search_list_providers',
            description='List configured local search providers from workspace search manifests and environment configuration.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_providers': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_search_list_providers,
        ),
        SigilTool(
            name='search_activate_provider',
            description='Set the active local search provider for the current workspace.',
            parameters={
                'type': 'object',
                'properties': {
                    'provider': {'type': 'string'},
                },
                'required': ['provider'],
            },
            handler=_search_activate_provider,
        ),
        SigilTool(
            name='web_search',
            description=(
                'Search the web and return ranked results (title, URL, snippet). Use for current '
                'information beyond your training data. Cite sources as markdown links in your answer; '
                'follow up with web_fetch to read a page in full.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'provider': {'type': 'string'},
                    'domains': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Only include results from these domains.',
                    },
                    'blocked_domains': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Never include results from these domains.',
                    },
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                },
                'required': ['query'],
            },
            handler=_web_search,
        ),
        SigilTool(
            name='tool_search',
            description='Search the active tool registry by tool name or description.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                },
                'required': ['query'],
            },
            handler=_tool_search,
        ),
        SigilTool(
            name='sleep',
            description='Pause execution briefly for bounded local wait flows.',
            parameters={
                'type': 'object',
                'properties': {
                    'seconds': {'type': 'number', 'minimum': 0.0, 'maximum': 5.0},
                },
                'required': ['seconds'],
            },
            handler=_sleep,
        ),
        SigilTool(
            name='ask_user_question',
            description='Request an answer from the local ask-user runtime using queued or interactive answers.',
            parameters={
                'type': 'object',
                'properties': {
                    'question': {'type': 'string'},
                    'header': {'type': 'string'},
                    'question_id': {'type': 'string'},
                    'choices': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'allow_free_text': {'type': 'boolean'},
                },
                'required': ['question'],
            },
            handler=_ask_user_question,
        ),
        SigilTool(
            name='account_status',
            description='Show local account runtime summary or a specific configured account profile.',
            parameters={
                'type': 'object',
                'properties': {
                    'profile': {'type': 'string'},
                },
            },
            handler=_account_status,
        ),
        SigilTool(
            name='account_list_profiles',
            description='List configured local account profiles from workspace account manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_profiles': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_account_list_profiles,
        ),
        SigilTool(
            name='account_login',
            description='Activate a local account profile or ephemeral account identity and persist it as the active account session.',
            parameters={
                'type': 'object',
                'properties': {
                    'target': {'type': 'string'},
                    'provider': {'type': 'string'},
                    'auth_mode': {'type': 'string'},
                },
                'required': ['target'],
            },
            handler=_account_login,
        ),
        SigilTool(
            name='account_logout',
            description='Clear the active local account session state.',
            parameters={
                'type': 'object',
                'properties': {
                    'reason': {'type': 'string'},
                },
            },
            handler=_account_logout,
        ),
        SigilTool(
            name='config_list',
            description='List merged or source-specific workspace config keys from local settings files.',
            parameters={
                'type': 'object',
                'properties': {
                    'source': {'type': 'string'},
                    'prefix': {'type': 'string'},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                },
            },
            handler=_config_list,
        ),
        SigilTool(
            name='config_get',
            description='Read a merged or source-specific workspace config value by dotted key path.',
            parameters={
                'type': 'object',
                'properties': {
                    'key_path': {'type': 'string'},
                    'source': {'type': 'string'},
                },
                'required': ['key_path'],
            },
            handler=_config_get,
        ),
        SigilTool(
            name='config_set',
            description='Write a workspace config value by dotted key path into a chosen config source.',
            parameters={
                'type': 'object',
                'properties': {
                    'key_path': {'type': 'string'},
                    'source': {'type': 'string'},
                    'value': {
                        'oneOf': [
                            {'type': 'string'},
                            {'type': 'number'},
                            {'type': 'integer'},
                            {'type': 'boolean'},
                            {'type': 'array'},
                            {'type': 'object'},
                            {'type': 'null'},
                        ]
                    },
                },
                'required': ['key_path', 'value'],
            },
            handler=_config_set,
        ),
        SigilTool(
            name='mcp_list_resources',
            description='List local MCP resources discovered from workspace MCP manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_resources': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_mcp_list_resources,
        ),
        SigilTool(
            name='mcp_read_resource',
            description='Read a local MCP resource by URI from workspace MCP manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'uri': {'type': 'string'},
                    'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': DEFAULT_MAX_RESULT_SIZE_CHARS},
                },
                'required': ['uri'],
            },
            handler=_mcp_read_resource,
        ),
        SigilTool(
            name='mcp_list_tools',
            description='List MCP tools exposed by configured MCP servers over real transport.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'server': {'type': 'string'},
                    'max_tools': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_mcp_list_tools,
        ),
        SigilTool(
            name='mcp_call_tool',
            description='Call an MCP tool exposed by a configured MCP server over real transport.',
            parameters={
                'type': 'object',
                'properties': {
                    'tool_name': {'type': 'string'},
                    'server': {'type': 'string'},
                    'arguments': {'type': 'object'},
                    'max_chars': {'type': 'integer', 'minimum': 1, 'maximum': DEFAULT_MAX_RESULT_SIZE_CHARS},
                },
                'required': ['tool_name'],
            },
            handler=_mcp_call_tool,
        ),
        SigilTool(
            name='remote_status',
            description='Show the local remote runtime summary or a specific configured remote profile.',
            parameters={
                'type': 'object',
                'properties': {
                    'profile': {'type': 'string'},
                },
            },
            handler=_remote_status,
        ),
        SigilTool(
            name='remote_list_profiles',
            description='List configured local remote profiles from workspace remote manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'mode': {'type': 'string'},
                    'max_profiles': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_remote_list_profiles,
        ),
        SigilTool(
            name='remote_connect',
            description='Activate a local remote target or configured remote profile and persist it as the active connection.',
            parameters={
                'type': 'object',
                'properties': {
                    'target': {'type': 'string'},
                    'mode': {'type': 'string'},
                },
                'required': ['target'],
            },
            handler=_remote_connect,
        ),
        SigilTool(
            name='remote_disconnect',
            description='Clear the active local remote connection state.',
            parameters={
                'type': 'object',
                'properties': {
                    'reason': {'type': 'string'},
                },
            },
            handler=_remote_disconnect,
        ),
        SigilTool(
            name='worktree_status',
            description='Show the current managed git worktree session status.',
            parameters={
                'type': 'object',
                'properties': {},
            },
            handler=_worktree_status,
        ),
        SigilTool(
            name='worktree_enter',
            description='Create an isolated git worktree and switch the current agent session into it.',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                },
            },
            handler=_worktree_enter,
        ),
        SigilTool(
            name='worktree_exit',
            description='Leave the active managed worktree session and optionally remove the worktree.',
            parameters={
                'type': 'object',
                'properties': {
                    'action': {'type': 'string'},
                    'discard_changes': {'type': 'boolean'},
                },
            },
            handler=_worktree_exit,
        ),
        SigilTool(
            name='workflow_list',
            description='List local workflow definitions discovered from workspace workflow manifests.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_workflows': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_workflow_list,
        ),
        SigilTool(
            name='workflow_get',
            description='Show one local workflow definition by name.',
            parameters={
                'type': 'object',
                'properties': {
                    'workflow_name': {'type': 'string'},
                },
                'required': ['workflow_name'],
            },
            handler=_workflow_get,
        ),
        SigilTool(
            name='workflow_run',
            description='Record and render a local workflow execution request from a workflow manifest.',
            parameters={
                'type': 'object',
                'properties': {
                    'workflow_name': {'type': 'string'},
                    'arguments': {'type': 'object'},
                },
                'required': ['workflow_name'],
            },
            handler=_workflow_run,
        ),
        SigilTool(
            name='remote_trigger',
            description='List, inspect, create, update, or run local remote triggers.',
            parameters={
                'type': 'object',
                'properties': {
                    'action': {'type': 'string'},
                    'trigger_id': {'type': 'string'},
                    'body': {'type': 'object'},
                    'query': {'type': 'string'},
                    'max_triggers': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
                'required': ['action'],
            },
            handler=_remote_trigger,
        ),
        SigilTool(
            name='plan_get',
            description='Show the current local runtime plan.',
            parameters={
                'type': 'object',
                'properties': {},
            },
            handler=_plan_get,
        ),
        SigilTool(
            name='update_plan',
            description='Replace the current local runtime plan with a structured multi-step plan and optionally sync it to tasks.',
            parameters={
                'type': 'object',
                'properties': {
                    'explanation': {'type': 'string'},
                    'sync_tasks': {'type': 'boolean'},
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'step': {'type': 'string'},
                                'status': {'type': 'string'},
                                'task_id': {'type': 'string'},
                                'description': {'type': 'string'},
                                'priority': {'type': 'string'},
                                'active_form': {'type': 'string'},
                                'owner': {'type': 'string'},
                                'depends_on': {
                                    'type': 'array',
                                    'items': {'type': 'string'},
                                },
                            },
                            'required': ['step'],
                        },
                    },
                },
                'required': ['items'],
            },
            handler=_update_plan,
        ),
        SigilTool(
            name='plan_clear',
            description='Clear the current local runtime plan and optionally sync the task runtime.',
            parameters={
                'type': 'object',
                'properties': {
                    'sync_tasks': {'type': 'boolean'},
                },
            },
            handler=_plan_clear,
        ),
        SigilTool(
            name='task_next',
            description='Show the next actionable tasks from the local runtime task list.',
            parameters={
                'type': 'object',
                'properties': {
                    'max_tasks': {'type': 'integer', 'minimum': 1, 'maximum': 50},
                },
            },
            handler=_task_next,
        ),
        SigilTool(
            name='team_list',
            description='List locally configured collaboration teams.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'max_teams': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_team_list,
        ),
        SigilTool(
            name='team_get',
            description='Show a locally configured collaboration team by name.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                },
                'required': ['team_name'],
            },
            handler=_team_get,
        ),
        SigilTool(
            name='team_create',
            description='Create a locally stored collaboration team.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                    'description': {'type': 'string'},
                    'members': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'},
                },
                'required': ['team_name'],
            },
            handler=_team_create,
        ),
        SigilTool(
            name='team_delete',
            description='Delete a locally stored collaboration team and its recorded messages.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                },
                'required': ['team_name'],
            },
            handler=_team_delete,
        ),
        SigilTool(
            name='send_message',
            description='Send a local collaboration message to a team or teammate and persist it in the team runtime.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                    'message': {'type': 'string'},
                    'sender': {'type': 'string'},
                    'recipient': {'type': 'string'},
                    'metadata': {'type': 'object'},
                },
                'required': ['team_name', 'message'],
            },
            handler=_send_message,
        ),
        SigilTool(
            name='team_messages',
            description='Show locally recorded collaboration messages for all teams or one team.',
            parameters={
                'type': 'object',
                'properties': {
                    'team_name': {'type': 'string'},
                    'limit': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_team_messages,
        ),
        SigilTool(
            name='task_list',
            description='List locally stored runtime tasks.',
            parameters={
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'actionable_only': {'type': 'boolean'},
                    'max_tasks': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_task_list,
        ),
        SigilTool(
            name='task_get',
            description='Show a locally stored runtime task by id.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_get,
        ),
        SigilTool(
            name='task_create',
            description='Create a locally stored runtime task.',
            parameters={
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'status': {'type': 'string'},
                    'priority': {'type': 'string'},
                    'task_id': {'type': 'string'},
                    'active_form': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'blocks': {'type': 'array', 'items': {'type': 'string'}},
                    'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'},
                },
                'required': ['title'],
            },
            handler=_task_create,
        ),
        SigilTool(
            name='task_update',
            description='Update a locally stored runtime task by id.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'title': {'type': 'string'},
                    'description': {'type': 'string'},
                    'status': {'type': 'string'},
                    'priority': {'type': 'string'},
                    'active_form': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'blocks': {'type': 'array', 'items': {'type': 'string'}},
                    'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                    'metadata': {'type': 'object'},
                },
                'required': ['task_id'],
            },
            handler=_task_update,
        ),
        SigilTool(
            name='task_start',
            description='Mark a task as in progress, or blocked if dependencies are unresolved.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'active_form': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_start,
        ),
        SigilTool(
            name='task_complete',
            description='Mark a task as completed and release blocked dependents when possible.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_complete,
        ),
        SigilTool(
            name='task_block',
            description='Mark a task as blocked with optional dependencies and reason.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                    'reason': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_block,
        ),
        SigilTool(
            name='task_cancel',
            description='Mark a task as cancelled with an optional reason.',
            parameters={
                'type': 'object',
                'properties': {
                    'task_id': {'type': 'string'},
                    'reason': {'type': 'string'},
                },
                'required': ['task_id'],
            },
            handler=_task_cancel,
        ),
        SigilTool(
            name='todo_write',
            description='Replace the current local runtime task list with a structured todo list.',
            parameters={
                'type': 'object',
                'properties': {
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'task_id': {'type': 'string'},
                                'title': {'type': 'string'},
                                'description': {'type': 'string'},
                                'status': {'type': 'string'},
                                'priority': {'type': 'string'},
                                'active_form': {'type': 'string'},
                                'owner': {'type': 'string'},
                                'blocks': {'type': 'array', 'items': {'type': 'string'}},
                                'blocked_by': {'type': 'array', 'items': {'type': 'string'}},
                                'metadata': {'type': 'object'},
                            },
                            'required': ['title'],
                        },
                    },
                },
                'required': ['items'],
            },
            handler=_todo_write,
        ),
        SigilTool(
            name='delegate_agent',
            description=(
                'Delegate a subtask to a nested coding agent and return its summary. '
                'Set agent to pick a type: "Explore" (fast, read-only search and reading; keeps '
                'big search output out of your context), "Plan" (read-only research that returns '
                'a plan), "general-purpose" (all tools), or the name of an agent defined in '
                '.sigil/agents/.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string'},
                    'agent': {
                        'type': 'string',
                        'description': 'Subagent type: Explore, Plan, general-purpose, or a .sigil/agents name.',
                    },
                    'subagent_type': {'type': 'string', 'description': 'Same as agent.'},
                    'isolation': {'type': 'string', 'enum': ['worktree'],
                                  'description': 'worktree: the child works in its own git worktree'},
                    'resume_agent': {'type': 'string',
                                     'description': 'Follow up with a finished subagent by its label or agent type'},
                    'subtasks': {
                        'type': 'array',
                        'items': {
                            'oneOf': [
                                {'type': 'string'},
                                {
                                    'type': 'object',
                                    'properties': {
                                        'prompt': {'type': 'string'},
                                        'label': {'type': 'string'},
                                        'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                                        'resume_session_id': {'type': 'string'},
                                        'session_id': {'type': 'string'},
                                        'depends_on': {
                                            'type': 'array',
                                            'items': {'type': 'string'},
                                        },
                                    },
                                    'required': ['prompt'],
                                },
                            ]
                        },
                    },
                    'resume_session_id': {'type': 'string'},
                    'session_id': {'type': 'string'},
                    'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 20},
                    'allow_write': {'type': 'boolean'},
                    'allow_shell': {'type': 'boolean'},
                    'include_parent_context': {'type': 'boolean'},
                    'continue_on_error': {'type': 'boolean'},
                    'max_failures': {'type': 'integer', 'minimum': 0, 'maximum': 20},
                    'strategy': {'type': 'string'},
                },
            },
            handler=_delegate_agent_placeholder,
        ),
        SigilTool(
            name='background_agent_start',
            description=(
                'Start a coding agent in a separate background process and return immediately with its id, '
                'so the conversation can continue while it works. Use for long, independent tasks. '
                'Use delegate_agent instead when you need the result before your next step. '
                'The agent only gets write/shell access if requested AND this session already has it.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string', 'description': 'Full, self-contained task description.'},
                    'max_turns': {'type': 'integer', 'minimum': 1, 'maximum': 40},
                    'allow_write': {'type': 'boolean'},
                    'allow_shell': {'type': 'boolean'},
                    'model': {'type': 'string', 'description': 'Optional model override; defaults to the current model.'},
                },
                'required': ['prompt'],
            },
            handler=_background_agent_start,
        ),
        SigilTool(
            name='background_agent_status',
            description=(
                'Without background_id: list all background agents and their status. '
                'With background_id: show its status and the last lines of its log (the final answer appears there when it finishes).'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'background_id': {'type': 'string'},
                    'tail': {'type': 'integer', 'minimum': 1, 'maximum': 400},
                },
            },
            handler=_background_agent_status,
        ),
        SigilTool(
            name='background_agent_restart',
            description=(
                'Start a stopped background agent again by id. It continues its saved session '
                '(or runs its prompt again if it stopped before saving); give prompt for new instructions.'
            ),
            parameters={
                'type': 'object',
                'properties': {'background_id': {'type': 'string'}, 'prompt': {'type': 'string'}},
                'required': ['background_id'],
            },
            handler=_background_agent_restart,
        ),
        SigilTool(
            name='background_agent_kill',
            description='Stop a running background agent and everything it started.',
            parameters={
                'type': 'object',
                'properties': {'background_id': {'type': 'string'}},
                'required': ['background_id'],
            },
            handler=_background_agent_kill,
        ),
        SigilTool(
            name='skill_search',
            description='Search the skills registry by name or description to find relevant skills for the current task.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query to match against skill names and descriptions.'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                },
                'required': ['query'],
            },
            handler=_skill_search,
        ),
        SigilTool(
            name='skill_get',
            description='Load and return the full content of a skill by name. Use this when you need detailed guidance from a skill.',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Name of the skill to load.'},
                },
                'required': ['name'],
            },
            handler=_skill_get,
        ),
        SigilTool(
            name='skill_list',
            description='List all available skills in the registry.',
            parameters={
                'type': 'object',
                'properties': {
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_skill_list,
        ),
        SigilTool(
            name='agent_search',
            description='Search the agents registry by name or description to find relevant specialist agents for the current task.',
            parameters={
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query to match against agent names and descriptions.'},
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                },
                'required': ['query'],
            },
            handler=_agent_search,
        ),
        SigilTool(
            name='agent_get',
            description='Load and return the full content of an agent persona by name. Use this to get the persona for delegate_agent.',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Name of the agent to load.'},
                },
                'required': ['name'],
            },
            handler=_agent_get,
        ),
        SigilTool(
            name='agent_list',
            description='List all available specialist agents in the registry.',
            parameters={
                'type': 'object',
                'properties': {
                    'max_results': {'type': 'integer', 'minimum': 1, 'maximum': 200},
                },
            },
            handler=_agent_list,
        ),
    ]
    if _powershell_tool_enabled():
        tools.append(SigilTool(
            name='powershell',
            description=(
                'Run a PowerShell command in the workspace (Windows). Needs --allow-shell; '
                'commands that delete or shut down need --unsafe.'
            ),
            parameters={
                'type': 'object',
                'properties': {
                    'command': {'type': 'string'},
                    'timeout_seconds': {'type': 'number', 'minimum': 1, 'maximum': MAX_BASH_TIMEOUT_SECONDS},
                },
                'required': ['command'],
            },
            handler=_run_powershell,
        ))
    return {tool.name: tool for tool in tools}


def serialize_tool_result(result: ToolExecutionResult) -> str:
    payload = {
        'tool': result.name,
        'ok': result.ok,
        'content': result.content,
    }
    if result.metadata:
        payload['metadata'] = result.metadata
    return json.dumps(payload, ensure_ascii=True, indent=2)


_READ_ALLOW_LARGE_MAX_CHARS = 500_000


def _truncate_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f'{head}\n...[truncated]...\n{tail}'


def _snapshot_text(text: str, limit: int = 240) -> str:
    normalized = ' '.join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + '...'


def _require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ToolExecutionError(f'{key} must be a non-empty string')
    return value


def _coerce_int(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolExecutionError(f'{key} must be an integer')
    return value


def _coerce_float(arguments: dict[str, Any], key: str, default: float) -> float:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolExecutionError(f'{key} must be a number')
    return float(value)


def _display_path(path: Path, context: ToolExecutionContext) -> Path:
    """Path relative to the workspace, or absolute when it is outside it."""
    try:
        return path.relative_to(context.root)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return path


def _path_access(context: ToolExecutionContext):
    """The workspace / --add-dir / other-directories rules for this context."""
    access = getattr(context, 'path_access', None)
    if access is None:
        from .path_access import path_access_for

        access = path_access_for(context.root, tuple(context.additional_working_directories))
        context.path_access = access
    return access


def _resolve_path(
    raw_path: str,
    context: ToolExecutionContext,
    *,
    allow_missing: bool = True,
    for_write: bool = False,
) -> Path:
    from .path_access import PathAccessError

    expanded = Path(raw_path).expanduser()
    candidate = expanded if expanded.is_absolute() else context.root / expanded
    # '..' is fine as long as the resolved path is allowed.
    resolved = candidate.resolve(strict=not allow_missing)
    try:
        return _path_access(context).check(resolved, write=for_write, label=raw_path)
    except PathAccessError as exc:
        raise ToolExecutionError(str(exc)) from exc


def _ensure_write_allowed(context: ToolExecutionContext) -> None:
    if not context.permissions.allow_file_write:
        raise ToolPermissionError(
            'File write tools are disabled. Re-run with --allow-write to enable edits.'
        )


def _redirect_safe_dirs(context: ToolExecutionContext) -> tuple[Path, ...]:
    """Folders shell output redirection (>, >>) may write into."""
    dirs: list[Path] = [context.scratchpad_root] if context.scratchpad_root else []
    access = _path_access(context)
    if access.enabled and context.permissions.allow_file_write:
        # With the other-directories switch on, the added and named folders too.
        dirs.extend(path for path in (*access.added, *access.allowed) if not access.is_protected(path))
    return tuple(dirs)


def _windows_sandbox_limits() -> dict[str, int] | None:
    from .sandbox import windows_limits

    return windows_limits()


def _run_in_windows_job(argv: list[str], context: ToolExecutionContext, timeout: float | None):
    """Run a command inside a limited Job Object (opt-in Windows sandbox)."""
    from .background_shells import _CREATE_SUSPENDED, _win_contain, _win_terminate_job

    process = subprocess.Popen(argv, cwd=context.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace',
                               env=_build_subprocess_env(context), creationflags=_WIN_CHILD_FLAGS | _CREATE_SUSPENDED)
    job = _win_contain(process, _windows_sandbox_limits())
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if job is not None:
            _win_terminate_job(job)
            job = None
        process.kill()
        process.communicate()
        raise
    finally:
        if job is not None:
            _win_terminate_job(job)  # nothing it started outlives the command
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _sandbox_argv(command: str, context: ToolExecutionContext) -> list[str] | None:
    """The bash command wrapped in the opt-in sandbox (see sandbox.py), or None."""
    from .sandbox import SandboxUnavailable, sandbox_argv

    extra = tuple(str(p) for p in (*context.additional_working_directories,
                                   *([context.scratchpad_root] if context.scratchpad_root else [])))
    try:
        return sandbox_argv(_BASH_EXECUTABLE, command, context.root, extra)
    except SandboxUnavailable as exc:
        raise ToolExecutionError(str(exc)) from exc


def _ensure_shell_allowed(command: str, context: ToolExecutionContext) -> str:
    """Raise when the command may not run; return a shell-guard warning (or '')."""
    from .bash_security import check_shell_security
    from .sandbox import shell_allowed

    allowed, message = check_shell_security(
        command,
        allow_shell=shell_allowed(context.permissions.allow_shell_commands),
        allow_destructive=context.permissions.allow_destructive_shell_commands,
        cwd=context.root,
        safe_write_dirs=_redirect_safe_dirs(context),
        on_ask=(lambda _reason: True) if context.approved else None,
        guard=context.shell_guard,
    )
    if not allowed:
        raise ToolPermissionError(message)
    return message


def _list_dir(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    raw_path = arguments.get('path', '.')
    if not isinstance(raw_path, str):
        raise ToolExecutionError('path must be a string')
    max_entries = _coerce_int(arguments, 'max_entries', 200)
    target = _resolve_path(raw_path, context)
    if not target.exists():
        raise ToolExecutionError(f'Path not found: {raw_path}')
    if not target.is_dir():
        raise ToolExecutionError(f'Path is not a directory: {raw_path}')
    entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    lines: list[str] = []
    for entry in entries[:max_entries]:
        kind = 'dir' if entry.is_dir() else 'file'
        rel = _display_path(entry, context)
        lines.append(f'{kind}\t{rel}')
    if len(entries) > max_entries:
        lines.append(f'... truncated at {max_entries} entries ...')
    return '\n'.join(lines) if lines else '(empty directory)'


_GREP_SKIP_DIRS = frozenset({'.git', '.hg', '.svn', 'node_modules', '__pycache__', '.venv', 'venv', '.tox',
                             '.mypy_cache', '.pytest_cache', '.ruff_cache', '.port_sessions'})


def _env_cap(name: str, default: int) -> int:
    """A size/count limit from the environment; 0 (or a negative number) means no limit."""
    try:
        value = int(os.environ.get(name, '') or default)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        value = default
    return value if value > 0 else 0


def _read_text_capped(target: Path, arguments: dict[str, Any]) -> tuple[str, str]:
    """The file's text, bounded by SIGIL_READ_MAX_BYTES (default 50 MB; 0 = no cap): a bigger file
    is read only in part (its first part, or with start_line/end_line the lines up to end_line), so a
    multi-GB file cannot fill memory. Returns (text, a note for the result or '')."""
    cap = _env_cap('SIGIL_READ_MAX_BYTES', 50 * 1024 * 1024)
    try:
        size = target.stat().st_size
    except OSError as _ignored:
        _log_ignored(_ignored)
        size = 0
    if not cap or size <= cap:
        return target.read_text(encoding='utf-8', errors='replace'), ''
    start_line = arguments.get('start_line')
    end_line = arguments.get('end_line')
    if isinstance(start_line, int) or isinstance(end_line, int):
        # why bounded: a line range used to read every line whole, so one huge line (a minified or
        # generated file without newlines) still came into memory in full
        first = start_line if isinstance(start_line, int) and start_line > 0 else 1
        last = end_line if isinstance(end_line, int) else (int(start_line) + 2000)
        lines: list[str] = []
        budget = cap
        note = ''
        with open(target, encoding='utf-8', errors='replace') as handle:
            number = 0
            while number < last:
                number += 1
                if number < first:
                    # lines before the range only keep their place; read in parts, never kept
                    while True:
                        part = handle.readline(1 << 20)
                        if not part or part.endswith('\n'):
                            break
                    if not part:
                        break
                    lines.append('\n')
                    continue
                line = handle.readline(budget + 1) if budget > 0 else ''
                if not line:
                    if budget <= 0 and handle.read(1):
                        note = (f'\n... [the range was cut at line {number}: {cap:,} characters read '
                                '(SIGIL_READ_MAX_BYTES). Ask for a later start_line for the rest.]')
                    break
                if len(line) > budget:
                    lines.append(line[:budget] + '\n')
                    note = (f'\n... [line {number} was cut after {cap:,} characters (SIGIL_READ_MAX_BYTES); '
                            'the rest of the range was not read.]')
                    break
                budget -= len(line)
                lines.append(line)
        return ''.join(lines), note
    with open(target, 'rb') as handle:
        data = handle.read(cap)
    return data.decode('utf-8', errors='replace'), (
        f'\n... [file is {size:,} bytes; only the first {cap:,} were read (SIGIL_READ_MAX_BYTES). '
        'Use start_line/end_line for a part further on.]')


def _read_file(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    path_str = _require_string(arguments, 'path')
    target = _resolve_path(path_str, context, allow_missing=False)
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    from .images import image_data_url, is_image, vision_enabled

    if target.suffix.lower() == '.pdf':
        return _read_pdf(target, path_str, arguments, context)
    if is_image(target):
        if not vision_enabled():
            return (f'{path_str} is an image; images are off (SIGIL_VISION=0).',
                    {'action': 'read_image', 'path': str(target), 'image_attached': False})
        try:
            image_data_url(target)  # checks type and size
        except ValueError as exc:
            raise ToolExecutionError(str(exc)) from exc
        return (f'Image {path_str} ({target.stat().st_size:,} bytes) is attached in the next message.',
                {'action': 'read_image', 'path': str(target), 'image_attached': True})
    
    text, size_note = _read_text_capped(target, arguments)
    
    # [SEMANTIC CACHING] Hash the file and check against session cache
    import hashlib
    current_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    # Check if we have a session to track hashes
    if context.session and hasattr(context.session, 'file_hashes'):
        rel_path = str(_display_path(target, context))
        previous_hash = context.session.file_hashes.get(rel_path)
        
        if previous_hash == current_hash and not arguments.get('force_read', False):
            return f"File '{rel_path}' unchanged since last read. Rely on your existing context."
        
        # Update cache
        context.session.file_hashes[rel_path] = current_hash
        # [BOUNDED GROWTH] Cap file_hashes to prevent unbounded memory growth
        # over long sessions. Evict oldest entries (Python 3.7+ dicts are ordered).
        _MAX_FILE_HASHES = 256
        while len(context.session.file_hashes) > _MAX_FILE_HASHES:
            oldest_key = next(iter(context.session.file_hashes))
            del context.session.file_hashes[oldest_key]

    # Track access turn for shredder decay
    if context.session:
        rel_path = str(_display_path(target, context))
        context.session.file_access_history[rel_path] = context.session.current_turn

    allow_large = arguments.get('allow_large', False)
    if not isinstance(allow_large, bool):
        raise ToolExecutionError('allow_large must be a boolean')
    output_limit = (
        max(context.max_output_chars, _READ_ALLOW_LARGE_MAX_CHARS)
        if allow_large
        else context.max_output_chars
    )

    start_line = arguments.get('start_line')
    end_line = arguments.get('end_line')
    if start_line is None and end_line is None:
        return _truncate_output(text + size_note, output_limit)
    if start_line is not None and (isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1):
        raise ToolExecutionError('start_line must be an integer >= 1')
    if end_line is not None and (isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < 1):
        raise ToolExecutionError('end_line must be an integer >= 1')
    lines = text.splitlines()
    start_idx = max((start_line or 1) - 1, 0)
    end_idx = end_line or len(lines)
    selected = lines[start_idx:end_idx]
    rendered = '\n'.join(f'{start_idx + idx + 1}: {line}' for idx, line in enumerate(selected))
    return _truncate_output(rendered + size_note, output_limit)


def _read_utf8_for_edit(target: Path) -> str:
    """Read a file for in-place editing, refusing files that are not valid UTF-8.

    Decoding with errors='replace' and writing back would silently replace every
    non-ASCII byte of a Windows-1252 / Shift-JIS / GBK file with U+FFFD.
    """
    try:
        return target.read_text(encoding='utf-8')
    except UnicodeDecodeError as exc:
        raise ToolExecutionError(
            f'Refusing to edit {target.name}: file is not valid UTF-8 '
            f'(byte offset {exc.start}). Editing it would corrupt non-ASCII characters.'
        ) from exc


def _write_file(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context, for_write=True)
    content = arguments.get('content')
    if not isinstance(content, str):
        raise ToolExecutionError('content must be a string')
    previous_text: str | None = None
    previous_sha256: str | None = None
    if target.exists() and target.is_file():
        previous_text = target.read_text(encoding='utf-8', errors='replace')
        previous_sha256 = hashlib.sha256(previous_text.encode('utf-8')).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    rel = _display_path(target, context)
    new_sha256 = hashlib.sha256(content.encode('utf-8')).hexdigest()

    # Track access turn
    if context.session:
        context.session.file_access_history[str(rel)] = context.session.current_turn
    return (
        f'wrote {rel} ({len(content)} chars)',
        {
            'action': 'write_file',
            'path': str(rel),
            'before_exists': previous_text is not None,
            'before_sha256': previous_sha256,
            'before_size': len(previous_text) if previous_text is not None else 0,
            'before_preview': (
                _snapshot_text(previous_text)
                if previous_text is not None
                else None
            ),
            'after_sha256': new_sha256,
            'after_size': len(content),
            'after_preview': _snapshot_text(content),
            'content_length': len(content),
        },
    )


def _edit_file(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context, allow_missing=False, for_write=True)
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    old_text = arguments.get('old_text')
    new_text = arguments.get('new_text')
    replace_all = arguments.get('replace_all', False)
    if not isinstance(old_text, str):
        raise ToolExecutionError('old_text must be a string')
    if not isinstance(new_text, str):
        raise ToolExecutionError('new_text must be a string')
    if not isinstance(replace_all, bool):
        raise ToolExecutionError('replace_all must be a boolean')
    current = _read_utf8_for_edit(target)
    occurrences = current.count(old_text)
    if occurrences == 0:
        raise ToolExecutionError('old_text was not found in the target file')
    if occurrences > 1 and not replace_all:
        raise ToolExecutionError(
            f'old_text matched {occurrences} times; pass replace_all=true to replace every match'
        )
    before_sha256 = hashlib.sha256(current.encode('utf-8')).hexdigest()
    updated = current.replace(old_text, new_text) if replace_all else current.replace(old_text, new_text, 1)
    target.write_text(updated, encoding='utf-8')
    rel = _display_path(target, context)
    replaced = occurrences if replace_all else 1
    after_sha256 = hashlib.sha256(updated.encode('utf-8')).hexdigest()
    
    # Track access turn
    if context.session:
        context.session.file_access_history[str(rel)] = context.session.current_turn

    return (
        f'edited {rel}; replaced {replaced} occurrence(s)',
        {
            'action': 'edit_file',
            'path': str(rel),
            'before_sha256': before_sha256,
            'after_sha256': after_sha256,
            'before_size': len(current),
            'after_size': len(updated),
            'before_preview': _snapshot_text(current),
            'after_preview': _snapshot_text(updated),
            'old_text_preview': _snapshot_text(old_text),
            'new_text_preview': _snapshot_text(new_text),
            'replaced_occurrences': replaced,
        },
    )


def _notebook_edit(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    target = _resolve_path(_require_string(arguments, 'path'), context, allow_missing=False, for_write=True)
    if target.suffix != '.ipynb':
        raise ToolExecutionError('notebook_edit requires a .ipynb target')
    if not target.is_file():
        raise ToolExecutionError(f'Path is not a file: {target}')
    cell_index = arguments.get('cell_index')
    if isinstance(cell_index, bool) or not isinstance(cell_index, int) or cell_index < 0:
        raise ToolExecutionError('cell_index must be an integer >= 0')
    source = arguments.get('source')
    if not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    cell_type = arguments.get('cell_type', 'code')
    if cell_type is not None and not isinstance(cell_type, str):
        raise ToolExecutionError('cell_type must be a string')
    create_cell = arguments.get('create_cell', False)
    if not isinstance(create_cell, bool):
        raise ToolExecutionError('create_cell must be a boolean')

    raw = _read_utf8_for_edit(target)
    before_sha256 = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolExecutionError(f'Notebook is not valid JSON: {target}') from exc
    if not isinstance(notebook, dict):
        raise ToolExecutionError('Notebook payload must be a JSON object')
    cells = notebook.get('cells')
    if not isinstance(cells, list):
        raise ToolExecutionError('Notebook does not contain a cells array')

    while len(cells) <= cell_index:
        if not create_cell:
            raise ToolExecutionError(
                f'Notebook cell {cell_index} does not exist; pass create_cell=true to append missing cells'
            )
        cells.append(
            {
                'cell_type': cell_type or 'code',
                'metadata': {},
                'source': [],
                'outputs': [],
                'execution_count': None,
            }
        )
    cell = cells[cell_index]
    if not isinstance(cell, dict):
        raise ToolExecutionError(f'Notebook cell {cell_index} is not a JSON object')
    existing_type = cell.get('cell_type')
    if not isinstance(existing_type, str):
        existing_type = 'code'
    source_lines = source.splitlines(keepends=True)
    if source and not source.endswith('\n'):
        source_lines = [*source_lines[:-1], source_lines[-1]]
    cell['cell_type'] = cell_type or existing_type
    cell['source'] = source_lines
    if cell['cell_type'] == 'code':
        cell.setdefault('outputs', [])
        cell.setdefault('execution_count', None)
    updated = json.dumps(notebook, ensure_ascii=True, indent=1) + '\n'
    target.write_text(updated, encoding='utf-8')
    after_sha256 = hashlib.sha256(updated.encode('utf-8')).hexdigest()
    rel = _display_path(target, context)
    return (
        f'updated notebook cell {cell_index} in {rel}',
        {
            'action': 'notebook_edit',
            'path': str(rel),
            'cell_index': cell_index,
            'cell_type': cell['cell_type'],
            'before_sha256': before_sha256,
            'after_sha256': after_sha256,
            'before_preview': _snapshot_text(raw),
            'after_preview': _snapshot_text(updated),
        },
    )


def _glob_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    pattern = _require_string(arguments, 'pattern')
    
    # 1. '..' is allowed: normalize it to an absolute pattern and check that below
    p = Path(pattern)
    if '..' in p.parts:
        p = Path(os.path.normpath(str(p if p.is_absolute() else context.root / p)))
        pattern = str(p)

    # 2. Handle absolute paths: glob from the allowed folder they start with
    search_root = context.root
    if p.is_absolute():
        main_root = context.root.resolve()
        allowed_roots = [main_root] + [
            d.resolve() for d in context.additional_working_directories
        ]
        # Longest root first, so a folder added inside the workspace wins.
        allowed_roots.sort(key=lambda r: len(str(r)), reverse=True)

        pattern_str = str(p)
        matched_root = False
        for root in allowed_roots:
            root_str = str(root)
            # Case-insensitive comparison for Windows paths
            if pattern_str.lower().startswith(root_str.lower()):
                rel = pattern_str[len(root_str):].lstrip('\\/')
                pattern = rel.replace('\\', '/')
                search_root = context.root if root == main_root else root
                matched_root = True
                break

        if not matched_root:
            # Other folders: glob from the pattern's fixed (wildcard-free) prefix.
            anchor = Path(p.anchor)
            rest: list[str] = []
            for index, part in enumerate(p.parts[1:], start=1):
                if any(ch in part for ch in '*?['):
                    rest = list(p.parts[index:])
                    break
                anchor = anchor / part
            if not rest or not _path_access(context).can_read(anchor):
                raise ToolExecutionError(f"Absolute glob pattern {pattern!r} is outside allowed directories")
            search_root = anchor
            pattern = '/'.join(rest)

    # 3. Perform glob search relative to the matched folder
    try:
        matches = sorted(search_root.glob(pattern))
    except NotImplementedError:
        # Final fallback for any other non-relative edge cases
        raise ToolExecutionError(f"Glob pattern {pattern!r} must be relative to the root directory")

    if not matches:
        return '(no matches)'
    root_resolved = search_root.resolve()
    validated: list[str] = []
    for path in matches:
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            continue
        if search_root is context.root:
            validated.append(str(path.relative_to(context.root)))
        else:
            # Outside the main folder: absolute, so read_file/edit can use it as is.
            validated.append(str(path))
    if not validated:
        return '(no matches)'
    return _truncate_output('\n'.join(validated), context.max_output_chars)


def _grep_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    pattern = _require_string(arguments, 'pattern')
    raw_path = arguments.get('path', '.')
    if not isinstance(raw_path, str):
        raise ToolExecutionError('path must be a string')
    literal = arguments.get('literal', False)
    if not isinstance(literal, bool):
        raise ToolExecutionError('literal must be a boolean')
    max_matches = _coerce_int(arguments, 'max_matches', 100)
    root = _resolve_path(raw_path, context)
    if not root.exists():
        raise ToolExecutionError(f'Path not found: {raw_path}')
    try:
        regex = re.compile(re.escape(pattern) if literal else pattern)
    except re.error as exc:
        raise ToolExecutionError(f'Invalid regex pattern: {exc}') from exc
    hits: list[str] = []
    # bounded walk: version-control, dependency and cache folders are skipped, files larger than
    # SIGIL_GREP_MAX_FILE_BYTES (default 10 MB) and binary files are skipped, the walk stops after
    # SIGIL_GREP_MAX_FILES files (default 20,000); lines are read one at a time (0 = no limit)
    max_file_bytes = _env_cap('SIGIL_GREP_MAX_FILE_BYTES', 10 * 1024 * 1024)
    max_files = _env_cap('SIGIL_GREP_MAX_FILES', 20000)
    skipped_large = 0
    scanned = 0
    stopped = False
    if root.is_dir():
        def _walk() -> Iterator[Path]:
            for folder, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in _GREP_SKIP_DIRS)
                for name in sorted(files):
                    yield Path(folder) / name
        file_iter: Any = _walk()
    else:
        file_iter = [root]
    for file_path in file_iter:
        if not file_path.is_file():
            continue
        if max_files and scanned >= max_files:
            stopped = True
            break
        scanned += 1
        try:
            size = file_path.stat().st_size
            if root.is_dir() and max_file_bytes and size > max_file_bytes:
                skipped_large += 1
                continue
            with open(file_path, 'rb') as probe:
                if b'\0' in probe.read(8192):
                    continue  # binary
            with open(file_path, encoding='utf-8', errors='replace') as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.rstrip('\r\n')
                    if regex.search(line):
                        rel = _display_path(file_path, context)
                        hits.append(f'{rel}:{line_no}: {line}')
                        if len(hits) >= max_matches:
                            return '\n'.join(hits + [f'... truncated at {max_matches} matches ...'])
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
    notes = []
    if skipped_large:
        notes.append(f'... {skipped_large} file(s) over SIGIL_GREP_MAX_FILE_BYTES ({max_file_bytes:,} bytes) skipped')
    if stopped:
        notes.append(f'... stopped after {max_files:,} files (SIGIL_GREP_MAX_FILES); narrow the path')
    return '\n'.join(hits + notes) if hits else '\n'.join(['(no matches)', *notes])


def _start_background_shell(command: str, context: ToolExecutionContext):
    from .background_shells import shell_manager

    from .sandbox import proxy_env

    output_dir = (context.scratchpad_root / 'background-shells') if context.scratchpad_root else None
    wrapped = _sandbox_argv(command, context)
    env = _build_subprocess_env(context)
    try:
        job = shell_manager().start(
            command, cwd=context.root, env=proxy_env(env) if wrapped else env,
            bash=_BASH_EXECUTABLE, output_dir=output_dir, argv=wrapped, limits=_windows_sandbox_limits(),
        )
    except RuntimeError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        '\n'.join([
            f'Started background shell {job.job_id} (pid {job.process.pid}).',
            f'Output file: {job.output_path}',
            f'Read new output with bash_output(id="{job.job_id}"); stop it with bash_kill(id="{job.job_id}").',
        ]),
        {'action': 'bash_background', 'command': command, 'shell_id': job.job_id,
         'pid': job.process.pid, 'output_path': str(job.output_path)},
    )


def _read_pdf(target: Path, path_str: str, arguments: dict[str, Any], context: ToolExecutionContext):
    """PDF text through the optional pypdf extra (pip install sigilcode[pdf])."""
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ToolExecutionError(
            'Reading PDFs needs the optional pypdf package: pip install "sigilcode[pdf]" (or pip install pypdf).'
        ) from exc
    try:
        reader = pypdf.PdfReader(str(target))
        pages = list(reader.pages)
    except Exception as exc:
        raise ToolExecutionError(f'Cannot read PDF {path_str}: {exc}') from exc
    first = max(1, _coerce_int(arguments, 'offset', 1)) if 'offset' in arguments else 1
    from . import prompt_constants

    default_pages = prompt_constants.PDF_MAX_PAGES_PER_READ
    count = _coerce_int(arguments, 'limit', default_pages) if 'limit' in arguments else default_pages
    chosen = pages[first - 1:first - 1 + max(1, count)]
    parts = []
    for number, page in enumerate(chosen, start=first):
        try:
            text = page.extract_text() or ''
        except Exception as _ignored:
            _log_ignored(_ignored)
            text = ''
        parts.append(f'--- page {number} ---\n{text.strip()}')
    header = f'PDF {path_str}: {len(pages)} page(s); showing {first}-{first + len(chosen) - 1}'
    return (_truncate_output(header + '\n\n' + '\n\n'.join(parts), context.max_output_chars),
            {'action': 'read_pdf', 'path': str(target), 'pages': len(pages)})


def _skill_tool(arguments: dict[str, Any], context: ToolExecutionContext):
    from types import SimpleNamespace

    from .custom_commands import discover_commands, expand_file_mentions, substitute_arguments

    name = _require_string(arguments, 'name').strip().lstrip('/')
    commands = discover_commands(context.root, context.skills_runtime)
    command = commands.get(name)
    if command is None:
        raise ToolExecutionError(f'No command or skill named {name!r}. Available: {", ".join(sorted(commands))}')
    text = substitute_arguments(command.body, str(arguments.get('arguments') or ''))
    text, _attached = expand_file_mentions(text, SimpleNamespace(tool_context=context))
    return (f'# /{name} ({command.source})\n\n{text}',
            {'action': 'skill', 'name': name, 'source': command.source})


def _powershell_executable() -> str:
    return shutil.which('pwsh') or shutil.which('powershell') or 'powershell'


def _powershell_tool_enabled() -> bool:
    flag = os.environ.get('SIGIL_POWERSHELL_TOOL', '').strip().lower()
    if flag in ('1', 'true', 'yes', 'on'):
        return True
    if flag in ('0', 'false', 'no', 'off'):
        return False
    return sys.platform == 'win32' and not os.path.exists(_BASH_EXECUTABLE)


_POWERSHELL_DESTRUCTIVE = re.compile(
    r'\b(Remove-Item|rm|del|rmdir|Format-Volume|Clear-Disk|Stop-Computer|Restart-Computer|'
    r'Remove-ItemProperty|Set-ExecutionPolicy)\b', re.I)


def _run_powershell(arguments: dict[str, Any], context: ToolExecutionContext):
    """PowerShell on Windows (when Git Bash is missing, or SIGIL_POWERSHELL_TOOL=1)."""
    command = _require_string(arguments, 'command')
    if not context.permissions.allow_shell_commands:
        raise ToolPermissionError('Shell commands are disabled. Re-run with --allow-shell to enable PowerShell.')
    if not context.permissions.allow_destructive_shell_commands and _POWERSHELL_DESTRUCTIVE.search(command):
        raise ToolPermissionError('Potentially destructive PowerShell command blocked. Re-run with --unsafe to allow it.')
    timeout = _bash_timeout(arguments, context)
    try:
        completed = subprocess.run(
            [_powershell_executable(), '-NoProfile', '-NonInteractive', '-Command', command],
            cwd=context.root, capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout, env=_build_subprocess_env(context), stdin=subprocess.DEVNULL,
            creationflags=_WIN_CHILD_FLAGS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(f'PowerShell command timed out after {timeout:.0f}s') from exc
    payload = [f'exit_code={completed.returncode}', '[stdout]', (completed.stdout or '').rstrip(),
               '[stderr]', (completed.stderr or '').rstrip()]
    return (_truncate_output('\n'.join(payload).strip(), context.max_output_chars),
            {'action': 'powershell', 'command': command, 'exit_code': completed.returncode})


def _wait_for_mcp_servers(arguments: dict[str, Any], context: ToolExecutionContext):
    runtime = context.mcp_runtime
    if runtime is None or not runtime.servers:
        return 'No MCP servers are configured.', {'action': 'wait_for_mcp_servers', 'servers': {}}
    timeout = _coerce_float(arguments, 'timeout_seconds', 30.0)
    statuses = runtime.wait_for_servers(max(0.0, min(timeout, 120.0)))
    lines = [f'- {name}: {status}' for name, status in statuses.items()]
    return '\n'.join(['# MCP servers', *lines]), {'action': 'wait_for_mcp_servers', 'servers': statuses}


def _bash_output(arguments: dict[str, Any], context: ToolExecutionContext):
    """New output of a background shell. `filter` (regex) keeps matching lines;
    `wait_seconds` (max 60) waits for new matching output - a simple monitor."""
    import time as _time

    from .background_shells import shell_manager

    job_id = _require_string(arguments, 'id')
    pattern = arguments.get('filter')
    try:
        regex = re.compile(pattern) if isinstance(pattern, str) and pattern else None
    except re.error as exc:
        raise ToolExecutionError(f'filter is not a valid regex: {exc}') from exc
    wait = max(0.0, min(_coerce_float(arguments, 'wait_seconds', 0.0), 60.0))
    deadline = _time.monotonic() + wait
    collected: list[str] = []
    skipped = 0
    while True:
        try:
            job, text = shell_manager().read_new(job_id, max_chars=context.max_output_chars)
        except KeyError as exc:
            raise ToolExecutionError(str(exc.args[0])) from exc
        for line in text.splitlines():
            if regex is None or regex.search(line):
                collected.append(line)
            else:
                skipped += 1
        if collected or job.exit_code is not None or _time.monotonic() >= deadline:
            break
        _time.sleep(0.2)
    text = '\n'.join(collected)
    if skipped:
        text += f'\n({skipped} lines did not match the filter)'
    code = job.exit_code
    header = f'status={job.status}' + ('' if code is None else f' exit_code={code}')
    if getattr(job, 'stop_reason', None):
        header += f' (stopped: {job.stop_reason})'
    body = text.rstrip() if text.strip() else '(no new output)'
    return (
        f'{header}\n{body}',
        {'action': 'bash_output', 'shell_id': job.job_id, 'status': job.status, 'exit_code': code},
    )


def _bash_kill(arguments: dict[str, Any], context: ToolExecutionContext):
    from .background_shells import shell_manager

    job_id = _require_string(arguments, 'id')
    try:
        job = shell_manager().kill(job_id)
    except KeyError as exc:
        raise ToolExecutionError(str(exc.args[0])) from exc
    return (
        f'Background shell {job.job_id}: {job.status} (exit_code={job.exit_code})',
        {'action': 'bash_kill', 'shell_id': job.job_id, 'status': job.status},
    )


def _run_bash(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    command = _normalize_windows_paths(_require_string(arguments, 'command'))
    guard_warning = _ensure_shell_allowed(command, context)
    if arguments.get('run_in_background') is True:
        return _start_background_shell(command, context)
    timeout = _bash_timeout(arguments, context)
    wrapped = _sandbox_argv(command, context)
    if wrapped is not None:
        from .sandbox import proxy_env

        completed = subprocess.run(
            wrapped,
            shell=False,
            cwd=context.root,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout,
            env=proxy_env(_build_subprocess_env(context)),
            stdin=subprocess.DEVNULL,
            **({'creationflags': _WIN_CHILD_FLAGS} if sys.platform == 'win32' else {}),
        )
    elif sys.platform == 'win32' and _windows_sandbox_limits() is not None:
        completed = _run_in_windows_job([_BASH_EXECUTABLE, '-c', command], context, timeout)
    elif sys.platform == 'win32':
        # shell=True on Windows uses cmd.exe semantics even when executable= is set,
        # which collides with bash's /c flag. Invoke bash explicitly instead. On a timeout the
        # whole tree ends (Job Object), not only bash.exe.
        from .background_shells import run_command_tree

        completed = run_command_tree([_BASH_EXECUTABLE, '-c', command], timeout=timeout, cwd=context.root,
                                     env=_build_subprocess_env(context), creationflags=_WIN_CHILD_FLAGS)
    else:
        from .background_shells import run_command_tree

        completed = run_command_tree(command, shell=True, executable=_BASH_EXECUTABLE, timeout=timeout,
                                     cwd=context.root, env=_build_subprocess_env(context))
    stdout = completed.stdout or ''
    stderr = completed.stderr or ''
    payload = [
        *([guard_warning] if guard_warning else []),
        f'exit_code={completed.returncode}',
        '[stdout]',
        stdout.rstrip(),
        '[stderr]',
        stderr.rstrip(),
    ]
    return (
        _truncate_output('\n'.join(payload).strip(), context.max_output_chars),
        {
            'action': 'bash',
            'command': command,
            'exit_code': completed.returncode,
            'stdout_preview': _snapshot_text(stdout),
            'stderr_preview': _snapshot_text(stderr),
            'output_preview': _snapshot_text('\n'.join(payload).strip()),
        },
    )


def _validate_url_not_internal(url: str) -> list[str]:
    """Refuse a URL whose host resolves to an internal address; return the checked addresses.

    The caller connects to exactly these addresses (`_pinned_opener`): a second lookup at connect
    time could answer differently (DNS rebinding) and reach a service on this machine."""
    import ipaddress
    import socket
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ToolExecutionError('Invalid URL: no hostname')
    try:
        resolved = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ToolExecutionError(f'Cannot resolve hostname: {hostname}') from exc
    blocked = [
        ipaddress.ip_network('0.0.0.0/8'),
        ipaddress.ip_network('127.0.0.0/8'),
        ipaddress.ip_network('10.0.0.0/8'),
        ipaddress.ip_network('172.16.0.0/12'),
        ipaddress.ip_network('192.168.0.0/16'),
        ipaddress.ip_network('169.254.0.0/16'),
        ipaddress.ip_network('::1/128'),
        ipaddress.ip_network('::ffff:0.0.0.0/96'),
        ipaddress.ip_network('fd00::/8'),
    ]
    cgnat = ipaddress.ip_network('100.64.0.0/10')
    checked: list[str] = []
    for _, _, _, _, sockaddr in resolved:
        ip = ipaddress.ip_address(str(sockaddr[0]).split('%', 1)[0])
        mapped = getattr(ip, 'ipv4_mapped', None)
        for candidate in (ip, mapped) if mapped is not None else (ip,):
            if (any(candidate in network for network in blocked if candidate.version == network.version)
                    or candidate.is_private or candidate.is_loopback or candidate.is_link_local
                    or candidate.is_reserved or candidate.is_unspecified or candidate.is_multicast
                    or (candidate.version == 4 and candidate in cgnat)):
                raise ToolExecutionError(
                    f'Fetching internal/private addresses is not allowed: {hostname} ({ip})')
        if str(ip) not in checked:
            checked.append(str(ip))
    return checked


def _pinned_opener(hostname: str, addresses: list[str]) -> urllib.request.OpenerDirector:
    """An opener whose connections to `hostname` go only to `addresses` (the ones that passed
    `_validate_url_not_internal`), so the name is not looked up a second time. TLS still checks
    the certificate against the hostname, and the Host header is unchanged. A connection to
    another host (a configured proxy) is left alone: the proxy resolves the target itself."""
    import socket

    host = hostname.lower().strip('[]')

    def create(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):  # noqa: ANN001, ANN202
        if str(address[0]).lower().strip('[]') != host:
            return socket.create_connection(address, timeout, source_address)
        last: OSError | None = None
        for ip in addresses:
            try:
                return socket.create_connection((ip, address[1]), timeout, source_address)
            except OSError as exc:
                last = exc
        raise last or OSError(f'no address to connect to for {hostname}')

    def pinned(connection_class):  # noqa: ANN001, ANN202
        class Pinned(connection_class):
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                super().__init__(*args, **kwargs)
                self._create_connection = create
        return Pinned

    class PinnedHTTP(urllib.request.HTTPHandler):
        def do_open(self, http_class, req, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return super().do_open(pinned(http_class), req, **kwargs)

    class PinnedHTTPS(urllib.request.HTTPSHandler):
        def do_open(self, http_class, req, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return super().do_open(pinned(http_class), req, **kwargs)

    return urllib.request.build_opener(_NoRedirect, PinnedHTTP, PinnedHTTPS)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


_WEB_FETCH_MAX_HTML_BYTES = 2_000_000


def _html_to_markdown(html_text: str, base_url: str | None = None) -> str:
    """Convert HTML to readable markdown (headings, lists, links, code, tables).

    Stdlib-only (no Turndown). Scripts/styles are
    dropped; if the page has a <main>/<article>, only that is kept so menus and
    footers don't drown the content; relative links are made absolute.
    """
    from html.parser import HTMLParser

    skip_tags = {'script', 'style', 'noscript', 'svg', 'template', 'iframe', 'button',
                 'form', 'select', 'canvas'}
    chrome_tags = {'nav', 'header', 'footer', 'aside'}
    void_tags = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed',
                 'source', 'track', 'wbr', 'param'}
    block_tags = {'p', 'div', 'section', 'article', 'main', 'br', 'tr', 'table', 'ul', 'ol',
                  'blockquote', 'hr', 'figure', 'figcaption', 'dl', 'dt', 'dd', 'details', 'summary'}
    headings = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

    class _Converter(HTMLParser):
        def __init__(self, drop_chrome: bool) -> None:
            super().__init__(convert_charrefs=True)
            self.drop_chrome = drop_chrome
            self.parts: list[str] = []
            self.title: list[str] = []
            self.in_title = False
            self.skip_stack: list[str] = []
            self.href: str | None = None
            self.link_text: list[str] = []
            self.pre_depth = 0
            self.main_parts: list[str] | None = None
            self.main_depth = 0
            self.main_tag: str | None = None
            self.list_stack: list[list[int | str]] = []  # [kind, counter]

        @property
        def out(self) -> list[str]:
            return self.parts

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == 'title':
                self.in_title = True
                return
            if self.skip_stack:
                if tag == self.skip_stack[-1] and tag not in void_tags:
                    self.skip_stack.append(tag)
                return
            attr = dict(attrs)
            is_chrome = tag in chrome_tags or (attr.get('role') or '').lower() in {
                'navigation', 'banner', 'contentinfo', 'search', 'complementary'}
            if tag in skip_tags or (self.drop_chrome and is_chrome) or tag == 'head' \
                    or attr.get('aria-hidden') == 'true' or 'hidden' in attr:
                if tag not in void_tags:
                    self.skip_stack.append(tag)
                return
            if tag in headings:
                self.parts.append('\n\n' + '#' * int(tag[1]) + ' ')
            elif tag in {'ul', 'ol'}:
                self.list_stack.append([tag, 0])
                self.parts.append('\n')
            elif tag == 'li':
                indent = '  ' * max(len(self.list_stack) - 1, 0)
                if self.list_stack and self.list_stack[-1][0] == 'ol':
                    self.list_stack[-1][1] = int(self.list_stack[-1][1]) + 1
                    self.parts.append(f'\n{indent}{self.list_stack[-1][1]}. ')
                else:
                    self.parts.append(f'\n{indent}- ')
            elif tag in {'td', 'th'}:
                self.parts.append(' | ')
            elif tag == 'pre':
                self.pre_depth += 1
                self.parts.append('\n```\n')
            elif tag == 'code' and not self.pre_depth:
                self.parts.append('`')
            elif tag in {'strong', 'b'}:
                self.parts.append('**')
            elif tag in {'em', 'i'}:
                self.parts.append('*')
            elif tag == 'blockquote':
                self.parts.append('\n> ')
            elif tag == 'img':
                alt = (attr.get('alt') or '').strip()
                if alt:
                    self.parts.append(f'[image: {alt}]')
            elif tag == 'a':
                href = (attr.get('href') or '').strip()
                if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                    if base_url:
                        href = urllib.parse.urljoin(base_url, href)
                    if href.startswith(('http://', 'https://')):
                        self.href = href
                        self.link_text = []
            elif tag in block_tags:
                self.parts.append('\n')

        def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            self.handle_starttag(tag, attrs)
            if tag not in void_tags:
                self.handle_endtag(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag == 'title':
                self.in_title = False
                return
            if self.skip_stack:
                if tag == self.skip_stack[-1]:
                    self.skip_stack.pop()
                return
            if tag == 'a' and self.href is not None:
                label = ' '.join(''.join(self.link_text).split())
                self.parts.append(f'[{label}]({self.href})' if label else '')
                self.href = None
            elif tag in {'ul', 'ol'}:
                if self.list_stack:
                    self.list_stack.pop()
                self.parts.append('\n')
            elif tag == 'pre':
                self.pre_depth = max(self.pre_depth - 1, 0)
                self.parts.append('\n```\n')
            elif tag == 'code' and not self.pre_depth:
                self.parts.append('`')
            elif tag in {'strong', 'b'}:
                self.parts.append('**')
            elif tag in {'em', 'i'}:
                self.parts.append('*')
            elif tag == 'tr':
                self.parts.append(' |\n')
            elif tag in block_tags or tag in headings:
                self.parts.append('\n')

        def handle_data(self, data: str) -> None:
            if self.in_title:
                self.title.append(data)
                return
            if self.skip_stack:
                return
            if not self.pre_depth:
                data = re.sub(r'\s+', ' ', data)
            if self.href is not None:
                self.link_text.append(data)
            else:
                self.parts.append(data)

    def _convert(fragment: str, drop_chrome: bool) -> tuple[str, str]:
        converter = _Converter(drop_chrome)
        converter.feed(fragment)
        converter.close()
        text = ''.join(converter.parts)
        # Drop empty emphasis markers left by whitespace-only <b>/<i>.
        text = re.sub(r'\*\*\s*\*\*|(?<!\*)\*\s\*(?!\*)', ' ', text)
        return text, ' '.join(''.join(converter.title).split())

    try:
        # Prefer the main content region when the page marks one.
        main_match = re.search(r'<(main|article)\b[^>]*>(.*)</\1\s*>', html_text, re.S | re.I)
        title = ''
        text = ''
        if main_match and len(main_match.group(2)) > 500:
            _, title = _convert(html_text[: main_match.start()], drop_chrome=True)
            text, _ = _convert(main_match.group(0), drop_chrome=False)
        if len(text.strip()) < 200:
            text, title = _convert(html_text, drop_chrome=True)
        if len(text.strip()) < 200:  # page is mostly "chrome"; keep everything
            text, title = _convert(html_text, drop_chrome=False)
    except Exception as _ignored:
        _log_ignored(_ignored)
        return html_text
    if title and title.lower() not in text[:300].lower():
        text = f'# {title}\n\n{text}'
    # Tidy whitespace outside ``` code blocks (odd segments are code).
    segments = text.split('```')
    for i in range(0, len(segments), 2):
        seg = re.sub(r'[ \t]+\n', '\n', segments[i])
        seg = re.sub(r'\n[ \t]+(?=[^-\d ])', '\n', seg)
        seg = re.sub(r'[ \t]{2,}', ' ', seg)
        segments[i] = re.sub(r'\n{3,}', '\n\n', seg)
    return '```'.join(segments).strip()


_WEB_FETCH_CACHE_TTL_SECONDS = 15 * 60
_WEB_FETCH_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_WEB_FETCH_CACHE_MAX = 64
_WEB_FETCH_LOCK = __import__('threading').Lock()  # why: web_fetch runs in parallel read-only batches
_WEB_FETCH_PROMPT_MAX_CHARS = 100_000
_WEB_FETCH_MAX_REDIRECTS = 10
_WEB_FETCH_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0 Safari/537.36 sigilcode/1.0'
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _web_fetch_cached(url: str, timeout: float) -> dict[str, Any]:
    now = time.time()
    with _WEB_FETCH_LOCK:
        cached = _WEB_FETCH_CACHE.get(url)
    if cached is not None and now - cached[0] < _WEB_FETCH_CACHE_TTL_SECONDS:
        return cached[1]
    result = _web_fetch_raw(url, timeout)
    with _WEB_FETCH_LOCK:
        _WEB_FETCH_CACHE[url] = (now, result)
        # why a hard cap: only expired pages were removed, so 15 minutes of fetching kept every page
        for key, (stamp, _value) in list(_WEB_FETCH_CACHE.items()):
            if now - stamp >= _WEB_FETCH_CACHE_TTL_SECONDS:
                _WEB_FETCH_CACHE.pop(key, None)
        while len(_WEB_FETCH_CACHE) > _WEB_FETCH_CACHE_MAX:
            _WEB_FETCH_CACHE.pop(next(iter(_WEB_FETCH_CACHE)))  # oldest first
    return result


def _web_fetch_raw(url: str, timeout: float) -> dict[str, Any]:
    """Fetch url, following same-host redirects only, validating every hop."""
    current = url
    for _ in range(_WEB_FETCH_MAX_REDIRECTS + 1):
        addresses = _validate_url_not_internal(current)
        opener = _pinned_opener(urllib.parse.urlparse(current).hostname or '', addresses)
        request = urllib.request.Request(
            current,
            headers={
                'User-Agent': _WEB_FETCH_USER_AGENT,
                'Accept': 'text/markdown, text/html;q=0.9, text/plain;q=0.8, application/json;q=0.8, */*;q=0.5',
            },
        )
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get('Location') if exc.headers else None
                if not location:
                    raise ToolExecutionError(f'Failed to fetch {current}: redirect without Location') from exc
                target = urllib.parse.urljoin(current, location)
                if urllib.parse.urlparse(target).scheme not in {'http', 'https'}:
                    raise ToolExecutionError(f'Refusing redirect to non-http URL: {target}') from exc
                old_host = (urllib.parse.urlparse(current).hostname or '').lower().removeprefix('www.')
                new_host = (urllib.parse.urlparse(target).hostname or '').lower().removeprefix('www.')
                if old_host != new_host:
                    return {'redirect_to': target}
                current = target
                continue
            raise ToolExecutionError(f'Failed to fetch {current}: HTTP {exc.code} {exc.reason}') from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ToolExecutionError(f'Failed to fetch {current}: {exc}') from exc
        with response:
            headers = response.headers
            content_type = headers.get_content_type() if headers is not None else None
            charset = (headers.get_content_charset() if headers is not None else None) or 'utf-8'
            raw_bytes = response.read(_WEB_FETCH_MAX_HTML_BYTES)
        if content_type and not (
            content_type.startswith('text/')
            or content_type in {'application/json', 'application/xml', 'application/xhtml+xml',
                                'application/javascript', 'application/x-yaml', 'application/rss+xml',
                                'application/atom+xml'}
            or content_type.endswith(('+json', '+xml'))
        ):
            return {
                'text': f'[Binary content ({content_type}, {len(raw_bytes):,} bytes) cannot be shown as text.]',
                'content_type': content_type,
                'final_url': current,
            }
        try:
            text = raw_bytes.decode(charset, errors='replace')
        except LookupError as _ignored:
            _log_ignored(_ignored)
            text = raw_bytes.decode('utf-8', errors='replace')
        if content_type in {'text/html', 'application/xhtml+xml'}:
            text = _html_to_markdown(text, base_url=current)
        return {'text': text, 'content_type': content_type, 'final_url': current}
    raise ToolExecutionError(f'Failed to fetch {url}: too many redirects')


def _web_fetch_apply_prompt(
    content: str, prompt: str, url: str, context: ToolExecutionContext, errors: list[str] | None = None
) -> str | None:
    """Answer the prompt from the page with a side model call."""
    try:
        from .anthropic_client import fast_client

        client = fast_client(context.model_config)
        body = content[:_WEB_FETCH_PROMPT_MAX_CHARS]
        if len(content) > len(body):
            body += '\n...[page truncated]...'
        turn = client.complete(
            [
                {
                    'role': 'user',
                    'content': (
                        f'Web page content from {url}:\n---\n{body}\n---\n\n{prompt}\n\n'
                        'Answer using only the page content above. Be concise but keep all '
                        'specifics the request asks for (exact values, code, commands, versions). '
                        'Quote short exact passages where precision matters. If the page does not '
                        'contain the answer, say so.'
                    ),
                }
            ],
            [],
        )
    except Exception as exc:
        from .debug_log import get_logger, sanitize_log, url_for_log

        get_logger('tools').warning('web_fetch prompt for %s failed: %s', url_for_log(url), sanitize_log(exc))
        if errors is not None:
            errors.append(f'{type(exc).__name__}: {exc}')
        return None
    answer = (getattr(turn, 'content', None) or '').strip()
    if not answer:
        return None
    return f'{answer}\n\nSource: {url}'


def _web_fetch(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    if _env_flag('SIGIL_DISABLE_WEB_FETCH'):
        raise ToolExecutionError('web_fetch is disabled by SIGIL_DISABLE_WEB_FETCH.')
    raw_url = _require_string(arguments, 'url').strip()
    max_chars = _coerce_int(arguments, 'max_chars', context.max_output_chars)
    prompt = arguments.get('prompt')
    if prompt is not None and not isinstance(prompt, str):
        raise ToolExecutionError('prompt must be a string')
    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme not in {'http', 'https'}:
        raise ToolExecutionError('url must use http or https scheme')
    if parsed.scheme == 'http':
        # Upgrade plain http to https.
        raw_url = urllib.parse.urlunparse(parsed._replace(scheme='https'))
    fetched = _web_fetch_cached(raw_url, context.command_timeout_seconds)
    if fetched.get('redirect_to'):
        message = (
            f'The URL redirected to a different host: {fetched["redirect_to"]}\n'
            'Call web_fetch again with that URL if you want to follow it.'
        )
        return message, {'action': 'web_fetch', 'url': raw_url, 'redirect_to': fetched['redirect_to']}
    text = fetched['text']
    content_type = fetched['content_type']
    if prompt and prompt.strip() and context.model_config is not None:
        # Why: a failed side call used to return the raw page with no word about the failure.
        prompt_errors: list[str] = []
        answer = _web_fetch_apply_prompt(text, prompt.strip(), raw_url, context, prompt_errors)
        if answer is None and prompt_errors:
            text = (f'[The prompt could not be applied ({prompt_errors[0]}); the page text follows.]\n\n'
                    + text)
        if answer is not None:
            return (
                _truncate_output(answer, context.max_output_chars),
                {
                    'action': 'web_fetch',
                    'url': raw_url,
                    'content_type': content_type,
                    'prompt_applied': True,
                    'fetched_chars': len(text),
                    'preview': _snapshot_text(answer),
                },
            )
    truncated = len(text) > max_chars
    rendered = text[:max_chars]
    if truncated:
        rendered += '\n...[truncated]...'
    return (
        _truncate_output(rendered, context.max_output_chars),
        {
            'action': 'web_fetch',
            'url': raw_url,
            'content_type': content_type,
            'truncated': truncated,
            'fetched_chars': len(text),
            'preview': _snapshot_text(rendered),
        },
    )


def _rollback_workspace_tool(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    if not context.permissions.allow_destructive_shell_commands:
        raise ToolPermissionError(
            'rollback_workspace is a destructive operation (git reset --hard + clean). '
            'Enable destructive shell commands to use it.')
    try:
        subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=context.root, capture_output=True)
        subprocess.run(['git', 'clean', '-fd'], cwd=context.root, capture_output=True)
        return "Workspace successfully rolled back to last checkpoint."
    except Exception as e:
        raise ToolExecutionError(f"Rollback failed: {e}")


def _search_status(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_search_runtime(context)
    provider = arguments.get('provider')
    if provider is not None and not isinstance(provider, str):
        raise ToolExecutionError('provider must be a string')
    if provider:
        return runtime.render_provider(provider)
    return '\n'.join(['# Search', '', runtime.render_summary()])


def _search_list_providers(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_search_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_providers = _coerce_int(arguments, 'max_providers', 20)
    providers = runtime.list_providers(query=query, limit=max_providers)
    lines = ['# Search Providers', '']
    if not providers:
        lines.append('No local search providers discovered.')
        return '\n'.join(lines)
    current = runtime.current_provider()
    current_name = current.name if current is not None else None
    for provider in providers:
        details = [provider.name, provider.provider, provider.base_url]
        if provider.api_key_env:
            details.append(f'api_key_env={provider.api_key_env}')
        if current_name == provider.name:
            details.append('active=true')
        lines.append('- ' + ' ; '.join(details))
    return '\n'.join(lines)


def _search_activate_provider(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    runtime = _require_search_runtime(context)
    provider = _require_string(arguments, 'provider')
    try:
        report = runtime.activate_provider(provider)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown search provider: {provider}') from exc
    return (
        '\n'.join(['# Search', '', report.as_text()]),
        {
            'action': 'search_activate_provider',
            'provider': report.provider_name,
            'provider_kind': report.provider_kind,
            'base_url': report.base_url,
        },
    )


def _web_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_search_runtime(context)
    query = _require_string(arguments, 'query')
    if context.session is not None:
        try:
            limit = int(os.environ.get('SIGIL_MAX_WEB_SEARCHES_PER_SESSION', '200'))
        except ValueError as _ignored:
            _log_ignored(_ignored)
            limit = 200
        if limit <= 0:
            limit = 200
        used = getattr(context.session, 'web_search_count', 0)
        if used >= limit:
            raise ToolExecutionError(
                f'Web search limit reached for this session ({used}/{limit}). '
                'This guards against runaway search loops; work with the results you have '
                'or raise SIGIL_MAX_WEB_SEARCHES_PER_SESSION.'
            )
        context.session.web_search_count = used + 1
    provider = arguments.get('provider')
    if provider is not None and not isinstance(provider, str):
        raise ToolExecutionError('provider must be a string')
    raw_domains = arguments.get('domains', ())
    if raw_domains is None:
        raw_domains = ()
    if not isinstance(raw_domains, (list, tuple)):
        raise ToolExecutionError('domains must be an array of strings')
    domains: tuple[str, ...] = tuple(
        item.strip() for item in raw_domains if isinstance(item, str) and item.strip()
    )
    if len(domains) != len(raw_domains):
        raise ToolExecutionError('domains must contain only non-empty strings')
    raw_blocked = arguments.get('blocked_domains') or ()
    if not isinstance(raw_blocked, (list, tuple)):
        raise ToolExecutionError('blocked_domains must be an array of strings')
    blocked_domains: tuple[str, ...] = tuple(
        item.strip() for item in raw_blocked if isinstance(item, str) and item.strip()
    )
    if domains and blocked_domains:
        raise ToolExecutionError('Use either domains or blocked_domains, not both')
    max_results = arguments.get('max_results')
    if max_results is None:
        default_provider = runtime.get_provider(provider) if provider else runtime.current_provider()
        max_results = default_provider.default_max_results if default_provider is not None else 5
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise ToolExecutionError('max_results must be an integer')
    if max_results < 1 or max_results > 20:
        raise ToolExecutionError('max_results must be between 1 and 20')
    try:
        selected_provider, results = runtime.search(
            query,
            provider_name=provider,
            max_results=max_results,
            domains=domains,
            timeout_seconds=context.command_timeout_seconds,
            blocked_domains=blocked_domains,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown search provider: {provider or exc.args[0]}') from exc
    except LookupError as exc:
        raise ToolExecutionError(str(exc)) from exc
    except (ValueError, OSError, urllib.error.URLError) as exc:
        raise ToolExecutionError(f'Web search failed: {exc}') from exc
    lines = ['# Web Search', '']
    lines.append(f'- Provider: {selected_provider.name} ({selected_provider.provider})')
    lines.append(f'- Query: {query}')
    lines.append(f'- Results: {len(results)}')
    if domains:
        lines.append('- Domains: ' + ', '.join(domains))
    if blocked_domains:
        lines.append('- Blocked domains: ' + ', '.join(blocked_domains))
    lines.append('')
    if not results:
        lines.append('No search results.')
    else:
        for index, result in enumerate(results, start=1):
            lines.append(f'{index}. [{result.title}]({result.url})')
            if result.snippet:
                lines.append(f'   {result.snippet}')
        lines.append('')
        lines.append(
            'REMINDER: when you use these results in your answer, cite the sources as '
            'markdown links [Title](URL). Use web_fetch on a URL (optionally with a '
            '`prompt`) to read a page in full.'
        )
    rendered = '\n'.join(lines)
    return (
        rendered,
        {
            'action': 'web_search',
            'provider': selected_provider.name,
            'provider_kind': selected_provider.provider,
            'query': query,
            'result_count': len(results),
            'domains': list(domains),
            'top_urls': [result.url for result in results[:5]],
        },
    )


def _tool_search(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    query = _require_string(arguments, 'query').lower()
    max_results = _coerce_int(arguments, 'max_results', 20)
    registry = context.tool_registry or default_tool_registry()
    matches: list[tuple[str, str]] = []
    for tool in registry.values():
        haystack = f'{tool.name} {tool.description}'.lower()
        if query in haystack:
            matches.append((tool.name, tool.description))
    if not matches:
        return '(no matching tools)'
    lines = ['# Tool Search', '']
    for name, description in matches[:max_results]:
        lines.append(f'- `{name}`: {description}')
    return '\n'.join(lines)


def _sleep(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    seconds = _coerce_float(arguments, 'seconds', 0.0)
    if seconds < 0.0 or seconds > 5.0:
        raise ToolExecutionError('seconds must be between 0.0 and 5.0')
    time.sleep(seconds)
    return (
        f'slept for {seconds:.3f} seconds',
        {
            'action': 'sleep',
            'seconds': seconds,
        },
    )


def _ask_user_question(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_ask_user_runtime(context)
    question = _require_string(arguments, 'question')
    header = arguments.get('header')
    question_id = arguments.get('question_id')
    if header is not None and not isinstance(header, str):
        raise ToolExecutionError('header must be a string')
    if question_id is not None and not isinstance(question_id, str):
        raise ToolExecutionError('question_id must be a string')
    raw_choices = arguments.get('choices', ())
    if raw_choices is None:
        raw_choices = ()
    if not isinstance(raw_choices, (list, tuple)):
        raise ToolExecutionError('choices must be an array of strings')
    choices = tuple(
        item.strip()
        for item in raw_choices
        if isinstance(item, str) and item.strip()
    )
    if len(choices) != len(raw_choices):
        raise ToolExecutionError('choices must contain only non-empty strings')
    allow_free_text = arguments.get('allow_free_text', True)
    if not isinstance(allow_free_text, bool):
        raise ToolExecutionError('allow_free_text must be a boolean')
    try:
        response = runtime.answer(
            question=question,
            choices=choices,
            question_id=question_id,
            header=header,
            allow_free_text=allow_free_text,
        )
    except LookupError as exc:
        raise ToolExecutionError(str(exc)) from exc
    lines = ['# Ask User', '']
    if header:
        lines.append(f'- Header: {header}')
    if question_id:
        lines.append(f'- Question ID: {question_id}')
    lines.append(f'- Question: {question}')
    if choices:
        lines.append('- Choices: ' + ', '.join(choices))
    answer_text = response.answer
    if response.source == 'queued':
        from .ask_user_runtime import manifest_label

        # N23: a queued answer is not the user's own reply; the model (and the transcript) see where it came from
        label = manifest_label(response.source_file) if response.source_file else 'a queued answers file'
        answer_text = f'Pre-set answer from {label} (not typed by the user): {response.answer}'
    lines.extend(
        [
            f'- Source: {response.source}',
            '',
            answer_text,
        ]
    )
    return (
        '\n'.join(lines),
        {
            'action': 'ask_user_question',
            'question': question,
            'question_id': question_id,
            'header': header,
            'source': response.source,
            'answer_preview': _snapshot_text(response.answer),
            'choices': list(choices),
        },
    )


def _account_status(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    profile = arguments.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise ToolExecutionError('profile must be a string')
    if profile:
        return runtime.render_profile(profile)
    return '\n'.join(['# Account', '', runtime.render_summary()])


def _account_list_profiles(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_profiles = _coerce_int(arguments, 'max_profiles', 20)
    profiles = runtime.list_profiles(query=query, limit=max_profiles)
    lines = ['# Account Profiles', '']
    if not profiles:
        lines.append('No local account profiles discovered.')
        return '\n'.join(lines)
    for profile in profiles:
        details = [profile.name, profile.provider, profile.identity]
        if profile.org:
            details.append(f'org={profile.org}')
        if profile.auth_mode:
            details.append(f'auth_mode={profile.auth_mode}')
        lines.append('- ' + ' ; '.join(details))
    return '\n'.join(lines)


def _account_login(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    target = _require_string(arguments, 'target')
    provider = arguments.get('provider')
    auth_mode = arguments.get('auth_mode')
    if provider is not None and not isinstance(provider, str):
        raise ToolExecutionError('provider must be a string')
    if auth_mode is not None and not isinstance(auth_mode, str):
        raise ToolExecutionError('auth_mode must be a string')
    report = runtime.login(target, provider=provider, auth_mode=auth_mode)
    return '\n'.join(['# Account', '', report.as_text()])


def _account_logout(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_account_runtime(context)
    reason = arguments.get('reason', 'tool_logout')
    if not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    report = runtime.logout(reason=reason)
    return '\n'.join(['# Account', '', report.as_text()])


def _config_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_config_runtime(context)
    source = arguments.get('source')
    prefix = arguments.get('prefix')
    if source is not None and not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    if prefix is not None and not isinstance(prefix, str):
        raise ToolExecutionError('prefix must be a string')
    limit = _coerce_int(arguments, 'limit', 100)
    try:
        rendered = runtime.render_keys(source=source, prefix=prefix, limit=limit)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown config source: {exc.args[0]}') from exc
    return '\n'.join(['# Config Keys', '', rendered])


def _config_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_config_runtime(context)
    key_path = _require_string(arguments, 'key_path')
    source = arguments.get('source')
    if source is not None and not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    try:
        rendered = runtime.render_value(key_path, source=source)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown config key or source: {exc.args[0]}') from exc
    return '\n'.join(['# Config Value', '', rendered])


# Settings that grant capability (allow rules, hooks that run commands, env, sandbox, other hosts
# and keys, commands Sigil runs). The model's config_set may change them only with the user's
# yes (an approval prompt), or when SIGIL_CONFIG_SET_SENSITIVE=allow.
SENSITIVE_CONFIG_KEYS = frozenset({
    'permissions', 'hooks', 'env', 'sandbox', 'statusLine', 'mcpServers', 'enabledMcpjsonServers',
    'profiles', 'fastModel', 'editorModel', 'model', 'extraHeaders', 'testCommand', 'lintCommand', 'autoTest',
    'autoMemoryDirectory', 'apiKeyHelper', 'plugins', 'disableSkillShellExecution', 'trustAllProjects',
})


def config_key_is_sensitive(key_path: str) -> bool:
    head = str(key_path or '').strip().split('.', 1)[0]
    return head in SENSITIVE_CONFIG_KEYS


def _config_set(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_config_runtime(context)
    key_path = _require_string(arguments, 'key_path')
    if (config_key_is_sensitive(key_path) and not context.approved
            and os.environ.get('SIGIL_CONFIG_SET_SENSITIVE', '').strip().lower() != 'allow'):
        raise ToolPermissionError(
            f'config_set {key_path}: this setting changes what Sigil may run or send, so it needs '
            'your approval (or SIGIL_CONFIG_SET_SENSITIVE=allow).')
    source = arguments.get('source', 'local')
    if not isinstance(source, str):
        raise ToolExecutionError('source must be a string')
    if 'value' not in arguments:
        raise ToolExecutionError('value is required')
    value = arguments.get('value')
    try:
        mutation = runtime.set_value(key_path, value, source=source)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown config source: {exc.args[0]}') from exc
    return (
        f'set {key_path} in {mutation.source_name} config',
        _config_mutation_metadata(mutation, value),
    )


def _mcp_list_resources(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_resources = _coerce_int(arguments, 'max_resources', 50)
    resources = runtime.list_resources(query=query, limit=max_resources)
    if not resources:
        return '(no MCP resources)'
    lines: list[str] = []
    for resource in resources:
        details = [resource.uri, f'server={resource.server_name}']
        if resource.name:
            details.append(f'name={resource.name}')
        if resource.mime_type:
            details.append(f'mime={resource.mime_type}')
        if resource.resolved_path:
            details.append(f'path={resource.resolved_path}')
        lines.append(' ; '.join(details))
    return '\n'.join(lines)


def _mcp_read_resource(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    uri = _require_string(arguments, 'uri')
    max_chars = _coerce_int(arguments, 'max_chars', context.max_output_chars)
    try:
        content = runtime.read_resource(uri, max_chars=max_chars)
    except FileNotFoundError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return content


def _mcp_list_tools(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    query = arguments.get('query')
    server = arguments.get('server')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    if server is not None and not isinstance(server, str):
        raise ToolExecutionError('server must be a string')
    max_tools = _coerce_int(arguments, 'max_tools', 50)
    tools = runtime.list_tools(query=query, server_name=server, limit=max_tools)
    if not tools:
        return '(no MCP tools)'
    lines: list[str] = []
    for tool in tools:
        details = [tool.name, f'server={tool.server_name}']
        if tool.description:
            details.append(tool.description)
        lines.append(' ; '.join(details))
    return '\n'.join(lines)


def _mcp_call_tool(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_mcp_runtime(context)
    tool_name = _require_string(arguments, 'tool_name')
    server = arguments.get('server')
    if server is not None and not isinstance(server, str):
        raise ToolExecutionError('server must be a string')
    raw_arguments = arguments.get('arguments', {})
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ToolExecutionError('arguments must be an object')
    max_chars = _coerce_int(arguments, 'max_chars', context.max_output_chars)
    try:
        content, metadata = runtime.call_tool(
            tool_name,
            arguments=raw_arguments,
            server_name=server,
            max_chars=max_chars,
        )
    except FileNotFoundError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        content,
        {
            'action': 'mcp_call_tool',
            'tool_name': tool_name,
            'server_name': metadata.get('server_name'),
            'mcp_is_error': metadata.get('is_error'),
            'rate_limited': metadata.get('rate_limited', False),
        },
    )


def _remote_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    runtime = _require_remote_runtime(context)
    profile = arguments.get('profile')
    if profile is not None and not isinstance(profile, str):
        raise ToolExecutionError('profile must be a string')
    if profile:
        return runtime.render_profile(profile)
    return '\n'.join(['# Remote', '', runtime.render_summary()])


def _remote_list_profiles(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    runtime = _require_remote_runtime(context)
    query = arguments.get('query')
    mode = arguments.get('mode')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    if mode is not None and not isinstance(mode, str):
        raise ToolExecutionError('mode must be a string')
    max_profiles = _coerce_int(arguments, 'max_profiles', 20)
    return runtime.render_profiles_index(query=query, mode=mode, limit=max_profiles)


def _remote_connect(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_remote_runtime(context)
    target = _require_string(arguments, 'target')
    mode = arguments.get('mode')
    if mode is not None and not isinstance(mode, str):
        raise ToolExecutionError('mode must be a string')
    report = runtime.connect(target, mode=mode)
    return (
        report.as_text(),
        {
            'action': 'remote_connect',
            'mode': report.mode,
            'target': report.target or target,
            'profile_name': report.profile_name,
            'session_url': report.session_url,
            'workspace_cwd': report.workspace_cwd,
        },
    )


def _remote_disconnect(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_remote_runtime(context)
    reason = arguments.get('reason')
    if reason is not None and not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    report = runtime.disconnect(reason=reason or 'tool_request')
    return (
        report.as_text(),
        {
            'action': 'remote_disconnect',
            'mode': report.mode,
            'target': report.target,
        },
    )


def _worktree_status(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    _ = arguments
    runtime = _require_worktree_runtime(context)
    return '\n'.join(['# Worktree', '', runtime.render_summary()])


def _worktree_enter(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    _ensure_write_allowed(context)  # S17: git worktree add changes the filesystem
    runtime = _require_worktree_runtime(context)
    name = arguments.get('name')
    if name is not None and not isinstance(name, str):
        raise ToolExecutionError('name must be a string')
    try:
        report = runtime.enter(name=name)
    except (RuntimeError, ValueError) as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        report.as_text(),
        {
            'action': 'worktree_enter',
            'cwd_update': report.worktree_path,
            'repo_root': report.repo_root,
            'worktree_path': report.worktree_path,
            'worktree_branch': report.worktree_branch,
            'session_name': report.session_name,
            'before_cwd': report.original_cwd,
            'after_cwd': report.worktree_path,
            'path': report.worktree_path,
        },
    )


def _worktree_exit(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    _ensure_write_allowed(context)  # S17: exit can remove the worktree and discard its changes
    runtime = _require_worktree_runtime(context)
    action = arguments.get('action', 'keep')
    discard_changes = arguments.get('discard_changes', False)
    if not isinstance(action, str):
        raise ToolExecutionError('action must be a string')
    if not isinstance(discard_changes, bool):
        raise ToolExecutionError('discard_changes must be a boolean')
    try:
        report = runtime.exit(
            action=action,
            discard_changes=discard_changes,
        )
    except (RuntimeError, ValueError) as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        report.as_text(),
        {
            'action': 'worktree_exit',
            'cwd_update': report.original_cwd or report.current_cwd,
            'repo_root': report.repo_root,
            'worktree_path': report.worktree_path,
            'worktree_branch': report.worktree_branch,
            'session_name': report.session_name,
            'before_cwd': report.worktree_path,
            'after_cwd': report.original_cwd or report.current_cwd,
            'exit_action': report.metadata.get('action'),
            'discard_changes': report.metadata.get('discard_changes'),
            'path': report.worktree_path,
        },
    )


def _workflow_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_workflow_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_workflows = _coerce_int(arguments, 'max_workflows', 50)
    workflows = runtime.list_workflows(query=query, limit=max_workflows)
    lines = ['# Workflows', '']
    if not workflows:
        lines.append('No local workflows discovered.')
        return '\n'.join(lines)
    for workflow in workflows:
        description = workflow.description or 'No description.'
        lines.append(f'- {workflow.name} ; steps={len(workflow.steps)} ; {description}')
    return '\n'.join(lines)


def _workflow_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_workflow_runtime(context)
    workflow_name = _require_string(arguments, 'workflow_name')
    try:
        return runtime.render_workflow(workflow_name)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown workflow: {workflow_name}') from exc


def _workflow_run(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_workflow_runtime(context)
    workflow_name = _require_string(arguments, 'workflow_name')
    raw_arguments = arguments.get('arguments', {})
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise ToolExecutionError('arguments must be an object')
    try:
        rendered = runtime.render_run_report(workflow_name, arguments=raw_arguments)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown workflow: {workflow_name}') from exc
    return (
        rendered,
        {
            'action': 'workflow_run',
            'workflow_name': workflow_name,
            'argument_keys': sorted(str(key) for key in raw_arguments.keys()),
        },
    )


def _remote_trigger(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[str, dict[str, Any]]:
    runtime = _require_remote_trigger_runtime(context)
    action = _require_string(arguments, 'action').strip().lower()
    if action == 'list':
        query = arguments.get('query')
        if query is not None and not isinstance(query, str):
            raise ToolExecutionError('query must be a string')
        max_triggers = _coerce_int(arguments, 'max_triggers', 50)
        rendered = runtime.render_trigger_index(query=query)
        return (
            rendered,
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'listed_triggers': len(runtime.list_triggers(query=query, limit=max_triggers)),
            },
        )
    if action == 'get':
        trigger_id = _require_string(arguments, 'trigger_id')
        try:
            rendered = runtime.render_trigger(trigger_id)
        except KeyError as exc:
            raise ToolExecutionError(f'Unknown remote trigger: {trigger_id}') from exc
        return (
            rendered,
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger_id,
            },
        )
    body = arguments.get('body', {})
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ToolExecutionError('body must be an object')
    if action == 'create':
        try:
            trigger = runtime.create_trigger(body)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolExecutionError(str(exc)) from exc
        return (
            runtime.render_trigger(trigger.trigger_id),
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger.trigger_id,
            },
        )
    trigger_id = _require_string(arguments, 'trigger_id')
    if action == 'update':
        try:
            trigger = runtime.update_trigger(trigger_id, body)
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolExecutionError(str(exc)) from exc
        return (
            runtime.render_trigger(trigger.trigger_id),
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger.trigger_id,
            },
        )
    if action == 'run':
        try:
            rendered = runtime.render_run_report(trigger_id, body=body)
        except KeyError as exc:
            raise ToolExecutionError(f'Unknown remote trigger: {trigger_id}') from exc
        return (
            rendered,
            {
                'action': 'remote_trigger',
                'remote_trigger_action': action,
                'trigger_id': trigger_id,
                'body_keys': sorted(str(key) for key in body.keys()),
            },
        )
    raise ToolExecutionError('action must be one of list, get, create, update, or run')


def _team_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_team_runtime(context)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_teams = _coerce_int(arguments, 'max_teams', 50)
    return runtime.render_teams_index(query=query, limit=max_teams)


def _team_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    try:
        return runtime.render_team(team_name)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc


def _team_create(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    description = arguments.get('description')
    members = arguments.get('members', ())
    metadata = arguments.get('metadata')
    if description is not None and not isinstance(description, str):
        raise ToolExecutionError('description must be a string')
    if not isinstance(members, (list, tuple)):
        raise ToolExecutionError('members must be an array of strings')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    try:
        team = runtime.create_team(
            team_name,
            description=description,
            members=members,
            metadata=metadata,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Team already exists: {team_name}') from exc
    return (
        f'created team {team.name}',
        {
            'action': 'team_create',
            'team_name': team.name,
            'member_count': len(team.members),
            'path': str(runtime.state_path),
        },
    )


def _team_delete(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    try:
        team = runtime.delete_team(team_name)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc
    return (
        f'deleted team {team.name}',
        {
            'action': 'team_delete',
            'team_name': team.name,
            'path': str(runtime.state_path),
        },
    )


def _send_message(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_team_runtime(context)
    team_name = _require_string(arguments, 'team_name')
    message = _require_string(arguments, 'message')
    sender = arguments.get('sender', 'agent')
    recipient = arguments.get('recipient')
    metadata = arguments.get('metadata')
    if sender is not None and not isinstance(sender, str):
        raise ToolExecutionError('sender must be a string')
    if recipient is not None and not isinstance(recipient, str):
        raise ToolExecutionError('recipient must be a string')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    try:
        stored = runtime.send_message(
            team_name=team_name,
            text=message,
            sender=sender or 'agent',
            recipient=recipient,
            metadata=metadata,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc
    return (
        f'sent message to team {stored.team_name}',
        {
            'action': 'send_message',
            'team_name': stored.team_name,
            'sender': stored.sender,
            'recipient': stored.recipient,
            'message_id': stored.message_id,
            'message_preview': _snapshot_text(stored.text),
            'path': str(runtime.state_path),
        },
    )


def _team_messages(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_team_runtime(context)
    team_name = arguments.get('team_name')
    if team_name is not None and not isinstance(team_name, str):
        raise ToolExecutionError('team_name must be a string')
    limit = _coerce_int(arguments, 'limit', 20)
    try:
        return runtime.render_messages(team_name=team_name, limit=limit)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown team: {team_name}') from exc


def _task_list(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_task_runtime(context)
    status = arguments.get('status')
    owner = arguments.get('owner')
    actionable_only = arguments.get('actionable_only', False)
    if status is not None and not isinstance(status, str):
        raise ToolExecutionError('status must be a string')
    if owner is not None and not isinstance(owner, str):
        raise ToolExecutionError('owner must be a string')
    if not isinstance(actionable_only, bool):
        raise ToolExecutionError('actionable_only must be a boolean')
    max_tasks = _coerce_int(arguments, 'max_tasks', 50)
    return runtime.render_tasks(
        status=status,
        owner=owner,
        actionable_only=actionable_only,
        limit=max_tasks,
    )


def _task_next(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_task_runtime(context)
    max_tasks = _coerce_int(arguments, 'max_tasks', 10)
    return runtime.render_next_tasks(limit=max_tasks)


def _task_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    runtime = _require_task_runtime(context)
    return runtime.render_task(_require_string(arguments, 'task_id'))


def _plan_get(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    del arguments
    runtime = _require_plan_runtime(context)
    return runtime.render_plan()


def _update_plan(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_plan_runtime(context)
    items = arguments.get('items')
    if not isinstance(items, list):
        raise ToolExecutionError('items must be an array of plan step objects')
    explanation = arguments.get('explanation')
    if explanation is not None and not isinstance(explanation, str):
        raise ToolExecutionError('explanation must be a string')
    sync_tasks = arguments.get('sync_tasks', True)
    if not isinstance(sync_tasks, bool):
        raise ToolExecutionError('sync_tasks must be a boolean')
    mutation = runtime.update_plan(
        [item for item in items if isinstance(item, dict)],
        explanation=explanation,
        task_runtime=context.task_runtime,
        sync_tasks=sync_tasks,
    )
    return (
        f'updated plan with {mutation.after_count} step(s)',
        _plan_mutation_metadata(
            action='update_plan',
            mutation=mutation,
            total_steps=mutation.after_count,
            sync_tasks=sync_tasks,
        ),
    )


def _plan_clear(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_plan_runtime(context)
    sync_tasks = arguments.get('sync_tasks', True)
    if not isinstance(sync_tasks, bool):
        raise ToolExecutionError('sync_tasks must be a boolean')
    mutation = runtime.clear_plan(
        task_runtime=context.task_runtime if sync_tasks else None,
    )
    return (
        'cleared local plan',
        _plan_mutation_metadata(
            action='plan_clear',
            mutation=mutation,
            total_steps=0,
            sync_tasks=sync_tasks,
        ),
    )


def _task_create(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    title = _require_string(arguments, 'title')
    description = arguments.get('description')
    status = arguments.get('status', 'todo')
    priority = arguments.get('priority')
    task_id = arguments.get('task_id')
    active_form = arguments.get('active_form')
    owner = arguments.get('owner')
    blocks = arguments.get('blocks', [])
    blocked_by = arguments.get('blocked_by', [])
    metadata = arguments.get('metadata')
    if description is not None and not isinstance(description, str):
        raise ToolExecutionError('description must be a string')
    if status is not None and not isinstance(status, str):
        raise ToolExecutionError('status must be a string')
    if priority is not None and not isinstance(priority, str):
        raise ToolExecutionError('priority must be a string')
    if task_id is not None and not isinstance(task_id, str):
        raise ToolExecutionError('task_id must be a string')
    if active_form is not None and not isinstance(active_form, str):
        raise ToolExecutionError('active_form must be a string')
    if owner is not None and not isinstance(owner, str):
        raise ToolExecutionError('owner must be a string')
    if not isinstance(blocks, list):
        raise ToolExecutionError('blocks must be an array of strings')
    if not isinstance(blocked_by, list):
        raise ToolExecutionError('blocked_by must be an array of strings')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    mutation = runtime.create_task(
        title=title,
        description=description,
        status=status or 'pending',
        priority=priority,
        task_id=task_id,
        active_form=active_form,
        owner=owner,
        blocks=blocks,
        blocked_by=blocked_by,
        metadata=metadata,
    )
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task creation succeeded but returned no task object')
    return (
        f'created task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_create',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_update(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    title = arguments.get('title')
    description = arguments.get('description')
    status = arguments.get('status')
    priority = arguments.get('priority')
    active_form = arguments.get('active_form')
    owner = arguments.get('owner')
    blocks = arguments.get('blocks')
    blocked_by = arguments.get('blocked_by')
    metadata = arguments.get('metadata')
    for key, value in (
        ('title', title),
        ('description', description),
        ('status', status),
        ('priority', priority),
        ('active_form', active_form),
        ('owner', owner),
    ):
        if value is not None and not isinstance(value, str):
            raise ToolExecutionError(f'{key} must be a string')
    if blocks is not None and not isinstance(blocks, list):
        raise ToolExecutionError('blocks must be an array of strings')
    if blocked_by is not None and not isinstance(blocked_by, list):
        raise ToolExecutionError('blocked_by must be an array of strings')
    if metadata is not None and not isinstance(metadata, dict):
        raise ToolExecutionError('metadata must be an object')
    try:
        mutation = runtime.update_task(
            task_id,
            title=title,
            description=description,
            status=status,
            priority=priority,
            active_form=active_form,
            owner=owner,
            blocks=blocks,
            blocked_by=blocked_by,
            metadata=metadata,
        )
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task update succeeded but returned no task object')
    return (
        f'updated task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_update',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_start(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    owner = arguments.get('owner')
    active_form = arguments.get('active_form')
    if owner is not None and not isinstance(owner, str):
        raise ToolExecutionError('owner must be a string')
    if active_form is not None and not isinstance(active_form, str):
        raise ToolExecutionError('active_form must be a string')
    try:
        mutation = runtime.start_task(task_id, owner=owner, active_form=active_form)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task start succeeded but returned no task object')
    return (
        f'started task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_start',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_complete(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    try:
        mutation = runtime.complete_task(task_id)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = runtime.get_task(task_id)
    if task is None:
        raise ToolExecutionError(f'Task completion succeeded but task {task_id} not found')
    return (
        f'completed task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_complete',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_block(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    blocked_by = arguments.get('blocked_by')
    reason = arguments.get('reason')
    if blocked_by is not None and not isinstance(blocked_by, list):
        raise ToolExecutionError('blocked_by must be an array of strings')
    if reason is not None and not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    try:
        mutation = runtime.block_task(task_id, blocked_by=blocked_by, reason=reason)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task block succeeded but returned no task object')
    return (
        f'blocked task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_block',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _task_cancel(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    task_id = _require_string(arguments, 'task_id')
    reason = arguments.get('reason')
    if reason is not None and not isinstance(reason, str):
        raise ToolExecutionError('reason must be a string')
    try:
        mutation = runtime.cancel_task(task_id, reason=reason)
    except KeyError as exc:
        raise ToolExecutionError(f'Unknown task id: {task_id}') from exc
    task = mutation.task
    if task is None:
        raise ToolExecutionError('Task cancellation succeeded but returned no task object')
    return (
        f'cancelled task {task.task_id}: {task.title} [{task.status}]',
        _task_mutation_metadata(
            action='task_cancel',
            mutation=mutation,
            task_id=task.task_id,
            task_status=task.status,
            total_tasks=mutation.after_count,
        ),
    )


def _todo_write(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    _ensure_write_allowed(context)
    runtime = _require_task_runtime(context)
    items = arguments.get('items')
    if not isinstance(items, list):
        raise ToolExecutionError('items must be an array of task objects')
    mutation = runtime.replace_tasks(
        [item for item in items if isinstance(item, dict)]
    )
    return (
        f'replaced todo list with {mutation.after_count} task(s)',
        _task_mutation_metadata(
            action='todo_write',
            mutation=mutation,
            total_tasks=mutation.after_count,
        ),
    )


def _bash_stream_cap(max_output_chars: int) -> int:
    """PC-safety cap on what one streamed bash command keeps in memory and streams:
    SIGIL_BASH_STREAM_MAX_CHARS (default 10 x max_output_chars; 0 = no cap)."""
    raw = os.environ.get('SIGIL_BASH_STREAM_MAX_CHARS', '').strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            from .debug_log import get_logger, sanitize_log

            get_logger('tools').debug('SIGIL_BASH_STREAM_MAX_CHARS=%s is not a number; default used',
                                      sanitize_log(raw, 40))
    return max(0, int(max_output_chars or 0)) * 10


class _HeadTailText:
    """Keeps the first and last limit/2 characters of a stream; the middle is counted, not kept."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.head: list[str] = []
        self.head_len = 0
        self.tail: 'collections.deque[str]' = collections.deque()
        self.tail_len = 0
        self.dropped = 0

    def add(self, text: str) -> None:
        if self.limit <= 0 or self.head_len < self.limit // 2:
            self.head.append(text)
            self.head_len += len(text)
            return
        self.tail.append(text)
        self.tail_len += len(text)
        while self.tail and self.tail_len > self.limit - self.limit // 2:
            removed = self.tail.popleft()
            self.tail_len -= len(removed)
            self.dropped += len(removed)

    def text(self) -> str:
        middle = (f'\n[... {self.dropped:,} characters cut (SIGIL_BASH_STREAM_MAX_CHARS={self.limit}) ...]\n'
                  if self.dropped else '')
        return ''.join(self.head) + middle + ''.join(self.tail)


def _stream_bash(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> Iterator[ToolStreamUpdate]:
    try:
        command = _normalize_windows_paths(_require_string(arguments, 'command'))
        guard_warning = _ensure_shell_allowed(command, context)
        timeout_seconds = _bash_timeout(arguments, context)
        wrapped = _sandbox_argv(command, context)
        subprocess_env = _build_subprocess_env(context)
        if wrapped is not None:
            from .sandbox import proxy_env

            popen_args = wrapped
            popen_kwargs = {'shell': False, 'stdin': subprocess.DEVNULL,
                            **({'creationflags': _WIN_CHILD_FLAGS} if sys.platform == 'win32' else {})}
            subprocess_env = proxy_env(subprocess_env)
        elif sys.platform == 'win32':
            popen_args = [_BASH_EXECUTABLE, '-c', command]
            # Own process group + no console: a child that sends Ctrl+C (e.g. via
            # os.kill(pid, 0) on Windows) can no longer interrupt the agent itself.
            popen_kwargs = {'shell': False, 'creationflags': _WIN_CHILD_FLAGS, 'stdin': subprocess.DEVNULL}
        else:
            popen_args = [command]
            popen_kwargs = {'shell': True, 'executable': _BASH_EXECUTABLE}
        win_limits = _windows_sandbox_limits() if wrapped is None and sys.platform == 'win32' else None
        if win_limits is not None:
            from .background_shells import _CREATE_SUSPENDED

            popen_kwargs['creationflags'] = popen_kwargs.get('creationflags', 0) | _CREATE_SUSPENDED

        process = subprocess.Popen(
            args=popen_args,
            **popen_kwargs,
            cwd=context.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=subprocess_env,
        )
        win_job = None
        if win_limits is not None:  # opt-in Windows sandbox: limits for the command and all it starts
            from .background_shells import _win_contain

            win_job = _win_contain(process, win_limits)
    except (ToolPermissionError, ToolExecutionError, OSError, subprocess.SubprocessError) as exc:
        yield ToolStreamUpdate(
            kind='result',
            result=ToolExecutionResult(name='bash', ok=False, content=str(exc)),
        )
        return

    import queue
    import threading

    def _reader(stream, stream_name, out_queue):
        try:
            for line in stream:
                out_queue.put(('delta', stream_name, line))
        finally:
            out_queue.put(('eof', stream_name, None))

    q = queue.Queue()
    t_out = threading.Thread(target=_reader, args=(process.stdout, 'stdout', q), daemon=True)
    t_err = threading.Thread(target=_reader, args=(process.stderr, 'stderr', q), daemon=True)
    t_out.start()
    t_err.start()

    stream_cap = _bash_stream_cap(context.max_output_chars)
    stdout_chunks = _HeadTailText(stream_cap)
    stderr_chunks = _HeadTailText(stream_cap)
    streamed_chars = 0
    stream_paused = False
    deadline = time.monotonic() + timeout_seconds
    timeout_error: str | None = None
    alive_streams = 2

    try:
        while alive_streams > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_error = (
                    f'Command timed out after {timeout_seconds:.1f}s: {command}. '
                    f'Re-run with a larger timeout_seconds (max {MAX_BASH_TIMEOUT_SECONDS:.0f}) '
                    'or start it in the background.'
                )
                _kill_process_tree(process)
                break
            
            try:
                msg_type, stream_name, line = q.get(timeout=min(remaining, 0.1))
                if msg_type == 'eof':
                    alive_streams -= 1
                else:
                    if stream_name == 'stdout':
                        stdout_chunks.add(line)
                    else:
                        stderr_chunks.add(line)
                    if stream_cap and streamed_chars + len(line) > stream_cap:
                        if not stream_paused:  # one notice, then keep draining without streaming
                            stream_paused = True
                            yield ToolStreamUpdate(
                                kind='delta',
                                content=(f'\n[output over {stream_cap:,} characters; live view paused, '
                                         'the end is kept (SIGIL_BASH_STREAM_MAX_CHARS)]\n'),
                                stream=stream_name,
                            )
                        continue
                    streamed_chars += len(line)
                    yield ToolStreamUpdate(
                        kind='delta',
                        content=line,
                        stream=stream_name,
                    )
            except queue.Empty as _ignored:
                _log_ignored(_ignored)
                continue

        exit_code = process.wait()
    finally:
        try:
            process.kill()
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass
        if win_job is not None:
            from .background_shells import _win_terminate_job

            _win_terminate_job(win_job)
    if timeout_error is not None:
        yield ToolStreamUpdate(
            kind='result',
            result=ToolExecutionResult(
                name='bash',
                ok=False,
                content=timeout_error,
                metadata={
                    'action': 'bash',
                    'command': command,
                    'exit_code': exit_code,
                    'timed_out': True,
                    'stdout_preview': _snapshot_text(stdout_chunks.text()),
                    'stderr_preview': _snapshot_text(stderr_chunks.text()),
                },
            ),
        )
        return

    stdout = stdout_chunks.text()
    stderr = stderr_chunks.text()
    cut = stdout_chunks.dropped + stderr_chunks.dropped
    payload = [
        *([guard_warning] if guard_warning else []),
        f'exit_code={exit_code}',
        *([f'[{cut:,} characters from the middle of the output were not kept '
           f'(SIGIL_BASH_STREAM_MAX_CHARS={stream_cap}; 0 = no cap)]'] if cut else []),
        '[stdout]',
        stdout.rstrip(),
        '[stderr]',
        stderr.rstrip(),
    ]
    yield ToolStreamUpdate(
        kind='result',
        result=ToolExecutionResult(
            name='bash',
            ok=True,
            content=_truncate_output('\n'.join(payload).strip(), context.max_output_chars),
            metadata={
                'action': 'bash',
                'command': command,
                'exit_code': exit_code,
                'streamed': True,
                'stdout_preview': _snapshot_text(stdout),
                'stderr_preview': _snapshot_text(stderr),
                'output_preview': _snapshot_text('\n'.join(payload).strip()),
            },
        ),
    )


def _background_max_turns() -> int:
    """Turn cap for background agents: SIGIL_BACKGROUND_MAX_TURNS (default 40)."""
    try:
        return max(1, int(os.environ.get('SIGIL_BACKGROUND_MAX_TURNS', '') or 40))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return 40


def _background_runtime_for(context: ToolExecutionContext):
    from .background_runtime import BackgroundSessionRuntime
    return BackgroundSessionRuntime(context.root / '.port_sessions' / 'background')


def _background_agent_start(arguments: dict[str, Any], context: ToolExecutionContext):
    """Launch a detached agent process; returns immediately with its id."""
    from .background_runtime import build_background_worker_command
    prompt = _require_string(arguments, 'prompt').strip()
    if not prompt:
        raise ToolExecutionError('prompt must not be empty')
    model_config = context.model_config
    if model_config is None:
        raise ToolExecutionError('background agents are unavailable: no model configuration in this runtime')
    max_turns = max(1, min(_coerce_int(arguments, 'max_turns', 12), _background_max_turns()))
    # Same rule as delegate_agent: a child can only get permissions the parent already has.
    allow_write = context.permissions.allow_file_write and bool(arguments.get('allow_write', False))
    allow_shell = context.permissions.allow_shell_commands and bool(arguments.get('allow_shell', False))
    model = str(arguments.get('model') or model_config.model)

    runtime = _background_runtime_for(context)
    from .background_runtime import background_limit_message

    limit_message = background_limit_message(runtime)
    if limit_message:
        raise ToolExecutionError(limit_message)
    background_id = runtime.create_id()
    forwarded = [
        '--cwd', str(context.root),
        '--max-turns', str(max_turns),
        '--model', model,
        '--base-url', str(model_config.base_url),
        '--timeout-seconds', str(getattr(model_config, 'timeout_seconds', 120.0)),
        '--background-root', str(runtime.root),
    ]
    if allow_write:
        forwarded.append('--allow-write')
    if allow_shell:
        forwarded.append('--allow-shell')
    for path in context.additional_working_directories:
        forwarded.extend(['--add-dir', str(path)])
    # The parent's live state: effort, sampling, pricing and permission mode.
    effort = getattr(model_config, 'effort', None)
    if effort:
        forwarded.extend(['--effort', str(effort)])
    if getattr(model_config, 'temperature', None) is not None:
        forwarded.extend(['--temperature', str(model_config.temperature)])
    pricing = getattr(model_config, 'pricing', None)
    if pricing is not None:
        forwarded.extend(['--input-cost-per-million', str(pricing.input_cost_per_million_tokens_usd),
                          '--output-cost-per-million', str(pricing.output_cost_per_million_tokens_usd)])
    if context.permission_mode in ('plan', 'acceptEdits', 'bypass', 'dontAsk'):
        forwarded.extend(['--permission-mode', context.permission_mode])
    # The parent's own limits go along, as with `sigil agent-bg`: a background agent never has
    # tools the session was started without.
    for rule in (*context.cli_allowed_tools, *context.session_allow_rules):
        forwarded.extend(['--allowed-tools', rule])
    for rule in context.cli_disallowed_tools:
        forwarded.extend(['--disallowed-tools', rule])
    if context.cli_tools is not None:
        forwarded.extend(['--tools', ','.join(context.cli_tools)])
    if context.permissions.allow_destructive_shell_commands and bool(arguments.get('allow_destructive', False)):
        forwarded.append('--unsafe')  # only when the parent may run destructive commands itself
    command = build_background_worker_command(
        background_id=background_id, prompt=prompt, forwarded_args=forwarded,
    )
    # API key goes through the environment (read by --api-key's default), never argv,
    # so it does not show up in the process list or the saved session record.
    env = dict(os.environ)
    from .main import _forward_key

    _forward_key(env, str(model_config.base_url or ''), str(model_config.api_key))
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SIGIL_BACKGROUND'] = '1'
    from .child_env import strip_ssh_agent

    strip_ssh_agent(env)  # N21: SIGIL_SSH_AGENT=off reaches background agents too
    record = runtime.launch(
        command,
        prompt=prompt,
        workspace_cwd=context.root,
        model=model,
        background_id=background_id,
        process_cwd=Path(__file__).resolve().parent.parent,
        env=env,
    )
    return (
        '\n'.join([
            f'Background agent started: {record.background_id}',
            f'- pid: {record.pid}',
            f'- model: {model}',
            f'- permissions: write={allow_write} shell={allow_shell}',
            f'- log: {record.log_path}',
            'It runs independently; keep working and check it with background_agent_status.',
        ]),
        {'action': 'background_agent_start', 'background_id': record.background_id, 'pid': record.pid},
    )


def _background_agent_status(arguments: dict[str, Any], context: ToolExecutionContext):
    """List all background agents, or show one agent's status plus the tail of its log."""
    runtime = _background_runtime_for(context)
    background_id = arguments.get('background_id')
    if not background_id:
        return runtime.render_ps(), {'action': 'background_agent_status'}
    if not isinstance(background_id, str) or not re.fullmatch(r'bg_[0-9a-f]{12}', background_id):
        raise ToolExecutionError('background_id must look like bg_<12 hex chars>')
    if not runtime.record_path(background_id).exists():
        raise ToolExecutionError(f'Unknown background agent: {background_id}')
    tail = max(1, min(_coerce_int(arguments, 'tail', 60), 400))
    runtime.check_stalled(runtime.load_record(background_id))
    text = runtime.render_logs(background_id, tail=tail)
    record = runtime.load_record(background_id)
    result = runtime.read_result(background_id)
    if result is not None:
        text += ('\n\n# Result\n' + str(result.get('final_output', '')).strip()
                 + f"\n\nstop_reason={result.get('stop_reason')} session_id={result.get('session_id')} "
                 f"cost_usd={result.get('total_cost_usd')}")
    return (
        _truncate_output(text, context.max_output_chars),
        {'action': 'background_agent_status', 'background_id': background_id, 'status': record.status},
    )


def _background_agent_restart(arguments: dict[str, Any], context: ToolExecutionContext):
    runtime = _background_runtime_for(context)
    background_id = _require_string(arguments, 'background_id')
    if not re.fullmatch(r'bg_[0-9a-f]{12}', background_id):
        raise ToolExecutionError('background_id must look like bg_<12 hex chars>')
    if not runtime.record_path(background_id).exists():
        raise ToolExecutionError(f'Unknown background agent: {background_id}')
    env = dict(os.environ)
    if context.model_config is not None:
        from .main import _forward_key

        _forward_key(env, str(context.model_config.base_url or ''), str(context.model_config.api_key))
    env['PYTHONIOENCODING'] = 'utf-8'
    env['SIGIL_BACKGROUND'] = '1'
    from .child_env import strip_ssh_agent

    strip_ssh_agent(env)
    prompt = arguments.get('prompt')
    from .background_runtime import RestartLimits

    # why (S05): the record file can be planted in the workspace; the worker gets no more than this
    # session has (SIGIL_BACKGROUND_RESTART_UNLIMITED=1 or `sigil agent-restart` keep the record's options).
    limits = RestartLimits(
        allow_write=context.permissions.allow_file_write,
        allow_shell=context.permissions.allow_shell_commands,
        allow_destructive=context.permissions.allow_destructive_shell_commands,
        permission_mode=str(context.permission_mode or 'default'),
        allowed_rules=tuple(str(rule) for rule in (*context.cli_allowed_tools, *context.session_allow_rules)),
        roots=tuple(Path(path).resolve() for path in (context.root, *context.additional_working_directories)),
        base_url=str(getattr(context.model_config, 'base_url', '') or os.environ.get('OPENAI_BASE_URL', '')),
    )
    try:
        record = runtime.restart(background_id, prompt=str(prompt) if prompt else None, env=env, limits=limits)
    except RuntimeError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        f'Background agent {record.background_id} restarted (pid {record.pid}); check it with background_agent_status.',
        {'action': 'background_agent_restart', 'background_id': record.background_id, 'pid': record.pid},
    )


def _background_agent_kill(arguments: dict[str, Any], context: ToolExecutionContext):
    runtime = _background_runtime_for(context)
    background_id = _require_string(arguments, 'background_id')
    if not re.fullmatch(r'bg_[0-9a-f]{12}', background_id):
        raise ToolExecutionError('background_id must look like bg_<12 hex chars>')
    if not runtime.record_path(background_id).exists():
        raise ToolExecutionError(f'Unknown background agent: {background_id}')
    record = runtime.kill(background_id)
    return (
        f'Background agent {record.background_id}: status={record.status}',
        {'action': 'background_agent_kill', 'background_id': background_id, 'status': record.status},
    )


def _delegate_agent_placeholder(
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> str:
    raise ToolExecutionError(
        'delegate_agent must be handled by the runtime and is not available as a standalone tool handler'
    )


def _lsp_query(arguments: dict[str, Any], context: ToolExecutionContext):
    runtime = _require_lsp_runtime(context)
    operation = _require_string(arguments, 'operation')
    file_path = _require_string(arguments, 'file_path')
    line = _coerce_int(arguments, 'line', 1)
    character = _coerce_int(arguments, 'character', 1)
    query = arguments.get('query')
    if query is not None and not isinstance(query, str):
        raise ToolExecutionError('query must be a string')
    max_results = _coerce_int(arguments, 'max_results', 50)
    try:
        result = runtime.query(
            operation,
            file_path=file_path,
            line=line,
            character=character,
            query=query,
            max_results=max_results,
        )
    except KeyError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return (
        result.content,
        {
            'action': 'lsp_query',
            'operation': result.operation,
            'file_path': file_path,
            'line': line,
            'character': character,
            'result_count': result.result_count,
            'file_count': result.file_count,
            'symbol_name': result.symbol_name,
        },
    )


def _require_account_runtime(context: ToolExecutionContext):
    if context.account_runtime is None:
        raise ToolExecutionError('No local account runtime is available.')
    return context.account_runtime


def _require_ask_user_runtime(context: ToolExecutionContext):
    if context.ask_user_runtime is None:
        raise ToolExecutionError('No local ask-user runtime is available.')
    return context.ask_user_runtime


def _require_search_runtime(context: ToolExecutionContext):
    if context.search_runtime is None or not context.search_runtime.has_search_runtime():
        raise ToolExecutionError(
            'No local search provider is available. Add a .sigil-search.json or .sigil/search.json manifest, '
            'or set SEARXNG_BASE_URL, BRAVE_SEARCH_API_KEY, or TAVILY_API_KEY.'
        )
    return context.search_runtime


def _require_config_runtime(context: ToolExecutionContext):
    if context.config_runtime is None:
        raise ToolExecutionError('No local config runtime is available.')
    return context.config_runtime


def _require_lsp_runtime(context: ToolExecutionContext):
    if context.lsp_runtime is None or not context.lsp_runtime.has_lsp_support():
        raise ToolExecutionError(
            'No local LSP runtime is available. Add supported source files to the workspace or a .sigil-lsp.json manifest.'
        )
    return context.lsp_runtime


def _require_mcp_runtime(context: ToolExecutionContext):
    if (
        context.mcp_runtime is None
        or (
            not context.mcp_runtime.resources
            and not context.mcp_runtime.servers
        )
    ):
        raise ToolExecutionError(
            'No MCP runtime is available. Add a .sigil-mcp.json, .mcp.json, or mcpServers manifest first.'
        )
    return context.mcp_runtime


def _require_remote_runtime(context: ToolExecutionContext):
    if context.remote_runtime is None:
        raise ToolExecutionError('Local remote runtime is not available.')
    return context.remote_runtime


def _require_remote_trigger_runtime(context: ToolExecutionContext):
    if context.remote_trigger_runtime is None:
        raise ToolExecutionError('Local remote trigger runtime is not available.')
    return context.remote_trigger_runtime


def _require_plan_runtime(context: ToolExecutionContext):
    if context.plan_runtime is None:
        raise ToolExecutionError('Local plan runtime is not available.')
    return context.plan_runtime


def _require_task_runtime(context: ToolExecutionContext):
    if context.task_runtime is None:
        raise ToolExecutionError('Local task runtime is not available.')
    return context.task_runtime


def _require_team_runtime(context: ToolExecutionContext):
    if context.team_runtime is None:
        raise ToolExecutionError('Local team runtime is not available.')
    return context.team_runtime


def _require_workflow_runtime(context: ToolExecutionContext):
    if context.workflow_runtime is None or not context.workflow_runtime.has_workflows():
        raise ToolExecutionError('Local workflow runtime is not available.')
    return context.workflow_runtime


def _require_worktree_runtime(context: ToolExecutionContext):
    if context.worktree_runtime is None:
        raise ToolExecutionError('Local worktree runtime is not available.')
    return context.worktree_runtime


def _task_mutation_metadata(
    *,
    action: str,
    mutation,
    task_id: str | None = None,
    task_status: str | None = None,
    total_tasks: int,
) -> dict[str, Any]:
    try:
        relative_path = str(Path(mutation.store_path).relative_to(Path.cwd()))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        relative_path = str(mutation.store_path)
    payload: dict[str, Any] = {
        'action': action,
        'path': relative_path,
        'before_sha256': mutation.before_sha256,
        'after_sha256': mutation.after_sha256,
        'before_preview': mutation.before_preview,
        'after_preview': mutation.after_preview,
        'before_task_count': mutation.before_count,
        'after_task_count': mutation.after_count,
        'total_tasks': total_tasks,
    }
    if task_id is not None:
        payload['task_id'] = task_id
    if task_status is not None:
        payload['task_status'] = task_status
    return payload


def _config_mutation_metadata(
    mutation,
    value: Any,
) -> dict[str, Any]:
    try:
        relative_path = str(Path(mutation.store_path).relative_to(Path.cwd()))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        relative_path = str(mutation.store_path)
    return {
        'action': 'config_set',
        'path': relative_path,
        'source_name': mutation.source_name,
        'key_path': mutation.key_path,
        'before_sha256': mutation.before_sha256,
        'after_sha256': mutation.after_sha256,
        'before_preview': mutation.before_preview,
        'after_preview': mutation.after_preview,
        'effective_key_count': mutation.effective_key_count,
        'value_preview': _snapshot_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True)
            if not isinstance(value, str)
            else value
        ),
    }


def _plan_mutation_metadata(
    *,
    action: str,
    mutation,
    total_steps: int,
    sync_tasks: bool,
) -> dict[str, Any]:
    try:
        relative_path = str(Path(mutation.store_path).relative_to(Path.cwd()))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        relative_path = str(mutation.store_path)
    payload: dict[str, Any] = {
        'action': action,
        'path': relative_path,
        'before_sha256': mutation.before_sha256,
        'after_sha256': mutation.after_sha256,
        'before_preview': mutation.before_preview,
        'after_preview': mutation.after_preview,
        'before_plan_count': mutation.before_count,
        'after_plan_count': mutation.after_count,
        'total_steps': total_steps,
        'sync_tasks': sync_tasks,
    }
    if mutation.explanation is not None:
        payload['explanation'] = mutation.explanation
    if mutation.synced_task_store_path is not None:
        try:
            payload['synced_task_store_path'] = str(
                Path(mutation.synced_task_store_path).relative_to(Path.cwd())
            )
        except ValueError as _ignored:
            _log_ignored(_ignored)
            payload['synced_task_store_path'] = mutation.synced_task_store_path
    if mutation.synced_task_sha256 is not None:
        payload['synced_task_sha256'] = mutation.synced_task_sha256
    if mutation.synced_tasks:
        payload['synced_tasks'] = mutation.synced_tasks
    return payload


def _drain_registered_streams(
    selector: selectors.BaseSelector,
    stdout_chunks: list[str],
    stderr_chunks: list[str],
) -> None:
    for key in list(selector.get_map().values()):
        try:
            remainder = key.fileobj.read()
        except OSError as _ignored:
            _log_ignored(_ignored)
            remainder = ''
        if not remainder:
            try:
                selector.unregister(key.fileobj)
            except (KeyError, ValueError, OSError) as _ignored:
                _log_ignored(_ignored)
                pass
            try:
                key.fileobj.close()
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass
            continue
        if key.data == 'stdout':
            stdout_chunks.append(remainder)
        else:
            stderr_chunks.append(remainder)
        try:
            selector.unregister(key.fileobj)
        except (KeyError, ValueError, OSError) as _ignored:
            _log_ignored(_ignored)
            pass
        try:
            key.fileobj.close()
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass


_SENSITIVE_ENV_KEYWORDS = (
    'SECRET',
    'TOKEN',
    'PASSWORD',
    'PRIVATE_KEY',
    'API_KEY',
    'CREDENTIAL',
    'AUTH',
)


def _is_sensitive_env_var(name: str) -> bool:
    """Return True if the environment variable name likely contains a secret."""
    upper = name.upper()
    return any(keyword in upper for keyword in _SENSITIVE_ENV_KEYWORDS)


def _build_subprocess_env(context: ToolExecutionContext) -> dict[str, str]:
    from .child_env import is_ssh_agent_var, strip_ssh_agent

    # N21: the SSH agent socket is not a secret value: bash keeps it (git over SSH) unless SIGIL_SSH_AGENT=off
    env = {
        key: value
        for key, value in os.environ.items()
        if is_ssh_agent_var(key) or not _is_sensitive_env_var(key)
    }
    env.update(context.extra_env)
    return strip_ssh_agent(env)


def _stream_static_text_result(
    result: ToolExecutionResult,
    *,
    chunk_size: int = 400,
) -> Iterator[ToolStreamUpdate]:
    content = result.content
    if content:
        for start in range(0, len(content), chunk_size):
            yield ToolStreamUpdate(
                kind='delta',
                content=content[start:start + chunk_size],
                stream='tool',
            )
    metadata = dict(result.metadata)
    metadata.setdefault('streamed', True)
    yield ToolStreamUpdate(
        kind='result',
        result=ToolExecutionResult(
            name=result.name,
            ok=result.ok,
            content=result.content,
            metadata=metadata,
        ),
    )
