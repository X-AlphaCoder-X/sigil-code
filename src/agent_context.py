from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache

from .prompt_constants import get_local_iso_date, get_scratchpad_instructions
from pathlib import Path

from .agent_plugin_cache import load_plugin_cache_summary
from .account_runtime import AccountRuntime
from .ask_user_runtime import AskUserRuntime
from .config_runtime import ConfigRuntime
from .hook_policy import HookPolicyRuntime
from .lsp_runtime import LSPRuntime
from .mcp_runtime import MCPRuntime
from .plan_runtime import PlanRuntime
from .plugin_runtime import PluginRuntime
from .remote_runtime import RemoteRuntime
from .remote_trigger_runtime import RemoteTriggerRuntime
from .search_runtime import SearchRuntime
from .skills_runtime import SkillsRuntime
from .task_runtime import TaskRuntime
from .team_runtime import TeamRuntime
from .workflow_runtime import WorkflowRuntime
from .worktree_runtime import WorktreeRuntime
from .agent_types import AgentRuntimeConfig
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

MAX_STATUS_CHARS = 2000
MAX_MEMORY_CHARACTER_COUNT = 40000
MEMORY_INSTRUCTION_PROMPT = (
    'Codebase and user instructions are shown below. Be sure to adhere to '
    'these instructions. IMPORTANT: These instructions override default '
    'behavior when they directly apply to the task.'
)

_SYSTEM_PROMPT_INJECTION: str | None = None


@dataclass(frozen=True)
class AgentContextSnapshot:
    cwd: Path
    shell: str
    platform_name: str
    os_version: str
    current_date: str
    is_git_repo: bool
    is_git_worktree: bool
    scratchpad_directory: str | None
    additional_working_directories: tuple[str, ...]
    user_context: dict[str, str]
    system_context: dict[str, str]


def clear_context_caches() -> None:
    _get_git_status_cached.cache_clear()
    _get_system_context_cached.cache_clear()
    _get_user_context_cached.cache_clear()


def get_system_prompt_injection() -> str | None:
    return _SYSTEM_PROMPT_INJECTION


def set_system_prompt_injection(value: str | None) -> None:
    global _SYSTEM_PROMPT_INJECTION
    _SYSTEM_PROMPT_INJECTION = value
    clear_context_caches()


def build_context_snapshot(
    runtime_config: AgentRuntimeConfig,
    *,
    scratchpad_directory: Path | None = None,
) -> AgentContextSnapshot:
    cwd = runtime_config.cwd.resolve()
    additional_dirs = tuple(
        str(path.resolve()) for path in runtime_config.additional_working_directories
    )
    return AgentContextSnapshot(
        cwd=cwd,
        shell=os.environ.get('SHELL', 'unknown'),
        platform_name=platform.system().lower() or os.name,
        os_version=_get_os_version(),
        current_date=get_local_iso_date(),
        is_git_repo=_is_git_repo(cwd),
        is_git_worktree=_is_git_worktree(cwd),
        scratchpad_directory=(
            str(scratchpad_directory.resolve()) if scratchpad_directory is not None else None
        ),
        additional_working_directories=additional_dirs,
        user_context=get_user_context(
            cwd,
            additional_dirs,
            runtime_config.disable_sigil_md_discovery,
            scratchpad_directory=scratchpad_directory,
        ),
        system_context=get_system_context(cwd, scratchpad_directory=scratchpad_directory),
    )


def get_git_status(cwd: Path) -> str | None:
    return _get_git_status_cached(str(cwd.resolve()))


def get_system_context(
    cwd: Path,
    *,
    scratchpad_directory: Path | None = None,
) -> dict[str, str]:
    scratchpad = str(scratchpad_directory.resolve()) if scratchpad_directory is not None else ''
    return dict(_get_system_context_cached(str(cwd.resolve()), scratchpad))


def get_user_context(
    cwd: Path,
    additional_working_directories: tuple[str, ...] = (),
    disable_sigil_md_discovery: bool = False,
    scratchpad_directory: Path | None = None,
) -> dict[str, str]:
    normalized_dirs = tuple(
        str(Path(path).resolve()) for path in additional_working_directories
    )
    return dict(
        _get_user_context_cached(
            str(cwd.resolve()),
            normalized_dirs,
            disable_sigil_md_discovery,
            str(scratchpad_directory.resolve()) if scratchpad_directory is not None else '',
        )
    )


def render_context_report(snapshot: AgentContextSnapshot, model: str) -> str:
    lines = [
        '# Context',
        '',
        '## Environment',
        f'- Primary working directory: {snapshot.cwd}',
        f'- Model: {model}',
        f'- Shell: {Path(snapshot.shell).name or snapshot.shell}',
        f'- Platform: {snapshot.platform_name}',
        f'- OS Version: {snapshot.os_version}',
        f'- Is a git repository: {snapshot.is_git_repo}',
        f'- Is a git worktree: {snapshot.is_git_worktree}',
        f'- Current date: {snapshot.current_date}',
    ]
    if snapshot.scratchpad_directory:
        lines.append(f'- Scratchpad directory: {snapshot.scratchpad_directory}')
    if snapshot.additional_working_directories:
        lines.extend(
            [
                '',
                '## Additional Working Directories',
                *[f'- {path}' for path in snapshot.additional_working_directories],
            ]
        )
    if snapshot.user_context:
        lines.extend(['', '## User Context'])
        for key, value in snapshot.user_context.items():
            lines.extend([f'### {key}', value, ''])
        while lines and not lines[-1]:
            lines.pop()
    if snapshot.system_context:
        lines.extend(['', '## System Context'])
        for key, value in snapshot.system_context.items():
            lines.extend([f'### {key}', value, ''])
        while lines and not lines[-1]:
            lines.pop()
    return '\n'.join(lines)


@lru_cache(maxsize=32)
def _get_system_context_cached(cwd: str, scratchpad_directory: str) -> dict[str, str]:
    context: dict[str, str] = {}
    # Opt-in: only include git status in the system prompt if the user
    # explicitly enables it. Keeping it off by default avoids leaking
    # untracked filenames and recent commit messages to the model
    # provider's safety filter and keeps sessions better isolated.
    if os.environ.get('SIGIL_INCLUDE_GIT_STATUS', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        git_status = _get_git_status_cached(cwd)
        if git_status is not None:
            context['gitStatus'] = git_status
    injection = get_system_prompt_injection()
    if injection:
        context['cacheBreaker'] = f'[CACHE_BREAKER: {injection}]'
    if scratchpad_directory:
        context['scratchpadDirectory'] = scratchpad_directory
    return context


@lru_cache(maxsize=32)
def _get_user_context_cached(
    cwd: str,
    additional_working_directories: tuple[str, ...],
    disable_sigil_md_discovery: bool,
    scratchpad_directory: str,
) -> dict[str, str]:
    context: dict[str, str] = {
        'currentDate': f"Today's date is {get_local_iso_date()}.",
    }
    if scratchpad_directory:
        context['scratchpad'] = get_scratchpad_instructions(scratchpad_directory)
    if disable_sigil_md_discovery:
        return context

    memory_bundle = _load_memory_bundle(Path(cwd), additional_working_directories)
    if memory_bundle:
        context['sigilMd'] = memory_bundle
    try:
        from .repo_map import build_repo_map, enabled as repo_map_enabled

        if repo_map_enabled(Path(cwd)):  # opt-in: SIGIL_REPO_MAP=1 or "repoMap": true
            context['repoMap'] = build_repo_map(Path(cwd))
    except Exception:
        from .debug_log import get_logger

        get_logger('context').warning('repo map failed for %s', cwd, exc_info=True)
    plugin_cache = load_plugin_cache_summary(Path(cwd), additional_working_directories)
    if plugin_cache:
        context['pluginCache'] = plugin_cache
    plugin_runtime = PluginRuntime.from_workspace(Path(cwd), additional_working_directories)
    if plugin_runtime.manifests:
        context['pluginRuntime'] = plugin_runtime.render_summary()
    hook_policy_runtime = HookPolicyRuntime.from_workspace(Path(cwd), additional_working_directories)
    if hook_policy_runtime.manifests:
        context['hookPolicy'] = hook_policy_runtime.render_summary()
        managed_settings = hook_policy_runtime.managed_settings()
        if managed_settings:
            context['managedSettings'] = '\n'.join(
                f'{key}={value}'
                for key, value in sorted(managed_settings.items())
            )
        safe_env_names = hook_policy_runtime.safe_env_names()
        if safe_env_names:  # names only: the values are passed to commands, never to the model
            context['safeEnv'] = '\n'.join(safe_env_names)
        context['trustMode'] = (
            'Workspace trust mode: trusted'
            if hook_policy_runtime.is_trusted()
            else 'Workspace trust mode: untrusted'
        )
    mcp_runtime = MCPRuntime.from_workspace(Path(cwd), additional_working_directories)
    if mcp_runtime.resources or mcp_runtime.servers:
        context['mcpRuntime'] = mcp_runtime.render_summary()
    remote_runtime = RemoteRuntime.from_workspace(Path(cwd), additional_working_directories)
    if remote_runtime.has_remote_config():
        context['remoteRuntime'] = remote_runtime.render_summary()
    remote_trigger_runtime = RemoteTriggerRuntime.from_workspace(
        Path(cwd),
        additional_working_directories,
    )
    if remote_trigger_runtime.has_state():
        context['remoteTriggerRuntime'] = remote_trigger_runtime.render_summary()
    search_runtime = SearchRuntime.from_workspace(Path(cwd), additional_working_directories)
    if search_runtime.has_search_runtime():
        context['searchRuntime'] = search_runtime.render_summary()
    account_runtime = AccountRuntime.from_workspace(Path(cwd), additional_working_directories)
    if account_runtime.has_account_state():
        context['accountRuntime'] = account_runtime.render_summary()
    ask_user_runtime = AskUserRuntime.from_workspace(Path(cwd), additional_working_directories)
    if ask_user_runtime.has_state():
        context['askUserRuntime'] = ask_user_runtime.render_summary()
    config_runtime = ConfigRuntime.from_workspace(Path(cwd))
    if config_runtime.has_config():
        context['configRuntime'] = config_runtime.render_summary()
    lsp_runtime = LSPRuntime.from_workspace(Path(cwd), additional_working_directories)
    if lsp_runtime.has_lsp_support():
        context['lspRuntime'] = lsp_runtime.render_summary()
    plan_runtime = PlanRuntime.from_workspace(Path(cwd))
    if plan_runtime.steps:
        context['planRuntime'] = plan_runtime.render_summary()
    task_runtime = TaskRuntime.from_workspace(Path(cwd))
    if task_runtime.tasks:
        context['taskRuntime'] = task_runtime.render_summary()
    team_runtime = TeamRuntime.from_workspace(Path(cwd), additional_working_directories)
    if team_runtime.has_team_state():
        context['teamRuntime'] = team_runtime.render_summary()
    workflow_runtime = WorkflowRuntime.from_workspace(Path(cwd), additional_working_directories)
    if workflow_runtime.has_workflows():
        context['workflowRuntime'] = workflow_runtime.render_summary()
    worktree_runtime = WorktreeRuntime.from_workspace(Path(cwd))
    if worktree_runtime.repo_root is not None or worktree_runtime.has_state():
        context['worktreeRuntime'] = worktree_runtime.render_summary()
    skills_runtime = SkillsRuntime.from_workspace(Path(cwd), additional_working_directories)
    if skills_runtime.has_skills_or_agents():
        context['skillsRuntime'] = skills_runtime.render_summary()
    return context


@lru_cache(maxsize=32)
def _get_git_status_cached(cwd: str) -> str | None:
    root = Path(cwd)
    if not _is_git_repo(root):
        return None

    branch = _run_command(['git', 'branch', '--show-current'], root)
    main_branch = _detect_default_branch(root)
    status = _run_command(['git', '--no-optional-locks', 'status', '--short'], root) or ''
    log = _run_command(['git', '--no-optional-locks', 'log', '--oneline', '-n', '5'], root) or '(none)'
    user_name = _run_command(['git', 'config', 'user.name'], root)

    if len(status) > MAX_STATUS_CHARS:
        status = (
            status[:MAX_STATUS_CHARS]
            + '\n... (truncated because it exceeds 2k characters. Use bash for full git status.)'
        )

    parts = [
        'This is the git status at the start of the conversation. It is a snapshot and does not update automatically during the run.',
        f'Current branch: {branch or "(unknown)"}',
        f'Main branch: {main_branch or "(unknown)"}',
    ]
    if user_name:
        parts.append(f'Git user: {user_name}')
    parts.extend(
        [
            f'Status:\n{status or "(clean)"}',
            f'Recent commits:\n{log}',
        ]
    )
    return '\n\n'.join(parts)


def _load_memory_bundle(cwd: Path, additional_working_directories: tuple[str, ...]) -> str | None:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for candidate in _discover_global_memory_files(cwd):
        _remember_path(candidate, discovered, seen)

    skip_agents = _sigil_md_at_or_above(cwd)
    for directory in _walk_upwards(cwd):
        for candidate in _discover_memory_files_for_directory(directory, skip_agents=skip_agents):
            _remember_path(candidate, discovered, seen)

    for raw_path in additional_working_directories:
        for candidate in _discover_memory_files_for_directory(Path(raw_path)):
            _remember_path(candidate, discovered, seen)
    discovered = _not_excluded(discovered, _memory_excludes(cwd))

    from .auto_memory import load_block

    auto_block = load_block(cwd)  # opt-in: SIGIL_AUTO_MEMORY=1 or autoMemoryEnabled
    if not discovered and not auto_block:
        return None

    blocks = [MEMORY_INSTRUCTION_PROMPT]
    for path in discovered:
        try:
            content = path.read_text(encoding='utf-8', errors='replace').strip()
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
        if not content:
            continue
        content = expand_memory_imports(content, path.parent, root=None if user_scope_file(path) else cwd)
        if len(content) > MAX_MEMORY_CHARACTER_COUNT:
            content = (
                content[:MAX_MEMORY_CHARACTER_COUNT]
                + '\n... (truncated because it exceeds the memory size limit)'
            )
        blocks.append(f'## {path}\n{content}')
    if auto_block:
        blocks.append(auto_block)
    if len(blocks) == 1:
        return None
    return '\n\n'.join(blocks)


_IMPORT_RE = re.compile(r'(?<![\w`@])@((?:~/|\.{1,2}/|/)?[\w.\-/\\]+\.[\w]+)')
MAX_IMPORT_DEPTH = 4


def user_scope_file(path: Path) -> bool:
    """A memory file the user wrote (~/.sigil, ~/.sigil): its imports load without a question."""
    try:
        resolved = Path(path).resolve()
        home = Path.home().resolve()
    except OSError as _ignored:
        _log_ignored(_ignored)
        return False
    return any(resolved.is_relative_to(home / name) for name in ('.sigil', '.sigil'))


def _resolve_import(raw: str, base_dir: Path) -> Path | None:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    try:
        return candidate.resolve()
    except OSError as _ignored:
        _log_ignored(_ignored)
        return None


def external_imports(cwd: Path, additional_working_directories: tuple[str, ...] = ()) -> list[tuple[Path, Path]]:
    """(memory file, imported file) for each `@path` in the folder's project memory files that
    resolves outside the folder (an external import needs the user's approval)."""
    root = Path(cwd).resolve()
    found: list[tuple[Path, Path]] = []
    for path in memory_file_paths(root, additional_working_directories):
        if user_scope_file(path):
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')[:MAX_MEMORY_CHARACTER_COUNT]
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
        for match in _IMPORT_RE.finditer(text):
            resolved = _resolve_import(match.group(1), path.parent)
            if resolved is not None and resolved.is_file() and not resolved.is_relative_to(root):
                found.append((path, resolved))
    return found


def _external_allowed(root: Path) -> bool:
    from .workspace_trust import is_trusted

    return is_trusted(root)


def expand_memory_imports(text: str, base_dir: Path, depth: int = 0, seen: set | None = None,
                          root: Path | None = None) -> str:
    """Inline `@path` imports in a memory file.

    Paths are relative to the file that imports them (`~/` = home); imports go
    up to 4 levels deep; code blocks and `code spans` are left alone; a file is
    imported once. With `root` (a project memory file), an import that resolves outside that
    folder loads only when the folder is trusted (workspace_trust lists it); otherwise it stays
    as plain text.
    """
    seen = set() if seen is None else seen
    if depth >= MAX_IMPORT_DEPTH or '@' not in text:
        return text
    out: list[str] = []
    in_fence = False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or '@' not in line:
            out.append(line)
            continue
        pieces = re.split(r'(`[^`]*`)', line)
        for index, piece in enumerate(pieces):
            if piece.startswith('`'):
                continue

            def _inline(match: 're.Match[str]') -> str:
                resolved = _resolve_import(match.group(1), base_dir)
                if resolved is None or not resolved.is_file() or resolved in seen:
                    return match.group(0)
                outside = root is not None and not resolved.is_relative_to(Path(root).resolve())
                if outside and not _external_allowed(Path(root).resolve()):
                    return match.group(0)  # external import of an untrusted project file: not loaded
                seen.add(resolved)
                try:
                    body = resolved.read_text(encoding='utf-8', errors='replace').strip()
                except OSError as _ignored:
                    _log_ignored(_ignored)
                    return match.group(0)
                body = expand_memory_imports(body, resolved.parent, depth + 1, seen, root)
                return f'{match.group(0)}\n<!-- imported from {resolved} -->\n{body}\n'

            pieces[index] = _IMPORT_RE.sub(_inline, piece)
        out.append(''.join(pieces))
    return '\n'.join(out)


def nested_memory_for(target: Path, cwd: Path, loaded: set) -> list[tuple[Path, str]]:
    """SIGIL.md / SIGIL.local.md in folders between cwd and target, not yet loaded."""
    try:
        target_dir = (target if target.is_dir() else target.parent).resolve()
        root = cwd.resolve()
        relative = target_dir.relative_to(root)
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return []
    found: list[tuple[Path, str]] = []
    folder = root
    skip_agents = _sigil_md_at_or_above(root)
    patterns = _memory_excludes(root)
    for part in relative.parts:
        folder = folder / part
        if folder in loaded:
            continue
        loaded.add(folder)
        names = ['SIGIL.md', 'SIGIL.local.md']
        fallback = None if skip_agents or _has_sigil_md(folder) else _agents_fallback(folder)
        if fallback is not None:
            names.insert(0, fallback.name)  # nested AGENTS.md: nearer folders come later and win
        for name in names:
            path = folder / name
            if path.is_file() and _not_excluded([path.resolve()], patterns):
                try:
                    body = path.read_text(encoding='utf-8', errors='replace').strip()
                except OSError as _ignored:
                    _log_ignored(_ignored)
                    continue
                if body:
                    found.append((path, expand_memory_imports(body, folder, root=root)[:MAX_MEMORY_CHARACTER_COUNT]))
    return found


def memory_file_paths(cwd: Path, additional_working_directories: tuple[str, ...] = ()) -> list[Path]:
    """The memory files a session loads (SIGIL.md, SIGIL.local.md, AGENTS.md, rules)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for candidate in _discover_global_memory_files(cwd):
        _remember_path(candidate, found, seen)
    skip_agents = _sigil_md_at_or_above(cwd)
    for directory in _walk_upwards(cwd):
        for candidate in _discover_memory_files_for_directory(directory, skip_agents=skip_agents):
            _remember_path(candidate, found, seen)
    for raw_path in additional_working_directories:
        for candidate in _discover_memory_files_for_directory(Path(raw_path)):
            _remember_path(candidate, found, seen)
    return _not_excluded(found, _memory_excludes(cwd))


def _discover_global_memory_files(cwd: Path | None = None) -> list[Path]:
    home = Path.home() / '.sigil'
    files = [home / 'SIGIL.md'] if (home / 'SIGIL.md').is_file() else []
    return files + _rule_files(home / 'rules', Path(cwd) if cwd is not None else Path.cwd())


# Read in place of SIGIL.md, first match per folder: AGENTS.override.md (as Codex CLI),
# AGENTS.md, then GEMINI.md.
AGENTS_FALLBACK_FILES = ('AGENTS.override.md', 'AGENTS.md', 'GEMINI.md')


def _has_sigil_md(directory: Path) -> bool:
    return any((directory / name).is_file() for name in ('SIGIL.md', 'SIGIL.local.md')) or \
        (directory / '.sigil' / 'SIGIL.md').is_file()


def _sigil_md_at_or_above(cwd: Path) -> bool:
    """Rule: any SIGIL.md / SIGIL.local.md in the cwd or above turns AGENTS.md off.

    ~/.sigil/SIGIL.md is the user's own memory, not a project file, so it does not count.
    """
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError) as _ignored:
        _log_ignored(_ignored)
        home = None
    for directory in _walk_upwards(cwd):
        if directory == home:
            if any((directory / name).is_file() for name in ('SIGIL.md', 'SIGIL.local.md')):
                return True
        elif _has_sigil_md(directory):
            return True
    return False


def _agents_fallback(directory: Path) -> Path | None:
    for name in AGENTS_FALLBACK_FILES:
        if (directory / name).is_file():
            return directory / name
    return None


def _memory_excludes(cwd: Path) -> list[str]:
    """sigilMdExcludes: glob patterns (matched against absolute paths) of memory files to skip."""
    try:
        from .settings import load_settings

        patterns = load_settings(cwd).get('sigilMdExcludes')
    except Exception as _ignored:
        _log_ignored(_ignored)
        patterns = None
    return [str(p) for p in patterns if isinstance(p, str) and p.strip()] if isinstance(patterns, list) else []


def _not_excluded(paths: list[Path], patterns: list[str]) -> list[Path]:
    if not patterns:
        return paths
    import fnmatch

    return [path for path in paths
            if not any(fnmatch.fnmatch(path.as_posix(), pattern.replace('\\', '/')) for pattern in patterns)]


def _discover_memory_files_for_directory(directory: Path, *, skip_agents: bool = False) -> list[Path]:
    files: list[Path] = []
    for candidate in (
        directory / 'SIGIL.md',
        directory / '.sigil' / 'SIGIL.md',
        directory / 'SIGIL.local.md',
    ):
        if candidate.is_file():
            files.append(candidate.resolve())

    # AGENTS.md fallback: only when no SIGIL.md / SIGIL.local.md exists here, in the cwd or above.
    agents_md = None if skip_agents or _has_sigil_md(directory) else _agents_fallback(directory)
    if agents_md is not None:
        files.insert(0, agents_md.resolve())

    files.extend(_rule_files(directory / '.sigil' / 'rules', directory))
    return files


def _rule_files(rules_dir: Path, workspace: Path) -> list[Path]:
    """Every *.md under a rules folder (subfolders too). A rule with `paths:` in its
    frontmatter loads only when the workspace has files matching one of the globs."""
    if not rules_dir.is_dir():
        return []
    picked: list[Path] = []
    for path in sorted(rules_dir.rglob('*.md')):
        if not path.is_file():
            continue
        try:
            from .custom_commands import parse_frontmatter, tool_list

            meta, _body = parse_frontmatter(path.read_text(encoding='utf-8', errors='replace'))
        except Exception as _ignored:
            _log_ignored(_ignored)
            meta = {}
        patterns = tool_list(meta.get('paths')) if meta.get('paths') else ()
        if patterns:
            try:
                if not any(next(workspace.glob(pattern), None) is not None for pattern in patterns):
                    continue
            except (OSError, ValueError, NotImplementedError) as _ignored:
                _log_ignored(_ignored)
                continue
        picked.append(path.resolve())
    return picked


def _walk_upwards(cwd: Path) -> list[Path]:
    parents = list(cwd.resolve().parents)
    parents.reverse()
    return [*parents, cwd.resolve()]


def _remember_path(path: Path, discovered: list[Path], seen: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    discovered.append(resolved)


def _detect_default_branch(cwd: Path) -> str | None:
    origin_head = _run_command(
        ['git', 'symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD'],
        cwd,
    )
    if origin_head and '/' in origin_head:
        return origin_head.split('/', 1)[1]

    for candidate in ('main', 'master'):
        try:
            completed = subprocess.run(
                ['git', 'show-ref', '--verify', f'refs/heads/{candidate}'],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        # Why: TimeoutExpired is a SubprocessError, not an OSError; a slow git (big repo,
        # antivirus) used to crash context building instead of being skipped.
        except (OSError, subprocess.SubprocessError) as _ignored:
            _log_ignored(_ignored)
            return None
        if completed.returncode == 0:
            return candidate
    return None


@lru_cache(maxsize=32)
def _is_git_repo(cwd: Path) -> bool:
    if (cwd / '.git').exists():
        return True
    try:
        completed = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    # Why: TimeoutExpired is a SubprocessError, not an OSError; a slow git (big repo,
    # antivirus) used to crash context building instead of being skipped.
    except (OSError, subprocess.SubprocessError) as _ignored:
        _log_ignored(_ignored)
        return False
    return completed.returncode == 0 and completed.stdout.strip() == 'true'


def _is_git_worktree(cwd: Path) -> bool:
    if not _is_git_repo(cwd):
        return False
    git_dir = _run_command(['git', 'rev-parse', '--git-dir'], cwd)
    git_common_dir = _run_command(['git', 'rev-parse', '--git-common-dir'], cwd)
    return bool(git_dir and git_common_dir and git_dir != git_common_dir)


def _run_command(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    # Why: TimeoutExpired is a SubprocessError, not an OSError; a slow git (big repo,
    # antivirus) used to crash context building instead of being skipped.
    except (OSError, subprocess.SubprocessError) as _ignored:
        _log_ignored(_ignored)
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _get_os_version() -> str:
    system = platform.system()
    release = platform.release()
    if system and release:
        return f'{system} {release}'
    return platform.platform()
