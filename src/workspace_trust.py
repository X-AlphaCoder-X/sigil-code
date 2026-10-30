"""Workspace trust: project files that run things need the user's yes first.

A folder's project items are what a cloned repository could make Sigil run or send without the
user typing it: hooks and a `statusLine` command in `.sigil/settings.json` /
`.sigil/settings.local.json`, MCP servers from project `.mcp.json`-style files (the folder and
its parents), environment values from the project's settings `env` block and `.env`, and the
settings that grant capability (allow rules, additionalDirectories, a non-default start mode,
sandbox, profiles, fastModel/editorModel, extraHeaders, test/lint commands, autoMemoryDirectory;
settings.local.json only when the repository supplies it), and Python custom tools in
`src/custom_tools/` (imported, so run, at agent start). `deny`/`ask` rules always apply.

The first time a folder has such items, the interactive chat lists them (commands and URLs;
env values by name only) and asks once: trust, or skip. The answer is kept per folder in
~/.sigil/trusted_folders.json, never in the repository. When those items change (their
fingerprint changes), Sigil asks again. Headless runs (-p, background workers, library use)
never prompt: untrusted items are skipped and named on stderr and in the log.

Items the user added themselves (`sigil mcp add --scope project`, `/statusline`) are recorded
as the user's own and always run. Items in ~/.sigil, ~/.sigil and `--mcp-config` /
`--plugin-dir` are the user's and are not checked here.

Off switch: SIGIL_TRUST_ALL=1, or "trustAllProjects": true in ~/.sigil/settings.json.
This is a permission prompt for the user; it adds nothing to the model's instructions.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_TRUTHY = {'1', 'true', 'yes', 'on'}
_REPORTED: set[str] = set()


@dataclass(frozen=True)
class TrustItem:
    kind: str  # 'hook' | 'statusLine' | 'mcp' | 'env' | 'setting' | 'command' | 'skill' | 'agent' | 'plugin' | 'file' | 'custom-tool'
    name: str
    detail: str  # command or URL (env: empty; the value is never shown)
    source: str
    secret: str = ''  # env value, only for the fingerprint

    @property
    def key(self) -> str:
        return f'{self.kind}\x00{self.name}\x00{self.detail}'

    def describe(self) -> str:
        # S16: name/detail come from repo files; strip terminal controls so an escape sequence in a
        # hook command or an MCP URL cannot reach the terminal when the trust question is shown
        from .terminal import strip_terminal_controls

        name = strip_terminal_controls(self.name)
        detail = strip_terminal_controls(self.detail)
        source = strip_terminal_controls(self.source)
        if self.kind == 'env':
            return f'env {name} (from {source})'
        if self.kind == 'custom-tool':
            return f'custom tools: {name} ({detail.split(" sha256:")[0]})'
        if self.kind == 'answers':
            return f'pre-set answers: {name} ({detail})'
        return f'{self.kind} {name}: {detail} (from {source})'


def _home() -> Path:
    from .settings import _home as settings_home

    return settings_home()


def state_path() -> Path:
    return _home() / '.sigil' / 'trusted_folders.json'


def trust_all() -> bool:
    if os.environ.get('SIGIL_TRUST_ALL', '').strip().lower() in _TRUTHY:
        return True
    from .settings import read_settings_file

    return read_settings_file(_home() / '.sigil' / 'settings.json').get('trustAllProjects') is True


def _folder(cwd: str | Path) -> str:
    try:
        return str(Path(cwd).resolve())
    except OSError as _ignored:
        _log_ignored(_ignored)
        return str(Path(cwd))


def _load_state() -> dict[str, Any]:
    from .settings import read_text_retry

    try:
        data = json.loads(read_text_retry(state_path()))
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(data: dict[str, Any]) -> None:
    from .settings import write_text_atomic

    write_text_atomic(state_path(), json.dumps(data, indent=2))
    _forget_status()  # trust changed: the next check reads it again


def _env_file_items(path: Path) -> list[TrustItem]:
    try:
        text = path.read_text(encoding='utf-8-sig')
    except OSError as _ignored:
        _log_ignored(_ignored)
        return []
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().removeprefix('export ').strip()
            if key:
                items.append(TrustItem('env', key, '', path.name, value.strip()))
    return items


def _hook_items(hooks: Any, source: str) -> list[TrustItem]:
    items: list[TrustItem] = []
    if not isinstance(hooks, dict):
        return items
    for event, groups in hooks.items():
        for group in groups if isinstance(groups, list) else []:
            entries = group.get('hooks') if isinstance(group, dict) else None
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                kind = str(entry.get('type') or 'command')
                detail = entry.get('command') or entry.get('url') or entry.get('prompt') or entry.get('tool') or ''
                if isinstance(entry.get('args'), list):
                    detail = ' '.join([str(detail), *(str(a) for a in entry['args'])])
                items.append(TrustItem('hook', f'{event} ({kind})', str(detail)[:300], source))
    return items


def _mcp_manifest_is_user_level(path: Path) -> bool:
    from .mcp_runtime import user_mcp_config_path

    try:
        return path.resolve() == user_mcp_config_path().resolve() or path.resolve().parent == _home().resolve()
    except OSError as _ignored:
        _log_ignored(_ignored)
        return False


def _grant_file_items(root: Path, limit: int = 500) -> list[TrustItem]:
    """Project commands and skills whose `allowed-tools` would pre-approve tools (and !`cmd`)."""
    from .custom_commands import parse_frontmatter, tool_list

    items: list[TrustItem] = []
    scanned = 0
    for folder, pattern, kind in ((root / '.sigil' / 'commands', '*.md', 'command'),
                                  (root / '.sigil' / 'skills', 'SKILL.md', 'skill'),
                                  (root / '.agents' / 'skills', 'SKILL.md', 'skill')):
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob(pattern)):
            scanned += 1
            if scanned > limit:  # bounded walk; the rest are still held back by custom_commands
                return items
            try:
                meta, _body = parse_frontmatter(path.read_text(encoding='utf-8', errors='replace')[:20000])
            except OSError as _ignored:
                _log_ignored(_ignored)
                continue
            tools = tool_list(meta.get('allowed-tools'))
            if tools:
                name = path.stem if kind == 'command' else path.parent.name
                items.append(TrustItem(kind, name, 'allowed-tools: ' + ', '.join(tools)[:300],
                                       path.relative_to(root).as_posix()))
    return items


def _search_items(root: Path) -> list[TrustItem]:
    """Repo search providers (.sigil-search.json, .sigil/search.json) that would get the user's
    search key on a host other than the provider's own."""
    from urllib.parse import urlsplit

    from .search_runtime import _default_base_url, _discover_manifest_paths, _load_profiles_from_manifest

    items: list[TrustItem] = []
    for path in _discover_manifest_paths(root, ()):
        for provider in _load_profiles_from_manifest(path):
            default = _default_base_url(provider.provider)
            if provider.api_key_env and not (default and urlsplit(default).hostname ==
                                             urlsplit(provider.base_url).hostname):
                items.append(TrustItem('search', provider.name, f'{provider.base_url} (key {provider.api_key_env})',
                                       path.name))
    return items


def _policy_secret_items(root: Path) -> list[TrustItem]:
    """Credential-looking names a repo policy file (.sigil-policy.json safeEnv) asks to pass to commands."""
    from .hook_policy import _discover_policy_paths, _load_policy_manifest, secret_like

    items: list[TrustItem] = []
    home = _home().resolve()
    for path in _discover_policy_paths(root, ()):
        if path.parent == home:
            continue  # the user's own policy file
        manifest = _load_policy_manifest(path)
        for name in (manifest.safe_env_names if manifest is not None else ()):
            if secret_like(name):
                items.append(TrustItem('env', name, 'safeEnv', path.name))
    return items


def custom_tool_detail(cwd: str | Path, path: str | Path) -> str:
    """'src/custom_tools/x.py sha256:<16 hex>': a changed file is a new item (asked again)."""
    root = Path(_folder(cwd))
    try:
        rel = Path(path).resolve().relative_to(root).as_posix()
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        rel = str(path)
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError as _ignored:
        _log_ignored(_ignored)
        digest = '-'
    return f'{rel} sha256:{digest}'


def _custom_tool_items(root: Path, limit: int = 200) -> list[TrustItem]:
    """Python files in the folder's src/custom_tools: Sigil imports (runs) them at agent start."""
    from .agent_tools import custom_tool_files, project_custom_tools_dir

    return [TrustItem('custom-tool', path.stem, custom_tool_detail(root, path), 'src/custom_tools')
            for path in custom_tool_files(project_custom_tools_dir(root))[:limit]]


def _ask_user_items(root: Path) -> list[TrustItem]:
    """N23: queued answers files answer the model's questions in the user's place."""
    items: list[TrustItem] = []
    for path in (root / '.sigil-ask-user.json', root / '.sigil' / 'ask-user.json'):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            answers = data.get('answers') if isinstance(data, dict) else None
            count = len(answers) if isinstance(answers, list) else 0
            content = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            count, content = 0, ''
        rel = path.relative_to(root).as_posix()
        items.append(TrustItem('answers', rel, f'{count} answers', rel, content))
    return items


def _agent_hook_items(root: Path) -> list[TrustItem]:
    """Hooks in the frontmatter of the repository's .sigil/agents/*.md (they run commands), and a
    start mode that grants more than the safe ones (`--agent NAME` applies it: S15)."""
    from .permission_modes import normalize_mode
    from .subagents import _load_agent_dir

    items: list[TrustItem] = []
    for name, definition in _load_agent_dir(root / '.sigil' / 'agents', 'project').items():
        for item in _hook_items(definition.hooks, f'agent {name}'):
            items.append(item)
        mode = normalize_mode(definition.permission_mode) if definition.permission_mode else None
        if mode == 'acceptEdits':  # bypass is never taken from a project file; the others only restrict
            items.append(TrustItem('agent', name, f'permissionMode {definition.permission_mode}', '.sigil/agents'))
    return items


def project_items(cwd: str | Path) -> list[TrustItem]:
    """Everything in the folder's project files that Sigil would run, connect to or put in env."""
    from .settings import (LOCAL_SETTINGS, PROJECT_SETTINGS, capability_entries, local_settings_from_repo,
                           read_settings_file)

    root = Path(_folder(cwd))
    items: list[TrustItem] = []
    for rel in (PROJECT_SETTINGS, LOCAL_SETTINGS):
        path = root / rel
        data = read_settings_file(path)
        source = rel.as_posix()
        if rel == PROJECT_SETTINGS or local_settings_from_repo(root):
            # allow rules, extra folders, start mode, sandbox, profiles, other models/hosts, commands
            items.extend(TrustItem('setting', kind, detail, source) for kind, detail in capability_entries(data))
        items.extend(_hook_items(data.get('hooks'), source))
        status = data.get('statusLine')
        command = status.get('command') if isinstance(status, dict) else status
        if isinstance(command, str) and command.strip():
            items.append(TrustItem('statusLine', 'statusLine', command.strip(), source))
        env = data.get('env')
        if isinstance(env, dict):
            items.extend(TrustItem('env', str(k), '', source, str(v)) for k, v in env.items())
    items.extend(_env_file_items(root / '.env'))
    items.extend(_grant_file_items(root))
    items.extend(_search_items(root))
    items.extend(_policy_secret_items(root))
    items.extend(_agent_hook_items(root))
    items.extend(_custom_tool_items(root))
    items.extend(_ask_user_items(root))
    try:
        from .plugin_runtime import _discover_plugin_manifest_paths

        # plugin manifests in the folder and the folders above it (hook text, tool aliases)
        items.extend(TrustItem('plugin', path.parent.name, str(path), 'plugin.json')
                     for path in _discover_plugin_manifest_paths(root, ()))
    except OSError as exc:
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
    try:
        from .agent_context import external_imports

        # SIGIL.md / rules `@path` imports that reach outside the folder (e.g. @~/.ssh/id_rsa)
        items.extend(TrustItem('file', '@import', str(target), source.name) for source, target in external_imports(root))
    except OSError as exc:
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
    try:
        from .mcp_runtime import _discover_manifest_paths, _load_manifest

        for path in _discover_manifest_paths(root, ()):
            if _mcp_manifest_is_user_level(path):
                continue
            for server in _load_manifest(path)[1]:
                detail = server.url or ' '.join([str(server.command or ''), *server.args]).strip()
                items.append(TrustItem('mcp', server.name, detail, str(path)))
    except Exception:  # a broken manifest is reported by the MCP loader itself
        from .debug_log import log_ignored_error

        log_ignored_error(sys.exc_info()[1])  # type: ignore[arg-type]
    user_items = set(_load_state().get(_folder(cwd), {}).get('user_items', []))
    return [item for item in items if item.key not in user_items]


def fingerprint(items: list[TrustItem]) -> str:
    rows = sorted([item.kind, item.name, item.detail, item.source,
                   hashlib.sha256(item.secret.encode('utf-8')).hexdigest()] for item in items)
    return hashlib.sha256(json.dumps(rows).encode('utf-8')).hexdigest()


_STATUS_TTL = 2.0  # seconds: settings are read many times a turn; the folder scan runs at most this often
_STATUS_CACHE: dict[str, tuple[float, tuple[str, list[TrustItem]]]] = {}
_STATUS_LOCK = threading.Lock()


def _forget_status() -> None:
    with _STATUS_LOCK:
        _STATUS_CACHE.clear()


def status(cwd: str | Path) -> tuple[str, list[TrustItem]]:
    """('trust-all' | 'nothing' | 'trusted' | 'skipped' | 'changed' | 'new', items)."""
    if trust_all():
        return 'trust-all', []
    key = _folder(cwd) + '|' + _signature(Path(_folder(cwd)))
    now = time.monotonic()
    with _STATUS_LOCK:
        cached = _STATUS_CACHE.get(key)
    if cached is not None and now - cached[0] < _STATUS_TTL:
        return cached[1][0], list(cached[1][1])
    result = _status_uncached(cwd)
    with _STATUS_LOCK:
        if len(_STATUS_CACHE) > 64:
            _STATUS_CACHE.clear()  # bounded
        _STATUS_CACHE[key] = (now, result)
    return result


def _signature(root: Path) -> str:
    """mtime and size of the files the trust answer depends on most (cheap stat calls), so an
    edit made a moment ago is seen at once and not served from the cache."""
    parts = []
    for path in (state_path(), root / '.sigil' / 'settings.json', root / '.sigil' / 'settings.local.json',
                 root / '.env', root / '.mcp.json', root / 'SIGIL.md', root / 'SIGIL.local.md',
                 root / '.sigil-policy.json', root / '.sigil-search.json', root / '.sigil' / 'search.json',
                 root / '.sigil' / 'commands', root / '.sigil' / 'skills', root / 'src' / 'custom_tools',
                 root / '.sigil-ask-user.json', root / '.sigil' / 'ask-user.json'):
        try:
            info = path.stat()
            parts.append(f'{info.st_mtime_ns}:{info.st_size}')
        except OSError:
            parts.append('-')
    return '|'.join(parts)


def _status_uncached(cwd: str | Path) -> tuple[str, list[TrustItem]]:
    items = project_items(cwd)
    if not items:
        return 'nothing', items
    record = _load_state().get(_folder(cwd))
    if not isinstance(record, dict) or 'trusted' not in record:
        return 'new', items
    if record.get('hash') != fingerprint(items):
        return 'changed', items
    return ('trusted' if record.get('trusted') else 'skipped'), items


def is_trusted(cwd: str | Path) -> bool:
    return status(cwd)[0] in ('trust-all', 'nothing', 'trusted')


def is_user_item(cwd: str | Path, kind: str, name: str, detail: str) -> bool:
    key = TrustItem(kind, name, detail, '').key
    return key in set(_load_state().get(_folder(cwd), {}).get('user_items', []))


def allowed(cwd: str | Path, kind: str, name: str, detail: str) -> bool:
    """May this project item run? Yes when the folder is trusted or the user added it."""
    return is_trusted(cwd) or is_user_item(cwd, kind, name, detail)


def record(cwd: str | Path, trusted: bool) -> None:
    data = _load_state()
    entry = data.get(_folder(cwd)) if isinstance(data.get(_folder(cwd)), dict) else {}
    entry.update({'trusted': bool(trusted), 'hash': fingerprint(project_items(cwd)),
                  'at': datetime.now(timezone.utc).isoformat()})
    data[_folder(cwd)] = entry
    _save_state(data)


def revoke(cwd: str | Path) -> None:
    data = _load_state()
    entry = data.get(_folder(cwd))
    if isinstance(entry, dict):
        entry.pop('trusted', None)
        entry.pop('hash', None)
        _save_state(data)


def mark_user_item(cwd: str | Path, kind: str, name: str, detail: str) -> None:
    """The user added this project item themselves (sigil mcp add, /statusline): it always runs."""
    data = _load_state()
    entry = data.get(_folder(cwd)) if isinstance(data.get(_folder(cwd)), dict) else {}
    keys = set(entry.get('user_items', []))
    keys.add(TrustItem(kind, name, detail, '').key)
    entry['user_items'] = sorted(keys)
    data[_folder(cwd)] = entry
    _save_state(data)


def describe(items: list[TrustItem]) -> str:
    return '\n'.join(f'  - {item.describe()}' for item in items)


def report_skipped(cwd: str | Path, items: list[TrustItem] | None = None) -> None:
    """Name the skipped project items on stderr and in the log (once per folder and content)."""
    state, found = status(cwd)
    items = items if items is not None else found
    if state in ('trust-all', 'nothing', 'trusted') or not items:
        return
    key = f'{_folder(cwd)}:{fingerprint(items)}'
    if key in _REPORTED:
        return
    _REPORTED.add(key)
    reason = 'skipped by the user' if state == 'skipped' else 'not trusted yet'
    message = (f'Sigil: project items in {_folder(cwd)} were not used ({reason}):\n{describe(items)}\n'
               'Trust them with `sigil trust grant` (or /trust grant in a chat), '
               'or set SIGIL_TRUST_ALL=1.')
    print(message, file=sys.stderr)
    from .debug_log import get_logger, sanitize_log

    get_logger('trust').warning('untrusted project items skipped in %s: %s', _folder(cwd),
                                sanitize_log('; '.join(item.describe() for item in items), 2000))


def ensure_trust(cwd: str | Path, ask: Callable[[str], str] | None = None,
                 output: Callable[[str], None] = print) -> bool:
    """Ask once (interactive) or report (headless). Returns whether project items may run."""
    state, items = status(cwd)
    if state in ('trust-all', 'nothing', 'trusted'):
        return True
    if state == 'skipped' or ask is None:
        report_skipped(cwd, items)
        return False
    what = 'changed since you trusted it' if state == 'changed' else 'has project settings that run things'
    output(f'This folder {what}:\n{describe(items)}\n'
           'Hooks and MCP servers run commands on this machine; custom tools run Python code; env values '
           'are set for the session; settings grant permissions, folders, models or hosts.')
    try:
        answer = str(ask('Trust this folder? [t]rust / [s]kip: ') or '').strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ''
    trusted = answer in ('t', 'trust', 'y', 'yes')
    record(cwd, trusted)
    if not trusted:
        output('Skipped: these project items stay off. Change it with /trust grant or `sigil trust grant`.')
    return trusted


def render(cwd: str | Path) -> str:
    state, items = status(cwd)
    lines = [f'Trust for {_folder(cwd)}: {state}']
    if state == 'trust-all':
        lines.append('SIGIL_TRUST_ALL=1 or trustAllProjects: every project item runs without asking.')
    elif items:
        lines.append(describe(items))
    else:
        lines.append('No project hooks, MCP servers, status line or env here.')
    return '\n'.join(lines)


def trust_command(cwd: str | Path, action: str) -> str:
    """`/trust` and `sigil trust`: status (default), grant, skip, revoke."""
    action = (action or 'status').strip().lower()
    if action in ('grant', 'yes', 'trust'):
        record(cwd, True)
        return render(cwd) + '\nTrusted. Start a new session (or /cd here again) to load them.'
    if action in ('skip', 'deny', 'no'):
        record(cwd, False)
        return render(cwd) + '\nSkipped.'
    if action in ('revoke', 'forget', 'reset'):
        revoke(cwd)
        return render(cwd) + '\nForgotten: Sigil asks again next time.'
    return render(cwd)
