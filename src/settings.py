"""Settings, merged from every settings file.

Order (later wins): ``~/.sigil/settings.json``, ``.sigil/settings.json``,
``.sigil/settings.local.json``, then ``--settings`` (a file path or a JSON
string). Lists are merged (items added once, in order); objects merge key by
key; any other value is replaced.

``SIGIL_DISABLE_USER_SETTINGS=1`` skips the user file (handy in CI).
"""
from __future__ import annotations

import json
import os
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

# The project folder of the agent running in this thread (set by LocalCodingAgent), for helpers
# that have no cwd argument: extraHeaders, fastModel, ${SIGIL_PROJECT_DIR}. Each thread (a
# subagent in a worktree, say) has its own value; without one the process folder is used.
_PROJECT_DIR: ContextVar[str | None] = ContextVar('sigil_project_dir', default=None)
_WARNED_BROKEN: set[str] = set()


def write_text_atomic(path: str | Path, text: str, encoding: str = 'utf-8') -> None:
    """Replace a file in one step: write a temp file in the same folder, then os.replace it.

    Why: state files (background records and results, settings, OAuth tokens, rate-limit and
    marketplace state, session names) are read by other Sigil processes; write_text truncates
    first, so a reader could see an empty or half-written file (JSONDecodeError). On Windows
    os.replace fails while a reader holds the old file open, so it is retried briefly.
    """
    import tempfile
    import time as _time

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=str(target.parent))
    try:
        with os.fdopen(fd, 'w', encoding=encoding) as handle:
            handle.write(text)
        for attempt in range(40):
            try:
                os.replace(tmp_name, target)
                return
            except PermissionError:
                if attempt == 39:
                    raise
                _time.sleep(0.005 * (attempt + 1))
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass  # removed by os.replace, or not ours to remove


def read_text_retry(path: str | Path, encoding: str = 'utf-8') -> str:
    """read_text for files another process replaces with write_text_atomic: on Windows a file
    that is being replaced refuses to open for a moment (PermissionError), so retry briefly."""
    import time as _time

    for attempt in range(40):
        try:
            return Path(path).read_text(encoding=encoding)
        except PermissionError:
            if attempt == 39:
                raise
            _time.sleep(0.005 * (attempt + 1))
    raise PermissionError(str(path))


def set_project_dir(path: str | Path) -> Any:
    """Set this thread's project folder; returns the token for _PROJECT_DIR.reset."""
    try:
        return _PROJECT_DIR.set(str(Path(path).resolve()))
    except OSError as _ignored:
        _log_ignored(_ignored)
        return _PROJECT_DIR.set(str(path))


def update_project_dir(path: str | Path) -> None:
    """Move the project folder of a running agent (/cd); outside an agent scope it does nothing."""
    if _PROJECT_DIR.get() is not None:
        set_project_dir(path)


class project_dir_scope:  # noqa: N801 - used as `with project_dir_scope(cwd):`
    """The agent's folder for the code inside the block; the previous value comes back after."""

    def __init__(self, path: str | Path) -> None:
        self.path = path
        self.token: Any = None

    def __enter__(self) -> 'project_dir_scope':
        self.token = set_project_dir(self.path)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.token is not None:
            _PROJECT_DIR.reset(self.token)


def project_dir() -> Path:
    value = _PROJECT_DIR.get()
    return Path(value) if value else Path.cwd()

USER_SETTINGS = Path('.sigil') / 'settings.json'
PROJECT_SETTINGS = Path('.sigil') / 'settings.json'
LOCAL_SETTINGS = Path('.sigil') / 'settings.local.json'


def _home() -> Path:
    return Path.home()


def setting_sources() -> set[str]:
    """--setting-sources (SIGIL_SETTING_SOURCES): which of user, project, local are read."""
    raw = os.environ.get('SIGIL_SETTING_SOURCES', '')
    chosen = {part.strip().lower() for part in raw.split(',') if part.strip()}
    return chosen or {'user', 'project', 'local'}


def extra_settings() -> dict[str, Any]:
    """--settings (SIGIL_SETTINGS): a JSON string or a file path, applied last."""
    return _parse_extra(os.environ.get('SIGIL_SETTINGS') or None)


def settings_paths(cwd: Path, *, include_user: bool = True) -> list[tuple[str, Path]]:
    """The settings files that apply to a workspace, lowest priority first."""
    root = Path(cwd).resolve()
    allowed = setting_sources()
    paths: list[tuple[str, Path]] = []
    if include_user and not os.environ.get('SIGIL_DISABLE_USER_SETTINGS') and 'user' in allowed:
        paths.append(('user', _home() / USER_SETTINGS))
    if 'project' in allowed:
        paths.append(('project', root / PROJECT_SETTINGS))
    if 'local' in allowed:
        paths.append(('local', root / LOCAL_SETTINGS))
    return paths


def merge_settings(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_settings(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            merged[key] = current + [item for item in value if item not in current]
        else:
            merged[key] = value
    return merged


def read_settings_file(path: Path) -> dict[str, Any]:
    """A settings file as a dict; missing or broken files give {}.

    A file that exists but is not a JSON object is reported once on stderr (and in the log):
    its permissions, hooks and env are not applied until it is fixed.
    """
    try:
        text = read_text_retry(path)  # a rule save in another process replaces the file
    except OSError as _ignored:
        _log_ignored(_ignored)
        return {}
    try:
        payload = json.loads(text)
        problem = None if isinstance(payload, dict) else f'it is a JSON {type(payload).__name__}, not an object'
    except ValueError as exc:
        payload, problem = None, str(exc)
    if problem is None:
        return payload
    _warn_broken(Path(path), problem)
    return {}


def _warn_broken(path: Path, problem: str) -> None:
    key = str(path)
    if key in _WARNED_BROKEN:
        return
    if len(_WARNED_BROKEN) < 256:  # bounded: one notice per file, not a growing set
        _WARNED_BROKEN.add(key)
    from .debug_log import get_logger, sanitize_log

    message = (f'Settings file {sanitize_log(path, 300)} is ignored: {sanitize_log(problem)}. '
               'Its permissions, hooks and env do not apply.')
    get_logger('settings').warning(message)
    print(message, file=sys.stderr)


def _parse_extra(extra: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if extra is None:
        return {}
    if isinstance(extra, dict):
        return extra
    text = str(extra).strip()
    if text.startswith('{'):
        try:
            payload = json.loads(text)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            return {}
        return payload if isinstance(payload, dict) else {}
    return read_settings_file(Path(text))


# Keys of a repository's own settings file that give Sigil more power (run commands, write or read
# more, send text or keys elsewhere). They apply only once the user trusts the folder
# (workspace_trust); `deny`/`ask` rules and other restricting keys always apply.
CAPABILITY_KEYS = ('sandbox', 'profiles', 'fastModel', 'editorModel', 'extraHeaders', 'testCommand',
                   'lintCommand', 'autoTest', 'autoMemoryDirectory', 'cleanupPeriodDays')
_SAFE_START_MODES = {'default', 'plan', 'dontAsk'}
_WARNED_MODES: set[str] = set()


def _start_mode(value: Any) -> str | None:
    from .permission_modes import normalize_mode

    return normalize_mode(str(value)) if value else None


def capability_entries(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """(setting, detail) for each capability-granting entry of a project settings payload."""
    entries: list[tuple[str, str]] = []
    permissions = payload.get('permissions')
    if isinstance(permissions, dict):
        for rule in permissions.get('allow') if isinstance(permissions.get('allow'), list) else []:
            entries.append(('permissions.allow', str(rule)))
        extra = permissions.get('additionalDirectories')
        for folder in extra if isinstance(extra, list) else []:
            entries.append(('permissions.additionalDirectories', str(folder)))
        mode = permissions.get('defaultMode')
        if mode and _start_mode(mode) not in _SAFE_START_MODES:
            entries.append(('permissions.defaultMode', str(mode)))
    for key in CAPABILITY_KEYS:
        if key in payload and payload[key] not in (None, False, '', {}, []):
            entries.append((key, json.dumps(payload[key], sort_keys=True, default=str)[:300]))
    return entries


def _drop_entries(payload: dict[str, Any], keep: Any) -> dict[str, Any]:
    """payload without the capability entries for which keep(setting, detail) is false."""
    out = dict(payload)
    permissions = out.get('permissions')
    if isinstance(permissions, dict):
        permissions = dict(permissions)
        if isinstance(permissions.get('allow'), list):
            permissions['allow'] = [r for r in permissions['allow'] if keep('permissions.allow', str(r))]
        if isinstance(permissions.get('additionalDirectories'), list):
            permissions['additionalDirectories'] = [
                d for d in permissions['additionalDirectories'] if keep('permissions.additionalDirectories', str(d))]
        mode = permissions.get('defaultMode')
        if mode and _start_mode(mode) not in _SAFE_START_MODES and not keep('permissions.defaultMode', str(mode)):
            permissions.pop('defaultMode', None)
        out['permissions'] = permissions
    for key in CAPABILITY_KEYS:
        if key in out and not keep(key, json.dumps(out[key], sort_keys=True, default=str)[:300]):
            out.pop(key)
    return out


def _never_bypass_from_project(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """A repository settings file never starts Sigil in bypass mode; --permission-mode or the
    user's ~/.sigil/settings.json still can."""
    permissions = payload.get('permissions')
    if not isinstance(permissions, dict) or _start_mode(permissions.get('defaultMode')) != 'bypass':
        return payload
    if str(path) not in _WARNED_MODES and len(_WARNED_MODES) < 256:
        _WARNED_MODES.add(str(path))
        from .debug_log import get_logger, sanitize_log

        get_logger('settings').warning('defaultMode %s in %s is ignored: a project file cannot start bypass mode '
                                       '(use --permission-mode or ~/.sigil/settings.json)',
                                       sanitize_log(permissions.get('defaultMode'), 40), sanitize_log(path, 300))
    payload = dict(payload)
    payload['permissions'] = {k: v for k, v in permissions.items() if k != 'defaultMode'}
    return payload


def local_settings_from_repo(root: Path) -> bool:
    """Is .sigil/settings.local.json supplied by the repository (tracked in git, or .sigil is a
    link) rather than the user's own file? Checked by reading .git/index, never by running git: a
    repository's git config can make git start programs."""
    folder = root / '.sigil'
    try:
        if folder.is_symlink() or (root / LOCAL_SETTINGS).is_symlink():
            return True
    except OSError as _ignored:
        _log_ignored(_ignored)
        return True
    for folder in (root, *root.parents):
        dot_git = folder / '.git'
        if not dot_git.exists():
            continue
        git_dir = dot_git
        if dot_git.is_file():  # worktree / submodule: "gitdir: <path>"
            try:
                line = dot_git.read_text(encoding='utf-8', errors='replace').strip()
            except OSError as _ignored:
                _log_ignored(_ignored)
                return True
            git_dir = Path(line.split(':', 1)[1].strip()) if line.startswith('gitdir:') else dot_git
            if not git_dir.is_absolute():
                git_dir = folder / git_dir
        try:
            relative = (root / LOCAL_SETTINGS).relative_to(folder).as_posix().encode('utf-8')
            with open(git_dir / 'index', 'rb') as handle:
                data = handle.read(64 * 1024 * 1024)
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            return False  # no index yet: nothing is tracked
        if data[4:8] == b'\x00\x00\x00\x04':  # index v4 compresses paths: match the file name
            return b'settings.local.json' in data
        return relative + b'\x00' in data
    return False


_IN_TRUST_CHECK: ContextVar[bool] = ContextVar('sigil_in_trust_check', default=False)


def _trust_filter(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    entries = capability_entries(payload)
    if not entries:
        return payload
    if _IN_TRUST_CHECK.get():
        # a settings read made by the trust check itself (it lists the folder's items, which reads
        # settings again): grant nothing and do not start another check (no recursion)
        return _drop_entries(payload, lambda kind, detail: False)
    from . import workspace_trust

    token = _IN_TRUST_CHECK.set(True)
    try:
        if workspace_trust.is_trusted(root):
            return payload
        filtered = _drop_entries(payload,
                                 lambda kind, detail: workspace_trust.is_user_item(root, 'setting', kind, detail))
        workspace_trust.report_skipped(root)
    finally:
        _IN_TRUST_CHECK.reset(token)
    return filtered


def load_settings(
    cwd: Path,
    *,
    extra: str | Path | dict[str, Any] | None = None,
    include_user: bool = True,
) -> dict[str, Any]:
    """The merged settings. A project's `.sigil/settings.json` (and a settings.local.json that
    the repository supplies) contribute capability-granting keys (CAPABILITY_KEYS, allow rules,
    additionalDirectories, a non-default start mode) only when the folder is trusted."""
    merged: dict[str, Any] = {}
    seen: set[Path] = set()
    root = Path(cwd).resolve()
    for label, path in settings_paths(cwd, include_user=include_user):
        try:
            resolved = path.resolve()
        except OSError as _ignored:
            _log_ignored(_ignored)
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        payload = read_settings_file(resolved)
        if label in ('project', 'local') and payload:
            payload = _never_bypass_from_project(payload, resolved)
            if label == 'project' or local_settings_from_repo(root):
                payload = _trust_filter(root, payload)
        merged = merge_settings(merged, payload)
    merged = merge_settings(merged, extra_settings())
    return merge_settings(merged, _parse_extra(extra))


def remove_local_permission_rule(cwd: Path, rule: str) -> list[str]:
    """Remove a rule from every list in .sigil/settings.local.json; returns the lists it was in."""
    path = Path(cwd).resolve() / LOCAL_SETTINGS
    payload = read_settings_file(path)
    permissions = payload.get('permissions')
    removed: list[str] = []
    if isinstance(permissions, dict):
        for kind in ('allow', 'ask', 'deny'):
            rules = permissions.get(kind)
            if isinstance(rules, list) and rule in rules:
                permissions[kind] = [item for item in rules if item != rule]
                removed.append(kind)
    if removed:
        write_text_atomic(path, json.dumps(payload, indent=2) + '\n')  # other processes read it
    return removed


def add_local_permission_rule(cwd: Path, kind: str, rule: str) -> Path:
    """Append a rule to permissions.<kind> in .sigil/settings.local.json."""
    if kind not in ('allow', 'ask', 'deny'):
        raise ValueError(f'unknown permission list: {kind}')
    path = Path(cwd).resolve() / LOCAL_SETTINGS
    payload = read_settings_file(path)
    permissions = payload.get('permissions')
    if not isinstance(permissions, dict):
        permissions = {}
    rules = permissions.get(kind)
    if not isinstance(rules, list):
        rules = []
    if rule not in rules:
        rules.append(rule)
    permissions[kind] = rules
    payload['permissions'] = permissions
    write_text_atomic(path, json.dumps(payload, indent=2) + '\n')  # other processes read it
    if kind == 'allow':  # the user's own answer: it applies even where the folder is not trusted
        try:
            from .workspace_trust import mark_user_item

            mark_user_item(cwd, 'setting', 'permissions.allow', rule)
        except OSError as exc:
            from .debug_log import log_ignored_error

            log_ignored_error(exc)
    return path
