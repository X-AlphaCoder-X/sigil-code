"""One place that decides whether a path may be read or written.

By default the agent works in the workspace (``--cwd``) and the ``--add-dir``
folders. The other-directories switch widens that:

    SIGIL_OTHER_DIRECTORIES=yes          read anywhere; write anywhere writes are allowed
    OTHER_DIRECTORIES=yes                same (plain name, e.g. in .env)
    SIGIL_ALLOWED_DIRECTORIES=D:/data;E:/out
                                         only these extra folders (os.pathsep-separated:
                                         ';' on Windows, ':' elsewhere)

Writes still need write permission (--allow-write, acceptEdits; plan mode
blocks them). Outside the workspace and added folders, a protected list is
write-protected by default, including Sigil's own key, trust, hook and agent files in ~/.sigil.
A ``permissions.protectedPaths`` list in settings.json adds folders to that list (a repository
file cannot empty it). ``SIGIL_PROTECTED_PATHS`` (an os.pathsep-separated list) replaces the
whole list; set it to an empty value to protect nothing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

TRUTHY = frozenset({'1', 'true', 'yes', 'on'})

DEFAULT_PROTECTED_PATHS = (
    '~/.ssh',
    '~/.aws',
    '~/.gnupg',
    'C:/Windows',
    'C:/Program Files',
    'C:/Program Files (x86)',
    '/etc',
    '/usr',
    '/bin',
    '/sbin',
    '/boot',
    '/System',
    # Sigil's own files: keys, folder trust, OAuth tokens, user settings (hooks), profiles,
    # user agents / commands / skills / rules / memory
    '~/.sigil/.env',
    '~/.sigil/settings.json',
    '~/.sigil/trusted_folders.json',
    '~/.sigil/mcp-oauth.json',
    '~/.sigil/mcp.json',
    '~/.sigil/profiles.json',
    '~/.sigil/agents',
    '~/.sigil/commands',
    '~/.sigil/skills',
    '~/.sigil/rules',
    '~/.sigil/SIGIL.md',
)


class PathAccessError(Exception):
    """A path is outside what the agent may read or write."""


def _split_paths(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(os.pathsep) if part.strip()]


def _norm(path: Path | str) -> Path:
    expanded = Path(str(path)).expanduser()
    try:
        return expanded.resolve()
    except OSError as _ignored:
        _log_ignored(_ignored)
        return expanded.absolute()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError as _ignored:
        _log_ignored(_ignored)
        pass
    if os.name == 'nt':  # case-insensitive file system
        p, r = str(path).lower(), str(root).lower().rstrip('\\/')
        return p == r or p.startswith(r + '\\') or p.startswith(r + '/')
    return False


@dataclass(frozen=True)
class PathAccess:
    workspace: Path
    added: tuple[Path, ...] = ()
    other_all: bool = False
    allowed: tuple[Path, ...] = ()
    protected: tuple[Path, ...] = ()

    @classmethod
    def from_env(
        cls,
        workspace: Path,
        added: tuple[Path, ...] | list[Path] = (),
        *,
        settings: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> 'PathAccess':
        env = os.environ if env is None else env
        switch = (env.get('SIGIL_OTHER_DIRECTORIES') or env.get('OTHER_DIRECTORIES') or '')
        other_all = switch.strip().lower() in TRUTHY
        allowed = tuple(_norm(p) for p in _split_paths(env.get('SIGIL_ALLOWED_DIRECTORIES', '')))
        if 'SIGIL_PROTECTED_PATHS' in env:
            protected_raw = _split_paths(env.get('SIGIL_PROTECTED_PATHS', ''))
        else:
            permissions = (settings or {}).get('permissions') if settings else None
            configured = permissions.get('protectedPaths') if isinstance(permissions, dict) else None
            # settings add to the defaults; only SIGIL_PROTECTED_PATHS (the user's env) replaces them
            protected_raw = list(DEFAULT_PROTECTED_PATHS) + (
                [str(p) for p in configured] if isinstance(configured, list) else [])
        return cls(
            workspace=_norm(workspace),
            added=tuple(_norm(p) for p in added),
            other_all=other_all,
            allowed=allowed,
            protected=tuple(_norm(p) for p in protected_raw),
        )

    @property
    def enabled(self) -> bool:
        return self.other_all or bool(self.allowed)

    def base_roots(self) -> tuple[Path, ...]:
        return (self.workspace, *self.added)

    def roots(self) -> tuple[Path, ...]:
        """Folders the agent may use; with the 'all' switch this is only a hint."""
        return (*self.base_roots(), *self.allowed)

    def _in_base(self, path: Path) -> bool:
        return any(_is_within(path, root) for root in self.base_roots())

    def is_protected(self, path: Path) -> bool:
        path = _norm(path)
        if self._in_base(path):
            return False
        return any(_is_within(path, root) for root in self.protected)

    def can_read(self, path: Path) -> bool:
        path = _norm(path)
        if self.other_all:
            return True
        return any(_is_within(path, root) for root in self.roots())

    def can_write(self, path: Path) -> bool:
        return self.can_read(path) and not self.is_protected(path)

    def check(self, path: Path, *, write: bool = False, label: str | None = None) -> Path:
        resolved = _norm(path)
        shown = label if label is not None else str(path)
        if not self.can_read(resolved):
            raise PathAccessError(
                f'Path {shown!r} escapes all allowed directories. '
                'Add the folder with --add-dir, or set SIGIL_OTHER_DIRECTORIES=yes '
                '(or SIGIL_ALLOWED_DIRECTORIES) to allow other folders.'
            )
        if write and self.is_protected(resolved):
            raise PathAccessError(
                f'Path {shown!r} is in a write-protected folder. '
                'Add to the list with permissions.protectedPaths; SIGIL_PROTECTED_PATHS replaces it.'
            )
        return resolved

    def describe(self) -> str:
        if self.other_all:
            return 'Other directories: allowed'
        if self.allowed:
            return 'Other directories: ' + ', '.join(str(p) for p in self.allowed)
        return 'Other directories: off (set SIGIL_OTHER_DIRECTORIES=yes to allow)'


def path_access_for(
    workspace: Path,
    added: tuple[Path, ...] | list[Path] = (),
    *,
    settings: Mapping[str, Any] | None = None,
) -> PathAccess:
    """PathAccess from the live environment and (optionally) loaded settings."""
    if settings is None and 'SIGIL_PROTECTED_PATHS' not in os.environ:
        from .settings import load_settings

        try:
            settings = load_settings(workspace)
        except Exception as _ignored:
            _log_ignored(_ignored)
            settings = {}
    return PathAccess.from_env(workspace, added, settings=settings)
