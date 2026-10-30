"""Plugin folders, loaded with --plugin-dir (SIGIL_PLUGIN_DIRS).

A plugin folder looks like this (every part is optional):

    my-plugin/
      .sigil-plugin/plugin.json   {"name", "description", "version", and optional
                                    "commands" / "agents" / "skills" (extra paths),
                                    "hooks" / "mcpServers" (a path or an inline object)}
      commands/*.md                slash commands: /my-plugin:name, and /name if free
      agents/*.md                  subagents:      my-plugin:name, and name if free
      skills/<name>/SKILL.md       skills
      hooks/hooks.json             {"hooks": {...}} in the settings.json format
      .mcp.json                    {"mcpServers": {...}}

${SIGIL_PLUGIN_ROOT} in hook commands, MCP server settings and command bodies
becomes the plugin folder. SIGIL_PLUGIN_DIRS is an os.pathsep list; --plugin-dir
(repeatable) adds to it. --safe-mode loads no plugins.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .debug_log import get_logger

ROOT_TOKEN = '${SIGIL_PLUGIN_ROOT}'
_log = get_logger('sigil_plugins')


@dataclass(frozen=True)
class SigilPlugin:
    name: str
    root: Path
    description: str = ''
    version: str = ''
    command_dirs: tuple[Path, ...] = ()
    agent_dirs: tuple[Path, ...] = ()
    skill_dirs: tuple[Path, ...] = ()
    hooks: dict[str, Any] = field(default_factory=dict)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    mcp_source: Path | None = None
    errors: tuple[str, ...] = ()


def plugin_dirs() -> list[Path]:
    if os.environ.get('SIGIL_SAFE_MODE') == '1':
        return []
    raw = os.environ.get('SIGIL_PLUGIN_DIRS', '')
    folders = [Path(part).expanduser().resolve() for part in raw.split(os.pathsep) if part.strip()]
    try:
        from .plugin_marketplace import installed_plugin_dirs

        folders += [folder for folder in installed_plugin_dirs() if folder not in folders]  # /plugin install
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    return folders


def plugin_data_dir(root: Path) -> Path:
    """${SIGIL_PLUGIN_DATA}: a folder the plugin may keep data in (kept across updates)."""
    from .settings import _home

    return _home() / '.sigil' / 'plugins' / 'data' / Path(root).name


def expand_root(value: Any, root: Path) -> Any:
    """Replace ${SIGIL_PLUGIN_ROOT}, ${SIGIL_PLUGIN_DATA} and ${SIGIL_PROJECT_DIR} in every string
    of a JSON-like value (the documented plugin variables)."""
    if isinstance(value, str):
        if '${SIGIL_PLUGIN_DATA}' in value:
            data = plugin_data_dir(root)
            data.mkdir(parents=True, exist_ok=True)
            value = value.replace('${SIGIL_PLUGIN_DATA}', str(data))
        if '${SIGIL_PROJECT_DIR}' in value:
            from .settings import project_dir

            value = value.replace('${SIGIL_PROJECT_DIR}', str(project_dir()))  # the agent's folder
        return value.replace(ROOT_TOKEN, str(root))
    if isinstance(value, list):
        return [expand_root(item, root) for item in value]
    if isinstance(value, dict):
        return {key: expand_root(item, root) for key, item in value.items()}
    return value


def _paths(value: Any, root: Path) -> list[Path]:
    items = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    return [(root / str(item)).resolve() for item in items if isinstance(item, str) and item.strip()]


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        errors.append(f'{path}: {exc}')
        return {}
    return payload if isinstance(payload, dict) else {}


def _object_or_file(value: Any, default: Path, root: Path, errors: list[str]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, dict):
        return value, root / '.sigil-plugin' / 'plugin.json'
    path = (root / value).resolve() if isinstance(value, str) and value.strip() else default
    if not path.is_file():
        return {}, None
    return _read_json(path, errors), path


def load_plugin(root: Path) -> SigilPlugin | None:
    root = Path(root).resolve()
    if not root.is_dir():
        _log.warning('plugin folder not found: %s', root)
        return None
    errors: list[str] = []
    manifest_path = root / '.sigil-plugin' / 'plugin.json'
    manifest = _read_json(manifest_path, errors) if manifest_path.is_file() else {}
    name = str(manifest.get('name') or root.name).strip() or root.name

    def dirs(key: str, default: str) -> tuple[Path, ...]:
        found = [root / default] if (root / default).is_dir() else []
        found += [p for p in _paths(manifest.get(key), root) if p.exists() and p not in found]
        return tuple(found)

    hooks_payload, _ = _object_or_file(manifest.get('hooks'), root / 'hooks' / 'hooks.json', root, errors)
    hooks = hooks_payload.get('hooks') if isinstance(hooks_payload.get('hooks'), dict) else hooks_payload
    mcp_payload, mcp_source = _object_or_file(manifest.get('mcpServers'), root / '.mcp.json', root, errors)
    servers = mcp_payload.get('mcpServers') if isinstance(mcp_payload.get('mcpServers'), dict) else mcp_payload
    plugin = SigilPlugin(
        name=name,
        root=root,
        description=str(manifest.get('description') or ''),
        version=str(manifest.get('version') or ''),
        command_dirs=dirs('commands', 'commands'),
        agent_dirs=dirs('agents', 'agents'),
        skill_dirs=dirs('skills', 'skills'),
        hooks=expand_root(hooks, root) if isinstance(hooks, dict) else {},
        mcp_servers=expand_root(servers, root) if isinstance(servers, dict) else {},
        mcp_source=mcp_source,
        errors=tuple(errors),
    )
    for error in errors:
        _log.warning('plugin %s: %s', name, error)
    return plugin


def loaded_plugins() -> list[SigilPlugin]:
    plugins: list[SigilPlugin] = []
    seen: set[str] = set()
    for folder in plugin_dirs():
        plugin = load_plugin(folder)
        if plugin is None or plugin.name in seen:
            continue
        seen.add(plugin.name)
        plugins.append(plugin)
    return plugins


def render_plugin_list(plugins: list[SigilPlugin] | None = None) -> str:
    plugins = loaded_plugins() if plugins is None else plugins
    if not plugins:
        return 'No plugin folders loaded (use --plugin-dir <folder>).'
    lines = ['Plugin folders:']
    for plugin in plugins:
        parts = [f'{len(plugin.command_dirs)} command dir(s)', f'{len(plugin.agent_dirs)} agent dir(s)',
                 f'{len(plugin.skill_dirs)} skill dir(s)', f'{len(plugin.hooks)} hook event(s)',
                 f'{len(plugin.mcp_servers)} MCP server(s)']
        version = f' {plugin.version}' if plugin.version else ''
        lines.append(f'  {plugin.name}{version}: {", ".join(parts)}  ({plugin.root})')
        if plugin.description:
            lines.append(f'    {plugin.description}')
        for error in plugin.errors:
            lines.append(f'    error: {error}')
    return '\n'.join(lines)
