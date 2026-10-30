from __future__ import annotations

import atexit
import collections
import json
import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import threading
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


MCP_PROTOCOL_VERSION = '2025-11-25'


# Rate limiting state (shared across all MCPRuntime instances)
_rate_limit_lock = threading.Lock()
_rate_limit_state: dict[str, dict[str, Any]] = {}
def _scrub_all() -> bool:
    return os.environ.get('SIGIL_SUBPROCESS_ENV_SCRUB', '').strip().lower() in ('1', 'true', 'yes', 'on')


def _credential_name(name: str) -> bool:
    from .hook_policy import secret_like

    return secret_like(name)


_RATE_LIMIT_STATE_FILE = Path.home() / '.sigil' / 'mcp_rate_limits.json'
_LEGACY_RATE_LIMIT_STATE_FILE = Path(__file__).parent / '.mcp_rate_limits.json'  # read if the new one is missing
_MONTH_KEEP_SECONDS = 32 * 86400  # a month window (startDate) never reaches further back


def _load_rate_limit_state() -> dict[str, dict[str, Any]]:
    """Load rate limit state from disk."""
    for path in (_RATE_LIMIT_STATE_FILE, _LEGACY_RATE_LIMIT_STATE_FILE):
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError) as _ignored:
            _log_ignored(_ignored)
            continue
    return {}


def _save_rate_limit_state() -> None:
    """Save rate limit state to disk."""
    try:
        from .settings import write_text_atomic

        write_text_atomic(_RATE_LIMIT_STATE_FILE, json.dumps(_rate_limit_state, indent=2))  # shared by processes
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass


# Load persisted state on module import
_rate_limit_state = _load_rate_limit_state()


def _check_rate_limit(server: MCPServerProfile) -> tuple[bool, str]:
    """Check if a request is allowed under rate limits.
    
    Returns (is_allowed, message) tuple.
    """
    if not server.rate_limits:
        return True, ''
    
    server_name = server.name
    rate_config = server.rate_limits
    
    requests_per_hour = rate_config.get('requestsPerHour')
    requests_per_month = rate_config.get('requestsPerMonth')
    
    if not requests_per_hour and not requests_per_month:
        return True, ''
    
    with _rate_limit_lock:
        if server_name not in _rate_limit_state:
            _rate_limit_state[server_name] = {
                'hourly_requests': [],
                'monthly_requests': [],
                'current_key_index': 0,
            }
        
        state = _rate_limit_state[server_name]
        now = time.time()
        hour_ago = now - 3600
        
        # Clean up old hourly requests
        state['hourly_requests'] = [t for t in state['hourly_requests'] if t > hour_ago]
        
        # Check key rotation
        key_rotation = rate_config.get('keyRotation', {})
        num_keys = 1
        if key_rotation.get('enabled') and key_rotation.get('keys'):
            num_keys = len(key_rotation['keys'])
        
        # Calculate effective limits with key rotation
        effective_hourly = (requests_per_hour or float('inf')) * num_keys
        effective_monthly = (requests_per_month or float('inf')) * num_keys
        
        # Check hourly limit
        if requests_per_hour and len(state['hourly_requests']) >= effective_hourly:
            return False, f'Hourly limit reached ({requests_per_hour} requests/hour per key, {num_keys} keys = {effective_hourly} total)'
        
        # Check monthly limit
        if requests_per_month:
            month_start = _get_month_start(rate_config.get('startDate'))
            month_requests = [t for t in state['monthly_requests'] if t > month_start]
            if len(month_requests) >= effective_monthly:
                return False, f'Monthly limit reached ({requests_per_month} requests/month per key, {num_keys} keys = {effective_monthly} total)'
        
        return True, ''


def _record_rate_limit_usage(server: MCPServerProfile) -> None:
    """Record a request for rate limit tracking."""
    if not server.rate_limits:
        return
    
    server_name = server.name
    now = time.time()
    
    with _rate_limit_lock:
        if server_name not in _rate_limit_state:
            _rate_limit_state[server_name] = {
                'hourly_requests': [],
                'monthly_requests': [],
                'current_key_index': 0,
            }
        
        state = _rate_limit_state[server_name]
        # keep only what a limit can still count (one hour, about one month)
        state['hourly_requests'] = [t for t in state.get('hourly_requests', []) if t > now - 3600]
        state['monthly_requests'] = [t for t in state.get('monthly_requests', []) if t > now - _MONTH_KEEP_SECONDS]
        state['hourly_requests'].append(now)
        state['monthly_requests'].append(now)
        
        # Rotate key if configured
        key_rotation = server.rate_limits.get('keyRotation', {})
        if key_rotation.get('enabled') and key_rotation.get('rotateOnLimit'):
            keys = key_rotation.get('keys', [])
            if len(keys) > 1:
                state['current_key_index'] = (state['current_key_index'] + 1) % len(keys)
        
        # Persist state to disk
        _save_rate_limit_state()


def _get_month_start(start_date_str: str | None) -> float:
    """Get the timestamp for the start of the current billing month."""
    from datetime import datetime
    
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            now = datetime.now()
            # Calculate current month's start based on the billing start day
            current_month_start = datetime(now.year, now.month, start_date.day)
            if current_month_start > now:
                # If we haven't reached this month's billing day, use last month
                if now.month == 1:
                    current_month_start = datetime(now.year - 1, 12, start_date.day)
                else:
                    current_month_start = datetime(now.year, now.month - 1, start_date.day)
            return current_month_start.timestamp()
        except (ValueError, TypeError) as _ignored:
            _log_ignored(_ignored)
            pass
    
    # Default to start of current calendar month
    now = datetime.now()
    return datetime(now.year, now.month, 1).timestamp()


def _get_active_api_key(server: MCPServerProfile) -> str | None:
    """Get the currently active API key for a server with key rotation."""
    key_rotation = server.rate_limits.get('keyRotation', {})
    if not key_rotation.get('enabled'):
        # Return the first API key found in env
        for key, value in server.env.items():
            if 'API_KEY' in key:
                return value
        return None
    
    keys = key_rotation.get('keys', [])
    if not keys:
        return None
    
    with _rate_limit_lock:
        state = _rate_limit_state.get(server.name, {})
        key_index = state.get('current_key_index', 0)
        key_name = keys[key_index % len(keys)]
        return server.env.get(key_name)


@dataclass(frozen=True)
class MCPResource:
    uri: str
    server_name: str
    source_manifest: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None
    resolved_path: str | None = None
    inline_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolRecord:
    name: str
    server_name: str
    source_manifest: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPServerProfile:
    name: str
    source_manifest: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rate_limits: dict[str, Any] = field(default_factory=dict)
    url: str | None = None  # http / sse servers
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPRuntime:
    resources: tuple[MCPResource, ...] = field(default_factory=tuple)
    servers: tuple[MCPServerProfile, ...] = field(default_factory=tuple)
    # Cache for remote tools to avoid repeated subprocess spawning
    _tools_cache: tuple[McpToolRecord, ...] | None = field(default=None, init=False, repr=False)
    _tools_cache_time: float = field(default=0.0, init=False, repr=False)
    _TOOLS_CACHE_TTL: float = field(default=60.0, init=False, repr=False)  # Cache for 60 seconds

    @classmethod
    def from_workspace(
        cls,
        cwd: Path,
        additional_working_directories: tuple[str, ...] = (),
    ) -> 'MCPRuntime':
        resources: list[MCPResource] = []
        servers: list[MCPServerProfile] = []
        if os.environ.get('SIGIL_SAFE_MODE') == '1':
            return cls()  # --safe-mode: no MCP
        extra = [Path(p).expanduser().resolve() for p in os.environ.get('SIGIL_MCP_CONFIG', '').split(os.pathsep)
                 if p.strip()]
        strict = os.environ.get('SIGIL_STRICT_MCP_CONFIG') == '1'
        paths = list(extra) if strict else [*_discover_manifest_paths(cwd, additional_working_directories), *extra]
        from .workspace_trust import allowed as trust_allows, is_trusted, report_skipped, _mcp_manifest_is_user_level

        trusted = is_trusted(cwd)
        skipped_any = False
        for path in paths:
            if not path.is_file():
                continue
            manifest_resources, manifest_servers = _load_manifest(path)
            if not trusted and path not in extra and not _mcp_manifest_is_user_level(path):
                # Why: a project's MCP servers used to start (and read ${VAR}s) with no question asked.
                # Untrusted: only the servers the user added themselves (sigil mcp add) stay.
                kept = [s for s in manifest_servers if trust_allows(
                    cwd, 'mcp', s.name, s.url or ' '.join([str(s.command or ''), *s.args]).strip())]
                skipped_any = skipped_any or len(kept) < len(manifest_servers)
                manifest_servers = kept
                manifest_resources = []
            resources.extend(manifest_resources)
            servers.extend(manifest_servers)
        if skipped_any:
            report_skipped(cwd)
        if not strict:
            from .sigil_plugins import loaded_plugins

            for plugin in loaded_plugins():  # --plugin-dir: the plugin's .mcp.json
                source = plugin.mcp_source or plugin.root / '.mcp.json'
                for name, item in plugin.mcp_servers.items():
                    if isinstance(name, str) and name.strip() and isinstance(item, dict):
                        server = _extract_server_profile(name.strip(), item, manifest_path=source)
                        if server is not None:
                            servers.append(server)
        disabled = disabled_mcp_servers(cwd)
        return cls(
            resources=tuple(resources),
            servers=tuple(server for server in _dedupe_servers(servers) if server.name not in disabled),
        )

    @property
    def manifests(self) -> tuple[str, ...]:
        seen: list[str] = []
        for entry in [*self.resources, *self.servers]:
            source_manifest = entry.source_manifest
            if source_manifest not in seen:
                seen.append(source_manifest)
        return tuple(seen)

    def has_transport_servers(self) -> bool:
        return any(server.transport in SUPPORTED_TRANSPORTS for server in self.servers)

    def list_resources(
        self,
        *,
        query: str | None = None,
        limit: int | None = None,
    ) -> tuple[MCPResource, ...]:
        resources = list(self.resources)
        resources.extend(self._list_remote_resources())
        filtered = _filter_resources(tuple(resources), query=query)
        if limit is not None and limit >= 0:
            filtered = filtered[:limit]
        return filtered

    def get_resource(self, uri: str) -> MCPResource | None:
        for resource in self.resources:
            if resource.uri == uri:
                return resource
        for resource in self._list_remote_resources():
            if resource.uri == uri:
                return resource
        return None

    def read_resource(self, uri: str, *, max_chars: int = 12000) -> str:
        for resource in self.resources:
            if resource.uri != uri:
                continue
            if resource.inline_text is not None:
                return _truncate(resource.inline_text, max_chars)
            if resource.resolved_path is not None:
                path = Path(resource.resolved_path)
                if not path.exists() or not path.is_file():
                    raise FileNotFoundError(f'MCP resource file not found: {path}')
                text = path.read_text(encoding='utf-8', errors='replace')
                return _truncate(text, max_chars)
        last_error: Exception | None = None
        candidate_servers: list[MCPServerProfile] = []
        discovered = self.get_resource(uri)
        if discovered is not None:
            server = self.get_server(discovered.server_name)
            if server is not None:
                candidate_servers.append(server)
        for server in self.servers:
            if server.transport not in SUPPORTED_TRANSPORTS:
                continue
            if all(existing.name != server.name for existing in candidate_servers):
                candidate_servers.append(server)
        for server in candidate_servers:
            try:
                result = _request_stdio(server, 'resources/read', {'uri': uri})
                result = _resolve_input_required(server, 'resources/read', {'uri': uri}, result, 30.0)
            except Exception as exc:
                last_error = exc
                continue
            rendered = _render_resource_contents(result.get('contents'))
            if rendered:
                return _truncate(rendered, max_chars)
        if last_error is not None:
            raise FileNotFoundError(f'Unable to read MCP resource {uri}: {last_error}') from last_error
        raise FileNotFoundError(f'Unknown MCP resource: {uri}')

    def list_tools(
        self,
        *,
        query: str | None = None,
        server_name: str | None = None,
        limit: int | None = None,
    ) -> tuple[McpToolRecord, ...]:
        # Use cache if available and not expired
        now = time.monotonic()
        if self._tools_cache is not None and (now - self._tools_cache_time) < self._TOOLS_CACHE_TTL:
            tools = self._tools_cache
        else:
            tools = self._list_remote_tools(server_name=None)
            # Update cache
            self._tools_cache = tools
            self._tools_cache_time = now
        
        # Filter by server_name if specified
        if server_name:
            tools = tuple(tool for tool in tools if tool.server_name == server_name)
        
        if query:
            needle = query.lower()
            tools = tuple(
                tool
                for tool in tools
                if needle in tool.name.lower()
                or needle in (tool.description or '').lower()
                or needle in tool.server_name.lower()
            )
        if limit is not None and limit >= 0:
            tools = tools[:limit]
        return tools

    def call_tool(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        server_name: str | None = None,
        max_chars: int = 12000,
        timeout_seconds: float = 30.0,
    ) -> tuple[str, dict[str, Any]]:
        tool = self._resolve_tool(tool_name, server_name=server_name)
        server = self.get_server(tool.server_name)
        if server is None:
            raise FileNotFoundError(f'Unknown MCP server: {tool.server_name}')

        # Check rate limits before making the call
        rate_limit_ok, rate_limit_msg = _check_rate_limit(server)
        if not rate_limit_ok:
            rendered = f'Rate limit exceeded for server {server.name}: {rate_limit_msg}'
            metadata = {
                'server_name': tool.server_name,
                'tool_name': tool.name,
                'is_error': True,
                'rate_limited': True,
            }
            return rendered, metadata

        payload = {
            'name': tool.name,
            'arguments': dict(arguments or {}),
        }
        result = _request_stdio(server, 'tools/call', payload, timeout_seconds=timeout_seconds)
        result = _resolve_input_required(server, 'tools/call', payload, result, timeout_seconds)

        # Record this request for rate limit tracking
        _record_rate_limit_usage(server)

        rendered = _truncate(_render_tool_call_result(result), max_chars)
        metadata = {
            'server_name': tool.server_name,
            'tool_name': tool.name,
            'is_error': bool(result.get('isError')),
        }
        return rendered, metadata

    def start_background(self) -> None:
        """Start every stdio server now, in the background (status in /mcp)."""
        connection_manager().start_background(self.servers)

    def wait_for_servers(self, timeout_seconds: float = 30.0) -> dict[str, str]:
        return connection_manager().wait([server.name for server in self.servers], timeout_seconds)

    def get_server(self, name: str) -> MCPServerProfile | None:
        needle = name.strip().lower()
        if not needle:
            return None
        for server in self.servers:
            if server.name.lower() == needle:
                return server
        return None

    def render_summary(self) -> str:
        if not self.resources and not self.servers:
            return 'No local MCP manifests, servers, or resources discovered.'
        lines = [
            f'Local MCP manifests: {len(self.manifests)}',
            f'Local MCP resources: {len(self.resources)}',
            f'Configured MCP servers: {len(self.servers)}',
        ]
        transport_counts: dict[str, int] = {}
        for server in self.servers:
            transport_counts[server.transport] = transport_counts.get(server.transport, 0) + 1
        for transport, count in sorted(transport_counts.items()):
            lines.append(f'- {transport}: {count} server(s)')
        by_server: dict[str, int] = {}
        for resource in self.resources:
            by_server[resource.server_name] = by_server.get(resource.server_name, 0) + 1
        for server_name, count in sorted(by_server.items()):
            lines.append(f'- local resources for {server_name}: {count}')
        for server in self.servers[:10]:
            details = [server.name, server.transport]
            if server.command:
                details.append(server.command)
            status = connection_manager().status.get(server.name)
            if status:
                details.append(status)
            lines.append('- Server: ' + ' ; '.join(details))
        return '\n'.join(lines)

    def render_resource_index(
        self,
        *,
        query: str | None = None,
        limit: int = 20,
    ) -> str:
        resources = self.list_resources(query=query, limit=limit)
        if not resources:
            return '# MCP Resources\n\nNo matching MCP resources discovered.'
        lines = ['# MCP Resources', '']
        for resource in resources:
            details = [resource.uri]
            details.append(f'server={resource.server_name}')
            if resource.name:
                details.append(f'name={resource.name}')
            if resource.mime_type:
                details.append(f'mime={resource.mime_type}')
            if resource.resolved_path:
                details.append(f'path={resource.resolved_path}')
            elif resource.inline_text is not None:
                details.append('source=inline')
            else:
                details.append('source=transport')
            lines.append('- ' + '; '.join(details))
        return '\n'.join(lines)

    def render_resource(self, uri: str, *, max_chars: int = 12000) -> str:
        resource = self.get_resource(uri)
        if resource is None:
            return f'# MCP Resource\n\nUnknown MCP resource: {uri}'
        lines = [
            '# MCP Resource',
            '',
            f'- URI: {resource.uri}',
            f'- Server: {resource.server_name}',
        ]
        if resource.name:
            lines.append(f'- Name: {resource.name}')
        if resource.mime_type:
            lines.append(f'- MIME Type: {resource.mime_type}')
        if resource.resolved_path:
            lines.append(f'- Path: {resource.resolved_path}')
        lines.extend(['', self.read_resource(uri, max_chars=max_chars)])
        return '\n'.join(lines)

    def render_tool_index(
        self,
        *,
        query: str | None = None,
        server_name: str | None = None,
        limit: int = 50,
    ) -> str:
        tools = self.list_tools(query=query, server_name=server_name, limit=limit)
        if not tools:
            return '# MCP Tools\n\nNo matching MCP tools discovered.'
        lines = ['# MCP Tools', '']
        for tool in tools:
            details = [tool.name, f'server={tool.server_name}']
            if tool.description:
                details.append(tool.description)
            lines.append('- ' + ' ; '.join(details))
        return '\n'.join(lines)

    def render_tool_call(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        server_name: str | None = None,
        max_chars: int = 12000,
    ) -> str:
        content, metadata = self.call_tool(
            tool_name,
            arguments=arguments,
            server_name=server_name,
            max_chars=max_chars,
        )
        lines = [
            '# MCP Tool Result',
            '',
            f'- Tool: {tool_name}',
            f'- Server: {metadata["server_name"]}',
            f'- is_error: {metadata["is_error"]}',
            '',
            content,
        ]
        return '\n'.join(lines)

    def _list_remote_resources(self) -> tuple[MCPResource, ...]:
        discovered: list[MCPResource] = []
        for server in self.servers:
            if server.transport not in SUPPORTED_TRANSPORTS:
                continue
            try:
                result = _request_stdio(server, 'resources/list', {})
            except OSError as _ignored:
                _log_ignored(_ignored)
                continue
            for item in _extract_remote_resources(server, result):
                discovered.append(item)
        return tuple(discovered)

    def list_prompts(self) -> list[dict[str, Any]]:
        """MCP prompts of every server: {server, name, description, arguments}."""
        found: list[dict[str, Any]] = []
        for server in self.servers:
            if server.transport not in SUPPORTED_TRANSPORTS:
                continue
            try:
                result = _request_stdio(server, 'prompts/list', {}, timeout_seconds=30.0)
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)  # no prompts capability, or the server is down
                continue
            for item in result.get('prompts') or []:
                if isinstance(item, dict) and isinstance(item.get('name'), str):
                    found.append({'server': server.name, 'name': item['name'],
                                  'description': str(item.get('description') or ''),
                                  'arguments': [a for a in item.get('arguments') or [] if isinstance(a, dict)]})
        return found

    def get_prompt(self, server_name: str, name: str, arguments: dict[str, str]) -> str:
        """Text of an MCP prompt (prompts/get), its messages joined."""
        server = self.get_server(server_name)
        if server is None:
            raise FileNotFoundError(f'Unknown MCP server: {server_name}')
        params = {'name': name, 'arguments': arguments}
        result = _request_stdio(server, 'prompts/get', params, timeout_seconds=30.0)
        result = _resolve_input_required(server, 'prompts/get', params, result, 30.0)
        parts: list[str] = []
        for message in result.get('messages') or []:
            content = message.get('content') if isinstance(message, dict) else None
            items = content if isinstance(content, list) else [content]
            for item in items:
                if isinstance(item, dict) and item.get('type') == 'text':
                    parts.append(str(item.get('text', '')))
                elif isinstance(item, dict) and item.get('type') == 'resource':
                    parts.append(str((item.get('resource') or {}).get('text', '')))
        return '\n\n'.join(part for part in parts if part)

    def _list_remote_tools(self, *, server_name: str | None = None) -> tuple[McpToolRecord, ...]:
        discovered: list[McpToolRecord] = []
        candidate_servers = (
            [self.get_server(server_name)] if server_name else list(self.servers)
        )
        for server in candidate_servers:
            if server is None or server.transport not in SUPPORTED_TRANSPORTS:
                continue
            try:
                result = _request_stdio(server, 'tools/list', {}, timeout_seconds=30.0)
            except OSError as _ignored:
                _log_ignored(_ignored)
                continue
            for item in _extract_remote_tools(server, result):
                discovered.append(item)
        return tuple(discovered)

    def _resolve_tool(self, tool_name: str, server_name: str | None = None) -> McpToolRecord:
        tools = self.list_tools(server_name=server_name)
        matches = [tool for tool in tools if tool.name == tool_name]
        if server_name:
            if not matches:
                raise FileNotFoundError(f'Unknown MCP tool: {tool_name} on server {server_name}')
            return matches[0]
        if not matches:
            raise FileNotFoundError(f'Unknown MCP tool: {tool_name}')
        if len(matches) > 1:
            raise FileNotFoundError(
                f'MCP tool {tool_name} exists on multiple servers. Pass server_name to disambiguate.'
            )
        return matches[0]


def _discover_manifest_paths(
    cwd: Path,
    additional_working_directories: tuple[str, ...],
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def remember(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            return
        seen.add(resolved)
        candidates.append(resolved)

    roots: list[Path] = []
    current = cwd.resolve()
    while True:
        roots.append(current)
        if current.parent == current:
            break
        current = current.parent
    roots.extend(Path(path).resolve() for path in additional_working_directories)

    for root in roots:
        remember(root / '.sigil-mcp.json')
        remember(root / '.mcp.json')
        remember(root / '.codex-mcp.json')
        remember(root / 'mcp.json')
    remember(user_mcp_config_path())  # sigil mcp add --scope user
    return tuple(candidates)


def user_mcp_config_path() -> Path:
    from .settings import _home

    return _home() / '.sigil' / 'mcp.json'


def disabled_mcp_servers(cwd: Path) -> set[str]:
    """Servers turned off with /mcp disable (disabledMcpServers in settings)."""
    try:
        from .settings import load_settings

        names = load_settings(cwd).get('disabledMcpServers')
    except Exception as _ignored:
        _log_ignored(_ignored)
        names = None
    return {str(name) for name in names} if isinstance(names, list) else set()


def _load_manifest(path: Path) -> tuple[list[MCPResource], list[MCPServerProfile]]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as _ignored:
        _log_ignored(_ignored)
        return [], []
    if not isinstance(payload, dict):
        return [], []

    resources: list[MCPResource] = []
    servers: list[MCPServerProfile] = []

    if isinstance(payload.get('resources'), list):
        resources.extend(
            _extract_resources(
                payload.get('name') if isinstance(payload.get('name'), str) else 'local',
                payload['resources'],
                manifest_path=path,
            )
        )

    raw_servers = payload.get('servers')
    if isinstance(raw_servers, list):
        for item in raw_servers:
            if not isinstance(item, dict):
                continue
            name = item.get('name')
            if not isinstance(name, str) or not name.strip():
                continue
            server_name = name.strip()
            raw_resources = item.get('resources')
            if isinstance(raw_resources, list):
                resources.extend(
                    _extract_resources(server_name, raw_resources, manifest_path=path)
                )
            server = _extract_server_profile(server_name, item, manifest_path=path)
            if server is not None:
                servers.append(server)

    raw_mcp_servers = payload.get('mcpServers')
    if isinstance(raw_mcp_servers, dict):
        # Extract global rate limits configuration
        global_rate_limits: dict[str, Any] = {}
        raw_rate_limits = payload.get('rateLimits')
        if isinstance(raw_rate_limits, dict):
            global_rate_limits = raw_rate_limits
        
        for server_name, item in raw_mcp_servers.items():
            if not isinstance(server_name, str) or not server_name.strip():
                continue
            if not isinstance(item, dict):
                continue
            # Get server-specific rate limits from global config
            server_rate_limits = global_rate_limits.get(server_name.strip(), {})
            server = _extract_server_profile(
                server_name.strip(), 
                item, 
                manifest_path=path,
                rate_limits=server_rate_limits if isinstance(server_rate_limits, dict) else {}
            )
            if server is not None:
                servers.append(server)

    return resources, servers


def _extract_server_profile(
    server_name: str,
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    rate_limits: dict[str, Any] | None = None,
) -> MCPServerProfile | None:
    transport_raw = payload.get('transport') or payload.get('type')
    url = payload.get('url')
    if isinstance(url, str) and url.strip() and (
            not isinstance(payload.get('command'), str) or str(transport_raw or '').lower() in HTTP_TRANSPORTS):
        kind = str(transport_raw or 'http').strip().lower()
        kind = 'http' if kind in ('streamable-http', 'streamable_http', 'http') else kind
        if kind not in HTTP_TRANSPORTS:
            return None
        headers = payload.get('headers')
        return MCPServerProfile(
            name=server_name,
            source_manifest=str(manifest_path),
            transport=kind,
            url=url.strip(),
            headers={str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {},
            description=payload.get('description') if isinstance(payload.get('description'), str) else None,
            metadata={'oauth': dict(payload['oauth'])} if isinstance(payload.get('oauth'), dict) else {},
            rate_limits=rate_limits if rate_limits else {},
        )
    command = payload.get('command')
    if not isinstance(command, str) or not command.strip():
        return None
    args = payload.get('args', ())
    if not isinstance(args, list):
        args = ()
    normalized_args = tuple(
        item for item in args if isinstance(item, str)
    )
    env = payload.get('env')
    normalized_env = {
        key: value
        for key, value in (env.items() if isinstance(env, dict) else [])
        if isinstance(key, str) and isinstance(value, str)
    }
    cwd = payload.get('cwd')
    resolved_cwd: str | None = None
    if isinstance(cwd, str) and cwd.strip():
        candidate = Path(cwd).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        resolved_cwd = str(candidate.resolve())
    description = payload.get('description') if isinstance(payload.get('description'), str) else None
    transport = payload.get('transport') or payload.get('type')
    if not isinstance(transport, str) or not transport.strip():
        transport = 'stdio'
    transport = transport.strip().lower()
    if transport != 'stdio':
        return None
    metadata = payload.get('metadata')
    return MCPServerProfile(
        name=server_name,
        source_manifest=str(manifest_path),
        transport=transport,
        command=command.strip(),
        args=normalized_args,
        env=normalized_env,
        cwd=resolved_cwd,
        description=description,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
        rate_limits=rate_limits if rate_limits else {},
    )


def _extract_resources(
    server_name: str,
    raw_resources: list[Any],
    *,
    manifest_path: Path,
) -> list[MCPResource]:
    resources: list[MCPResource] = []
    seen_uris: set[str] = set()
    for item in raw_resources:
        if not isinstance(item, dict):
            continue
        uri = item.get('uri')
        if not isinstance(uri, str) or not uri.strip():
            continue
        uri = uri.strip()
        if uri in seen_uris:
            continue
        seen_uris.add(uri)
        raw_path = item.get('path')
        if raw_path is None:
            raw_path = item.get('file')
        resolved_path: str | None = None
        if isinstance(raw_path, str) and raw_path.strip():
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            resolved_path = str(candidate.resolve())
        inline_text = item.get('text')
        if not isinstance(inline_text, str):
            inline_text = None
        metadata = item.get('metadata')
        resources.append(
            MCPResource(
                uri=uri,
                server_name=server_name,
                source_manifest=str(manifest_path),
                name=item.get('name') if isinstance(item.get('name'), str) else None,
                description=(
                    item.get('description')
                    if isinstance(item.get('description'), str)
                    else None
                ),
                mime_type=(
                    item.get('mimeType')
                    if isinstance(item.get('mimeType'), str)
                    else item.get('mime_type')
                    if isinstance(item.get('mime_type'), str)
                    else None
                ),
                resolved_path=resolved_path,
                inline_text=inline_text,
                metadata=dict(metadata) if isinstance(metadata, dict) else {},
            )
        )
    return resources


def _extract_remote_resources(
    server: MCPServerProfile,
    payload: dict[str, Any],
) -> tuple[MCPResource, ...]:
    raw_resources = payload.get('resources')
    if not isinstance(raw_resources, list):
        return ()
    resources: list[MCPResource] = []
    for item in raw_resources:
        if not isinstance(item, dict):
            continue
        uri = item.get('uri')
        if not isinstance(uri, str) or not uri.strip():
            continue
        resources.append(
            MCPResource(
                uri=uri.strip(),
                server_name=server.name,
                source_manifest=server.source_manifest,
                name=item.get('name') if isinstance(item.get('name'), str) else None,
                description=(
                    item.get('description')
                    if isinstance(item.get('description'), str)
                    else None
                ),
                mime_type=(
                    item.get('mimeType')
                    if isinstance(item.get('mimeType'), str)
                    else item.get('mime_type')
                    if isinstance(item.get('mime_type'), str)
                    else None
                ),
                metadata={
                    'transport': server.transport,
                    'server_command': server.command,
                },
            )
        )
    return tuple(resources)


def _extract_remote_tools(
    server: MCPServerProfile,
    payload: dict[str, Any],
) -> tuple[McpToolRecord, ...]:
    raw_tools = payload.get('tools')
    if not isinstance(raw_tools, list):
        return ()
    tools: list[McpToolRecord] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        if not isinstance(name, str) or not name.strip():
            continue
        input_schema = item.get('inputSchema')
        if not isinstance(input_schema, dict):
            input_schema = item.get('input_schema')
        tools.append(
            McpToolRecord(
                name=name.strip(),
                server_name=server.name,
                source_manifest=server.source_manifest,
                description=(
                    item.get('description')
                    if isinstance(item.get('description'), str)
                    else None
                ),
                input_schema=dict(input_schema) if isinstance(input_schema, dict) else {},
                metadata={
                    'transport': server.transport,
                    'server_command': server.command,
                },
            )
        )
    return tuple(tools)


def _filter_resources(
    resources: tuple[MCPResource, ...],
    *,
    query: str | None = None,
) -> tuple[MCPResource, ...]:
    if not query:
        return resources
    needle = query.lower()
    return tuple(
        resource
        for resource in resources
        if needle in resource.uri.lower()
        or needle in resource.server_name.lower()
        or needle in (resource.name or '').lower()
        or needle in (resource.description or '').lower()
    )


def _dedupe_servers(servers: list[MCPServerProfile]) -> list[MCPServerProfile]:
    seen: set[tuple[str, str, str | None, tuple[str, ...]]] = set()
    deduped: list[MCPServerProfile] = []
    for server in servers:
        key = (server.name.lower(), server.transport, server.command, server.args)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(server)
    return deduped


_WIN_SUSPENDED = 0x00000004  # CREATE_SUSPENDED: resumed by background_shells._win_contain


def _end_process_tree(process: 'subprocess.Popen[Any]') -> None:
    """End a stdio server and its children: terminate() alone leaves node running under
    npx/cmd on Windows (and under a shell wrapper elsewhere)."""
    if sys.platform == 'win32':
        try:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True,
                           timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.SubprocessError) as exc:
            _log_close_problem(process, 'taskkill', exc)
    else:
        import signal

        from .process_group import safe_killpg

        safe_killpg(process.pid, signal.SIGTERM)  # its own session's group only, never ours
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        if sys.platform != 'win32':
            import signal

            from .process_group import safe_killpg

            safe_killpg(process.pid, signal.SIGKILL)
        process.kill()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            _log_close_problem(process, 'wait after kill', exc)


def _log_close_problem(process: 'subprocess.Popen[Any]', step: str, exc: BaseException) -> None:
    from .debug_log import get_logger

    get_logger('mcp').debug('closing MCP server pid %s: %s failed: %s', process.pid, step, exc)


class _StdioMCPConnection:
    def __init__(self, server: MCPServerProfile, *, timeout_seconds: float = 10.0) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        # last 200 lines only: a session-long connection must not grow without bound
        self.stderr_lines: 'collections.deque[str]' = collections.deque(maxlen=200)
        self._request_id = 0
        # Thread-safe queue for stdout lines (Windows-compatible approach)
        # bounded: a server that floods stdout faster than it is read is slowed (back-pressure)
        self._stdout_queue: queue.Queue[str | None] = queue.Queue(maxsize=10000)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def __enter__(self) -> '_StdioMCPConnection':
        try:
            command = [self.server.command or '', *self.server.args]
            if not command[0]:
                raise OSError(f'MCP server {self.server.name} has no executable command')
            env = os.environ.copy()
            for name in list(env):
                # Sigil's own model keys are not for third-party MCP servers (a server that needs a
                # key gets it through its `env` in the manifest); SIGIL_SUBPROCESS_ENV_SCRUB=1 also
                # removes every other credential-looking variable
                if name.upper().startswith('SIGIL_API_KEY') or (_scrub_all() and _credential_name(name)):
                    env.pop(name, None)
            from .child_env import strip_ssh_agent

            # N21: no SSH agent socket for MCP servers unless SIGIL_MCP_SSH_AGENT=1 (their own `env` may set it)
            strip_ssh_agent(env, mcp=True)
            env.update(self.server.env)
            
            # Ensure Node.js is in PATH for npx.cmd on Windows
            # npx.cmd requires node.exe to be accessible
            if sys.platform == 'win32':
                nodejs_path = os.environ.get('NODEJS_PATH', r'C:\Program Files\nodejs')
                if os.path.isdir(nodejs_path):
                    current_path = env.get('PATH', '')
                    if nodejs_path not in current_path:
                        env['PATH'] = f"{nodejs_path};{current_path}"
            
            # Apply active API key rotation for rate-limited servers
            if self.server.rate_limits and self.server.rate_limits.get('keyRotation', {}).get('enabled'):
                active_key = _get_active_api_key(self.server)
                if active_key:
                    key_rotation = self.server.rate_limits.get('keyRotation', {})
                    keys = key_rotation.get('keys', [])
                    if keys:
                        # Get the key name that should be active
                        with _rate_limit_lock:
                            state = _rate_limit_state.get(self.server.name, {})
                            key_index = state.get('current_key_index', 0)
                            active_key_name = keys[key_index % len(keys)]
                            # Set the primary API key env var to the active key
                            env[active_key_name] = active_key
            
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.server.cwd or None,
                env=env,
                # own process group on POSIX, so close() can end what the server started; on
                # Windows a Job Object (started suspended, then resumed inside it): close() ends
                # every process it started, also those whose parent already exited
                start_new_session=(sys.platform != 'win32'),
                **({'creationflags': _WIN_SUSPENDED} if sys.platform == 'win32' else {}),
            )
            if sys.platform == 'win32':
                from .background_shells import _win_contain

                self._win_job = _win_contain(self.process)
            assert self.process.stdout is not None
            assert self.process.stderr is not None
            
            # Start background threads to read stdout and stderr (Windows-compatible)
            self._stdout_thread = threading.Thread(
                target=self._read_stdout, daemon=True
            )
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._read_stderr, daemon=True
            )
            self._stderr_thread.start()
            
            self._initialize()
            return self
        except Exception:
            self.close()
            raise

    def _read_stdout(self) -> None:
        """Background thread to read stdout lines."""
        if self.process is None or self.process.stdout is None:
            return
        try:
            for line in self.process.stdout:
                self._stdout_queue.put(line)
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
        finally:
            self._stdout_queue.put(None)  # Signal EOF

    def _read_stderr(self) -> None:
        """Background thread to read stderr lines."""
        if self.process is None or self.process.stderr is None:
            return
        try:
            for line in self.process.stderr:
                self.stderr_lines.append(line.rstrip())
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        process = self.process
        job, self._win_job = getattr(self, '_win_job', None), None
        if job is not None:
            from .background_shells import _win_terminate_job

            try:
                _win_terminate_job(job)  # the whole tree, orphans included
            except OSError as exc:
                _log_close_problem(process, 'TerminateJobObject', exc)
        if process is not None and sys.platform == 'win32' and process.poll() is None:
            # taskkill /T finds children through their parent, so end the tree while it runs
            _end_process_tree(process)
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)
            if sys.platform != 'win32':
                # the process group outlives its leader: signal it even if the server already quit
                _end_process_tree(process)
        # Wait for threads to finish
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=0.5)
            self._stdout_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=0.5)
            self._stderr_thread = None
        if process is None:
            return
        for stream_name in ('stdout', 'stderr'):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception as _ignored:
                    from .debug_log import log_ignored_error

                    log_ignored_error(_ignored)
        self.process = None

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        self._send(
            {
                'jsonrpc': '2.0',
                'id': request_id,
                'method': method,
                'params': params,
            }
        )
        response = self._await_response(request_id)
        error = response.get('error')
        if isinstance(error, dict):
            message = error.get('message')
            raise MCPRequestError(
                f'MCP {method} failed for server {self.server.name}: {message or error}'
            )
        result = response.get('result')
        if not isinstance(result, dict):
            return {}
        return result

    def _initialize(self) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._send(
            {
                'jsonrpc': '2.0',
                'id': request_id,
                'method': 'initialize',
                'params': {
                    'protocolVersion': MCP_PROTOCOL_VERSION,
                    'capabilities': {'elicitation': {}},
                    'clientInfo': {
                        'name': 'sigilcode',
                        'version': '0.1.0',
                    },
                },
            }
        )
        response = self._await_response(request_id)
        error = response.get('error')
        if isinstance(error, dict):
            raise OSError(
                f'MCP initialize failed for server {self.server.name}: {error.get("message") or error}'
            )
        self._send(
            {
                'jsonrpc': '2.0',
                'method': 'notifications/initialized',
                'params': {},
            }
        )
        # Some MCP servers (like omega-memory) need a brief delay after the
        # notifications/initialized notification before they can handle tool calls.
        # This gives the server time to process the notification and enter the
        # "initialized" state before subsequent requests.
        time.sleep(0.5)

    def _send(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise OSError(f'MCP server {self.server.name} is not running')
        self.process.stdin.write(json.dumps(payload, ensure_ascii=True) + '\n')
        self.process.stdin.flush()

    def _await_response(self, request_id: int) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stderr = '\n'.join(list(self.stderr_lines)[-5:])
                raise TimeoutError(
                    f'Timed out waiting for MCP response from {self.server.name}'
                    + (f' stderr={stderr}' if stderr else '')
                )
            try:
                # Use queue with timeout for Windows compatibility
                line = self._stdout_queue.get(timeout=min(remaining, 0.1))
                if line is None:
                    # EOF reached
                    stderr = '\n'.join(list(self.stderr_lines)[-5:])
                    raise OSError(
                        f'MCP server {self.server.name} closed connection'
                        + (f' stderr={stderr}' if stderr else '')
                    )
            except queue.Empty as _ignored:
                _log_ignored(_ignored)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as _ignored:
                _log_ignored(_ignored)
                continue
            if not isinstance(payload, dict):
                continue
            if 'method' in payload:
                if 'id' in payload:  # a request from the server (elicitation/create, ping)
                    self._answer_server_request(payload)
                continue
            if payload.get('id') == request_id:
                return payload

    def _answer_server_request(self, payload: dict[str, Any]) -> None:
        try:
            self._send(answer_server_request(self.server.name, payload))
        except OSError as _ignored:
            _log_ignored(_ignored)
            pass


def answer_server_request(server_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The reply to a request a server sends us (elicitation/create, ping; others: not supported)."""
    method = payload.get('method')
    reply: dict[str, Any] = {'jsonrpc': '2.0', 'id': payload.get('id')}
    if method == 'elicitation/create':
        params = payload.get('params') if isinstance(payload.get('params'), dict) else {}
        try:
            reply['result'] = _ELICITATION_HANDLER(server_name, params)
        except Exception as _ignored:
            _log_ignored(_ignored)
            reply['result'] = {'action': 'decline'}
    elif method == 'ping':
        reply['result'] = {}
    else:
        reply['error'] = {'code': -32601, 'message': f'Method not supported: {method}'}
    return reply


def _resolve_input_required(server: Any, method: str, params: dict[str, Any], result: Any,
                             timeout_seconds: float) -> Any:
    """MCP 2026-07-28: a result {"resultType": "input_required", "inputRequests": {key: request}} is
    answered (elicitation/create through the handler) and the call is sent again with
    "inputResponses" under the same keys and the unchanged "requestState"."""
    for _round in range(5):
        if not isinstance(result, dict) or result.get('resultType') != 'input_required':
            return result
        responses: dict[str, Any] = {}
        for key, item in (result.get('inputRequests') or {}).items():
            if isinstance(item, dict) and item.get('method') == 'elicitation/create':
                try:
                    responses[key] = _ELICITATION_HANDLER(server.name, item.get('params') or {})
                except Exception as _ignored:
                    _log_ignored(_ignored)
                    responses[key] = {'action': 'decline'}
            else:
                responses[key] = {'action': 'decline'}
        retry = {**params, 'inputResponses': responses}
        if 'requestState' in result:
            retry['requestState'] = result['requestState']
        result = _request_stdio(server, method, retry, timeout_seconds=timeout_seconds)
    return result


def _decline_elicitation(server_name: str, params: dict[str, Any]) -> dict[str, Any]:
    return {'action': 'decline'}


# (server name, elicitation params) -> {"action": "accept" | "decline" | "cancel", "content": {...}}.
# The chat sets one that asks the user; elsewhere requests are declined.
_ELICITATION_HANDLER: Callable[[str, dict[str, Any]], dict[str, Any]] = _decline_elicitation


def set_elicitation_handler(handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None) -> None:
    global _ELICITATION_HANDLER
    _ELICITATION_HANDLER = handler or _decline_elicitation


HTTP_TRANSPORTS = frozenset({'http', 'sse'})
SUPPORTED_TRANSPORTS = frozenset({'stdio'}) | HTTP_TRANSPORTS


def _connection_class(server: MCPServerProfile) -> Any:
    if server.transport == 'http':
        from .mcp_http import HttpMCPConnection

        return HttpMCPConnection
    if server.transport == 'sse':
        from .mcp_http import SseMCPConnection

        return SseMCPConnection
    return _StdioMCPConnection


class MCPRequestError(OSError):
    """The server answered with a JSON-RPC error (the connection itself is fine)."""


_IDEMPOTENT_METHODS = frozenset({'tools/list', 'resources/list', 'resources/read', 'prompts/list'})


def _persistent_enabled() -> bool:
    return os.environ.get('SIGIL_MCP_PERSISTENT', '1').strip().lower() not in {'0', 'false', 'no', 'off'}


class MCPConnectionManager:
    """Keeps one running process per MCP server for the whole session.

    Servers start on first use, or in the background at chat start. A server
    that died is restarted on the next request; list/read requests are retried
    once, tool calls are not (they may have side effects). At most
    SIGIL_MCP_MAX_CONNECTIONS (default 16) stay open; the least recently used is
    closed first. SIGIL_MCP_PERSISTENT=0 goes back to one process per request.
    """

    def __init__(self) -> None:
        self._connections: dict[tuple, _StdioMCPConnection] = {}
        self._locks: dict[tuple, threading.Lock] = {}
        self._last_used: dict[tuple, float] = {}
        self._guard = threading.Lock()
        self.status: dict[str, str] = {}

    @staticmethod
    def _key(server: MCPServerProfile) -> tuple:
        return (server.name, server.source_manifest, server.command or server.url, tuple(server.args))

    def _lock_for(self, key: tuple) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def _alive(self, connection: _StdioMCPConnection | None) -> bool:
        return connection is not None and connection.process is not None and connection.process.poll() is None

    def _open(self, server: MCPServerProfile, timeout_seconds: float) -> _StdioMCPConnection:
        key = self._key(server)
        self.status[server.name] = 'connecting'
        connection = _connection_class(server)(server, timeout_seconds=max(timeout_seconds, 10.0))
        try:
            connection.__enter__()
        except Exception as exc:
            self.status[server.name] = f'failed: {exc}'
            raise
        self._connections[key] = connection
        self.status[server.name] = 'connected'
        self._evict(keep=key)
        return connection

    def _evict(self, keep: tuple) -> None:
        try:
            limit = int(os.environ.get('SIGIL_MCP_MAX_CONNECTIONS', '16') or 16)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            limit = 16
        while limit > 0 and len(self._connections) > limit:
            candidates = sorted((k for k in list(self._connections) if k != keep),
                                key=lambda k: self._last_used.get(k, 0.0))
            for oldest in candidates:
                lock = self._lock_for(oldest)
                if lock.acquire(blocking=False):  # in use by another call: leave it, try the next
                    try:
                        self._drop(oldest)
                    finally:
                        lock.release()
                    break
            else:
                return  # every other connection is busy: keep them all for now

    def _drop(self, key: tuple) -> None:
        connection = self._connections.pop(key, None)
        if connection is not None:
            try:
                connection.close()
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)
            self.status[key[0]] = 'closed'

    def connection(self, server: MCPServerProfile, timeout_seconds: float = 10.0) -> _StdioMCPConnection:
        key = self._key(server)
        with self._lock_for(key):
            existing = self._connections.get(key)
            if self._alive(existing):
                return existing  # type: ignore[return-value]
            self._drop(key)
            return self._open(server, timeout_seconds)

    def request(self, server: MCPServerProfile, method: str, params: dict[str, Any], *,
                timeout_seconds: float = 10.0) -> dict[str, Any]:
        key = self._key(server)
        with self._lock_for(key):
            for attempt in (1, 2):
                connection = self._connections.get(key)
                if not self._alive(connection):
                    self._drop(key)
                    connection = self._open(server, timeout_seconds)
                connection.timeout_seconds = timeout_seconds
                self._last_used[key] = time.monotonic()
                try:
                    return connection.request(method, params)
                except MCPRequestError:
                    raise
                except (OSError, TimeoutError):
                    self._drop(key)
                    if attempt == 2 or method not in _IDEMPOTENT_METHODS:
                        raise
        raise OSError(f'MCP {method} failed for server {server.name}')

    def start_background(self, servers: tuple[MCPServerProfile, ...]) -> list[threading.Thread]:
        threads = []
        for server in servers:
            if server.transport not in SUPPORTED_TRANSPORTS or self._alive(self._connections.get(self._key(server))):
                continue
            self.status[server.name] = 'connecting'

            def _start(profile: MCPServerProfile = server) -> None:
                try:
                    self.connection(profile, 30.0)
                except Exception as _ignored:
                    from .debug_log import log_ignored_error

                    log_ignored_error(_ignored)  # status already says failed

            thread = threading.Thread(target=_start, name=f'mcp-start-{server.name}', daemon=True)
            thread.start()
            threads.append(thread)
        return threads

    def wait(self, names: list[str], timeout_seconds: float = 30.0) -> dict[str, str]:
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() < deadline:
            if not any(self.status.get(name) == 'connecting' for name in names):
                break
            time.sleep(0.1)
        return {name: self.status.get(name, 'not started') for name in names}

    def close_all(self) -> None:
        for key in list(self._connections):
            self._drop(key)


_CONNECTIONS = MCPConnectionManager()
atexit.register(_CONNECTIONS.close_all)


def connection_manager() -> MCPConnectionManager:
    return _CONNECTIONS


def _request_stdio(
    server: MCPServerProfile,
    method: str,
    params: dict[str, Any],
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    rotating = bool(server.rate_limits and server.rate_limits.get('keyRotation', {}).get('enabled'))
    if _persistent_enabled() and not rotating:
        return _CONNECTIONS.request(server, method, params, timeout_seconds=timeout_seconds)
    with _connection_class(server)(server, timeout_seconds=timeout_seconds) as connection:
        return connection.request(method, params)


def _render_resource_contents(contents: Any) -> str:
    if not isinstance(contents, list):
        return ''
    parts: list[str] = []
    for item in contents:
        if not isinstance(item, dict):
            continue
        text = item.get('text')
        if isinstance(text, str):
            parts.append(text)
            continue
        blob = item.get('blob')
        if isinstance(blob, str):
            mime_type = item.get('mimeType') if isinstance(item.get('mimeType'), str) else 'application/octet-stream'
            parts.append(f'[blob:{mime_type}] {blob}')
            continue
        parts.append(json.dumps(item, ensure_ascii=True, indent=2))
    return '\n\n'.join(parts).strip()


def _render_tool_call_result(result: dict[str, Any]) -> str:
    parts: list[str] = []
    content = result.get('content')
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get('text')
            if isinstance(text, str):
                parts.append(text)
                continue
            structured = item.get('structuredContent')
            if structured is not None:
                parts.append(json.dumps(structured, ensure_ascii=True, indent=2))
                continue
            parts.append(json.dumps(item, ensure_ascii=True, indent=2))
    structured_content = result.get('structuredContent')
    if structured_content is not None:
        parts.append(json.dumps(structured_content, ensure_ascii=True, indent=2))
    if not parts:
        parts.append(json.dumps(result, ensure_ascii=True, indent=2))
    return '\n\n'.join(part for part in parts if part).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f'{head}\n...[truncated]...\n{tail}'
