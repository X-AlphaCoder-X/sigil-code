"""MCP over HTTP: Streamable HTTP (one POST per request) and the older SSE transport.

Server entries in .mcp.json / .sigil-mcp.json:
    {"type": "http", "url": "https://host/mcp", "headers": {"Authorization": "Bearer ..."}}
    {"type": "sse",  "url": "https://host/sse"}
Header values may use ${VAR} to read the environment. Responses may be plain
JSON or a ``text/event-stream`` body; the ``Mcp-Session-Id`` header is kept for
the session.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import urllib.parse
from typing import Any
from urllib import error, request
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

MCP_PROTOCOL_VERSION = '2025-06-18'
_ENV_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


# Sigil's own model keys read as empty in a remote server's URL or headers: a manifest naming
# ${SIGIL_API_KEY} or ${OPENAI_API_KEY} would send the user's model key to that server.
_OWN_KEYS = ('SIGIL_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN')


def _env_value(name: str) -> str:
    if name.upper() in _OWN_KEYS or name.upper().startswith('SIGIL_API_KEY_'):
        return ''
    return os.environ.get(name, '')


def expand_env(value: str) -> str:
    return _ENV_RE.sub(lambda m: _env_value(m.group(1)), value)


def _origin(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f'{parts.scheme.lower()}://{(parts.hostname or "").lower()}:{parts.port or ""}'


def _sse_limit() -> int:
    try:
        return max(int(os.environ.get('SIGIL_MCP_SSE_MAX_CHARS', '') or 10_000_000), 1)
    except ValueError as exc:
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
        return 10_000_000


def _sse_events(lines: Any) -> Any:
    """Yield (event, data) pairs from an iterable of SSE lines (bytes or str)."""
    event, data = 'message', []
    limit = _sse_limit()
    size = 0
    for raw in lines:
        line = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if not line:
            if data and size <= limit:
                yield event, '\n'.join(data)
            elif data:
                from .debug_log import get_logger

                get_logger('mcp').warning('SSE event over SIGIL_MCP_SSE_MAX_CHARS (%d chars) dropped', size)
            event, data, size = 'message', [], 0
            continue
        if line.startswith(':'):
            continue
        field, _, value = line.partition(':')
        value = value[1:] if value.startswith(' ') else value
        if field == 'event':
            event = value
        elif field == 'data':
            size += len(value)
            if size <= limit:  # past the cap the event is not kept (and is dropped when it ends)
                data.append(value)
    if data:
        yield event, '\n'.join(data)


class HttpMCPConnection:
    """Streamable HTTP. Matches _StdioMCPConnection: request(), close(), alive()."""

    def __init__(self, server: Any, *, timeout_seconds: float = 30.0) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds
        self.session_id: str | None = None
        self._request_id = 0
        self._open = False

    def _headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream',
                   'MCP-Protocol-Version': MCP_PROTOCOL_VERSION}
        for key, value in (getattr(self.server, 'headers', None) or {}).items():
            headers[str(key)] = expand_env(str(value))
        if not any(key.lower() == 'authorization' for key in headers):
            from .mcp_oauth import access_token

            token = access_token(self.server.url)
            if token:
                headers['Authorization'] = f'Bearer {token}'
        if self.session_id:
            headers['Mcp-Session-Id'] = self.session_id
        return headers

    def _auth_failed(self, exc: error.HTTPError, retried: bool) -> bool:
        """True when a 401 was fixed by refreshing the saved login (so retry once)."""
        if exc.code != 401 or retried:
            return False
        from .mcp_oauth import refresh

        return refresh(self.server.url) is not None

    def _login_hint(self, exc: error.HTTPError) -> str:
        if exc.code != 401:
            return ''
        return f' (log in with /mcp login {self.server.name})'

    def _post(self, payload: dict[str, Any], *, expect_reply: bool, _retried: bool = False) -> dict[str, Any] | None:
        req = request.Request(self.server.url, data=json.dumps(payload).encode('utf-8'),
                              headers=self._headers(), method='POST')
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                session = response.headers.get('Mcp-Session-Id') if response.headers else None
                if session:
                    self.session_id = session
                kind = (response.headers.get('Content-Type') or '') if response.headers else ''
                if not expect_reply:
                    return None
                if 'text/event-stream' in kind:
                    for _event, data in _sse_events(iter(response.readline, b'')):
                        try:
                            message = json.loads(data)
                        except ValueError as _ignored:
                            _log_ignored(_ignored)
                            continue
                        if isinstance(message, dict) and 'method' in message:
                            if 'id' in message:  # the server asks us something (elicitation/create)
                                from .mcp_runtime import answer_server_request

                                self._post(answer_server_request(self.server.name, message), expect_reply=False)
                            continue
                        if isinstance(message, dict) and message.get('id') == payload.get('id'):
                            return message
                    raise OSError(f'MCP server {self.server.name} ended the stream without a reply')
                body = response.read().decode('utf-8', errors='replace')
        except error.HTTPError as exc:
            if self._auth_failed(exc, _retried):
                return self._post(payload, expect_reply=expect_reply, _retried=True)
            detail = exc.read().decode('utf-8', errors='replace')[:500]
            raise OSError(f'MCP server {self.server.name} returned HTTP {exc.code}: {detail}{self._login_hint(exc)}') from exc
        except error.URLError as exc:
            raise OSError(f'Cannot reach MCP server {self.server.name} at {self.server.url}: {exc.reason}') from exc
        if not body.strip():
            return {}
        message = json.loads(body)
        if isinstance(message, list):
            message = next((m for m in message if isinstance(m, dict) and m.get('id') == payload.get('id')), {})
        return message if isinstance(message, dict) else {}

    def __enter__(self) -> 'HttpMCPConnection':
        self._request_id += 1
        reply = self._post({'jsonrpc': '2.0', 'id': self._request_id, 'method': 'initialize', 'params': {
            'protocolVersion': MCP_PROTOCOL_VERSION, 'capabilities': {'elicitation': {}},
            'clientInfo': {'name': 'sigilcode', 'version': '1.0.0'}}}, expect_reply=True)
        if reply and isinstance(reply.get('error'), dict):
            raise OSError(f'MCP initialize failed for server {self.server.name}: {reply["error"].get("message")}')
        self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}}, expect_reply=False)
        self._open = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def process(self) -> Any:
        return self if self._open else None

    def poll(self) -> None:
        return None  # "running" for the connection manager's liveness check

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from .mcp_runtime import MCPRequestError

        self._request_id += 1
        reply = self._post({'jsonrpc': '2.0', 'id': self._request_id, 'method': method, 'params': params},
                           expect_reply=True) or {}
        if isinstance(reply.get('error'), dict):
            raise MCPRequestError(f'MCP {method} failed for server {self.server.name}: {reply["error"].get("message")}')
        result = reply.get('result')
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        if self._open and self.session_id:
            try:
                req = request.Request(self.server.url, headers=self._headers(), method='DELETE')
                request.urlopen(req, timeout=5).close()
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)
        self._open = False


class SseMCPConnection(HttpMCPConnection):
    """The older HTTP+SSE transport: GET opens a stream that names a POST endpoint."""

    def __init__(self, server: Any, *, timeout_seconds: float = 30.0) -> None:
        super().__init__(server, timeout_seconds=timeout_seconds)
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10000)  # bounded (back-pressure)
        self._endpoint: str | None = None
        self._stream: Any = None
        self._reader: threading.Thread | None = None

    def _read(self) -> None:
        try:
            for event, data in _sse_events(iter(self._stream.readline, b'')):
                if event == 'endpoint':
                    endpoint = urllib.parse.urljoin(self.server.url, data.strip())
                    if _origin(endpoint) != _origin(self.server.url):
                        # the POSTs carry the server's headers and OAuth token: never to another origin
                        self.endpoint_error = f'endpoint on another origin refused: {_origin(endpoint)}'
                        continue
                    self._endpoint = endpoint
                    continue
                try:
                    message = json.loads(data)
                except ValueError as _ignored:
                    _log_ignored(_ignored)
                    continue
                if isinstance(message, dict):
                    self._messages.put(message)
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
        finally:
            self._open = False

    def __enter__(self) -> 'SseMCPConnection':
        headers = {k: v for k, v in self._headers().items() if k != 'Content-Type'}
        headers['Accept'] = 'text/event-stream'
        try:
            self._stream = request.urlopen(request.Request(self.server.url, headers=headers),
                                           timeout=self.timeout_seconds)
        except (error.URLError, OSError) as exc:
            raise OSError(f'Cannot open SSE stream for MCP server {self.server.name}: {exc}') from exc
        self._open = True
        self._reader = threading.Thread(target=self._read, name=f'mcp-sse-{self.server.name}', daemon=True)
        self._reader.start()
        deadline = time.monotonic() + self.timeout_seconds
        while self._endpoint is None and time.monotonic() < deadline and self._open:
            time.sleep(0.02)
        if self._endpoint is None and getattr(self, 'endpoint_error', None):
            raise OSError(f'MCP server {self.server.name}: {self.endpoint_error}')
        if self._endpoint is None:
            self.close()
            raise OSError(f'MCP server {self.server.name} sent no endpoint event')
        self._request_id += 1
        reply = self._call({'jsonrpc': '2.0', 'id': self._request_id, 'method': 'initialize', 'params': {
            'protocolVersion': '2024-11-05', 'capabilities': {},
            'clientInfo': {'name': 'sigilcode', 'version': '1.0.0'}}})
        if isinstance(reply.get('error'), dict):
            raise OSError(f'MCP initialize failed for server {self.server.name}: {reply["error"].get("message")}')
        self._send({'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})
        return self

    def _send(self, payload: dict[str, Any], _retried: bool = False) -> None:
        req = request.Request(self._endpoint or self.server.url, data=json.dumps(payload).encode('utf-8'),
                              headers=self._headers(), method='POST')
        try:
            request.urlopen(req, timeout=self.timeout_seconds).close()
        except error.HTTPError as exc:
            if self._auth_failed(exc, _retried):
                return self._send(payload, _retried=True)
            raise OSError(f'MCP server {self.server.name} returned HTTP {exc.code}{self._login_hint(exc)}') from exc

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._send(payload)
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                if not self._open:
                    raise OSError(f'MCP server {self.server.name} closed the SSE stream')
                continue
            if message.get('id') == payload.get('id'):
                return message
        raise TimeoutError(f'Timed out waiting for MCP response from {self.server.name}')

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        from .mcp_runtime import MCPRequestError

        self._request_id += 1
        reply = self._call({'jsonrpc': '2.0', 'id': self._request_id, 'method': method, 'params': params})
        if isinstance(reply.get('error'), dict):
            raise MCPRequestError(f'MCP {method} failed for server {self.server.name}: {reply["error"].get("message")}')
        result = reply.get('result')
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        self._open = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)
