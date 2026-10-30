"""OAuth login for remote (HTTP / SSE) MCP servers.

Flow (MCP authorization spec, OAuth 2.1 with PKCE):
  1. Find the authorization server: <origin>/.well-known/oauth-protected-resource,
     then <issuer>/.well-known/oauth-authorization-server (fallback: the origin).
  2. Register a client (dynamic client registration) unless the server entry has
     "oauth": {"client_id": "..."}.
  3. Open the browser at the authorization endpoint (code + S256 PKCE) and catch
     the redirect on a one-shot local server at http://127.0.0.1:<port>/callback.
  4. Exchange the code for tokens and save them in ~/.sigil/mcp-oauth.json.
Requests then send "Authorization: Bearer <token>"; an expired token, or a 401,
is refreshed with the refresh token. Log in with /mcp login <server> or
`sigil mcp login <server>`; /mcp logout <server> forgets the tokens.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from .debug_log import get_logger, url_for_log
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

CALLBACK_PATH = '/callback'
_STORE_LOCK = threading.RLock()  # parallel MCP calls may refresh at the same time
_log = get_logger('mcp_oauth')


def token_store_path() -> Path:
    from .settings import _home

    return _home() / '.sigil' / 'mcp-oauth.json'


def _load_store() -> dict[str, Any]:
    try:
        from .settings import read_text_retry

        data = json.loads(read_text_retry(token_store_path()))
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return {}
    return data if isinstance(data, dict) else {}


def _save_store(data: dict[str, Any]) -> None:
    path = token_store_path()
    from .settings import write_text_atomic

    write_text_atomic(path, json.dumps(data, indent=2))  # several Sigil processes refresh tokens
    try:
        os.chmod(path, 0o600)  # owner only (no effect on Windows ACLs)
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass


def stored_tokens(server_url: str) -> dict[str, Any] | None:
    entry = _load_store().get(server_url)
    return entry if isinstance(entry, dict) else None


def _update_store(server_url: str, entry: dict[str, Any] | None) -> None:
    with _STORE_LOCK:
        data = _load_store()
        if entry is None:
            data.pop(server_url, None)
        else:
            data[server_url] = entry
        _save_store(data)


def forget_tokens(server_url: str) -> bool:
    with _STORE_LOCK:
        if server_url not in _load_store():
            return False
        _update_store(server_url, None)
    _log.info('forgot OAuth tokens for %s', url_for_log(server_url))
    return True


_LOOPBACK_HOSTS = frozenset({'127.0.0.1', '::1', 'localhost'})


def _check_oauth_url(url: str, what: str) -> None:
    """Refuse an OAuth endpoint URL that is not https (http only for loopback). The URLs come from
    the server's own metadata / WWW-Authenticate, so a hostile server could otherwise name a
    file:// or a plain-http collector. Override: SIGIL_MCP_ALLOW_INSECURE_OAUTH=1 (fail closed)."""
    if os.environ.get('SIGIL_MCP_ALLOW_INSECURE_OAUTH', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        return
    parsed = urllib.parse.urlparse(str(url))
    scheme = parsed.scheme.lower()
    if scheme == 'https':
        return
    host = (parsed.hostname or '').lower()
    if scheme == 'http' and host in _LOOPBACK_HOSTS:
        return
    raise OSError(f'{what} must be https (or http on localhost): refused {url!r} '
                  '(set SIGIL_MCP_ALLOW_INSECURE_OAUTH=1 to allow plain http)')


def _get_json(url: str, timeout: float = 15.0) -> dict[str, Any] | None:
    try:
        _check_oauth_url(url, 'OAuth metadata endpoint')
    except OSError as _ignored:
        _log_ignored(_ignored)
        return None
    try:
        with request.urlopen(request.Request(url, headers={'Accept': 'application/json'}), timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except (error.URLError, OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None
    return data if isinstance(data, dict) else None


def _post(url: str, data: dict[str, Any], *, as_json: bool = False, timeout: float = 15.0) -> dict[str, Any]:
    if as_json:
        body, kind = json.dumps(data).encode('utf-8'), 'application/json'
    else:
        body, kind = urllib.parse.urlencode(data).encode('utf-8'), 'application/x-www-form-urlencoded'
    _check_oauth_url(url, 'OAuth endpoint')
    req = request.Request(url, data=body, method='POST', headers={'Content-Type': kind, 'Accept': 'application/json'})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode('utf-8') or '{}')
    except error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:400]
        raise OSError(f'OAuth request to {url} failed: HTTP {exc.code} {detail}') from exc
    except (error.URLError, OSError) as exc:
        raise OSError(f'OAuth request to {url} failed: {exc}') from exc
    if not isinstance(payload, dict):
        raise OSError(f'OAuth endpoint {url} returned a non-object reply')
    return payload


def _probe_www_authenticate(server_url: str) -> str:
    """The WWW-Authenticate header of an unauthenticated MCP request (a 401), or ''."""
    body = json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
        'protocolVersion': '2025-06-18', 'capabilities': {}, 'clientInfo': {'name': 'sigilcode', 'version': '1.0.0'}}})
    req = request.Request(server_url, data=body.encode('utf-8'), method='POST', headers={
        'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'})
    try:
        with request.urlopen(req, timeout=15) as response:
            response.read()
        return ''
    except error.HTTPError as exc:
        return (exc.headers.get('WWW-Authenticate') or '') if exc.code == 401 and exc.headers else ''
    except (error.URLError, OSError) as _ignored:
        _log_ignored(_ignored)
        return ''


def _auth_param(header: str, name: str) -> str | None:
    import re

    match = re.search(rf'{name}\s*=\s*(?:"([^"]*)"|([^\s,]+))', header or '')
    return (match.group(1) if match.group(1) is not None else match.group(2)) if match else None


def discover(server_url: str, www_authenticate: str | None = None) -> dict[str, Any]:
    """Authorization server metadata for an MCP server URL (MCP authorization spec).

    Protected resource metadata comes from the 401's WWW-Authenticate resource_metadata when
    there is one, else /.well-known/oauth-protected-resource (path form, then root). The
    authorization server metadata must exist, its issuer must match, and it must offer S256 PKCE.
    """
    header = _probe_www_authenticate(server_url) if www_authenticate is None else www_authenticate
    parsed = urllib.parse.urlparse(server_url)
    origin = f'{parsed.scheme}://{parsed.netloc}'
    path = parsed.path.rstrip('/')
    candidates = [url for url in [_auth_param(header, 'resource_metadata')] if url]
    candidates += ([origin + '/.well-known/oauth-protected-resource' + path] if path else []) + \
        [origin + '/.well-known/oauth-protected-resource']
    resource = None
    for candidate in candidates:
        resource = _get_json(candidate)
        if resource:
            break
    issuer = origin  # the issuer identifier exactly as given; the URLs below are built from it
    if resource and isinstance(resource.get('authorization_servers'), list) and resource['authorization_servers']:
        issuer = str(resource['authorization_servers'][0])
    issuer_parsed = urllib.parse.urlparse(issuer)
    issuer_origin = f'{issuer_parsed.scheme}://{issuer_parsed.netloc}'
    issuer_path = issuer_parsed.path.rstrip('/')
    order = ([issuer_origin + '/.well-known/oauth-authorization-server' + issuer_path,
              issuer_origin + '/.well-known/openid-configuration' + issuer_path,
              issuer + '/.well-known/openid-configuration'] if issuer_path else
             [issuer_origin + '/.well-known/oauth-authorization-server',
              issuer_origin + '/.well-known/openid-configuration'])
    metadata = None
    for candidate in order:
        metadata = _get_json(candidate)
        if metadata and metadata.get('authorization_endpoint') and metadata.get('token_endpoint'):
            break
        metadata = None
    _log.info('OAuth discovery for %s: issuer %s (%s)', url_for_log(server_url), url_for_log(issuer),
              'found' if metadata else 'none')
    if not metadata:
        raise OSError(f'No OAuth authorization server metadata for {issuer} (tried {", ".join(order)})')
    if metadata.get('issuer') != issuer:  # RFC 8414: identical, no normalization
        raise OSError(f'Authorization server metadata names issuer {metadata.get("issuer")!r}, not {issuer!r}')
    methods = metadata.get('code_challenge_methods_supported')
    if not isinstance(methods, list) or 'S256' not in methods:
        raise OSError(f'{issuer} does not offer PKCE S256 (code_challenge_methods_supported); not logging in')
    scope = _auth_param(header, 'scope')
    if not scope and resource and isinstance(resource.get('scopes_supported'), list):
        scope = ' '.join(str(s) for s in resource['scopes_supported'])
    return {**metadata, '_issuer': issuer, '_scope': scope or ''}


def _pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode('ascii')
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).rstrip(b'=').decode('ascii')
    return verifier, challenge


class _CallbackServer(HTTPServer):
    result: dict[str, str]


class _CallbackHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        self.server.result.update(query)  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        # why (S08): the callback page carries the authorization code; lock it down so nothing is
        # cached, sniffed into another type, loaded as a sub-resource, or sent on in a referrer
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'none'")
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(b'<html><body>Login finished. You can close this tab and go back to the terminal.</body></html>')


def login(server: Any, *, open_browser: Callable[[str], Any] | None = None, timeout_seconds: float = 300.0,
          output: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the browser login for an HTTP/SSE MCP server and save its tokens."""
    server_url = server.url
    oauth = server.metadata.get('oauth') if isinstance(getattr(server, 'metadata', None), dict) else None
    oauth = oauth if isinstance(oauth, dict) else {}
    metadata = discover(server_url)
    issuer = metadata['_issuer']
    client_id = oauth.get('client_id')
    client_secret = oauth.get('client_secret')
    known = (_load_store().get('_clients') or {}).get(issuer) if not client_id else None
    port = int(oauth.get('callback_port') or 0)
    if isinstance(known, dict) and known.get('client_id') and not port:
        # reuse the client registered for this issuer, on the port it was registered with
        registered_port = urllib.parse.urlparse((known.get('redirect_uris') or [''])[0]).port
        try:
            callback = _CallbackServer(('127.0.0.1', int(registered_port or 0)), _CallbackHandler)
            client_id, client_secret = known['client_id'], known.get('client_secret')
        except OSError as _ignored:
            _log_ignored(_ignored)
            callback = _CallbackServer(('127.0.0.1', 0), _CallbackHandler)  # port taken: register again
    else:
        callback = _CallbackServer(('127.0.0.1', port), _CallbackHandler)
    callback.result = {}
    redirect_uri = f'http://127.0.0.1:{callback.server_address[1]}{CALLBACK_PATH}'
    if not client_id:
        endpoint = metadata.get('registration_endpoint')
        if not endpoint:
            callback.server_close()
            raise OSError(f'{server.name} needs "oauth": {{"client_id": ...}} (the server has no client registration)')
        registered = _post(str(endpoint), {
            'client_name': 'Sigil Code', 'application_type': 'native', 'redirect_uris': [redirect_uri],
            'grant_types': ['authorization_code', 'refresh_token'], 'response_types': ['code'],
            'token_endpoint_auth_method': 'none'}, as_json=True)
        client_id = registered.get('client_id')
        client_secret = registered.get('client_secret') or client_secret
        if not client_id:
            callback.server_close()
            raise OSError('client registration returned no client_id')
        with _STORE_LOCK:  # clients are kept per issuer
            data = _load_store()
            data.setdefault('_clients', {})[issuer] = {'client_id': client_id, 'client_secret': client_secret,
                                                       'redirect_uris': [redirect_uri]}
            _save_store(data)
    verifier, challenge = _pkce()
    state = secrets.token_urlsafe(16)
    params = {'response_type': 'code', 'client_id': client_id, 'redirect_uri': redirect_uri,
              'code_challenge': challenge, 'code_challenge_method': 'S256', 'state': state, 'resource': server_url}
    scope = oauth.get('scopes') or oauth.get('scope') or metadata.get('_scope')
    if scope:
        params['scope'] = str(scope)
    _check_oauth_url(str(metadata['authorization_endpoint']), 'authorization endpoint')
    url = str(metadata['authorization_endpoint']) + '?' + urllib.parse.urlencode(params)
    thread = threading.Thread(target=callback.serve_forever, daemon=True)
    thread.start()
    try:
        output(f'Opening the login page for {server.name}. If no browser opens, visit:\n{url}')
        (open_browser or webbrowser.open)(url)
        deadline = time.monotonic() + timeout_seconds
        while 'code' not in callback.result and 'error' not in callback.result:
            if time.monotonic() > deadline:
                raise OSError(f'No login callback within {int(timeout_seconds)} seconds')
            time.sleep(0.1)
    finally:
        callback.shutdown()
        callback.server_close()
    result = callback.result
    # RFC 9207, checked first and compared exactly: on a mismatch nothing in the response is used
    returned_issuer = result.get('iss')
    if returned_issuer is not None and returned_issuer != metadata['issuer']:
        raise OSError(f'Login callback came from issuer {returned_issuer!r}, expected {metadata["issuer"]!r}')
    if returned_issuer is None and metadata.get('authorization_response_iss_parameter_supported'):
        raise OSError('Login callback is missing the iss parameter the authorization server promised')
    if result.get('error'):
        raise OSError(f'Login refused: {result.get("error")} {result.get("error_description", "")}'.strip())
    if result.get('state') != state:
        raise OSError('Login callback had the wrong state (possible CSRF); try again')
    form = {'grant_type': 'authorization_code', 'code': result['code'], 'redirect_uri': redirect_uri,
            'client_id': client_id, 'code_verifier': verifier, 'resource': server_url}
    if client_secret:
        form['client_secret'] = client_secret
    tokens = _post(str(metadata['token_endpoint']), form)
    if not tokens.get('access_token'):
        raise OSError('The token endpoint returned no access_token')
    entry = {
        'access_token': tokens['access_token'],
        'refresh_token': tokens.get('refresh_token'),
        'expires_at': time.time() + float(tokens['expires_in']) if tokens.get('expires_in') else None,
        'token_endpoint': metadata['token_endpoint'],
        'issuer': issuer,
        'client_id': client_id,
        'client_secret': client_secret,
    }
    _update_store(server_url, entry)
    _log.info('OAuth login saved for %s (client %s)', server.name, client_id)
    return entry


_REFRESH_LOCKS: dict[str, threading.Lock] = {}
_REFRESH_LOCKS_GUARD = threading.Lock()


def refresh(server_url: str) -> dict[str, Any] | None:
    """Refresh the access token; one refresh per server at a time. A caller that waited while
    another thread refreshed gets that fresh token instead of spending the (single-use, rotated)
    refresh token a second time."""
    with _REFRESH_LOCKS_GUARD:
        lock = _REFRESH_LOCKS.setdefault(server_url, threading.Lock())
    before = (stored_tokens(server_url) or {}).get('refresh_token')
    with lock:
        current = stored_tokens(server_url)
        if current and current.get('refresh_token') and current.get('refresh_token') != before:
            return current  # refreshed by another thread while this one waited
        return _refresh_locked(server_url)


def _refresh_locked(server_url: str) -> dict[str, Any] | None:
    entry = stored_tokens(server_url)
    if not entry or not entry.get('refresh_token'):
        return None
    form = {'grant_type': 'refresh_token', 'refresh_token': entry['refresh_token'], 'client_id': entry.get('client_id'),
            'resource': server_url}
    if entry.get('client_secret'):
        form['client_secret'] = entry['client_secret']
    try:
        tokens = _post(str(entry['token_endpoint']), form)
    except OSError as exc:
        _log.warning('OAuth refresh failed for %s: %s', url_for_log(server_url), exc)
        return None
    if not tokens.get('access_token'):
        _log.warning('OAuth refresh for %s returned no access_token', url_for_log(server_url))
        return None
    entry.update({
        'access_token': tokens['access_token'],
        'refresh_token': tokens.get('refresh_token') or entry.get('refresh_token'),
        'expires_at': time.time() + float(tokens['expires_in']) if tokens.get('expires_in') else None,
    })
    _update_store(server_url, entry)
    _log.info('OAuth token refreshed for %s', url_for_log(server_url))
    return entry


def access_token(server_url: str) -> str | None:
    """A usable token for this server (refreshed when it is about to expire), or None."""
    entry = stored_tokens(server_url)
    if not entry:
        return None
    expires = entry.get('expires_at')
    if isinstance(expires, (int, float)) and expires - 30 < time.time():
        entry = refresh(server_url)
    return entry.get('access_token') if entry else None


def mcp_auth_command(runtime: Any, verb: str, name: str, **login_kwargs: Any) -> str:
    """Shared by /mcp login|logout and `sigil mcp login|logout`."""
    server = runtime.get_server(name) if runtime is not None else None
    if server is None:
        return f'Unknown MCP server: {name}'
    if not getattr(server, 'url', None):
        return f'{name} is a local (stdio) server; only http/sse servers use OAuth login'
    if hasattr(runtime, '_tools_cache'):
        runtime._tools_cache = None  # the tool list cached before the login (or logout) is stale
    if verb == 'logout':
        return f'Logged out of {name}' if forget_tokens(server.url) else f'No saved login for {name}'
    try:
        login(server, **login_kwargs)
    except OSError as exc:
        _log.warning('OAuth login to %s failed: %s', name, exc)
        return f'Login to {name} failed: {exc}'
    from .mcp_runtime import connection_manager

    manager = connection_manager()  # reconnect now: clears a "failed: HTTP 401" from start-up
    manager._drop(manager._key(server))
    try:
        manager.connection(server, 30.0)
    except Exception as exc:
        return f'Logged in to {name}, but connecting still failed: {exc}'
    return f'Logged in to {name}; the token is saved in {token_store_path()}'
