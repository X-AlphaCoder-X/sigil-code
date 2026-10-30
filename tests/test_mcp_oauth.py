"""MCP OAuth: discovery, client registration, PKCE browser login with a local
callback, bearer tokens, refresh on expiry and on 401, logout, /mcp login and
sigil mcp login. One fake server plays both the OAuth server and the MCP server."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src import mcp_oauth
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.main import main
from src.mcp_http import HttpMCPConnection
from src.mcp_runtime import MCPRuntime, connection_manager

REAL_URLOPEN = urllib.request.urlopen


class _State:
    def reset(self) -> None:
        self.registered: list[dict] = []
        self.challenges: dict[str, str] = {}
        self.valid_tokens = {'tok-1'}
        self.token_requests: list[dict] = []
        self.mcp_auth: list[str | None] = []
        self.refresh_tokens = {'refresh-1': 'tok-2'}
        self.deny = False
        self.prm_path = None  # serve protected-resource metadata only here (named in WWW-Authenticate)
        self.issuer_override = None
        self.s256 = True
        self.iss_override = None
        self.scopes: list[str] = []


STATE = _State()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet
        return

    def _json(self, payload: dict, status: int = 200, headers: dict | None = None) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        base = f'http://127.0.0.1:{self.server.server_address[1]}'
        parsed = urllib.parse.urlparse(self.path)
        prm = {'resource': base + '/mcp', 'authorization_servers': [base + '/auth']}
        if STATE.prm_path and parsed.path == STATE.prm_path:
            return self._json(prm)
        if parsed.path == '/.well-known/oauth-protected-resource' and not STATE.prm_path:
            return self._json(prm)
        if parsed.path == '/.well-known/oauth-authorization-server/auth':  # RFC 8414 path form
            return self._json({'issuer': STATE.issuer_override or base + '/auth',
                               'authorization_endpoint': base + '/auth/authorize',
                               'token_endpoint': base + '/auth/token', 'registration_endpoint': base + '/auth/register',
                               'code_challenge_methods_supported': ['S256'] if STATE.s256 else ['plain'],
                               'authorization_response_iss_parameter_supported': True})
        if parsed.path == '/auth/authorize':
            query = dict(urllib.parse.parse_qsl(parsed.query))
            assert query['code_challenge_method'] == 'S256'
            STATE.scopes.append(query.get('scope', ''))
            target = query['redirect_uri'] + '?' + urllib.parse.urlencode(
                {'error': 'access_denied', 'state': query['state'], 'iss': STATE.iss_override or base + '/auth'}
                if STATE.deny else  # RFC 9207: iss is on error responses too
                {'code': 'code-1', 'state': query['state'], 'iss': STATE.iss_override or base + '/auth'})
            STATE.challenges['code-1'] = query['code_challenge']
            self.send_response(302)
            self.send_header('Location', target)
            self.end_headers()
            return
        self._json({'error': 'not found'}, 404)

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
        if self.path == '/auth/register':
            body = json.loads(raw)
            STATE.registered.append(body)
            return self._json({'client_id': 'client-1', **body}, 201)
        if self.path == '/auth/token':
            form = dict(urllib.parse.parse_qsl(raw.decode('utf-8')))
            STATE.token_requests.append(form)
            if form['grant_type'] == 'authorization_code':
                digest = hashlib.sha256(form['code_verifier'].encode('ascii')).digest()
                expected = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
                if STATE.challenges.get(form['code']) != expected or form['client_id'] != 'client-1':
                    return self._json({'error': 'invalid_grant'}, 400)
                return self._json({'access_token': 'tok-1', 'refresh_token': 'refresh-1', 'expires_in': 3600,
                                   'token_type': 'Bearer'})
            new = STATE.refresh_tokens.get(form.get('refresh_token', ''))
            if not new:
                return self._json({'error': 'invalid_grant'}, 400)
            STATE.valid_tokens.add(new)
            return self._json({'access_token': new, 'expires_in': 3600, 'token_type': 'Bearer'})
        if self.path == '/mcp':
            auth = self.headers.get('Authorization')
            STATE.mcp_auth.append(auth)
            if not auth or auth.removeprefix('Bearer ') not in STATE.valid_tokens:
                base = f'http://127.0.0.1:{self.server.server_address[1]}'
                header = (f'Bearer resource_metadata="{base}{STATE.prm_path}", scope="mcp.read"'
                          if STATE.prm_path else 'Bearer')
                return self._json({'error': 'unauthorized'}, 401, {'WWW-Authenticate': header})
            message = json.loads(raw)
            if 'id' not in message:
                self.send_response(202)
                self.end_headers()
                return
            method = message.get('method')
            if method == 'initialize':
                result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'o'}}
            elif method == 'tools/list':
                result = {'tools': [{'name': 'whoami', 'description': 'Who', 'inputSchema': {'type': 'object'}}]}
            elif method == 'tools/call':
                result = {'content': [{'type': 'text', 'text': 'user:' + auth}], 'isError': False}
            else:
                result = {}
            return self._json({'jsonrpc': '2.0', 'id': message['id'], 'result': result})
        self._json({'error': 'not found'}, 404)

    def do_DELETE(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()


def follow_login(url: str) -> None:
    """Stands in for the browser: load the authorize page and follow its redirect."""
    threading.Thread(target=lambda: REAL_URLOPEN(url, timeout=10).read(), daemon=True).start()


class MCPOAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        STATE.reset()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(connection_manager().close_all)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        home = self.ws / 'home'
        home.mkdir()
        patcher = patch('src.settings._home', return_value=home)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.url = self.base + '/mcp'
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'web': {'type': 'http', 'url': self.url}}}),
                                           encoding='utf-8')

    def runtime(self) -> MCPRuntime:
        return MCPRuntime.from_workspace(self.ws)

    def store(self) -> dict:
        return json.loads(mcp_oauth.token_store_path().read_text(encoding='utf-8'))

    def login(self) -> str:
        return mcp_oauth.mcp_auth_command(self.runtime(), 'login', 'web', open_browser=follow_login,
                                          timeout_seconds=20, output=lambda _text: None)

    def test_login_registers_client_uses_pkce_and_saves_tokens(self) -> None:
        message = self.login()
        self.assertTrue(message.startswith('Logged in to web'), message)
        self.assertEqual(len(STATE.registered), 1)
        self.assertTrue(STATE.registered[0]['redirect_uris'][0].startswith('http://127.0.0.1:'))
        exchange = STATE.token_requests[0]
        self.assertEqual(exchange['grant_type'], 'authorization_code')
        self.assertEqual(exchange['resource'], self.url)
        saved = self.store()[self.url]
        self.assertEqual(saved['access_token'], 'tok-1')
        self.assertEqual(saved['refresh_token'], 'refresh-1')
        self.assertGreater(saved['expires_at'], time.time())
        tools = self.runtime().list_tools()
        self.assertEqual([t.name for t in tools], ['whoami'])
        self.assertIn('Bearer tok-1', STATE.mcp_auth)

    def test_configured_client_id_skips_registration(self) -> None:
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'web': {
            'type': 'http', 'url': self.url, 'oauth': {'client_id': 'client-1', 'scopes': 'read'}}}}),
            encoding='utf-8')
        self.assertEqual(self.runtime().servers[0].metadata['oauth']['client_id'], 'client-1')
        self.assertTrue(self.login().startswith('Logged in'))
        self.assertEqual(STATE.registered, [])

    def test_401_refreshes_token_and_retries(self) -> None:
        self.assertTrue(self.login().startswith('Logged in'))
        STATE.valid_tokens = set()  # the server revokes tok-1; refresh-1 still gives tok-2
        tools = self.runtime().list_tools()
        self.assertEqual([t.name for t in tools], ['whoami'])
        self.assertIn('Bearer tok-1', STATE.mcp_auth)
        self.assertEqual(STATE.mcp_auth[-1], 'Bearer tok-2')
        self.assertEqual(self.store()[self.url]['access_token'], 'tok-2')
        self.assertEqual(self.store()[self.url]['refresh_token'], 'refresh-1')  # kept when not rotated

    def test_expired_token_is_refreshed_before_use(self) -> None:
        self.assertTrue(self.login().startswith('Logged in'))
        data = self.store()
        data[self.url]['expires_at'] = time.time() - 5
        mcp_oauth._save_store(data)
        self.assertEqual(mcp_oauth.access_token(self.url), 'tok-2')
        self.assertEqual(STATE.token_requests[-1]['grant_type'], 'refresh_token')

    def test_no_login_gives_hint_and_logout_forgets(self) -> None:
        with self.assertRaises(OSError) as caught:
            with HttpMCPConnection(self.runtime().servers[0], timeout_seconds=5):
                pass
        self.assertIn('HTTP 401', str(caught.exception))
        self.assertIn('/mcp login web', str(caught.exception))
        self.assertEqual(mcp_oauth.mcp_auth_command(self.runtime(), 'logout', 'web'), 'No saved login for web')
        self.assertTrue(self.login().startswith('Logged in'))
        self.assertEqual(mcp_oauth.mcp_auth_command(self.runtime(), 'logout', 'web'), 'Logged out of web')
        self.assertNotIn(self.url, self.store())

    def test_denied_login_and_bad_state(self) -> None:
        STATE.deny = True
        message = self.login()
        self.assertIn('Login to web failed', message)
        self.assertIn('access_denied', message)
        STATE.deny = False

        def wrong_state(url: str) -> None:
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            target = query['redirect_uri'] + '?' + urllib.parse.urlencode(
                {'code': 'code-1', 'state': 'forged', 'iss': self.base + '/auth'})  # right issuer, forged state
            threading.Thread(target=lambda: REAL_URLOPEN(target, timeout=10).read(), daemon=True).start()

        message = mcp_oauth.mcp_auth_command(self.runtime(), 'login', 'web', open_browser=wrong_state,
                                             timeout_seconds=20, output=lambda _text: None)
        self.assertIn('wrong state', message)
        store = mcp_oauth.token_store_path()
        self.assertTrue(not store.exists() or self.url not in self.store())  # no tokens saved


    def test_spec_rules(self) -> None:
        # resource metadata named by the 401's WWW-Authenticate (and its scope) is used first
        STATE.prm_path = '/meta/prm'
        self.assertTrue(self.login().startswith('Logged in'), STATE.scopes)
        self.assertEqual(STATE.scopes[-1], 'mcp.read')
        self.assertEqual(STATE.registered[0]['application_type'], 'native')
        self.assertTrue(self.login().startswith('Logged in'))
        self.assertEqual(len(STATE.registered), 1)  # the client registered for this issuer is reused
        STATE.prm_path = None
        STATE.issuer_override = 'http://elsewhere.example'
        self.assertIn('names issuer', self.login())
        STATE.issuer_override = None
        STATE.s256 = False
        self.assertIn('PKCE S256', self.login())
        STATE.s256 = True
        STATE.iss_override = 'http://elsewhere.example'
        self.assertIn('expected', self.login())

    def test_issuer_and_iss_compared_exactly(self) -> None:
        auth = self.base + '/auth/'  # differs from the issuer only by a trailing slash
        STATE.issuer_override = auth
        self.assertIn('names issuer', self.login())
        STATE.issuer_override = None
        STATE.iss_override = auth
        self.assertIn('expected', self.login())
        STATE.deny = True  # a wrong iss is caught before the error response is used
        message = self.login()
        self.assertIn('expected', message)
        self.assertNotIn('access_denied', message)

    def test_stdio_and_unknown_servers(self) -> None:
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'local': {'command': 'x'}}}), encoding='utf-8')
        self.assertIn('only http/sse', mcp_oauth.mcp_auth_command(self.runtime(), 'login', 'local'))
        self.assertEqual(mcp_oauth.mcp_auth_command(self.runtime(), 'login', 'nope'), 'Unknown MCP server: nope')

    def test_cli_login_and_logout(self) -> None:
        out = io.StringIO()
        with patch('src.mcp_oauth.webbrowser.open', side_effect=follow_login), redirect_stdout(out):
            self.assertEqual(main(['mcp', 'login', 'web', '--cwd', str(self.ws)]), 0)
            self.assertEqual(main(['mcp', 'logout', 'web', '--cwd', str(self.ws)]), 0)
            self.assertEqual(main(['mcp', 'logout', 'web', '--cwd', str(self.ws)]), 1)
        text = out.getvalue()
        self.assertIn('Opening the login page for web', text)
        self.assertIn('Logged in to web', text)
        self.assertIn('Logged out of web', text)

    def test_slash_login_then_model_calls_protected_tool(self) -> None:
        responses = [
            {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'c1', 'type': 'function',
                'function': {'name': 'mcp__web__whoami', 'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}]},
        ]
        recorded: list[dict] = []

        def dispatch(request_obj, timeout=None):  # noqa: ANN001
            url = request_obj if isinstance(request_obj, str) else request_obj.full_url
            if url.startswith(self.base):  # the OAuth / MCP server is real; the model is faked
                return REAL_URLOPEN(request_obj, timeout=timeout)
            recorded.append(json.loads(request_obj.data.decode('utf-8')))
            return _FakeResponse(responses.pop(0))

        def make_agent() -> LocalCodingAgent:
            return LocalCodingAgent(
                model_config=ModelConfig(model='m'),
                runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.port_sessions' / 'agent',
                                                  scratchpad_root=self.ws / '.port_sessions' / 'scratchpad',
                                                  nudge_strategy='off'))

        agent = make_agent()  # start-up hits HTTP 401: not logged in yet
        self.assertNotIn('mcp__web__whoami', agent.tool_registry)
        with patch('src.mcp_oauth.webbrowser.open', side_effect=follow_login), redirect_stdout(io.StringIO()):
            result = agent.run('/mcp login web')
        self.assertIn('Logged in to web', result.final_output)
        self.assertEqual(connection_manager().status.get('web'), 'connected')
        self.assertIn('mcp__web__whoami', agent.tool_registry)  # registered without a restart
        self.assertIn('mcp__web__whoami', make_agent().tool_registry)  # and in the next session
        with patch('src.openai_compat.request.urlopen', side_effect=dispatch):
            agent.run('who am i on web?')
        tool_text = [str(m.get('content')) for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any('user:Bearer tok-1' in t for t in tool_text), tool_text)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> '_FakeResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


if __name__ == '__main__':
    unittest.main()
