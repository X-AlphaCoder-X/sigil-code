"""Batch 17b (P3): /plugin marketplaces (git), OpenTelemetry export, Windows sandbox limits."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig

REAL_URLOPEN = urllib.request.urlopen
BASE = 'http://127.0.0.1:8000/v1'


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason':
                                             'tool_calls' if message.get('tool_calls') else 'stop'}],
                                'usage': {'prompt_tokens': 120, 'completion_tokens': 7}}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def git(*args: str, cwd: Path) -> None:
    subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=a', *args], cwd=cwd, check=True,
                   capture_output=True)


class MarketplaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.home = self.base / 'home'
        self.home.mkdir()
        patcher = patch('src.settings._home', return_value=self.home)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {'SIGIL_PLUGIN_DIRS': '', 'SIGIL_SAFE_MODE': ''})
        env.start()
        self.addCleanup(env.stop)

    def make_market(self, folder: Path, use_git: bool = True) -> None:
        plugin = folder / 'plugins' / 'fmt'
        (plugin / '.sigil-plugin').mkdir(parents=True)
        (plugin / '.sigil-plugin' / 'plugin.json').write_text('{"name": "fmt", "version": "0.2.0"}', encoding='utf-8')
        (plugin / 'commands').mkdir()
        (plugin / 'commands' / 'hello.md').write_text('Say hello from fmt', encoding='utf-8')
        (folder / '.sigil-plugin').mkdir()
        (folder / '.sigil-plugin' / 'marketplace.json').write_text(json.dumps({'name': 'team-tools', 'plugins': [
            {'name': 'fmt', 'source': './plugins/fmt', 'version': '0.2.0'},
            {'name': 'escape', 'source': '../outside'}]}), encoding='utf-8')
        if use_git:
            git('init', '-q', cwd=folder)
            git('add', '.', cwd=folder)
            git('commit', '-q', '-m', 'market', cwd=folder)

    def test_add_install_load_uninstall(self) -> None:
        from src.sigil_plugins import plugin_dirs
        from src.custom_commands import discover_commands
        from src.plugin_marketplace import command, root

        market = self.base / 'market'
        market.mkdir()
        self.make_market(market)
        self.assertIn('Added marketplace team-tools (2 plugin(s))', command(f'marketplace add {market}'))
        self.assertTrue((root() / 'marketplaces' / 'team-tools' / '.git').exists())  # cloned
        self.assertIn('team-tools: fmt, escape', command('marketplace list'))
        self.assertIn('Installed fmt@team-tools', command('install fmt'))
        installed = root() / 'installed' / 'fmt@team-tools'
        self.assertTrue((installed / 'commands' / 'hello.md').exists())
        self.assertIn(installed.resolve(), plugin_dirs())
        commands = discover_commands(self.base)
        self.assertIn('fmt:hello', commands)
        self.assertIn('outside the marketplace', command('install escape'))
        self.assertIn('No plugin', command('install nothing'))
        out = io.StringIO()
        from src.main import main

        with redirect_stdout(out):
            self.assertEqual(main(['plugin', 'list']), 0)
        self.assertIn('fmt@team-tools', out.getvalue())
        self.assertIn('Uninstalled fmt@team-tools', command('uninstall fmt'))
        self.assertNotIn('fmt:hello', discover_commands(self.base))
        self.assertIn('Removed marketplace team-tools', command('marketplace remove team-tools'))

    def test_plain_folder_marketplace_and_slash_command(self) -> None:
        market = self.base / 'local-market'
        market.mkdir()
        self.make_market(market, use_git=False)
        with tempfile.TemporaryDirectory() as ws:
            agent = LocalCodingAgent(model_config=ModelConfig(model='m'),
                                     runtime_config=AgentRuntimeConfig(cwd=Path(ws), session_directory=Path(ws) / 's'))
            self.assertIn(str(market), agent.run(f'/plugin marketplace add {market}').final_output)
            self.assertIn('Installed fmt@team-tools', agent.run('/plugin install fmt@team-tools').final_output)
            self.assertIn('Usage: /plugin', agent.run('/plugin frobnicate').final_output)


class _Collector(BaseHTTPRequestHandler):
    received: list = []

    def log_message(self, *args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        type(self).received.append((self.path, dict(self.headers),
                                    json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'{}')


class TelemetryTests(unittest.TestCase):
    def test_model_and_tool_records_reach_the_collector(self) -> None:
        from src import telemetry

        _Collector.received = []
        server = ThreadingHTTPServer(('127.0.0.1', 0), _Collector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        telemetry._PENDING.clear()
        replies = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't1', 'type': 'function', 'function': {
            'name': 'list_dir', 'arguments': '{}'}}]}, {'role': 'assistant', 'content': 'done'}]

        def fake(req, timeout=None):  # noqa: ANN001
            if not req.full_url.endswith('/chat/completions'):
                return REAL_URLOPEN(req, timeout=timeout)
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as ws, patch('src.openai_compat.request.urlopen', side_effect=fake), \
                patch.dict(os.environ, {'SIGIL_ENABLE_TELEMETRY': '1', 'SIGIL_OTEL_INTERVAL': '3600',
                                        'OTEL_EXPORTER_OTLP_ENDPOINT': f'http://127.0.0.1:{server.server_address[1]}',
                                        'OTEL_EXPORTER_OTLP_HEADERS': 'x-team=tools'}):
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                     runtime_config=AgentRuntimeConfig(cwd=Path(ws), session_directory=Path(ws) / 's',
                                                                       nudge_strategy='off'))
            agent.run('look around')
            self.assertEqual(telemetry.flush(), 3)
        path, headers, payload = _Collector.received[-1]
        self.assertEqual(path, '/v1/logs')
        self.assertEqual({k.lower(): v for k, v in headers.items()}.get('x-team'), 'tools')
        records = payload['resourceLogs'][0]['scopeLogs'][0]['logRecords']
        names = [r['body']['stringValue'] for r in records]
        self.assertEqual(sorted(names), sorted(['sigil.model_request', 'sigil.tool_result', 'sigil.model_request']))
        usage = next(r for r in records if r['body']['stringValue'] == 'sigil.model_request')
        attrs = {a['key']: a['value'] for a in usage['attributes']}
        self.assertEqual(attrs['input_tokens'], {'intValue': '120'})
        self.assertEqual(attrs['model'], {'stringValue': 'm'})
        self.assertNotIn('look around', json.dumps(payload))  # no prompt text leaves the machine

    def test_off_by_default(self) -> None:
        from src import telemetry

        telemetry._PENDING.clear()
        with patch.dict(os.environ, {'SIGIL_ENABLE_TELEMETRY': '', 'OTEL_EXPORTER_OTLP_ENDPOINT': ''}):
            os.environ.pop('OTEL_EXPORTER_OTLP_ENDPOINT', None)
            telemetry.record({'type': 'model_usage', 'model': 'm'})
        self.assertEqual(telemetry._PENDING, [])


class WindowsSandboxTests(unittest.TestCase):
    def test_limits_only_when_on(self) -> None:
        from src.sandbox import windows_limits

        with patch.dict(os.environ, {'SIGIL_SANDBOX': ''}):
            self.assertIsNone(windows_limits())
        with patch.dict(os.environ, {'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_MEMORY_MB': '512'}), \
                patch('src.sandbox._platform', return_value='win32'):
            limits = windows_limits()
        self.assertEqual(limits['memory_bytes'], 512 * 1024 * 1024)
        self.assertEqual(limits['max_processes'], 64)

    @unittest.skipUnless(sys.platform == 'win32', 'Job Objects exist only on Windows')
    def test_bash_runs_inside_a_limited_job(self) -> None:
        replies = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'b1', 'type': 'function', 'function': {
            'name': 'bash', 'arguments': json.dumps({'command': 'echo inside-job'})}}]},
            {'role': 'assistant', 'content': 'ok'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'b2', 'type': 'function', 'function': {
                'name': 'bash', 'arguments': json.dumps({'command': 'echo a | cat | cat'})}}]},
            {'role': 'assistant', 'content': 'ok'}]
        recorded: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as ws, patch('src.openai_compat.request.urlopen', side_effect=fake):
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                     runtime_config=AgentRuntimeConfig(
                                         cwd=Path(ws), session_directory=Path(ws) / 's', nudge_strategy='off',
                                         permissions=AgentPermissions(allow_shell_commands=True)))
            with patch.dict(os.environ, {'SIGIL_SANDBOX': '1'}):
                agent.run('echo')
                with patch.dict(os.environ, {'SIGIL_SANDBOX_MAX_PROCESSES': '1'}):
                    agent.run('pipe')
        first = [m for m in recorded[1]['messages'] if m.get('role') == 'tool'][-1]['content']
        self.assertIn('inside-job', first)
        self.assertIn('exit_code=0', first)
        second = [m for m in recorded[3]['messages'] if m.get('role') == 'tool'][-1]['content']
        self.assertNotIn('exit_code=0', second)  # one process allowed: the pipeline cannot start


if __name__ == '__main__':
    unittest.main()
