"""Opt-in bash sandbox: bubblewrap / sandbox-exec command lines, the allow-list
proxy, and an agent run through a recording stand-in for bwrap."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src import sandbox
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig

PY = sys.executable.replace('\\', '/')
FAKE_BWRAP = r'''
import json, subprocess, sys
args = sys.argv[1:]
with open(sys.argv[0] + '.calls', 'a', encoding='utf-8') as f:
    f.write(json.dumps(args) + '\n')
inner = args[args.index('--') + 1:]
sys.exit(subprocess.run(inner).returncode)
'''


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason':
                                             'tool_calls' if message.get('tool_calls') else 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _bash(call_id: str, command: str, **extra) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function',
        'function': {'name': 'bash', 'arguments': json.dumps({'command': command, **extra})}}]}


class ArgvTests(unittest.TestCase):
    def test_linux_bwrap_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_BWRAP': '/usr/bin/bwrap', 'SIGIL_SANDBOX_DOMAINS': '',
                'SIGIL_SANDBOX_NETWORK': ''}):
            cwd = Path(tmp).resolve()
            argv = sandbox.build_argv('/bin/bash', 'ls', cwd, ('/extra',), platform='linux')
            self.assertEqual(argv[:5], ['/usr/bin/bwrap', '--ro-bind', '/', '/', '--dev'])
            self.assertIn(['--bind', str(cwd), str(cwd)], [argv[i:i + 3] for i in range(len(argv))])
            self.assertIn('--unshare-all', argv)
            self.assertNotIn('--share-net', argv)
            self.assertEqual(argv[-4:], ['--', '/bin/bash', '-c', 'ls'])
            with patch.dict(os.environ, {'SIGIL_SANDBOX_DOMAINS': 'pypi.org'}):
                # allow-list: no shared network; only the proxy, through the socket and the relay
                isolated = sandbox.build_argv('/bin/bash', 'pip install x', cwd, platform='linux', proxy_port=4321,
                                              proxy_socket='/tmp/sigil-proxy-1/proxy.sock')
                self.assertNotIn('--share-net', isolated)
                self.assertIn(['--bind', '/tmp/sigil-proxy-1', '/tmp/sigil-proxy-1'],
                              [isolated[i:i + 3] for i in range(len(isolated))])
                tail = isolated[isolated.index('--', isolated.index('--chdir')) + 1:]
                self.assertTrue(tail[1].endswith('sandbox_relay.py'))
                self.assertEqual(tail[2:], ['unix:/tmp/sigil-proxy-1/proxy.sock', '4321', '--',
                                            '/bin/bash', '-c', 'pip install x'])
                # the bind comes after "--tmpfs /tmp", so the socket folder stays visible
                self.assertLess(isolated.index('--tmpfs'), isolated.index('/tmp/sigil-proxy-1'))
            with patch.dict(os.environ, {'SIGIL_SANDBOX_NETWORK': '1'}):
                self.assertIn('--share-net', sandbox.build_argv('/bin/bash', 'ls', cwd, platform='linux'))

    def test_macos_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_SANDBOX': '1',
                                                                           'SIGIL_SANDBOX_NETWORK': ''}), \
                patch('src.sandbox.shutil.which', return_value='/usr/bin/sandbox-exec'):
            cwd = Path(tmp).resolve()
            argv = sandbox.build_argv('/bin/bash', 'make', cwd, platform='darwin', proxy_port=4321)
            self.assertEqual(argv[0], 'sandbox-exec')
            profile = argv[2]
            self.assertIn('(deny file-write*)', profile)
            self.assertIn(f'(subpath "{cwd}")', profile)
            self.assertIn('(deny network*)', profile)
            self.assertIn('localhost:4321', profile)

    def test_off_by_default_windows_and_missing_tool(self) -> None:
        with patch.dict(os.environ, {'SIGIL_SANDBOX': ''}):
            self.assertIsNone(sandbox.sandbox_argv('bash', 'ls', Path('.')))
            self.assertFalse(sandbox.shell_allowed(False))
            self.assertTrue(sandbox.shell_allowed(True))
        with patch.dict(os.environ, {'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_BWRAP': ''}), \
                patch('src.sandbox._platform', return_value='win32'):
            self.assertIsNone(sandbox.sandbox_argv('bash', 'ls', Path('.')))
            self.assertFalse(sandbox.active())
        with patch.dict(os.environ, {'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_BWRAP': ''}), \
                patch('src.sandbox._platform', return_value='linux'), \
                patch('src.sandbox.shutil.which', return_value=None):
            self.assertFalse(sandbox.active())
            with self.assertRaises(sandbox.SandboxUnavailable) as caught:
                sandbox.sandbox_argv('bash', 'ls', Path('.'))
            self.assertIn('SIGIL_SANDBOX=0', str(caught.exception))

    def test_host_allow_list(self) -> None:
        self.assertTrue(sandbox.host_allowed('api.github.com', ['github.com']))
        self.assertTrue(sandbox.host_allowed('github.com', ['github.com']))
        self.assertFalse(sandbox.host_allowed('evilgithub.com', ['github.com']))
        self.assertTrue(sandbox.host_allowed('a.b.foo.io', ['*.foo.io']))


class _Hello(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        body = b'hello through proxy'
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ProxyTests(unittest.TestCase):
    def test_proxy_allows_listed_hosts_only(self) -> None:
        server = ThreadingHTTPServer(('127.0.0.1', 0), _Hello)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_address[1]}/x'
        for domains, ok in ((['127.0.0.1'], True), (['example.com'], False)):
            port = sandbox.start_proxy(domains)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': f'http://127.0.0.1:{port}'}))
            if ok:
                self.assertEqual(opener.open(url, timeout=10).read(), b'hello through proxy')
            else:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    opener.open(url, timeout=10)
                self.assertEqual(caught.exception.code, 403)


class RelayTests(unittest.TestCase):
    """The relay used inside the Linux sandbox, run for real here in its TCP mode."""

    def test_relay_reaches_only_allowed_hosts_through_the_proxy(self) -> None:
        import socket
        import subprocess

        server = ThreadingHTTPServer(('127.0.0.1', 0), _Hello)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        relay = Path(sandbox.__file__).with_name('sandbox_relay.py')
        child = ('import os, urllib.request, urllib.error\n'
                 'opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": os.environ["P"]}))\n'
                 'try:\n'
                 '    print(opener.open(os.environ["U"], timeout=10).read().decode())\n'
                 'except urllib.error.HTTPError as exc:\n'
                 '    print("HTTP", exc.code)\n')
        for domains, expected in ((['127.0.0.1'], 'hello through proxy'), (['example.com'], 'HTTP 403')):
            proxy_port = sandbox.start_proxy(domains)
            probe = socket.socket()
            probe.bind(('127.0.0.1', 0))
            relay_port = probe.getsockname()[1]
            probe.close()
            env = {**os.environ, 'P': f'http://127.0.0.1:{relay_port}',
                   'U': f'http://127.0.0.1:{server.server_address[1]}/x'}
            done = subprocess.run([sys.executable, str(relay), f'tcp:127.0.0.1:{proxy_port}', str(relay_port), '--',
                                   sys.executable, '-c', child], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip(), expected)

    def test_sandbox_setting_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_SANDBOX_DOMAINS': '',
                                                                           'SIGIL_SANDBOX_NETWORK': ''}), \
                patch('src.sandbox._settings', return_value={
                    'network': {'allowedDomains': ['github.com', '*.npmjs.org']},
                    'filesystem': {'allowWrite': [str(Path(tmp) / 'build')]}}):
            allow_all, domains = sandbox._network()
            paths = sandbox._write_paths(Path(tmp), ())
        self.assertEqual((allow_all, domains), (False, ['github.com', '*.npmjs.org']))
        self.assertIn(str((Path(tmp) / 'build').resolve()), paths)
        self.assertEqual(sandbox.proxy_env({})['ALL_PROXY'] if sandbox._PROXY else 'x', sandbox.proxy_env({}).get(
            'ALL_PROXY', 'x'))


class AgentSandboxTests(unittest.TestCase):
    def test_agent_bash_runs_inside_the_sandbox_without_allow_shell(self) -> None:
        replies = [_bash('c1', 'echo sandboxed-hello'), _bash('c2', 'echo bg-hi', run_in_background=True),
                   _bash('c3', 'rm -rf /'), {'role': 'assistant', 'content': 'done'}]
        recorded: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            fake_bwrap = ws / 'fake_bwrap.py'
            fake_bwrap.write_text(FAKE_BWRAP, encoding='utf-8')
            env = {'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_BWRAP': f'"{PY}" "{fake_bwrap.as_posix()}"',
                   'SIGIL_SANDBOX_DOMAINS': '', 'SIGIL_SANDBOX_NETWORK': ''}
            with patch.dict(os.environ, env), patch('src.sandbox._platform', return_value='linux'), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                    runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s' / 'agent',
                                                      scratchpad_root=ws / '.s' / 'scratch', nudge_strategy='off'))
                self.assertFalse(agent.runtime_config.permissions.allow_shell_commands)
                agent.run('say hello')
                tool_replies = [str(m.get('content')) for m in recorded[-1]['messages'] if m.get('role') == 'tool']
                self.assertIn('sandboxed-hello', tool_replies[0])
                self.assertIn('Started background shell', tool_replies[1])
                self.assertNotIn('exit_code=', tool_replies[2])  # the destructive check still applies
                calls = fake_bwrap.with_name('fake_bwrap.py.calls')
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and (not calls.exists() or
                                                       len(calls.read_text(encoding='utf-8').splitlines()) < 2):
                    time.sleep(0.1)
                lines = [json.loads(line) for line in calls.read_text(encoding='utf-8').splitlines()]
                from src.background_shells import shell_manager

                for job in list(shell_manager().jobs.values()):  # let the temp folder go on Windows
                    try:
                        job.process.wait(timeout=10)
                    except Exception:
                        shell_manager().kill(job.job_id)
            self.assertEqual(len(lines), 2)
            for argv in lines:
                self.assertIn('--unshare-all', argv)
                self.assertIn(str(ws), argv)
                self.assertEqual(argv[argv.index('--') + 2], '-c')
            self.assertEqual(lines[0][-1], 'echo sandboxed-hello')
            self.assertEqual(lines[1][-1], 'echo bg-hi')

    def test_streaming_bash_is_wrapped_too(self) -> None:
        from src.agent_tools import build_tool_context, default_tool_registry, execute_tool_streaming

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            fake_bwrap = ws / 'fake_bwrap.py'
            fake_bwrap.write_text(FAKE_BWRAP, encoding='utf-8')
            env = {'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_BWRAP': f'"{PY}" "{fake_bwrap.as_posix()}"',
                   'SIGIL_SANDBOX_DOMAINS': '', 'SIGIL_SANDBOX_NETWORK': ''}
            with patch.dict(os.environ, env), patch('src.sandbox._platform', return_value='linux'):
                registry = default_tool_registry()
                context = build_tool_context(AgentRuntimeConfig(cwd=ws), tool_registry=registry)
                updates = list(execute_tool_streaming(registry, 'bash', {'command': 'echo streamed-hi'}, context))
            result = [u.result for u in updates if u.kind == 'result'][-1]
            self.assertTrue(result.ok, result.content)
            self.assertIn('streamed-hi', result.content)
            calls = [json.loads(line) for line in
                     fake_bwrap.with_name('fake_bwrap.py.calls').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(calls[0][-1], 'echo streamed-hi')

    def test_missing_bwrap_reports_how_to_fix(self) -> None:
        replies = [_bash('c1', 'echo hi'), {'role': 'assistant', 'content': 'done'}]
        recorded: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            with patch.dict(os.environ, {'SIGIL_SANDBOX': '1', 'SIGIL_SANDBOX_BWRAP': ''}), \
                    patch('src.sandbox._platform', return_value='linux'), \
                    patch('src.sandbox.shutil.which', return_value=None), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake):
                from src.agent_types import AgentPermissions

                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                    runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s' / 'agent',
                                                      nudge_strategy='off',
                                                      permissions=AgentPermissions(allow_shell_commands=True)))
                agent.run('say hi')
            tool_replies = [str(m.get('content')) for m in recorded[-1]['messages'] if m.get('role') == 'tool']
            self.assertIn('bubblewrap', tool_replies[0])
            self.assertIn('SIGIL_SANDBOX=0', tool_replies[0])


if __name__ == '__main__':
    unittest.main()
