"""Batch 15a (P3): --version, --session-id, --fork-session, --no-session-persistence,
--system-prompt-file, --append-system-prompt-file, --disable-slash-commands,
--include-partial-messages, --replay-user-messages, --permission-prompt-tool, --init-only,
`sigil doctor` and `sigil completion`."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import SigilTool
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.main import _permission_tool_prompt, _run_print_mode, _stream_json_events, build_parser, main
from src.session_store import list_agent_sessions, load_agent_session

PY = sys.executable.replace('\\', '/')
BASE = 'http://127.0.0.1:8000/v1'


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


class FakeModel:
    def __init__(self, replies: list[dict] | None = None) -> None:
        self.replies = list(replies or [])
        self.requests: list[dict] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001
        self.requests.append(json.loads(req.data.decode('utf-8')))
        return _Resp(self.replies.pop(0) if self.replies else {'role': 'assistant', 'content': 'answer'})


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        self.model = FakeModel()
        for patcher in (patch('src.openai_compat.request.urlopen', side_effect=self.model),
                        patch.dict(os.environ, {})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.session_dir = None

    def run_main(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([*argv, '--cwd', str(self.ws), '--model', 'm', '--base-url', BASE])
        return code, out.getvalue()

    def sessions(self) -> list[str]:
        from src.main import _session_directory_for

        return [sid for sid, _, _ in list_agent_sessions(_session_directory_for(self.ws))]

    def test_version_doctor_and_completion(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            main(['--version'])
        self.assertEqual(caught.exception.code, 0)
        self.assertRegex(out.getvalue(), r'^Sigil Code \S+')
        code, text = self.run_main('doctor')
        self.assertEqual(code, 0)
        self.assertIn('Python version', text)
        for shell, marker in (('bash', 'complete -W'), ('zsh', '#compdef'), ('powershell', 'Register-ArgumentCompleter')):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(['completion', shell]), 0)
            self.assertIn(marker, out.getvalue())
            self.assertIn('agent-chat', out.getvalue())
            self.assertIn('--version', out.getvalue())
            self.assertNotIn('agent-bg-worker', out.getvalue())
            from src.main import _CLI_NAME

            self.assertIn(_CLI_NAME, out.getvalue())
            if shell == 'bash':
                self.assertTrue(out.getvalue().endswith(f'" {_CLI_NAME}\n'))

    def test_session_id_fork_and_no_persistence(self) -> None:
        code, text = self.run_main('-p', 'first', '--session-id', 'my-session-1', '--output-format', 'json')
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)['session_id'], 'my-session-1')
        self.assertIn('my-session-1', self.sessions())
        with self.assertRaises(SystemExit):
            self.run_main('-p', 'again', '--session-id', 'my-session-1')
        code, text = self.run_main('-p', 'second', '-c', '--fork-session', '--output-format', 'json')
        forked = json.loads(text)['session_id']
        self.assertNotEqual(forked, 'my-session-1')
        from src.main import _session_directory_for

        directory = _session_directory_for(self.ws)
        original = load_agent_session('my-session-1', directory=directory)
        copy = load_agent_session(forked, directory=directory)
        self.assertGreater(len(copy.messages), len(original.messages))  # the fork got the new turn
        self.assertIn('first', json.dumps(copy.messages))
        before = set(self.sessions())
        code, text = self.run_main('-p', 'temp', '--no-session-persistence', '--output-format', 'json')
        self.assertEqual(code, 0)
        self.assertEqual(set(self.sessions()), before)

    def test_prompt_files_and_disabled_slash_commands(self) -> None:
        (self.ws / 'sys.txt').write_text('CUSTOM SYSTEM FROM FILE', encoding='utf-8')
        (self.ws / 'extra.txt').write_text('APPENDED FROM FILE', encoding='utf-8')
        self.run_main('-p', 'hi', '--system-prompt-file', str(self.ws / 'sys.txt'),
                      '--append-system-prompt-file', str(self.ws / 'extra.txt'))
        system = self.model.requests[-1]['messages'][0]['content']
        self.assertIn('CUSTOM SYSTEM FROM FILE', system)
        self.assertIn('APPENDED FROM FILE', system)
        with self.assertRaises(SystemExit):
            self.run_main('-p', 'hi', '--system-prompt-file', str(self.ws / 'missing.txt'))
        count = len(self.model.requests)
        self.run_main('-p', '/help')
        self.assertEqual(len(self.model.requests), count)  # /help answered locally
        self.run_main('-p', '/help', '--disable-slash-commands')
        self.assertEqual(len(self.model.requests), count + 1)
        self.assertIn('/help', json.dumps(self.model.requests[-1]['messages']))

    def test_stream_json_partial_and_replay(self) -> None:
        args = build_parser().parse_args(['print', 'x', '--output-format', 'stream-json'])
        agent = LocalCodingAgent(model_config=ModelConfig(model='m'),
                                 runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s'))
        lines: list[str] = []
        emit = _stream_json_events(agent, args, lines.append)
        emit({'type': 'content_delta', 'delta': 'pa'})
        emit({'type': 'tool_start', 'tool_name': 'bash'})
        self.assertEqual([json.loads(line)['type'] for line in lines], ['tool_start'])
        args = build_parser().parse_args(['print', 'x', '--output-format', 'stream-json',
                                          '--include-partial-messages'])
        emit = _stream_json_events(agent, args, lines.append)
        emit({'type': 'content_delta', 'delta': 'pa'})
        self.assertEqual(json.loads(lines[-1])['type'], 'content_delta')
        self.assertTrue(agent.runtime_config.stream_model_responses)

        args = build_parser().parse_args(['print', '--input-format', 'stream-json', '--output-format',
                                          'stream-json', '--replay-user-messages', '--cwd', str(self.ws),
                                          '--model', 'm', '--base-url', BASE])
        out: list[str] = []
        stdin = io.StringIO(json.dumps({'type': 'user', 'message': {'role': 'user', 'content': 'hello there'}}) + '\n')
        self.assertEqual(_run_print_mode(args, stdin=stdin, out=out.append), 0)
        kinds = [json.loads(line)['type'] for line in out]
        self.assertEqual(kinds[0], 'user')
        self.assertEqual(json.loads(out[0])['message']['content'], 'hello there')
        self.assertEqual(kinds[-1], 'result')

    def test_permission_prompt_tool(self) -> None:
        answers = {'decision': 'allow'}
        seen: list[dict] = []

        def decide(arguments, context):  # noqa: ANN001
            seen.append(arguments)
            return json.dumps({'behavior': answers['decision']}), {}

        self.model.replies = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                'name': 'write_file', 'arguments': json.dumps({'path': 'a.txt', 'content': 'A'})}}]},
            {'role': 'assistant', 'content': 'done'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c2', 'type': 'function', 'function': {
                'name': 'write_file', 'arguments': json.dumps({'path': 'b.txt', 'content': 'B'})}}]},
            {'role': 'assistant', 'content': 'done'},
        ]
        agent = LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url=BASE),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s', nudge_strategy='off'))
        agent.tool_registry = {**agent.tool_registry, 'mcp__perm__decide': SigilTool(
            name='mcp__perm__decide', description='d', parameters={'type': 'object'}, handler=decide)}
        agent.permission_prompt = _permission_tool_prompt(agent, 'mcp__perm__decide')
        agent.run('write a')
        self.assertEqual((self.ws / 'a.txt').read_text(encoding='utf-8'), 'A')
        self.assertEqual(seen[0]['tool_name'], 'write_file')
        self.assertEqual(seen[0]['input']['path'], 'a.txt')
        answers['decision'] = 'deny'
        agent.run('write b')
        self.assertFalse((self.ws / 'b.txt').exists())
        self.assertEqual(_permission_tool_prompt(agent, 'mcp__missing__x')({'tool': 'bash'}), 'deny')

    def test_init_only_runs_setup_hooks(self) -> None:
        marker = self.ws / 'setup-ran.json'
        script = self.ws / 'hook.py'
        script.write_text('import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), '
                          'encoding="utf-8")\n', encoding='utf-8')
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': {'Setup': [{'hooks': [{
            'type': 'command', 'command': f'"{PY}" "{script.as_posix()}" "{marker.as_posix()}"'}]}]}}),
            encoding='utf-8')
        code, text = self.run_main('agent', 'unused prompt', '--init-only')
        self.assertEqual(code, 0)
        self.assertIn('Setup hooks done', text)
        payload = json.loads(marker.read_text(encoding='utf-8'))
        self.assertEqual((payload['hook_event_name'], payload['trigger']), ('Setup', 'init'))
        self.assertEqual(self.model.requests, [])  # no model call


if __name__ == '__main__':
    unittest.main()
