"""Batch 10 (P2): hook options (if, async, shell, args, statusMessage) and outputs
(systemMessage, updatedToolOutput), more hook events, recursive rules with paths:,
--settings, --setting-sources, --mcp-config, --strict-mcp-config, --safe-mode,
--input-format stream-json."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.agent_context import memory_file_paths
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.command_hooks import CommandHookRuntime
from src.main import main
from src.mcp_runtime import MCPRuntime
from src.settings import load_settings

PY = sys.executable.replace('\\', '/')


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def fake_server(responses: list[dict[str, object]], recorded: list[dict[str, object]]):
    queued = [FakeHTTPResponse(payload) for payload in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        recorded.append(json.loads(request_obj.data.decode('utf-8')))
        return queued.pop(0)

    return _fake_urlopen


def text_reply(text: str) -> dict[str, object]:
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}


def tool_reply(name: str, arguments: dict[str, object], call_id: str = 'call_1') -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def write_settings(ws: Path, payload: dict) -> None:
    (ws / '.sigil').mkdir(exist_ok=True)
    (ws / '.sigil' / 'settings.json').write_text(json.dumps(payload), encoding='utf-8')


def logging_script(ws: Path, log: Path, extra: str = '') -> str:
    script = ws / 'hook.py'
    script.write_text('import json, sys\nd = json.load(sys.stdin)\n'
                      f'open({str(log)!r}, "a").write(d["hook_event_name"] + "\\n")\n' + extra, encoding='utf-8')
    return f'"{PY}" "{script.as_posix()}"'


def isolated(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SIGIL_SETTING', 'SIGIL_MCP', 'SIGIL_SAFE',
                                                                     'SIGIL_STRICT'))}
    env.update({'SIGIL_DISABLE_USER_HOOKS': '1', 'SIGIL_DISABLE_USER_SETTINGS': '1'})
    env.update(extra)
    return env


class HookOptionTests(unittest.TestCase):
    def test_if_condition_and_updated_output_and_system_message(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('bash', {'command': 'echo secret-token'}),
                     tool_reply('bash', {'command': 'echo other'}, call_id='c2'), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
                patch('sys.stderr', io.StringIO()) as err:
            ws = Path(tmp)
            log = ws / 'events.log'
            command = logging_script(ws, log, 'print(json.dumps({"systemMessage": "redacted output", '
                                              '"hookSpecificOutput": {"updatedToolOutput": "[redacted]"}}))\n')
            write_settings(ws, {'hooks': {'PostToolUse': [{'matcher': 'Bash', 'hooks': [
                {'type': 'command', 'command': command, 'if': 'Bash(echo secret*)',
                 'statusMessage': 'checking output'}]}]}})
            make_agent(ws, allow_shell_commands=True).run('run')
            events = log.read_text(encoding='utf-8').split()
            messages = err.getvalue()
        self.assertEqual(events, ['PostToolUse'])  # only the call matching "if"
        tools = [m['content'] for m in recorded[2]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any('[redacted]' in t for t in tools))
        self.assertFalse(any('secret-token' in t for t in tools))
        self.assertTrue(any('other' in t for t in tools))
        self.assertIn('redacted output', messages)
        self.assertIn('checking output', messages)

    def test_args_shell_and_async(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(), clear=True):
            ws = Path(tmp)
            out = ws / 'out.txt'
            script = ws / 'write.py'
            script.write_text('import sys, time\ntime.sleep(float(sys.argv[2]))\n'
                              'open(sys.argv[1], "a").write("ran\\n")\n', encoding='utf-8')
            write_settings(ws, {'hooks': {'Notification': [{'hooks': [
                {'type': 'command', 'command': sys.executable, 'args': [str(script), str(out), '0']},
                {'type': 'command', 'command': sys.executable, 'args': [str(script), str(out), '5'],
                 'async': True},
            ]}]}})
            runtime = CommandHookRuntime.from_workspace(ws)
            started = time.monotonic()
            runtime.run('Notification', {'message': 'hi'})
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 4.0)  # the 5-second async hook was not waited for
            self.assertEqual(out.read_text(encoding='utf-8').split(), ['ran'])
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and len(out.read_text(encoding='utf-8').split()) < 2:
                time.sleep(0.1)
            self.assertEqual(out.read_text(encoding='utf-8').split(), ['ran', 'ran'])
            if sys.platform == 'win32':
                write_settings(ws, {'hooks': {'Notification': [{'hooks': [
                    {'type': 'command', 'shell': 'powershell',
                     'command': f"Add-Content -Path '{out}' -Value 'from-ps'"}]}]}})
                CommandHookRuntime.from_workspace(ws).run('Notification', {})
                self.assertIn('from-ps', out.read_text(encoding='utf-8'))


class HookEventTests(unittest.TestCase):
    def test_events(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('task_create', {'title': 'Write docs'}),
            text_reply('created'),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            (ws / 'SIGIL.md').write_text('rules', encoding='utf-8')
            (ws / '.sigil' / 'commands').mkdir(parents=True)
            (ws / '.sigil' / 'commands' / 'go.md').write_text('Go now', encoding='utf-8')
            log = ws / 'events.log'
            command = logging_script(ws, log)
            write_settings(ws, {'hooks': {event: [{'hooks': [{'type': 'command', 'command': command}]}] for event in (
                'InstructionsLoaded', 'TaskCreated', 'PostToolBatch', 'PreModelSwitch', 'PostModelSwitch',
                'ConfigChange', 'UserPromptExpansion')}})
            agent = make_agent(ws, allow_file_write=True)
            agent.run('make a task')
            agent.set_model('other-model')
            agent.run('/permissions allow Read')
            write_settings(ws, {'hooks': {'UserPromptExpansion': [{'hooks': [
                {'type': 'command', 'command': f'"{PY}" -c "import sys; sys.exit(2)"'}]}]}})
            blocked_agent = make_agent(ws)
            blocked = blocked_agent.run('/go')
            events = log.read_text(encoding='utf-8').split()
        for event in ('InstructionsLoaded', 'TaskCreated', 'PostToolBatch', 'PreModelSwitch', 'PostModelSwitch',
                      'ConfigChange'):
            self.assertIn(event, events)
        self.assertIn('Blocked by UserPromptExpansion hook', blocked.final_output)

    def test_pre_model_switch_can_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(), clear=True):
            ws = Path(tmp)
            write_settings(ws, {'hooks': {'PreModelSwitch': [{'hooks': [
                {'type': 'command', 'command': f'"{PY}" -c "import sys; sys.exit(2)"'}]}]}})
            agent = make_agent(ws)
            agent.set_model('blocked-model')
        self.assertEqual(agent.model_config.model, 'm')


class RulesTests(unittest.TestCase):
    def test_recursive_rules_with_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            rules = ws / '.sigil' / 'rules'
            (rules / 'backend').mkdir(parents=True)
            (rules / 'general.md').write_text('general rule', encoding='utf-8')
            (rules / 'backend' / 'python.md').write_text('---\npaths: ["**/*.py"]\n---\npython rule', encoding='utf-8')
            (rules / 'backend' / 'rust.md').write_text('---\npaths: ["**/*.rs"]\n---\nrust rule', encoding='utf-8')
            (ws / 'app.py').write_text('x', encoding='utf-8')
            names = {p.name for p in memory_file_paths(ws)}
        self.assertIn('general.md', names)
        self.assertIn('python.md', names)
        self.assertNotIn('rust.md', names)


class FlagTests(unittest.TestCase):
    def run_print(self, argv: list[str], responses: list[dict[str, object]], stdin_text: str = '', tty=True):
        recorded: list[dict[str, object]] = []

        class _Stdin(io.StringIO):
            def isatty(self) -> bool:
                return tty

        out = io.StringIO()
        with patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
                patch('sys.stdin', _Stdin(stdin_text)), redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue(), recorded

    def test_settings_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(), clear=True):
            ws = Path(tmp)
            write_settings(ws, {'outputStyle': 'from-project', 'permissions': {'deny': ['Bash']}})
            os.environ['SIGIL_SETTINGS'] = json.dumps({'outputStyle': 'from-flag'})
            self.assertEqual(load_settings(ws)['outputStyle'], 'from-flag')
            os.environ['SIGIL_SETTING_SOURCES'] = 'local'
            self.assertNotIn('permissions', load_settings(ws))
            os.environ.pop('SIGIL_SETTINGS')
            code, _, recorded = self.run_print(
                ['-p', 'hi', '--cwd', tmp, '--settings', json.dumps({'permissions': {'deny': ['WebFetch']}}),
                 '--setting-sources', 'project'], [text_reply('ok')])
            self.assertEqual(code, 0)
            self.assertEqual(os.environ['SIGIL_SETTING_SOURCES'], 'project')

    def test_mcp_config_flags_and_safe_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other, \
                patch.dict(os.environ, isolated(), clear=True):
            ws = Path(tmp)
            (ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'local': {'command': 'x'}}}), encoding='utf-8')
            extra = Path(other) / 'extra.json'
            extra.write_text(json.dumps({'mcpServers': {'extra': {'command': 'y'}}}), encoding='utf-8')
            os.environ['SIGIL_MCP_CONFIG'] = str(extra)
            self.assertEqual({s.name for s in MCPRuntime.from_workspace(ws).servers}, {'local', 'extra'})
            os.environ['SIGIL_STRICT_MCP_CONFIG'] = '1'
            self.assertEqual({s.name for s in MCPRuntime.from_workspace(ws).servers}, {'extra'})
            os.environ['SIGIL_SAFE_MODE'] = '1'
            self.assertEqual(MCPRuntime.from_workspace(ws).servers, ())
            write_settings(ws, {'hooks': {'Stop': [{'hooks': [{'type': 'command', 'command': 'x'}]}]}})
            self.assertEqual(CommandHookRuntime.from_workspace(ws).hooks, ())
            (ws / 'SIGIL.md').write_text('MEMORY-MARKER', encoding='utf-8')
            os.environ.pop('SIGIL_SAFE_MODE')
            code, _, recorded = self.run_print(['-p', 'hi', '--cwd', tmp, '--safe-mode'], [text_reply('ok')])
            self.assertEqual(code, 0)
            self.assertNotIn('MEMORY-MARKER', json.dumps(recorded[0]['messages']))

    def test_stream_json_input(self) -> None:
        lines = '\n'.join([
            json.dumps({'type': 'user', 'message': {'role': 'user', 'content': 'my number is 7'}}),
            json.dumps({'type': 'user', 'message': {'role': 'user', 'content': [{'type': 'text', 'text': 'what is it?'}]}}),
        ]) + '\n'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(), clear=True):
            code, out, recorded = self.run_print(
                ['-p', '--input-format', 'stream-json', '--output-format', 'json', '--cwd', tmp],
                [text_reply('noted'), text_reply('7')], stdin_text=lines, tty=False)
        self.assertEqual(code, 0)
        results = [json.loads(line) for line in out.strip().splitlines()]
        self.assertEqual([r['result'] for r in results], ['noted', '7'])
        self.assertEqual(results[0]['session_id'], results[1]['session_id'])
        self.assertIn('my number is 7', json.dumps(recorded[1]['messages']))


if __name__ == '__main__':
    unittest.main()
