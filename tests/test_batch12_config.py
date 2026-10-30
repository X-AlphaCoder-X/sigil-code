"""Batch 12 (P2): stable prompt prefix, max_tokens, fast model, cleared-results note,
provider profiles, /test /lint and autoTest, sigil mcp add|list|remove|get, /mcp reconnect|enable|disable."""
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
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.main import _run_agent_chat_loop, main
from src.mcp_runtime import MCPRuntime


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


def tool_reply(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def make_agent(ws: Path, model: str = 'm', **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model=model),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def isolated(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({'SIGIL_DISABLE_USER_SETTINGS': '1', 'SIGIL_FAST_MODEL': '', 'SIGIL_MAX_TOKENS': ''})
    env.update(extra)
    return env


class PromptTests(unittest.TestCase):
    def test_system_prompt_is_the_same_across_sessions(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated()), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('a'), text_reply('b')], recorded)):
            make_agent(Path(tmp)).run('first')
            make_agent(Path(tmp)).run('second')
        first_system = recorded[0]['messages'][0]['content']
        second_system = recorded[1]['messages'][0]['content']
        self.assertEqual(first_system, second_system)
        self.assertNotIn('scratchpadDirectory', first_system)
        self.assertIn('Older tool results', first_system)
        reminder = recorded[0]['messages'][1]['content']
        self.assertIn('scratchpadDirectory', reminder)

    def test_old_layout_switch(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(SIGIL_STABLE_PREFIX='0')), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('a')], recorded)):
            make_agent(Path(tmp)).run('first')
        self.assertIn('scratchpadDirectory', recorded[0]['messages'][0]['content'])


class ModelSettingTests(unittest.TestCase):
    def test_max_tokens(self) -> None:
        recorded: list[dict[str, object]] = []
        from src.openai_compat import OpenAICompatClient

        with patch.dict(os.environ, isolated()), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('a'), text_reply('b'), text_reply('c')], recorded)):
            OpenAICompatClient(ModelConfig(model='m', max_tokens=256)).complete([{'role': 'user', 'content': 'x'}], [])
            OpenAICompatClient(ModelConfig(model='m')).complete([{'role': 'user', 'content': 'x'}], [])
            with patch.dict(os.environ, {'SIGIL_MAX_TOKENS': '99'}):
                OpenAICompatClient(ModelConfig(model='m')).complete([{'role': 'user', 'content': 'x'}], [])
        self.assertEqual(recorded[0]['max_tokens'], 256)
        self.assertNotIn('max_tokens', recorded[1])
        self.assertEqual(recorded[2]['max_tokens'], 99)

    def test_fast_model_for_compaction(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('one'), text_reply('two'), text_reply('Summary.')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated(SIGIL_FAST_MODEL='tiny-model')), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp), model='big-model')
            first = agent.run('a')
            from src.session_store import load_agent_session

            agent.resume('b', load_agent_session(first.session_id, directory=agent.runtime_config.session_directory))
            agent.compact(load_agent_session(first.session_id, directory=agent.runtime_config.session_directory))
        self.assertEqual([r['model'] for r in recorded], ['big-model', 'big-model', 'tiny-model'])

    def test_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, isolated(MY_CLOUD_KEY='sk-cloud')):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'profiles': {'cloud': {
                'base_url': 'https://openrouter.ai/api/v1', 'model': 'big/model', 'api_key_env': 'MY_CLOUD_KEY'}}}),
                encoding='utf-8')
            agent = make_agent(ws)
            listing = agent.run('/profile').final_output
            agent.run('/profile cloud')
            from src.main import _build_agent, build_parser

            via_flag = _build_agent(build_parser().parse_args(['agent', 'x', '--cwd', tmp, '--profile', 'cloud']))
        self.assertIn('cloud: big/model', listing)
        self.assertEqual(agent.model_config.model, 'big/model')
        self.assertEqual(agent.model_config.api_key, 'sk-cloud')
        self.assertEqual(via_flag.model_config.base_url, 'https://openrouter.ai/api/v1')


class CheckCommandTests(unittest.TestCase):
    def _settings(self, ws: Path, payload: dict) -> None:
        (ws / '.sigil').mkdir(exist_ok=True)
        (ws / '.sigil' / 'settings.json').write_text(json.dumps(payload), encoding='utf-8')

    def test_test_command_feeds_failures(self) -> None:
        recorded: list[dict[str, object]] = []
        failing = f'"{sys.executable}" -c "print(\'1 failed: test_x\'); raise SystemExit(1)"'
        passing = f'"{sys.executable}" -c "print(\'all good\')"'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated()), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('fixing')], recorded)):
            ws = Path(tmp)
            self._settings(ws, {'testCommand': failing, 'lintCommand': passing})
            agent = make_agent(ws, allow_shell_commands=True)
            agent.run('/test')
            lint = agent.run('/lint').final_output
        self.assertIn('1 failed: test_x', json.dumps(recorded[0]['messages']))
        self.assertIn('Lint passed', lint)

    def test_auto_test_after_edits(self) -> None:
        recorded: list[dict[str, object]] = []
        failing = f'"{sys.executable}" -c "raise SystemExit(\'boom in test\')"'
        responses = [tool_reply('write_file', {'path': 'a.py', 'content': 'x = 1'}), text_reply('wrote it'),
                     text_reply('fixed it')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated()), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            self._settings(ws, {'testCommand': failing, 'autoTest': True})
            agent = make_agent(ws, allow_shell_commands=True, allow_file_write=True)
            prompts = iter(['make a.py'])

            def _input(_label: str) -> str:
                try:
                    return next(prompts)
                except StopIteration:
                    raise EOFError

            _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                                 input_func=_input, output_func=lambda _t: None,
                                 result_printer=lambda _r, show_transcript: None)
        self.assertEqual(len(recorded), 3)
        self.assertIn('boom in test', json.dumps(recorded[2]['messages']))


class McpManagementTests(unittest.TestCase):
    def test_add_list_get_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch('src.settings._home', return_value=Path(home)), patch.dict(os.environ, isolated()):
            out = io.StringIO()
            with redirect_stdout(out):
                main(['mcp', 'add', 'files', '--cwd', tmp, '--env', 'TOKEN=x', '--', 'npx', '-y', 'server-files'])
                main(['mcp', 'add', 'remote', '--cwd', tmp, '--scope', 'user', '--url', 'https://h/mcp',
                      '--header', 'Authorization: Bearer ${KEY}'])
                main(['mcp', 'list', '--cwd', tmp])
                main(['mcp', 'get', 'remote', '--cwd', tmp])
            text = out.getvalue()
            project = json.loads((Path(tmp) / '.mcp.json').read_text(encoding='utf-8'))
            self.assertEqual(project['mcpServers']['files'],
                             {'command': 'npx', 'args': ['-y', 'server-files'], 'env': {'TOKEN': 'x'}})
            self.assertIn('files (project)', text)
            self.assertIn('remote (user): https://h/mcp', text)
            self.assertIn('"Authorization": "Bearer ${KEY}"', text)
            self.assertEqual({s.name for s in MCPRuntime.from_workspace(Path(tmp)).servers}, {'files', 'remote'})
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['mcp', 'remove', 'files', '--cwd', tmp]), 0)
            self.assertEqual({s.name for s in MCPRuntime.from_workspace(Path(tmp)).servers}, {'remote'})

    def test_enable_disable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated()):
            ws = Path(tmp)
            (ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'a': {'command': 'x'}, 'b': {'command': 'y'}}}),
                                          encoding='utf-8')
            agent = make_agent(ws)
            agent.run('/mcp disable a')
            self.assertEqual({s.name for s in agent.mcp_runtime.servers}, {'b'})
            agent.run('/mcp enable a')
            self.assertEqual({s.name for s in agent.mcp_runtime.servers}, {'a', 'b'})
            self.assertIn('Unknown MCP server', agent.run('/mcp reconnect nope').final_output)


if __name__ == '__main__':
    unittest.main()
