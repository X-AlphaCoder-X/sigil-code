"""Batch 7 (P2): /agents, /add-dir, /btw, /copy clipboard, /memory add|edit, /model list,
dontAsk mode, /permissions add|remove, permissions.additionalDirectories,
notifications, --fallback-model."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_bar import AgentBar
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.anthropic_client import FallbackClient, make_client
from src.desktop import copy_to_clipboard, notify
from src.main import _build_runtime_config, build_parser
from src.permission_modes import apply_mode, normalize_mode


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def fake_server(responses: list[object], recorded: list[dict[str, object]]):
    queued = list(responses)

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        body = request_obj.data
        recorded.append({'url': request_obj.full_url, 'body': json.loads(body.decode('utf-8')) if body else None})
        item = queued.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeHTTPResponse(item)

    return _fake_urlopen


def text_reply(text: str) -> dict[str, object]:
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}


def tool_reply(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def make_agent(ws: Path, model: str = 'm') -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model=model),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def env(**extra: str) -> dict[str, str]:
    values = dict(os.environ)
    values['SIGIL_DISABLE_USER_SETTINGS'] = '1'
    values.update(extra)
    return values


class CommandTests(unittest.TestCase):
    def test_agents_lists_definitions_and_background_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / '.sigil' / 'agents').mkdir(parents=True)
            (ws / '.sigil' / 'agents' / 'linter.md').write_text('---\nname: linter\ndescription: Lints\n---\nx',
                                                                 encoding='utf-8')
            output = AgentBar(ws).handle_command('/agents')
        self.assertIn('- Explore [built-in]', output)
        self.assertIn('- linter [project]: Lints', output)
        self.assertIn('# Background Sessions', output)

    def test_add_dir_during_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as extra, \
                patch.dict(os.environ, env(SIGIL_OTHER_DIRECTORIES='')):
            (Path(extra) / 'data.txt').write_text('extra data', encoding='utf-8')
            agent = make_agent(Path(tmp))
            from src.agent_tools import execute_tool

            before = execute_tool(agent.tool_registry, 'read_file', {'path': str(Path(extra) / 'data.txt')},
                                  agent.tool_context)
            message = agent.run(f'/add-dir {extra}').final_output
            after = execute_tool(agent.tool_registry, 'read_file', {'path': str(Path(extra) / 'data.txt')},
                                 agent.tool_context)
        self.assertFalse(before.ok)
        self.assertIn('Added folder', message)
        self.assertTrue(after.ok, after.content)

    def test_btw_is_not_saved(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('first answer'), text_reply('side answer'), text_reply('third')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp))
            first = agent.run('main question')
            side = agent.run('/btw what is 2+2?')
            from src.session_store import load_agent_session

            stored = load_agent_session(first.session_id, directory=agent.runtime_config.session_directory)
            agent.resume('follow up', stored)
        self.assertEqual(side.final_output, 'side answer')
        self.assertIn('main question', json.dumps(recorded[1]['body']['messages']))
        self.assertNotIn('what is 2+2', json.dumps(recorded[2]['body']['messages']))

    def test_copy_uses_the_clipboard(self) -> None:
        calls: list[tuple[list[str], bytes]] = []

        class _Done:
            returncode = 0

        def _run(command, input=None, **kwargs):  # noqa: A002, ANN001
            calls.append((command, input))
            return _Done()

        with patch.dict(os.environ, {'SIGIL_CLIPBOARD': ''}), patch('src.desktop.shutil.which', return_value='x'), \
                patch('src.desktop.subprocess.run', side_effect=_run):
            method = copy_to_clipboard('héllo')
        self.assertIn(method, ('clip', 'pbcopy', 'wl-copy'))
        expected = 'héllo'.encode('utf-16') if sys.platform == 'win32' else 'héllo'.encode('utf-8')
        self.assertEqual(calls[0][1], expected)
        stream = io.StringIO()
        with patch.dict(os.environ, {'SIGIL_CLIPBOARD': ''}), patch('src.desktop.shutil.which', return_value=None):
            self.assertEqual(copy_to_clipboard('abc', stream=stream), 'osc52')
        self.assertIn('\x1b]52;c;YWJj', stream.getvalue())
        with patch.dict(os.environ, {'SIGIL_CLIPBOARD': 'off'}):
            self.assertIsNone(copy_to_clipboard('x'))

    def test_memory_add_and_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.run('/memory add Use tabs, not spaces')
            self.assertIn('- Use tabs, not spaces', (ws / 'SIGIL.md').read_text(encoding='utf-8'))
            editor = ws / 'editor.py'
            editor.write_text('import sys\nopen(sys.argv[1], "a").write("- edited\\n")\n', encoding='utf-8')
            command = f'"{sys.executable}" "{editor}"' if os.name == 'nt' else f'{sys.executable} {editor}'
            with patch.dict(os.environ, {'EDITOR': command, 'VISUAL': ''}):
                agent.run('/memory edit')
            self.assertIn('- edited', (ws / 'SIGIL.md').read_text(encoding='utf-8'))

    def test_model_lists_server_models(self) -> None:
        recorded: list[dict[str, object]] = []
        models = {'data': [{'id': 'qwen3-coder'}, {'id': 'm'}, {'id': 'llama3'}]}
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([models], recorded)):
            output = make_agent(Path(tmp)).run('/model').final_output
        self.assertTrue(recorded[0]['url'].endswith('/models'))
        self.assertIn('* m', output)
        self.assertIn('- qwen3-coder', output)

    def test_permissions_add_and_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env()):
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.run('/permissions deny Bash(rm *)')
            local = json.loads((ws / '.sigil' / 'settings.local.json').read_text(encoding='utf-8'))
            self.assertEqual(local['permissions']['deny'], ['Bash(rm *)'])
            self.assertIn('Bash(rm *)', agent.run('/permissions').final_output)
            self.assertEqual(agent.permission_rules.deny[0].raw, 'Bash(rm *)')
            agent.run('/permissions remove Bash(rm *)')
            local = json.loads((ws / '.sigil' / 'settings.local.json').read_text(encoding='utf-8'))
            self.assertEqual(local['permissions']['deny'], [])

    def test_additional_directories_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as extra, patch.dict(os.environ, env()):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(
                json.dumps({'permissions': {'additionalDirectories': [extra]}}), encoding='utf-8')
            config = _build_runtime_config(build_parser().parse_args(['agent', 'x', '--cwd', tmp]))
        self.assertIn(Path(extra).resolve(), config.additional_working_directories)


class ModeAndNotifyTests(unittest.TestCase):
    def test_dont_ask_refuses_without_prompting(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('write_file', {'path': 'f.txt', 'content': 'x'}), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env()), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp))
            self.assertEqual(normalize_mode('dont-ask'), 'dontAsk')
            apply_mode(agent, 'dontAsk')
            asked: list[object] = []
            agent.permission_prompt = lambda request: (asked.append(request), 'once')[1]
            agent.run('write')
            self.assertFalse((Path(tmp) / 'f.txt').exists())
        self.assertEqual(asked, [])
        tool_text = json.dumps(recorded[1]['body']['messages'])
        self.assertIn('dontAsk', tool_text)

    def test_notify(self) -> None:
        stream = io.StringIO()
        with patch.dict(os.environ, {'SIGIL_NOTIFY': 'bell'}):
            self.assertEqual(notify('Sigil Code', 'finished', stream=stream), 'bell')
        self.assertIn('\a', stream.getvalue())
        self.assertIn('\x1b]9;Sigil Code: finished', stream.getvalue())
        with patch.dict(os.environ, {'SIGIL_NOTIFY': 'off'}):
            self.assertIsNone(notify('a', 'b', stream=io.StringIO()))


class FallbackTests(unittest.TestCase):
    def test_fallback_model_after_an_error(self) -> None:
        from urllib import error as urlerror

        recorded: list[dict[str, object]] = []
        failure = urlerror.HTTPError('http://x', 503, 'down', {}, io.BytesIO(b'overloaded'))
        args = build_parser().parse_args(['agent', 'x', '--fallback-model', 'small-model, other@http://127.0.0.1:9/v1'])
        from src.main import _build_model_config

        config = _build_model_config(args)
        self.assertEqual(config.fallback_models, ('small-model', 'other@http://127.0.0.1:9/v1'))
        client = make_client(config)
        self.assertIsInstance(client, FallbackClient)
        with patch('src.openai_compat.request.urlopen', side_effect=fake_server([failure, text_reply('from fallback')],
                                                                                 recorded)), \
                patch('time.sleep'):
            turn = client.complete([{'role': 'user', 'content': 'hi'}], [])
        self.assertEqual(turn.content, 'from fallback')
        self.assertEqual(recorded[-1]['body']['model'], 'small-model')


if __name__ == '__main__':
    unittest.main()
