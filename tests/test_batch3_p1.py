"""Batch 3 (P1): reasoning and text tool calls, plan-mode tools, !cmd, /resume,
built-in /init /review /security-review, user-level config."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.main import _apply_user_config, _run_agent_chat_loop, main
from src.permission_modes import apply_mode
from src.reasoning import parse_text_tool_calls, split_reasoning


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


def text_reply(text: str, **extra: object) -> dict[str, object]:
    message = {'role': 'assistant', 'content': text, **extra}
    return {'choices': [{'message': message, 'finish_reason': 'stop'}]}


def tool_reply(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def make_agent(ws: Path, *, shell: bool = False, write: bool = False) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(allow_shell_commands=shell, allow_file_write=write),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def tool_messages(payload: dict[str, object]) -> list[str]:
    return [str(m.get('content')) for m in payload['messages'] if m.get('role') == 'tool']


def chat(agent: LocalCodingAgent, lines: list[str]) -> list[str]:
    queue = iter(lines)
    outputs: list[str] = []

    def _input(_label: str) -> str:
        try:
            return next(queue)
        except StopIteration:
            raise EOFError

    _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                         input_func=_input, output_func=outputs.append,
                         result_printer=lambda _r, show_transcript: None)
    return outputs


class ReasoningTests(unittest.TestCase):
    def test_split_reasoning(self) -> None:
        self.assertEqual(split_reasoning('<think>hmm</think>Answer'), ('Answer', 'hmm'))
        self.assertEqual(split_reasoning('plan first</think>Answer'), ('Answer', 'plan first'))
        self.assertEqual(split_reasoning('Answer', 'from field'), ('Answer', 'from field'))
        self.assertEqual(split_reasoning('<think>cut off'), ('', 'cut off'))
        with patch.dict(os.environ, {'SIGIL_STRIP_THINK': '0'}):
            self.assertEqual(split_reasoning('<think>x</think>y')[0], '<think>x</think>y')

    def test_think_not_sent_back(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('<think>secret chain</think>The answer is 4.', reasoning_content='more'),
                     text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp))
            first = agent.run('2+2?')
            reasoning = agent.last_reasoning
            agent.resume('thanks', __import__('src.session_store', fromlist=['x']).load_agent_session(
                first.session_id, directory=agent.runtime_config.session_directory))
        self.assertEqual(first.final_output, 'The answer is 4.')
        self.assertIn('secret chain', reasoning)
        self.assertIn('more', reasoning)
        self.assertNotIn('secret chain', json.dumps(recorded[1]['messages']))

    def test_text_tool_call_formats(self) -> None:
        names = {'read_file', 'list_dir'}
        rest, calls = parse_text_tool_calls(
            'Reading.\n<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>', names)
        self.assertEqual(calls, [('read_file', {'path': 'a.py'})])
        self.assertEqual(rest, 'Reading.')
        _, calls = parse_text_tool_calls(
            '<function=read_file>\n<parameter=path>\nb.py\n</parameter>\n<parameter=limit>5</parameter>\n</function>',
            names)
        self.assertEqual(calls, [('read_file', {'path': 'b.py', 'limit': 5})])
        _, calls = parse_text_tool_calls('```json\n{"name": "list_dir", "arguments": {"path": "."}}\n```', names)
        self.assertEqual(calls, [('list_dir', {'path': '.'})])
        self.assertEqual(parse_text_tool_calls('{"name": "rm_rf", "arguments": {}}', names)[1], [])
        self.assertEqual(parse_text_tool_calls('Just prose.', names)[1], [])

    def test_text_tool_call_runs_without_a_nudge(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            text_reply('<tool_call>{"name": "read_file", "arguments": {"path": "hello.txt"}}</tool_call>'),
            text_reply('It says hi.'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            (ws / 'hello.txt').write_text('hi there', encoding='utf-8')
            result = make_agent(ws).run('read hello.txt')
        self.assertEqual(result.final_output, 'It says hi.')
        self.assertEqual(len(recorded), 2)
        self.assertTrue(any('hi there' in t for t in tool_messages(recorded[1])))

    def test_text_tool_calls_can_be_turned_off(self) -> None:
        recorded: list[dict[str, object]] = []
        reply = '<tool_call>{"name": "read_file", "arguments": {"path": "x"}}</tool_call>'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_TEXT_TOOL_CALLS': '0'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply(reply)], recorded)):
            result = make_agent(Path(tmp)).run('go')
        self.assertIn('<tool_call>', result.final_output)


class PlanToolTests(unittest.TestCase):
    def test_exit_plan_mode_approved_and_denied(self) -> None:
        for answer, expect_mode in (('once', 'default'), ('deny', 'plan')):
            recorded: list[dict[str, object]] = []
            responses = [tool_reply('exit_plan_mode', {'plan': '1. edit a.py'}), text_reply('ok')]
            with tempfile.TemporaryDirectory() as tmp, \
                    patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
                agent = make_agent(Path(tmp), write=True)
                apply_mode(agent, 'plan')
                seen: list[dict[str, object]] = []
                agent.permission_prompt = lambda request, a=answer: (seen.append(request), a)[1]
                agent.run('plan it')
            self.assertEqual(agent.permission_mode, expect_mode, answer)
            self.assertIn('1. edit a.py', seen[0]['arguments']['plan'])

    def test_enter_plan_mode(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('enter_plan_mode', {}), tool_reply('write_file', {'path': 'x.txt', 'content': 'x'}),
                     text_reply('planned')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agent = make_agent(ws, write=True)
            agent.run('think first')
            self.assertEqual(agent.permission_mode, 'plan')
            self.assertFalse((ws / 'x.txt').exists())


class ChatCommandTests(unittest.TestCase):
    def test_bang_command_output_goes_with_next_prompt(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('seen')], recorded)):
            outputs = chat(make_agent(Path(tmp), shell=True), ['!echo bang-output', 'what did it print?'])
        self.assertTrue(any('bang-output' in line for line in outputs))
        user = [m['content'] for m in recorded[0]['messages'] if m.get('role') == 'user'][-1]
        self.assertIn('<bash-input>echo bang-output</bash-input>', user)
        self.assertIn('what did it print?', user)

    def test_bang_command_respects_shell_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outputs = chat(make_agent(Path(tmp), shell=False), ['!echo nope'])
        self.assertTrue(any('--allow-shell' in line for line in outputs))

    def test_resume_in_chat(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('first answer'), text_reply('second answer')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            first = make_agent(ws).run('remember the code word')
            outputs = chat(make_agent(ws), ['/resume', '0', 'what was it?'])
        self.assertTrue(any('remember the code word' in line for line in outputs))
        self.assertIn(f'resuming_session_id={first.session_id}', outputs)
        contents = [m.get('content') for m in recorded[1]['messages']]
        self.assertIn('remember the code word', contents)

    def test_builtin_commands(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('a'), text_reply('b'), text_reply('c')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.run('/init')
            agent.run('/review src/app.py')
            agent.run('/security-review')
        users = [[m['content'] for m in r['messages'] if m.get('role') == 'user'][-1] for r in recorded]
        self.assertIn('SIGIL.md', users[0])
        self.assertIn('src/app.py', users[1])
        self.assertIn('security review', users[2])


class UserConfigTests(unittest.TestCase):
    def test_env_files_and_settings_fill_missing_values(self) -> None:
        names = ('SIGIL_T_FROM_CWD_ENV', 'SIGIL_T_FROM_HOME_ENV', 'SIGIL_T_FROM_SETTINGS', 'SIGIL_T_KEEP',
                 'OPENAI_MODEL')
        env = {k: v for k, v in os.environ.items() if k not in names}
        env['SIGIL_T_KEEP'] = 'real'
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch.dict(os.environ, env, clear=True), patch('src.settings._home', return_value=Path(home)):
            ws = Path(tmp)
            (ws / '.env').write_text('SIGIL_T_FROM_CWD_ENV=cwd\nSIGIL_T_KEEP=from-file\n', encoding='utf-8')
            (Path(home) / '.sigil').mkdir()
            (Path(home) / '.sigil' / '.env').write_text('SIGIL_T_FROM_HOME_ENV=home\n', encoding='utf-8')
            (Path(home) / '.sigil' / 'settings.json').write_text(
                json.dumps({'model': 'my-local-model', 'env': {'SIGIL_T_FROM_SETTINGS': 'yes'}}), encoding='utf-8')
            _apply_user_config(ws)
            self.assertEqual(os.environ['SIGIL_T_FROM_CWD_ENV'], 'cwd')
            self.assertEqual(os.environ['SIGIL_T_FROM_HOME_ENV'], 'home')
            self.assertEqual(os.environ['SIGIL_T_FROM_SETTINGS'], 'yes')
            self.assertEqual(os.environ['SIGIL_T_KEEP'], 'real')
            self.assertEqual(os.environ['OPENAI_MODEL'], 'my-local-model')

    def test_model_from_user_settings_reaches_the_request(self) -> None:
        recorded: list[dict[str, object]] = []

        class _Stdin(io.StringIO):
            def isatty(self) -> bool:
                return True

        env = {k: v for k, v in os.environ.items() if k != 'OPENAI_MODEL'}
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch.dict(os.environ, env, clear=True), patch('src.settings._home', return_value=Path(home)), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('hi')], recorded)), \
                patch('sys.stdin', _Stdin('')), redirect_stdout(io.StringIO()):
            (Path(home) / '.sigil').mkdir()
            (Path(home) / '.sigil' / 'settings.json').write_text(json.dumps({'model': 'from-user-settings'}),
                                                                  encoding='utf-8')
            main(['-p', 'hi', '--cwd', tmp])
        self.assertEqual(recorded[0]['model'], 'from-user-settings')


if __name__ == '__main__':
    unittest.main()
