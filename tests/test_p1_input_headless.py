"""P1: multiline keys, saved prompt history, -c/--continue, --allowed-tools,
--disallowed-tools, --tools and --permission-mode (fake model server)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from src.agent_bar import AgentBar, build_prompt_kwargs
from src.main import _build_agent, _prompt_history, build_parser, main


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


class FakeStdin(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Event:
    def __init__(self, buffer: Buffer) -> None:
        self.current_buffer = buffer


def handler_for(bindings, keys: tuple[str, ...]):
    for binding in bindings.bindings:
        if tuple(str(k.value if hasattr(k, 'value') else k) for k in binding.keys) == keys:
            return binding
    raise AssertionError(f'no binding for {keys}')


def isolated_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env['SIGIL_DISABLE_USER_SETTINGS'] = '1'
    env.update(extra)
    return env


class MultilineAndHistoryTests(unittest.TestCase):
    def test_newline_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bindings = build_prompt_kwargs(AgentBar(Path(tmp)))['key_bindings']
        for keys in (('c-j',), ('escape', 'c-m')):
            buffer = Buffer(document=Document('first'))
            handler_for(bindings, keys).handler(_Event(buffer))
            self.assertEqual(buffer.text, 'first\n', keys)

    def test_backslash_enter_continues_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bindings = build_prompt_kwargs(AgentBar(Path(tmp)))['key_bindings']
        enter_bindings = [b for b in bindings.bindings
                          if tuple(str(getattr(k, 'value', k)) for k in b.keys) == ('c-m',)]
        continue_binding = next(b for b in enter_bindings if b.handler.__name__ == '_continue_line')
        buffer = Buffer(document=Document('line one \\'))
        continue_binding.handler(_Event(buffer))
        self.assertEqual(buffer.text, 'line one \n')

    def test_history_is_saved_to_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hist'
            with patch.dict(os.environ, {'SIGIL_HISTORY_FILE': str(path)}):
                history = _prompt_history()
                history.store_string('remember this prompt')
                again = list(_prompt_history().load_history_strings())
            self.assertIn('remember this prompt', again)
            with patch.dict(os.environ, {'SIGIL_HISTORY_FILE': ''}):
                self.assertEqual(type(_prompt_history()).__name__, 'InMemoryHistory')


class HeadlessFlagTests(unittest.TestCase):
    def run_main(self, argv: list[str], responses: list[dict[str, object]]):
        recorded: list[dict[str, object]] = []
        out = io.StringIO()
        with patch.dict(os.environ, isolated_env(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
                patch('sys.stdin', FakeStdin('')), redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue(), recorded

    def test_continue_with_print(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code1, _, _ = self.run_main(['-p', 'my name is Ada', '--cwd', tmp], [text_reply('hi Ada')])
            code2, out, recorded = self.run_main(['-c', '-p', 'what is my name?', '--cwd', tmp],
                                                 [text_reply('Ada')])
        self.assertEqual((code1, code2), (0, 0))
        contents = [m.get('content') for m in recorded[0]['messages']]
        self.assertIn('my name is Ada', contents)
        self.assertIn('what is my name?', contents)

    def test_top_level_continue_resumes_chat(self) -> None:
        args = build_parser().parse_args(['-c'])
        self.assertTrue(args.top_continue)
        seen: dict[str, object] = {}

        def _fake_loop(agent, *, initial_prompt, resume_session_id, show_transcript, **_kw):
            seen['resume'] = resume_session_id
            return 0

        with patch('src.main._run_agent_chat_loop', side_effect=_fake_loop), \
                patch('src.main._build_agent', return_value=object()):
            main(['--continue'])
        self.assertEqual(seen['resume'], 'latest')

    def test_allowed_tools_skip_the_write_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, _, recorded = self.run_main(
                ['-p', 'write', '--cwd', tmp, '--allowed-tools', 'Write(notes/**) Read'],
                [tool_reply('write_file', {'path': 'notes/a.txt', 'content': 'ok'}), text_reply('done')])
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / 'notes' / 'a.txt').exists())

    def test_disallowed_and_tools_limit_the_tool_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, recorded = self.run_main(
                ['-p', 'x', '--cwd', tmp, '--disallowed-tools', 'Bash', '--disallowed-tools', 'WebFetch'],
                [text_reply('ok')])
            names = {t['function']['name'] for t in recorded[0]['tools']}
            self.assertNotIn('bash', names)
            self.assertNotIn('web_fetch', names)
            self.assertIn('read_file', names)
            _, _, recorded = self.run_main(['-p', 'x', '--cwd', tmp, '--tools', 'Read,Grep'], [text_reply('ok')])
            self.assertEqual({t['function']['name'] for t in recorded[0]['tools']}, {'read_file', 'grep_search'})

    def test_disallowed_rule_with_specifier_denies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / '.env').write_text('SECRET', encoding='utf-8')
            _, _, recorded = self.run_main(
                ['-p', 'x', '--cwd', tmp, '--disallowed-tools', 'Read(./.env)'],
                [tool_reply('read_file', {'path': '.env'}), text_reply('ok')])
        tool_text = [str(m.get('content')) for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any('Denied by permission rule' in t for t in tool_text))

    def test_permission_mode_flag_and_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated_env(), clear=True):
            agent = _build_agent(build_parser().parse_args(['agent', 'x', '--cwd', tmp, '--permission-mode', 'plan']))
            self.assertEqual(agent.permission_mode, 'plan')
            (Path(tmp) / '.sigil').mkdir()
            (Path(tmp) / '.sigil' / 'settings.json').write_text(
                json.dumps({'permissions': {'defaultMode': 'acceptEdits'}}), encoding='utf-8')
            agent = _build_agent(build_parser().parse_args(['agent', 'x', '--cwd', tmp]))
            self.assertEqual(agent.permission_mode, 'acceptEdits')
            self.assertTrue(agent.tool_context.permissions.allow_file_write)


if __name__ == '__main__':
    unittest.main()
