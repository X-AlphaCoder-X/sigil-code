"""3.5 print mode (-p) and --output-format text|json|stream-json, through main()
and a fake model server."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.main import _translate_print_argv, _subcommand_names, build_parser, main


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
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 11, 'completion_tokens': 7, 'total_tokens': 18}}


def tool_reply(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


class FakeStdin(io.StringIO):
    def __init__(self, text: str, tty: bool = False) -> None:
        super().__init__(text)
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def run_main(argv: list[str], responses: list[dict[str, object]], *, stdin_text: str = '',
             tty: bool = True) -> tuple[int, str, list[dict[str, object]]]:
    recorded: list[dict[str, object]] = []
    buffer = io.StringIO()
    with patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
            patch('sys.stdin', FakeStdin(stdin_text, tty=tty)), redirect_stdout(buffer):
        code = main(argv)
    return code, buffer.getvalue(), recorded


class TranslationTests(unittest.TestCase):
    def test_flag_becomes_print_subcommand(self) -> None:
        names = _subcommand_names(build_parser())
        self.assertEqual(_translate_print_argv(['-p', 'hi', '--model', 'm'], names),
                         ['print', 'hi', '--model', 'm'])
        self.assertEqual(_translate_print_argv(['--output-format', 'json', '--print', 'q'], names),
                         ['print', 'q', '--output-format', 'json'])
        self.assertEqual(_translate_print_argv(['-p'], names), ['print'])
        self.assertEqual(_translate_print_argv(['agent', 'x'], names), ['agent', 'x'])


class PrintModeTests(unittest.TestCase):
    def test_text_output_is_only_the_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_main(['-p', 'say hi', '--cwd', tmp], [text_reply('Hello there.')])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), 'Hello there.')

    def test_piped_stdin_goes_before_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, recorded = run_main(['-p', 'explain this log', '--cwd', tmp], [text_reply('It failed.')],
                                           stdin_text='ERROR disk full\n', tty=False)
        self.assertEqual(code, 0)
        users = [m['content'] for m in recorded[0]['messages'] if m.get('role') == 'user']
        self.assertEqual(users[-1], 'ERROR disk full\n\nexplain this log')

    def test_stdin_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, recorded = run_main(['-p', '--cwd', tmp], [text_reply('ok')],
                                           stdin_text='just stdin', tty=False)
        self.assertEqual(code, 0)
        users = [m['content'] for m in recorded[0]['messages'] if m.get('role') == 'user']
        self.assertEqual(users[-1], 'just stdin')

    def test_no_prompt_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stderr', io.StringIO()):
            code, _out, recorded = run_main(['-p', '--cwd', tmp], [], tty=True)
        self.assertEqual(code, 2)
        self.assertEqual(recorded, [])

    def test_json_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_main(['-p', 'q', '--output-format', 'json', '--cwd', tmp], [text_reply('answer')])
            payload = json.loads(out)
            saved = (Path(tmp) / '.port_sessions' / 'agent' / f"{payload['session_id']}.json").exists()
        self.assertTrue(saved)
        self.assertEqual(code, 0)
        for key in ('result', 'session_id', 'total_cost_usd', 'usage', 'stop_reason', 'is_error', 'num_turns'):
            self.assertIn(key, payload)
        self.assertEqual(payload['type'], 'result')
        self.assertEqual(payload['result'], 'answer')
        self.assertFalse(payload['is_error'])
        self.assertEqual(payload['usage']['output_tokens'], 7)

    def test_stream_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'a.txt').write_text('x', encoding='utf-8')
            code, out, _ = run_main(
                ['-p', 'list', '--output-format', 'stream-json', '--cwd', tmp],
                [tool_reply('list_dir', {'path': '.'}), text_reply('one file')])
        self.assertEqual(code, 0)
        lines = [json.loads(line) for line in out.strip().splitlines()]
        self.assertEqual(lines[0]['type'], 'system')
        self.assertEqual(lines[0]['subtype'], 'init')
        self.assertEqual(lines[-1]['type'], 'result')
        self.assertEqual(lines[-1]['result'], 'one file')
        self.assertEqual(lines[-1]['session_id'], lines[0]['session_id'])
        types = {line.get('type') for line in lines[1:-1]}
        self.assertTrue(types, 'expected run events between init and result')

    def test_budget_stop_exits_non_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_main(
                ['-p', 'loop', '--max-turns', '1', '--output-format', 'json', '--cwd', tmp],
                [tool_reply('list_dir', {'path': '.'}), text_reply('never')])
        self.assertNotEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(payload['is_error'])
        self.assertEqual(payload['stop_reason'], 'max_turns')

    def test_backend_error_exits_non_zero(self) -> None:
        def _boom(request_obj, timeout=None):  # noqa: ANN001
            raise OSError('connection refused')

        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=_boom), \
                patch('sys.stdin', FakeStdin('', tty=True)), patch('time.sleep'), redirect_stdout(buffer):
            code = main(['-p', 'q', '--output-format', 'json', '--cwd', tmp])
        self.assertNotEqual(code, 0)
        self.assertTrue(json.loads(buffer.getvalue().strip().splitlines()[-1])['is_error'])


if __name__ == '__main__':
    unittest.main()
