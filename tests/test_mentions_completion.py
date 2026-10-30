"""3.7 @file mentions and tab completion (/commands, @paths)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.document import Document

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.completion import build_completer, complete_path, complete_slash, slash_command_names
from src.main import _run_agent_chat_loop


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


def make_agent(ws: Path, added: tuple[Path, ...] = ()) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, additional_working_directories=added,
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def chat_once(agent: LocalCodingAgent, prompt: str) -> list[str]:
    prompts = iter([prompt])
    outputs: list[str] = []

    def _input(_label: str) -> str:
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError

    _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                         input_func=_input, output_func=outputs.append,
                         result_printer=lambda _r, show_transcript: None)
    return outputs


class CompletionTests(unittest.TestCase):
    def test_slash_completion(self) -> None:
        names = slash_command_names()
        self.assertIn('/help', complete_slash('/he', names))
        self.assertIn('/exit', complete_slash('/ex', names))
        self.assertIn('/bashes', complete_slash('/bas', names))
        self.assertEqual(complete_slash('/help me', names), [])
        self.assertEqual(complete_slash('hello', names), [])

    def test_custom_commands_are_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / '.sigil' / 'commands').mkdir(parents=True)
            (ws / '.sigil' / 'commands' / 'deploy-now.md').write_text('go', encoding='utf-8')
            names = slash_command_names(make_agent(ws))
        self.assertIn('/deploy-now', complete_slash('/deploy', names))

    def test_path_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'src' / 'pkg').mkdir(parents=True)
            (ws / 'src' / 'main.py').write_text('x', encoding='utf-8')
            (ws / 'setup.cfg').write_text('x', encoding='utf-8')
            (ws / '.hidden').write_text('x', encoding='utf-8')
            self.assertEqual(complete_path('s', [ws]), ['src/', 'setup.cfg'])
            self.assertEqual(complete_path('src/', [ws]), ['src/pkg/', 'src/main.py'])
            self.assertEqual(complete_path('src/ma', [ws]), ['src/main.py'])
            self.assertNotIn('.hidden', complete_path('', [ws]))
            self.assertIn('.hidden', complete_path('.h', [ws]))

    def test_prompt_toolkit_completer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'notes.md').write_text('x', encoding='utf-8')
            completer = build_completer(make_agent(ws))
            at = [c.text for c in completer.get_completions(Document('read @no'), None)]
            slash = [c.text for c in completer.get_completions(Document('/hel'), None)]
            email = [c.text for c in completer.get_completions(Document('mail me@no'), None)]
        self.assertEqual(at, ['notes.md'])
        self.assertIn('/help', slash)
        self.assertEqual(email, [])


class MentionTests(unittest.TestCase):
    def test_mention_adds_file_content(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('ok')], recorded)):
            ws = Path(tmp)
            (ws / 'spec.txt').write_text('the spec says 42', encoding='utf-8')
            outputs = chat_once(make_agent(ws), 'summarize @spec.txt and @missing.txt, mail a@b.com')
        user = [m['content'] for m in recorded[0]['messages'] if m.get('role') == 'user'][-1]
        self.assertIn('summarize @spec.txt and @missing.txt, mail a@b.com', user)
        self.assertIn('<file path="spec.txt">', user)
        self.assertIn('the spec says 42', user)
        self.assertNotIn('<file path="missing.txt">', user)
        self.assertTrue(any('attached: spec.txt' in line for line in outputs))

    def test_mention_respects_path_access_and_size(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside, \
                patch.dict(os.environ, {'SIGIL_OTHER_DIRECTORIES': ''}), \
                patch('src.custom_commands.MAX_MENTION_CHARS', 10), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('ok')], recorded)):
            ws = Path(tmp)
            secret = Path(outside) / 'secret.txt'
            secret.write_text('outside secret', encoding='utf-8')
            (ws / 'big.txt').write_text('0123456789ABCDEF', encoding='utf-8')
            chat_once(make_agent(ws), f'look @{secret.as_posix()} @big.txt')
        user = [m['content'] for m in recorded[0]['messages'] if m.get('role') == 'user'][-1]
        self.assertNotIn('outside secret', user)
        self.assertIn('0123456789', user)
        self.assertNotIn('ABCDEF', user)
        self.assertIn('truncated', user)


if __name__ == '__main__':
    unittest.main()
