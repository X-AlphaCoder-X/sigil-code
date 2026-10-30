"""Thinking / effort reaches each provider in the form its docs require (checked 2026-10-29):
reasoning_effort 'none' with a one-time fallback to 'low', and per-model Claude thinking."""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock
from urllib import error

from src.agent_types import ModelConfig
from src.openai_compat import OpenAICompatClient
from src.providers import claude_thinking, claude_version


class _Resp:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


OK = {'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]}


class OpenAIEffortTests(unittest.TestCase):
    def test_off_sends_none_and_falls_back_once(self) -> None:
        sent: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            body = json.loads(req.data.decode('utf-8'))
            sent.append(body)
            if body.get('reasoning_effort') == 'none':
                raise error.HTTPError(req.full_url, 400, 'bad', {}, __import__('io').BytesIO(
                    b'{"error": "Unsupported value: reasoning_effort none"}'))
            return _Resp(OK)

        with mock.patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': '', 'SIGIL_EFFORT_FIELD': ''}), \
                mock.patch('src.openai_compat.request.urlopen', side_effect=fake):
            os.environ.pop('SIGIL_EFFORT_LEVEL', None)
            client = OpenAICompatClient(ModelConfig(model='gpt-5', base_url='https://api.openai.com/v1', effort='off'))
            client.complete([{'role': 'user', 'content': 'hi'}], [])
            client.complete([{'role': 'user', 'content': 'again'}], [])
        self.assertEqual([b.get('reasoning_effort') for b in sent], ['none', 'low', 'low'])

    def test_none_accepted_is_kept(self) -> None:
        sent: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            sent.append(json.loads(req.data.decode('utf-8')))
            return _Resp(OK)

        with mock.patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': '', 'SIGIL_EFFORT_FIELD': ''}), \
                mock.patch('src.openai_compat.request.urlopen', side_effect=fake):
            os.environ.pop('SIGIL_EFFORT_LEVEL', None)
            client = OpenAICompatClient(ModelConfig(model='gpt-5.5', base_url='https://api.openai.com/v1', effort='off'))
            client.complete([{'role': 'user', 'content': 'hi'}], [])
        self.assertEqual(sent[0]['reasoning_effort'], 'none')


class ClaudeRoutingTests(unittest.TestCase):
    def test_versions(self) -> None:
        self.assertEqual(claude_version('claude-opus-5-5'), ('opus', 5, 5))
        self.assertEqual(claude_version('claude-sonnet-4-20250514'), ('sonnet', 4, 0))
        self.assertEqual(claude_version('claude-haiku-4-5-20251001'), ('haiku', 4, 5))
        self.assertEqual(claude_version('claude-3-7-sonnet-latest'), ('sonnet', 3, 7))
        self.assertEqual(claude_version('anthropic/claude-fable-5-1'), ('fable', 5, 1))
        self.assertIsNone(claude_version('gpt-5'))

    def test_future_models_use_the_current_form(self) -> None:
        thinking, effort, temperature_ok = claude_thinking('claude-opus-6', 'xhigh')
        self.assertEqual((thinking, effort, temperature_ok), ({'type': 'adaptive'}, 'xhigh', False))
        self.assertEqual(claude_thinking('claude-opus-4-6', 'medium'), ({'type': 'adaptive'}, 'medium', True))
        self.assertEqual(claude_thinking('claude-opus-4-7', None), (None, None, False))
        self.assertEqual(claude_thinking('claude-haiku-5-5', 'off'), ({'type': 'disabled'}, None, False))
        self.assertEqual(claude_thinking('claude-fable-5-1', 'off'), ({'type': 'adaptive'}, 'low', False))

    def test_native_request_through_a_fake_server(self) -> None:
        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig
        import tempfile
        from pathlib import Path

        sent: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            sent.append(json.loads(req.data.decode('utf-8')))
            return _Resp({'id': 'm', 'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': 'ok'}],
                          'stop_reason': 'end_turn', 'usage': {'input_tokens': 3, 'output_tokens': 1}})

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': '', 'SIGIL_ANTHROPIC_NATIVE': '1'}), \
                mock.patch('src.openai_compat.request.urlopen', side_effect=fake):
            os.environ.pop('SIGIL_EFFORT_LEVEL', None)
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='claude-sonnet-5', base_url='https://api.anthropic.com/v1',
                                         api_key='sk-ant-x', effort='high', temperature=0.3),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp), session_directory=Path(tmp) / 's',
                                                  nudge_strategy='off'))
            agent.run('hi')
            agent.run('/effort off')
            agent.run('hi again')
        self.assertEqual((sent[0]['thinking'], sent[0]['output_config']), ({'type': 'adaptive'}, {'effort': 'high'}))
        self.assertNotIn('temperature', sent[0])
        self.assertEqual(sent[-1]['thinking'], {'type': 'disabled'})


class EffortNoneAliasTests(unittest.TestCase):
    """'none' = 'off' everywhere a level is typed."""

    def test_slash_env_and_flag(self) -> None:
        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig
        from src.main import build_parser
        import tempfile
        from pathlib import Path

        sent: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            sent.append(json.loads(req.data.decode('utf-8')))
            return _Resp(OK)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': '', 'SIGIL_EFFORT_FIELD': ''}), \
                mock.patch('src.openai_compat.request.urlopen', side_effect=fake):
            os.environ.pop('SIGIL_EFFORT_LEVEL', None)
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='gpt-5.5', base_url='https://api.openai.com/v1'),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp), session_directory=Path(tmp) / 's',
                                                  nudge_strategy='off'))
            reply = agent.run('/effort none')
            agent.run('hi')
            os.environ['SIGIL_EFFORT_LEVEL'] = 'none'
            OpenAICompatClient(ModelConfig(model='gpt-5.5', base_url='https://api.openai.com/v1')).complete(
                [{'role': 'user', 'content': 'env'}], [])
        self.assertIn('Set effort level to: off', reply.final_output)
        self.assertEqual(agent.model_config.effort, 'off')
        self.assertEqual([b.get('reasoning_effort') for b in sent], ['none', 'none'])
        for level in ('none', 'off', 'xhigh', 'max'):
            self.assertEqual(build_parser().parse_args(['agent', 'hi', '--effort', level]).effort, level)


if __name__ == '__main__':
    unittest.main()
