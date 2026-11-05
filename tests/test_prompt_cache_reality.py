"""Prompt caching, second review: fixes found by checking the cache work against provider docs and
real request payloads. Offline: urllib.request.urlopen is replaced by a fake model.

- OpenAI's extended retention (prompt_cache_retention) goes only to the models OpenAI lists.
- DeepSeek and Z.ai cache reads are priced from their published cache-hit prices.
- Input totals in /cost, /stats, print mode and the query engine include cached tokens again.
- Resuming a session keeps the 1-hour cache write count.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.request
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from src import prompt_cache
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig, ModelPricing
from src.openai_compat import OpenAICompatClient

_CLEAN = {k: '' for k in ('SIGIL_PROMPT_CACHE', 'SIGIL_PROMPT_CACHE_TTL', 'SIGIL_ANTHROPIC_NATIVE',
                          'SIGIL_ANTHROPIC_STREAM', 'SIGIL_EXCLUDE_PARAMS', 'SIGIL_FAST_MODEL', 'SIGIL_EFFORT_LEVEL')}
MESSAGES = [{'role': 'system', 'content': 'You are a coding agent.'}, {'role': 'user', 'content': 'hi'}]


class _Response(io.BytesIO):
    headers = {'Content-Type': 'application/json'}

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()


class FakeModel:
    """Answers each request with the next reply (the last one repeats) and keeps the request bodies."""

    def __init__(self, *replies: dict) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001, ANN204
        self.bodies.append(json.loads(req.data.decode('utf-8')))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return _Response(json.dumps(reply).encode('utf-8'))


def _chat(content: str = 'ok', usage: dict | None = None, tool: bool = False) -> dict:
    message: dict = {'role': 'assistant', 'content': content}
    if tool:
        message['tool_calls'] = [{'id': 'call_1', 'type': 'function',
                                  'function': {'name': 'read_file', 'arguments': '{"path": "a.txt"}'}}]
    return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if tool else 'stop'}],
            'usage': usage or {'prompt_tokens': 10, 'completion_tokens': 1}}


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, _CLEAN)
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        prompt_cache.set_session(clear=True)
        self.addCleanup(prompt_cache.set_session, clear=True)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name)
        (self.cwd / 'a.txt').write_text('hello\n', encoding='utf-8')

    def _agent(self, model: str, base_url: str, pricing: ModelPricing | None = None) -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model=model, base_url=base_url, api_key='test-key',
                                     pricing=pricing or ModelPricing()),
            runtime_config=AgentRuntimeConfig(cwd=self.cwd))


class OpenAIRetentionTests(_Base):
    """OpenAI lists the models with extended retention; any other model may reject the field, and a
    rejection drops every cache field (prompt_cache_key too) for the rest of the session."""

    LISTED = ('gpt-5.5', 'gpt-5.5-pro', 'gpt-5.4', 'gpt-5.2', 'gpt-5.1-codex-max', 'gpt-5.1', 'gpt-5.1-codex',
              'gpt-5.1-codex-mini', 'gpt-5.1-chat-latest', 'gpt-5', 'gpt-5-codex', 'gpt-4.1', 'gpt-4.1-2025-04-14')
    NOT_LISTED = ('gpt-5-mini', 'gpt-5-nano', 'gpt-5.3', 'gpt-4.1-mini', 'gpt-4.1-nano', 'gpt-4o', 'gpt-5.6')

    def _body(self, model: str) -> dict:
        fake = FakeModel(_chat())
        with mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
            OpenAICompatClient(ModelConfig(model=model, base_url='https://api.openai.com/v1')).complete(MESSAGES, [])
        return fake.bodies[-1]

    def test_retention_only_for_listed_models(self) -> None:
        for model in self.LISTED:
            body = self._body(model)
            self.assertEqual(body.get('prompt_cache_retention'), '24h', model)
            self.assertRegex(body.get('prompt_cache_key', ''), r'^sigil-[0-9a-f]{32}$', model)
        for model in self.NOT_LISTED:
            body = self._body(model)
            self.assertNotIn('prompt_cache_retention', body, model)
            self.assertRegex(body.get('prompt_cache_key', ''), r'^sigil-[0-9a-f]{32}$', model)

    def test_short_ttl_sends_no_retention(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_TTL': '5m'}):
            self.assertNotIn('prompt_cache_retention', self._body('gpt-5.5'))


class PublishedCacheReadPriceTests(_Base):
    """DeepSeek and Z.ai publish cache-hit prices (api-docs.deepseek.com/quick_start/pricing,
    docs.z.ai/guides/overview/pricing); reads were charged at the full input price."""

    def test_ratios_from_the_price_pages(self) -> None:
        cases = {
            ('https://api.deepseek.com', 'deepseek-flash'): 0.003 / 0.15,
            ('https://api.deepseek.com', 'deepseek-v4-pro'): 0.022 / 0.66,
            ('https://api.z.ai/api/paas/v4', 'glm-5.2'): 0.26 / 1.4,
            ('https://api.z.ai/api/paas/v4', 'glm-5.1'): 0.26 / 1.4,
            ('https://api.z.ai/api/paas/v4', 'glm-5'): 0.2 / 1.0,
            ('https://open.bigmodel.cn/api/paas/v4', 'glm-4.7'): 0.11 / 0.6,
            ('https://api.z.ai/api/paas/v4', 'glm-4.6'): 0.11 / 0.6,
            ('https://api.z.ai/api/paas/v4', 'glm-5.3-flashx-legal'): 0.075 / 0.37,
        }
        for (url, model), ratio in cases.items():
            got = prompt_cache.price_ratios(url, model)
            self.assertAlmostEqual(got.read, ratio, places=6, msg=model)
            self.assertEqual((got.write_5m, got.write_1h), (1.0, 1.0), model)  # no write surcharge listed
        for url, model in (('https://api.deepseek.com', 'deepseek-chat'), ('https://api.z.ai/api/paas/v4', 'glm-9')):
            self.assertIsNone(prompt_cache.price_ratios(url, model).read, model)  # unknown model: input price

    def test_cost_of_a_cached_turn_through_the_agent(self) -> None:
        from src.main import _cache_pricing

        base = ModelPricing(input_cost_per_million_tokens_usd=1.32, output_cost_per_million_tokens_usd=0)
        pricing = _cache_pricing(base, Namespace(model='deepseek-v4-pro'), 'https://api.deepseek.com')
        fake = FakeModel(_chat('done', {'prompt_tokens': 1_000_000, 'completion_tokens': 0,
                                        'prompt_cache_hit_tokens': 900_000, 'prompt_cache_miss_tokens': 100_000}))
        with mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
            result = self._agent('deepseek-v4-pro', 'https://api.deepseek.com', pricing).run('hi')
        self.assertAlmostEqual(result.total_cost_usd, 0.1 * 1.32 + 0.9 * 1.32 * (0.022 / 0.66), places=6)


class InputTotalsIncludeCacheTests(_Base):
    """Since cached tokens are split out of input_tokens, 'total input' must add them back."""

    def _run(self) -> tuple[LocalCodingAgent, object]:
        fake = FakeModel(_chat('reading', {'prompt_tokens': 4020, 'completion_tokens': 5,
                                           'prompt_tokens_details': {'cached_tokens': 0}}, tool=True),
                         _chat('done', {'prompt_tokens': 4130, 'completion_tokens': 5,
                                        'prompt_tokens_details': {'cached_tokens': 4000}}))
        agent = self._agent('gpt-5.5', 'https://api.openai.com/v1')
        with mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
            result = agent.run('read a.txt')
        return agent, result

    def test_cost_stats_and_print_mode(self) -> None:
        from src.agent_slash_commands import _handle_cost, _handle_stats
        from src.main import _print_agent_result

        agent, result = self._run()
        self.assertEqual((agent.cumulative_usage.prompt_tokens, agent.cumulative_usage.input_tokens), (8150, 4150))
        cost = _handle_cost(agent, '', '/cost').output
        self.assertIn('Total input tokens:    8,150', cost)
        self.assertIn('Uncached input tokens: 4,150', cost)
        self.assertIn('Input (cumul.):  8,150 tokens', _handle_stats(agent, '', '/stats').output)
        out = io.StringIO()
        with redirect_stdout(out):
            _print_agent_result(result, show_transcript=False)
        self.assertIn('input_tokens_cumulative=8150', out.getvalue())

    def test_query_engine_turn_usage(self) -> None:
        from src.query_engine import QueryEnginePort

        fake = FakeModel(_chat('done', {'prompt_tokens': 1000, 'completion_tokens': 3,
                                        'prompt_tokens_details': {'cached_tokens': 600}}))
        agent = self._agent('gpt-5.5', 'https://api.openai.com/v1')
        with mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
            turn = QueryEnginePort.from_runtime_agent(agent).submit_message('hi')
        self.assertEqual((turn.usage.input_tokens, turn.usage.output_tokens), (1000, 3))


class ResumeKeepsOneHourWritesTests(_Base):
    def test_resume_keeps_the_1h_write_count(self) -> None:
        from src.session_store import load_agent_session

        def native(text: str, written_1h: int) -> dict:
            return {'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
                    'usage': {'input_tokens': 5, 'output_tokens': 1, 'cache_read_input_tokens': 0,
                              'cache_creation_input_tokens': written_1h,
                              'cache_creation': {'ephemeral_5m_input_tokens': 0,
                                                 'ephemeral_1h_input_tokens': written_1h}}}

        with mock.patch.object(urllib.request, 'urlopen', side_effect=FakeModel(native('one', 4000))):
            first = self._agent('claude-sonnet-5-5', 'https://api.anthropic.com')
            result = first.run('hi')
        self.assertEqual(first.cumulative_usage.cache_creation_1h_input_tokens, 4000)
        resumed = self._agent('claude-sonnet-5-5', 'https://api.anthropic.com')  # a new process: -c
        stored = load_agent_session(result.session_id, directory=resumed.runtime_config.session_directory)
        with mock.patch.object(urllib.request, 'urlopen', side_effect=FakeModel(native('two', 100))):
            resumed.resume('again', stored)
        self.assertEqual(resumed.cumulative_usage.cache_creation_1h_input_tokens, 4100)
        self.assertEqual(resumed.cumulative_usage.cache_creation_input_tokens, 4100)


if __name__ == '__main__':
    unittest.main()
