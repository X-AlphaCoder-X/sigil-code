"""Prompt caching when the model or endpoint changes, and the one-time retry without cache fields.
Offline: urllib.request.urlopen is replaced by a fake model, or a local fake server answers.

- Cache prices are derived again after /model, /profile, agent-resume --model/--base-url, a subagent
  model and a fallback model; a price set with a --cache-*-cost-per-million flag is kept.
- Each turn is priced by the endpoint that answered it, and earlier turns keep their cost.
- A background agent on another model or endpoint gets that model's cache prices.
- A 400/422 that does not name a cache field is not retried; one that does is retried once.
- A background agent started from the chat sends the cache fields, honours /cache and reports its
  usage and cost back (end to end, with a real worker process).
- SIGIL_PROMPT_CACHE_PROVIDER names the provider of a server Sigil does not recognise.
"""
from __future__ import annotations

import http.server
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from unittest import mock

from src import prompt_cache
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig, ModelPricing
from src.anthropic_client import AnthropicClient
from src.openai_compat import OpenAICompatClient, OpenAICompatError

_CLEAN = {k: '' for k in ('SIGIL_PROMPT_CACHE', 'SIGIL_PROMPT_CACHE_TTL', 'SIGIL_PROMPT_CACHE_PROVIDER',
                          'SIGIL_ANTHROPIC_NATIVE', 'SIGIL_ANTHROPIC_STREAM', 'SIGIL_EXCLUDE_PARAMS',
                          'SIGIL_FAST_MODEL', 'SIGIL_EFFORT_LEVEL')}
MESSAGES = [{'role': 'system', 'content': 'You are a coding agent.'}, {'role': 'user', 'content': 'hi'}]
MILLION = 1_000_000


class _Response(io.BytesIO):
    headers = {'Content-Type': 'application/json'}

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()


class RoutedModel:
    """Fake urlopen. handler(url, body) -> (status, reply dict); keeps (url, body) of each request."""

    def __init__(self, handler) -> None:  # noqa: ANN001
        self.handler = handler
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001, ANN204
        body = json.loads(req.data.decode('utf-8'))
        self.requests.append((req.full_url, body))
        status, reply = self.handler(req.full_url, body)
        data = json.dumps(reply).encode('utf-8')
        if status != 200:
            raise urllib.error.HTTPError(req.full_url, status, 'error', {}, io.BytesIO(data))
        return _Response(data)


def _chat(content: str = 'ok', usage: dict | None = None) -> dict:
    return {'choices': [{'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
            'usage': usage or {'prompt_tokens': 10, 'completion_tokens': 1}}


def _native(text: str = 'ok', cache_read: int = 0) -> dict:
    return {'content': [{'type': 'text', 'text': text}], 'stop_reason': 'end_turn',
            'usage': {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': cache_read,
                      'cache_creation_input_tokens': 0}}


def _derived(input_price: float, base_url: str, model: str, **flags) -> ModelPricing:  # noqa: ANN003
    """Pricing as `sigil agent` builds it (main._cache_pricing)."""
    from src.main import _cache_pricing

    return _cache_pricing(ModelPricing(input_cost_per_million_tokens_usd=input_price),
                          Namespace(model=model, **flags), base_url)


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
        with prompt_cache._LOCK:
            prompt_cache._WARNED.clear()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()

    def _agent(self, model: str, base_url: str, pricing: ModelPricing | None = None,
               **config) -> LocalCodingAgent:  # noqa: ANN003
        return LocalCodingAgent(
            model_config=ModelConfig(model=model, base_url=base_url, api_key='test-key',
                                     pricing=pricing or ModelPricing(), **config),
            runtime_config=AgentRuntimeConfig(cwd=self.cwd))


# ---------------------------------------------------------------- OQ3: retry only for cache errors

class RetryOnlyForCacheErrorsTests(_Base):
    def _openai(self, status: int, error: dict) -> tuple[RoutedModel, object]:
        def handler(url, body):  # noqa: ANN001, ANN202
            if prompt_cache.has_cache_fields(body):
                return status, error
            return 200, _chat()
        fake = RoutedModel(handler)
        client = OpenAICompatClient(ModelConfig(model='gpt-5.5', base_url='https://api.openai.com/v1'))
        return fake, client

    def test_an_unrelated_400_is_sent_once_and_logged(self) -> None:
        from src.debug_log import get_logger

        fake, client = self._openai(400, {'error': {'message': 'context too long: 300000 > 272000 tokens'}})
        with mock.patch.object(urllib.request, 'urlopen', side_effect=fake), \
                self.assertLogs(get_logger('prompt_cache').name, level='INFO') as logs:
            with self.assertRaises(OpenAICompatError):
                client.complete(MESSAGES, [])
        self.assertEqual(len(fake.requests), 1)
        self.assertTrue(any('not retried' in line for line in logs.output), logs.output)
        self.assertIsNone(prompt_cache.rejected_reason('https://api.openai.com/v1', 'gpt-5.5'))

    def test_errors_that_name_a_cache_field_are_retried_once(self) -> None:
        from src.debug_log import get_logger

        cases = (
            (400, {'error': {'message': 'Unrecognized request argument supplied: prompt_cache_retention'}}),
            (422, {'detail': [{'type': 'extra_forbidden', 'loc': ['body', 'prompt_cache_key'],
                               'msg': 'Extra inputs are not permitted'}]}),
            (400, {'error': {'message': 'extra fields not permitted'}}),  # unknown field, no name
        )
        for status, error in cases:
            prompt_cache.set_session(clear=True)
            fake, client = self._openai(status, error)
            with mock.patch.object(urllib.request, 'urlopen', side_effect=fake), \
                    self.assertLogs(get_logger('prompt_cache').name, level='INFO') as logs:
                self.assertEqual(client.complete(MESSAGES, []).content, 'ok')
            self.assertEqual(len(fake.requests), 2, error)
            self.assertFalse(prompt_cache.has_cache_fields(fake.requests[1][1]))
            self.assertTrue(any('retrying once without them' in line for line in logs.output), logs.output)

    def test_an_unknown_field_error_about_another_field_is_not_retried(self) -> None:
        payload = {'model': 'm', 'messages': [], 'temperature': 0.2, 'prompt_cache_key': 'sigil-x'}
        other = OpenAICompatError('HTTP 400 from local model backend: Unrecognized request argument supplied: '
                                  'temperature')
        ours = OpenAICompatError('HTTP 400 from local model backend: Unrecognized request argument supplied: '
                                 'prompt_cache_key')
        self.assertFalse(prompt_cache.is_cache_rejection(other, payload))
        self.assertTrue(prompt_cache.is_cache_rejection(ours, payload))
        # a field we did not send does not count
        self.assertFalse(prompt_cache.is_cache_rejection(
            OpenAICompatError('HTTP 400 from x: keep_alive is invalid'), payload))

    def test_native_anthropic_retries_only_cache_control_errors(self) -> None:
        for message, expected_requests in (('messages: text content blocks must be non-empty', 1),
                                           ('system.0.cache_control: Extra inputs are not permitted', 2)):
            prompt_cache.set_session(clear=True)

            def handler(url, body, message=message):  # noqa: ANN001, ANN202
                if prompt_cache.has_cache_fields(body):
                    return 400, {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}}
                return 200, _native()
            fake = RoutedModel(handler)
            client = AnthropicClient(ModelConfig(model='claude-sonnet-5-5', base_url='https://api.anthropic.com'))
            with mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
                if expected_requests == 1:
                    with self.assertRaises(OpenAICompatError):
                        client.complete(MESSAGES, [])
                else:
                    client.complete(MESSAGES, [])
            self.assertEqual(len(fake.requests), expected_requests, message)


# ---------------------------------------------------------------- OQ1: prices follow a model switch

class RepriceOnSwitchTests(_Base):
    ANTHROPIC = 'https://api.anthropic.com'

    def test_model_command_rederives_cache_prices(self) -> None:
        from src.agent_slash_commands import _handle_model

        agent = self._agent('claude-sonnet-4-5', self.ANTHROPIC, _derived(3.0, self.ANTHROPIC, 'claude-sonnet-4-5'))
        self.assertAlmostEqual(agent.model_config.pricing.cache_read_input_cost_per_million_tokens_usd, 0.3)
        _handle_model(agent, 'claude-fable-5-1', '/model claude-fable-5-1')
        pricing = agent.model_config.pricing
        self.assertAlmostEqual(pricing.cache_read_input_cost_per_million_tokens_usd, 3.0 * 0.025)
        self.assertAlmostEqual(pricing.cache_creation_input_cost_per_million_tokens_usd, 3.75)
        self.assertAlmostEqual(pricing.cache_creation_1h_input_cost_per_million_tokens_usd, 6.0)
        self.assertAlmostEqual(agent.tool_context.model_config.pricing.cache_read_input_cost_per_million_tokens_usd,
                               0.075)  # background agents and subagents see the new prices too
        fake = RoutedModel(lambda url, body: (200, _native('done', MILLION)))
        with mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
            result = agent.run('hi')
        self.assertAlmostEqual(result.total_cost_usd, 0.075, places=9)

    def test_a_price_set_with_a_flag_is_kept(self) -> None:
        pricing = _derived(3.0, self.ANTHROPIC, 'claude-sonnet-4-5', cache_read_cost_per_million=0.5)
        agent = self._agent('claude-sonnet-4-5', self.ANTHROPIC, pricing)
        agent.set_model('claude-fable-5-1')
        self.assertEqual(agent.model_config.pricing.cache_read_input_cost_per_million_tokens_usd, 0.5)
        self.assertAlmostEqual(agent.model_config.pricing.cache_creation_input_cost_per_million_tokens_usd, 3.75)

    def test_profile_switch_to_another_provider(self) -> None:
        from src.profiles import apply_profile

        agent = self._agent('gpt-5.5', 'https://api.openai.com/v1',
                            _derived(1.32, 'https://api.openai.com/v1', 'gpt-5.5'))
        self.assertAlmostEqual(agent.model_config.pricing.cache_read_input_cost_per_million_tokens_usd, 0.132)
        with mock.patch.dict(os.environ, {'SIGIL_API_KEY_API_DEEPSEEK_COM': 'deepseek-test-key'}):
            apply_profile(agent, 'ds', {'ds': {'base_url': 'https://api.deepseek.com', 'model': 'deepseek-v4-pro'}})
        pricing = agent.model_config.pricing
        self.assertAlmostEqual(pricing.cache_read_input_cost_per_million_tokens_usd, 1.32 * 0.022 / 0.66)
        self.assertAlmostEqual(pricing.cache_creation_input_cost_per_million_tokens_usd, 1.32)  # no surcharge

    def test_switch_to_a_model_with_no_published_price_unsets_the_derived_price(self) -> None:
        agent = self._agent('gpt-4o', 'https://api.openai.com/v1', _derived(2.0, 'https://api.openai.com/v1', 'gpt-4o'))
        self.assertAlmostEqual(agent.model_config.pricing.cache_read_input_cost_per_million_tokens_usd, 1.0)
        agent.set_model('my-finetune')  # unknown: charged at the input price, not gpt-4o's discount
        self.assertEqual(agent.model_config.pricing.cache_read_input_cost_per_million_tokens_usd, 0.0)

    def test_subagent_definition_model_gets_its_own_prices(self) -> None:
        agents = self.cwd / '.sigil' / 'agents'
        agents.mkdir(parents=True)
        (agents / 'cheap.md').write_text('---\nname: cheap\ndescription: cheap helper\nmodel: claude-fable-5-1\n'
                                         '---\nAnswer briefly.\n', encoding='utf-8')
        agent = self._agent('claude-sonnet-4-5', self.ANTHROPIC, _derived(3.0, self.ANTHROPIC, 'claude-sonnet-4-5'))
        seen: list[ModelConfig] = []
        real_run = LocalCodingAgent.run

        def spy(child, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            seen.append(child.model_config)
            return real_run(child, *args, **kwargs)
        with mock.patch.object(LocalCodingAgent, 'run', spy), \
                mock.patch.object(urllib.request, 'urlopen',
                                  side_effect=RoutedModel(lambda u, b: (200, _native('child done', MILLION)))):
            outcome = agent._execute_delegate_agent({'prompt': 'help', 'subagent_type': 'cheap'})
        self.assertTrue(outcome.ok, outcome.content)
        self.assertEqual(seen[-1].model, 'claude-fable-5-1')
        self.assertAlmostEqual(seen[-1].pricing.cache_read_input_cost_per_million_tokens_usd, 0.075)

    def test_earlier_turns_keep_their_price_after_a_switch(self) -> None:
        from src.session_store import load_agent_session

        agent = self._agent('claude-sonnet-4-5', self.ANTHROPIC, _derived(3.0, self.ANTHROPIC, 'claude-sonnet-4-5'))
        with mock.patch.object(urllib.request, 'urlopen',
                               side_effect=RoutedModel(lambda u, b: (200, _native('one', MILLION)))):
            first = agent.run('hi')
        self.assertAlmostEqual(first.total_cost_usd, 0.3, places=9)
        agent.set_model('claude-fable-5-1')
        stored = load_agent_session(first.session_id, directory=agent.runtime_config.session_directory)
        with mock.patch.object(urllib.request, 'urlopen',
                               side_effect=RoutedModel(lambda u, b: (200, _native('two', MILLION)))):
            second = agent.resume('again', stored)
        self.assertAlmostEqual(second.total_cost_usd, 0.3 + 0.075, places=9)  # not 2 x 0.075
        self.assertAlmostEqual(agent.cumulative_cost_usd, 0.375, places=9)

    def test_a_fallback_turn_is_priced_by_the_endpoint_that_answered(self) -> None:
        def handler(url, body):  # noqa: ANN001, ANN202
            if 'api.openai.com' in url:
                return 400, {'error': {'message': 'The model gpt-5.5 does not exist'}}
            return 200, _chat('from deepseek', {'prompt_tokens': MILLION, 'completion_tokens': 0,
                                                'prompt_cache_hit_tokens': 900_000,
                                                'prompt_cache_miss_tokens': 100_000})
        agent = self._agent('gpt-5.5', 'https://api.openai.com/v1',
                            _derived(1.32, 'https://api.openai.com/v1', 'gpt-5.5'),
                            fallback_models=('deepseek-v4-pro@https://api.deepseek.com',))
        fake = RoutedModel(handler)
        with mock.patch.dict(os.environ, {'SIGIL_API_KEY_API_DEEPSEEK_COM': 'deepseek-test-key'}), \
                mock.patch.object(urllib.request, 'urlopen', side_effect=fake):
            result = agent.run('hi')
        self.assertEqual(result.final_output, 'from deepseek')
        expected = 0.1 * 1.32 + 0.9 * 1.32 * (0.022 / 0.66)
        self.assertAlmostEqual(result.total_cost_usd, expected, places=9)
        usage_events = [e for e in result.events if e.get('type') == 'model_usage']
        self.assertEqual(usage_events[-1]['model'], 'deepseek-v4-pro')

    def test_agent_resume_with_another_model_and_no_price_flags(self) -> None:
        from src.main import _build_resumed_agent, build_parser

        url = 'https://api.openai.com/v1'
        agent = self._agent('gpt-4o', url, _derived(2.0, url, 'gpt-4o'))
        with mock.patch.object(urllib.request, 'urlopen', side_effect=RoutedModel(lambda u, b: (200, _chat()))):
            first = agent.run('hi')
        args = build_parser().parse_args(['agent-resume', first.session_id, 'again', '--cwd', str(self.cwd),
                                          '--model', 'gpt-5.5', '--base-url', url])
        resumed, _stored = _build_resumed_agent(args)
        pricing = resumed.model_config.pricing
        self.assertEqual(pricing.input_cost_per_million_tokens_usd, 2.0)  # not reset to 0 by the parser default
        self.assertAlmostEqual(pricing.cache_read_input_cost_per_million_tokens_usd, 0.2)  # gpt-5.5: 0.1 x input
        args = build_parser().parse_args(['agent-resume', first.session_id, 'again', '--cwd', str(self.cwd),
                                          '--model', 'gpt-4o', '--base-url', url, '--input-cost-per-million', '4',
                                          '--cache-write-cost-per-million', '9'])
        resumed, _stored = _build_resumed_agent(args)
        pricing = resumed.model_config.pricing
        self.assertAlmostEqual(pricing.cache_read_input_cost_per_million_tokens_usd, 2.0)  # 0.5 x the new input
        self.assertEqual(pricing.cache_creation_input_cost_per_million_tokens_usd, 9.0)  # the flag wins


# ---------------------------------------------------------------- OQ2: background agent prices

class _Record:
    background_id = 'bg_000000000001'
    pid = 1
    log_path = 'log'


class BackgroundAgentPricingTests(_Base):
    def _start(self, agent: LocalCodingAgent, arguments: dict) -> list[str]:
        from src.agent_tools import execute_tool

        captured: dict = {}

        def fake_launch(_runtime, command, **kwargs):  # noqa: ANN001, ANN003, ANN202
            captured['command'] = command
            return _Record()
        with mock.patch('src.background_runtime.BackgroundSessionRuntime.launch', fake_launch), \
                mock.patch.dict(os.environ, {'SIGIL_API_KEY_API_DEEPSEEK_COM': 'deepseek-test-key'}):
            result = execute_tool(agent.tool_registry, 'background_agent_start', arguments, agent.tool_context)
        self.assertTrue(result.ok, result.content)
        return captured['command']

    @staticmethod
    def _flag(command: list[str], name: str) -> float | None:
        return float(command[command.index(name) + 1]) if name in command else None

    def test_another_endpoint_gets_its_own_cache_prices(self) -> None:
        url = 'https://api.openai.com/v1'
        agent = self._agent('gpt-5.5', url, _derived(1.32, url, 'gpt-5.5'))
        command = self._start(agent, {'prompt': 'go', 'model': 'deepseek-v4-pro@https://api.deepseek.com'})
        self.assertAlmostEqual(self._flag(command, '--cache-read-cost-per-million'), 1.32 * 0.022 / 0.66)
        self.assertAlmostEqual(self._flag(command, '--cache-write-cost-per-million'), 1.32)

    def test_another_model_on_the_same_endpoint(self) -> None:
        url = 'https://api.openai.com/v1'
        agent = self._agent('gpt-4o', url, _derived(2.0, url, 'gpt-4o'))
        command = self._start(agent, {'prompt': 'go', 'model': 'gpt-5.5'})
        self.assertAlmostEqual(self._flag(command, '--cache-read-cost-per-million'), 0.2)

    def test_a_flag_price_still_goes_along(self) -> None:
        url = 'https://api.openai.com/v1'
        agent = self._agent('gpt-4o', url, _derived(2.0, url, 'gpt-4o', cache_read_cost_per_million=0.7))
        command = self._start(agent, {'prompt': 'go', 'model': 'gpt-5.5'})
        self.assertEqual(self._flag(command, '--cache-read-cost-per-million'), 0.7)


# ---------------------------------------------------------------- SIGIL_PROMPT_CACHE_PROVIDER

class CacheProviderOverrideTests(_Base):
    def test_applies_to_servers_not_recognised_by_host(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_PROVIDER': 'ollama'}):
            plan = prompt_cache.plan_for('http://10.0.0.5:9000/v1', 'qwen3')
            self.assertEqual(plan.provider, 'ollama')
            self.assertEqual(plan.body, {'keep_alive': '1h'})
            # a known host keeps its own provider
            self.assertEqual(prompt_cache.plan_for('https://api.openai.com/v1', 'gpt-5.5').provider, 'openai')
            self.assertEqual(prompt_cache.provider_of('http://10.0.0.5:9000/v1'), 'openai-compatible')

    def test_prices_follow_the_override(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_PROVIDER': 'openrouter'}):
            self.assertEqual(prompt_cache.price_ratios('http://proxy.internal/v1', 'anthropic/claude-sonnet-4.5'),
                             prompt_cache.CacheRatios(0.1, 1.25, 2.0))

    def test_an_unknown_value_warns_and_is_ignored(self) -> None:
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_PROVIDER': 'not-a-provider'}), \
                mock.patch('sys.stderr', stderr):
            self.assertEqual(prompt_cache.plan_for('http://10.0.0.5:9000/v1', 'm').provider, 'openai-compatible')
        self.assertIn('SIGIL_PROMPT_CACHE_PROVIDER', stderr.getvalue())


# ---------------------------------------------------------------- end to end: a real background agent

class FakeServer:
    """A local HTTP server answering chat completions; keeps (body, headers) of each POST."""

    def __init__(self, reply: bytes) -> None:
        self.requests: list[tuple[dict, dict]] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
                outer.requests.append((body, {k.lower(): v for k, v in self.headers.items()}))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(404)
                self.send_header('Content-Length', '0')
                self.end_headers()

            def log_message(self, *args) -> None:  # noqa: ANN002
                return None

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class BackgroundAgentEndToEndTests(_Base):
    """The parent runs on one model; the background agent on another model behind the same server
    (an OpenRouter-style proxy, named with SIGIL_PROMPT_CACHE_PROVIDER). /cache is set to on and 5m
    in the parent and must reach the worker."""

    def test_background_agent_sends_cache_fields_and_reports_usage(self) -> None:
        from src.agent_tools import execute_tool
        from src.background_runtime import BackgroundSessionRuntime

        server = FakeServer(json.dumps(_chat('background done', {
            'prompt_tokens': 1000, 'completion_tokens': 7, 'prompt_tokens_details': {'cached_tokens': 600}})).encode())
        self.addCleanup(server.close)
        env = {'SIGIL_PROMPT_CACHE_PROVIDER': 'openrouter', 'SIGIL_DISABLE_USER_SETTINGS': '1',
               'SIGIL_DETECT_MODEL': '0', 'SIGIL_FALLBACK_MODEL': '', 'OPENAI_MODEL': '', 'OPENAI_BASE_URL': ''}
        patch_env = mock.patch.dict(os.environ, env)
        patch_env.start()
        self.addCleanup(patch_env.stop)
        prompt_cache.set_session(mode='on', ttl='5m')  # /cache on 5m in the parent
        parent_pricing = _derived(2.0, server.url, 'openai/gpt-4o')
        parent_pricing = replace(parent_pricing, output_cost_per_million_tokens_usd=4.0)
        self.assertAlmostEqual(parent_pricing.cache_read_input_cost_per_million_tokens_usd, 1.0)  # gpt-4o: 0.5
        agent = self._agent('openai/gpt-4o', server.url, parent_pricing)

        launched: dict = {}
        real_launch = BackgroundSessionRuntime.launch

        def spy(runtime, command, **kwargs):  # noqa: ANN001, ANN003, ANN202
            launched['env'] = dict(kwargs.get('env') or {})
            launched['command'] = list(command)
            return real_launch(runtime, command, **kwargs)
        with mock.patch.object(BackgroundSessionRuntime, 'launch', spy):
            started = execute_tool(agent.tool_registry, 'background_agent_start',
                                   {'prompt': 'say done', 'model': 'anthropic/claude-sonnet-4.5', 'max_turns': 2},
                                   agent.tool_context)
        self.assertTrue(started.ok, started.content)
        background_id = started.metadata['background_id']
        runtime = BackgroundSessionRuntime(self.cwd / '.port_sessions' / 'background')
        deadline = time.monotonic() + 120
        result = None
        while time.monotonic() < deadline:
            result = runtime.read_result(background_id)
            if result is not None and runtime.load_record(background_id).status != 'running':
                break
            time.sleep(0.25)
        log = Path(runtime.log_path(background_id)).read_text(encoding='utf-8', errors='replace')
        self.assertIsNotNone(result, log[-3000:])

        # environment: the /cache choice went along, no value is the text "None"
        child_env = launched['env']
        self.assertEqual((child_env.get('SIGIL_PROMPT_CACHE'), child_env.get('SIGIL_PROMPT_CACHE_TTL')), ('on', '5m'))
        self.assertEqual(child_env.get('SIGIL_PROMPT_CACHE_PROVIDER'), 'openrouter')
        self.assertNotIn('None', [value for value in child_env.values()])
        self.assertNotIn('None', launched['command'])

        # requests: Claude behind OpenRouter gets markers (5m: no ttl field) and the session header
        self.assertTrue(server.requests, log[-3000:])
        body, headers = server.requests[0]
        self.assertEqual(body['model'], 'anthropic/claude-sonnet-4.5')
        markers = [part['cache_control'] for message in body['messages'] if isinstance(message.get('content'), list)
                   for part in message['content'] if isinstance(part, dict) and 'cache_control' in part]
        self.assertTrue(markers, body['messages'])
        self.assertTrue(all(marker == {'type': 'ephemeral'} for marker in markers), markers)
        self.assertLessEqual(len(markers), prompt_cache.MAX_BREAKPOINTS)
        self.assertRegex(headers.get('x-session-id', ''), r'^sigil-[0-9a-f]{32}$')

        # usage and cost come back: 400 uncached + 600 read at Claude's 0.1 x input, 7 output
        usage = result['usage']
        self.assertEqual((usage['input_tokens'], usage['cache_read_input_tokens'], usage['output_tokens']),
                         (400, 600, 7))
        expected = (400 * 2.0 + 600 * 0.2 + 7 * 4.0) / MILLION
        self.assertAlmostEqual(result['total_cost_usd'], expected, places=12)
        status = execute_tool(agent.tool_registry, 'background_agent_status', {'background_id': background_id},
                              agent.tool_context)
        self.assertIn('background done', status.content)
        self.assertIn('input_tokens=1000', status.content)  # cached included
        self.assertIn('cache_read=600', status.content)


if __name__ == '__main__':
    unittest.main()
