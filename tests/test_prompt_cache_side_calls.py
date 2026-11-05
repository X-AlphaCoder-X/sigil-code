"""Prompt caching, third review: prices that follow the model, side calls in the totals, /compact
keeping the totals, and summary calls that reuse the main conversation's cache.
Offline: urllib.request.urlopen is replaced by a fake model (and a fake /models list).

- Input and output prices follow a model switch (server price list or the modelPrices table); a
  price given with a flag is kept; a fallback turn is priced with the fallback's own prices.
- Side calls (/btw, compaction, rewind summaries, the web_fetch prompt, prompt and agent hooks,
  subagents, the editor, the fast model) add their usage and cost once, priced by the model that
  served them, and stay in the totals after the next prompt and after a restart.
- /compact keeps the real token and cost totals; only the context estimate shrinks.
- /btw, compaction and rewind summaries send the main loop's tools, system and messages unchanged,
  so the summary request reads the cache the main loop wrote; no tool they ask for is run.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from unittest import mock

from src import prompt_cache
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig, ModelPricing

MILLION = 1_000_000
ANTHROPIC = 'https://api.anthropic.com'
OPENROUTER = 'https://openrouter.ai/api/v1'
_CLEAN = {k: '' for k in ('SIGIL_PROMPT_CACHE', 'SIGIL_PROMPT_CACHE_TTL', 'SIGIL_PROMPT_CACHE_PROVIDER',
                          'SIGIL_ANTHROPIC_NATIVE', 'SIGIL_ANTHROPIC_STREAM', 'SIGIL_EXCLUDE_PARAMS',
                          'SIGIL_FAST_MODEL', 'SIGIL_EFFORT_LEVEL', 'SIGIL_MODEL_PRICES', 'SIGIL_ARCHITECT',
                          'SIGIL_EDITOR_MODEL')}
TABLE = {'claude-sonnet-4-5': {'input': 3, 'output': 15},
         'claude-haiku-4-5': {'input': 1, 'output': 5}}
SONNET = ModelPricing(input_cost_per_million_tokens_usd=3.0, output_cost_per_million_tokens_usd=15.0,
                      cache_creation_input_cost_per_million_tokens_usd=3.75,
                      cache_read_input_cost_per_million_tokens_usd=0.3,
                      cache_creation_1h_input_cost_per_million_tokens_usd=6.0)


class _Response(io.BytesIO):
    headers = {'Content-Type': 'application/json'}

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.close()


class RoutedModel:
    """Fake urlopen for model calls (POST) and the /models list (GET).
    handler(url, body or None) -> (status, reply dict). Keeps (url, body) of each request."""

    def __init__(self, handler) -> None:  # noqa: ANN001
        self.handler = handler
        self.requests: list[tuple[str, dict | None]] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001, ANN204
        body = json.loads(req.data.decode('utf-8')) if req.data else None
        self.requests.append((req.full_url, body))
        status, reply = self.handler(req.full_url, body)
        data = json.dumps(reply).encode('utf-8')
        if status != 200:
            raise urllib.error.HTTPError(req.full_url, status, 'error', {}, io.BytesIO(data))
        return _Response(data)

    def posts(self) -> list[dict]:
        return [body for _url, body in self.requests if body is not None]


def _native(text: str = 'ok', *, tool: str | None = None, input_tokens: int = 0, output: int = 0,
            read: int = 0, write: int = 0) -> dict:
    content: list[dict] = []
    if text:
        content.append({'type': 'text', 'text': text})
    if tool:
        content.append({'type': 'tool_use', 'id': 'toolu_1', 'name': tool, 'input': {'path': 'a.txt'}})
    return {'content': content, 'stop_reason': 'tool_use' if tool else 'end_turn',
            'usage': {'input_tokens': input_tokens, 'output_tokens': output, 'cache_read_input_tokens': read,
                      'cache_creation_input_tokens': write}}


def _chat(text: str = 'ok', *, tool: str | None = None, usage: dict | None = None) -> dict:
    message: dict = {'role': 'assistant', 'content': text}
    if tool:
        message['tool_calls'] = [{'id': 'call_1', 'type': 'function',
                                  'function': {'name': tool, 'arguments': '{"path": "a.txt"}'}}]
    return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if tool else 'stop'}],
            'usage': usage or {'prompt_tokens': 10, 'completion_tokens': 1}}


def _is_summary(body: dict | None) -> bool:
    text = json.dumps(body or {})
    return 'Side question' in text or 'Do NOT call any tools' in text or 'Web page content from' in text


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
        from src import model_info

        model_info._CACHE.clear()
        self.addCleanup(model_info._CACHE.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cwd = Path(tmp.name).resolve()
        (self.cwd / 'a.txt').write_text('hello\n', encoding='utf-8')

    def _agent(self, model: str = 'claude-sonnet-4-5', base_url: str = ANTHROPIC,
               pricing: ModelPricing = SONNET, **config) -> LocalCodingAgent:  # noqa: ANN003
        return LocalCodingAgent(
            model_config=ModelConfig(model=model, base_url=base_url, api_key='test-key', pricing=pricing, **config),
            runtime_config=AgentRuntimeConfig(cwd=self.cwd, compact_preserve_messages=1))

    def _patch(self, fake: RoutedModel) -> None:
        patcher = mock.patch.object(urllib.request, 'urlopen', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)


def replace_ns(namespace, **changes):  # noqa: ANN001, ANN003, ANN201
    from types import SimpleNamespace

    return SimpleNamespace(**{**vars(namespace), **changes})


# ---------------------------------------------------------------- OQ4: input and output prices

class PricesFollowTheModelTests(_Base):
    MODELS = {'data': [
        {'id': 'anthropic/claude-sonnet-4.5', 'pricing': {
            'prompt': '0.000003', 'completion': '0.000015', 'input_cache_read': '0.0000003',
            'input_cache_write': '0.00000375', 'input_cache_write_1h': '0.000006'}},
        {'id': 'openai/gpt-4o', 'pricing': {'prompt': '0.0000025', 'completion': '0.00001',
                                            'input_cache_read': '0.00000125'}},
        {'id': 'openrouter/auto', 'pricing': {'prompt': '-1', 'completion': '-1'}},
    ]}

    def _config(self, *argv: str) -> ModelConfig:
        from src.main import _build_model_config, build_parser

        return _build_model_config(build_parser().parse_args(['agent', 'hi', *argv]))

    def test_server_price_list_gives_prices_and_a_switch_rederives_them(self) -> None:
        def handler(url, body):  # noqa: ANN001, ANN202
            if body is None:
                return 200, self.MODELS
            return 200, _chat('done', usage={'prompt_tokens': MILLION, 'completion_tokens': 1000,
                                             'prompt_tokens_details': {'cached_tokens': 600_000}})
        self._patch(RoutedModel(handler))
        with mock.patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}):
            config = self._config('--model', 'anthropic/claude-sonnet-4.5', '--base-url', OPENROUTER)
            pricing = config.pricing
            self.assertEqual((pricing.input_cost_per_million_tokens_usd, pricing.output_cost_per_million_tokens_usd),
                             (3.0, 15.0))
            self.assertAlmostEqual(pricing.cache_read_input_cost_per_million_tokens_usd, 0.3)
            self.assertAlmostEqual(pricing.cache_creation_1h_input_cost_per_million_tokens_usd, 6.0)
            agent = LocalCodingAgent(model_config=replace(config, api_key='test-key'),
                                     runtime_config=AgentRuntimeConfig(cwd=self.cwd))
            agent.set_model('openai/gpt-4o')
            pricing = agent.model_config.pricing
            self.assertAlmostEqual(pricing.input_cost_per_million_tokens_usd, 2.5)
            self.assertAlmostEqual(pricing.output_cost_per_million_tokens_usd, 10.0)
            self.assertAlmostEqual(pricing.cache_read_input_cost_per_million_tokens_usd, 1.25)
            result = agent.run('hi')
        self.assertAlmostEqual(result.total_cost_usd, 0.4 * 2.5 + 0.6 * 1.25 + 0.001 * 10.0, places=9)

    def test_a_negative_server_price_is_ignored(self) -> None:
        self._patch(RoutedModel(lambda url, body: (200, self.MODELS)))
        with mock.patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}):
            config = self._config('--model', 'openrouter/auto', '--base-url', OPENROUTER)
        self.assertEqual(config.pricing.input_cost_per_million_tokens_usd, 0.0)

    def test_nanogpt_detailed_list(self) -> None:
        fake = RoutedModel(lambda url, body: (200, {'data': [{'id': 'zai-org/glm-5', 'pricing': {
            'prompt': 0.6, 'completion': 2.2, 'cacheReadInputPer1kTokens': 0.00011,
            'currency': 'USD', 'unit': 'per_million_tokens'}}]}))
        self._patch(fake)
        with mock.patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}):
            config = self._config('--model', 'zai-org/glm-5', '--base-url', 'https://nano-gpt.com/api/v1')
        self.assertIn('detailed=true', fake.requests[0][0])
        self.assertAlmostEqual(config.pricing.input_cost_per_million_tokens_usd, 0.6)
        self.assertAlmostEqual(config.pricing.output_cost_per_million_tokens_usd, 2.2)
        self.assertAlmostEqual(config.pricing.cache_read_input_cost_per_million_tokens_usd, 0.11)

    def test_price_table_and_a_flag_price(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_MODEL_PRICES': json.dumps(TABLE)}):
            config = self._config('--model', 'claude-sonnet-4-5', '--base-url', ANTHROPIC)
            self.assertEqual((config.pricing.input_cost_per_million_tokens_usd,
                              config.pricing.output_cost_per_million_tokens_usd), (3.0, 15.0))
            agent = self._agent(pricing=config.pricing)
            agent.set_model('claude-haiku-4-5')
            self.assertEqual((agent.model_config.pricing.input_cost_per_million_tokens_usd,
                              agent.model_config.pricing.output_cost_per_million_tokens_usd), (1.0, 5.0))
            self.assertAlmostEqual(agent.model_config.pricing.cache_read_input_cost_per_million_tokens_usd, 0.1)
            flagged = self._config('--model', 'claude-sonnet-4-5', '--base-url', ANTHROPIC,
                                   '--input-cost-per-million', '7')
            agent = self._agent(pricing=flagged.pricing)
            agent.set_model('claude-haiku-4-5')
        self.assertEqual(agent.model_config.pricing.input_cost_per_million_tokens_usd, 7.0)  # flag kept
        self.assertEqual(agent.model_config.pricing.output_cost_per_million_tokens_usd, 5.0)

    def test_a_fallback_turn_uses_the_fallbacks_input_price(self) -> None:
        def handler(url, body):  # noqa: ANN001, ANN202
            if body['model'] == 'claude-sonnet-4-5':
                return 400, {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model not found'}}
            return 200, _native('from haiku', input_tokens=MILLION)
        self._patch(RoutedModel(handler))
        with mock.patch.dict(os.environ, {'SIGIL_MODEL_PRICES': json.dumps(TABLE)}):
            agent = self._agent(fallback_models=('claude-haiku-4-5',))
            result = agent.run('hi')
        self.assertEqual(result.final_output, 'from haiku')
        self.assertAlmostEqual(result.total_cost_usd, 1.0, places=9)  # haiku's input price, not sonnet's 3

    def test_agent_resume_with_another_model(self) -> None:
        from src.main import _build_resumed_agent, build_parser

        self._patch(RoutedModel(lambda url, body: (200, _native('ok', input_tokens=10))))
        with mock.patch.dict(os.environ, {'SIGIL_MODEL_PRICES': json.dumps(TABLE)}):
            first = self._agent().run('hi')
            args = build_parser().parse_args(['agent-resume', first.session_id, 'again', '--cwd', str(self.cwd),
                                              '--model', 'claude-haiku-4-5', '--base-url', ANTHROPIC])
            resumed, _stored = _build_resumed_agent(args)
        self.assertEqual((resumed.model_config.pricing.input_cost_per_million_tokens_usd,
                          resumed.model_config.pricing.output_cost_per_million_tokens_usd), (1.0, 5.0))


# ---------------------------------------------------------------- OQ2 + OQ3: side calls and /compact

class SideCallsAreCountedTests(_Base):
    def _run_then(self, replies: list[dict]) -> tuple[LocalCodingAgent, RoutedModel]:
        queue = list(replies)
        fake = RoutedModel(lambda url, body: (200, queue.pop(0) if len(queue) > 1 else queue[0]))
        self._patch(fake)
        agent = self._agent()
        agent.run('hi')
        return agent, fake

    def test_btw_counted_once_and_kept_after_the_next_prompt_and_a_restart(self) -> None:
        from src.agent_slash_commands import _handle_btw, _handle_cost
        from src.session_store import load_agent_session
        from src.status_line import status_payload

        agent, _fake = self._run_then([_native('first', input_tokens=1000, output=100),
                                       _native('side answer', read=1000, output=50),
                                       _native('again', input_tokens=10, output=1)])
        run_cost = (1000 * 3 + 100 * 15) / MILLION
        btw_cost = (1000 * 0.3 + 50 * 15) / MILLION
        self.assertIn('side answer', _handle_btw(agent, 'what file?', '/btw what file?').output)
        self.assertAlmostEqual(agent.cumulative_cost_usd, run_cost + btw_cost, places=12)
        self.assertEqual(agent.cumulative_usage.cache_read_input_tokens, 1000)
        self.assertIn('Side calls', _handle_cost(agent, '', '/cost').output)
        self.assertAlmostEqual(status_payload(agent)['cost']['total_cost_usd'], run_cost + btw_cost, places=12)
        stored = load_agent_session(agent.active_session_id, directory=agent.runtime_config.session_directory)
        agent.resume('again', stored)  # the next REPL prompt
        total = run_cost + btw_cost + (10 * 3 + 1 * 15) / MILLION
        self.assertAlmostEqual(agent.cumulative_cost_usd, total, places=12)
        self.assertEqual(agent.cumulative_usage.cache_read_input_tokens, 1000)
        restarted = self._agent()  # a new process: sigil -c
        stored = load_agent_session(agent.active_session_id, directory=agent.runtime_config.session_directory)
        with mock.patch.object(urllib.request, 'urlopen',
                               side_effect=RoutedModel(lambda u, b: (200, _native('x', input_tokens=0, output=0)))):
            restarted.resume('x', stored)
        self.assertAlmostEqual(restarted.cumulative_cost_usd, total, places=12)

    def test_print_mode_and_json_results_include_side_calls(self) -> None:
        from contextlib import redirect_stdout

        from src.agent_slash_commands import _handle_btw
        from src.main import _print_agent_result, _result_payload, _with_side_calls

        agent, _fake = self._run_then([_native('first', input_tokens=1000, output=100),
                                       _native('side answer', read=1000, output=50)])
        result = agent.last_run_result
        _handle_btw(agent, 'what file?', '/btw what file?')
        total = (1000 * 3 + 100 * 15 + 1000 * 0.3 + 50 * 15) / MILLION
        self.assertAlmostEqual(_result_payload(_with_side_calls(agent, result), 0)['total_cost_usd'], total, places=12)
        out = io.StringIO()
        with redirect_stdout(out):
            _print_agent_result(_with_side_calls(agent, result), show_transcript=False)
        self.assertIn('input_tokens_cumulative=2000', out.getvalue())  # 1000 + the /btw cache read

    def test_compact_keeps_the_real_totals_and_counts_the_summary(self) -> None:
        from src.session_store import load_agent_session

        agent, _fake = self._run_then([_native('', tool='read_file', input_tokens=1000, read=500, output=10),
                                       _native('done', input_tokens=1000, read=500, output=10),
                                       _native('<summary>S</summary>', read=2000, output=200)])
        directory = agent.runtime_config.session_directory
        stored = load_agent_session(agent.active_session_id, directory=directory)
        self.assertEqual(stored.usage['cache_read_input_tokens'], 1000)
        result = agent.compact(stored)
        self.assertIsNone(result.error)
        after = load_agent_session(agent.active_session_id, directory=directory)
        self.assertEqual((after.usage['input_tokens'], after.usage['cache_read_input_tokens']), (2000, 1000))
        runs = 2 * (1000 * 3 + 500 * 0.3 + 10 * 15) / MILLION
        summary = (2000 * 0.3 + 200 * 15) / MILLION
        self.assertAlmostEqual(after.total_cost_usd + after.side_cost_usd, runs + summary, places=12)
        restarted = self._agent()
        with mock.patch.object(urllib.request, 'urlopen',
                               side_effect=RoutedModel(lambda u, b: (200, _native('x')))):
            restarted.resume('x', after)
        self.assertEqual(restarted.cumulative_usage.cache_read_input_tokens, 3000)
        self.assertAlmostEqual(restarted.cumulative_cost_usd, runs + summary, places=12)

    def test_rewind_summary_counted(self) -> None:
        from src.rewind import list_prompt_points, summarize_session
        from src.session_store import load_agent_session

        agent, _fake = self._run_then([_native('one', input_tokens=100), _native('two', input_tokens=100),
                                       _native('<summary>S</summary>', read=1000, output=100)])
        directory = agent.runtime_config.session_directory
        agent.resume('second', load_agent_session(agent.active_session_id, directory=directory))
        before = agent.cumulative_cost_usd
        stored = load_agent_session(agent.active_session_id, directory=directory)
        summarize_session(agent, stored, list_prompt_points(stored.messages)[-1], 'summarize-to')
        self.assertAlmostEqual(agent.cumulative_cost_usd - before, (1000 * 0.3 + 100 * 15) / MILLION, places=12)

    def test_web_fetch_prompt_counted(self) -> None:
        def handler(url, body):  # noqa: ANN001, ANN202
            text = json.dumps(body)
            if 'Web page content from' in text:
                return 200, _native('the answer', input_tokens=2000, output=20)
            if '"tool_result"' in text:
                return 200, _native('done', input_tokens=10)
            return 200, {'content': [{'type': 'tool_use', 'id': 'toolu_w', 'name': 'web_fetch',
                                      'input': {'url': 'https://example.com', 'prompt': 'what?'}}],
                         'stop_reason': 'tool_use', 'usage': {'input_tokens': 10, 'output_tokens': 0}}
        self._patch(RoutedModel(handler))
        page = {'text': 'page text', 'content_type': 'text/plain', 'final_url': 'https://example.com'}
        with mock.patch('src.agent_tools._web_fetch_cached', return_value=page):
            agent = self._agent()
            result = agent.run('fetch it')
        side = (2000 * 3 + 20 * 15) / MILLION
        self.assertAlmostEqual(agent.cumulative_cost_usd - result.total_cost_usd, side, places=12)

    def test_prompt_and_agent_hooks_counted(self) -> None:
        self._patch(RoutedModel(lambda url, body: (200, _native('{"ok": true}', input_tokens=1000, output=10))))
        agent = self._agent()
        agent._complete_for_prompt_hook('check', 'payload', 5.0)
        agent._run_agent_for_hook('check', 'payload', 5.0)
        self.assertAlmostEqual(agent.cumulative_cost_usd, 2 * (1000 * 3 + 10 * 15) / MILLION, places=12)

    def test_fast_model_priced_with_its_own_prices(self) -> None:
        from src.compact import compact_conversation

        agent, _fake = self._run_then([_native('', tool='read_file', input_tokens=10),
                                       _native('done', input_tokens=10),
                                       _native('<summary>S</summary>', input_tokens=MILLION)])
        before = agent.cumulative_cost_usd
        with mock.patch.dict(os.environ, {'SIGIL_FAST_MODEL': 'claude-haiku-4-5',
                                          'SIGIL_MODEL_PRICES': json.dumps(TABLE)}):
            self.assertIsNone(compact_conversation(agent, trigger='manual').error)
        self.assertAlmostEqual(agent.cumulative_cost_usd - before, 1.0, places=9)  # haiku input price

    def test_compaction_fallback_counted(self) -> None:
        from src.compact import compact_conversation

        agent, _fake = self._run_then([_native('', tool='read_file', input_tokens=10),
                                       _native('done', input_tokens=10),
                                       _native('', read=1000),                       # no text: fall back
                                       _native('<summary>S</summary>', input_tokens=500)])
        before = agent.cumulative_cost_usd
        self.assertIsNone(compact_conversation(agent, trigger='manual').error)
        self.assertAlmostEqual(agent.cumulative_cost_usd - before, (1000 * 0.3 + 500 * 3) / MILLION, places=12)

    def test_reactive_compaction_counted_once(self) -> None:
        from types import SimpleNamespace

        agent, _fake = self._run_then([_native('', tool='read_file', input_tokens=10),
                                       _native('done', input_tokens=10),
                                       _native('<summary>S</summary>', read=3000, output=300)])
        over = SimpleNamespace(exceeds_soft_limit=True, exceeds_hard_limit=False, projected_input_tokens=9000,
                               soft_input_limit_tokens=5000, hard_input_limit_tokens=8000,
                               soft_overflow_tokens=4000, overflow_tokens=1000)
        fits = replace_ns(over, exceeds_soft_limit=False)
        side_before = agent.side_totals()[0].total_tokens
        with mock.patch('src.agent_runtime.calculate_token_budget', side_effect=[over, over, fits]), \
                mock.patch.object(agent, '_reduce_context_pressure'), \
                mock.patch.object(agent, '_can_auto_compact_with_summary', return_value=True):
            preflight = agent._preflight_prompt_length(agent.last_session, [], turn_index=0)
        counted = preflight.usage_increment.total_tokens + agent.side_totals()[0].total_tokens - side_before
        self.assertEqual(counted, 3300)  # the summary's 3000 cache reads + 300 output, once
        self.assertEqual(preflight.model_calls_increment, 1)

    def test_subagent_and_editor_counted(self) -> None:
        from src.architect import run_editor

        self._patch(RoutedModel(lambda url, body: (200, _native('child done', input_tokens=1000))))
        agent = self._agent()
        outcome = agent._execute_delegate_agent({'prompt': 'help'})
        self.assertTrue(outcome.ok, outcome.content)
        run_editor(agent, 'change nothing')
        self.assertAlmostEqual(agent.cumulative_cost_usd, 2 * 1000 * 3 / MILLION, places=12)


# ---------------------------------------------------------------- OQ1: summaries reuse the cache

def _strip(value):  # noqa: ANN001, ANN202
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _marked(message: dict) -> bool:
    content = message.get('content')
    return isinstance(content, list) and any(isinstance(p, dict) and 'cache_control' in p for p in content)


class SummaryReusesTheCacheTests(_Base):
    def _capture(self, base_url: str, model: str) -> tuple[dict, list[dict]]:
        """A run with one tool call, then /btw, /compact and a rewind summary. Returns the main
        loop's last request and the three summary requests."""
        from src.agent_slash_commands import _handle_btw
        from src.rewind import list_prompt_points, summarize_session
        from src.session_store import load_agent_session

        native = base_url == ANTHROPIC

        def handler(url, body):  # noqa: ANN001, ANN202
            if _is_summary(body):
                return 200, (_native('<summary>S</summary>') if native else _chat('<summary>S</summary>'))
            done = '"tool_result"' in json.dumps(body) or any(m.get('role') == 'tool' for m in body['messages'])
            if native:
                return 200, _native('done') if done else _native('', tool='read_file')
            return 200, _chat('done') if done else _chat('', tool='read_file')
        fake = RoutedModel(handler)
        self._patch(fake)
        agent = self._agent(model=model, base_url=base_url)
        agent.run('read a.txt')
        main_last = fake.posts()[-1]
        _handle_btw(agent, 'which file?', '/btw which file?')
        directory = agent.runtime_config.session_directory
        stored = load_agent_session(agent.active_session_id, directory=directory)  # each starts from here
        self.assertIsNone(agent.compact(stored).error)
        summarize_session(agent, stored, list_prompt_points(stored.messages)[-1], 'summarize-from')
        summaries = [b for b in fake.posts() if _is_summary(b)]
        return main_last, summaries

    def _check_prefix(self, main_last: dict, summary: dict, system_key: str | None) -> None:
        self.assertEqual(json.dumps(summary.get('tools')), json.dumps(main_last.get('tools')))
        self.assertNotIn('tool_choice', set(summary) - set(main_last))
        self.assertEqual(summary.get('tool_choice'), main_last.get('tool_choice'))
        if system_key:
            self.assertEqual(json.dumps(summary[system_key]), json.dumps(main_last[system_key]))
        k = len(main_last['messages']) - 1  # the main loop's last breakpoint
        self.assertTrue(_marked(main_last['messages'][k]))
        self.assertTrue(_marked(summary['messages'][k]), 'the summary request has a breakpoint there too')
        self.assertEqual(json.dumps(_strip(summary['messages'][:k + 1])),
                         json.dumps(_strip(main_last['messages'][:k + 1])))

    def test_native_anthropic(self) -> None:
        main_last, summaries = self._capture(ANTHROPIC, 'claude-sonnet-4-5')
        self.assertEqual(len(summaries), 3)
        for summary in summaries:
            self._check_prefix(main_last, summary, 'system')

    def test_openrouter_claude_openai_format(self) -> None:
        main_last, summaries = self._capture(OPENROUTER, 'anthropic/claude-sonnet-4.5')
        self.assertEqual(len(summaries), 3)
        for summary in summaries:
            self._check_prefix(main_last, summary, None)

    def test_a_tool_call_in_a_summary_is_never_run(self) -> None:
        from src.session_store import load_agent_session

        replies = [_native('', tool='read_file'), _native('done'),
                   _native('', tool='read_file'), _native('<summary>S</summary>')]
        fake = RoutedModel(lambda url, body: (200, replies.pop(0) if len(replies) > 1 else replies[0]))
        self._patch(fake)
        agent = self._agent()
        agent.run('read a.txt')
        stored = load_agent_session(agent.active_session_id, directory=agent.runtime_config.session_directory)
        with mock.patch('src.agent_tools._read_file', side_effect=AssertionError('a summary ran a tool')):
            result = agent.compact(stored)
        self.assertIsNone(result.error)
        self.assertIn('S', result.summary_text)
        summaries = [b for b in fake.posts() if _is_summary(b)]
        self.assertEqual(len(summaries), 2)  # asked again once, with the instruction repeated
        self.assertEqual(json.dumps(summaries[0].get('tools')), json.dumps(summaries[1].get('tools')))


if __name__ == '__main__':
    unittest.main()
