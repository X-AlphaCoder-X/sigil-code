"""Prompt caching: provider plans, breakpoints, the one-time retry without cache fields, usage and
prices, /cache, background agents, microcompact batching and rewind. Offline: local fake servers."""
from __future__ import annotations

import http.server
import io
import json
import os
import re
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from src import prompt_cache
from src.agent_types import ModelConfig, ModelPricing, UsageStats
from src.anthropic_client import AnthropicClient, FallbackClient, add_breakpoints, anthropic_usage, make_client
from src.openai_compat import OpenAICompatClient, OpenAICompatError, _parse_usage

_CLEAN = {k: '' for k in ('SIGIL_PROMPT_CACHE', 'SIGIL_PROMPT_CACHE_TTL', 'SIGIL_ANTHROPIC_NATIVE',
                          'SIGIL_ANTHROPIC_STREAM', 'SIGIL_MICROCOMPACT_BATCH', 'SIGIL_EXCLUDE_PARAMS')}


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, _CLEAN)
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        self.settings = settings.start()
        self.addCleanup(settings.stop)
        prompt_cache.set_session(clear=True)
        self.addCleanup(prompt_cache.set_session, clear=True)


class FakeServer:
    """A local HTTP server. reply(body: dict, headers: dict) -> (status, content_type, bytes)."""

    def __init__(self, reply) -> None:  # noqa: ANN001
        self.requests: list[tuple[dict, dict]] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
                headers = {k.lower(): v for k, v in self.headers.items()}
                outer.requests.append((body, headers))
                status, ctype, data = reply(body, headers)
                self.send_response(status)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:  # noqa: ANN002
                return None

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _chat_reply(usage: dict | None = None) -> bytes:
    return json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                       'usage': usage or {'prompt_tokens': 100, 'completion_tokens': 5,
                                          'prompt_tokens_details': {'cached_tokens': 60}}}).encode()


def _chat_stream() -> bytes:
    chunks = [{'choices': [{'delta': {'content': 'ok'}}]},
              {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
              {'choices': [], 'usage': {'prompt_tokens': 50, 'completion_tokens': 2,
                                        'prompt_tokens_details': {'cached_tokens': 40}}}]
    return (''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + 'data: [DONE]\n\n').encode()


def _rejects_cache_fields(status: int):  # noqa: ANN202
    def reply(body: dict, _headers: dict):  # noqa: ANN202
        if 'prompt_cache_key' in body or 'cache_control' in json.dumps(body):
            return status, 'application/json', json.dumps(
                {'error': {'message': 'Unrecognized request argument supplied: prompt_cache_key'}}).encode()
        if body.get('stream'):
            return 200, 'text/event-stream', _chat_stream()
        return 200, 'application/json', _chat_reply()
    return reply


MESSAGES = [{'role': 'system', 'content': 'You are a coding agent.'}, {'role': 'user', 'content': 'hi'}]


class ProviderPlanTests(_Base):
    def test_hosts_match_exactly_or_by_subdomain(self) -> None:
        cases = {
            'https://api.anthropic.com/v1': 'anthropic', 'https://evilanthropic.com/v1': 'openai-compatible',
            'https://anthropic.com.evil.io/v1': 'openai-compatible', 'https://openrouter.ai/api/v1': 'openrouter',
            'https://api.openai.com/v1': 'openai', 'https://x.openai.azure.com/openai': 'azure',
            'https://api.deepseek.com': 'deepseek', 'https://api.x.ai/v1': 'xai',
            'https://dashscope-intl.aliyuncs.com/compatible-mode/v1': 'dashscope',
            'https://api.moonshot.ai/v1': 'moonshot', 'https://api.z.ai/api/paas/v4': 'zai',
            'https://api.mistral.ai/v1': 'mistral', 'https://api.groq.com/openai/v1': 'groq',
            'https://api.fireworks.ai/inference/v1': 'fireworks', 'https://api.together.xyz/v1': 'together',
            'https://nano-gpt.com/api/v1': 'nanogpt', 'https://llm.chutes.ai/v1': 'chutes',
            'https://generativelanguage.googleapis.com/v1beta/openai': 'gemini',
            'https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1': 'bedrock',
            'http://127.0.0.1:11434/v1': 'ollama', 'http://localhost:1234/v1': 'lmstudio',
            'http://127.0.0.1:8080/v1': 'llamacpp', 'http://127.0.0.1:8000/v1': 'vllm',
            'http://127.0.0.1:9999/v1': 'openai-compatible',
        }
        for url, expected in cases.items():
            self.assertEqual(prompt_cache.provider_of(url), expected, url)

    def test_plan_per_provider(self) -> None:
        plan = prompt_cache.plan_for
        self.assertTrue(plan('https://api.anthropic.com', 'claude-x', native=True).markers)
        self.assertFalse(plan('https://api.anthropic.com/v1', 'claude-x').markers)  # compat endpoint: no cache
        orc = plan('https://openrouter.ai/api/v1', 'anthropic/claude-sonnet-5')
        self.assertTrue(orc.markers and orc.marker_ttl)
        self.assertEqual(orc.key_headers, ('x-session-id',))
        org = plan('https://openrouter.ai/api/v1', 'google/gemini-3-pro')
        self.assertTrue(org.markers and not org.marker_ttl)
        self.assertFalse(plan('https://openrouter.ai/api/v1', 'deepseek/deepseek-v4').markers)
        self.assertTrue(plan('https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen3-max').markers)
        self.assertEqual(plan('https://api.openai.com/v1', 'gpt-5.5').key_field, 'prompt_cache_key')
        self.assertIsNone(plan('https://x.openai.azure.com/openai', 'gpt-4.1').key_field)
        self.assertEqual(plan('https://api.mistral.ai/v1', 'mistral-large').key_field, 'prompt_cache_key')
        self.assertEqual(plan('https://api.x.ai/v1', 'grok-5').key_headers, ('x-grok-conv-id',))
        self.assertEqual(plan('https://api.fireworks.ai/inference/v1', 'm').key_headers, ('x-session-affinity',))
        for url in ('https://api.deepseek.com', 'https://api.groq.com/openai/v1', 'http://127.0.0.1:8000/v1',
                    'https://generativelanguage.googleapis.com/v1beta/openai'):
            p = plan(url, 'm')
            self.assertFalse(p.markers or p.key_field or p.key_headers or p.body, url)
        for url in ('https://llm.chutes.ai/v1', 'http://localhost:1234/v1',
                    'https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1', 'http://127.0.0.1:9999/v1'):
            self.assertEqual(plan(url, 'm').key_field, 'prompt_cache_key', url)  # OpenAI approach

    def test_ttl_mapping(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_TTL': '1h'}):
            self.assertEqual(prompt_cache.plan_for('https://api.anthropic.com', 'c', native=True).marker(),
                             {'type': 'ephemeral', 'ttl': '1h'})
            self.assertEqual(prompt_cache.plan_for('https://dashscope.aliyuncs.com/v1', 'qwen').marker(),
                             {'type': 'ephemeral'})  # no TTL choice there
            self.assertEqual(prompt_cache.plan_for('https://api.openai.com/v1', 'gpt-5.5').body,
                             {'prompt_cache_retention': '24h'})
            self.assertEqual(prompt_cache.plan_for('https://api.openai.com/v1', 'gpt-5.6').body, {})
            self.assertEqual(prompt_cache.plan_for('https://api.moonshot.ai/v1', 'kimi-k3').body,
                             {'prompt_cache_options': {'mode': 'implicit', 'ttl': '1h'}})
            self.assertEqual(prompt_cache.plan_for('http://127.0.0.1:11434/v1', 'llama').body, {'keep_alive': '1h'})
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_TTL': '5m'}):
            self.assertEqual(prompt_cache.plan_for('https://api.anthropic.com', 'c', native=True).marker(),
                             {'type': 'ephemeral'})
            self.assertEqual(prompt_cache.plan_for('https://api.openai.com/v1', 'gpt-5.5').body, {})
            self.assertEqual(prompt_cache.plan_for('http://127.0.0.1:11434/v1', 'llama').body, {})
        # the default (user decision 2026-11-04) is 1 hour, for the main agent and every subagent
        self.assertEqual(prompt_cache.current_settings().ttl, '1h')
        self.assertEqual(prompt_cache.plan_for('https://api.anthropic.com', 'c', native=True).marker(),
                         {'type': 'ephemeral', 'ttl': '1h'})

    def test_off_sends_nothing_and_on_marks_anywhere(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'off'}):
            for url in ('https://openrouter.ai/api/v1', 'https://api.openai.com/v1', 'https://api.x.ai/v1'):
                plan = prompt_cache.plan_for(url, 'anthropic/claude-x')
                self.assertFalse(plan.active or plan.markers)
                payload, headers = {'model': 'm', 'messages': MESSAGES}, {}
                self.assertFalse(prompt_cache.apply_hints(plan, payload, headers))
                self.assertEqual((payload, headers), ({'model': 'm', 'messages': MESSAGES}, {}))
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'on'}):
            self.assertTrue(prompt_cache.plan_for('http://127.0.0.1:8000/v1', 'm').markers)

    def test_settings_precedence_and_invalid_values(self) -> None:
        self.settings.return_value = {'mode': 'off', 'ttl': '1h'}
        self.assertEqual(prompt_cache.current_settings(), prompt_cache.CacheSettings('off', '1h', 'settings', 'settings'))
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'on', 'SIGIL_PROMPT_CACHE_TTL': '5m'}):
            got = prompt_cache.current_settings()
            self.assertEqual((got.mode, got.ttl, got.mode_source), ('on', '5m', 'SIGIL_PROMPT_CACHE'))
            prompt_cache.set_session(mode='auto', ttl='24h')
            got = prompt_cache.current_settings()
            self.assertEqual((got.mode, got.ttl, got.mode_source), ('auto', '24h', '/cache'))
        prompt_cache.set_session(clear=True)
        self.settings.return_value = {}
        err = io.StringIO()
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'sometimes', 'SIGIL_PROMPT_CACHE_TTL': '7m'}), \
                mock.patch('sys.stderr', err):
            got = prompt_cache.current_settings()
        self.assertEqual((got.mode, got.ttl), ('auto', '1h'))
        self.assertIn('SIGIL_PROMPT_CACHE=sometimes is not a prompt cache mode', err.getvalue())
        self.assertIn('SIGIL_PROMPT_CACHE_TTL=7m is not a prompt cache TTL', err.getvalue())

    def test_cache_key_is_a_hash(self) -> None:
        secret_path = str(Path.home())
        messages = [{'role': 'system', 'content': f'cwd {secret_path} user {os.environ.get("USERNAME", "me")}'}]
        key = prompt_cache.cache_key('gpt-5.5', messages)
        self.assertRegex(key, r'^sigil-[0-9a-f]{32}$')
        self.assertNotIn(Path.home().name.lower(), key)
        self.assertEqual(key, prompt_cache.cache_key('gpt-5.5', messages))
        self.assertNotEqual(key, prompt_cache.cache_key('gpt-5.4', messages))

    def test_rejected_endpoints_are_bounded_and_thread_safe(self) -> None:
        def worker(n: int) -> None:
            for i in range(50):
                prompt_cache.remember_rejected(f'http://127.0.0.1:{n}/v1', f'm{i}', 'HTTP 400')
                prompt_cache.set_session(ttl='1h')
        threads = [threading.Thread(target=worker, args=(9000 + n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertLessEqual(len(prompt_cache._REJECTED), 64)  # documented bound, oldest endpoint dropped first


class BreakpointTests(_Base):
    def test_openai_format_breakpoints(self) -> None:
        marker = {'type': 'ephemeral'}
        messages = [{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA'}}]},
                    {'role': 'assistant', 'content': 'a1'},
                    {'role': 'tool', 'tool_call_id': 'c', 'content': '   '},
                    {'role': 'user', 'content': 'next'}]
        marked = prompt_cache.with_breakpoints(messages, marker)
        spots = [i for i, m in enumerate(marked) if 'cache_control' in json.dumps(m)]
        self.assertEqual(spots, [0, 1, 4])  # system, end of the previous request, end of this one
        self.assertEqual(marked[1]['content'][0]['cache_control'], marker)  # text part, not the image
        self.assertNotIn('cache_control', json.dumps(marked[3]))  # whitespace is never marked
        self.assertEqual(messages[1]['content'][0], {'type': 'text', 'text': 'look'})  # input not changed
        self.assertLessEqual(len(spots), prompt_cache.MAX_BREAKPOINTS)

    def test_native_breakpoints_skip_thinking_and_empty(self) -> None:
        payload = {'tools': [{'name': 'a'}, {'name': 'b'}],
                   'system': [{'type': 'text', 'text': 'sys'}],
                   'messages': [
                       {'role': 'user', 'content': [{'type': 'text', 'text': 'q'},
                                                    {'type': 'image', 'source': {'type': 'base64', 'data': 'AA'}}]},
                       {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 't', 'signature': 's'},
                                                         {'type': 'tool_use', 'id': 'x', 'name': 'a', 'input': {}}]},
                       {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'x', 'content': 'r'},
                                                    {'type': 'text', 'text': ' '}]}]}
        count = add_breakpoints(payload, {'type': 'ephemeral'})
        self.assertEqual(count, 4)
        self.assertIn('cache_control', payload['tools'][-1])
        self.assertIn('cache_control', payload['system'][-1])
        self.assertIn('cache_control', payload['messages'][0]['content'][1])  # the image block
        self.assertIn('cache_control', payload['messages'][2]['content'][0])  # tool_result, not the ' ' text
        self.assertNotIn('cache_control', payload['messages'][2]['content'][1])
        self.assertNotIn('cache_control', json.dumps(payload['messages'][1]))  # never on thinking
        self.assertEqual(json.dumps(payload).count('cache_control'), 4)

    def test_rewind_keeps_the_cached_prefix(self) -> None:
        """After a rewind the next request's previous-request breakpoint sits where an earlier request
        wrote its cache, with identical content before it (a hit; docs: writes occur only at breakpoints)."""
        client = AnthropicClient(ModelConfig(model='claude-x', base_url='https://api.anthropic.com'))
        history = [{'role': 'system', 'content': 'sys'}]
        requests = []
        for i in range(4):
            history.append({'role': 'user', 'content': f'prompt {i}'})
            requests.append(client.build_payload(list(history), []))
            history.append({'role': 'assistant', 'content': f'reply {i}'})
        kept = history[:history.index({'role': 'user', 'content': 'prompt 2'})]  # rewind to before prompt 2
        after = client.build_payload(kept + [{'role': 'user', 'content': 'different prompt'}], [])

        def upto_marker(messages: list, which: int) -> str:
            plain = json.loads(json.dumps(messages))
            marked = [i for i, m in enumerate(plain) if 'cache_control' in json.dumps(m)]
            cut = marked[which]
            text = json.dumps(plain[:cut + 1])
            return re.sub(r', "cache_control": \{[^}]*\}', '', text)

        earlier = requests[1]  # the request that produced "reply 1", the last kept reply
        self.assertEqual(upto_marker(after['messages'], 0), upto_marker(earlier['messages'], -1))
        self.assertEqual(after['system'], earlier['system'])


class RetryWithoutCacheFieldsTests(_Base):
    def _client(self, server: FakeServer, model: str = 'm') -> OpenAICompatClient:
        return OpenAICompatClient(ModelConfig(model=model, base_url=server.url + '/v1', timeout_seconds=10))

    def test_400_and_422_retry_once_and_remember(self) -> None:
        for status in (400, 422):
            server = FakeServer(_rejects_cache_fields(status))
            self.addCleanup(server.close)
            client = self._client(server)
            from src.debug_log import get_logger

            with self.assertLogs(get_logger('prompt_cache').name, level='INFO') as logs:
                turn = client.complete(MESSAGES, [])
            self.assertEqual(turn.content, 'ok')
            self.assertEqual(len(server.requests), 2)
            self.assertIn('prompt_cache_key', server.requests[0][0])
            self.assertNotIn('prompt_cache_key', server.requests[1][0])
            self.assertTrue(any('cache fields rejected' in line for line in logs.output))
            client.complete(MESSAGES, [])  # remembered: sent without cache fields straight away
            self.assertEqual(len(server.requests), 3)
            self.assertNotIn('prompt_cache_key', server.requests[2][0])

    def test_stream_retries_before_the_first_event(self) -> None:
        server = FakeServer(_rejects_cache_fields(400))
        self.addCleanup(server.close)
        events = list(self._client(server).stream(MESSAGES, []))
        self.assertEqual(''.join(e.delta for e in events if e.type == 'content_delta'), 'ok')
        self.assertEqual(len(server.requests), 2)
        usage = [e.usage for e in events if e.type == 'usage'][0]
        self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens), (10, 40))

    def test_unrelated_400_without_cache_fields_is_not_retried(self) -> None:
        server = FakeServer(lambda body, h: (400, 'application/json', b'{"error": "bad"}'))
        self.addCleanup(server.close)
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'off'}):
            with self.assertRaises(OpenAICompatError):
                self._client(server).complete(MESSAGES, [])
        self.assertEqual(len(server.requests), 1)

    def test_a_400_that_is_not_about_cache_fields_is_not_retried(self) -> None:
        # The error does not name a cache field we sent, so it is shown at once (no second request).
        server = FakeServer(lambda body, h: (400, 'application/json', b'{"error": "context too long"}'))
        self.addCleanup(server.close)
        with self.assertRaises(OpenAICompatError):
            self._client(server).complete(MESSAGES, [])
        self.assertEqual(len(server.requests), 1)
        self.assertIsNone(prompt_cache.rejected_reason(server.url + '/v1', 'm'))  # not remembered

    def test_headers_and_key_reach_the_server(self) -> None:
        server = FakeServer(lambda body, h: (200, 'application/json', _chat_reply()))
        self.addCleanup(server.close)
        client = self._client(server)
        with mock.patch('src.prompt_cache.provider_of', return_value='xai'):
            client.complete(MESSAGES, [])
        body, headers = server.requests[-1]
        self.assertRegex(headers.get('x-grok-conv-id', ''), r'^sigil-[0-9a-f]{32}$')
        with mock.patch('src.prompt_cache.provider_of', return_value='openai'):
            client.complete(MESSAGES, [])
        self.assertRegex(server.requests[-1][0].get('prompt_cache_key', ''), r'^sigil-[0-9a-f]{32}$')

    def test_fallback_client_plans_per_endpoint(self) -> None:
        bad = FakeServer(lambda body, h: (500, 'application/json', b'{}'))
        good = FakeServer(lambda body, h: (200, 'application/json', _chat_reply()))
        self.addCleanup(bad.close)
        self.addCleanup(good.close)
        config = ModelConfig(model='m', base_url=bad.url + '/v1', timeout_seconds=10,
                             fallback_models=(f'm2@{good.url}/v1',))
        client = make_client(config)
        self.assertIsInstance(client, FallbackClient)
        with mock.patch.dict(os.environ, {'SIGIL_RETRY_MAX': '0'}):
            self.assertEqual(client.complete(MESSAGES, []).content, 'ok')
        self.assertIn('prompt_cache_key', good.requests[-1][0])


def _sse(events: list[dict]) -> bytes:
    return ''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n' for e in events).encode()


class NativeAnthropicTests(_Base):
    def _client(self, server: FakeServer) -> AnthropicClient:
        return AnthropicClient(ModelConfig(model='claude-x', base_url=server.url, timeout_seconds=10))

    def test_sse_stream_with_tools_thinking_and_usage(self) -> None:
        events = [
            {'type': 'message_start', 'message': {'usage': {
                'input_tokens': 12, 'cache_read_input_tokens': 900, 'cache_creation_input_tokens': 300,
                'cache_creation': {'ephemeral_5m_input_tokens': 100, 'ephemeral_1h_input_tokens': 200},
                'output_tokens': 1}}},
            {'type': 'ping'},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'hmm'}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'signature_delta', 'signature': 'sig'}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'Let me '}},
            {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'look.'}},
            {'type': 'content_block_stop', 'index': 1},
            {'type': 'content_block_start', 'index': 2, 'content_block': {
                'type': 'tool_use', 'id': 'tu1', 'name': 'read_file', 'input': {}}},
            {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '{"pa'}},
            {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': 'th": "a"}'}},
            {'type': 'content_block_stop', 'index': 2},
            {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': {'output_tokens': 40}},
            {'type': 'message_stop'}]
        server = FakeServer(lambda body, h: (200, 'text/event-stream', _sse(events)))
        self.addCleanup(server.close)
        client = self._client(server)
        out = list(client.stream(MESSAGES, [{'type': 'function', 'function': {'name': 'read_file'}}]))
        self.assertTrue(server.requests[0][0]['stream'])
        self.assertEqual(''.join(e.delta for e in out if e.type == 'content_delta'), 'Let me look.')
        calls = [e for e in out if e.type == 'tool_call_delta']
        self.assertEqual((calls[0].tool_call_id, calls[0].tool_name), ('tu1', 'read_file'))
        self.assertEqual(json.loads(''.join(c.arguments_delta for c in calls)), {'path': 'a'})
        usage = [e.usage for e in out if e.type == 'usage']
        self.assertEqual(len(usage), 1)
        self.assertEqual((usage[0].input_tokens, usage[0].output_tokens, usage[0].cache_read_input_tokens,
                          usage[0].cache_creation_input_tokens, usage[0].cache_creation_1h_input_tokens),
                         (12, 40, 900, 300, 200))
        self.assertEqual(out[-1].finish_reason, 'tool_calls')
        original = client.pop_original_content()
        self.assertEqual(original[0], {'type': 'thinking', 'thinking': 'hmm', 'signature': 'sig'})
        self.assertEqual(original[2]['input'], {'path': 'a'})

    def test_stream_error_event_raises(self) -> None:
        events = [{'type': 'message_start', 'message': {'usage': {}}},
                  {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}]
        server = FakeServer(lambda body, h: (200, 'text/event-stream', _sse(events)))
        self.addCleanup(server.close)
        with self.assertRaisesRegex(OpenAICompatError, 'overloaded_error'):
            list(self._client(server).stream(MESSAGES, []))

    def test_stream_cut_before_message_stop_raises(self) -> None:
        events = [{'type': 'message_start', 'message': {'usage': {'input_tokens': 3}}},
                  {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                  {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'half'}}]
        server = FakeServer(lambda body, h: (200, 'text/event-stream', _sse(events)))
        self.addCleanup(server.close)
        with self.assertRaisesRegex(OpenAICompatError, 'ended before message_stop'):
            list(self._client(server).stream(MESSAGES, []))

    def test_json_answer_to_a_stream_request_is_used_whole(self) -> None:
        reply = {'content': [{'type': 'thinking', 'thinking': 't', 'signature': 's'},
                             {'type': 'text', 'text': 'whole'}],
                 'stop_reason': 'end_turn', 'usage': {'input_tokens': 2, 'cache_read_input_tokens': 8}}
        server = FakeServer(lambda body, h: (200, 'application/json', json.dumps(reply).encode()))
        self.addCleanup(server.close)
        client = self._client(server)
        out = list(client.stream(MESSAGES, []))
        self.assertEqual(''.join(e.delta for e in out if e.type == 'content_delta'), 'whole')
        self.assertEqual([e.usage.cache_read_input_tokens for e in out if e.type == 'usage'], [8])
        self.assertEqual(client.pop_original_content()[0]['signature'], 's')

    def test_native_retry_without_cache_control(self) -> None:
        def reply(body: dict, _h: dict):  # noqa: ANN202
            if 'cache_control' in json.dumps(body):
                return 400, 'application/json', b'{"error": {"message": "cache_control: Extra inputs"}}'
            return 200, 'application/json', json.dumps({'content': [{'type': 'text', 'text': 'fine'}],
                                                        'stop_reason': 'end_turn', 'usage': {}}).encode()
        server = FakeServer(reply)
        self.addCleanup(server.close)
        with mock.patch.dict(os.environ, {'SIGIL_ANTHROPIC_STREAM': '0'}):
            self.assertEqual(self._client(server).complete(MESSAGES, []).content, 'fine')
        self.assertEqual(len(server.requests), 2)

    def test_usage_breakdown(self) -> None:
        usage = anthropic_usage({'input_tokens': 5, 'output_tokens': 7, 'cache_read_input_tokens': 50,
                                 'cache_creation': {'ephemeral_5m_input_tokens': 10, 'ephemeral_1h_input_tokens': 30}})
        self.assertEqual((usage.cache_creation_input_tokens, usage.cache_creation_1h_input_tokens), (40, 30))
        self.assertEqual(anthropic_usage({'input_tokens': -3, 'output_tokens': 'x'}), UsageStats())


class UsageAndPriceTests(_Base):
    def test_parse_usage_per_provider(self) -> None:
        cases = [
            ({'prompt_tokens': 100, 'completion_tokens': 5, 'prompt_tokens_details': {'cached_tokens': 60}},
             (40, 60, 0)),  # OpenAI: cached inside prompt_tokens
            ({'prompt_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 50, 'cache_write_tokens': 30}},
             (20, 50, 30)),  # OpenAI 5.6+ / OpenRouter writes
            ({'prompt_tokens': 100, 'prompt_cache_hit_tokens': 70, 'prompt_cache_miss_tokens': 30}, (30, 70, 0)),
            ({'prompt_tokens': 100, 'cached_tokens': 25}, (75, 25, 0)),  # Together top level
            ({'prompt_tokens': 100, 'prompt_tokens_details': {'cache_creation_input_tokens': 90}}, (10, 0, 90)),
            ({'input_tokens': 10, 'cache_read_input_tokens': 80, 'cache_creation_input_tokens': 20}, (10, 80, 20)),
            ({'prompt_tokens': 100, 'cache_read_input_tokens': 80}, (20, 80, 0)),  # proxy: inclusive total
            ({'prompt_tokens': 10, 'prompt_tokens_details': {'cached_tokens': 500}}, (0, 10, 0)),  # clamped
        ]
        for raw, (uncached, read, write) in cases:
            usage = _parse_usage(raw)
            self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens),
                             (uncached, read, write), raw)
            self.assertGreaterEqual(usage.input_tokens, 0)

    def test_cost_with_ttl_split_and_hit_rate(self) -> None:
        pricing = ModelPricing(input_cost_per_million_tokens_usd=10, output_cost_per_million_tokens_usd=0,
                               cache_creation_input_cost_per_million_tokens_usd=12.5,
                               cache_read_input_cost_per_million_tokens_usd=1,
                               cache_creation_1h_input_cost_per_million_tokens_usd=20)
        usage = UsageStats(input_tokens=1_000_000, cache_creation_input_tokens=2_000_000,
                           cache_creation_1h_input_tokens=1_000_000, cache_read_input_tokens=1_000_000)
        self.assertAlmostEqual(pricing.estimate_cost_usd(usage), 10 + 12.5 + 20 + 1)
        self.assertAlmostEqual(usage.cache_hit_rate, 0.25)
        self.assertIsNone(UsageStats().cache_hit_rate)
        unpriced = ModelPricing(input_cost_per_million_tokens_usd=10)
        self.assertAlmostEqual(unpriced.estimate_cost_usd(usage), 40)  # unset cache prices: the input price

    def test_derived_and_flag_prices(self) -> None:
        from src.main import _cache_pricing

        base = ModelPricing(input_cost_per_million_tokens_usd=10)
        args = Namespace(model='claude-opus-5-5', cache_read_cost_per_million=None,
                         cache_write_cost_per_million=None, cache_write_1h_cost_per_million=None)
        got = _cache_pricing(base, args, 'https://api.anthropic.com')
        self.assertEqual((got.cache_read_input_cost_per_million_tokens_usd,
                          got.cache_creation_input_cost_per_million_tokens_usd,
                          got.cache_creation_1h_input_cost_per_million_tokens_usd), (0.5, 12.5, 20.0))
        args.cache_read_cost_per_million = 0.3
        self.assertEqual(_cache_pricing(base, args, 'https://api.anthropic.com')
                         .cache_read_input_cost_per_million_tokens_usd, 0.3)
        unknown = _cache_pricing(base, Namespace(model='m'), 'http://127.0.0.1:9999/v1')
        self.assertEqual(unknown.cache_read_input_cost_per_million_tokens_usd, 0.0)
        gpt = _cache_pricing(base, Namespace(model='gpt-5.5'), 'https://api.openai.com/v1')
        self.assertEqual(gpt.cache_read_input_cost_per_million_tokens_usd, 1.0)

    def test_sessions_round_trip_and_old_sessions_load(self) -> None:
        from src.session_store import _deserialize_pricing, serialize_model_config, usage_from_payload

        usage = UsageStats(input_tokens=1, cache_creation_input_tokens=5, cache_creation_1h_input_tokens=3)
        self.assertEqual(usage_from_payload(usage.to_dict()), usage)
        self.assertEqual(usage_from_payload({'input_tokens': 4}), UsageStats(input_tokens=4))  # pre-1.1.0
        config = ModelConfig(model='m', pricing=ModelPricing(cache_creation_1h_input_cost_per_million_tokens_usd=2))
        self.assertEqual(_deserialize_pricing(serialize_model_config(config)['pricing']), config.pricing)
        self.assertEqual(_deserialize_pricing({'input_cost_per_million_tokens_usd': 1}),
                         ModelPricing(input_cost_per_million_tokens_usd=1))
        self.assertEqual((usage + usage).cache_creation_1h_input_tokens, 6)

    def test_telemetry_carries_cache_fields(self) -> None:
        from src import telemetry

        with mock.patch.object(telemetry, '_PENDING', []) as pending, mock.patch.object(telemetry, '_start'), \
                mock.patch.object(telemetry, 'enabled', return_value=True):
            telemetry.record({'type': 'model_usage', 'model': 'm', 'input_tokens': 1,
                              'cache_creation_input_tokens': 7, 'cache_creation_1h_input_tokens': 2,
                              'cache_hit_rate': 0.5})
            self.assertEqual(len(pending), 1)
            attrs = {a['key']: a['value'] for a in pending[0]['attributes']}
        self.assertEqual(attrs['cache_creation_input_tokens'], {'intValue': '7'})
        self.assertEqual(attrs['cache_creation_1h_input_tokens'], {'intValue': '2'})
        self.assertEqual(attrs['cache_hit_rate'], {'doubleValue': 0.5})


class CommandTests(_Base):
    def _agent(self, usage: UsageStats):  # noqa: ANN202
        return Namespace(cumulative_usage=usage, cumulative_cost_usd=0.0,
                         model_config=ModelConfig(model='claude-x', base_url='https://api.anthropic.com'),
                         runtime_config=Namespace(cwd=Path(tempfile.gettempdir())),
                         render_context_report=lambda prompt=None: 'Context: 1 token\n')

    def test_cache_command_sets_and_shows(self) -> None:
        from src.agent_slash_commands import _handle_cache, _handle_context, _handle_cost

        agent = self._agent(UsageStats(input_tokens=100, cache_read_input_tokens=300, cache_creation_input_tokens=100,
                                       cache_creation_1h_input_tokens=40))
        out = _handle_cache(agent, 'off', '/cache off').output
        self.assertIn('Prompt caching is off for this session.', out)
        self.assertEqual(prompt_cache.current_settings().mode, 'off')
        out = _handle_cache(agent, 'ttl 1h', '/cache ttl 1h').output
        self.assertIn('TTL:       1h (from /cache)', out)
        self.assertIn('Cache hit rate:        60%', out)
        self.assertIn('(40 with the 1h TTL)', out)
        self.assertEqual(_handle_cache(agent, '2h', '/cache 2h').output.splitlines()[0],
                         # explicit caches added list / delete / explicit (same exact check)
                         'Usage: /cache [on|off|auto|ttl 5m|1h|24h|reset|explicit on|off|auto|list|delete <name|all>]')
        _handle_cache(agent, 'reset', '/cache reset')
        self.assertEqual(prompt_cache.session_overrides(), {})
        self.assertIn('Cache hit rate:        60%', _handle_cost(agent, '', '/cost').output)
        self.assertIn('Prompt cache (this session):', _handle_context(agent, '', '/usage').output)
        self.assertNotIn('Prompt cache', _handle_context(self._agent(UsageStats()), '', '/usage').output)

    def test_status_line_payload(self) -> None:
        from src.status_line import _cache_payload

        self.assertEqual(_cache_payload(UsageStats(input_tokens=1, cache_read_input_tokens=3))['hit_rate'], 0.75)
        self.assertIsNone(_cache_payload(UsageStats())['hit_rate'])

    def test_background_agents_get_the_session_choice_only_when_set(self) -> None:
        env: dict[str, str] = {}
        prompt_cache.forward_env(env)
        self.assertEqual(env, {})
        prompt_cache.set_session(mode='off', ttl='1h')
        prompt_cache.forward_env(env)
        self.assertEqual(env, {'SIGIL_PROMPT_CACHE': 'off', 'SIGIL_PROMPT_CACHE_TTL': '1h'})
        self.assertNotIn('None', json.dumps(env))


class MicrocompactBatchTests(_Base):
    def _session(self, n_tools: int):  # noqa: ANN202
        from src.agent_session import AgentSessionState

        state = AgentSessionState.create(['system'], 'start')
        for i in range(n_tools):
            call = {'id': f'call_{i}', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
            state.append_assistant('', (call,))
            state.append_tool('read_file', f'call_{i}', f'R{i}:' + 'x' * 9000)
        return state

    def test_batch_waits_then_moves_together(self) -> None:
        from src.microcompact import microcompact_session

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(microcompact_session(self._session(7), Path(tmp), keep_recent=5, batch=3).offloaded_count, 0)
            self.assertEqual(microcompact_session(self._session(8), Path(tmp), keep_recent=5, batch=3).offloaded_count, 3)
            self.assertEqual(microcompact_session(self._session(7), Path(tmp), keep_recent=5).offloaded_count, 2)

    def test_batch_size_follows_the_cache(self) -> None:
        from src.microcompact import cache_friendly_batch

        self.assertEqual(cache_friendly_batch(None, 100.0), 1)  # nothing cached yet in this process
        self.assertEqual(cache_friendly_batch(90.0, 100.0), 3)  # warm
        self.assertEqual(cache_friendly_batch(0.0, 1000.0), 3)  # within the default 1h TTL: warm
        self.assertEqual(cache_friendly_batch(0.0, 4000.0), 1)  # older than 1h: cold
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_TTL': '5m'}):
            self.assertEqual(cache_friendly_batch(0.0, 1000.0), 1)  # older than 5m: cold
        with mock.patch.dict(os.environ, {'SIGIL_MICROCOMPACT_BATCH': '5'}):
            self.assertEqual(cache_friendly_batch(90.0, 100.0), 5)
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'off'}):
            self.assertEqual(cache_friendly_batch(90.0, 100.0), 1)


class EnvFileTests(_Base):
    def test_env_file_with_a_bom_and_the_ttl_line(self) -> None:
        """.env files saved with a UTF-8 BOM: the first comment line must not become a variable, and
        SIGIL_PROMPT_CACHE_TTL=1h set there reaches the cache settings."""
        from src.main import _load_env_file

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.env'
            path.write_bytes(b'\xef\xbb\xbf# ==== header ====\r\n# SIGIL_PROMPT_CACHE=auto  # comment\r\n'
                             b'SIGIL_PROMPT_CACHE_TTL=1h\r\n')
            with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_TTL': ''}):
                os.environ.pop('SIGIL_PROMPT_CACHE_TTL')
                _load_env_file(path)
                self.assertNotIn('﻿#', os.environ)
                self.assertEqual(os.environ.get('SIGIL_PROMPT_CACHE_TTL'), '1h')
                self.assertEqual(prompt_cache.current_settings().ttl_source, 'SIGIL_PROMPT_CACHE_TTL')
            os.environ.pop('﻿#', None)


class EveryCallCachesTests(_Base):
    def test_side_clients_use_the_same_plan(self) -> None:
        from src.anthropic_client import fast_client

        config = ModelConfig(model='anthropic/claude-sonnet-5', base_url='https://openrouter.ai/api/v1')
        with mock.patch.dict(os.environ, {'SIGIL_FAST_MODEL': ''}), \
                mock.patch('src.settings.load_settings', return_value={}):
            side = fast_client(config)
        payload = side._build_payload(messages=MESSAGES, tools=[], stream=False, output_schema=None)
        self.assertIn('cache_control', json.dumps(payload['messages']))
        native = make_client(ModelConfig(model='claude-x', base_url='https://api.anthropic.com'))
        self.assertIsInstance(native, AnthropicClient)
        self.assertIn('cache_control', json.dumps(native.build_payload(MESSAGES, [])))


if __name__ == '__main__':
    unittest.main()
