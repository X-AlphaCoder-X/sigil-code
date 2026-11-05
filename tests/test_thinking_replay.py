"""Replayed reasoning is never the reason a request fails: Anthropic thinking-block binding
(block_binding drop_block, strip-and-retry), Bedrock reasoningContent, and the OpenAI-compatible
round trip of reasoning_content (DeepSeek, Kimi) and Gemini 3 thought signatures. Offline: local
fake servers only."""
from __future__ import annotations

import http.server
import json
import os
import threading
import unittest
from unittest import mock

from src import thinking_replay
from src.agent_session import AgentMessage
from src.agent_types import ModelConfig
from src.anthropic_client import AnthropicClient, FallbackClient
from src.bedrock_client import BedrockConverseClient
from src.debug_log import get_logger
from src.openai_compat import OpenAICompatClient, OpenAICompatError

_CLEAN = {k: '' for k in ('SIGIL_THINKING_BINDING', 'SIGIL_REASONING_REPLAY', 'SIGIL_PROMPT_CACHE_PROVIDER',
                          'SIGIL_EXTRA_HEADERS', 'SIGIL_ANTHROPIC_STREAM', 'SIGIL_ANTHROPIC_NATIVE',
                          'SIGIL_EXCLUDE_PARAMS', 'SIGIL_EFFORT_LEVEL', 'SIGIL_RETRY_MAX')}
_CLEAN.update({'SIGIL_PROMPT_CACHE': 'off', 'SIGIL_EXPLICIT_CACHE': 'off'})

# The error from a user's session, as the Anthropic API sent it.
USER_ERROR = ('HTTP 400 from the Anthropic API: {"type":"error","error":{"type":"invalid_request_error",'
              '"message":"messages.1.content.2: Invalid `signature` in `thinking` block. The block is bound to a '
              'different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` '
              'to \\"drop_block\\". That setting requires the `thinking-binding-controls-2026-08-01` value in the '
              '`anthropic-beta` header. Content that preceded this block when it was created is missing from this '
              'request, starting at `messages.0.content.29`."},"request_id":"req_x"}')


class FakeServer:
    """A local HTTP server. reply(body, headers) -> (status, content_type, bytes)."""

    def __init__(self, reply) -> None:  # noqa: ANN001
        self.requests: list[tuple[dict, dict]] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
                outer.requests.append((body, {k.lower(): v for k, v in self.headers.items()}))
                status, ctype, data = reply(body, outer.requests)
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


def _anthropic_ok(extra: dict | None = None) -> tuple[int, str, bytes]:
    body = {'content': [{'type': 'text', 'text': 'done'}], 'stop_reason': 'end_turn',
            'usage': {'input_tokens': 5, 'output_tokens': 2}}
    body.update(extra or {})
    return 200, 'application/json', json.dumps(body).encode()


def _anthropic_error(message: str) -> tuple[int, str, bytes]:
    return 400, 'application/json', json.dumps(
        {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}}).encode()


def _thinking(sig: str) -> dict:
    return {'type': 'thinking', 'thinking': 'plan', 'signature': sig}


def _history() -> list[dict]:
    """A session with two tool loops whose assistant turns kept their thinking blocks."""
    return [
        {'role': 'system', 'content': 'You are helpful.'},
        {'role': 'user', 'content': 'fix it'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 't1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "a"}'}}],
         'anthropic_content': [_thinking('sigA'), {'type': 'text', 'text': 'reading'},
                               {'type': 'tool_use', 'id': 't1', 'name': 'read_file', 'input': {'path': 'a'}}]},
        {'role': 'tool', 'tool_call_id': 't1', 'content': 'file a'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 't2', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "b"}'}}],
         'anthropic_content': [_thinking('sigB'),
                               {'type': 'tool_use', 'id': 't2', 'name': 'read_file', 'input': {'path': 'b'}}]},
        {'role': 'tool', 'tool_call_id': 't2', 'content': 'file b'},
    ]


def _signatures(body: dict) -> list[str]:
    return [b.get('signature') for m in body['messages'] for b in m['content']
            if isinstance(b, dict) and b.get('type') == 'thinking']


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, _CLEAN)
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)

    def _server(self, reply) -> FakeServer:  # noqa: ANN001
        server = FakeServer(reply)
        self.addCleanup(server.close)
        return server


class AnthropicBindingTests(_Base):
    def _client(self, server: FakeServer, model: str = 'claude-opus-5-5') -> AnthropicClient:
        return AnthropicClient(ModelConfig(model=model, base_url=server.url, timeout_seconds=10))

    def test_claude_51_plus_asks_to_drop_bound_blocks_with_the_beta(self) -> None:
        server = self._server(lambda body, reqs: _anthropic_ok())
        self._client(server).complete(_history(), [])
        body, headers = server.requests[0]
        self.assertEqual(body['thinking'], {'type': 'adaptive',
                                            'block_binding': {'prefix_mismatch_behavior': 'drop_block'}})
        self.assertIn(thinking_replay.BINDING_BETA, headers['anthropic-beta'])
        self.assertEqual(_signatures(body), ['sigA', 'sigB'])  # replayed unchanged

    def test_user_beta_header_is_kept(self) -> None:
        server = self._server(lambda body, reqs: _anthropic_ok())
        with mock.patch.dict(os.environ, {'SIGIL_EXTRA_HEADERS': 'anthropic-beta: my-beta-1'}):
            self._client(server).complete(_history(), [])
        self.assertEqual(server.requests[0][1]['anthropic-beta'], f'my-beta-1,{thinking_replay.BINDING_BETA}')

    def test_error_mode_and_off(self) -> None:
        server = self._server(lambda body, reqs: _anthropic_ok())
        with mock.patch.dict(os.environ, {'SIGIL_THINKING_BINDING': 'error'}):
            self._client(server).complete(_history(), [])
        with mock.patch.dict(os.environ, {'SIGIL_THINKING_BINDING': 'off'}):
            self._client(server).complete(_history(), [])
        self.assertEqual(server.requests[0][0]['thinking']['block_binding'], {'prefix_mismatch_behavior': 'error'})
        self.assertNotIn('thinking', server.requests[1][0])
        self.assertNotIn('anthropic-beta', server.requests[1][1])

    def test_models_without_the_prefix_check_or_with_thinking_off_get_no_binding(self) -> None:
        server = self._server(lambda body, reqs: _anthropic_ok())
        self._client(server, 'claude-sonnet-4-5').complete(_history(), [])
        with mock.patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': 'off'}):
            self._client(server, 'claude-sonnet-5-5').complete(_history(), [])  # between_tools
        self.assertNotIn('block_binding', json.dumps(server.requests[0][0]))
        self.assertEqual(server.requests[1][0]['thinking'], {'type': 'between_tools'})
        self.assertNotIn('anthropic-beta', server.requests[1][1])

    def test_the_users_error_strips_the_named_block_and_later_ones_and_retries(self) -> None:
        def reply(body, reqs):  # noqa: ANN001
            return _anthropic_error(USER_ERROR.split('"message":"', 1)[1].split('"},"request_id"')[0]
                                    .replace('\\"', '"')) if len(reqs) == 1 else _anthropic_ok()
        server = self._server(reply)
        client = self._client(server, 'claude-x')  # no binding: the API enforces anyway
        turn = client.complete(_history(), [])
        self.assertEqual(turn.content, 'done')
        self.assertEqual(len(server.requests), 2)
        # messages.1.content.2 is not a thinking block here, so everything from that point on goes:
        # sigA sits at messages.1.content.0 (before it) and stays; sigB (later) is dropped
        self.assertEqual(_signatures(server.requests[0][0]), ['sigA', 'sigB'])
        self.assertEqual(_signatures(server.requests[1][0]), ['sigA'])
        # the tool calls and text of each turn stay
        self.assertIn('tool_use', json.dumps(server.requests[1][0]['messages'][3]))
        # the next request of the session does not send the refused block again
        client.complete(_history(), [])
        self.assertEqual(len(server.requests), 3)
        self.assertEqual(_signatures(server.requests[2][0]), ['sigA'])

    def test_an_error_that_names_no_block_drops_all_replayed_thinking(self) -> None:
        reply = lambda body, reqs: (_anthropic_error('Invalid `signature` in `thinking` block')  # noqa: E731
                                    if len(reqs) == 1 else _anthropic_ok())
        server = self._server(reply)
        self._client(server, 'claude-x').complete(_history(), [])
        self.assertEqual(_signatures(server.requests[1][0]), [])

    def test_a_proxy_that_refuses_block_binding_gets_it_no_more(self) -> None:
        reply = lambda body, reqs: (_anthropic_error('thinking.block_binding: Extra inputs are not permitted')  # noqa: E731
                                    if len(reqs) == 1 else _anthropic_ok())
        server = self._server(reply)
        client = self._client(server)
        client.complete(_history(), [])
        client.complete(_history(), [])
        self.assertIn('block_binding', server.requests[0][0]['thinking'])
        for body, headers in server.requests[1:]:
            self.assertNotIn('block_binding', body.get('thinking', {}))
            self.assertNotIn('anthropic-beta', headers)

    def test_stream_recovers_before_the_first_token(self) -> None:
        events = [{'type': 'message_start', 'message': {'usage': {'input_tokens': 3}}},
                  {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                  {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'hi'}},
                  {'type': 'content_block_stop', 'index': 0},
                  {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
                  {'type': 'message_stop'}]
        sse = ''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n' for e in events).encode()
        reply = lambda body, reqs: (_anthropic_error('messages.3.content.0: Invalid `signature` in `thinking` '  # noqa: E731
                                                     'block. The block is bound to a different conversation.')
                                    if len(reqs) == 1 else (200, 'text/event-stream', sse))
        server = self._server(reply)
        out = list(self._client(server, 'claude-x').stream(_history(), []))
        self.assertEqual(''.join(e.delta for e in out if e.type == 'content_delta'), 'hi')
        self.assertEqual(_signatures(server.requests[1][0]), ['sigA'])

    def test_a_dropped_block_report_is_logged(self) -> None:
        transformed = {'input_transformations': [
            {'type': 'thinking_dropped', 'path': 'messages.3.content.0', 'reason': 'prefix_binding_mismatch'}]}
        server = self._server(lambda body, reqs: _anthropic_ok(transformed))
        with self.assertLogs(get_logger('model'), level='WARNING') as logs:
            self._client(server, 'claude-opus-5-5-logtest').complete(_history(), [])
        self.assertIn('Anthropic dropped replayed thinking', '\n'.join(logs.output))
        self.assertIn('messages.3.content.0', '\n'.join(logs.output))

    def test_fallback_client_passes_kept_blocks_through(self) -> None:
        reply = {'content': [_thinking('sigN'), {'type': 'tool_use', 'id': 't9', 'name': 'x', 'input': {}}],
                 'stop_reason': 'tool_use', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
        server = self._server(lambda body, reqs: (200, 'application/json', json.dumps(reply).encode()))
        config = ModelConfig(model='claude-x', base_url=server.url, timeout_seconds=10, fallback_models=('other',))
        with mock.patch('src.anthropic_client.use_native_anthropic', return_value=True):
            client = FallbackClient(config)
            client.complete([{'role': 'user', 'content': 'go'}], [])
        self.assertEqual(client.pop_original_content()[0]['signature'], 'sigN')


class BedrockReasoningTests(_Base):
    def test_bound_reasoning_content_is_stripped_and_sent_again(self) -> None:
        client = BedrockConverseClient(ModelConfig(
            model='us.anthropic.claude-opus-5-5', base_url='https://bedrock-runtime.us-east-1.amazonaws.com'))
        history = [{'role': 'user', 'content': 'go'},
                   {'role': 'assistant', 'content': '', 'tool_calls': [
                       {'id': 't1', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}],
                    'anthropic_content': [
                        {'reasoningContent': {'reasoningText': {'text': 'plan', 'signature': 'bsig'}}},
                        {'toolUse': {'toolUseId': 't1', 'name': 'x', 'input': {}}}]},
                   {'role': 'tool', 'tool_call_id': 't1', 'content': 'ok'}]
        ok = {'output': {'message': {'role': 'assistant', 'content': [{'text': 'done'}]}},
              'stopReason': 'end_turn', 'usage': {'inputTokens': 1, 'outputTokens': 1}}
        error = OpenAICompatError('HTTP 400 from Amazon Bedrock: ValidationException: messages.1.content.0: '
                                  'Invalid `signature` in `thinking` block. The block is bound to a different '
                                  'conversation.')
        with mock.patch.object(client, '_post', side_effect=[error, ok]) as post:
            turn = client.complete(history, [])
        self.assertEqual(turn.content, 'done')
        self.assertIn('bsig', json.dumps(post.call_args_list[0].args[0]))
        sent = post.call_args_list[1].args[0]
        self.assertNotIn('bsig', json.dumps(sent))
        self.assertIn('toolUse', json.dumps(sent['messages'][1]))
        with mock.patch.object(client, '_post', return_value=ok) as post:
            client.complete(history, [])
        self.assertNotIn('bsig', json.dumps(post.call_args.args[0]))


def _chat(message: dict) -> tuple[int, str, bytes]:
    return 200, 'application/json', json.dumps({'choices': [{'message': message, 'finish_reason': 'stop'}],
                                                 'usage': {'prompt_tokens': 3, 'completion_tokens': 1}}).encode()


def _compat_error(message: str) -> tuple[int, str, bytes]:
    return 400, 'application/json', json.dumps({'error': {'message': message}}).encode()


class CompatReplayTests(_Base):
    def _client(self, server: FakeServer, model: str) -> OpenAICompatClient:
        return OpenAICompatClient(ModelConfig(model=model, base_url=server.url, timeout_seconds=10))

    def _round_trip(self, client: OpenAICompatClient, first_reply: dict) -> list[dict]:
        """One reply, kept the way the runtime keeps it, then the history for the next request."""
        turn = client.complete([{'role': 'user', 'content': 'go'}], [])
        stored = AgentMessage(role='assistant', content=turn.content, tool_calls=tuple(
            {'id': c.id, 'type': 'function', 'function': {'name': c.name, 'arguments': json.dumps(c.arguments)}}
            for c in turn.tool_calls), metadata={'provider_replay': client.pop_replay()})
        return [{'role': 'user', 'content': 'go'}, stored.to_openai_message(),
                {'role': 'tool', 'tool_call_id': turn.tool_calls[0].id, 'content': 'result'}]

    def test_deepseek_gets_reasoning_content_back(self) -> None:
        reply = {'role': 'assistant', 'content': '', 'reasoning_content': 'think first',
                 'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]}
        server = self._server(lambda body, reqs: _chat(reply) if len(reqs) == 1 else _chat({'content': 'ok'}))
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_PROVIDER': 'deepseek'}):
            client = self._client(server, 'deepseek-v4-pro')
            history = self._round_trip(client, reply)
            client.complete(history, [{'type': 'function', 'function': {'name': 'x', 'parameters': {}}}])
        assistant = server.requests[1][0]['messages'][1]
        self.assertEqual(assistant['reasoning_content'], 'think first')
        self.assertNotIn('provider_replay', json.dumps(server.requests[1][0]))

    def test_a_server_that_asks_for_reasoning_content_gets_it_from_then_on(self) -> None:
        asked = 'thinking is enabled but reasoning_content is missing in assistant tool call message at index 1'

        def reply(body, reqs):  # noqa: ANN001
            return _compat_error(asked) if len(reqs) == 1 else _chat({'content': 'ok'})
        server = self._server(reply)
        client = self._client(server, 'some-thinking-model')
        history = [{'role': 'user', 'content': 'go'},
                   {'role': 'assistant', 'content': '', 'tool_calls': [
                       {'id': 'c1', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]},
                   {'role': 'tool', 'tool_call_id': 'c1', 'content': 'r'}]
        self.assertEqual(client.complete(history, []).content, 'ok')
        client.complete(history, [])
        self.assertNotIn('reasoning_content', server.requests[0][0]['messages'][1])
        self.assertEqual(server.requests[1][0]['messages'][1]['reasoning_content'], '')
        self.assertEqual(server.requests[2][0]['messages'][1]['reasoning_content'], '')

    def test_gemini_3_thought_signatures_go_back_and_foreign_calls_get_the_skip_value(self) -> None:
        signed = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'g1', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'},
             'extra_content': {'google': {'thought_signature': 'SIG_A'}}}]}
        server = self._server(lambda body, reqs: _chat(signed) if len(reqs) == 1 else _chat({'content': 'ok'}))
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_PROVIDER': 'gemini'}):
            client = self._client(server, 'gemini-3.1-pro-preview')
            history = self._round_trip(client, signed)
            foreign = {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'o1', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]}
            client.complete([foreign, {'role': 'tool', 'tool_call_id': 'o1', 'content': 'r'}] + history, [])
        messages = server.requests[1][0]['messages']
        self.assertEqual(messages[0]['tool_calls'][0]['extra_content'],
                         {'google': {'thought_signature': thinking_replay.SKIP_SIGNATURE}})
        self.assertEqual(messages[3]['tool_calls'][0]['extra_content'], {'google': {'thought_signature': 'SIG_A'}})

    def test_gemini_3_parallel_calls_follow_googles_first_call_rule(self) -> None:
        # Google: only the first parallel call carries a signature. The later calls the API wrote stay
        # unsigned, and a step from another model gets the skip value on its first call only.
        key = thinking_replay.endpoint_key('https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-3.1-pro')
        own = {'role': 'assistant', 'content': '', 'provider_replay': {
            'endpoint': key, 'tool_extras': {'p1': {'google': {'thought_signature': 'SIG_FIRST'}}}},
            'tool_calls': [{'id': f'p{i}', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}
                           for i in (1, 2, 3)]}
        foreign = {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': f'f{i}', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}} for i in (1, 2)]}
        out = thinking_replay.prepare([own, foreign], base_url='https://generativelanguage.googleapis.com/v1beta/openai',
                                      model='gemini-3.1-pro', provider='gemini', echo=False, fill=False,
                                      dummy_signatures=False)
        signed = [c.get('extra_content') for c in out[0]['tool_calls']]
        self.assertEqual(signed, [{'google': {'thought_signature': 'SIG_FIRST'}}, None, None])
        skipped = [c.get('extra_content') for c in out[1]['tool_calls']]
        self.assertEqual(skipped, [{'google': {'thought_signature': thinking_replay.SKIP_SIGNATURE}}, None])
        # a step that has a real signature anywhere gets no skip value at all
        late = dict(own, provider_replay={'endpoint': key, 'tool_extras': {'p2': {'google': {'thought_signature': 'S2'}}}})
        out = thinking_replay.prepare([late], base_url='https://generativelanguage.googleapis.com/v1beta/openai',
                                      model='gemini-3.1-pro', provider='gemini', echo=False, fill=False,
                                      dummy_signatures=False)
        self.assertEqual([c.get('extra_content') for c in out[0]['tool_calls']],
                         [None, {'google': {'thought_signature': 'S2'}}, None])

    def test_strict_providers_never_see_the_fields(self) -> None:
        replay = {'endpoint': 'https://other.example|m', 'reasoning_content': 'x',
                  'tool_extras': {'c1': {'google': {'thought_signature': 'S'}}}}
        server = self._server(lambda body, reqs: _chat({'content': 'ok'}))
        client = self._client(server, 'gpt-5.5')
        client.complete([{'role': 'user', 'content': 'go'},
                         {'role': 'assistant', 'content': '', 'provider_replay': replay, 'tool_calls': [
                             {'id': 'c1', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]},
                         {'role': 'tool', 'tool_call_id': 'c1', 'content': 'r'}], [])
        sent = json.dumps(server.requests[0][0])
        for word in ('provider_replay', 'reasoning_content', 'extra_content', 'thought_signature'):
            self.assertNotIn(word, sent)

    def test_a_server_that_refuses_the_fields_stops_getting_them(self) -> None:
        reply = lambda body, reqs: (_compat_error("Unrecognized request argument supplied: reasoning_content")  # noqa: E731
                                    if len(reqs) == 1 else _chat({'content': 'ok'}))
        server = self._server(reply)
        replay = {'endpoint': thinking_replay.endpoint_key(server.url, 'deepseek-v4-pro'), 'reasoning_content': 'r'}
        history = [{'role': 'user', 'content': 'go'},
                   {'role': 'assistant', 'content': 'a', 'provider_replay': replay}]
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_PROVIDER': 'deepseek'}):
            client = self._client(server, 'deepseek-v4-pro')
            self.assertEqual(client.complete(history, []).content, 'ok')
            client.complete(history, [])
        self.assertIn('reasoning_content', json.dumps(server.requests[0][0]))
        self.assertNotIn('reasoning_content', json.dumps(server.requests[1][0]))
        self.assertNotIn('reasoning_content', json.dumps(server.requests[2][0]))

    def test_an_unrelated_400_is_not_retried(self) -> None:
        server = self._server(lambda body, reqs: _compat_error('context length exceeded'))
        with self.assertRaises(OpenAICompatError):
            self._client(server, 'm').complete([{'role': 'user', 'content': 'go'}], [])
        self.assertEqual(len(server.requests), 1)

    def test_streamed_reasoning_and_signatures_are_kept(self) -> None:
        chunks = [{'choices': [{'delta': {'reasoning_content': 'step '}}]},
                  {'choices': [{'delta': {'reasoning_content': 'two'}}]},
                  {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'g1', 'function': {'name': 'x', 'arguments': '{}'},
                                                          'extra_content': {'google': {'thought_signature': 'S1'}}}]}}]},
                  {'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]}]
        sse = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks).encode() + b'data: [DONE]\n\n'
        server = self._server(lambda body, reqs: (200, 'text/event-stream', sse))
        client = self._client(server, 'gemini-3.1-pro-preview')
        list(client.stream([{'role': 'user', 'content': 'go'}], []))
        replay = client.pop_replay()
        self.assertEqual(replay['reasoning_content'], 'step two')
        self.assertEqual(replay['tool_extras'], {'g1': {'google': {'thought_signature': 'S1'}}})
        self.assertIsNone(client.pop_replay())


class ClassifyTests(unittest.TestCase):
    def test_the_users_error_is_recognised(self) -> None:
        self.assertEqual(thinking_replay.signature_error(USER_ERROR), (1, 2))
        self.assertFalse(thinking_replay.binding_rejected(USER_ERROR))
        self.assertIsNone(thinking_replay.signature_error('HTTP 429 rate limited: Invalid `signature`'))

    def test_session_message_keeps_replay_for_assistant_only(self) -> None:
        replay = {'endpoint': 'e', 'reasoning_content': 'r'}
        self.assertEqual(AgentMessage(role='assistant', content='a', metadata={'provider_replay': replay})
                         .to_openai_message()['provider_replay'], replay)
        self.assertNotIn('provider_replay', AgentMessage(role='user', content='u', metadata={'provider_replay': replay})
                         .to_openai_message())


if __name__ == '__main__':
    unittest.main()
