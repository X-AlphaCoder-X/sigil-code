"""Amazon Bedrock Converse (cachePoint caching, SigV4 and bearer auth, eventstream), Fireworks
cached-token headers and Vertex explicit-cache prices. Offline: urlopen is a fake AWS / Fireworks."""
from __future__ import annotations

import datetime
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib import error

from src import prompt_cache
from src.agent_types import ModelConfig, ModelPricing
from src.bedrock_client import (BedrockConverseClient, bedrock_usage, encode_frame, model_kind, sign,
                                use_bedrock_converse)
from src.debug_log import get_logger
from src.openai_compat import OpenAICompatClient, OpenAICompatError

BASE = 'https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1'
CLAUDE = 'us.anthropic.claude-sonnet-5-v1:0'
NOVA = 'us.amazon.nova-pro-v1:0'
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read',
                                            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]
MESSAGES = [{'role': 'system', 'content': 'You are a coding agent.'}, {'role': 'user', 'content': 'hi'}]
ENV = {'SIGIL_PROMPT_CACHE': '', 'SIGIL_PROMPT_CACHE_TTL': '', 'SIGIL_BEDROCK_CONVERSE': '',
       'AWS_BEARER_TOKEN_BEDROCK': '', 'AWS_ACCESS_KEY_ID': '', 'AWS_SECRET_ACCESS_KEY': '', 'AWS_SESSION_TOKEN': '',
       'AWS_PROFILE': '', 'AWS_REGION': '', 'AWS_DEFAULT_REGION': '', 'AWS_SHARED_CREDENTIALS_FILE': '',
       'SIGIL_ANTHROPIC_NATIVE': '', 'SIGIL_EXPLICIT_CACHE': ''}


class _Resp:
    def __init__(self, body: bytes, content_type: str = 'application/json', headers: dict | None = None) -> None:
        self.stream = io.BytesIO(body)
        self.headers = {'Content-Type': content_type, **(headers or {})}

    def read(self, n: int = -1) -> bytes:
        return self.stream.read(n)

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        return None


def converse_reply(text: str = 'ok', usage: dict | None = None, content: list | None = None, stop: str = 'end_turn') -> bytes:
    return json.dumps({'output': {'message': {'role': 'assistant', 'content': content or [{'text': text}]}},
                       'stopReason': stop,
                       'usage': usage or {'inputTokens': 10, 'outputTokens': 3, 'cacheReadInputTokens': 900,
                                          'cacheWriteInputTokens': 300,
                                          'cacheDetails': [{'inputTokens': 300, 'ttl': '1h'}]}}).encode()


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, ENV)
        patcher.start()
        self.addCleanup(patcher.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        prompt_cache.set_session(clear=True)
        self.addCleanup(prompt_cache.set_session, clear=True)
        self.calls: list = []
        self.replies: list = []

    def fake(self, req, timeout=None):  # noqa: ANN001
        self.calls.append((req.full_url, {k.lower(): v for k, v in req.header_items()},
                           json.loads(req.data) if req.data else None))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def client(self, model: str = CLAUDE) -> BedrockConverseClient:
        return BedrockConverseClient(ModelConfig(model=model, base_url=BASE, api_key='local-token', timeout_seconds=5))


class SigV4Tests(_Base):
    def test_aws_test_suite_vector_get_vanilla(self) -> None:
        """AWS SigV4 test suite 'get-vanilla': GET / on example.amazonaws.com, 20150830T123600Z."""
        creds = {'key': 'AKIDEXAMPLE', 'secret': 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY'}
        now = datetime.datetime(2015, 8, 30, 12, 36, 0, tzinfo=datetime.timezone.utc)
        headers = sign('GET', 'https://example.amazonaws.com/', {}, b'', creds, 'us-east-1', service='service', now=now)
        self.assertEqual(headers['Authorization'],
                         'AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, '
                         'SignedHeaders=host;x-amz-date, '
                         'Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31')

    def test_signed_request_to_converse_with_session_token(self) -> None:
        with mock.patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': 'AKIDTEST', 'AWS_SECRET_ACCESS_KEY': 'secret-xyz',
                                          'AWS_SESSION_TOKEN': 'session-tok'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.replies.append(_Resp(converse_reply()))
            self.client().complete(MESSAGES, TOOLS)
        url, headers, _body = self.calls[0]
        self.assertEqual(url, 'https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-5-v1%3A0/converse')
        self.assertRegex(headers['authorization'], r'^AWS4-HMAC-SHA256 Credential=AKIDTEST/\d{8}/us-east-1/bedrock/'
                         r'aws4_request, SignedHeaders=[a-z;-]*x-amz-security-token[a-z;-]*, Signature=[0-9a-f]{64}$')
        self.assertEqual(headers['x-amz-security-token'], 'session-tok')
        self.assertNotIn('secret-xyz', json.dumps(headers))

    def test_bearer_key_and_shared_credentials_file(self) -> None:
        with mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-bedrock-key'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.replies.append(_Resp(converse_reply()))
            self.client().complete(MESSAGES, [])
        self.assertEqual(self.calls[-1][1]['authorization'], 'Bearer ABSK-bedrock-key')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'credentials'
            path.write_text('[work]\naws_access_key_id = AKIDFILE\naws_secret_access_key = file-secret\n', encoding='utf-8')
            with mock.patch.dict(os.environ, {'AWS_SHARED_CREDENTIALS_FILE': str(path), 'AWS_PROFILE': 'work'}), \
                    mock.patch('urllib.request.urlopen', side_effect=self.fake):
                self.replies.append(_Resp(converse_reply()))
                self.client().complete(MESSAGES, [])
        self.assertIn('Credential=AKIDFILE/', self.calls[-1][1]['authorization'])

    def test_no_credentials_is_a_clear_error(self) -> None:
        with self.assertRaisesRegex(OpenAICompatError, 'AWS_BEARER_TOKEN_BEDROCK'):
            self.client().complete(MESSAGES, [])


class RoutingAndPayloadTests(_Base):
    def test_claude_and_nova_use_converse_by_default(self) -> None:
        from src.anthropic_client import make_client

        self.assertIsInstance(make_client(ModelConfig(model=CLAUDE, base_url=BASE)), BedrockConverseClient)
        self.assertIsInstance(make_client(ModelConfig(model=NOVA, base_url=BASE)), BedrockConverseClient)
        self.assertIsInstance(make_client(ModelConfig(model='openai.gpt-oss-120b-1:0', base_url=BASE)),
                              OpenAICompatClient)
        with mock.patch.dict(os.environ, {'SIGIL_BEDROCK_CONVERSE': '0'}):
            self.assertIsInstance(make_client(ModelConfig(model=CLAUDE, base_url=BASE)), OpenAICompatClient)
        self.assertEqual((model_kind(CLAUDE), model_kind(NOVA), model_kind('meta.llama4')), ('claude', 'nova', ''))
        self.assertFalse(use_bedrock_converse(ModelConfig(model=CLAUDE, base_url='https://evil-amazonaws.com.io/v1')))

    def test_claude_cache_points_with_1h(self) -> None:
        history = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'q1'},
                   {'role': 'assistant', 'content': 'a1'}, {'role': 'user', 'content': 'q2'}]
        payload = self.client().build_payload(history, TOOLS)
        point = {'cachePoint': {'type': 'default', 'ttl': '1h'}}
        self.assertEqual(payload['toolConfig']['tools'][-1], point)
        self.assertEqual(payload['system'][-1], point)
        self.assertEqual(payload['messages'][0]['content'][-1], point)   # end of the previous request
        self.assertEqual(payload['messages'][-1]['content'][-1], point)  # end of this request
        self.assertEqual(json.dumps(payload).count('cachePoint'), 4)
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE_TTL': '5m'}):
            short = self.client().build_payload(MESSAGES, TOOLS)
        self.assertNotIn('ttl', json.dumps(short))
        with mock.patch.dict(os.environ, {'SIGIL_PROMPT_CACHE': 'off'}):
            self.assertNotIn('cachePoint', json.dumps(self.client().build_payload(MESSAGES, TOOLS)))

    def test_nova_has_no_tools_point_and_no_ttl(self) -> None:
        payload = self.client(NOVA).build_payload(MESSAGES, TOOLS)
        self.assertNotIn('cachePoint', json.dumps(payload['toolConfig']))
        self.assertEqual(payload['system'][-1], {'cachePoint': {'type': 'default'}})
        self.assertNotIn('additionalModelRequestFields', payload)

    def test_images_tools_thinking_and_reasoning_round_trip(self) -> None:
        reasoning = {'reasoningContent': {'reasoningText': {'text': 'plan', 'signature': 'sig-1'}}}
        tool_use = {'toolUse': {'toolUseId': 't1', 'name': 'read_file', 'input': {'path': 'a'}}}
        messages = [{'role': 'system', 'content': 'sys'},
                    {'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJD'}}]},
                    {'role': 'assistant', 'content': '', 'anthropic_content': [reasoning, tool_use],
                     'tool_calls': [{'id': 't1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 't1', 'content': 'file text'}]
        client = BedrockConverseClient(ModelConfig(model=CLAUDE, base_url=BASE, effort='high'))
        payload = client.build_payload(messages, TOOLS)
        self.assertEqual(payload['messages'][0]['content'][1], {'image': {'format': 'png', 'source': {'bytes': 'QUJD'}}})
        self.assertEqual(payload['messages'][1]['content'][:2], [reasoning, tool_use])
        self.assertEqual(payload['messages'][2]['content'][0]['toolResult']['toolUseId'], 't1')
        self.assertIn('thinking', payload['additionalModelRequestFields'])

    def test_usage_cost_and_a_full_agent_run(self) -> None:
        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig

        usage = bedrock_usage({'inputTokens': 10, 'outputTokens': 3, 'cacheReadInputTokens': 900,
                               'cacheWriteInputTokens': 300, 'cacheDetails': [{'inputTokens': 200, 'ttl': '1h'},
                                                                              {'inputTokens': 100, 'ttl': '5m'}]})
        self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens,
                          usage.cache_creation_1h_input_tokens, usage.prompt_tokens), (10, 900, 300, 200, 1210))
        pricing = ModelPricing(input_cost_per_million_tokens_usd=3.0, output_cost_per_million_tokens_usd=15.0,
                               cache_read_input_cost_per_million_tokens_usd=0.3,
                               cache_creation_input_cost_per_million_tokens_usd=3.75,
                               cache_creation_1h_input_cost_per_million_tokens_usd=6.0)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-key'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.replies.append(_Resp(converse_reply('hello')))
            ws = Path(tmp)
            agent = LocalCodingAgent(model_config=ModelConfig(model=CLAUDE, base_url=BASE, pricing=pricing),
                                     runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s',
                                                                       nudge_strategy='off'))
            result = agent.run('say hello')
        self.assertEqual(result.final_output, 'hello')
        self.assertEqual(agent.cumulative_usage.cache_read_input_tokens, 900)
        expected = (10 * 3.0 + 3 * 15.0 + 900 * 0.3 + 300 * 6.0) / 1e6
        self.assertAlmostEqual(agent.cumulative_cost_usd, expected, places=9)
        self.assertIn('cachePoint', json.dumps(self.calls[0][2]))


class StreamTests(_Base):
    def frames(self, events: list[tuple[str, dict]], tail: bytes = b'') -> bytes:
        return b''.join(encode_frame({':message-type': 'event', ':event-type': name, ':content-type': 'application/json'},
                                     json.dumps(body).encode()) for name, body in events) + tail

    def test_text_tool_reasoning_usage(self) -> None:
        events = [('messageStart', {'role': 'assistant'}),
                  ('contentBlockDelta', {'contentBlockIndex': 0, 'delta': {'reasoningContent': {'text': 'think'}}}),
                  ('contentBlockDelta', {'contentBlockIndex': 0, 'delta': {'reasoningContent': {'signature': 'sg'}}}),
                  ('contentBlockStop', {'contentBlockIndex': 0}),
                  ('contentBlockDelta', {'contentBlockIndex': 1, 'delta': {'text': 'Reading.'}}),
                  ('contentBlockStop', {'contentBlockIndex': 1}),
                  ('contentBlockStart', {'contentBlockIndex': 2, 'start': {'toolUse': {'toolUseId': 't9', 'name': 'read_file'}}}),
                  ('contentBlockDelta', {'contentBlockIndex': 2, 'delta': {'toolUse': {'input': '{"path"'}}}),
                  ('contentBlockDelta', {'contentBlockIndex': 2, 'delta': {'toolUse': {'input': ': "a"}'}}}),
                  ('contentBlockStop', {'contentBlockIndex': 2}),
                  ('messageStop', {'stopReason': 'tool_use'}),
                  ('metadata', {'usage': {'inputTokens': 5, 'outputTokens': 7, 'cacheReadInputTokens': 50,
                                          'cacheWriteInputTokens': 20, 'cacheDetails': [{'inputTokens': 20, 'ttl': '1h'}]}})]
        client = self.client()
        with mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-key'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.replies.append(_Resp(self.frames(events), 'application/vnd.amazon.eventstream'))
            out = list(client.stream(MESSAGES, TOOLS))
        self.assertTrue(self.calls[0][0].endswith('/converse-stream'))
        self.assertEqual(''.join(e.delta for e in out if e.type == 'content_delta'), 'Reading.')
        calls = [e for e in out if e.type == 'tool_call_delta']
        self.assertEqual((calls[0].tool_call_id, calls[0].tool_name), ('t9', 'read_file'))
        self.assertEqual(json.loads(''.join(c.arguments_delta for c in calls)), {'path': 'a'})
        usage = [e.usage for e in out if e.type == 'usage']
        self.assertEqual(len(usage), 1)
        self.assertEqual((usage[0].cache_read_input_tokens, usage[0].cache_creation_1h_input_tokens), (50, 20))
        self.assertEqual(out[-1].finish_reason, 'tool_calls')
        original = client.pop_original_content()
        self.assertEqual(original[0], {'reasoningContent': {'reasoningText': {'text': 'think', 'signature': 'sg'}}})

    def test_exception_frame_cut_stream_and_bad_crc(self) -> None:
        bad = [encode_frame({':message-type': 'exception', ':exception-type': 'throttlingException'},
                            b'{"message": "Too many requests"}')]
        cases = [(b''.join(bad), 'throttlingException'),
                 (self.frames([('messageStart', {'role': 'assistant'})]), 'ended before messageStop'),
                 (self.frames([('messageStart', {})])[:-1] + b'\x00', 'message check')]
        for body, pattern in cases:
            with mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-key'}), \
                    mock.patch('urllib.request.urlopen', side_effect=self.fake):
                self.replies.append(_Resp(body, 'application/vnd.amazon.eventstream'))
                with self.assertRaisesRegex(OpenAICompatError, pattern):
                    list(self.client().stream(MESSAGES, []))


class ErrorTests(_Base):
    def http_error(self, code: int, kind: str, message: str) -> error.HTTPError:
        return error.HTTPError(BASE, code, 'err', {'x-amzn-ErrorType': f'{kind}:http://internal'},
                               io.BytesIO(json.dumps({'message': message}).encode()))

    def test_cache_point_rejection_is_retried_once_without_points(self) -> None:
        with mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-key'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake), \
                self.assertLogs(get_logger('prompt_cache').name, level='INFO') as logs:
            self.replies += [self.http_error(400, 'ValidationException', 'cachePoint is not supported for this model'),
                             _Resp(converse_reply('fine'))]
            self.assertEqual(self.client().complete(MESSAGES, TOOLS).content, 'fine')
        self.assertIn('cachePoint', json.dumps(self.calls[0][2]))
        self.assertNotIn('cachePoint', json.dumps(self.calls[1][2]))
        self.assertTrue(any('cachePoint' in line for line in logs.output))

    def test_throttling_and_other_errors_carry_the_aws_type_and_are_not_retried(self) -> None:
        with mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-secret-key'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.replies.append(self.http_error(429, 'ThrottlingException', 'Rate exceeded'))
            with self.assertRaises(OpenAICompatError) as caught:
                self.client().complete(MESSAGES, TOOLS)
        self.assertEqual(str(caught.exception), 'HTTP 429 from Amazon Bedrock: ThrottlingException: Rate exceeded')
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('ABSK-secret-key', str(caught.exception))


class FireworksHeaderTests(_Base):
    def test_header_and_stream_perf_metrics(self) -> None:
        client = OpenAICompatClient(ModelConfig(model='accounts/fireworks/models/x', timeout_seconds=5,
                                                base_url='https://api.fireworks.ai/inference/v1'))
        body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                           'usage': {'prompt_tokens': 1000, 'completion_tokens': 5}}).encode()
        with mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.replies.append(_Resp(body, headers={'fireworks-cached-prompt-tokens': '800',
                                                     'fireworks-prompt-tokens': '1000'}))
            usage = client.complete(MESSAGES, []).usage
        self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens), (200, 800))
        state: dict = {}
        events = list(client._parse_stream_payload({'choices': [{'delta': {}, 'finish_reason': 'stop'}],
                                                    'perf_metrics': {'cached-prompt-tokens': 600}}, state))
        events += list(client._parse_stream_payload({'choices': [], 'usage': {'prompt_tokens': 1000,
                                                                              'completion_tokens': 2}}, state))
        usage = [e.usage for e in events if e.type == 'usage'][0]
        self.assertEqual((usage.input_tokens, usage.cache_read_input_tokens), (400, 600))
        reported = list(client._parse_stream_payload(
            {'choices': [], 'usage': {'prompt_tokens': 1000, 'prompt_tokens_details': {'cached_tokens': 100}}},
            {'cached': 900}))
        self.assertEqual([e.usage.cache_read_input_tokens for e in reported if e.type == 'usage'], [100])  # body wins


class VertexPriceTests(_Base):
    def test_vertex_prices_are_verified_now(self) -> None:
        from src import explicit_cache

        self.assertEqual(explicit_cache.VERTEX_PRICES['gemini-2.5-pro'], (1.25, 0.125, 4.50))
        self.assertEqual(explicit_cache.VERTEX_PRICES['gemini-2.5-flash'][2], 1.00)
        self.assertNotIn('gemini-3.8-flash', explicit_cache.VERTEX_PRICES)  # not in the explicit-caching table


if __name__ == '__main__':
    unittest.main()
