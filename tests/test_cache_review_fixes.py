"""Fixes from the review of explicit caches (Gemini, Vertex), Bedrock Converse and the cache displays.
Offline: urllib's urlopen is replaced by fake Google / AWS servers, and one test kills a real child
process to show a crashed run leaves no cache behind."""
from __future__ import annotations

import datetime
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest import mock
from urllib import error

from src import explicit_cache, google_auth, prompt_cache
from src.agent_types import ModelConfig, ModelPricing, OutputSchemaConfig, UsageStats
from src.bedrock_client import (BedrockConverseClient, encode_frame, eventstream_frames, sign,
                                use_bedrock_converse)
from src.openai_compat import OpenAICompatClient, OpenAICompatError

GEMINI = 'https://generativelanguage.googleapis.com/v1beta/openai'
VERTEX = 'https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/endpoints/openapi'
BEDROCK = 'https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1'
FIPS = 'https://bedrock-runtime-fips.us-east-1.amazonaws.com/openai/v1'
VPCE = 'https://vpce-0a1b2c-xyz.bedrock-runtime.us-east-1.vpce.amazonaws.com/openai/v1'
CLAUDE = 'us.anthropic.claude-sonnet-5-v1:0'
NOVA = 'us.amazon.nova-pro-v1:0'
NOVA2 = 'us.amazon.nova-2-lite-v1:0'
SYSTEM = 'You are a coding agent. ' + 'Rules. ' * 4000
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
                                            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]
MESSAGES = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'hi'}]
REPO = Path(__file__).resolve().parent.parent
_ENV = {'SIGIL_PROMPT_CACHE': '', 'SIGIL_PROMPT_CACHE_TTL': '', 'SIGIL_EXPLICIT_CACHE': '',
        'SIGIL_EXPLICIT_CACHE_MAX': '', 'SIGIL_GOOGLE_TOKEN_COMMAND': '', 'SIGIL_EXCLUDE_PARAMS': '',
        'SIGIL_BEDROCK_CONVERSE': '', 'AWS_BEARER_TOKEN_BEDROCK': '', 'AWS_ACCESS_KEY_ID': '',
        'AWS_SECRET_ACCESS_KEY': '', 'AWS_SESSION_TOKEN': '', 'AWS_PROFILE': '', 'AWS_REGION': '',
        'AWS_DEFAULT_REGION': '', 'SIGIL_EXTRA_HEADERS': '', 'SIGIL_EFFORT_LEVEL': '', 'SIGIL_FAST_MODEL': ''}


class _Resp:
    def __init__(self, body: Any = None, raw: bytes | None = None, content_type: str = 'application/json') -> None:
        self.body = raw if raw is not None else json.dumps(body).encode()
        self.stream = io.BytesIO(self.body)
        self.headers = {'Content-Type': content_type}

    def read(self, n: int = -1) -> bytes:
        return self.stream.read(n)

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        return None


def _http_error(url: str, status: int, text: str) -> error.HTTPError:
    return error.HTTPError(url, status, 'err', {}, io.BytesIO(json.dumps({'error': {'message': text}}).encode()))


def _named_cache(body: dict) -> str | None:
    return ((body.get('extra_body') or {}).get('google') or {}).get('cached_content')


class FakeGoogle:
    """cachedContents create / patch / delete / list and chat completions; records every call. A chat
    call that names a cache answers `chat_error` (status, text) while `errors_left` > 0."""

    def __init__(self, cache_id: str = 'AbC1') -> None:
        self.calls: list[tuple[str, str, dict]] = []
        self.cache_id = cache_id
        self.chat_error: tuple[int, str] | None = None
        self.errors_left = 0
        self.lock = threading.Lock()

    def __call__(self, req, timeout=None):  # noqa: ANN001
        url, method = req.full_url, req.get_method()
        body = json.loads(req.data) if req.data else {}
        with self.lock:
            self.calls.append((method, url, body))
        if url.endswith('/chat/completions'):
            if _named_cache(body) and self.chat_error and self.errors_left:
                self.errors_left -= 1
                raise _http_error(url, *self.chat_error)
            return _Resp({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': 7200, 'completion_tokens': 3}})
        if method == 'POST' and url.endswith('/cachedContents'):
            return _Resp({'name': f'cachedContents/{self.cache_id}', 'usageMetadata': {'totalTokenCount': 7100},
                          'expireTime': '2099-01-01T00:00:00Z'})
        if method == 'GET':
            return _Resp({'cachedContents': []})
        return _Resp({})

    def of(self, method: str, part: str = '') -> list:
        with self.lock:
            return [c for c in self.calls if c[0] == method and part in c[1]]


def _join_cache_threads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith('cache-'):
            thread.join(5)


class _GoogleBase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(_join_cache_threads)
        env = mock.patch.dict(os.environ, {**_ENV, 'HOME': self.home.name, 'USERPROFILE': self.home.name,
                                           'SIGIL_EXPLICIT_CACHE': 'on'})
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        prompt_cache.set_session(clear=True)
        explicit_cache.reset_for_tests()
        google_auth.reset()
        self.addCleanup(explicit_cache.reset_for_tests)
        self.google = FakeGoogle()
        patcher = mock.patch('urllib.request.urlopen', side_effect=self.google)
        patcher.start()
        self.addCleanup(patcher.stop)

    def client(self) -> OpenAICompatClient:
        explicit_cache._SWEPT.add(explicit_cache.endpoint_for(GEMINI, 'gemini-2.5-flash').id)
        return OpenAICompatClient(ModelConfig(model='gemini-2.5-flash', base_url=GEMINI, api_key='AIza-k',
                                              timeout_seconds=5))


class ExplicitCacheErrorTests(_GoogleBase):
    """B-EC1: only an error about the cache drops it; a rate limit, an auth error or another 400 keeps it."""

    def test_errors_not_about_the_cache_keep_it_and_are_not_resent(self) -> None:
        for status, text in ((429, 'Resource has been exhausted (e.g. check quota).'),
                             (401, 'Request had invalid authentication credentials.'),
                             (400, 'The input token count exceeds the maximum number of tokens allowed.')):
            explicit_cache.reset_for_tests()
            self.google.calls.clear()
            self.google.chat_error, self.google.errors_left = (status, text), 1
            client = self.client()
            with self.assertRaises(OpenAICompatError):
                client.complete(MESSAGES, TOOLS)
            _join_cache_threads()
            self.assertEqual(len(self.google.of('POST', '/chat/completions')), 1, status)  # not resent at once
            self.assertEqual(self.google.of('DELETE'), [], status)                        # the cache is kept
            self.assertEqual([c['name'] for c in explicit_cache.live_caches()], ['cachedContents/AbC1'], status)

    def test_errors_about_the_cache_resend_without_it(self) -> None:
        for status, text in ((403, 'CachedContent not found (or permission denied)'),
                             (404, 'Not found'),
                             (400, 'CachedContent can not be used with GenerateContent request setting '
                                   'system_instruction, tools or tool_config.')):
            explicit_cache.reset_for_tests()
            self.google.calls.clear()
            self.google.chat_error, self.google.errors_left = (status, text), 1
            self.assertEqual(self.client().complete(MESSAGES, TOOLS).content, 'ok', status)
            chats = self.google.of('POST', '/chat/completions')
            self.assertEqual(len(chats), 2, status)
            self.assertIsNone(_named_cache(chats[1][2]), status)
            self.assertEqual(chats[1][2]['messages'][0]['role'], 'system', status)
            _join_cache_threads()
            self.assertEqual(explicit_cache.live_caches(), [], status)


class CachedPriceTests(unittest.TestCase):
    """G1: cached reads on Gemini / Vertex (10%), Bedrock Nova (25%) and SambaNova (10%) were priced at 100%."""

    def test_read_ratios(self) -> None:
        with mock.patch.dict(os.environ, _ENV), mock.patch('src.prompt_cache._settings_value', return_value={}):
            self.assertEqual(prompt_cache.price_ratios(GEMINI, 'gemini-2.5-flash').read, 0.1)
            self.assertEqual(prompt_cache.price_ratios(VERTEX, 'google/gemini-2.5-pro').read, 0.1)
            self.assertEqual(prompt_cache.price_ratios(BEDROCK, NOVA).read, 0.25)
            self.assertEqual(prompt_cache.price_ratios('https://api.sambanova.ai/v1', 'MiniMax-M2.7').read, 0.1)
            self.assertEqual(prompt_cache.price_ratios(BEDROCK, CLAUDE).write_1h, 2.0)  # Claude unchanged

    def test_derived_gemini_cached_price(self) -> None:
        from src.main import _cache_pricing

        with mock.patch.dict(os.environ, _ENV), mock.patch('src.prompt_cache._settings_value', return_value={}), \
                mock.patch('src.prompt_cache.known_prices', return_value={}):
            got = _cache_pricing(ModelPricing(input_cost_per_million_tokens_usd=0.30), Namespace(model='gemini-2.5-flash'),
                                 GEMINI)
        self.assertAlmostEqual(got.cache_read_input_cost_per_million_tokens_usd, 0.03)


class _AwsBase(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, {**_ENV, 'AWS_BEARER_TOKEN_BEDROCK': 'ABSK-test'})
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        prompt_cache.set_session(clear=True)
        self.addCleanup(prompt_cache.set_session, clear=True)
        from src import bedrock_client

        bedrock_client._OUTPUT_CONFIG_REFUSED.clear()
        self.addCleanup(bedrock_client._OUTPUT_CONFIG_REFUSED.clear)
        self.calls: list = []
        self.replies: list = []

    def fake(self, req, timeout=None):  # noqa: ANN001
        self.calls.append((req.full_url, {k.lower(): v for k, v in req.header_items()}, json.loads(req.data)))
        reply = self.replies.pop(0) if self.replies else _Resp({
            'output': {'message': {'role': 'assistant', 'content': [{'text': 'ok'}]}}, 'stopReason': 'end_turn',
            'usage': {'inputTokens': 1, 'outputTokens': 1}})
        if isinstance(reply, Exception):
            raise reply
        return reply

    def client(self, model: str = CLAUDE, base: str = BEDROCK, **extra: Any) -> BedrockConverseClient:
        return BedrockConverseClient(ModelConfig(model=model, base_url=base, timeout_seconds=5, **extra))


class BedrockHostTests(_AwsBase):
    """B-BR1: FIPS endpoints use Converse on their own host; other amazonaws hosts get no cache markers."""

    def test_fips_uses_converse_on_the_fips_host(self) -> None:
        self.assertTrue(use_bedrock_converse(ModelConfig(model=CLAUDE, base_url=FIPS)))
        with mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.client(base=FIPS).complete(MESSAGES[:1] + [{'role': 'user', 'content': 'hi'}], [])
        self.assertTrue(self.calls[0][0].startswith('https://bedrock-runtime-fips.us-east-1.amazonaws.com/model/'))

    def test_other_amazonaws_hosts_get_no_markers(self) -> None:
        self.assertFalse(use_bedrock_converse(ModelConfig(model=CLAUDE, base_url=VPCE)))
        plan = prompt_cache.plan_for(VPCE, CLAUDE)
        self.assertFalse(plan.markers)
        self.assertTrue(prompt_cache.plan_for(BEDROCK, CLAUDE).markers)


class BedrockSigningTests(_AwsBase):
    """B-BR3 and B-SIG1."""

    def test_extra_headers_are_signed(self) -> None:
        with mock.patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': '', 'AWS_ACCESS_KEY_ID': 'AKIDTEST',
                                          'AWS_SECRET_ACCESS_KEY': 'secret-xyz',
                                          'SIGIL_EXTRA_HEADERS': '{"X-Trace-Id": "t1", "Content-Type": "application/json"}'}), \
                mock.patch('urllib.request.urlopen', side_effect=self.fake):
            self.client().complete([{'role': 'user', 'content': 'hi'}], [])
        _url, headers, _body = self.calls[0]
        signed = headers['authorization'].split('SignedHeaders=')[1].split(',')[0].split(';')
        self.assertIn('x-trace-id', signed)
        self.assertEqual(headers['x-trace-id'], 't1')

    def test_aws_vectors_header_value_trim_multiline_and_encoded_query_order(self) -> None:
        creds = {'key': 'AKIDEXAMPLE', 'secret': 'wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY'}
        now = datetime.datetime(2015, 8, 30, 12, 36, 0, tzinfo=datetime.timezone.utc)

        def signature(url: str, headers: dict) -> str:
            out = sign('GET', url, headers, b'', creds, 'us-east-1', service='service', now=now)
            return out['Authorization'].rsplit('Signature=', 1)[1]
        # awslabs/aws-c-auth tests/aws-signing-test-suite/v4
        self.assertEqual(signature('https://example.amazonaws.com/', {'My-Header1': ' value1',
                                                                       'My-Header2': ' "a   b   c"'}),
                         'acc3ed3afb60bb290fc8d2dd0098b9911fcaa05412b367055dee359757a9c736')
        self.assertEqual(signature('https://example.amazonaws.com/', {'My-Header1': 'value1\n  value2\n    value3'}),
                         'cfd34249e4b1c8d6b91ef74165d41a32e5fab3306300901bb65a51a73575eefd')
        self.assertEqual(signature('https://example.amazonaws.com/?Param-3=Value3&Param=Value2&%E1%88%B4=Value1', {}),
                         '371d3713e185cc334048618a97f809c9ffe339c62934c032af5a0e595648fcac')


class BedrockParsingTests(_AwsBase):
    """B-BR4 and the tool-input shape."""

    def test_header_bytes_past_the_frame_are_a_clear_error(self) -> None:
        import struct
        import zlib

        raw = bytes([3]) + b'abc' + bytes([7]) + struct.pack('>H', 200) + b'x'   # says 200 bytes, has 1
        total = 12 + len(raw) + 4
        prelude = struct.pack('>II', total, len(raw))
        prelude += struct.pack('>I', zlib.crc32(prelude) & 0xffffffff)
        message = prelude + raw
        frame = message + struct.pack('>I', zlib.crc32(message) & 0xffffffff)
        with self.assertRaises(OpenAICompatError):
            list(eventstream_frames(io.BytesIO(frame)))
        prelude = struct.pack('>II', 16 + 4, 9)                                    # header length past the end
        prelude += struct.pack('>I', zlib.crc32(prelude) & 0xffffffff)
        message = prelude + b'\x00\x00\x00\x00'
        with self.assertRaises(OpenAICompatError):
            list(eventstream_frames(io.BytesIO(message + struct.pack('>I', zlib.crc32(message) & 0xffffffff))))

    def test_tool_input_that_is_not_an_object(self) -> None:
        turn = self.client()._turn({'output': {'message': {'content': [
            {'toolUse': {'toolUseId': 't1', 'name': 'read_file', 'input': ['x']}},
            {'toolUse': 'garbage'}]}}, 'stopReason': 'tool_use'})
        self.assertEqual([(c.id, c.arguments) for c in turn.tool_calls], [('t1', {})])


class NovaReasoningTests(_AwsBase):
    """Nova 2 reasoningConfig from the effort setting (docs.aws.amazon.com/nova/latest/nova2-userguide)."""

    def test_effort_levels(self) -> None:
        cases = {'low': 'low', 'medium': 'medium', 'high': 'high', 'max': 'high'}
        for effort, expected in cases.items():
            payload = self.client(NOVA2, effort=effort, temperature=0.5).build_payload(MESSAGES, [])
            self.assertEqual(payload['additionalModelRequestFields']['reasoningConfig'],
                             {'type': 'enabled', 'maxReasoningEffort': expected}, effort)
            # "Temperature, topP and topK cannot be used with maxReasoningEffort set to high"
            self.assertEqual('temperature' in payload['inferenceConfig'], expected != 'high', effort)
        for effort in (None, 'off', 'auto'):
            payload = self.client(NOVA2, effort=effort).build_payload(MESSAGES, [])
            self.assertNotIn('additionalModelRequestFields', payload, effort)
        self.assertNotIn('additionalModelRequestFields', self.client(NOVA, effort='high').build_payload(MESSAGES, []))


class BedrockStructuredOutputTests(_AwsBase):
    """outputConfig.textFormat (docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html)."""

    SCHEMA = OutputSchemaConfig(name='answer', schema={'type': 'object', 'properties': {
        'a': {'type': 'string'}, 'b': {'type': 'object', 'properties': {'c': {'type': 'integer'}}}}})

    def test_output_config_shape(self) -> None:
        payload = self.client().build_payload([{'role': 'user', 'content': 'hi'}], [], self.SCHEMA)
        json_schema = payload['outputConfig']['textFormat']['structure']['jsonSchema']
        self.assertEqual(payload['outputConfig']['textFormat']['type'], 'json_schema')
        self.assertEqual(json_schema['name'], 'answer')
        schema = json.loads(json_schema['schema'])
        self.assertIs(schema['additionalProperties'], False)
        self.assertIs(schema['properties']['b']['additionalProperties'], False)
        self.assertNotIn('system', payload)  # no text instruction when the API does it

    def test_rejected_output_config_falls_back_once_and_is_remembered(self) -> None:
        self.replies.append(error.HTTPError(BEDROCK, 400, 'bad', {'x-amzn-ErrorType': 'ValidationException:x'},
                                            io.BytesIO(b'{"message": "outputConfig is not supported for this model"}')))
        client = self.client(NOVA)
        with mock.patch('urllib.request.urlopen', side_effect=self.fake):
            client.complete([{'role': 'user', 'content': 'hi'}], [], output_schema=self.SCHEMA)
            client.complete([{'role': 'user', 'content': 'hi'}], [], output_schema=self.SCHEMA)
        self.assertIn('outputConfig', self.calls[0][2])
        self.assertNotIn('outputConfig', self.calls[1][2])
        self.assertIn('JSON schema', json.dumps(self.calls[1][2]['system']))
        self.assertNotIn('outputConfig', self.calls[2][2])                # remembered for this model


class BedrockPrefixTests(_AwsBase):
    """Three turns: at most 4 cache points, and each request's read point sits where the previous
    request wrote, with the same bytes before it."""

    def _strip(self, value: Any) -> Any:
        return json.loads(json.dumps(value).replace(', {"cachePoint": {"type": "default", "ttl": "1h"}}', '')
                          .replace(', {"cachePoint": {"type": "default"}}', ''))

    def test_claude_and_nova(self) -> None:
        for model, with_tools in ((CLAUDE, True), (NOVA, False)):
            history = [{'role': 'system', 'content': SYSTEM}]
            payloads = []
            for turn in range(3):
                history.append({'role': 'user', 'content': f'question {turn}'})
                payloads.append(self.client(model).build_payload(list(history), TOOLS))
                history.append({'role': 'assistant', 'content': f'answer {turn}'})
            for index, payload in enumerate(payloads):
                text = json.dumps(payload)
                self.assertLessEqual(text.count('cachePoint'), 4, model)
                self.assertEqual('cachePoint' in json.dumps(payload['toolConfig']), with_tools, model)
                if index:
                    before = payloads[index - 1]
                    end = len(before['messages']) - 1
                    self.assertIn('cachePoint', json.dumps(payload['messages'][end]), model)  # read point
                    self.assertEqual(self._strip(payload['messages'][:end + 1]), self._strip(before['messages']), model)
                    self.assertEqual(payload['system'], before['system'], model)
                    self.assertEqual(payload['toolConfig'], before['toolConfig'], model)


class BedrockCoverageTests(_AwsBase):
    """/cache, side calls (fast model), background agents and the status line on Bedrock."""

    def test_cache_command_side_client_and_status_line(self) -> None:
        from src.agent_slash_commands import _handle_cache
        from src.anthropic_client import fast_client
        from src.bedrock_client import bedrock_usage
        from src.status_line import status_payload

        config = ModelConfig(model=CLAUDE, base_url=BEDROCK)
        agent = Namespace(cumulative_usage=bedrock_usage({'inputTokens': 10, 'outputTokens': 2,
                                                          'cacheReadInputTokens': 900, 'cacheWriteInputTokens': 90}),
                          cumulative_cost_usd=0.1, side_totals=lambda: (UsageStats(), 0.0, 0), model_config=config,
                          runtime_config=Namespace(cwd=Path(tempfile.gettempdir())), active_session_id='s',
                          _session_name=None, permission_mode='default')
        out = _handle_cache(agent, '', '/cache').output
        self.assertIn('cachePoint', out)
        self.assertIn('Cache read tokens:     900', out)
        with mock.patch('src.settings.load_settings', return_value={}):
            self.assertIsInstance(fast_client(config), BedrockConverseClient)
        self.assertEqual(status_payload(agent)['cache']['read_tokens'], 900)

    def test_background_agent_keeps_the_aws_variables(self) -> None:
        from src import agent_tools
        from src.agent_types import AgentPermissions

        seen: dict = {}

        class Runtime:
            root = Path(tempfile.gettempdir())

            def create_id(self) -> str:
                return 'bg_0123456789ab'

            def launch(self, command, **kwargs):  # noqa: ANN001, ANN003, ANN201
                seen.update(kwargs['env'])
                return Namespace(background_id='bg_0123456789ab', pid=1, log_path='x')

        context = agent_tools.ToolExecutionContext(
            root=Path(tempfile.gettempdir()), command_timeout_seconds=5, max_output_chars=1000,
            permissions=AgentPermissions(), model_config=ModelConfig(model=CLAUDE, base_url=BEDROCK))
        with mock.patch.object(agent_tools, '_background_runtime_for', return_value=Runtime()), \
                mock.patch('src.background_runtime.background_limit_message', return_value=None):
            agent_tools._background_agent_start({'prompt': 'do it'}, context)
        self.assertEqual(seen.get('AWS_BEARER_TOKEN_BEDROCK'), 'ABSK-test')


class DisplayFixTests(_GoogleBase):
    """B-CMD1 (/cache delete keeps the case of the name), B-CMD2 (/cost line order), E9 (storage in
    budgets and the printed / background record)."""

    def _agent(self) -> Namespace:
        return Namespace(cumulative_usage=UsageStats(input_tokens=1, cache_creation_input_tokens=40,
                                                     cache_creation_1h_input_tokens=40),
                         cumulative_cost_usd=1.0, side_totals=lambda *a: (UsageStats(), 0.0, 0),
                         model_config=ModelConfig(model='gemini-2.5-flash', base_url=GEMINI),
                         runtime_config=Namespace(cwd=Path(self.home.name)), active_session_id='s',
                         _session_name=None, permission_mode='default')

    def test_cache_delete_by_full_name_with_capitals(self) -> None:
        from src.agent_slash_commands import _handle_cache

        self.client().complete(MESSAGES, TOOLS)
        out = _handle_cache(self._agent(), 'delete cachedContents/AbC1', '/cache delete cachedContents/AbC1').output
        self.assertIn('Deleted 1 explicit cache(s): cachedContents/AbC1', out)
        self.assertEqual(explicit_cache.live_caches(), [])

    def test_cost_keeps_the_1h_line_under_cache_creation(self) -> None:
        from src.agent_slash_commands import _handle_cost

        self.client().complete(MESSAGES, TOOLS)
        lines = _handle_cost(self._agent(), '', '/cost').output.splitlines()
        at = next(i for i, line in enumerate(lines) if line.startswith('Cache creation tokens'))
        self.assertTrue(lines[at + 1].startswith('  with the 1h TTL'), lines)

    def test_storage_counts_in_budget_and_printed_record(self) -> None:
        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig, BudgetConfig
        from src.main import _with_side_calls

        with mock.patch('src.explicit_cache.storage_cost_usd', return_value=0.5):
            ws = Path(self.home.name)
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:9/v1'),
                                     runtime_config=AgentRuntimeConfig(
                                         cwd=ws, session_directory=ws / '.s', nudge_strategy='off',
                                         budget_config=BudgetConfig(max_total_cost_usd=1.0)))
            decision = agent._check_budget(UsageStats(), 0.6, tool_call_count=0, delegated_tasks=0,
                                           model_calls=0, session_turns=0)
            self.assertTrue(decision.exceeded, decision)                  # 0.6 + 0.5 storage > 1.0
            result = Namespace(session_id='s1', usage=UsageStats(), total_cost_usd=1.0)
            with mock.patch('src.main.replace', side_effect=lambda obj, **kw: Namespace(**{**vars(obj), **kw})):
                shown = _with_side_calls(agent, result)
        self.assertAlmostEqual(shown.total_cost_usd, 1.5)


class CrashLeakTests(unittest.TestCase):
    """A real child process creates a cache and is killed (no atexit); the next run deletes it."""

    CHILD = textwrap.dedent('''
        import json, os, sys, time
        from unittest import mock
        sys.path.insert(0, sys.argv[2])
        from src import explicit_cache
        from src.agent_types import ModelConfig
        from src.openai_compat import OpenAICompatClient
        log = sys.argv[1]
        class Resp:
            def __init__(self, body): self.body = json.dumps(body).encode(); self.headers = {}
            def read(self, n=-1): return self.body
            def __enter__(self): return self
            def __exit__(self, *a): return None
        def fake(req, timeout=None):
            with open(log, 'a', encoding='utf-8') as f:
                f.write(json.dumps([req.get_method(), req.full_url]) + chr(10))
            if req.full_url.endswith('/chat/completions'):
                return Resp({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
            if req.get_method() == 'POST':
                return Resp({'name': 'cachedContents/k1', 'usageMetadata': {'totalTokenCount': 7100},
                             'expireTime': '2099-01-01T00:00:00Z'})
            return Resp({'cachedContents': []})
        base = 'https://generativelanguage.googleapis.com/v1beta/openai'
        with mock.patch('urllib.request.urlopen', side_effect=fake):
            explicit_cache._SWEPT.add(explicit_cache.endpoint_for(base, 'gemini-2.5-flash').id)
            client = OpenAICompatClient(ModelConfig(model='gemini-2.5-flash', base_url=base, api_key='AIza-k'))
            client.complete([{'role': 'system', 'content': 'Rules. ' * 4000}, {'role': 'user', 'content': 'hi'}], [])
            print('READY', flush=True)
            time.sleep(120)
    ''')

    def test_killed_run_leaves_a_registry_entry_and_the_next_run_deletes_it(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            script, log = Path(home) / 'child.py', Path(home) / 'calls.log'
            script.write_text(self.CHILD, encoding='utf-8')
            env = {**os.environ, **_ENV, 'HOME': home, 'USERPROFILE': home, 'SIGIL_EXPLICIT_CACHE': 'on',
                   'PYTHONIOENCODING': 'utf-8'}
            child = subprocess.Popen([sys.executable, str(script), str(log), str(REPO)], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, cwd=home)
            try:
                line = child.stdout.readline().decode('utf-8', 'replace').strip()
                self.assertEqual(line, 'READY', child.stderr.read(2000) if line != 'READY' else '')
            finally:
                child.kill()     # TerminateProcess / SIGKILL: no atexit, no release_all
                child.wait(10)
                child.stdout.close()
                child.stderr.close()
            calls = [json.loads(x) for x in log.read_text(encoding='utf-8').splitlines()]
            self.assertIn('POST', [m for m, u in calls if u.endswith('/cachedContents')])
            self.assertFalse([c for c in calls if c[0] == 'DELETE'])          # the kill skipped clean-up
            registry = Path(home) / '.sigil' / 'explicit-caches.json'
            self.assertEqual(json.loads(registry.read_text(encoding='utf-8'))['cachedContents/k1']['pid'], child.pid)
            deleted: list = []

            def fake(req, timeout=None):  # noqa: ANN001
                if req.get_method() == 'DELETE':
                    deleted.append(req.full_url.rsplit('/', 1)[-1])
                return _Resp({'cachedContents': []})
            with mock.patch.dict(os.environ, {'HOME': home, 'USERPROFILE': home}), \
                    mock.patch('urllib.request.urlopen', side_effect=fake):
                explicit_cache.reset_for_tests()
                try:
                    count = explicit_cache.sweep(explicit_cache.endpoint_for(GEMINI, 'gemini-2.5-flash'), lambda: 'k')
                finally:
                    explicit_cache.reset_for_tests()
                self.assertEqual((count, deleted), (1, ['k1']))
                self.assertEqual(json.loads(registry.read_text(encoding='utf-8')), {})


if __name__ == '__main__':
    unittest.main()
