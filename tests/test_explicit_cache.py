"""Explicit context caches (Gemini API, Vertex AI): the price rule, create / renew / delete calls,
what a request leaves out, the fallbacks, crash clean-up, the cap, /cache and the Vertex token.
Offline: urllib's urlopen is replaced by a fake Google server."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock
from urllib import error

from src import explicit_cache, google_auth, prompt_cache
from src.agent_types import ModelConfig
from src.openai_compat import OpenAICompatClient

GEMINI = 'https://generativelanguage.googleapis.com/v1beta/openai'
VERTEX = 'https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/endpoints/openapi'
SYSTEM = 'You are a coding agent. ' + 'Rules. ' * 4000          # about 7,000 tokens: above every minimum
TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file',
                                            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]
MESSAGES = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'hi'}]


class _Resp:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode()
        self.headers = {'Content-Type': 'application/json'}

    def read(self) -> bytes:
        return self.body

    def __enter__(self):  # noqa: ANN204
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        return None


class FakeGoogle:
    """Answers cachedContents create/list/patch/delete and chat completions; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict, dict]] = []
        self.created = 0
        self.fail_create = 0          # how many creates answer 400
        self.fail_chat_with_cache = 0  # how many chat calls naming a cache answer 404
        self.listing: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, req, timeout=None):  # noqa: ANN001
        url, method = req.full_url, req.get_method()
        body = json.loads(req.data) if req.data else {}
        headers = {k.lower(): v for k, v in req.header_items()}
        with self.lock:
            self.calls.append((method, url, body, headers))
        if url.endswith('/chat/completions'):
            if explicit_cache_name(body) and self.fail_chat_with_cache:
                self.fail_chat_with_cache -= 1
                raise error.HTTPError(url, 404, 'nf', {}, io.BytesIO(b'{"error": "CachedContent not found"}'))
            return _Resp({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': 7200, 'completion_tokens': 3,
                                    'prompt_tokens_details': {'cached_tokens': 0}}})
        if method == 'POST' and url.endswith('/cachedContents'):
            if self.fail_create:
                self.fail_create -= 1
                raise error.HTTPError(url, 400, 'bad', {}, io.BytesIO(b'{"error": "invalid tools"}'))
            self.created += 1
            parent = url.split('/v1beta/')[-1].split('/v1/')[-1].rsplit('cachedContents', 1)[0]
            return _Resp({'name': f'{parent}cachedContents/c{self.created}', 'usageMetadata': {'totalTokenCount': 7100},
                          'expireTime': '2099-01-01T00:00:00.123456789Z'})
        if method == 'GET' and '/cachedContents' in url:
            return _Resp({'cachedContents': self.listing})
        if method == 'PATCH':
            return _Resp({'expireTime': '2099-01-01T01:00:00Z'})
        if method == 'DELETE':
            return _Resp({})
        raise AssertionError(f'unexpected call {method} {url}')

    def of(self, method: str, part: str = '') -> list:
        with self.lock:
            return [c for c in self.calls if c[0] == method and part in c[1]]


def _join_cache_threads() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith('cache-'):
            thread.join(5)


def explicit_cache_name(body: dict) -> str | None:
    return ((body.get('extra_body') or {}).get('google') or {}).get('cached_content')


def wait_for(check, seconds: float = 3.0) -> bool:  # noqa: ANN001
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return check()


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.addCleanup(_join_cache_threads)  # runs first: background deletes finish before the home goes
        env = mock.patch.dict(os.environ, {'HOME': self.home.name, 'USERPROFILE': self.home.name,
                                           'SIGIL_PROMPT_CACHE': '', 'SIGIL_PROMPT_CACHE_TTL': '',
                                           'SIGIL_EXPLICIT_CACHE': '', 'SIGIL_EXPLICIT_CACHE_MAX': '',
                                           'SIGIL_GOOGLE_TOKEN_COMMAND': '', 'SIGIL_EXCLUDE_PARAMS': ''})
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

    def client(self, base: str = GEMINI, model: str = 'gemini-2.5-flash', key: str = 'AIza-test-key') -> OpenAICompatClient:
        endpoint = explicit_cache.endpoint_for(base, model)
        explicit_cache._SWEPT.add(endpoint.id)  # the crash sweep has its own tests
        return OpenAICompatClient(ModelConfig(model=model, base_url=base, api_key=key, timeout_seconds=5))

    def warm_up(self, client: OpenAICompatClient, misses: int = 3) -> None:
        """Three replies without implicit hits, one per minute: 60 requests/hour, miss rate 1."""
        start = time.time() - 240
        with mock.patch('time.time', return_value=start):  # one reply that shows the field is reported
            explicit_cache.observe(client.config.base_url, client.config.model,
                                   Namespace(prompt_tokens=7200, cache_read_input_tokens=50))
        for index in range(misses):
            with mock.patch('time.time', return_value=start + 60 * (index + 1)):
                explicit_cache.observe(client.config.base_url, client.config.model,
                                       Namespace(prompt_tokens=7200, cache_read_input_tokens=0))


class PriceRuleTests(_Base):
    def test_rule_and_its_numbers(self) -> None:
        flash = explicit_cache.GEMINI_PRICES['gemini-2.5-flash']        # 0.30 / 0.03 / 1.00
        self.assertTrue(explicit_cache.worth_it(4, 1.0, *flash))       # 4 x 0.27 = 1.08 > 1.00
        self.assertFalse(explicit_cache.worth_it(3, 1.0, *flash))      # 0.81 < 1.00
        self.assertFalse(explicit_cache.worth_it(60, 0.05, *flash))    # implicit hits 95%: 0.81
        pro = explicit_cache.GEMINI_PRICES['gemini-3.1-pro-preview']   # 2.00 / 0.20 / 4.50
        self.assertFalse(explicit_cache.worth_it(2.5, 1.0, *pro))      # 4.50, not more than storage
        self.assertTrue(explicit_cache.worth_it(3, 1.0, *pro))
        self.assertEqual(explicit_cache.min_tokens('gemini-2.5-flash'), 2048)
        self.assertEqual(explicit_cache.min_tokens('gemini-3.8-flash'), 4096)

    def test_auto_waits_for_evidence_then_creates(self) -> None:
        client = self.client()
        client.complete(MESSAGES, TOOLS)
        self.assertEqual(self.google.of('POST', '/cachedContents'), [])      # no evidence yet
        self.warm_up(client)
        client.complete(MESSAGES, TOOLS)
        create = self.google.of('POST', '/cachedContents')
        self.assertEqual(len(create), 1)
        _, url, body, headers = create[0]
        self.assertEqual(url, 'https://generativelanguage.googleapis.com/v1beta/cachedContents')
        self.assertEqual(body['model'], 'models/gemini-2.5-flash')
        self.assertEqual(body['ttl'], '3600s')                               # the 1h timer
        self.assertEqual(body['systemInstruction'], {'parts': [{'text': SYSTEM}]})
        declaration = body['tools'][0]['functionDeclarations'][0]
        self.assertEqual((declaration['name'], declaration['parametersJsonSchema']['type']), ('read_file', 'object'))
        self.assertRegex(body['displayName'], r'^sigil-[0-9a-f]{12}-\d+-[0-9a-f]{6}$')
        self.assertEqual(headers['x-goog-api-key'], 'AIza-test-key')
        self.assertNotIn('AIza', json.dumps(body))
        chat = self.google.of('POST', '/chat/completions')[-1][2]
        self.assertEqual(explicit_cache_name(chat), 'cachedContents/c1')
        self.assertEqual([m['role'] for m in chat['messages']], ['user'])    # the system prompt is in the cache
        self.assertNotIn('tools', chat)
        self.assertNotIn('tool_choice', chat)
        self.assertNotIn(explicit_cache.FULL_KEY, chat)
        client.complete(MESSAGES, TOOLS)                                     # reused, not created again
        self.assertEqual(len(self.google.of('POST', '/cachedContents')), 1)

    def test_implicit_hits_mean_no_explicit_cache(self) -> None:
        client = self.client()
        start = time.time() - 180
        for index in range(4):
            with mock.patch('time.time', return_value=start + 60 * index):
                explicit_cache.observe(GEMINI, 'gemini-2.5-flash',
                                       Namespace(prompt_tokens=7200, cache_read_input_tokens=7000))
        client.complete(MESSAGES, TOOLS)
        self.assertEqual(self.google.of('POST', '/cachedContents'), [])

    def test_small_prefix_unknown_price_and_implicit_only_models(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            self.client().complete([{'role': 'system', 'content': 'short'}, {'role': 'user', 'content': 'hi'}], [])
            self.client(model='gemini-4-argon').complete(MESSAGES, TOOLS)
        self.assertEqual(self.google.of('POST', '/cachedContents'), [])
        client = self.client(model='gemini-9-unknown')
        self.warm_up(client)
        client.complete(MESSAGES, TOOLS)                                     # auto: no storage price known
        self.assertEqual(self.google.of('POST', '/cachedContents'), [])

    def test_off_creates_nothing(self) -> None:
        for env in ({'SIGIL_EXPLICIT_CACHE': 'off'}, {'SIGIL_PROMPT_CACHE': 'off', 'SIGIL_EXPLICIT_CACHE': 'on'}):
            with mock.patch.dict(os.environ, env):
                client = self.client()
                self.warm_up(client)
                client.complete(MESSAGES, TOOLS)
        self.assertEqual(self.google.of('POST', '/cachedContents'), [])
        self.assertTrue(all(explicit_cache_name(c[2]) is None for c in self.google.of('POST', '/chat')))


class EvidenceTests(_Base):
    def test_zeros_from_an_endpoint_that_never_reports_cached_tokens_prove_nothing(self) -> None:
        client = self.client()
        start = time.time() - 240
        for index in range(4):
            with mock.patch('time.time', return_value=start + 60 * index):
                explicit_cache.observe(GEMINI, 'gemini-2.5-flash', Namespace(prompt_tokens=7200, cache_read_input_tokens=0))
        client.complete(MESSAGES, TOOLS)
        self.assertEqual(self.google.of('POST', '/cachedContents'), [])

    def test_vertex_has_no_assumed_price_and_its_own_minimums(self) -> None:
        # Vertex prices verified on 2026-11-05 (model-garden/pricing): a listed model follows the rule
        client = self.client(base=VERTEX, model='google/gemini-2.5-flash')
        self.warm_up(client)
        client.complete(MESSAGES, TOOLS)
        self.assertEqual(len(self.google.of('POST', '/cachedContents')), 1)
        # a model without a verified Vertex storage price still gets no cache in auto
        unpriced = self.client(base=VERTEX, model='google/gemini-3.8-flash')
        self.warm_up(unpriced)
        unpriced.complete(MESSAGES, TOOLS)
        self.assertEqual(len(self.google.of('POST', '/cachedContents')), 1)
        self.assertEqual(explicit_cache.min_tokens('gemini-3.1-pro-preview', 'vertex'), 6144)
        self.assertEqual(explicit_cache.min_tokens('gemini-2.0-flash', 'vertex'), 2048)
        self.assertEqual(explicit_cache.min_tokens('gemini-3.1-pro-preview'), 4096)


class OtherProviderTests(_Base):
    def test_cerebras_deepinfra_kimi_sambanova(self) -> None:
        from src.openai_compat import _parse_usage

        plan = prompt_cache.plan_for
        for url in ('https://api.cerebras.ai/v1', 'https://api.deepinfra.com/v1/openai', 'https://api.moonshot.ai/v1'):
            self.assertEqual(plan(url, 'm').key_field, 'prompt_cache_key', url)
        sambanova = plan('https://api.sambanova.ai/v1', 'MiniMax-M3')
        self.assertFalse(sambanova.key_field or sambanova.markers or sambanova.body)
        self.assertEqual(prompt_cache.price_ratios('https://api.cerebras.ai/v1', 'm').read, 1.0)
        usage = _parse_usage({'prompt_tokens': 5000, 'prompt_tokens_details': {'cache_creation_tokens': 4100}})
        self.assertEqual((usage.input_tokens, usage.cache_creation_input_tokens), (900, 4100))


class LifecycleTests(_Base):
    def test_ttl_is_renewed_when_half_is_left(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            client = self.client()
            client.complete(MESSAGES, TOOLS)
            live = next(iter(explicit_cache._LIVE.values()))
            live.expire = time.time() + 600                                  # 10 of 60 minutes left
            client.complete(MESSAGES, TOOLS)
        self.assertTrue(wait_for(lambda: self.google.of('PATCH')))
        _, url, body, _ = self.google.of('PATCH')[0]
        self.assertEqual(url, 'https://generativelanguage.googleapis.com/v1beta/cachedContents/c1?updateMask=ttl')
        self.assertEqual(body, {'ttl': '3600s'})

    def test_changed_prefix_gets_a_new_cache_and_the_stale_one_goes(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            client = self.client()
            client.complete(MESSAGES, TOOLS)
            for live in explicit_cache._LIVE.values():
                live.last_used -= 600                                        # not used for 10 minutes
            client.complete([{'role': 'system', 'content': SYSTEM + ' New rule.'}, {'role': 'user', 'content': 'hi'}],
                            TOOLS)
        self.assertEqual(len(self.google.of('POST', '/cachedContents')), 2)
        self.assertEqual([c[1].rsplit('/', 1)[-1] for c in self.google.of('DELETE')], ['c1'])
        self.assertEqual([c['name'] for c in explicit_cache.live_caches()], ['cachedContents/c2'])

    def test_cap(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on', 'SIGIL_EXPLICIT_CACHE_MAX': '1'}):
            client = self.client()
            client.complete(MESSAGES, TOOLS)
            client.complete(MESSAGES, TOOLS + [{'type': 'function', 'function': {'name': 'other', 'parameters': {}}}])
        self.assertEqual(len(explicit_cache.live_caches()), 1)
        self.assertEqual(len(self.google.of('DELETE')), 1)

    def test_create_failure_never_breaks_the_request(self) -> None:
        self.google.fail_create = 5
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            client = self.client()
            from src.debug_log import get_logger

            # the real logger name (the root differs between builds), not a hard-coded one
            with self.assertLogs(get_logger('explicit_cache').name, level='WARNING') as logs:
                for _ in range(3):
                    self.assertEqual(client.complete(MESSAGES, TOOLS).content, 'ok')
        self.assertEqual(len(self.google.of('POST', '/cachedContents')), 2)  # then off for this endpoint
        chats = self.google.of('POST', '/chat/completions')
        self.assertTrue(all(c[2]['messages'][0]['role'] == 'system' and 'tools' in c[2] for c in chats))
        self.assertTrue(any('off for this endpoint' in line for line in logs.output))

    def test_request_naming_a_dead_cache_is_sent_again_without_it(self) -> None:
        self.google.fail_chat_with_cache = 1
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            client = self.client()
            self.assertEqual(client.complete(MESSAGES, TOOLS).content, 'ok')
        chats = self.google.of('POST', '/chat/completions')
        self.assertEqual(len(chats), 2)
        self.assertEqual(explicit_cache_name(chats[0][2]), 'cachedContents/c1')
        self.assertIsNone(explicit_cache_name(chats[1][2]))
        self.assertEqual(chats[1][2]['messages'][0]['role'], 'system')      # the full request
        self.assertTrue(wait_for(lambda: self.google.of('DELETE')))
        self.assertEqual(explicit_cache.live_caches(), [])

    def test_stream_falls_back_too(self) -> None:
        self.google.fail_chat_with_cache = 1
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            client = self.client()
            with mock.patch.object(OpenAICompatClient, '_stream_payload',
                                   side_effect=lambda payload: iter(_stream_answer(self.google, payload))):
                events = list(client.stream(MESSAGES, TOOLS))
        self.assertEqual(''.join(e.delta for e in events if e.type == 'content_delta'), 'ok')

    def test_session_end_and_exit_delete_the_caches(self) -> None:
        from src.command_hooks import fire_session_end

        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            self.client().complete(MESSAGES, TOOLS)
        registry = Path(self.home.name) / '.sigil' / 'explicit-caches.json'
        self.assertIn('cachedContents/c1', json.loads(registry.read_text(encoding='utf-8')))
        fire_session_end(None, 's1', 'exit')
        self.assertEqual(len(self.google.of('DELETE')), 1)
        self.assertEqual(json.loads(registry.read_text(encoding='utf-8')), {})
        self.assertEqual(explicit_cache.live_caches(), [])

    def test_exit_delete_uses_a_short_timeout(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            self.client().complete(MESSAGES, TOOLS)
        seen: list = []
        original = self.google.__call__

        def recording(req, timeout=None):  # noqa: ANN001
            seen.append((req.get_method(), timeout))
            return original(req, timeout)
        with mock.patch('urllib.request.urlopen', side_effect=recording):
            explicit_cache.release_all('exit')
        self.assertEqual(seen, [('DELETE', 5.0)])
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT': '2'}):
            self.assertEqual(explicit_cache.delete_timeout(), 2.0)

    def test_crash_clean_up_touches_only_this_install_and_dead_processes(self) -> None:
        install = explicit_cache._install_id()
        dead = 2 ** 22 + 12345                                               # no such process
        registry = Path(self.home.name) / '.sigil' / 'explicit-caches.json'
        registry.write_text(json.dumps({
            'cachedContents/r1': {
                'name': 'cachedContents/r1', 'version_root': 'https://generativelanguage.googleapis.com/v1beta',
                'parent': '', 'pid': dead, 'display_name': f'sigil-{install}-{dead}-abcdef'},
            # a registry entry without this install's id in its name is never touched
            'cachedContents/r2': {
                'name': 'cachedContents/r2', 'version_root': 'https://generativelanguage.googleapis.com/v1beta',
                'parent': '', 'pid': dead}}), encoding='utf-8')
        self.google.listing = [
            {'name': 'cachedContents/l1', 'displayName': f'sigil-{install}-{dead}-abcdef'},          # orphan
            {'name': 'cachedContents/l2', 'displayName': f'sigil-{install}-{os.getpid()}-abcdef'},   # alive (us)
            {'name': 'cachedContents/l3', 'displayName': f'sigil-000000000000-{dead}-abcdef'},       # other install
            {'name': 'cachedContents/l4', 'displayName': 'someone-else'}]
        endpoint = explicit_cache.endpoint_for(GEMINI, 'gemini-2.5-flash')
        deleted = explicit_cache.sweep(endpoint, lambda: 'AIza-test-key')
        self.assertEqual(deleted, 2)
        self.assertEqual(sorted(c[1].rsplit('/', 1)[-1] for c in self.google.of('DELETE')), ['l1', 'r1'])
        self.assertEqual(explicit_cache.sweep(endpoint, lambda: 'k'), 0)     # once per endpoint

    def test_storage_cost_and_displays(self) -> None:
        from src.agent_slash_commands import _handle_cache
        from src.status_line import status_payload

        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            self.client().complete(MESSAGES, TOOLS)
        live = next(iter(explicit_cache._LIVE.values()))
        live.tokens, live.created = 1_000_000, time.time() - 1800           # 1M tokens for 30 minutes
        self.assertAlmostEqual(explicit_cache.storage_cost_usd(), 0.5, places=3)
        agent = Namespace(cumulative_usage=None, cumulative_cost_usd=1.0, side_totals=lambda: (None, 0.0),
                          model_config=ModelConfig(model='gemini-2.5-flash', base_url=GEMINI),
                          runtime_config=Namespace(cwd=Path(self.home.name)), active_session_id='s',
                          _session_name=None, permission_mode='default')
        out = _handle_cache(agent, 'list', '/cache list').output
        self.assertIn('cachedContents/c1  gemini-2.5-flash  1,000,000 tokens', out)
        self.assertIn('Cache storage:         $0.5000', out)
        payload = status_payload(agent)
        self.assertAlmostEqual(payload['cost']['cache_storage_usd'], 0.5, places=3)
        self.assertAlmostEqual(payload['cost']['total_cost_usd'], 1.5, places=3)
        self.assertEqual(payload['cache']['explicit'][0]['name'], 'cachedContents/c1')
        out = _handle_cache(agent, 'delete all', '/cache delete all').output
        self.assertIn('Deleted 1 explicit cache(s): cachedContents/c1', out)
        self.assertEqual(len(self.google.of('DELETE')), 1)
        self.assertAlmostEqual(explicit_cache.storage_cost_usd(), 0.5, places=3)  # spent stays counted

    def test_background_agents_get_the_explicit_choice(self) -> None:
        env: dict[str, str] = {}
        explicit_cache.forward_env(env)
        self.assertEqual(env, {})
        explicit_cache.set_session_mode('off')
        explicit_cache.forward_env(env)
        self.assertEqual(env, {'SIGIL_EXPLICIT_CACHE': 'off'})

    def test_telemetry_record(self) -> None:
        from src import telemetry

        with mock.patch.object(telemetry, '_PENDING', []) as pending, mock.patch.object(telemetry, '_start'), \
                mock.patch.object(telemetry, 'enabled', return_value=True), \
                mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on'}):
            self.client().complete(MESSAGES, TOOLS)
            attrs = {a['key']: a['value'] for a in pending[0]['attributes']}
        self.assertEqual(attrs['event.name'], {'stringValue': 'sigil.explicit_cache'})
        self.assertEqual(attrs['action'], {'stringValue': 'create'})
        self.assertEqual(attrs['tokens'], {'intValue': '7100'})


def _stream_answer(google: FakeGoogle, payload: dict):  # noqa: ANN202
    from src.agent_types import StreamEvent
    from src.openai_compat import OpenAICompatError

    with google.lock:
        google.calls.append(('POST', GEMINI + '/chat/completions', payload, {}))
    if explicit_cache_name(payload) and google.fail_chat_with_cache:
        google.fail_chat_with_cache -= 1
        raise OpenAICompatError('HTTP 404 from local model backend: CachedContent not found')
    yield StreamEvent(type='content_delta', delta='ok')
    yield StreamEvent(type='message_stop', finish_reason='stop')


class VertexTests(_Base):
    def test_vertex_urls_names_and_refreshed_token(self) -> None:
        command = f'"{sys.executable}" -c "print(\'ya29.token-1\')"'
        with mock.patch.dict(os.environ, {'SIGIL_EXPLICIT_CACHE': 'on', 'SIGIL_GOOGLE_TOKEN_COMMAND': command}):
            client = self.client(base=VERTEX, model='google/gemini-2.5-flash', key='stale-key')
            from src.debug_log import get_logger

            with self.assertLogs(get_logger('').name, level='DEBUG') as logs:
                client.complete(MESSAGES, TOOLS)
        _, url, body, headers = self.google.of('POST', '/cachedContents')[0]
        self.assertEqual(url, 'https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/'
                              'cachedContents')
        self.assertEqual(body['model'], 'projects/p1/locations/us-central1/publishers/google/models/gemini-2.5-flash')
        self.assertEqual(headers['authorization'], 'Bearer ya29.token-1')
        chat_headers = self.google.of('POST', '/chat/completions')[-1][3]
        self.assertEqual(chat_headers['authorization'], 'Bearer ya29.token-1')
        self.assertEqual(explicit_cache_name(self.google.of('POST', '/chat/completions')[-1][2]),
                         'projects/p1/locations/us-central1/cachedContents/c1')
        self.assertFalse(any('ya29' in line for line in logs.output))       # never logged

    def test_token_is_cached_and_refreshed_in_time(self) -> None:
        command = f'"{sys.executable}" -c "import time; print(\'tok-%d\' % (time.time() * 1000))"'
        with mock.patch.dict(os.environ, {'SIGIL_GOOGLE_TOKEN_COMMAND': command}):
            first = google_auth.access_token('fallback')
            self.assertEqual(google_auth.access_token('fallback'), first)   # cached
            with mock.patch('time.monotonic', return_value=time.monotonic() + 2800):
                self.assertNotEqual(google_auth.access_token('fallback'), first)  # 45 minutes: new token
        with mock.patch.dict(os.environ, {'SIGIL_GOOGLE_TOKEN_COMMAND': f'"{sys.executable}" -c "raise SystemExit(3)"'}):
            google_auth.reset()
            self.assertEqual(google_auth.access_token('fallback'), 'fallback')

    def test_endpoint_parsing_is_strict(self) -> None:
        self.assertIsNone(explicit_cache.endpoint_for('http://generativelanguage.googleapis.com/v1beta/openai', 'm'))
        self.assertIsNone(explicit_cache.endpoint_for(
            'https://us-central1-aiplatform.googleapis.com/v1/projects/a..b%2F/locations/x/endpoints/openapi', 'm'))
        self.assertIsNone(explicit_cache.endpoint_for('https://evil-generativelanguage.googleapis.com.io/v1beta', 'm'))
        self.assertIsNone(explicit_cache.endpoint_for(GEMINI, '../../x'))
        with self.assertRaises(explicit_cache.ExplicitCacheError):
            explicit_cache._resource_url(explicit_cache.endpoint_for(GEMINI, 'gemini-2.5-flash'), '../admin')


if __name__ == '__main__':
    unittest.main()
