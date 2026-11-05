"""Explicit cache storage kept across resume, the start-up crash sweep, the Nova cache limit, dated
price rows and Nova cache-write prices. Offline: fake model clients and a fake Google server."""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from src import explicit_cache, prompt_cache
from src.agent_types import AssistantTurn, ModelConfig, UsageStats

GEMINI = 'https://generativelanguage.googleapis.com/v1beta/openai'
GEMINI_ROOT = 'https://generativelanguage.googleapis.com/v1beta'
VERTEX_ROOT = 'https://us-central1-aiplatform.googleapis.com/v1'
BEDROCK = 'https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1'
NOVA = 'us.amazon.nova-pro-v1:0'
CLAUDE = 'us.anthropic.claude-sonnet-5-v1:0'
DEAD_PID = 2 ** 22 + 4321
_ENV = {'SIGIL_PROMPT_CACHE': '', 'SIGIL_PROMPT_CACHE_TTL': '', 'SIGIL_EXPLICIT_CACHE': '',
        'SIGIL_GOOGLE_TOKEN_COMMAND': '', 'SIGIL_BEDROCK_CONVERSE': '', 'SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT': '',
        'SIGIL_API_KEY_GENERATIVELANGUAGE_GOOGLEAPIS_COM': ''}


class _Stub:
    """A model client that answers 'done' (or raises `error`)."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.config = None

    def complete(self, messages, tools, output_schema=None):  # noqa: ANN001, ANN201
        if self.error is not None:
            raise self.error
        return AssistantTurn(content='done', tool_calls=(), finish_reason='stop',
                             raw_message={'role': 'assistant', 'content': 'done'},
                             usage=UsageStats(input_tokens=3, output_tokens=2))


class _HomeBase(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        env = mock.patch.dict(os.environ, {**_ENV, 'HOME': self.home.name, 'USERPROFILE': self.home.name})
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        explicit_cache.reset_for_tests()
        self.addCleanup(explicit_cache.reset_for_tests)
        self.ws = Path(self.home.name) / 'ws'
        self.ws.mkdir()

    def agent(self, *, subagent: bool = False, base: str = 'http://127.0.0.1:9/v1', key: str = 'local-token'):  # noqa: ANN201
        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig

        agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url=base, api_key=key),
                                 runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s',
                                                                   nudge_strategy='off'),
                                 is_subagent=subagent)
        agent.client = _Stub()
        return agent


class StorageAcrossResumeTests(_HomeBase):
    """Explicit cache storage is saved in the session file and restored on resume, never twice."""

    def stored(self, session_id: str):  # noqa: ANN201
        from src.session_store import load_agent_session

        return load_agent_session(session_id, directory=self.ws / '.s')

    def test_saved_restored_and_not_double_counted(self) -> None:
        explicit_cache._SPENT[0] = 0.4                     # storage this process spent on caches
        first = self.agent()
        session_id = first.run('hi').session_id
        self.assertAlmostEqual(self.stored(session_id).cache_storage_usd, 0.4)
        explicit_cache.reset_for_tests()                  # a new process (-c, -r, agent-resume, /resume)
        second = self.agent()
        second.resume('again', self.stored(session_id))
        self.assertAlmostEqual(explicit_cache.storage_cost_usd(), 0.4)       # restored
        self.assertAlmostEqual(self.stored(session_id).cache_storage_usd, 0.4)  # saved once, not 0.8
        explicit_cache._SPENT[0] += 0.1                   # more storage in this process
        second.resume('third', self.stored(session_id))   # the chat loop resumes the same session again
        self.assertAlmostEqual(explicit_cache.storage_cost_usd(), 0.5)
        self.assertAlmostEqual(self.stored(session_id).cache_storage_usd, 0.5)
        from src.status_line import status_payload

        self.assertAlmostEqual(status_payload(second)['cost']['cache_storage_usd'], 0.5)

    def test_old_session_files_load_and_subagents_save_none(self) -> None:
        from src.session_store import load_agent_session

        folder = self.ws / '.s'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'old1.json').write_text(json.dumps({'session_id': 'old1', 'messages': [], 'usage': {}}),
                                          encoding='utf-8')
        self.assertEqual(load_agent_session('old1', directory=folder).cache_storage_usd, 0.0)
        explicit_cache._SPENT[0] = 0.3
        child = self.agent(subagent=True)
        child_id = child.run('child task').session_id
        self.assertEqual(self.stored(child_id).cache_storage_usd, 0.0)  # the parent's session holds it

    def test_clear_starts_a_new_count(self) -> None:
        explicit_cache._SPENT[0] = 0.25
        agent = self.agent()
        agent.run('hi')
        agent.clear_runtime_state()
        self.assertAlmostEqual(explicit_cache.storage_cost_usd(), 0.0)
        explicit_cache._SPENT[0] += 0.05
        self.assertAlmostEqual(explicit_cache.storage_cost_usd(), 0.05)

    def test_interrupted_run_keeps_side_totals_and_storage(self) -> None:
        explicit_cache._SPENT[0] = 0.2
        agent = self.agent()
        agent.record_side_usage(UsageStats(input_tokens=5), cost_usd=0.07)
        agent.client = _Stub(KeyboardInterrupt())
        result = agent.run('hi')
        self.assertEqual(result.stop_reason, 'interrupted')
        stored = self.stored(result.session_id)
        self.assertAlmostEqual(stored.side_cost_usd, 0.07)
        self.assertAlmostEqual(stored.cache_storage_usd, 0.2)


class _FakeGoogle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, req, timeout=None):  # noqa: ANN001
        with self.lock:
            self.calls.append((req.get_method(), req.full_url))

        class Resp:
            def __init__(self) -> None:
                self.body = json.dumps({'cachedContents': []}).encode()

            def read(self, n: int = -1) -> bytes:
                return self.body

            def __enter__(self):  # noqa: ANN204
                return self

            def __exit__(self, *a) -> None:  # noqa: ANN002
                return None
        return Resp()


class StartupSweepTests(_HomeBase):
    """Crash leftovers of this install are deleted at start-up, in the background, for every endpoint
    with a usable key; other installs' caches are never touched."""

    def write_registry(self, entries: dict) -> Path:
        path = Path(self.home.name) / '.sigil' / 'explicit-caches.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding='utf-8')
        return path

    def entry(self, name: str, root: str, parent: str = '', kind: str = 'gemini', install: str | None = None) -> dict:
        install = install or explicit_cache._install_id()
        return {'name': name, 'version_root': root, 'parent': parent, 'kind': kind, 'model': 'gemini-2.5-flash',
                'display_name': f'sigil-{install}-{DEAD_PID}-abcdef', 'pid': DEAD_PID}

    def test_start_up_deletes_this_installs_leftovers_only(self) -> None:
        registry = self.write_registry({
            'cachedContents/mine1': self.entry('cachedContents/mine1', GEMINI_ROOT),
            'cachedContents/other1': self.entry('cachedContents/other1', GEMINI_ROOT, install='000000000000'),
            'projects/p1/locations/us-central1/cachedContents/v1': self.entry(
                'projects/p1/locations/us-central1/cachedContents/v1', VERTEX_ROOT,
                'projects/p1/locations/us-central1', 'vertex')})
        google = _FakeGoogle()
        with mock.patch('urllib.request.urlopen', side_effect=google), \
                self.assertLogs(explicit_cache.get_logger('explicit_cache').name, level='INFO') as logs:
            self.agent(base=GEMINI, key='AIza-k')        # start-up of a top-level agent
            self.assertTrue(explicit_cache.wait_for_start_sweep(10))
        deletes = [url.rsplit('/', 1)[-1] for method, url in google.calls if method == 'DELETE']
        self.assertEqual(deletes, ['mine1'])             # not other1 (another install), not v1 (no token)
        left = json.loads(registry.read_text(encoding='utf-8'))
        self.assertEqual(sorted(left), ['cachedContents/other1', 'projects/p1/locations/us-central1/cachedContents/v1'])
        self.assertTrue(any('no usable key' in line for line in logs.output))
        # the first-use clean-up still runs for the endpoint that was skipped
        vertex = explicit_cache.endpoint_for(VERTEX_ROOT + '/projects/p1/locations/us-central1/endpoints/openapi', 'm')
        self.assertNotIn(vertex.id, explicit_cache._SWEPT)

    def test_time_limit_and_tampered_hosts(self) -> None:
        self.write_registry({
            'cachedContents/a': self.entry('cachedContents/a', GEMINI_ROOT),
            'cachedContents/evil': self.entry('cachedContents/evil', 'https://evil.example.com/v1beta')})
        google = _FakeGoogle()
        with mock.patch('urllib.request.urlopen', side_effect=google), \
                self.assertLogs(explicit_cache.get_logger('explicit_cache').name, level='WARNING') as logs:
            explicit_cache.sweep_at_start(lambda base: 'AIza-k', time_limit=0.0, wait=True)
        self.assertEqual(google.calls, [])                # the time limit was reached before any call
        self.assertTrue(any('time limit' in line for line in logs.output))
        explicit_cache.reset_for_tests()
        with mock.patch('urllib.request.urlopen', side_effect=google):
            explicit_cache.sweep_at_start(lambda base: 'AIza-k', wait=True)
        self.assertTrue(all('evil.example.com' not in url for _m, url in google.calls))  # never another host

    def test_subagents_do_not_start_it(self) -> None:
        self.write_registry({'cachedContents/mine1': self.entry('cachedContents/mine1', GEMINI_ROOT)})
        google = _FakeGoogle()
        with mock.patch('urllib.request.urlopen', side_effect=google):
            self.agent(subagent=True, base=GEMINI, key='AIza-k')
            time.sleep(0.2)
        self.assertEqual(google.calls, [])


class NovaCacheLimitTests(unittest.TestCase):
    """Amazon Nova caches at most 20K tokens (Bedrock prompt-caching page): no cachePoint past it."""

    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, _ENV)
        env.start()
        self.addCleanup(env.stop)
        settings = mock.patch('src.prompt_cache._settings_value', return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        prompt_cache.set_session(clear=True)
        self.addCleanup(prompt_cache.set_session, clear=True)
        from src import bedrock_client

        bedrock_client._LIMIT_NOTED.clear()

    def payload(self, model: str, turns: int, words: int):  # noqa: ANN201
        from src.bedrock_client import BedrockConverseClient

        history = [{'role': 'system', 'content': 'rules ' * 2000}]
        for turn in range(turns):
            history.append({'role': 'user', 'content': f'q{turn} ' + 'word ' * words})
            history.append({'role': 'assistant', 'content': f'a{turn} ' + 'word ' * words})
        history.append({'role': 'user', 'content': 'last question'})
        return BedrockConverseClient(ModelConfig(model=model, base_url=BEDROCK)).build_payload(history, [])

    def prefix_tokens(self, payload: dict) -> list[int]:
        from src.bedrock_client import estimate_tokens

        found, total = [], 0
        for block in payload.get('system') or []:
            if 'cachePoint' in block:
                found.append(total)
            else:
                total += estimate_tokens(block)
        for message in payload['messages']:
            for block in message['content']:
                if 'cachePoint' in block:
                    found.append(total)
                else:
                    total += estimate_tokens(block)
        return found

    def test_points_stay_under_the_limit_and_move_back(self) -> None:
        from src.bedrock_client import NOVA_CACHE_MAX_TOKENS

        from src.debug_log import get_logger

        with self.assertLogs(get_logger('bedrock').name, level='INFO') as logs:
            payload = self.payload(NOVA, turns=6, words=3000)
            self.payload(NOVA, turns=6, words=3000)
        points = self.prefix_tokens(payload)
        self.assertTrue(points, 'some cache point must remain')
        self.assertTrue(all(p <= NOVA_CACHE_MAX_TOKENS for p in points), points)
        self.assertGreater(max(points), NOVA_CACHE_MAX_TOKENS // 2)  # moved back, not just dropped
        self.assertEqual(sum('20,000' in line for line in logs.output), 1)  # logged once

    def test_claude_is_not_limited(self) -> None:
        payload = self.payload(CLAUDE, turns=6, words=3000)
        self.assertGreater(max(self.prefix_tokens(payload)), 20000)


class DatedPriceTests(unittest.TestCase):
    """Price rows with a change date (aistudio.google.com/docs/pricing: 3.6 / 3.8 Flash from 2027-01-01)."""

    def test_rows_follow_the_date(self) -> None:
        import datetime

        endpoint = explicit_cache.endpoint_for(GEMINI, 'gemini-3.8-flash')
        before = datetime.datetime(2026, 12, 31, 23, 0, tzinfo=datetime.timezone.utc).timestamp()
        after = datetime.datetime(2027, 1, 1, 0, 30, tzinfo=datetime.timezone.utc).timestamp()
        self.assertEqual(explicit_cache.prices_for(endpoint, when=before), (0.75, 0.075, 0.50))
        self.assertEqual(explicit_cache.prices_for(endpoint, when=after), (1.50, 0.15, 1.00))
        self.assertEqual(explicit_cache.prices_for(explicit_cache.endpoint_for(GEMINI, 'gemini-3.6-flash'), when=after),
                         (1.50, 0.15, 1.00))
        self.assertEqual(explicit_cache.prices_for(explicit_cache.endpoint_for(GEMINI, 'gemini-2.5-flash'), when=after),
                         (0.30, 0.03, 1.00))


class NovaWritePriceTests(unittest.TestCase):
    """AWS Price List (AmazonBedrock, us-east-1): Nova cache writes are $0.00 per 1K tokens."""

    def test_writes_cost_nothing(self) -> None:
        from src.agent_types import ModelPricing
        from src.main import _cache_pricing

        with mock.patch.dict(os.environ, _ENV), mock.patch('src.prompt_cache._settings_value', return_value={}), \
                mock.patch('src.prompt_cache.known_prices', return_value={}):
            pricing = _cache_pricing(ModelPricing(input_cost_per_million_tokens_usd=0.8), Namespace(model=NOVA), BEDROCK)
        cost = pricing.estimate_cost_usd(UsageStats(cache_creation_input_tokens=1_000_000,
                                                    cache_read_input_tokens=1_000_000))
        self.assertAlmostEqual(cost, 0.20, places=4)  # reads 0.25 x 0.80; writes free


if __name__ == '__main__':
    unittest.main()
