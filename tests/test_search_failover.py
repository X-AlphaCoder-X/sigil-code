from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from src import search_runtime
from src.search_runtime import SearchProviderProfile, SearchUsageTracker, _search_tavily


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _provider() -> SearchProviderProfile:
    return SearchProviderProfile(
        name='tavily', provider='tavily', source_manifest='test',
        base_url='https://api.tavily.com/search', api_key_env='TAVILY_API_KEY',
    )


class TavilyFailoverTests(unittest.TestCase):
    def test_rejected_key_fails_over_to_second_key(self) -> None:
        keys_used: list[str] = []

        def fake_urlopen(req, timeout=None):
            key = json.loads(req.data)['api_key']
            keys_used.append(key)
            if key == 'k1':
                raise HTTPError(req.full_url, 432, 'plan limit', {}, None)
            body = {'results': [{'url': 'https://a.test/x', 'title': 'A', 'content': 'snip'}]}
            return _Resp(json.dumps(body).encode())

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'TAVILY_API_KEY': 'k1', 'TAVILY_API_KEY_2': 'k2'}), \
                patch.object(search_runtime.request, 'urlopen', side_effect=fake_urlopen):
            tracker = SearchUsageTracker(state_path=Path(tmp) / 'state.json')
            results = _search_tavily(
                _provider(), 'q', max_results=3, domains=(), timeout_seconds=5,
                usage_tracker=tracker, blocked_domains=('bad.test',),
            )
            self.assertEqual(keys_used, ['k1', 'k2'])
            self.assertEqual(results[0].url, 'https://a.test/x')
            self.assertTrue(tracker.is_key_exhausted('TAVILY_API_KEY', 1000))
            self.assertEqual(tracker.get_usage().get('TAVILY_API_KEY_2'), 1)

    def test_non_quota_error_is_raised(self) -> None:
        def fake_urlopen(req, timeout=None):
            raise HTTPError(req.full_url, 500, 'server error', {}, None)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'TAVILY_API_KEY': 'k1'}), \
                patch.object(search_runtime.request, 'urlopen', side_effect=fake_urlopen):
            tracker = SearchUsageTracker(state_path=Path(tmp) / 'state.json')
            with self.assertRaises(LookupError):
                _search_tavily(_provider(), 'q', max_results=3, domains=(), timeout_seconds=5, usage_tracker=tracker)


if __name__ == '__main__':
    unittest.main()
