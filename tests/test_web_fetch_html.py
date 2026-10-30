from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import agent_tools
from src.agent_tools import _html_to_markdown, _web_fetch


class HtmlToMarkdownTests(unittest.TestCase):
    def test_strips_markup_keeps_structure(self) -> None:
        html = (
            '<html><head><title>t</title><style>.a{color:red}</style></head><body>'
            '<script>var x=1;</script><h1>Title</h1><p>Hello <b>world</b>, see '
            '<a href="https://example.com/x">the docs</a>.</p>'
            '<ul><li>one</li><li>two</li></ul><ol><li>first</li><li>second</li></ol>'
            '<pre>a  b\n c</pre></body></html>'
        )
        out = _html_to_markdown(html)
        self.assertIn('# Title', out)
        self.assertIn('Hello **world**, see [the docs](https://example.com/x).', out)
        self.assertIn('- one', out)
        self.assertIn('- two', out)
        self.assertIn('1. first', out)
        self.assertIn('2. second', out)
        self.assertIn('a  b\n c', out)
        self.assertNotIn('color:red', out)
        self.assertNotIn('var x', out)
        self.assertNotIn('<', out)

    def test_relative_links_and_chrome_removed(self) -> None:
        body = '<p>' + 'Real content. ' * 40 + '<a href="/guide">guide</a></p>'
        html = (
            '<html><head><title>Page</title></head><body>'
            '<nav><a href="/menu">MENU_ITEM</a></nav>'
            '<div role="navigation">SIDEBAR</div>'
            f'<main>{body}</main><footer>FOOTER_TEXT</footer></body></html>'
        )
        out = _html_to_markdown(html, base_url='https://site.test/docs/page')
        self.assertIn('[guide](https://site.test/guide)', out)
        self.assertIn('# Page', out)
        for junk in ('MENU_ITEM', 'SIDEBAR', 'FOOTER_TEXT'):
            self.assertNotIn(junk, out)


class WebFetchBehaviourTests(unittest.TestCase):
    def setUp(self) -> None:
        agent_tools._WEB_FETCH_CACHE.clear()

    def _ctx(self, model_config=None):
        return SimpleNamespace(max_output_chars=50_000, command_timeout_seconds=5, model_config=model_config)

    def test_http_upgraded_and_cached(self) -> None:
        calls = []

        def fake_raw(url, timeout):
            calls.append(url)
            return {'text': 'hello page', 'content_type': 'text/plain', 'final_url': url}

        with patch.object(agent_tools, '_web_fetch_raw', side_effect=fake_raw):
            out1, _ = _web_fetch({'url': 'http://example.com/a'}, self._ctx())
            out2, _ = _web_fetch({'url': 'https://example.com/a'}, self._ctx())
        self.assertEqual(calls, ['https://example.com/a'])
        self.assertEqual(out1, 'hello page')
        self.assertEqual(out2, 'hello page')

    def test_cross_host_redirect_is_reported(self) -> None:
        with patch.object(agent_tools, '_web_fetch_raw', return_value={'redirect_to': 'https://other.test/x'}):
            out, meta = _web_fetch({'url': 'https://example.com/a'}, self._ctx())
        self.assertIn('https://other.test/x', out)
        self.assertEqual(meta['redirect_to'], 'https://other.test/x')

    def test_prompt_returns_model_answer(self) -> None:
        page = {'text': 'The version is 4.2.1', 'content_type': 'text/plain', 'final_url': 'https://e.test'}
        with patch.object(agent_tools, '_web_fetch_raw', return_value=page), \
                patch('src.openai_compat.OpenAICompatClient') as client_cls:
            client_cls.return_value.complete.return_value = SimpleNamespace(content='Version 4.2.1')
            out, meta = _web_fetch(
                {'url': 'https://e.test', 'prompt': 'What version?'}, self._ctx(model_config=object())
            )
        self.assertTrue(meta['prompt_applied'])
        self.assertIn('Version 4.2.1', out)
        sent = client_cls.return_value.complete.call_args[0][0][0]['content']
        self.assertIn('The version is 4.2.1', sent)
        self.assertIn('What version?', sent)


if __name__ == '__main__':
    unittest.main()
