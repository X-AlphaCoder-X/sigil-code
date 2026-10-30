from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace

from src.agent_tools import _mcp_list_tools


class _FakeRuntime:
    servers = ('demo',)
    resources = ()

    def list_tools(self, query=None, server_name=None, limit=50):
        return [SimpleNamespace(name='echo', server_name='demo', description='')]


class McpListToolsQuietTests(unittest.TestCase):
    def test_list_tools_writes_nothing_to_stderr(self) -> None:
        """Regression: leftover DEBUG prints leaked into the user's terminal."""
        context = SimpleNamespace(mcp_runtime=_FakeRuntime())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = _mcp_list_tools({}, context)
        self.assertEqual(result, 'echo ; server=demo')
        self.assertEqual(err.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
