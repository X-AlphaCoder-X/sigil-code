"""MCP connection manager: one server process per session, restart after a crash,
background start, wait_for_mcp_servers."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.agent_types import AgentRuntimeConfig
from src.mcp_runtime import MCPRuntime, connection_manager

SERVER = '''
import json, os, sys, time
with open(sys.argv[1], "a") as f:
    f.write(str(os.getpid()) + "\\n")
if len(sys.argv) > 2:
    time.sleep(float(sys.argv[2]))
TOOLS = [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}]
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    msg = json.loads(raw)
    method = msg.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "f", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        if msg["params"]["arguments"].get("crash"):
            sys.exit(3)
        result = {"content": [{"type": "text", "text": "pid:" + str(os.getpid())}], "isError": False}
    elif method == "resources/list":
        result = {"resources": []}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32601, "message": "nope"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg.get("id"), "result": result}), flush=True)
'''


def make_workspace(tmp: str, *, delay: float | None = None) -> tuple[Path, Path]:
    ws = Path(tmp)
    script = ws / 'server.py'
    script.write_text(SERVER, encoding='utf-8')
    starts = ws / 'starts.txt'
    args = ['-u', str(script), str(starts)] + ([str(delay)] if delay is not None else [])
    (ws / '.sigil-mcp.json').write_text(
        json.dumps({'mcpServers': {'fake': {'command': sys.executable, 'args': args}}}), encoding='utf-8')
    return ws, starts


def start_count(starts: Path) -> int:
    return len(starts.read_text(encoding='utf-8').split()) if starts.exists() else 0


class MCPConnectionTests(unittest.TestCase):
    def tearDown(self) -> None:
        connection_manager().close_all()

    def test_one_process_for_many_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'}):
            ws, starts = make_workspace(tmp)
            runtime = MCPRuntime.from_workspace(ws)
            runtime.list_tools()
            first, _ = runtime.call_tool('echo', arguments={})
            second, _ = runtime.call_tool('echo', arguments={})
            self.assertEqual(start_count(starts), 1)
            self.assertEqual(first, second)  # same pid answered both
            self.assertIn('connected', runtime.render_summary())
            connection_manager().close_all()

    def test_per_request_mode_still_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '0'}):
            ws, starts = make_workspace(tmp)
            runtime = MCPRuntime.from_workspace(ws)
            runtime.call_tool('echo', arguments={})
            runtime.call_tool('echo', arguments={})
            # Every request starts its own process (tool lookup + two calls).
            self.assertGreaterEqual(start_count(starts), 2)

    def test_crashed_server_restarts_on_next_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'}):
            ws, starts = make_workspace(tmp)
            runtime = MCPRuntime.from_workspace(ws)
            runtime.call_tool('echo', arguments={})
            with self.assertRaises(Exception):
                runtime.call_tool('echo', arguments={'crash': True})  # not retried: may have side effects
            self.assertEqual(start_count(starts), 1)
            text, _ = runtime.call_tool('echo', arguments={})
            self.assertIn('pid:', text)
            self.assertEqual(start_count(starts), 2)
            connection_manager().close_all()

    def test_background_start_and_wait_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'}):
            ws, starts = make_workspace(tmp, delay=1.0)
            config = AgentRuntimeConfig(cwd=ws)
            runtime = MCPRuntime.from_workspace(ws)
            context = build_tool_context(config, mcp_runtime=runtime)
            runtime.start_background()
            self.assertIn(connection_manager().status.get('fake'), ('connecting', 'connected'))
            started = time.monotonic()
            result = execute_tool(default_tool_registry(), 'wait_for_mcp_servers', {'timeout_seconds': 30}, context)
            self.assertTrue(result.ok, result.content)
            self.assertIn('fake: connected', result.content)
            self.assertLess(time.monotonic() - started, 30)
            runtime.call_tool('echo', arguments={})
            self.assertEqual(start_count(starts), 1)  # the background process was reused
            connection_manager().close_all()


if __name__ == '__main__':
    unittest.main()
