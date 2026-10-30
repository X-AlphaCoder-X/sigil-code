"""Batch 5 (P1): native Anthropic client, MCP over HTTP/SSE, mcp__server__tool tools."""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent, native_mcp_tool_name
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.anthropic_client import AnthropicClient, convert_messages, make_client
from src.mcp_runtime import MCPRuntime, connection_manager
from src.openai_compat import OpenAICompatClient


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def recording_server(responses, recorded):
    queued = [FakeHTTPResponse(r) for r in responses]

    def _fake(request_obj, timeout=None):  # noqa: ANN001
        recorded.append({'url': request_obj.full_url, 'headers': dict(request_obj.header_items()),
                         'body': json.loads(request_obj.data.decode('utf-8'))})
        return queued.pop(0)

    return _fake


ANTHROPIC = 'https://api.anthropic.com/v1'


class AnthropicClientTests(unittest.TestCase):
    def test_message_conversion(self) -> None:
        messages = [
            {'role': 'system', 'content': 'You are helpful.'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'look'},
                                         {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJD'}}]},
            {'role': 'assistant', 'content': 'Reading.', 'tool_calls': [
                {'id': 't1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "a"}'}}]},
            {'role': 'tool', 'tool_call_id': 't1', 'content': 'file text'},
            {'role': 'system', 'content': 'Remember the plan.'},
        ]
        system, converted = convert_messages(messages)
        self.assertEqual(system, [{'type': 'text', 'text': 'You are helpful.'}])
        self.assertEqual([m['role'] for m in converted], ['user', 'assistant', 'user'])
        self.assertEqual(converted[0]['content'][1]['type'], 'image')
        self.assertEqual(converted[1]['content'][1], {'type': 'tool_use', 'id': 't1', 'name': 'read_file',
                                                      'input': {'path': 'a'}})
        last = converted[2]['content']
        self.assertEqual(last[0]['type'], 'tool_result')
        self.assertIn('<system-reminder>', last[1]['text'])

    def test_payload_caching_temperature_and_thinking(self) -> None:
        tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'd',
                                                   'parameters': {'type': 'object', 'properties': {}}}}]
        messages = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'hi'}]
        with patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': ''}):
            plain = AnthropicClient(ModelConfig(model='claude-x', base_url=ANTHROPIC)).build_payload(messages, tools)
            # Claude 4.5 and earlier: manual budget (Anthropic API docs, extended thinking)
            thinking = AnthropicClient(ModelConfig(model='claude-sonnet-4-5', base_url=ANTHROPIC, effort='high',
                                                   temperature=0.2)).build_payload(messages, tools)
            warm = AnthropicClient(ModelConfig(model='claude-sonnet-4-5', base_url=ANTHROPIC,
                                               temperature=1.0)).build_payload(messages, tools)
            # Claude 4.7+ / 5.x: adaptive thinking + output_config.effort, no temperature below 1
            modern = AnthropicClient(ModelConfig(model='claude-opus-5-5', base_url=ANTHROPIC, effort='high',
                                                 temperature=0.2)).build_payload(messages, tools)
            modern_plain = AnthropicClient(ModelConfig(model='claude-sonnet-5', base_url=ANTHROPIC,
                                                       temperature=0.2)).build_payload(messages, tools)
        self.assertEqual(modern['thinking'], {'type': 'adaptive'})
        self.assertEqual(modern['output_config'], {'effort': 'high'})
        self.assertNotIn('temperature', modern)
        self.assertNotIn('temperature', modern_plain)
        self.assertNotIn('thinking', modern_plain)
        self.assertNotIn('temperature', plain)
        self.assertEqual(plain['system'][-1]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual(plain['tools'][-1]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual(plain['messages'][-1]['content'][-1]['cache_control'], {'type': 'ephemeral'})
        self.assertEqual(thinking['thinking'], {'type': 'enabled', 'budget_tokens': 16000})
        self.assertNotIn('temperature', thinking)
        self.assertGreater(thinking['max_tokens'], 16000)
        self.assertEqual(warm['temperature'], 1.0)

    def test_selection_is_opt_in(self) -> None:
        config = ModelConfig(model='claude-x', base_url=ANTHROPIC)
        with patch.dict(os.environ, {'SIGIL_ANTHROPIC_NATIVE': ''}):
            self.assertIsInstance(make_client(config), OpenAICompatClient)
        with patch.dict(os.environ, {'SIGIL_ANTHROPIC_NATIVE': '1'}):
            self.assertIsInstance(make_client(config), AnthropicClient)
            self.assertIsInstance(make_client(ModelConfig(model='m')), OpenAICompatClient)

    def test_agent_tool_loop_over_messages_api(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            {'content': [{'type': 'thinking', 'thinking': 'private'}, {'type': 'text', 'text': 'Let me look.'},
                         {'type': 'tool_use', 'id': 'tu1', 'name': 'read_file', 'input': {'path': 'hello.txt'}}],
             'stop_reason': 'tool_use',
             'usage': {'input_tokens': 100, 'output_tokens': 10, 'cache_read_input_tokens': 80,
                       'cache_creation_input_tokens': 20}},
            {'content': [{'type': 'text', 'text': 'It says hello.'}], 'stop_reason': 'end_turn',
             'usage': {'input_tokens': 120, 'output_tokens': 5}},
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_ANTHROPIC_NATIVE': '1', 'SIGIL_EFFORT_LEVEL': ''}), \
                patch('src.openai_compat.request.urlopen', side_effect=recording_server(responses, recorded)):
            ws = Path(tmp)
            (ws / 'hello.txt').write_text('hello world', encoding='utf-8')
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='claude-x', base_url=ANTHROPIC, api_key='sk-ant-test'),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.port_sessions' / 'agent',
                                                  scratchpad_root=ws / '.port_sessions' / 'scratchpad',
                                                  nudge_strategy='off'),
            )
            result = agent.run('read hello.txt')
        self.assertEqual(result.final_output, 'It says hello.')
        self.assertTrue(recorded[0]['url'].endswith('/v1/messages'))
        headers = {k.lower(): v for k, v in recorded[0]['headers'].items()}
        self.assertEqual(headers['x-api-key'], 'sk-ant-test')
        self.assertEqual(headers['anthropic-version'], '2023-06-01')
        second = recorded[1]['body']['messages']
        self.assertEqual(second[-1]['content'][0]['type'], 'tool_result')
        self.assertIn('hello world', second[-1]['content'][0]['content'])
        # Thinking blocks go back unchanged with the tool results (Anthropic API docs, thinking
        # with tool use: "Required: within a tool-use turn, pass thinking blocks back").
        assistant = next(m for m in second if m['role'] == 'assistant')
        self.assertEqual(assistant['content'][0], {'type': 'thinking', 'thinking': 'private'})
        self.assertEqual([b['type'] for b in assistant['content']], ['thinking', 'text', 'tool_use'])
        self.assertEqual(result.usage.cache_read_input_tokens, 80)


# ---------------------------------------------------------------- MCP over HTTP

TOOLS = [{'name': 'shout', 'description': 'Upper-case text', 'inputSchema': {'type': 'object'}}]


def _reply(message: dict) -> dict:
    method = message.get('method')
    if method == 'initialize':
        result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'h'}}
    elif method == 'tools/list':
        result = {'tools': TOOLS}
    elif method == 'tools/call':
        text = message['params']['arguments'].get('text', '')
        result = {'content': [{'type': 'text', 'text': text.upper()}], 'isError': False}
    else:
        result = {}
    return {'jsonrpc': '2.0', 'id': message.get('id'), 'result': result}


class _MCPHandler(BaseHTTPRequestHandler):
    sse_queues: list = []
    seen_headers: list = []

    def log_message(self, *args) -> None:  # quiet
        return

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
        type(self).seen_headers.append({'token': self.headers.get('X-Token'),
                                        'session': self.headers.get('Mcp-Session-Id')})
        if self.path.startswith('/messages'):  # legacy SSE transport
            self.send_response(202)
            self.end_headers()
            if 'id' in body:
                type(self).sse_queues[-1].put(_reply(body))
            return
        if 'id' not in body:
            self.send_response(202)
            self.end_headers()
            return
        reply = json.dumps(_reply(body))
        self.send_response(200)
        self.send_header('Mcp-Session-Id', 'session-123')
        if body.get('method') == 'tools/call':  # answer as an event stream
            payload = f'event: message\ndata: {reply}\n\n'.encode('utf-8')
            self.send_header('Content-Type', 'text/event-stream')
        else:
            payload = reply.encode('utf-8')
            self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_DELETE(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        outbox: queue.Queue = queue.Queue()
        type(self).sse_queues.append(outbox)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        self.wfile.write(b'event: endpoint\ndata: /messages?session=1\n\n')
        self.wfile.flush()
        while True:
            try:
                message = outbox.get(timeout=5)
            except queue.Empty:
                return
            self.wfile.write(f'event: message\ndata: {json.dumps(message)}\n\n'.encode('utf-8'))
            self.wfile.flush()


class MCPHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        _MCPHandler.sse_queues = []
        _MCPHandler.seen_headers = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), _MCPHandler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]
        self.addCleanup(self.server.shutdown)
        self.addCleanup(connection_manager().close_all)

    def _runtime(self, tmp: str, entry: dict) -> MCPRuntime:
        (Path(tmp) / '.sigil-mcp.json').write_text(json.dumps({'mcpServers': {'web': entry}}), encoding='utf-8')
        return MCPRuntime.from_workspace(Path(tmp))

    def test_streamable_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'TEST_MCP_TOKEN': 'tok-1'}):
            runtime = self._runtime(tmp, {'type': 'http', 'url': f'http://127.0.0.1:{self.port}/mcp',
                                          'headers': {'X-Token': '${TEST_MCP_TOKEN}'}})
            self.assertEqual(runtime.servers[0].transport, 'http')
            tools = runtime.list_tools()
            text, meta = runtime.call_tool('shout', arguments={'text': 'hi'})
            connection_manager().close_all()
        self.assertEqual([t.name for t in tools], ['shout'])
        self.assertIn('HI', text)
        self.assertEqual(meta.get('server_name'), 'web')
        self.assertTrue(all(h['token'] == 'tok-1' for h in _MCPHandler.seen_headers))
        self.assertEqual(_MCPHandler.seen_headers[-1]['session'], 'session-123')

    def test_legacy_sse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp, {'type': 'sse', 'url': f'http://127.0.0.1:{self.port}/sse'})
            text, _ = runtime.call_tool('shout', arguments={'text': 'quiet'})
            connection_manager().close_all()
        self.assertIn('QUIET', text)


class NativeMCPToolTests(unittest.TestCase):
    def tearDown(self) -> None:
        connection_manager().close_all()

    def test_mcp_tools_become_real_tools(self) -> None:
        from tests.test_mcp_connections import make_workspace

        recorded: list[dict[str, object]] = []
        responses = [
            {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'c1', 'type': 'function',
                'function': {'name': 'mcp__fake__echo', 'arguments': '{}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [{'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}]},
        ]

        def _fake(request_obj, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(request_obj.data.decode('utf-8')))
            return FakeHTTPResponse(responses.pop(0))

        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=_fake):
            ws, _ = make_workspace(tmp)
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m'),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.port_sessions' / 'agent',
                                                  scratchpad_root=ws / '.port_sessions' / 'scratchpad',
                                                  nudge_strategy='off'),
            )
            self.assertIn('mcp__fake__echo', agent.tool_registry)
            agent.run('echo please')
            connection_manager().close_all()
        names = {t['function']['name'] for t in recorded[0]['tools']}
        self.assertIn('mcp__fake__echo', names)
        tool_text = [str(m.get('content')) for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any('pid:' in t for t in tool_text), tool_text)

    def test_name_and_switch(self) -> None:
        self.assertEqual(native_mcp_tool_name('my server', 'do.it'), 'mcp__my_server__do_it')
        self.assertLessEqual(len(native_mcp_tool_name('s' * 50, 't' * 50)), 64)


if __name__ == '__main__':
    unittest.main()
