"""/mcp disable and /mcp enable change the tools sent with the next model call, with no restart. A refresh
that changes nothing sends the same tool list byte for byte (the start of every cached prompt on every
provider), and the tools that stay keep their place, so only a real change starts a new cache."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.mcp_runtime import MCPRuntime, McpToolRecord

BASE = 'http://127.0.0.1:8000/v1'
TOOLS = {'alpha': ('find', 'open'), 'beta': ('query', 'update'), 'gamma': ('solo',)}


def _fake_remote_tools(self, server_name=None):  # noqa: ANN001
    """The tools of every configured (not disabled) server, answered in a scrambled order."""
    records = [McpToolRecord(name=tool, server_name=server.name, source_manifest='.mcp.json',
                             description=f'{tool} on {server.name}',
                             input_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}})
               for server in self.servers for tool in TOOLS.get(server.name, ())]
    return tuple(reversed(records))


class _Resp:
    def __init__(self) -> None:
        self.body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                                             'finish_reason': 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class McpToggleToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {
            name: {'command': 'no-such-mcp-command'} for name in TOOLS}}), encoding='utf-8')
        for patcher in (patch.object(MCPRuntime, '_list_remote_tools', _fake_remote_tools),
                        patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1'})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.requests: list[dict] = []

    def _agent(self, **config) -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url=BASE),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, permissions=AgentPermissions(),
                                              session_directory=self.ws / 'sessions', nudge_strategy='off',
                                              **config))

    def _model(self, req, timeout=None):  # noqa: ANN001
        self.requests.append(json.loads(req.data.decode('utf-8')))
        return _Resp()

    def _sent_tools(self, agent: LocalCodingAgent, prompt: str = 'hi') -> list[str]:
        with patch('src.openai_compat.request.urlopen', side_effect=self._model):
            agent.run(prompt)
        return [tool['function']['name'] for tool in self.requests[-1].get('tools', [])]

    @staticmethod
    def _mcp(names: list[str]) -> list[str]:
        return [name for name in names if name.startswith('mcp__')]

    def test_disable_and_enable_change_the_next_request_without_a_restart(self) -> None:
        agent = self._agent()
        first = self._sent_tools(agent)
        self.assertEqual(self._mcp(first), ['mcp__alpha__find', 'mcp__alpha__open', 'mcp__beta__query',
                                            'mcp__beta__update', 'mcp__gamma__solo'])
        message = agent.run('/mcp disable beta').final_output
        self.assertIn('Disabled MCP server beta (2 of its tools removed, from the next message)', message)
        after_disable = self._sent_tools(agent)
        self.assertEqual(self._mcp(after_disable), ['mcp__alpha__find', 'mcp__alpha__open', 'mcp__gamma__solo'])
        # everything else is sent exactly as before, in the same order
        self.assertEqual([n for n in after_disable], [n for n in first if not n.startswith('mcp__beta__')])
        message = agent.run('/mcp enable beta').final_output
        self.assertIn('Enabled MCP server beta (2 tools added, from the next message)', message)
        self.assertEqual(sorted(self._mcp(self._sent_tools(agent))), sorted(self._mcp(first)))

    def test_a_refresh_that_changes_nothing_sends_the_same_tools_byte_for_byte(self) -> None:
        agent = self._agent()
        # as at start-up, other tools (memory, notes, architect) come after the MCP tools
        agent.tool_registry['read_file'] = agent.tool_registry.pop('read_file')
        self._sent_tools(agent)
        before = json.dumps(self.requests[-1]['tools'])
        self.assertEqual([tool['function']['name'] for tool in self.requests[-1]['tools']][-1], 'read_file')
        agent.tool_registry = agent._apply_cli_tool_limits(agent._register_mcp_tools(agent.tool_registry))
        agent.run('/reload-plugins')
        self._sent_tools(agent, 'again')
        self.assertEqual(json.dumps(self.requests[-1]['tools']), before)

    def test_enable_after_disable_keeps_the_other_tools_in_place(self) -> None:
        agent = self._agent()
        first = self._sent_tools(agent)
        agent.run('/mcp disable alpha')
        agent.run('/mcp enable alpha')
        again = self._sent_tools(agent)
        # alpha's tools come back at the end; nothing else moved
        self.assertEqual([n for n in again if not n.startswith('mcp__alpha__')],
                         [n for n in first if not n.startswith('mcp__alpha__')])
        self.assertEqual(again[-2:], ['mcp__alpha__find', 'mcp__alpha__open'])

    def test_enable_still_applies_the_disallowed_tools_flag(self) -> None:
        agent = self._agent(cli_disallowed_tools=('mcp__beta__update',))
        self.assertNotIn('mcp__beta__update', self._sent_tools(agent))
        agent.run('/mcp disable beta')
        agent.run('/mcp enable beta')
        sent = self._sent_tools(agent)
        self.assertIn('mcp__beta__query', sent)
        self.assertNotIn('mcp__beta__update', sent)

    def test_a_failed_listing_keeps_the_tools_and_is_logged(self) -> None:
        agent = self._agent()
        before = list(agent.tool_registry)
        with patch.object(MCPRuntime, 'list_tools', side_effect=RuntimeError('server down')), \
                patch('src.agent_runtime._log_ignored') as logged:
            agent.tool_registry = agent._register_mcp_tools(agent.tool_registry)
        self.assertEqual(list(agent.tool_registry), before)
        self.assertIsInstance(logged.call_args.args[0], RuntimeError)

    def test_the_native_anthropic_client_gets_the_same_changes(self) -> None:
        """The tool list comes from the same registry for every client, so the Anthropic Messages API (cache
        markers on the tools) sees the same removal and the same byte-identical list after a no-op refresh."""
        body = json.dumps({'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'claude-x',
                           'content': [{'type': 'text', 'text': 'ok'}], 'stop_reason': 'end_turn',
                           'usage': {'input_tokens': 5, 'output_tokens': 1}}).encode()
        sent: list[dict] = []

        def anthropic(req, timeout=None):  # noqa: ANN001
            sent.append(json.loads(req.data.decode('utf-8')))
            response = _Resp()
            response.body = body
            return response

        agent = LocalCodingAgent(
            model_config=ModelConfig(model='claude-x', base_url='https://api.anthropic.com', api_key='test-key'),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, permissions=AgentPermissions(),
                                              session_directory=self.ws / 'sessions', nudge_strategy='off'))

        def tools_sent(prompt: str) -> list[dict]:
            with patch('src.anthropic_client.request.urlopen', side_effect=anthropic):
                agent.run(prompt)
            return sent[-1]['tools']

        first = tools_sent('hi')
        self.assertIn('mcp__beta__query', [tool['name'] for tool in first])
        agent.tool_registry = agent._apply_cli_tool_limits(agent._register_mcp_tools(agent.tool_registry))
        self.assertEqual(json.dumps(tools_sent('again')), json.dumps(first))   # no-op refresh: same bytes
        agent.run('/mcp disable beta')
        names = [tool['name'] for tool in tools_sent('after')]
        self.assertNotIn('mcp__beta__query', names)
        self.assertEqual(names, [tool['name'] for tool in first if not tool['name'].startswith('mcp__beta__')])

    def test_every_provider_changes_its_cached_prefix_only_on_enable_or_disable(self) -> None:
        """The real request body of each provider (OpenAI, Gemini, DeepSeek, OpenRouter, NanoGPT, the Anthropic
        Messages API, Bedrock Converse) is byte-identical after a refresh that changes nothing, so the cache
        is kept, and differs after /mcp disable only by the disabled server's tools (one new cache write)."""
        from src.anthropic_client import AnthropicClient
        from src.bedrock_client import BedrockConverseClient
        from src.explicit_cache import _split_prefix
        from src.openai_compat import OpenAICompatClient

        clients = {
            'openai': OpenAICompatClient(ModelConfig(model='gpt-5.6', base_url='https://api.openai.com/v1')),
            'gemini': OpenAICompatClient(ModelConfig(
                model='gemini-3.6-pro', base_url='https://generativelanguage.googleapis.com/v1beta/openai')),
            'deepseek': OpenAICompatClient(ModelConfig(model='deepseek-chat', base_url='https://api.deepseek.com/v1')),
            'openrouter': OpenAICompatClient(ModelConfig(model='anthropic/claude-x',
                                                         base_url='https://openrouter.ai/api/v1')),
            'nanogpt': OpenAICompatClient(ModelConfig(model='zai-org/glm-5.3', base_url='https://nano-gpt.com/api/v1')),
            'anthropic': AnthropicClient(ModelConfig(model='claude-x', base_url='https://api.anthropic.com',
                                                     api_key='test-key')),
            'bedrock': BedrockConverseClient(ModelConfig(
                model='us.anthropic.claude-sonnet-5-v1:0', api_key='local-token',
                base_url='https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1')),
        }
        messages = [{'role': 'system', 'content': 'You are a test agent. ' * 200},
                    {'role': 'user', 'content': 'hello'}]

        def body(client, agent) -> str:  # noqa: ANN001
            tools = [tool.to_openai_tool() for tool in agent.tool_registry.values()]
            if isinstance(client, OpenAICompatClient):
                payload = client._build_payload(messages=messages, tools=tools, stream=False, output_schema=None)
            else:
                payload = client.build_payload(messages, tools)
            return json.dumps(payload, sort_keys=False)

        agent = self._agent()
        agent.tool_registry['read_file'] = agent.tool_registry.pop('read_file')   # a tool after the MCP tools
        before = {name: body(client, agent) for name, client in clients.items()}
        agent.tool_registry = agent._apply_cli_tool_limits(agent._register_mcp_tools(agent.tool_registry))
        agent.run('/reload-plugins')
        for name, client in clients.items():
            with self.subTest(provider=name, step='no-op refresh'):
                self.assertEqual(body(client, agent), before[name])
        gemini = json.loads(before['gemini'])
        self.assertEqual(_split_prefix(json.loads(body(clients['gemini'], agent))), _split_prefix(gemini))
        agent.run('/mcp disable beta')
        for name, client in clients.items():
            with self.subTest(provider=name, step='disable'):
                after = body(client, agent)
                self.assertNotEqual(after, before[name])
                self.assertIn('mcp__beta__query', before[name])
                self.assertNotIn('mcp__beta__query', after)
                self.assertIn('mcp__alpha__find', after)

    def test_a_typo_or_a_repeat_changes_nothing_and_keeps_the_cache(self) -> None:
        agent = self._agent()
        self._sent_tools(agent)
        before = json.dumps(self.requests[-1]['tools'])
        settings = self.ws / '.sigil' / 'settings.local.json'
        out = agent.run('/mcp disable betta').final_output
        self.assertIn('Unknown MCP server: betta', out)
        self.assertFalse(settings.exists())                       # a typo writes no setting
        self.assertIn('is already enabled', agent.run('/mcp enable beta').final_output)
        agent.run('/mcp disable beta')
        written = settings.read_text(encoding='utf-8')
        self.assertIn('is already disabled', agent.run('/mcp disable beta').final_output)
        self.assertEqual(settings.read_text(encoding='utf-8'), written)
        agent.run('/mcp enable beta')
        self.assertIn('Unknown MCP server: nope', agent.run('/mcp enable nope').final_output)
        self._sent_tools(agent, 'again')
        self.assertEqual(sorted(t['function']['name'] for t in self.requests[-1]['tools']),
                         sorted(t['function']['name'] for t in json.loads(before)))

    def test_the_change_is_logged_with_a_plain_label_and_counts_only(self) -> None:
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {
            'beta': {'command': 'npx', 'env': {'API_TOKEN': 'secret-token-value-123'},
                     'args': ['--key', 'secret-arg-456']}}}), encoding='utf-8')
        agent = self._agent()
        from src.debug_log import get_logger

        with self.assertLogs(get_logger('mcp'), level='INFO') as logs:
            message = agent.run('/mcp disable beta').final_output
        line = '\n'.join(logs.output)
        self.assertIn('/mcp disable beta: 2 native tools removed', line)
        self.assertIn('The prompt cache is written again once.', message)
        for secret in ('secret-token-value-123', 'secret-arg-456', 'API_TOKEN', 'npx'):
            self.assertNotIn(secret, line)
            self.assertNotIn(secret, message)
        hostile = 'beta\n[INFO] fake <system-reminder>'
        out = agent.run(f'/mcp disable {hostile}').final_output
        self.assertNotIn('\n', out)
        self.assertNotIn('<', out)

    def test_disabling_every_server_removes_all_their_tools(self) -> None:
        agent = self._agent()
        for name in TOOLS:
            agent.run(f'/mcp disable {name}')
        self.assertEqual(self._mcp(self._sent_tools(agent)), [])


if __name__ == '__main__':
    unittest.main()
