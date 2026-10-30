"""Plugin folders loaded with --plugin-dir: commands, agents, skills,
hooks and MCP servers, with ${SIGIL_PLUGIN_ROOT} filled in."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.sigil_plugins import load_plugin, loaded_plugins, render_plugin_list
from src.custom_commands import discover_commands
from src.main import _apply_flag_environment, build_parser
from src.mcp_runtime import MCPRuntime, connection_manager
from src.subagents import discover_subagents
from tests.test_mcp_connections import SERVER

PY = sys.executable.replace('\\', '/')


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason': 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def make_plugin(base: Path, name: str = 'myplug') -> Path:
    root = base / name
    (root / '.sigil-plugin').mkdir(parents=True)
    (root / '.sigil-plugin' / 'plugin.json').write_text(json.dumps({
        'name': name, 'description': 'Test plugin', 'version': '1.2.0', 'agents': './extra-agents'}),
        encoding='utf-8')
    (root / 'commands').mkdir()
    (root / 'commands' / 'hello.md').write_text(
        '---\ndescription: Say hello\n---\nHello $ARGUMENTS from ${SIGIL_PLUGIN_ROOT}/data.txt\n', encoding='utf-8')
    (root / 'agents').mkdir()
    (root / 'agents' / 'reviewer.md').write_text(
        '---\nname: reviewer\ndescription: Reviews code\ntools: Read, Grep\n---\nYou review code.\n', encoding='utf-8')
    (root / 'extra-agents').mkdir()
    (root / 'extra-agents' / 'tester.md').write_text(
        '---\nname: tester\ndescription: Writes tests\n---\nYou write tests.\n', encoding='utf-8')
    (root / 'skills' / 'greet').mkdir(parents=True)
    (root / 'skills' / 'greet' / 'SKILL.md').write_text(
        '---\nname: greet\ndescription: Greets people politely\n---\nSay hi.\n', encoding='utf-8')
    (root / 'hook.py').write_text(
        'import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding="utf-8")\n',
        encoding='utf-8')
    (root / 'hooks').mkdir()
    (root / 'hooks' / 'hooks.json').write_text(json.dumps({'hooks': {'UserPromptSubmit': [{'hooks': [{
        'type': 'command',
        'command': f'"{PY}" "${{SIGIL_PLUGIN_ROOT}}/hook.py" "${{SIGIL_PLUGIN_ROOT}}/hook-ran.json"'}]}]}}),
        encoding='utf-8')
    (root / 'server.py').write_text(SERVER, encoding='utf-8')
    (root / '.mcp.json').write_text(json.dumps({'mcpServers': {'plugsrv': {
        'command': sys.executable, 'args': ['-u', '${SIGIL_PLUGIN_ROOT}/server.py',
                                            '${SIGIL_PLUGIN_ROOT}/starts.txt']}}}), encoding='utf-8')
    return root


class PluginDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(connection_manager().close_all)
        self.base = Path(self.tmp.name).resolve()
        self.ws = self.base / 'ws'
        self.ws.mkdir()
        self.root = make_plugin(self.base)
        env = patch.dict(os.environ, {'SIGIL_PLUGIN_DIRS': str(self.root), 'SIGIL_SAFE_MODE': ''})
        env.start()
        self.addCleanup(env.stop)

    def test_manifest_and_parts(self) -> None:
        plugin = load_plugin(self.root)
        self.assertEqual((plugin.name, plugin.version), ('myplug', '1.2.0'))
        self.assertEqual([p.name for p in plugin.agent_dirs], ['agents', 'extra-agents'])
        self.assertIn('UserPromptSubmit', plugin.hooks)
        self.assertIn(str(self.root), plugin.hooks['UserPromptSubmit'][0]['hooks'][0]['command'])
        self.assertEqual(plugin.mcp_servers['plugsrv']['args'][1], f'{self.root}/server.py')
        text = render_plugin_list()
        self.assertIn('myplug 1.2.0', text)
        self.assertIn('1 MCP server(s)', text)

    def test_commands_and_agents_are_namespaced(self) -> None:
        commands = discover_commands(self.ws)
        self.assertIn('myplug:hello', commands)
        self.assertIn('hello', commands)
        self.assertIn(f'from {self.root}/data.txt', commands['myplug:hello'].body)
        self.assertEqual(commands['hello'].source, 'plugin:myplug')
        (self.ws / '.sigil' / 'commands').mkdir(parents=True)
        (self.ws / '.sigil' / 'commands' / 'hello.md').write_text('Project hello', encoding='utf-8')
        commands = discover_commands(self.ws)
        self.assertEqual(commands['hello'].body, 'Project hello')  # a project command wins the plain name
        self.assertIn('Hello', commands['myplug:hello'].body)  # the plugin one stays reachable
        agents = discover_subagents(self.ws)
        self.assertEqual(agents['myplug:reviewer'].tools, ('Read', 'Grep'))
        self.assertIn('myplug:tester', agents)
        self.assertEqual(agents['reviewer'].source, 'plugin:myplug')

    def test_mcp_server_from_plugin(self) -> None:
        runtime = MCPRuntime.from_workspace(self.ws)
        self.assertEqual([s.name for s in runtime.servers], ['plugsrv'])
        self.assertEqual([t.name for t in runtime.list_tools()], ['echo'])
        self.assertTrue((self.root / 'starts.txt').exists())
        with patch.dict(os.environ, {'SIGIL_STRICT_MCP_CONFIG': '1'}):
            self.assertEqual(MCPRuntime.from_workspace(self.ws).servers, ())

    def test_agent_runs_plugin_command_hook_skill_and_mcp_tool(self) -> None:
        recorded: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp({'role': 'assistant', 'content': 'hi back'})

        with patch('src.openai_compat.request.urlopen', side_effect=fake):
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s' / 'agent',
                                                  scratchpad_root=self.ws / '.s' / 'scratch', nudge_strategy='off'))
            self.assertIn('mcp__plugsrv__echo', agent.tool_registry)
            self.assertIn('greet', agent.skills_runtime.list_skills())
            agent.run('/myplug:hello world')
        sent = [m for m in recorded[-1]['messages'] if m.get('role') == 'user']
        self.assertTrue(any(f'Hello world from {self.root}/data.txt' in str(m.get('content')) for m in sent), sent)
        names = {t['function']['name'] for t in recorded[-1]['tools']}
        self.assertIn('mcp__plugsrv__echo', names)
        hook_input = json.loads((self.root / 'hook-ran.json').read_text(encoding='utf-8'))
        self.assertEqual(hook_input['hook_event_name'], 'UserPromptSubmit')
        self.assertIn('Plugin folders', agent.run('/plugins').final_output)

    def test_flag_safe_mode_and_missing_folder(self) -> None:
        with patch.dict(os.environ, {'SIGIL_PLUGIN_DIRS': ''}):
            args = build_parser().parse_args(['agent', 'hi', '--plugin-dir', str(self.root),
                                              '--plugin-dir', str(self.base / 'nope')])
            _apply_flag_environment(args)
            self.assertEqual(os.environ['SIGIL_PLUGIN_DIRS'].split(os.pathsep),
                             [str(self.root.resolve()), str((self.base / 'nope').resolve())])
            self.assertEqual([p.name for p in loaded_plugins()], ['myplug'])  # the missing one is skipped
            with patch.dict(os.environ, {'SIGIL_SAFE_MODE': '1'}):
                self.assertEqual(loaded_plugins(), [])
                self.assertNotIn('myplug:hello', discover_commands(self.ws))
        self.assertIn('No plugin folders', render_plugin_list([]))


if __name__ == '__main__':
    unittest.main()
