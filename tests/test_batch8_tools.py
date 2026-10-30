"""Batch 8 (P2): PDF reading, monitor filter, PowerShell tool, parallel read-only tool calls,
Skill tool, skill list token cap and .agents/skills, output styles."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.background_shells import shell_manager
from src.skills_runtime import SkillsRuntime, _skills_within_budget


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def fake_server(responses: list[dict[str, object]], recorded: list[dict[str, object]]):
    queued = [FakeHTTPResponse(payload) for payload in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        recorded.append(json.loads(request_obj.data.decode('utf-8')))
        return queued.pop(0)

    return _fake_urlopen


def text_reply(text: str) -> dict[str, object]:
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}


def multi_tool_reply(calls: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': f'call_{i}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(calls)]},
        'finish_reason': 'tool_calls'}]}


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


class PdfTests(unittest.TestCase):
    def test_pdf_with_and_without_pypdf(self) -> None:
        class _Page:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        fake = types.ModuleType('pypdf')
        fake.PdfReader = lambda path: types.SimpleNamespace(pages=[_Page('Page one text'), _Page('Page two text')])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'spec.pdf').write_bytes(b'%PDF-1.4 fake')
            context = build_tool_context(AgentRuntimeConfig(cwd=Path(tmp)))
            with patch.dict(sys.modules, {'pypdf': fake}):
                ok = execute_tool(default_tool_registry(), 'read_file', {'path': 'spec.pdf'}, context)
            with patch.dict(sys.modules, {'pypdf': None}):
                missing = execute_tool(default_tool_registry(), 'read_file', {'path': 'spec.pdf'}, context)
        self.assertTrue(ok.ok, ok.content)
        self.assertIn('2 page(s)', ok.content)
        self.assertIn('Page two text', ok.content)
        self.assertFalse(missing.ok)
        self.assertIn('pip install', missing.content)


class MonitorTests(unittest.TestCase):
    def tearDown(self) -> None:
        shell_manager().kill_all()

    def test_filter_and_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = AgentRuntimeConfig(cwd=Path(tmp), permissions=AgentPermissions(allow_shell_commands=True),
                                        scratchpad_root=Path(tmp) / 'scratch')
            context = build_tool_context(config)
            registry = default_tool_registry()
            started = execute_tool(registry, 'bash', {
                'command': 'echo info one; echo ERROR boom; sleep 1; echo ERROR late; sleep 30',
                'run_in_background': True}, context)
            job_id = started.metadata['shell_id']
            first = execute_tool(registry, 'bash_output', {'id': job_id, 'filter': 'ERROR', 'wait_seconds': 10}, context)
            second = execute_tool(registry, 'bash_output', {'id': job_id, 'filter': 'ERROR', 'wait_seconds': 10},
                                  context)
            shell_manager().kill_all()
            time.sleep(0.3)
        self.assertIn('ERROR boom', first.content)
        self.assertNotIn('info one', first.content)
        self.assertIn('did not match', first.content)
        self.assertIn('ERROR late', second.content)


class PowerShellTests(unittest.TestCase):
    def test_tool_switch(self) -> None:
        with patch.dict(os.environ, {'SIGIL_POWERSHELL_TOOL': '1'}):
            self.assertIn('powershell', default_tool_registry())
        with patch.dict(os.environ, {'SIGIL_POWERSHELL_TOOL': '0'}):
            self.assertNotIn('powershell', default_tool_registry())

    def test_permissions_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_POWERSHELL_TOOL': '1'}):
            registry = default_tool_registry()
            off = execute_tool(registry, 'powershell', {'command': 'Write-Output hi'},
                               build_tool_context(AgentRuntimeConfig(cwd=Path(tmp))))
            shell = build_tool_context(AgentRuntimeConfig(cwd=Path(tmp), permissions=AgentPermissions(
                allow_shell_commands=True)))
            destructive = execute_tool(registry, 'powershell', {'command': 'Remove-Item x -Recurse'}, shell)
            if sys.platform == 'win32':
                ran = execute_tool(registry, 'powershell', {'command': 'Write-Output ps-ok'}, shell)
                self.assertIn('ps-ok', ran.content)
        self.assertFalse(off.ok)
        self.assertIn('--allow-shell', off.content)
        self.assertFalse(destructive.ok)
        self.assertIn('--unsafe', destructive.content)


class ParallelToolTests(unittest.TestCase):
    def test_read_only_calls_run_together(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [multi_tool_reply([('read_file', {'path': 'a.txt'}), ('read_file', {'path': 'b.txt'}),
                                       ('list_dir', {'path': '.'})]), text_reply('done')]
        active = {'now': 0, 'peak': 0}
        lock = threading.Lock()
        import src.agent_tools as agent_tools

        original = agent_tools._read_file

        def _slow_read(arguments, context):  # noqa: ANN001
            with lock:
                active['now'] += 1
                active['peak'] = max(active['peak'], active['now'])
            time.sleep(0.4)
            try:
                return original(arguments, context)
            finally:
                with lock:
                    active['now'] -= 1

        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            (ws / 'a.txt').write_text('alpha', encoding='utf-8')
            (ws / 'b.txt').write_text('beta', encoding='utf-8')
            agent = make_agent(ws)
            tool = agent.tool_registry['read_file']
            agent.tool_registry['read_file'] = type(tool)(name=tool.name, description=tool.description,
                                                          parameters=tool.parameters, handler=_slow_read)
            result = agent.run('read both')
        self.assertEqual(result.final_output, 'done')
        self.assertEqual(active['peak'], 2)
        tools = [m for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertEqual([m['tool_call_id'] for m in tools], ['call_0', 'call_1', 'call_2'])
        self.assertIn('alpha', tools[0]['content'])
        self.assertIn('beta', tools[1]['content'])

    def test_switch_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_PARALLEL_TOOLS': '0'}):
            agent = make_agent(Path(tmp))
            from src.agent_types import ToolCall

            calls = (ToolCall('a', 'list_dir', {}), ToolCall('b', 'list_dir', {}))
            self.assertEqual(agent._precompute_read_only_calls(calls), {})


class SkillTests(unittest.TestCase):
    def test_skill_tool_runs_a_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / '.sigil' / 'commands').mkdir(parents=True)
            (ws / '.sigil' / 'commands' / 'release.md').write_text('Prepare release $ARGUMENTS.', encoding='utf-8')
            agent = make_agent(ws)
            result = execute_tool(agent.tool_registry, 'skill', {'name': '/release', 'arguments': 'v2'},
                                  agent.tool_context)
            missing = execute_tool(agent.tool_registry, 'skill', {'name': 'nope'}, agent.tool_context)
        self.assertTrue(result.ok, result.content)
        self.assertIn('Prepare release v2.', result.content)
        self.assertFalse(missing.ok)

    def test_token_budget_and_agents_folder(self) -> None:
        items = [(f'skill-{i}', 'x' * 150) for i in range(100)]
        with patch.dict(os.environ, {'SIGIL_SKILL_LIST_TOKENS': '400'}):
            small = _skills_within_budget(items)
        with patch.dict(os.environ, {'SIGIL_SKILL_LIST_TOKENS': '4000'}):
            large = _skills_within_budget(items)
        self.assertLess(len(small), len(large))
        self.assertGreater(len(small), 0)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / '.agents' / 'skills' / 'shared'
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text('---\nname: shared\ndescription: from .agents\n---\nbody',
                                             encoding='utf-8')
            self.assertIn('shared', SkillsRuntime.from_workspace(Path(tmp)).list_skills())


class OutputStyleTests(unittest.TestCase):
    def test_style_in_system_prompt(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('ok')], recorded)):
            ws = Path(tmp)
            (ws / '.sigil' / 'output-styles').mkdir(parents=True)
            (ws / '.sigil' / 'output-styles' / 'pirate.md').write_text(
                '---\nname: pirate\ndescription: Talk like a pirate\n---\nAnswer like a pirate.', encoding='utf-8')
            agent = make_agent(ws)
            listing = agent.run('/output-style').final_output
            agent.run('/output-style pirate')
            agent.run('hello')
            saved = json.loads((ws / '.sigil' / 'settings.local.json').read_text(encoding='utf-8'))
        self.assertIn('- concise', listing)
        self.assertIn('- pirate: Talk like a pirate', listing)
        self.assertEqual(saved['outputStyle'], 'pirate')
        system = '\n'.join(str(m['content']) for m in recorded[0]['messages'] if m['role'] == 'system')
        self.assertIn('Answer like a pirate.', system)


if __name__ == '__main__':
    unittest.main()
