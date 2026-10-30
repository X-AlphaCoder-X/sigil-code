"""3.4 subagent types (Explore, Plan, general-purpose) and .sigil/agents, through
a fake model server."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import default_tool_registry
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.skills_runtime import SkillsRuntime
from src.subagents import READ_ONLY_TOOLS, discover_subagents


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


def tool_reply(name: str, arguments: dict[str, object], call_id: str = 'call_1') -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def tool_names(payload: dict[str, object]) -> set[str]:
    return {t['function']['name'] for t in payload.get('tools', [])}


def system_text(payload: dict[str, object]) -> str:
    return '\n'.join(str(m.get('content')) for m in payload['messages'] if m.get('role') == 'system')


def make_agent(ws: Path) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='parent-model'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws,
            permissions=AgentPermissions(allow_file_write=True, allow_shell_commands=True),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad',
            nudge_strategy='off',
        ),
    )


class SubagentTypeTests(unittest.TestCase):
    def test_explore_cannot_write(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('delegate_agent', {'prompt': 'look around', 'agent': 'Explore',
                                          'allow_write': True, 'allow_shell': True}),
            tool_reply('write_file', {'path': 'hacked.txt', 'content': 'x'}, call_id='child_1'),
            text_reply('child found things'),
            text_reply('parent done'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            result = make_agent(ws).run('explore please')
            self.assertFalse((ws / 'hacked.txt').exists())
        child_tools = tool_names(recorded[1])
        self.assertNotIn('write_file', child_tools)
        self.assertNotIn('bash', child_tools)
        self.assertIn('read_file', child_tools)
        self.assertTrue(child_tools <= READ_ONLY_TOOLS)
        self.assertIn('Explore subagent', system_text(recorded[1]))
        self.assertEqual(result.final_output, 'parent done')

    def test_custom_agent_tools_model_and_prompt(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('delegate_agent', {'prompt': 'review src', 'subagent_type': 'reviewer'}),
            text_reply('review written'),
            text_reply('parent done'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agents = ws / '.sigil' / 'agents' / 'quality'
            agents.mkdir(parents=True)
            (agents / 'reviewer.md').write_text(
                '---\nname: reviewer\ndescription: Reviews code\ntools: Read, Grep\n'
                'model: small-review-model\n---\nYou are a strict code reviewer.',
                encoding='utf-8')
            make_agent(ws).run('get a review')
        self.assertEqual(recorded[0]['model'], 'parent-model')
        self.assertEqual(recorded[1]['model'], 'small-review-model')
        self.assertEqual(tool_names(recorded[1]), {'read_file', 'grep_search'})
        self.assertIn('You are a strict code reviewer.', system_text(recorded[1]))

    def test_unknown_agent_type_lists_choices(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('delegate_agent', {'prompt': 'x', 'agent': 'nope'}), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            make_agent(Path(tmp)).run('delegate')
        tool_messages = [str(m.get('content')) for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any("Unknown agent type 'nope'" in m and 'Explore' in m for m in tool_messages))

    def test_discovery_sources_and_inherit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch('src.subagents._home', return_value=Path(home)):
            ws = Path(tmp)
            (Path(home) / '.sigil' / 'agents').mkdir(parents=True)
            (Path(home) / '.sigil' / 'agents' / 'mine.md').write_text(
                '---\nname: mine\ndescription: personal\nmodel: inherit\ndisallowedTools: Bash\n---\nbody',
                encoding='utf-8')
            agents = discover_subagents(ws)
            self.assertEqual(agents['mine'].source, 'user')
            filtered = agents['mine'].filter_tools(default_tool_registry())
            self.assertNotIn('bash', filtered)
            self.assertIn('write_file', filtered)
            for name in ('general-purpose', 'Explore', 'Plan'):
                self.assertIn(name, agents)

    def test_skill_repo_folder_is_a_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            repo_agents = ws / 'my-skills' / 'agents'
            repo_agents.mkdir(parents=True)
            (repo_agents / 'helper.md').write_text(
                '---\nname: helper\ndescription: A helper\n---\nHelp.', encoding='utf-8')
            with patch.dict(os.environ, {'SIGIL_SKILL_REPOS': 'my-skills'}):
                runtime = SkillsRuntime.from_workspace(ws)
                found = discover_subagents(ws, runtime)
            self.assertIn('helper', found)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop('SIGIL_SKILL_REPOS', None)
                runtime = SkillsRuntime.from_workspace(ws)
            self.assertNotIn('helper', discover_subagents(ws, runtime))


if __name__ == '__main__':
    unittest.main()
