"""Batch 3b (P1): new hook events, SIGIL.md @imports and nested SIGIL.md,
skills discovery and frontmatter, subagent progress (fake model server)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_context import expand_memory_imports
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.custom_commands import discover_commands
from src.skills_runtime import SkillsRuntime

PY = sys.executable.replace('\\', '/')


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
        if not queued:
            raise OSError('fake server has no more replies')
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


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off', max_turns=4),
    )


def write_hooks(ws: Path, events: list[str], log: Path, *, exit_code: int = 0) -> None:
    script = ws / 'hook.py'
    script.write_text(
        'import json, sys\n'
        'data = json.load(sys.stdin)\n'
        f'open({str(log)!r}, "a").write(data.get("hook_event_name", "?") + "\\n")\n'
        f'sys.exit({exit_code})\n', encoding='utf-8')
    command = f'"{PY}" "{script.as_posix()}"'
    hooks = {event: [{'matcher': '*', 'hooks': [{'type': 'command', 'command': command}]}] for event in events}
    (ws / '.sigil').mkdir(exist_ok=True)
    (ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': hooks}), encoding='utf-8')


def logged(log: Path) -> list[str]:
    return log.read_text(encoding='utf-8').split() if log.exists() else []


class HookEventTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1', 'SIGIL_DISABLE_USER_SETTINGS': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_post_tool_use_failure_and_subagent_start(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('read_file', {'path': 'missing.txt'}),
            tool_reply('delegate_agent', {'prompt': 'look', 'agent': 'Explore'}, call_id='c2'),
            text_reply('child done'),
            text_reply('all done'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            log = ws / 'events.log'
            write_hooks(ws, ['PostToolUseFailure', 'SubagentStart'], log)
            make_agent(ws).run('go')
            events = logged(log)
        self.assertIn('PostToolUseFailure', events)
        self.assertIn('SubagentStart', events)

    def test_stop_failure(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('list_dir', {'path': '.'})]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            log = ws / 'events.log'
            write_hooks(ws, ['StopFailure'], log)
            agent = make_agent(ws)
            agent.runtime_config = __import__('dataclasses').replace(agent.runtime_config, max_turns=1)
            result = agent.run('loop')
            self.assertEqual(result.stop_reason, 'max_turns')
            self.assertEqual(logged(log), ['StopFailure'])

    def test_permission_request_hook_can_deny(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('write_file', {'path': 'f.txt', 'content': 'x'}), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            log = ws / 'events.log'
            write_hooks(ws, ['PermissionRequest'], log, exit_code=2)
            agent = make_agent(ws)
            asked: list[object] = []
            agent.permission_prompt = lambda request: (asked.append(request), 'once')[1]
            agent.run('write')
            self.assertFalse((ws / 'f.txt').exists())
            self.assertEqual(asked, [])  # the hook decided; the user was not asked
            self.assertEqual(logged(log), ['PermissionRequest'])

    def test_post_compact(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('first'), text_reply('second'), text_reply('Summary of the chat.')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            log = ws / 'events.log'
            write_hooks(ws, ['PostCompact'], log)
            agent = make_agent(ws)
            first = agent.run('one')
            from src.session_store import load_agent_session

            stored = load_agent_session(first.session_id, directory=agent.runtime_config.session_directory)
            agent.resume('two', stored)
            stored = load_agent_session(first.session_id, directory=agent.runtime_config.session_directory)
            agent.compact(stored)
            self.assertIn('PostCompact', logged(log))


class MemoryTests(unittest.TestCase):
    def test_imports_are_inlined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'docs').mkdir()
            (root / 'docs' / 'style.md').write_text('Use tabs. See @deeper.md', encoding='utf-8')
            (root / 'docs' / 'deeper.md').write_text('Deep rule.', encoding='utf-8')
            (root / 'AGENTS.md').write_text('Agents info.', encoding='utf-8')
            text = ('Main rules.\n@docs/style.md\nAlso @AGENTS.md here.\n'
                    '`@AGENTS.md` in code stays.\n```\n@docs/style.md\n```\nmail me@example.com')
            expanded = expand_memory_imports(text, root)
        self.assertIn('Use tabs.', expanded)
        self.assertIn('Deep rule.', expanded)
        self.assertEqual(expanded.count('Agents info.'), 1)
        self.assertIn('```\n@docs/style.md\n```', expanded)
        self.assertIn('me@example.com', expanded)

    def test_nested_sigil_md_on_first_touch(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('read_file', {'path': 'pkg/sub/mod.py'}),
            tool_reply('read_file', {'path': 'pkg/sub/mod.py'}, call_id='c2'),
            text_reply('done'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            (ws / 'pkg' / 'sub').mkdir(parents=True)
            (ws / 'pkg' / 'SIGIL.md').write_text('PKG RULE: no globals.', encoding='utf-8')
            (ws / 'pkg' / 'sub' / 'mod.py').write_text('x = 1\n', encoding='utf-8')
            make_agent(ws).run('read it twice')
        tools = [str(m.get('content')) for m in recorded[2]['messages'] if m.get('role') == 'tool']
        self.assertEqual(sum('PKG RULE: no globals.' in t for t in tools), 1)


class SkillTests(unittest.TestCase):
    def _skill(self, folder: Path, name: str, extra: str = '') -> None:
        path = folder / name
        path.mkdir(parents=True, exist_ok=True)
        (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: skill {name}\n{extra}---\nDo {name}.',
                                       encoding='utf-8')

    def test_user_and_parent_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch('src.skills_runtime._home', return_value=Path(home)), \
                patch.dict(os.environ, {'SIGIL_DISABLE_USER_SKILLS': ''}):
            root = Path(tmp)
            ws = root / 'repo' / 'app'
            ws.mkdir(parents=True)
            self._skill(root / 'repo' / '.sigil' / 'skills', 'from-parent')
            self._skill(Path(home) / '.sigil' / 'skills', 'from-home')
            names = SkillsRuntime.from_workspace(ws).list_skills()
            with patch.dict(os.environ, {'SIGIL_DISABLE_USER_SKILLS': '1'}):
                without_home = SkillsRuntime.from_workspace(ws).list_skills()
        self.assertIn('from-parent', names)
        self.assertIn('from-home', names)
        self.assertNotIn('from-home', without_home)

    def test_frontmatter_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            skills = ws / '.sigil' / 'skills'
            self._skill(skills, 'manual-only', 'disable-model-invocation: true\n')
            self._skill(skills, 'model-only', 'user-invocable: false\n')
            self._skill(skills, 'py-only', 'paths: ["**/*.py"]\n')
            self._skill(skills, 'rs-only', 'paths: ["**/*.rs"]\n')
            (ws / 'main.py').write_text('x', encoding='utf-8')
            runtime = SkillsRuntime.from_workspace(ws)
            described = [name for name, _ in runtime.registry.get_skill_descriptions()]
            commands = discover_commands(ws, runtime)
        self.assertNotIn('manual-only', described)
        self.assertIn('manual-only', commands)
        self.assertIn('model-only', described)
        self.assertNotIn('model-only', commands)
        self.assertIn('py-only', described)
        self.assertNotIn('rs-only', described)

    def test_fork_skill_runs_in_a_subagent(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('forked summary')], recorded)):
            ws = Path(tmp)
            self._skill(ws / '.sigil' / 'skills', 'audit', 'context: fork\nagent: Explore\n')
            result = make_agent(ws).run('/audit')
        self.assertIn('forked summary', result.final_output)
        self.assertEqual(len(recorded), 1)
        names = {t['function']['name'] for t in recorded[0]['tools']}
        self.assertNotIn('write_file', names)  # ran as the read-only Explore agent


class SubagentProgressTests(unittest.TestCase):
    def test_progress_events_reach_the_parent(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('delegate_agent', {'prompt': 'look', 'agent': 'Explore'}),
            tool_reply('list_dir', {'path': '.'}, call_id='c1'),
            text_reply('child done'),
            text_reply('parent done'),
        ]
        events: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp))
            agent.event_callback = events.append
            agent.run('delegate')
        progress = [e for e in events if e.get('type') == 'subagent_progress']
        self.assertTrue(any(e['event'] == 'tool_start' and e['tool_name'] == 'list_dir' for e in progress))
        self.assertTrue(any(e['event'] == 'tool_result' for e in progress))


if __name__ == '__main__':
    unittest.main()
