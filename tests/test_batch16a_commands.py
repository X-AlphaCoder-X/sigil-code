"""Batch 16a (P3): /plan <task>, /todos, /recap, /cd, /theme, /keybindings, /export to Markdown,
/bug, /focus, think / think hard / ultrathink, /read-only."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig

BASE = 'http://127.0.0.1:8000/v1'


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason':
                                             'tool_calls' if message.get('tool_calls') else 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def call(name: str, **arguments) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': f'c_{name}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        self.replies: list[dict] = []
        self.requests: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            self.requests.append(json.loads(req.data.decode('utf-8')))
            return _Resp(self.replies.pop(0) if self.replies else {'role': 'assistant', 'content': 'answer'})

        for patcher in (patch('src.openai_compat.request.urlopen', side_effect=fake),
                        patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': '', 'SIGIL_THEME': ''})):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop('SIGIL_EFFORT_LEVEL', None)
        os.environ.pop('SIGIL_THEME', None)

    def agent(self, **perms) -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url=BASE),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s', nudge_strategy='off',
                                              permissions=AgentPermissions(**perms)))

    def last_user(self) -> str:
        return [m for m in self.requests[-1]['messages'] if m.get('role') == 'user'][-1]['content']


class PlanAndPromptCommands(Base):
    def test_plan_with_task_enters_plan_mode_and_runs(self) -> None:
        agent = self.agent(allow_file_write=True)
        self.assertIn('plan', agent.run('/plan').final_output.lower())
        self.assertEqual(self.requests, [])
        self.replies = [call('write_file', path='x.txt', content='no'), {'role': 'assistant', 'content': 'plan: ...'}]
        agent.run('/plan add a login page')
        self.assertEqual(agent.permission_mode, 'plan')
        sent = json.dumps(self.requests[0]['messages'])
        self.assertIn('add a login page', sent)
        self.assertIn('Plan mode is active', sent)
        self.assertFalse((self.ws / 'x.txt').exists())  # plan mode refused the write

    def test_recap_todos_and_keybindings(self) -> None:
        agent = self.agent()
        agent.run('/recap tests')
        self.assertIn('short recap of this session', self.last_user())
        self.assertIn('Focus on: tests', self.last_user())
        count = len(self.requests)
        self.assertEqual(agent.run('/todos').final_output, agent.run('/todo').final_output)
        self.assertIn('Shift+Tab', agent.run('/keybindings').final_output)
        self.assertEqual(len(self.requests), count)

    def test_think_keywords_raise_effort_for_one_prompt(self) -> None:
        agent = self.agent()
        seen: list = []
        original = agent.set_effort

        def spy(level):
            seen.append(level)
            original(level)

        agent.set_effort = spy
        agent.run('please ultrathink about this')
        self.assertEqual(seen, ['max', 'auto'])
        self.assertIsNone(agent.model_config.effort)  # back to the earlier level
        agent.set_effort('high')
        seen.clear()
        agent.run('think about it')
        self.assertEqual(seen, [])  # high already beats think (medium)
        with patch.dict(os.environ, {'SIGIL_THINK_KEYWORDS': '0'}):
            agent.run('think hard')
        self.assertEqual(seen, [])


class SessionCommands(Base):
    def test_cd_changes_folder_and_fires_hook(self) -> None:
        sub = self.ws / 'sub'
        sub.mkdir()
        agent = self.agent()
        fired: list = []
        agent._run_hook_event = lambda event, payload, **kw: fired.append((event, payload)) or None
        self.assertIn('Working folder is now', agent.run('/cd sub').final_output)
        self.assertEqual(agent.runtime_config.cwd, sub.resolve())
        self.assertEqual(agent.tool_context.root, sub.resolve())
        self.assertEqual(fired[-1][0], 'CwdChanged')
        self.assertIn('Not a folder', agent.run('/cd nope').final_output)

    def test_theme_light_rewrites_colours(self) -> None:
        from src.terminal import LightStream, apply_theme, to_light

        self.assertEqual(to_light('\x1b[97mhi\x1b[0m \x1b[1;93mx'), '\x1b[30mhi\x1b[0m \x1b[1;33mx')
        agent = self.agent()
        original = sys.stdout
        try:
            self.assertIn('Theme set to light', agent.run('/theme light').final_output)
            self.assertIsInstance(sys.stdout, LightStream)
            settings = json.loads((self.ws / '.sigil' / 'settings.local.json').read_text(encoding='utf-8'))
            self.assertEqual(settings['theme'], 'light')
            apply_theme('dark')
            self.assertIs(sys.stdout, original)
            self.assertIn('Unknown theme', agent.run('/theme pink').final_output)
        finally:
            sys.stdout = original

    def test_export_markdown_bug_and_focus(self) -> None:
        self.replies = [call('read_file', path='a.txt'), {'role': 'assistant', 'content': 'The file says hi.'}]
        (self.ws / 'a.txt').write_text('hi', encoding='utf-8')
        agent = self.agent()
        agent.run('what is in a.txt?')
        agent.run('/export notes.md')
        text = (self.ws / 'notes.md').read_text(encoding='utf-8')
        self.assertTrue(text.startswith('# Conversation'))
        self.assertIn('## User', text)
        self.assertIn('## Tool: read_file', text)
        focus = agent.run('/focus').final_output
        self.assertIn('> what is in a.txt?', focus)
        self.assertIn('Tools: read_file', focus)
        self.assertIn('The file says hi.', focus)
        with patch('webbrowser.open') as opened:
            report = agent.run('/bug crash on start').final_output
        url = opened.call_args[0][0]
        self.assertIn('/issues/new?', url)
        self.assertIn('crash+on+start', url)
        self.assertIn(url, report)


class ReadOnlyTests(Base):
    def test_read_only_file_is_sent_and_protected(self) -> None:
        (self.ws / 'spec.md').write_text('THE SPEC TEXT', encoding='utf-8')
        agent = self.agent(allow_file_write=True)
        self.assertIn('Read-only for this session', agent.run('/read-only spec.md').final_output)
        self.replies = [call('edit_file', path='spec.md', old_text='THE', new_text='A'),
                        {'role': 'assistant', 'content': 'ok'},
                        call('write_file', path='other.txt', content='fine'),
                        {'role': 'assistant', 'content': 'ok'}]
        agent.run('update the spec')
        self.assertIn('THE SPEC TEXT', json.dumps(self.requests[0]['messages']))
        self.assertEqual((self.ws / 'spec.md').read_text(encoding='utf-8'), 'THE SPEC TEXT')
        tool = [m for m in self.requests[1]['messages'] if m.get('role') == 'tool'][-1]['content']
        self.assertIn('read-only in this session', tool)
        agent.run('write other')
        self.assertEqual((self.ws / 'other.txt').read_text(encoding='utf-8'), 'fine')
        self.assertIn('spec.md', agent.run('/read-only').final_output)


if __name__ == '__main__':
    unittest.main()
