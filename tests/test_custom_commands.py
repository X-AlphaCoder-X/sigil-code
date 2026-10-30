"""3.3 custom slash commands (.sigil/commands) and skills as /name, through a
fake model server."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.custom_commands import parse_frontmatter, substitute_arguments, tool_list
from src.providers import resolve_model_alias


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


def tool_reply(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def last_user_text(payload: dict[str, object]) -> str:
    users = [m for m in payload['messages'] if m.get('role') == 'user']
    return str(users[-1]['content'])


def make_agent(ws: Path, *, shell: bool = False) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='main-model'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws,
            permissions=AgentPermissions(allow_shell_commands=shell),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad',
            nudge_strategy='off',
        ),
    )


def write_command(ws: Path, name: str, text: str) -> None:
    folder = ws / '.sigil' / 'commands'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.md').write_text(text, encoding='utf-8')


class HelperTests(unittest.TestCase):
    def test_argument_substitution(self) -> None:
        self.assertEqual(substitute_arguments('Fix $ARGUMENTS now', 'issue 12'), 'Fix issue 12 now')
        self.assertEqual(substitute_arguments('from $0 to $1', 'a "b c"'), 'from a to b c')
        self.assertEqual(substitute_arguments('first=$ARGUMENTS[0]', 'x y'), 'first=x')
        self.assertEqual(substitute_arguments('Review the code', 'src/a.py'),
                         'Review the code\n\nARGUMENTS: src/a.py')
        self.assertEqual(substitute_arguments('Plain', ''), 'Plain')

    def test_frontmatter_and_tool_lists(self) -> None:
        meta, body = parse_frontmatter('---\ndescription: d\nallowed-tools: Bash(git add *), Read\n---\nBody')
        self.assertEqual(meta['description'], 'd')
        self.assertEqual(body.strip(), 'Body')
        self.assertEqual(tool_list(meta['allowed-tools']), ('Bash(git add *)', 'Read'))
        self.assertEqual(tool_list(['Edit', 'Write']), ('Edit', 'Write'))

    def test_builtin_parser_without_pyyaml(self) -> None:
        text = ('---\nname: reviewer\ndescription: >\n  Reviews code\n  for bugs\ntools:\n  - Read\n'
                '  - Grep\nallowed-tools: Bash(git diff *) Read\nnotes: |\n  line one\n  line two\n'
                'model: "inherit"\n---\nBody text')
        with patch('src.custom_commands.yaml', None):
            meta, body = parse_frontmatter(text)
        self.assertEqual(meta['description'], 'Reviews code for bugs')
        self.assertEqual(meta['tools'], ['Read', 'Grep'])
        self.assertEqual(tool_list(meta['allowed-tools']), ('Bash(git diff *)', 'Read'))
        self.assertEqual(meta['notes'], 'line one\nline two')
        self.assertEqual(meta['model'], 'inherit')
        self.assertEqual(body.strip(), 'Body text')

    def test_skills_runtime_uses_builtin_parser(self) -> None:
        from src.skills_runtime import SkillsRegistry

        with patch('src.skills_runtime.yaml', None):
            meta, _ = SkillsRegistry()._parse_frontmatter('---\nname: x\ntools:\n  - Read\n  - Bash\n---\nb')
        self.assertEqual(meta['tools'], ['Read', 'Bash'])

    def test_model_aliases(self) -> None:
        with patch.dict(os.environ, {'SIGIL_MODEL_HAIKU': 'small-local'}):
            self.assertEqual(resolve_model_alias('haiku'), 'small-local')
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_model_alias('sonnet'))
        self.assertIsNone(resolve_model_alias('inherit'))
        self.assertEqual(resolve_model_alias('qwen3-coder'), 'qwen3-coder')


class CustomCommandRunTests(unittest.TestCase):
    def test_project_command_expands_arguments(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('ok')], recorded)):
            ws = Path(tmp)
            write_command(ws, 'fix-issue', '---\ndescription: Fix an issue\n---\nFix issue #$0 in $1 carefully.')
            make_agent(ws).run('/fix-issue 42 src/app.py')
        self.assertEqual(last_user_text(recorded[0]), 'Fix issue #42 in src/app.py carefully.')

    def test_user_command_and_project_wins(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch('src.custom_commands._home', return_value=Path(home)), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('a'), text_reply('b')], recorded)):
            ws = Path(tmp)
            write_command(Path(home), 'mine', 'personal command')
            write_command(Path(home), 'shared', 'from home')
            write_command(ws, 'shared', 'from project')
            agent = make_agent(ws)
            agent.run('/mine')
            agent.run('/shared')
        self.assertEqual(last_user_text(recorded[0]), 'personal command')
        self.assertEqual(last_user_text(recorded[1]), 'from project')

    def test_shell_injection_respects_permissions(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('a'), text_reply('b'), text_reply('c')], recorded)):
            ws = Path(tmp)
            write_command(ws, "show-state", "Output: !`echo injected-ok`")
            write_command(ws, 'allowed', '---\nallowed-tools: Bash(echo *)\n---\nOutput: !`echo via-rule`')
            make_agent(ws, shell=True).run("/show-state")
            make_agent(ws, shell=False).run("/show-state")
            make_agent(ws, shell=False).run('/allowed')
        self.assertEqual(last_user_text(recorded[0]), 'Output: injected-ok')
        self.assertIn('command not run', last_user_text(recorded[1]))
        self.assertEqual(last_user_text(recorded[2]), 'Output: via-rule')

    def test_file_mention_adds_content(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('ok')], recorded)):
            ws = Path(tmp)
            (ws / 'notes.txt').write_text('remember the milk', encoding='utf-8')
            write_command(ws, 'explain', 'Explain @notes.txt please')
            make_agent(ws).run('/explain')
        text = last_user_text(recorded[0])
        self.assertIn('Explain @notes.txt please', text)
        self.assertIn('remember the milk', text)

    def test_model_and_allowed_tools_apply_for_one_turn(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('write_file', {'path': 'out.txt', 'content': 'x'}), text_reply('wrote'),
            tool_reply('write_file', {'path': 'again.txt', 'content': 'y'}), text_reply('no'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            write_command(ws, 'save', '---\nmodel: other-model\nallowed-tools: Write\n---\nSave a file')
            agent = make_agent(ws)
            agent.run('/save')
            self.assertTrue((ws / 'out.txt').exists())
            agent.run('save another')
            self.assertFalse((ws / 'again.txt').exists())
        self.assertEqual(recorded[0]['model'], 'other-model')
        self.assertEqual(recorded[1]['model'], 'other-model')
        self.assertEqual(recorded[2]['model'], 'main-model')

    def test_skill_runs_as_slash_name(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('ok')], recorded)):
            ws = Path(tmp)
            skill = ws / '.sigil' / 'skills' / 'deploy-check'
            skill.mkdir(parents=True)
            (skill / 'SKILL.md').write_text(
                '---\nname: deploy-check\ndescription: Check a deploy\n---\nCheck the deploy of $ARGUMENTS.',
                encoding='utf-8')
            make_agent(ws).run('/deploy-check staging')
        self.assertEqual(last_user_text(recorded[0]), 'Check the deploy of staging.')

    def test_unknown_name_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = make_agent(Path(tmp)).run('/no-such-command')
        self.assertIn('Unknown skill: no-such-command', result.final_output)


if __name__ == '__main__':
    unittest.main()
