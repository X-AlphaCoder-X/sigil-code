"""3.2 permission rules (permissions.allow/ask/deny) and approval prompts."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.main import _append_agent_forwarded_args, build_parser, build_permission_prompt
from src.permission_rules import PermissionRules, parse_rule, rule_matches, suggest_rule


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


def tool_results(recorded: list[dict[str, object]]) -> list[str]:
    return [str(m.get('content')) for payload in recorded for m in payload['messages']
            if m.get('role') == 'tool']


def isolated_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ('SIGIL_BACKGROUND_ASK',)}
    env['SIGIL_DISABLE_USER_SETTINGS'] = '1'
    env.update(extra)
    return env


def write_settings(workspace: Path, permissions: dict[str, list[str]], *, local: bool = False) -> None:
    folder = workspace / '.sigil'
    folder.mkdir(exist_ok=True)
    name = 'settings.local.json' if local else 'settings.json'
    (folder / name).write_text(json.dumps({'permissions': permissions}), encoding='utf-8')


def make_agent(workspace: Path, *, write: bool = False, shell: bool = False) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=workspace,
            permissions=AgentPermissions(allow_file_write=write, allow_shell_commands=shell),
            session_directory=workspace / '.port_sessions' / 'agent',
            scratchpad_root=workspace / '.port_sessions' / 'scratchpad',
            nudge_strategy='off',
        ),
    )


class RuleMatchingTableTests(unittest.TestCase):
    def test_rule_matching_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / 'src').mkdir()
            cases = [
                ('Bash', 'bash', {'command': 'rm -rf /'}, True),
                ('Bash(git diff *)', 'bash', {'command': 'git diff HEAD'}, True),
                ('Bash(git diff *)', 'bash', {'command': 'git diff'}, True),
                ('Bash(git diff *)', 'bash', {'command': 'git status'}, False),
                ('Bash(ls *)', 'bash', {'command': 'lsof'}, False),
                ('Bash(npm run test:*)', 'bash', {'command': 'npm run test:unit'}, True),
                ('bash(echo *)', 'bash', {'command': 'echo hi'}, True),
                ('Read(./.env)', 'read_file', {'path': '.env'}, True),
                ('Read(./.env)', 'read_file', {'path': 'other.txt'}, False),
                ('Read(*.secret)', 'read_file', {'path': 'deep/dir/key.secret'}, True),
                ('Read', 'grep_search', {'pattern': 'x'}, True),
                ('Edit(src/**)', 'write_file', {'path': 'src/a/b.py'}, True),
                ('Edit(src/**)', 'edit_file', {'path': 'tests/a.py'}, False),
                ('Edit(/src/**)', 'notebook_edit', {'path': 'src/n.ipynb'}, True),
                ('Write', 'edit_file', {'path': 'x'}, False),
                ('WebFetch(domain:example.com)', 'web_fetch', {'url': 'https://docs.example.com/a'}, True),
                ('WebFetch(domain:example.com)', 'web_fetch', {'url': 'https://example.org/'}, False),
                ('mcp__github', 'mcp__github__create_issue', {}, True),
                ('mcp__github__create_issue', 'mcp__github__create_issue', {}, True),
                ('mcp__github', 'mcp__gitlab__x', {}, False),
            ]
            for raw, tool, args, expected in cases:
                rule = parse_rule(raw)
                self.assertIsNotNone(rule, raw)
                self.assertEqual(rule_matches(rule, tool, args, cwd=cwd), expected, (raw, tool, args))

    def test_compound_commands(self) -> None:
        rule = parse_rule('Bash(git *)')
        cwd = Path('.')
        self.assertFalse(rule_matches(rule, 'bash', {'command': 'git status && rm -rf x'}, cwd=cwd,
                                      for_allow=True))
        self.assertTrue(rule_matches(rule, 'bash', {'command': 'git status && git log'}, cwd=cwd,
                                     for_allow=True))
        deny = parse_rule('Bash(rm *)')
        self.assertTrue(rule_matches(deny, 'bash', {'command': 'ls && rm -rf x'}, cwd=cwd))

    def test_deny_beats_ask_beats_allow(self) -> None:
        rules = PermissionRules.from_settings({'permissions': {
            'allow': ['Bash'], 'ask': ['Bash(git push *)'], 'deny': ['Bash(git push --force *)']}})
        cwd = Path('.')
        self.assertEqual(rules.decide('bash', {'command': 'ls'}, cwd=cwd).behavior, 'allow')
        self.assertEqual(rules.decide('bash', {'command': 'git push origin'}, cwd=cwd).behavior, 'ask')
        self.assertEqual(rules.decide('bash', {'command': 'git push --force x'}, cwd=cwd).behavior, 'deny')
        self.assertIsNone(rules.decide('read_file', {'path': 'x'}, cwd=cwd))

    def test_bad_rules_are_reported(self) -> None:
        rules = PermissionRules.from_settings({'permissions': {'allow': ['Bash(', 42]}})
        self.assertEqual(len(rules.errors), 2)

    def test_suggested_rules_are_narrow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            self.assertEqual(suggest_rule('bash', {'command': 'npm test'}, cwd), 'Bash(npm test)')
            self.assertEqual(suggest_rule('write_file', {'path': 'a/b.txt'}, cwd), 'Write(./a/b.txt)')
            self.assertEqual(suggest_rule('web_fetch', {'url': 'https://x.dev/p'}, cwd), 'WebFetch(domain:x.dev)')


class AgentRuleTests(unittest.TestCase):
    def test_deny_rule_blocks_read(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('read_file', {'path': '.env'}), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated_env(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            (ws / '.env').write_text('SECRET=1', encoding='utf-8')
            write_settings(ws, {'deny': ['Read(./.env)']})
            make_agent(ws).run('read env')
        results = tool_results(recorded)
        self.assertTrue(any('Denied by permission rule Read(./.env)' in r for r in results), results)
        self.assertFalse(any('SECRET=1' in r for r in results))

    def test_allow_rule_runs_without_write_flag(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('write_file', {'path': 'out/a.txt', 'content': 'hi'}), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated_env(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            write_settings(ws, {'allow': ['Write(out/**)']})
            make_agent(ws).run('write it')
            self.assertEqual((ws / 'out' / 'a.txt').read_text(encoding='utf-8'), 'hi')

    def test_prompt_once_and_deny(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('write_file', {'path': 'one.txt', 'content': '1'}),
            tool_reply('write_file', {'path': 'two.txt', 'content': '2'}),
            text_reply('done'),
        ]
        answers = iter(['once', 'deny'])
        requests: list[dict[str, object]] = []

        def _prompt(request: dict[str, object]) -> str:
            requests.append(request)
            return next(answers)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated_env(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.permission_prompt = _prompt
            agent.run('write two files')
            self.assertTrue((ws / 'one.txt').exists())
            self.assertFalse((ws / 'two.txt').exists())
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]['rule'], 'Write(./one.txt)')
        self.assertTrue(any('The user denied this call' in r for r in tool_results(recorded)))

    def test_always_rule_is_saved_and_loads_next_session(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('write_file', {'path': 'keep.txt', 'content': 'first'}), text_reply('saved'),
            tool_reply('write_file', {'path': 'keep.txt', 'content': 'second'}), text_reply('saved again'),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated_env(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.permission_prompt = lambda _request: 'always'
            agent.run('save')
            local = json.loads((ws / '.sigil' / 'settings.local.json').read_text(encoding='utf-8'))
            self.assertIn('Write(./keep.txt)', local['permissions']['allow'])
            # A new session (no prompt available) uses the saved rule.
            make_agent(ws).run('save again')
            self.assertEqual((ws / 'keep.txt').read_text(encoding='utf-8'), 'second')

    def test_headless_ask_rule_blocks_unless_allowed(self) -> None:
        for env_value, expect_written in (('', False), ('allow', True)):
            recorded: list[dict[str, object]] = []
            responses = [tool_reply('write_file', {'path': 'f.txt', 'content': 'x'}), text_reply('ok')]
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, isolated_env(SIGIL_BACKGROUND_ASK=env_value), clear=True), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
                    patch('builtins.print'):
                ws = Path(tmp)
                write_settings(ws, {'ask': ['Write']})
                make_agent(ws, write=True).run('write')
                self.assertEqual((ws / 'f.txt').exists(), expect_written, env_value)
            if not expect_written:
                self.assertTrue(any('Blocked: needs approval' in r for r in tool_results(recorded)))

    def test_headless_without_rules_keeps_flag_refusal(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('write_file', {'path': 'f.txt', 'content': 'x'}), text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, isolated_env(SIGIL_BACKGROUND_ASK='allow'), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            make_agent(ws).run('write')
            self.assertFalse((ws / 'f.txt').exists())
        self.assertTrue(any('--allow-write' in r for r in tool_results(recorded)))

    def test_shell_guard_ask_goes_to_prompt(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('bash', {'command': 'echo approved-run'}), text_reply('ok')]
        requests: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, isolated_env(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp))  # shell is off: the prompt decides
            agent.permission_prompt = lambda request: (requests.append(request), 'once')[1]
            agent.run('run it')
        self.assertEqual(len(requests), 1)
        self.assertIn('Shell commands are disabled', str(requests[0]['reason']))
        self.assertTrue(any('approved-run' in r for r in tool_results(recorded)))


class PromptAndFlagTests(unittest.TestCase):
    def test_prompt_answers(self) -> None:
        request = {'tool': 'bash', 'arguments': {'command': 'ls'}, 'reason': 'r', 'rule': 'Bash(ls)'}
        for typed, expected in (('y', 'once'), ('a', 'always'), ('n', 'deny'), ('', 'deny')):
            prompt = build_permission_prompt(lambda _p, typed=typed: typed, output_func=lambda _t: None)
            self.assertEqual(prompt(request), expected)

    def test_permission_prompts_flag_is_forwarded(self) -> None:
        args = build_parser().parse_args(['agent-bg', 'x', '--permission-prompts', 'allow'])
        forwarded: list[str] = []
        _append_agent_forwarded_args(forwarded, args, include_backend=True)
        self.assertEqual(forwarded[forwarded.index('--permission-prompts') + 1], 'allow')


if __name__ == '__main__':
    unittest.main()
