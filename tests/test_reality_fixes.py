"""Fixes from the reality-check audit (2026-10-29). Each test failed before its fix;
model calls go through a fake urlopen (no network, no real server)."""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig

BASE = 'http://127.0.0.1:8000/v1'


class _Resp:
    def __init__(self, payload: dict) -> None:
        self.body = json.dumps(payload).encode('utf-8')

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def chat_reply(message: dict, usage: dict | None = None) -> dict:
    payload = {'choices': [{'message': message,
                            'finish_reason': 'tool_calls' if message.get('tool_calls') else 'stop'}]}
    if usage:
        payload['usage'] = usage
    return payload


def text(content: str) -> dict:
    return {'role': 'assistant', 'content': content}


def call(name: str, call_id: str = 'c1', **arguments) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class FakeModel:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001
        self.requests.append(json.loads(req.data.decode('utf-8')))
        item = self.replies.pop(0) if self.replies else chat_reply(text('done'))
        return _Resp(item)


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m', base_url=BASE),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms), session_directory=ws / '.s' / 'agent',
            scratchpad_root=ws / '.s' / 'scratch', nudge_strategy='off'))


def _remove_when_released(path: str) -> None:
    """A process killed on Windows lets go of its working folder a moment later."""
    import shutil

    for _ in range(50):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.2)
    shutil.rmtree(path, ignore_errors=True)


def _logger(name: str):
    """The Sigil logger `name` (by object, so the test does not depend on the logger's root name)."""
    from src.debug_log import get_logger

    return get_logger(name)


def tool_texts(request: dict) -> list[str]:
    return [str(m.get('content')) for m in request['messages'] if m.get('role') == 'tool']


class RateLimiterDefaultTests(unittest.TestCase):
    """F51: the limiter is off unless SIGIL_RPM_LIMIT / SIGIL_TPM_LIMIT are set."""

    def test_limits_from_env(self) -> None:
        from src.token_router import TokenRouter, limits_from_env

        self.assertEqual(limits_from_env({}), (0, 0))
        self.assertEqual(limits_from_env({'SIGIL_TPM_LIMIT': 'lots', 'SIGIL_RPM_LIMIT': '-3'}), (0, 0))
        self.assertEqual(limits_from_env({'SIGIL_TPM_LIMIT': '1000', 'SIGIL_RPM_LIMIT': '5'}), (5, 1000))
        router = TokenRouter(rpm_limit=0, tpm_limit=0)
        for _ in range(100):
            router.request_timestamps.append(time.time())
            router.record_usage(10 ** 7)
        self.assertEqual(router._calculate_wait(), 0.0)
        capped = TokenRouter(rpm_limit=0, tpm_limit=1000)
        capped.record_usage(5000)
        self.assertGreater(capped._calculate_wait(), 0.0)  # an explicit limit still applies

    def test_big_prompts_do_not_stall_the_agent_loop(self) -> None:
        from src.token_router import TokenRouter, limits_from_env

        previous = TokenRouter._instance
        self.addCleanup(setattr, TokenRouter, '_instance', previous)
        TokenRouter.configure(*limits_from_env({}))
        usage = {'prompt_tokens': 60000, 'completion_tokens': 10}
        model = FakeModel([chat_reply(call('list_dir', 'a', path='.'), usage),
                           chat_reply(call('list_dir', 'b', path='.'), usage),
                           chat_reply(call('list_dir', 'c', path='.'), usage),
                           chat_reply(text('finished'), usage)])
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch('src.token_router.time.sleep', side_effect=AssertionError('rate limiter waited')):
            result = make_agent(Path(tmp)).run('look around')
        self.assertEqual(result.final_output, 'finished')
        self.assertEqual(len(model.requests), 4)


class SessionLoadTests(unittest.TestCase):
    """F46: an unknown session id fails at once instead of after 5 s of retries."""

    def test_missing_session_fails_fast(self) -> None:
        from src.session_store import load_agent_session

        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            with self.assertRaises(OSError):
                load_agent_session('does-not-exist', directory=Path(tmp))
            self.assertLess(time.monotonic() - started, 1.0)


class ShredderTests(unittest.TestCase):
    """F47: outlines use byte offsets on bytes, so non-ASCII text before a name is fine."""

    def test_non_ascii_before_definition(self) -> None:
        from src import shredder

        source = '# café ééééééé\ndef hello(a, b):\n    pass\n'
        data = source.encode('utf-8')
        # the byte-offset slicing itself, on a stand-in parse tree (works without tree-sitter)

        class Node:
            def __init__(self, kind, start=0, end=0, children=(), **fields):  # noqa: ANN001, ANN003
                self.type, self.start_byte, self.end_byte = kind, start, end
                self.children, self._fields = list(children), fields

            def child_by_field_name(self, name):  # noqa: ANN001, ANN201
                return self._fields.get(name)

        name_at = data.index(b'hello')
        params_at = data.index(b'(a, b)')
        function = Node('function_definition', name=Node('identifier', name_at, name_at + 5),
                        parameters=Node('parameters', params_at, params_at + 6))
        tree = Node('module', children=[Node('comment'), function])
        self.assertEqual(shredder._extract_python_outline(tree, shredder._ByteText(data)), 'def hello(a, b): ...')
        if shredder.PY_LANGUAGE is not None:  # with tree-sitter: the real parser
            self.assertEqual(shredder.get_file_outline(source, '.py'), 'def hello(a, b): ...')
        else:  # without it: a short file comes back as it is (head/tail outline)
            self.assertEqual(shredder.get_file_outline(source, '.py'), source)


class SearchJsonTests(unittest.TestCase):
    """F45: a non-JSON reply from a search backend is a LookupError, not a crash."""

    def test_html_reply_is_a_lookup_error(self) -> None:
        from src.search_runtime import SearchRuntime

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / '.sigil-search.json').write_text(
                '{"providers":[{"name":"local","provider":"searxng","baseUrl":"http://127.0.0.1:8080"}]}',
                encoding='utf-8')
            runtime = SearchRuntime.from_workspace(ws)

            class _Html(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            with patch('src.search_runtime.request.urlopen', return_value=_Html(b'<html>proxy login</html>')):
                with self.assertRaises(LookupError):
                    runtime.search('x')
            with patch('src.search_runtime.request.urlopen', return_value=_Html(b'[1, 2]')):
                with self.assertRaises(LookupError):
                    runtime.search('x')


class PermissionRuleSplitTests(unittest.TestCase):
    """F37: an allow rule must cover every command on the line, including newline and '&'
    separated ones, and never matches command substitution."""

    def test_rule_matching(self) -> None:
        from src.permission_rules import parse_rule, rule_matches

        rule = parse_rule('Bash(git *)')
        cwd = Path('.')
        for command in ('git status\nrm -rf build', 'git status & rm -rf build', 'git log $(rm -rf build)',
                        'git log `rm -rf build`', 'git diff <(rm -rf build)', 'git status\r\nrm -rf build'):
            self.assertFalse(rule_matches(rule, 'bash', {'command': command}, cwd=cwd, for_allow=True), command)
        for command in ('git status', 'git status 2>&1', 'git log && git status', 'git status &> out.log'):
            self.assertTrue(rule_matches(rule, 'bash', {'command': command}, cwd=cwd, for_allow=True), command)
        deny = parse_rule('Bash(rm *)')
        self.assertTrue(rule_matches(deny, 'bash', {'command': 'ls\nrm -rf build'}, cwd=cwd))
        self.assertTrue(rule_matches(deny, 'bash', {'command': 'ls & rm -rf build'}, cwd=cwd))

    def test_newline_does_not_ride_on_an_allow_rule(self) -> None:
        model = FakeModel([chat_reply(call('bash', command='echo hi\ntouch pwned.txt')), chat_reply(text('ok'))])
        env = {k: v for k, v in os.environ.items() if k != 'SIGIL_BACKGROUND_ASK'}
        env['SIGIL_DISABLE_USER_SETTINGS'] = '1'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=model), patch('builtins.print'):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'permissions': {'allow': ['Bash(echo *)']}}),
                                                          encoding='utf-8')
            make_agent(ws).run('say hi')
            self.assertFalse((ws / 'pwned.txt').exists())


class SessionFolderDefaultTests(unittest.TestCase):
    """F01: a config built with only cwd keeps its sessions under that cwd, not the process cwd."""

    def test_defaults_follow_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            config = AgentRuntimeConfig(cwd=ws)
            self.assertEqual(config.session_directory, ws.resolve() / '.port_sessions' / 'agent')
            # the scratchpad keeps its old default (process folder; the CLI passes <cwd>/.port_sessions)
            self.assertEqual(config.scratchpad_root, (Path('.port_sessions') / 'scratchpad').resolve())
            explicit = AgentRuntimeConfig(cwd=ws, session_directory=ws / 'mine')
            self.assertEqual(explicit.session_directory, ws / 'mine')

    def test_agent_run_writes_inside_its_workspace(self) -> None:
        here = Path.cwd() / '.port_sessions' / 'agent'
        before = sorted(here.glob('*.json')) if here.is_dir() else []
        model = FakeModel([chat_reply(text('saved'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                     runtime_config=AgentRuntimeConfig(cwd=ws, nudge_strategy='off'))
            result = agent.run('hello')
            self.assertTrue((ws / '.port_sessions' / 'agent' / f'{result.session_id}.json').exists())
        after = sorted(here.glob('*.json')) if here.is_dir() else []
        self.assertEqual(before, after)


class AnthropicThinkingTests(unittest.TestCase):
    """F30: with the native Messages API, the assistant's thinking blocks (with signature) go back
    unchanged with the tool results (Anthropic API docs: thinking blocks go back within a tool-use turn)."""

    def run_loop(self, stream: bool) -> list[dict]:
        replies = [
            {'content': [{'type': 'thinking', 'thinking': 'plan: read it', 'signature': 'sig-abc'},
                         {'type': 'text', 'text': 'Reading.'},
                         {'type': 'tool_use', 'id': 'tu1', 'name': 'read_file', 'input': {'path': 'a.txt'}}],
             'stop_reason': 'tool_use', 'usage': {'input_tokens': 5, 'output_tokens': 3}},
            {'content': [{'type': 'redacted_thinking', 'data': 'opaque'},
                         {'type': 'tool_use', 'id': 'tu2', 'name': 'read_file', 'input': {'path': 'a.txt'}}],
             'stop_reason': 'tool_use', 'usage': {'input_tokens': 5, 'output_tokens': 3}},
            {'content': [{'type': 'text', 'text': 'It says hi.'}], 'stop_reason': 'end_turn',
             'usage': {'input_tokens': 5, 'output_tokens': 3}},
        ]
        sent: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            sent.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_ANTHROPIC_NATIVE': '1', 'SIGIL_EFFORT_LEVEL': ''}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake):
            ws = Path(tmp)
            (ws / 'a.txt').write_text('hi', encoding='utf-8')
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='claude-sonnet-5', base_url='https://api.anthropic.com/v1',
                                         api_key='fake-anthropic-key'),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s', nudge_strategy='off',
                                                  stream_model_responses=stream))
            result = agent.run('read a.txt')
        self.assertEqual(result.final_output, 'It says hi.')
        return sent

    def check(self, sent: list[dict]) -> None:
        assistants = [m for m in sent[2]['messages'] if m['role'] == 'assistant']
        self.assertEqual(assistants[0]['content'][0],
                         {'type': 'thinking', 'thinking': 'plan: read it', 'signature': 'sig-abc'})
        self.assertEqual([b['type'] for b in assistants[0]['content']], ['thinking', 'text', 'tool_use'])
        self.assertEqual(assistants[1]['content'][0], {'type': 'redacted_thinking', 'data': 'opaque'})
        self.assertNotIn("anthropic_content", json.dumps(sent))  # internal key never sent

    def test_blocks_go_back_with_tool_results(self) -> None:
        self.check(self.run_loop(stream=False))

    def test_blocks_go_back_when_streaming(self) -> None:
        self.check(self.run_loop(stream=True))

    def test_openai_compatible_payload_drops_the_internal_key(self) -> None:
        from src.openai_compat import OpenAICompatClient

        client = OpenAICompatClient(ModelConfig(model='m', base_url=BASE))
        payload = client._build_payload(messages=[
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'x', 'anthropic_content': [{'type': 'text', 'text': 'x'}]}],
            tools=[], stream=False, output_schema=None)
        self.assertNotIn('anthropic_content', json.dumps(payload))


class BareFlagsTests(unittest.TestCase):
    """F32: `sigil -w feature-x -n feature-x` (README) and other agent flags without a subcommand
    start the chat instead of exiting with an argparse error."""

    def capture(self, argv: list[str]) -> tuple[int, dict]:
        from src import main as main_module

        seen: dict = {}

        def fake_build(args):  # noqa: ANN001
            seen['args'] = args
            return object()

        def fake_loop(agent, *, initial_prompt, resume_session_id, show_transcript, **_kw):
            seen.update(prompt=initial_prompt, resume=resume_session_id)
            return 0

        with patch.object(main_module, '_build_agent', side_effect=fake_build), \
                patch.object(main_module, '_run_agent_chat_loop', side_effect=fake_loop):
            code = main_module.main(argv)
        return code, seen

    def test_readme_worktree_and_name(self) -> None:
        code, seen = self.capture(['-w', 'feature-x', '-n', 'feature-x'])
        self.assertEqual(code, 0)
        self.assertEqual((seen['args'].worktree, seen['args'].session_name), ('feature-x', 'feature-x'))

    def test_model_flag_prompt_and_continue(self) -> None:
        code, seen = self.capture(['--model', 'qwen', 'fix the bug'])
        self.assertEqual((code, seen['args'].model, seen['prompt']), (0, 'qwen', 'fix the bug'))
        code, seen = self.capture(['-c', '--model', 'qwen'])
        self.assertEqual((code, seen['resume'], seen['args'].model), (0, 'latest', 'qwen'))
        code, seen = self.capture(['--allow-write'])
        self.assertEqual(code, 0)
        self.assertTrue(seen['args'].allow_write)


class BashLookupTests(unittest.TestCase):
    """F16: no machine-specific fallback; Git Bash found the documented ways; WSL launcher skipped."""

    def resolve(self, env: dict, on_path: dict, existing: set) -> str:
        from src.agent_tools import _resolve_bash_executable

        return _resolve_bash_executable(env=env, which=lambda name: on_path.get(name),
                                        exists=lambda path: str(path) in existing, platform='win32')

    def test_order(self) -> None:
        git_bash = str(Path('D:/Tools/Git/bin/bash.exe'))
        own = str(Path('E:/b/bash.exe'))
        self.assertEqual(self.resolve({'SIGIL_GIT_BASH_PATH': own}, {}, {own}), own)
        wsl = {'bash': str(Path('C:/Windows/System32/bash.exe')), 'git': str(Path('D:/Tools/Git/cmd/git.exe'))}
        self.assertEqual(self.resolve({'SystemRoot': str(Path('C:/Windows'))}, wsl, {wsl['bash'], git_bash}),
                         git_bash)
        program_files = str(Path('C:/Program Files/Git/bin/bash.exe'))
        self.assertEqual(self.resolve({'ProgramFiles': str(Path('C:/Program Files'))}, {}, {program_files}),
                         program_files)

    def test_no_machine_specific_path(self) -> None:
        conda = str(Path('C:/ProgramData/miniconda3/Library/usr/bin/bash.exe'))
        self.assertEqual(self.resolve({}, {}, {conda}), '/bin/bash')  # none found: the PowerShell tool takes over


PNG = (b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
       b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82')


class ImageResumeTests(unittest.TestCase):
    """F13: an image message saved in a session is still an image after resume (not its base64 as text)."""

    def test_round_trip(self) -> None:
        from src.agent_session import AgentMessage

        url = 'data:image/png;base64,QUJD'
        message = AgentMessage(role='user', content='look at this', metadata={'images': [url]})
        restored = AgentMessage.from_openai_message(message.to_transcript_entry())
        self.assertEqual(restored.content, 'look at this')
        self.assertEqual(restored.to_openai_message(), message.to_openai_message())

    def test_resumed_session_sends_the_image_as_an_image(self) -> None:
        from src.session_store import load_agent_session

        model = FakeModel([chat_reply(text('a red pixel')), chat_reply(text('still red'))])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_VISION': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            (ws / 'shot.png').write_bytes(PNG)
            agent = make_agent(ws)
            self.assertEqual(agent.attach_images([str(ws / 'shot.png')]), [])
            first = agent.run('what is in the picture?')
            agent.resume('and now?', load_agent_session(first.session_id, directory=ws / '.s' / 'agent'))
        second = model.requests[1]['messages']
        self.assertFalse(any(isinstance(m.get('content'), str) and 'base64' in m['content'] for m in second))
        parts = [p for m in second if isinstance(m.get('content'), list) for p in m['content']]
        texts = [p.get('text', '') for p in parts if p.get('type') == 'text']
        self.assertFalse(any('base64' in t or "'image_url'" in t for t in texts), texts)
        self.assertTrue(any(p.get('type') == 'image_url' for p in parts))


class CwdChangeRegistryTests(unittest.TestCase):
    """F09: changing the working folder (/cd, worktrees) keeps the session's tool limits and extras."""

    def test_tools_survive_cd(self) -> None:
        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_SESSION_MEMORY': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            (ws / 'sub').mkdir()
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m', base_url=BASE),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s', nudge_strategy='off',
                                                  cli_disallowed_tools=('Bash',)))
            self.assertNotIn('bash', agent.tool_registry)
            self.assertIn('session_note', agent.tool_registry)
            self.assertIn('Working folder is now', agent.run('/cd sub').final_output)
            self.assertNotIn('bash', agent.tool_registry)  # the --disallowed-tools limit still holds
            self.assertIn('session_note', agent.tool_registry)
            agent.run('hi')
        names = {t['function']['name'] for t in model.requests[-1]['tools']}
        self.assertNotIn('bash', names)
        self.assertIn('session_note', names)

    def test_restricted_agent_gains_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'sub').mkdir()
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m', base_url=BASE),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s', nudge_strategy='off',
                                                  cli_tools=('Read', 'Grep')))
            before = set(agent.tool_registry)
            agent._apply_runtime_cwd_update(ws / 'sub')
            self.assertLessEqual(set(agent.tool_registry), before | {'session_note', 'memory_save'})
            self.assertNotIn('write_file', agent.tool_registry)


class ParallelGateTests(unittest.TestCase):
    """F05/F06: calls run ahead in parallel only when the normal loop would run them without
    a block or a question (policy, plugin, plan mode, rules, /read-only, budgets)."""

    def two_delegates(self, ws: Path, **runtime) -> list[dict]:
        sent: list[dict] = []

        def route(req, timeout=None):  # noqa: ANN001
            body = json.loads(req.data.decode('utf-8'))
            sent.append(body)
            users = json.dumps([m.get('content') for m in body['messages'] if m.get('role') == 'user'])
            if 'task A' in users or 'task B' in users:
                return _Resp(chat_reply(text('child ran')))
            if any(m.get('role') == 'tool' for m in body['messages']):
                return _Resp(chat_reply(text('parent done')))
            return _Resp({'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'a', 'type': 'function', 'function': {'name': 'delegate_agent',
                                                              'arguments': json.dumps({'prompt': 'task A'})}},
                {'id': 'b', 'type': 'function', 'function': {'name': 'delegate_agent',
                                                              'arguments': json.dumps({'prompt': 'task B'})}}]},
                'finish_reason': 'tool_calls'}]})

        with patch('src.openai_compat.request.urlopen', side_effect=route):
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m', base_url=BASE),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s', nudge_strategy='off',
                                                  **runtime))
            self.final = agent.run('split it').final_output
        return sent

    @staticmethod
    def child_calls(sent: list[dict]) -> int:
        count = 0
        for body in sent:
            users = json.dumps([m.get('content') for m in body['messages'] if m.get('role') == 'user'])
            if ('task A' in users or 'task B' in users) and 'split it' not in users:
                count += 1
        return count

    def test_policy_denied_delegates_never_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / '.sigil-policy.json').write_text('{"denyTools": ["delegate_agent"]}', encoding='utf-8')
            sent = self.two_delegates(ws)
        self.assertEqual(self.child_calls(sent), 0)

    def test_budget_limited_delegates_never_run_ahead(self) -> None:
        from src.agent_types import BudgetConfig

        with tempfile.TemporaryDirectory() as tmp:
            sent = self.two_delegates(Path(tmp), budget_config=BudgetConfig(max_delegated_tasks=1))
        self.assertLessEqual(self.child_calls(sent), 1)

    def test_allowed_delegates_still_run_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sent = self.two_delegates(Path(tmp))
        self.assertEqual(self.child_calls(sent), 2)
        self.assertEqual(self.final, 'parent done')


class ProjectEnvNoticeTests(unittest.TestCase):
    """F34: values a project's .env or .sigil/settings env fill in are named on stderr (never their
    values), so a cloned repo cannot quietly change the endpoint or the shell guard."""

    def test_names_are_shown(self) -> None:
        from contextlib import redirect_stderr, redirect_stdout

        from src.main import main

        names = ('OPENAI_BASE_URL', 'SIGIL_SHELL_GUARD', 'SIGIL_T_PROJECT', 'SIGIL_QUIET_PROJECT_ENV')
        env = {k: v for k, v in os.environ.items() if k not in names}
        model = FakeModel([chat_reply(text('hi'))])
        err, out = io.StringIO(), io.StringIO()

        class _Stdin(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=model), patch('sys.stdin', _Stdin('')), \
                redirect_stderr(err), redirect_stdout(out):
            ws = Path(tmp)
            (ws / '.env').write_text('OPENAI_BASE_URL=http://127.0.0.1:8000/v1\nSIGIL_SHELL_GUARD=off\n',
                                     encoding='utf-8')
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'env': {'SIGIL_T_PROJECT': 'secret-x'}}),
                                                          encoding='utf-8')
            self.assertEqual(main(['-p', 'hi', '--cwd', tmp]), 0)
        notice = err.getvalue()
        for name in ('OPENAI_BASE_URL', 'SIGIL_SHELL_GUARD', 'SIGIL_T_PROJECT'):
            self.assertIn(name, notice)
        self.assertNotIn('secret-x', notice)
        self.assertNotIn('127.0.0.1', notice)


class StatusLineTests(unittest.TestCase):
    """F48: the status line never blocks the prompt, a hung command and its children are ended,
    ~/ is expanded, and the payload has the model.id / model.display_name shape."""

    def test_hung_command_does_not_block_and_is_killed(self) -> None:
        import sys

        from src.background_runtime import _is_process_running
        from src.status_line import StatusLine

        tmp = tempfile.mkdtemp()
        self.addCleanup(_remove_when_released, tmp)
        if True:
            ws = Path(tmp)
            pid_file = ws / 'child.pid'
            script = ws / 'hang.py'
            script.write_text(
                'import subprocess, sys, time, pathlib\n'
                'child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])\n'
                f'pathlib.Path(r"{pid_file}").write_text(str(child.pid))\n'
                'time.sleep(60)\n', encoding='utf-8')
            line = StatusLine(f'"{sys.executable}" "{script}"', ws)
            started = time.monotonic()
            self.assertEqual(line.render({'model': {'id': 'm'}}), '')  # first call waits at most ~2 s
            self.assertLess(time.monotonic() - started, 4.0)
            started = time.monotonic()
            for _ in range(20):
                line.render({'model': {'id': 'm'}})
            self.assertLess(time.monotonic() - started, 0.5)  # later calls never wait
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and (not pid_file.exists() or
                                                   _is_process_running(int(pid_file.read_text() or 0))):
                time.sleep(0.2)
            self.assertTrue(pid_file.exists())
            self.assertFalse(_is_process_running(int(pid_file.read_text())), 'grandchild left running')

    def test_home_expansion_and_payload_shape(self) -> None:
        from types import SimpleNamespace

        from src.status_line import expand_home, status_payload

        self.assertEqual(expand_home('python ~/x.py --a ~'), f'python {Path.home()}/x.py --a ~')
        with tempfile.TemporaryDirectory() as tmp:
            agent = SimpleNamespace(active_session_id='s1', model_config=SimpleNamespace(model='qwen3'),
                                    runtime_config=SimpleNamespace(cwd=Path(tmp)), permission_mode='plan',
                                    cumulative_cost_usd=0.5, _session_name='feat')
            payload = status_payload(agent)
        self.assertEqual(payload['model'], {'id': 'qwen3', 'display_name': 'qwen3'})
        self.assertEqual(payload['workspace']['current_dir'], str(Path(tmp)))
        self.assertEqual(payload['cost']['total_cost_usd'], 0.5)
        self.assertEqual((payload['session_id'], payload['session_name']), ('s1', 'feat'))


class HookDocsTests(unittest.TestCase):
    """F55: HTTP hooks treat non-2xx as a non-blocking error unless onFailure is "block"
    (hooks reference); every hook gets transcript_path and permission_mode."""

    def serve(self, status: int, body: str) -> str:
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_address[1]}/hook'

    def test_non_2xx_is_non_blocking_unless_on_failure_block(self) -> None:
        from src.command_hooks import CommandHookRuntime

        url = self.serve(403, 'forbidden')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'}):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            settings = ws / '.sigil' / 'settings.json'
            settings.write_text(json.dumps({'hooks': {'PreToolUse': [{'hooks': [{'type': 'http', 'url': url}]}]}}),
                                encoding='utf-8')
            outcome = CommandHookRuntime.from_workspace(ws).run('PreToolUse', {}, tool_name='bash')
            self.assertFalse(outcome.blocked)
            self.assertEqual(len(outcome.errors), 1)
            settings.write_text(json.dumps({'hooks': {'PreToolUse': [{'hooks': [
                {'type': 'http', 'url': url, 'onFailure': 'block'}]}]}}), encoding='utf-8')
            self.assertTrue(CommandHookRuntime.from_workspace(ws).run('PreToolUse', {}, tool_name='bash').blocked)

    def test_common_input_fields(self) -> None:
        import sys

        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            out = ws / 'payload.json'
            script = ws / 'hook.py'
            script.write_text(f'import sys, pathlib\npathlib.Path(r"{out}").write_text(sys.stdin.read())\n',
                              encoding='utf-8')
            (ws / '.sigil').mkdir()
            command = f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}"'
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': {'UserPromptSubmit': [{'hooks': [
                {'type': 'command', 'command': command}]}]}}), encoding='utf-8')
            agent = make_agent(ws)
            agent.run('hello')
            payload = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(payload['permission_mode'], 'default')
        self.assertTrue(payload['transcript_path'].endswith(f"{payload['session_id']}.json"))


class SkillDescriptionTests(unittest.TestCase):
    """F56: a skill without `description` uses the first non-empty line of its body
    (skills reference) instead of being dropped without a word."""

    def test_fallback_description(self) -> None:
        from src.skills_runtime import SkillsRuntime

        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            folder = ws / '.sigil' / 'skills' / 'deploy'
            folder.mkdir(parents=True)
            (folder / 'SKILL.md').write_text('---\nname: deploy\n---\n\n# Deploy the app to staging\nSteps...\n',
                                             encoding='utf-8')
            runtime = SkillsRuntime.from_workspace(ws)
            self.assertIn('deploy', runtime.list_skills())
            self.assertEqual(runtime.get_skill('deploy').metadata.description, 'Deploy the app to staging')
            make_agent(ws).run('/deploy now')
        self.assertIn('Steps...', json.dumps(model.requests[0]['messages']))


class PolicyBudgetKeyTests(unittest.TestCase):
    """F26: the documented policy key maxToolCalls limits tool calls."""

    def test_max_tool_calls(self) -> None:
        model = FakeModel([chat_reply({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'list_dir', 'arguments': '{"path": "."}'}},
            {'id': 'b', 'type': 'function', 'function': {'name': 'list_dir', 'arguments': '{"path": "."}'}}]}),
            chat_reply(text('done'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            (ws / '.sigil-policy.json').write_text('{"budget": {"maxToolCalls": 1}}', encoding='utf-8')
            result = make_agent(ws).run('look twice')
        self.assertEqual(result.stop_reason, 'budget_exceeded')
        self.assertIn('tool-call budget', result.final_output)


class TestRunIsolationTests(unittest.TestCase):
    """F01/F52: the suite runs outside the repo and with a throw-away home folder."""

    def test_cwd_and_home_are_temporary(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        self.assertNotEqual(Path.cwd().resolve(), repo)
        home = Path.home().resolve()
        self.assertIn('sigil-test-home-', home.name)
        # temp folders and the working folder sit inside the throw-away home
        self.assertTrue(str(Path(tempfile.gettempdir()).resolve()).startswith(str(home)), tempfile.gettempdir())
        self.assertTrue(str(Path.cwd().resolve()).startswith(str(home)), Path.cwd())
        with tempfile.TemporaryDirectory() as tmp:
            from src.skills_runtime import SkillsRuntime

            self.assertEqual(SkillsRuntime.from_workspace(Path(tmp)).list_skills(), [])  # no real ~/.sigil


class SubagentRunTests(unittest.TestCase):
    """F07: parallel delegates in one turn get their own session ids. F08: an error outside the
    child's own try block is reported as a failed child, not lost."""

    def test_parallel_children_do_not_share_a_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            ParallelGateTests.two_delegates(self, ws)
            children = sorted(p.name for p in (ws / '.s').glob('*--sub--*.json'))
        self.assertEqual(len(children), 2, children)

    def test_error_outside_the_child_run_is_reported(self) -> None:
        model = FakeModel([chat_reply(call('delegate_agent', prompt='task A')), chat_reply(text('child ran')),
                           chat_reply(text('parent done'))])
        real_preview = LocalCodingAgent._preview_text
        raised: list[bool] = []

        def preview(self, value, limit=220):  # noqa: ANN001
            if str(value) == 'child ran' and not raised:  # the child's summary, in its thread
                raised.append(True)
                raise RuntimeError('summary broke')
            return real_preview(self, value, limit)

        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch.object(LocalCodingAgent, '_preview_text', preview):
            make_agent(Path(tmp)).run('split it')
        tool_reply = ' '.join(tool_texts(model.requests[-1]))
        self.assertIn('summary broke', tool_reply)


class BackgroundApprovalDirTests(unittest.TestCase):
    """F33: the chat answers approval questions from the folder the agent tool starts background
    agents in (<--cwd>/.port_sessions/background), not the process folder."""

    def test_same_folder_as_the_agent_tool(self) -> None:
        from src import main as sigil_main
        from src.agent_tools import _background_runtime_for

        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.run('hello')
            self.assertNotEqual(Path.cwd().resolve(), ws.resolve())
            self.assertEqual(sigil_main._background_dir_for(agent), _background_runtime_for(agent.tool_context).root)
            self.assertEqual(sigil_main._background_dir_for(agent), ws.resolve() / '.port_sessions' / 'background')


class SmallLimitsTests(unittest.TestCase):
    """F27: the image limit is on the base64 size. F35: -c / -r skip subagent and background
    sessions. F40: MCP stderr lines are capped."""

    def test_image_limit_counts_base64(self) -> None:
        from src.images import image_data_url

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.png'
            path.write_bytes(b'\x89PNG' + b'0' * 296)  # 300 bytes -> 400 base64 chars
            with patch.dict(os.environ, {'SIGIL_MAX_IMAGE_BYTES': '350'}):
                with self.assertRaises(ValueError):
                    image_data_url(path)
            with patch.dict(os.environ, {'SIGIL_MAX_IMAGE_BYTES': '400'}):
                self.assertTrue(image_data_url(path).startswith('data:image/png;base64,'))

    def test_latest_session_skips_children(self) -> None:
        from src.main import _lookup_latest_session_id

        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            result = make_agent(ws).run('hello')
            folder = ws / '.s' / 'agent'
            time.sleep(0.05)
            for name in (f'{result.session_id}--sub--1-abc123', 'bg_1234'):
                (folder / f'{name}.json').write_text('{}', encoding='utf-8')
            self.assertEqual(_lookup_latest_session_id(str(ws), session_dir=folder), result.session_id)

    def test_mcp_stderr_is_capped(self) -> None:
        from src.mcp_runtime import MCPServerProfile, _StdioMCPConnection

        connection = _StdioMCPConnection(MCPServerProfile(name='x', source_manifest='t', transport='stdio'))
        for number in range(5000):
            connection.stderr_lines.append(f'line {number}')
        self.assertLessEqual(len(connection.stderr_lines), 200)
        self.assertEqual(list(connection.stderr_lines)[-1], 'line 4999')


class GitTimeoutTests(unittest.TestCase):
    """F02: a git call that hits its 2 s timeout must not crash context building."""

    def test_slow_git_does_not_end_the_run(self) -> None:
        import subprocess

        real_run = subprocess.run

        def slow_git(command, *args, **kwargs):  # noqa: ANN001
            if isinstance(command, (list, tuple)) and command and command[0] == 'git':
                raise subprocess.TimeoutExpired(command, kwargs.get('timeout') or 2.0)
            return real_run(command, *args, **kwargs)

        model = FakeModel([chat_reply(text('fine'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch('src.agent_context.subprocess.run', side_effect=slow_git):
            result = make_agent(Path(tmp)).run('hello')
        self.assertEqual(result.final_output, 'fine')


class ToolCrashTests(unittest.TestCase):
    """F17: an unexpected exception inside a tool handler becomes a failed tool result."""

    def test_handler_error_goes_back_to_the_model(self) -> None:
        from src.agent_tools import SigilTool

        def broken(arguments, context):  # noqa: ANN001
            raise RuntimeError('server said no')

        model = FakeModel([chat_reply(call('mcp__x__boom')), chat_reply(text('recovered'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            agent = make_agent(Path(tmp))
            agent.tool_registry['mcp__x__boom'] = SigilTool(
                name='mcp__x__boom', description='boom', parameters={'type': 'object', 'properties': {}},
                handler=broken)
            result = agent.run('use the tool')
        self.assertEqual(result.final_output, 'recovered')
        self.assertTrue(any('server said no' in t for t in tool_texts(model.requests[1])), tool_texts(model.requests[1]))


class InvalidArgumentsTests(unittest.TestCase):
    """F38: cut-off tool arguments in a non-streamed reply go back to the model as a failed call
    (as the streaming path does) instead of ending the run."""

    def test_cut_off_arguments_are_retried(self) -> None:
        broken = {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'c1', 'type': 'function', 'function': {'name': 'write_file', 'arguments': '{"path": "a.txt", "con'}}]}
        model = FakeModel([chat_reply(broken), chat_reply(text('retried'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            result = make_agent(ws, allow_file_write=True).run('write a file')
            self.assertFalse((ws / 'a.txt').exists())
        self.assertEqual(result.final_output, 'retried')
        self.assertTrue(any('not valid JSON' in t for t in tool_texts(model.requests[1])), tool_texts(model.requests[1]))


_MCP_SERVER = r'''
import json, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
open(sys.argv[1], 'w').write(str(child.pid))
for line in sys.stdin:
    msg = json.loads(line)
    if 'id' in msg:
        result = {'protocolVersion': '2025-06-18', 'capabilities': {}, 'serverInfo': {'name': 'fake'}}
        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}) + '\n')
        sys.stdout.flush()
'''


def _pid_alive(pid: int) -> bool:
    import subprocess
    import sys

    if sys.platform == 'win32':
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/NH'], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class MCPCloseTreeTests(unittest.TestCase):
    """F39: closing a stdio MCP server also ends the processes it started (npx -> cmd -> node)."""

    def test_close_kills_grandchild(self) -> None:
        import sys

        from src.mcp_runtime import MCPServerProfile, _StdioMCPConnection

        tmp = tempfile.mkdtemp()
        self.addCleanup(_remove_when_released, tmp)
        script = Path(tmp) / 'server.py'
        script.write_text(_MCP_SERVER, encoding='utf-8')
        pid_file = Path(tmp) / 'child.pid'
        profile = MCPServerProfile(name='fake', source_manifest='test', transport='stdio',
                                   command=sys.executable, args=(str(script), str(pid_file)))
        connection = _StdioMCPConnection(profile, timeout_seconds=10.0).__enter__()
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            time.sleep(0.05)
        child = int(pid_file.read_text())
        self.assertTrue(_pid_alive(child))
        started = time.monotonic()
        connection.close()
        took = time.monotonic() - started
        # before the fix close() waited until the grandchild ended by itself (~60 s here)
        self.assertLess(took, 10.0, f'close() took {took:.1f} s')
        for _ in range(30):
            if not _pid_alive(child):
                break
            time.sleep(0.1)
        alive = _pid_alive(child)
        if alive:  # do not leave it behind
            import subprocess
            subprocess.run(['taskkill', '/PID', str(child), '/F'] if sys.platform == 'win32' else ['kill', str(child)],
                           capture_output=True)
        self.assertFalse(alive, 'grandchild outlived close()')


def _python_in_bash() -> str:
    import sys

    return '"' + Path(sys.executable).as_posix() + '"'


class BashStreamCapTests(unittest.TestCase):
    """F21: streamed bash output stops growing past SIGIL_BASH_STREAM_MAX_CHARS (head and tail
    kept, the rest drained), so a chatty command cannot fill memory or the session."""

    def test_chatty_command_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'gen.py').write_text("import sys\nfor i in range(20000):\n    sys.stdout.write('x' * 99 + '\\n')\n"
                                       "print('LAST LINE')\n", encoding='utf-8')
            model = FakeModel([chat_reply(call('bash', command=f'{_python_in_bash()} gen.py')),
                               chat_reply(text('done'))])
            with patch('src.openai_compat.request.urlopen', side_effect=model), \
                    patch.dict(os.environ, {'SIGIL_BASH_STREAM_MAX_CHARS': '5000'}):
                result = make_agent(ws, allow_shell_commands=True).run('run it')
        deltas = [e for e in result.events if e.get('type') == 'tool_delta']
        self.assertLess(len(deltas), 200, len(deltas))  # 20,001 lines before
        self.assertLess(sum(len(e.get('delta') or '') for e in deltas), 8000)
        reply = ' '.join(tool_texts(model.requests[1]))
        self.assertIn('LAST LINE', reply)  # the tail is kept
        self.assertIn('SIGIL_BASH_STREAM_MAX_CHARS', reply)

    def test_default_cap(self) -> None:
        from src.agent_tools import _bash_stream_cap

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SIGIL_BASH_STREAM_MAX_CHARS', None)
            self.assertEqual(_bash_stream_cap(12000), 120000)
            os.environ['SIGIL_BASH_STREAM_MAX_CHARS'] = '0'
            self.assertEqual(_bash_stream_cap(12000), 0)  # 0 = no cap


class StalePidTests(unittest.TestCase):
    """F22: a stale 'running' record whose PID now belongs to another process is not killed;
    the record is marked exited instead."""

    def test_kill_skips_a_reused_pid(self) -> None:
        import subprocess
        import sys
        from datetime import datetime, timedelta, timezone

        from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime

        victim = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        self.addCleanup(victim.wait)
        self.addCleanup(victim.kill)
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            root = ws / '.port_sessions' / 'background'
            runtime = BackgroundSessionRuntime(root)
            bid = 'bg_0123456789ab'
            runtime.save_record(BackgroundSessionRecord(
                background_id=bid, pid=victim.pid, prompt='p', workspace_cwd=str(ws), model='m', mode='agent',
                status='running', log_path=str(root / f'{bid}.log'), record_path=str(root / f'{bid}.json'),
                started_at=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(), command=()))
            model = FakeModel([chat_reply(call('background_agent_kill', background_id=bid)), chat_reply(text('ok'))])
            with patch('src.openai_compat.request.urlopen', side_effect=model):
                make_agent(ws, allow_shell_commands=True).run('stop it')
            time.sleep(0.3)
            self.assertIsNone(victim.poll(), 'an unrelated process with the reused PID was killed')
            self.assertEqual(runtime.load_record(bid).status, 'exited')
        self.assertIn('status=exited', ' '.join(tool_texts(model.requests[1])))

    def test_note_survives_refresh(self) -> None:
        """F23: why a worker stopped ('blocked: needs approval ...') is kept when it exits."""
        import subprocess
        import sys

        from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime

        gone = subprocess.Popen([sys.executable, '-c', 'pass'])
        gone.wait()
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            root = ws / '.port_sessions' / 'background'
            runtime = BackgroundSessionRuntime(root)
            bid = 'bg_0123456789cd'
            runtime.save_record(BackgroundSessionRecord(
                background_id=bid, pid=gone.pid, prompt='p', workspace_cwd=str(ws), model='m', mode='agent',
                status='running', log_path=str(root / f'{bid}.log'), record_path=str(root / f'{bid}.json'),
                started_at='2026-01-01T00:00:00+00:00', command=(), note='blocked: needs approval (Bash(git push *))'))
            model = FakeModel([chat_reply(call('background_agent_status', background_id=bid)), chat_reply(text('ok'))])
            with patch('src.openai_compat.request.urlopen', side_effect=model):
                make_agent(ws, allow_shell_commands=True).run('how is it')
            self.assertEqual(runtime.load_record(bid).note, 'blocked: needs approval (Bash(git push *))')
        self.assertIn('needs approval', ' '.join(tool_texts(model.requests[1])))

    def test_own_process_still_matches(self) -> None:
        import subprocess
        import sys

        from src.background_runtime import _process_start_time

        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])
        try:
            started = _process_start_time(child.pid)
            self.assertIsNotNone(started)
            self.assertLess(abs(started - time.time()), 30)
        finally:
            child.kill()
            child.wait()


class BackgroundShellCapTests(unittest.TestCase):
    """F24: a watchdog stops a background shell at SIGIL_BACKGROUND_SHELL_MAX_BYTES or
    SIGIL_BACKGROUND_SHELL_TIMEOUT even when nobody reads its output; reads stay bounded."""

    def setUp(self) -> None:
        from src import background_shells

        self.ws = Path(tempfile.mkdtemp())
        self.addCleanup(_remove_when_released, str(self.ws))  # runs last
        # a manager of its own: jobs and the watchdog of an earlier test cannot reach this one (S01)
        self.manager = background_shells.BackgroundShellManager()
        manager_patch = patch.object(background_shells, '_MANAGER', self.manager)
        manager_patch.start()
        self.addCleanup(manager_patch.stop)
        self.addCleanup(self.manager.kill_all)  # runs first: the log file is closed before removal

    def start(self, ws: Path, script: str, env: dict) -> str:
        (ws / 'job.py').write_text(script, encoding='utf-8')
        model = FakeModel([chat_reply(call('bash', command=f'{_python_in_bash()} job.py', run_in_background=True)),
                           chat_reply(text('started'))])
        with patch('src.openai_compat.request.urlopen', side_effect=model), patch.dict(os.environ, env):
            make_agent(ws, allow_shell_commands=True).run('start it')
        self.assertEqual(len(self.manager.jobs), 1, 'the bash tool started no background shell')
        job = list(self.manager.jobs.values())[0]
        deadline = time.monotonic() + 30  # stopping a process tree can take seconds on a loaded machine
        while job.exit_code is None and time.monotonic() < deadline:
            time.sleep(0.1)
        return job

    def test_size_cap_without_reads(self) -> None:
        if True:
            job = self.start(self.ws, "import sys, time\nwhile True:\n    sys.stdout.write('y' * 10000 + '\\n')\n"
                             "    sys.stdout.flush()\n    time.sleep(0.01)\n",
                             {'SIGIL_BACKGROUND_SHELL_MAX_BYTES': '200000', 'SIGIL_BACKGROUND_SHELL_CHECK_SECONDS': '0.2'})
            self.assertIsNotNone(job.exit_code, 'still running: no size cap')
            self.assertEqual(job.status, 'killed')
            self.assertIn('SIGIL_BACKGROUND_SHELL_MAX_BYTES', job.stop_reason or '')
            self.assertLess(job.output_path.stat().st_size, 200000 + 2_000_000)

    def test_timeout_without_reads(self) -> None:
        if True:
            job = self.start(self.ws, 'import time\ntime.sleep(60)\n',
                             {'SIGIL_BACKGROUND_SHELL_TIMEOUT': '1', 'SIGIL_BACKGROUND_SHELL_CHECK_SECONDS': '0.2'})
            self.assertEqual(job.status, 'killed')
            self.assertIn('SIGIL_BACKGROUND_SHELL_TIMEOUT', job.stop_reason or '')

    def test_job_object_failure_is_logged(self) -> None:
        """F25: opt-in Windows limits that cannot be applied are reported in the log."""
        import ctypes
        import sys

        if sys.platform != 'win32':
            self.skipTest('Windows Job Objects')
        from src.background_shells import shell_manager

        real = ctypes.WinDLL

        class Kernel32:
            def __init__(self, lib) -> None:  # noqa: ANN001
                self._lib = lib

            def __getattr__(self, name: str):
                if name == 'AssignProcessToJobObject':
                    return lambda *args: 0
                return getattr(self._lib, name)

        def fake(name, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            lib = real(name, *args, **kwargs)
            return Kernel32(lib) if name == 'kernel32' else lib

        with patch('ctypes.WinDLL', side_effect=fake), self.assertLogs(_logger('sandbox'), level='WARNING') as logs:
            shell_manager().start('exit 0', cwd=self.ws, env=None, bash='bash', output_dir=self.ws,
                                  argv=[sys.executable, '-c', 'pass'], limits={'max_processes': 4})
        self.assertTrue(any('without its limits' in line for line in logs.output), logs.output)

    def test_read_new_reads_only_the_tail(self) -> None:
        from src.background_shells import shell_manager

        if True:
            job = self.start(self.ws, "print('z' * 3_000_000)\nprint('END')\n", {})
            job, text = shell_manager().read_new(job.job_id, max_chars=1000)
        self.assertIn('END', text)
        self.assertLess(len(text), 1300)
        self.assertIn('bytes skipped', text)


class ProjectFolderTests(unittest.TestCase):
    """F03 / F29 / F42: lookups without a cwd argument use the agent's folder, not the process
    folder; extraHeaders settings are read again only when a settings file changes."""

    def test_user_rule_paths_match_the_session_folder(self) -> None:
        rules = Path.home() / '.sigil' / 'rules'
        rules.mkdir(parents=True, exist_ok=True)
        rule = rules / 'python-only.md'
        rule.write_text("---\npaths: ['*.py']\n---\nPY RULE TEXT\n", encoding='utf-8')
        self.addCleanup(rule.unlink)
        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            ws = Path(tmp)
            (ws / 'a.py').write_text('x = 1\n', encoding='utf-8')
            self.assertFalse(any(Path.cwd().glob('*.py')))
            make_agent(ws).run('hello')
        self.assertIn('PY RULE TEXT', json.dumps(model.requests[0]))

    def test_extra_headers_come_from_the_agent_folder(self) -> None:
        seen: list = []

        def fake(req, timeout=None):  # noqa: ANN001
            seen.append(req.get_header('X-project'))
            return _Resp(chat_reply(text('ok')))

        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=fake):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'extraHeaders': {'X-Project': 'here'}}),
                                                         encoding='utf-8')
            make_agent(ws).run('hello')
        self.assertEqual(seen[0], 'here')

    def test_extra_headers_settings_are_cached(self) -> None:
        from src import settings as settings_module
        from src.openai_compat import extra_headers

        calls: list = []
        real = settings_module.load_settings

        def counting(*args, **kwargs):  # noqa: ANN002, ANN003
            calls.append(args)
            return real(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, settings_module.project_dir_scope(tmp):
            with patch.object(settings_module, 'load_settings', counting):
                for _ in range(5):
                    extra_headers()
        self.assertEqual(len(calls), 1, calls)

    def test_agent_folder_does_not_outlive_the_agent(self) -> None:
        from src.settings import project_dir

        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            make_agent(Path(tmp)).run('hello')
        self.assertEqual(project_dir(), Path.cwd())  # plugin ${SIGIL_PROJECT_DIR} outside an agent


class QuietFailureTests(unittest.TestCase):
    """F04: a failing repo map is logged. F10: a broken settings file is reported, not silently
    dropped (with its deny rules). F11: an "always" rule that cannot be saved is reported and
    kept for the session. F12: the Ctrl+C backup is not written into the project."""

    def test_repo_map_failure_is_logged(self) -> None:
        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch.dict(os.environ, {'SIGIL_REPO_MAP': '1'}), \
                patch('src.repo_map.build_repo_map', side_effect=RuntimeError('map broke')), \
                self.assertLogs(_logger('context'), level='WARNING') as logs:
            make_agent(Path(tmp)).run('hello')
        self.assertTrue(any('repo map failed' in line for line in logs.output), logs.output)

    def test_broken_settings_file_is_reported(self) -> None:
        model = FakeModel([chat_reply(text('ok'))])
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch('sys.stderr', err):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text('{"permissions": {"deny": ["Bash"]},', encoding='utf-8')
            make_agent(ws).run('hello')
        self.assertIn('settings.json is ignored', err.getvalue())

    def test_unsaved_always_rule_is_reported_and_kept(self) -> None:
        # the same command twice: the offered rule is for that command (Bash(echo one))
        model = FakeModel([chat_reply(call('bash', 'c1', command='echo one')),
                           chat_reply(call('bash', 'c2', command='echo one')), chat_reply(text('done'))])
        asked: list = []
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch('src.settings.add_local_permission_rule', side_effect=OSError('read-only disk')), \
                patch('sys.stderr', err):
            ws = Path(tmp)
            # shell is not allowed up front, so the first call asks (an explicit `ask` rule would
            # ask every time, ahead of allow rules)
            agent = make_agent(ws)
            agent.permission_prompt = lambda request: asked.append(request) or 'always'
            agent.run('echo twice')
        self.assertEqual(len(asked), 1, 'asked again: the rule was lost')
        self.assertIn('Could not save', err.getvalue())

    def test_interrupt_backup_stays_out_of_the_project(self) -> None:
        from src.main import _lookup_latest_session_id

        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=KeyboardInterrupt):
            ws = Path(tmp)
            try:
                make_agent(ws).run('hello')
            except KeyboardInterrupt:
                pass
            self.assertFalse((ws / 'session_backup.json').exists(), 'backup written into the project')
            self.assertTrue((ws / '.s' / 'session_backup.json').exists())
            self.assertTrue(_lookup_latest_session_id(str(ws), is_backup=True, session_dir=ws / '.s' / 'agent'))


class SmallP3Tests(unittest.TestCase):
    """F14 key help, F15 release notes when installed, F18 read_file schema, F19 web_fetch prompt
    failure, F31 log rollover, F36 background CLI --cwd, F41 MCP rate-limit state, F43 profile key,
    F49 display buffer, F50 skill paths scan cap."""

    def test_key_help_lists_the_keys(self) -> None:
        from src.agent_slash_commands import KEY_HELP

        for key in ('Ctrl+S', 'Ctrl+G', 'Ctrl+O', 'Ctrl+T', 'Alt+P', 'Alt+T', 'Alt+V', 'Ctrl+Enter', '?'):
            self.assertIn(key, KEY_HELP)

    def test_release_notes_without_a_local_changelog(self) -> None:
        from src import agent_slash_commands

        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(Path(tmp))
            with patch.object(agent_slash_commands, '_changelog_candidates', return_value=[Path(tmp) / 'none.md']):
                result = agent_slash_commands._handle_release_notes(agent, '', '/release-notes')
        self.assertIn('github.com', str(getattr(result, 'output', result)))

    def test_read_file_schema_names_images_pdfs_and_pages(self) -> None:
        model = FakeModel([chat_reply(text('ok'))])
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model):
            make_agent(Path(tmp)).run('hello')
        spec = next(t['function'] for t in model.requests[0]['tools'] if t['function']['name'] == 'read_file')
        self.assertIn('PDF', spec['description'])
        self.assertIn('image', spec['description'])
        self.assertIn('offset', spec['parameters']['properties'])
        self.assertIn('limit', spec['parameters']['properties'])

    def test_web_fetch_prompt_failure_is_said(self) -> None:
        model = FakeModel([chat_reply(call('web_fetch', url='https://example.com/a', prompt='what is it')),
                           chat_reply(text('ok'))])
        page = {'text': 'PAGE BODY TEXT', 'content_type': 'text/html', 'final_url': 'https://example.com/a'}
        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=model), \
                patch('src.agent_tools._web_fetch_cached', return_value=page), \
                patch('src.anthropic_client.fast_client', side_effect=RuntimeError('side model down')), \
                self.assertLogs(_logger('tools'), level='WARNING'):
            make_agent(Path(tmp)).run('fetch it')
        reply = ' '.join(tool_texts(model.requests[1]))
        self.assertIn('PAGE BODY TEXT', reply)
        self.assertIn('side model down', reply)

    def test_log_rollover_failure_is_quiet(self) -> None:
        import logging

        from src.debug_log import _RotatingFile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shared.log'
            handler = _RotatingFile(path, maxBytes=200, backupCount=2, encoding='utf-8', delay=True)
            log = logging.getLogger('sigil-test-rollover')
            log.addHandler(handler)
            log.propagate = False
            err = io.StringIO()
            try:
                with patch('os.rename', side_effect=PermissionError('in use by another process')), \
                        patch('sys.stderr', err):
                    for number in range(30):
                        log.warning('line %d %s', number, 'x' * 40)
            finally:
                log.removeHandler(handler)
                handler.close()
            written = path.read_text(encoding='utf-8')
        self.assertNotIn('Logging error', err.getvalue())
        self.assertIn('line 29', written)

    def test_background_cli_takes_cwd(self) -> None:
        import contextlib
        import subprocess
        import sys

        from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime
        from src.main import main

        gone = subprocess.Popen([sys.executable, '-c', 'pass'])
        gone.wait()
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            root = ws / '.port_sessions' / 'background'
            bid = 'bg_00000000cafe'
            BackgroundSessionRuntime(root).save_record(BackgroundSessionRecord(
                background_id=bid, pid=gone.pid, prompt='p', workspace_cwd=str(ws), model='m', mode='agent',
                status='completed', log_path=str(root / f'{bid}.log'), record_path=str(root / f'{bid}.json'),
                started_at='2026-01-01T00:00:00+00:00', command=()))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(['agent-ps', '--cwd', str(ws)])
        self.assertEqual(code, 0)
        self.assertIn(bid, out.getvalue())

    def test_mcp_rate_limit_state_is_pruned_and_outside_the_package(self) -> None:
        from src import mcp_runtime

        self.assertFalse(str(mcp_runtime._RATE_LIMIT_STATE_FILE).startswith(str(Path(mcp_runtime.__file__).parent)))
        profile = mcp_runtime.MCPServerProfile(name='limited-test', source_manifest='t', transport='stdio',
                                               rate_limits={'requestsPerMonth': 1000})
        old = time.time() - 90 * 86400
        with mcp_runtime._rate_limit_lock:
            mcp_runtime._rate_limit_state['limited-test'] = {
                'hourly_requests': [old] * 500, 'monthly_requests': [old] * 5000, 'current_key_index': 0}
        try:
            mcp_runtime._record_rate_limit_usage(profile)
            state = mcp_runtime._rate_limit_state['limited-test']
            self.assertEqual(len(state['monthly_requests']), 1)
            self.assertEqual(len(state['hourly_requests']), 1)
        finally:
            mcp_runtime._rate_limit_state.pop('limited-test', None)

    def test_profile_on_another_host_does_not_get_the_key(self) -> None:
        from src.profiles import profile_config

        config = ModelConfig(model='m', base_url='https://api.one.example/v1', api_key='fake-key-one')
        other = profile_config(config, {'base_url': 'https://api.two.example/v1', 'model': 'x'})
        self.assertNotEqual(other.api_key, 'fake-key-one')
        same_host = profile_config(config, {'base_url': 'https://api.one.example/v2', 'model': 'y'})
        self.assertEqual(same_host.api_key, 'fake-key-one')
        with patch.dict(os.environ, {'TWO_KEY': 'fake-key-two'}):
            named = profile_config(config, {'base_url': 'https://api.two.example/v1', 'api_key_env': 'TWO_KEY'})
        self.assertEqual(named.api_key, 'fake-key-two')

    def test_display_does_not_keep_all_streamed_text(self) -> None:
        from src.sigil_display import SigilDisplay

        display = SigilDisplay(out=io.StringIO())
        for _ in range(2000):
            display.on_event({'type': 'content_delta', 'delta': 'x' * 500})
        buffer = getattr(display, '_content_buffer', '')
        self.assertLess(len(buffer), 10000)

    def test_skill_paths_scan_is_capped(self) -> None:
        from src.skills_runtime import _any_match

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            for number in range(300):
                (ws / f'f{number:03}.txt').write_text('', encoding='utf-8')
            (ws / 'zz').mkdir()
            (ws / 'zz' / 'late.py').write_text('', encoding='utf-8')
            self.assertTrue(_any_match(ws, '**/*.py'))
            self.assertTrue(_any_match(ws, 'zz/*.py'))
            self.assertFalse(_any_match(ws, '*.py'))
            with patch.dict(os.environ, {'SIGIL_SKILL_PATHS_SCAN_LIMIT': '50'}):
                self.assertFalse(_any_match(ws, '**/*.py'))  # stopped after 50 entries


class IgnoredErrorLogTests(unittest.TestCase):
    """Security rules, Rule 12: an error that is ignored on purpose is still written to the
    always-on log file (INFO): at once the first time, then at most one line a minute per place
    with the number of errors since, so every error shows up within a minute."""

    def test_ignored_errors_are_logged_and_rate_limited(self) -> None:
        from src import debug_log
        from src.debug_log import get_logger, log_ignored_error, sanitize_log

        clock = [1000.0]

        def fail() -> None:
            try:
                raise OSError('pipe closed\nFAKE LOG LINE')
            except OSError as exc:
                log_ignored_error(exc)

        with patch.object(debug_log.time, 'monotonic', lambda: clock[0]), \
                self.assertLogs(get_logger('ignored'), level='INFO') as logs:
            for _ in range(250):
                fail()  # same place, same minute: one line
            clock[0] += 61
            fail()  # a minute later: one line with the count
        self.assertEqual(len(logs.records), 2, logs.output)
        self.assertIn('test_reality_fixes.py', logs.output[0])
        self.assertIn('250 times since the last line', logs.output[1])
        self.assertNotIn('\n', logs.records[0].getMessage())  # no forged log lines (Rule 1)
        self.assertEqual(sanitize_log('a\r\nb\tc'), 'a  b c')


class LogRetentionTests(unittest.TestCase):
    """Logs are kept 2 days by default (SIGIL_LOG_RETENTION_DAYS), and a long-running session
    removes old ones too, not only at start."""

    def test_two_day_default_and_cleanup_while_running(self) -> None:
        import logging

        from src import debug_log

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SIGIL_LOG_RETENTION_DAYS', None)
            folder = Path(tmp)
            day_and_half = folder / 'sigil-2026-10-27.log'
            three_days = folder / 'sigil-2026-10-26.log'
            for path, age in ((day_and_half, 1.5), (three_days, 2.5)):
                path.write_text('x', encoding='utf-8')
                os.utime(path, (time.time() - age * 86400,) * 2)
            self.assertEqual(debug_log.remove_old_logs(folder), 1)  # 2.5 days old: gone
            self.assertTrue(day_and_half.exists())
            # while running: the handler cleans up again (hourly) without a restart
            os.utime(day_and_half, (time.time() - 3 * 86400,) * 2)
            handler = debug_log._RotatingFile(folder / 'sigil-2026-10-29.log', maxBytes=10 ** 6,
                                              backupCount=1, encoding='utf-8', delay=True)
            handler._next_cleanup = 0.0
            log = logging.getLogger('sigil-test-retention')
            log.addHandler(handler)
            log.propagate = False
            try:
                log.warning('hello')
            finally:
                log.removeHandler(handler)
                handler.close()
            self.assertFalse(day_and_half.exists())


class KeyLeakTests(unittest.TestCase):
    """F59: a `model@url` on another host (fallback model, fastModel, editor model) used to get the
    main API key. F60: HTTP hook headers resolved every environment variable; the docs resolve
    only the ones in `allowedEnvVars`."""

    def test_fallback_model_on_another_host_does_not_get_the_main_key(self) -> None:
        seen: list[tuple[str, str]] = []

        def fake(req, timeout=None):  # noqa: ANN001
            seen.append((req.full_url, req.get_header('Authorization')))
            if '127.0.0.1:8000' in req.full_url:
                from urllib.error import HTTPError

                raise HTTPError(req.full_url, 500, 'down', {}, io.BytesIO(b'down'))
            return _Resp(chat_reply(text('from fallback')))

        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=fake), \
                patch.dict(os.environ, {'SIGIL_RETRY_MAX_ATTEMPTS': '1'}):
            ws = Path(tmp)
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m', base_url=BASE, api_key='main-provider-key',
                                         fallback_models=('other@https://other.example/v1',)),
                runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s', nudge_strategy='off'))
            agent.run('hello')
        other = [auth for url, auth in seen if 'other.example' in url]
        self.assertTrue(other, seen)
        self.assertNotIn('main-provider-key', ' '.join(auth or '' for auth in other))

    def test_endpoint_key_from_host_variable_and_same_host(self) -> None:
        from src.profiles import config_for_endpoint

        config = ModelConfig(model='m', base_url='https://api.one.example/v1', api_key='key-one')
        self.assertEqual(config_for_endpoint(config, 'x', 'https://api.one.example/v2').api_key, 'key-one')
        self.assertNotEqual(config_for_endpoint(config, 'x', 'https://other.example/v1').api_key, 'key-one')
        with patch.dict(os.environ, {'SIGIL_API_KEY_OTHER_EXAMPLE': 'key-other'}):
            self.assertEqual(config_for_endpoint(config, 'x', 'https://other.example/v1').api_key, 'key-other')
        from src.architect import editor_model_config

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_EDITOR_MODEL': 'ed@https://other.example/v1'}):
            self.assertNotEqual(editor_model_config(config, Path(tmp)).api_key, 'key-one')

    def test_hook_headers_resolve_only_allowed_env_vars(self) -> None:
        import http.server
        import threading

        from src.command_hooks import CommandHookRuntime

        received: dict = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                received['leak'] = self.headers.get('X-Leak')
                received['ok'] = self.headers.get('Authorization')
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_address[1]}/hook'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
                'OPENAI_API_KEY': 'main-provider-key', 'HOOK_TOKEN': 'hook-secret', 'SIGIL_DISABLE_USER_HOOKS': '1'}):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': {'PreToolUse': [{'hooks': [{
                'type': 'http', 'url': url, 'allowedEnvVars': ['HOOK_TOKEN'],
                'headers': {'Authorization': 'Bearer $HOOK_TOKEN', 'X-Leak': '${OPENAI_API_KEY}'}}]}]}}),
                encoding='utf-8')
            CommandHookRuntime.from_workspace(ws).run('PreToolUse', {}, tool_name='bash')
        self.assertEqual(received.get('ok'), 'Bearer hook-secret')
        self.assertFalse(received.get('leak'), received)


class RaceTests(unittest.TestCase):
    """Shared caches used from several threads (parallel subagents, reader threads) and JSON
    state files read while another process writes them."""

    def setUp(self) -> None:
        import sys

        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # switch threads very often: makes interleavings likely
        self.addCleanup(sys.setswitchinterval, previous)

    def hammer(self, work, threads: int = 8) -> list:
        import threading

        errors: list = []

        def run(index: int) -> None:
            try:
                work(index)
            except Exception as exc:  # noqa: BLE001 - the test collects every failure
                errors.append(repr(exc))

        pool = [threading.Thread(target=run, args=(i,)) for i in range(threads)]
        for thread in pool:
            thread.start()
        for thread in pool:
            thread.join(60)
        return errors

    def test_ignored_error_log_is_thread_safe(self) -> None:
        from src.debug_log import log_ignored_error

        def work(index: int) -> None:
            for number in range(4000):
                exec(compile('\n' * (number % 700) + 'log_ignored_error(OSError("x"))', f'place{index}', 'exec'),
                     {'log_ignored_error': log_ignored_error})

        self.assertEqual(self.hammer(work), [])

    def test_extra_headers_cache_is_thread_safe(self) -> None:
        from src import settings as settings_module
        from src.openai_compat import _configured_extra_headers

        folders = [tempfile.mkdtemp() for _ in range(80)]
        for folder in folders:
            self.addCleanup(_remove_when_released, folder)

        def work(index: int) -> None:
            for number in range(300):
                with settings_module.project_dir_scope(folders[(index * 7 + number) % len(folders)]):
                    _configured_extra_headers()

        self.assertEqual(self.hammer(work), [])

    def test_background_record_is_never_read_half_written(self) -> None:
        import threading

        from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime

        with tempfile.TemporaryDirectory() as tmp:
            runtime = BackgroundSessionRuntime(Path(tmp))
            bid = 'bg_0123456789ef'
            record = BackgroundSessionRecord(
                background_id=bid, pid=0, prompt='p' * 20000, workspace_cwd=tmp, model='m', mode='agent',
                status='completed', log_path=str(Path(tmp) / f'{bid}.log'), record_path=str(runtime.record_path(bid)),
                started_at='2026-01-01T00:00:00+00:00', command=())
            runtime.save_record(record)
            stop = threading.Event()
            errors: list = []

            def writer() -> None:
                while not stop.is_set():
                    runtime.save_record(record)

            thread = threading.Thread(target=writer)
            thread.start()
            try:
                for _ in range(1500):
                    try:
                        runtime.load_record(bid)
                    except (ValueError, OSError) as exc:  # JSONDecodeError is a ValueError
                        errors.append(repr(exc))
            finally:
                stop.set()
                thread.join(30)
        self.assertEqual(errors[:3], [])

    def test_subagent_limit_holds_under_parallel_delegation(self) -> None:
        """Check-then-increment of the spawn count (SIGIL_MAX_SUBAGENTS_PER_SESSION) must be atomic."""
        started: list = []

        def route(req, timeout=None):  # noqa: ANN001
            body = json.loads(req.data.decode('utf-8'))
            if any('child task' in str(m.get('content')) for m in body['messages'] if m.get('role') == 'user'):
                started.append(1)
                time.sleep(0.05)
            return _Resp(chat_reply(text('ok')))

        with tempfile.TemporaryDirectory() as tmp, patch('src.openai_compat.request.urlopen', side_effect=route), \
                patch.dict(os.environ, {'SIGIL_MAX_SUBAGENTS_PER_SESSION': '2'}):
            agent = make_agent(Path(tmp))
            errors = self.hammer(lambda index: agent._execute_delegate_agent({'prompt': f'child task {index}'}),
                                 threads=6)
        self.assertEqual(errors, [])
        self.assertLessEqual(len(started), 2, f'{len(started)} children started with a limit of 2')
        self.assertLessEqual(agent.subagent_spawn_count, 2)

    def test_web_fetch_cache_is_bounded(self) -> None:
        from src import agent_tools

        def fake_fetch(url, timeout):  # noqa: ANN001
            return {'text': 'x' * 1000, 'content_type': 'text/html', 'final_url': url}

        agent_tools._WEB_FETCH_CACHE.clear()
        with patch.object(agent_tools, '_web_fetch_raw', side_effect=fake_fetch):
            for number in range(300):
                agent_tools._web_fetch_cached(f'https://example.com/{number}', 5.0)
        self.assertLessEqual(len(agent_tools._WEB_FETCH_CACHE), 64)
        agent_tools._WEB_FETCH_CACHE.clear()


class OwnSurfaceTests(unittest.TestCase):
    """The command and tool lists are Sigil's own (no snapshot of another program's files)."""

    def test_chat_loop_commands_exist_in_the_chat_loop(self) -> None:
        from src.commands import CHAT_LOOP_COMMANDS

        source = (Path(__file__).resolve().parents[1] / 'src' / 'main.py').read_text(encoding='utf-8')
        for name in CHAT_LOOP_COMMANDS:
            self.assertIn(f"'/{name}", source, name)

    def test_no_upstream_source_paths_in_the_surfaces(self) -> None:
        from src.commands import PORTED_COMMANDS
        from src.tools import PORTED_TOOLS

        for module in (*PORTED_COMMANDS, *PORTED_TOOLS):
            self.assertTrue(module.source_hint.startswith('src/') and module.source_hint.endswith('.py'),
                            module)
        # the loaders read no snapshot file any more
        src = Path(__file__).resolve().parents[1] / 'src'
        for name in ('commands.py', 'tools.py', 'parity_audit.py'):
            self.assertNotIn('reference_data', (src / name).read_text(encoding='utf-8'), name)


class WorkspaceTrustTests(unittest.TestCase):
    """F61: a folder's project hooks, MCP servers, status line and env need the user's trust
    (asked once per folder, kept in ~/.sigil, asked again when they change); SIGIL_TRUST_ALL=1
    turns it off; headless runs name what they skipped; user-level items are not affected."""

    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('SIGIL_TRUST_ALL', None)  # conftest turns trust off for the other tests
        self.ws = Path(tempfile.mkdtemp())
        self.addCleanup(_remove_when_released, str(self.ws))
        self.marker = self.ws / 'hook-ran.txt'
        script = self.ws / 'hook.py'
        script.write_text(f'import pathlib\npathlib.Path(r"{self.marker}").write_text("ran")\n', encoding='utf-8')
        import sys

        self.command = f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}"'
        (self.ws / '.sigil').mkdir()
        self.write_hooks(self.command)

    def write_hooks(self, command: str) -> None:
        (self.ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': {'UserPromptSubmit': [
            {'hooks': [{'type': 'command', 'command': command}]}]}}), encoding='utf-8')

    def run_agent(self) -> str:
        err = io.StringIO()
        model = FakeModel([chat_reply(text('ok'))])
        with patch('src.openai_compat.request.urlopen', side_effect=model), patch('sys.stderr', err):
            make_agent(self.ws).run('hello')
        return err.getvalue()

    def test_untrusted_project_hook_does_not_run_and_is_named(self) -> None:
        err = self.run_agent()
        self.assertFalse(self.marker.exists(), 'an untrusted project hook ran')
        self.assertIn('hook.py', err)  # headless: the skipped item is named
        self.assertIn('trust grant', err)  # how to trust it (the command name differs per build)

    def test_trusted_hook_runs(self) -> None:
        from src.workspace_trust import ensure_trust

        asked: list = []
        self.assertTrue(ensure_trust(self.ws, ask=lambda prompt: asked.append(prompt) or 't', output=lambda s: None))
        self.run_agent()
        self.assertEqual(len(asked), 1)
        self.assertTrue(self.marker.exists())

    def test_changed_items_ask_again(self) -> None:
        from src.workspace_trust import ensure_trust, status

        ensure_trust(self.ws, ask=lambda prompt: 't', output=lambda s: None)
        self.assertEqual(status(self.ws)[0], 'trusted')
        self.write_hooks(self.command + ' --changed')
        self.assertEqual(status(self.ws)[0], 'changed')
        asked: list = []
        ensure_trust(self.ws, ask=lambda prompt: asked.append(prompt) or 's', output=lambda s: None)
        self.assertEqual(len(asked), 1)
        self.run_agent()
        self.assertFalse(self.marker.exists())  # skipped after the change

    def test_trust_all_switch_skips_the_question(self) -> None:
        with patch.dict(os.environ, {'SIGIL_TRUST_ALL': '1'}):
            self.run_agent()
        self.assertTrue(self.marker.exists())

    def test_user_level_items_are_not_affected(self) -> None:
        (self.ws / '.sigil' / 'settings.json').unlink()
        user = Path.home() / '.sigil'
        user.mkdir(parents=True, exist_ok=True)
        settings = user / 'settings.json'
        self.addCleanup(lambda: settings.unlink() if settings.exists() else None)
        settings.write_text(json.dumps({'hooks': {'UserPromptSubmit': [
            {'hooks': [{'type': 'command', 'command': self.command}]}]}}), encoding='utf-8')
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'repo-server': {'command': 'evil'}}}),
                                          encoding='utf-8')
        with patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': ''}):
            err = self.run_agent()
        self.assertTrue(self.marker.exists(), 'a user-level hook was blocked')
        self.assertIn('repo-server', err)  # the project's MCP server is the one held back

    def test_mcp_servers_need_trust_but_user_added_ones_run(self) -> None:
        from src.main import main
        from src.mcp_runtime import MCPRuntime

        (self.ws / '.sigil' / 'settings.json').unlink()
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'repo-server': {'command': 'evil-tool'}}}),
                                          encoding='utf-8')
        with patch('sys.stderr', io.StringIO()):
            self.assertEqual([s.name for s in MCPRuntime.from_workspace(self.ws).servers], [])
            with contextlib_redirect():
                main(['mcp', 'add', 'mine', '--cwd', str(self.ws), '--', 'my-tool', '--flag'])
            names = sorted(s.name for s in MCPRuntime.from_workspace(self.ws).servers)
        self.assertEqual(names, ['mine'])

    def test_trust_cli_and_slash_command(self) -> None:
        from src.agent_slash_commands import preprocess_slash_command
        from src.main import main

        out = io.StringIO()
        with contextlib_redirect(out):
            main(['trust', 'grant', '--cwd', str(self.ws)])
        self.assertIn('trusted', out.getvalue().lower())
        agent = make_agent(self.ws)
        result = preprocess_slash_command(agent, '/trust revoke')
        self.assertIn('Forgotten', str(getattr(result, 'output', result)))
        self.assertIn('new', str(getattr(preprocess_slash_command(agent, '/trust'), 'output', '')))


def contextlib_redirect(target=None):  # noqa: ANN001, ANN201
    import contextlib

    return contextlib.redirect_stdout(target or io.StringIO())


class ApprovalsThroughChatTests(unittest.TestCase):
    """F33 through the chat loop: an approval request in <--cwd>/.port_sessions/background is
    asked in the chat although the process runs in another folder."""

    def test_chat_loop_answers_requests_from_the_agent_folder(self) -> None:
        import threading

        from src.background_approvals import ask, pending
        from src.main import _run_agent_chat_loop

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve()
            root = ws / '.port_sessions' / 'background'
            root.mkdir(parents=True)
            self.assertNotEqual(Path.cwd().resolve(), ws)
            box: dict = {}
            worker = threading.Thread(target=lambda: box.update(answer=ask(
                root, 'bg_c', {'tool': 'bash', 'arguments': {'command': 'git push'}}, timeout=15)))
            worker.start()
            for _ in range(300):
                if pending(root):
                    break
                time.sleep(0.05)
            inputs = iter(['a'])

            def scripted(_label: str) -> str:
                try:
                    return next(inputs)
                except StopIteration:
                    raise EOFError from None

            output: list[str] = []
            model = FakeModel([])
            with patch('src.openai_compat.request.urlopen', side_effect=model), \
                    patch('src.main.HAS_PROMPT_TOOLKIT', False):
                agent = make_agent(ws)
                _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                                     input_func=scripted, output_func=output.append)
            worker.join(20)
        self.assertEqual(box.get('answer'), 'always', output)


if __name__ == '__main__':
    unittest.main()
