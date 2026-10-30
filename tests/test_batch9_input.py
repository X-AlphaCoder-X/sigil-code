"""Batch 9 (P2): Ctrl+G editor key, vim mode, status line, live todos, rewind of bash
changes to tracked files, -w/--worktree, -n/--name."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_bar import AgentBar, build_prompt_kwargs
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.checkpoints import CheckpointStore
from src.main import _run_agent_chat_loop, _vim_mode, build_parser, main
from src.status_line import StatusLine, status_line_for


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


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def binding_named(bar: AgentBar, name: str):
    bindings = build_prompt_kwargs(bar)['key_bindings']
    return [b for b in bindings.bindings if b.handler.__name__ == name]


class KeyAndSettingTests(unittest.TestCase):
    def test_editor_and_todo_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bar = AgentBar(Path(tmp))
            editor_bindings = binding_named(bar, '_edit_in_editor')
            keys = {tuple(str(getattr(k, 'value', k)) for k in b.keys) for b in editor_bindings}
            self.assertIn(('c-g',), keys)
            self.assertIn(('c-x', 'c-e'), keys)

            class _App:
                def invalidate(self) -> None:
                    pass

            class _Event:
                app = _App()

            toggle = binding_named(bar, '_toggle_todos')[0]
            toggle.handler(_Event())
            self.assertTrue(bar.show_todos)

    def test_vim_mode_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1', 'SIGIL_EDITOR_MODE': ''}):
            self.assertFalse(_vim_mode(Path(tmp)))
            (Path(tmp) / '.sigil').mkdir()
            (Path(tmp) / '.sigil' / 'settings.json').write_text(json.dumps({'editorMode': 'vim'}), encoding='utf-8')
            self.assertTrue(_vim_mode(Path(tmp)))

    def test_status_line_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1'}):
            ws = Path(tmp)
            script = ws / 'status.py'
            script.write_text('import json, sys\nd = json.load(sys.stdin)\nprint("[" + d["model"] + "] " + '
                              'd["permission_mode"])\nprint("second line ignored")\n', encoding='utf-8')
            agent = make_agent(ws)
            agent.run(f'/statusline "{sys.executable}" "{script}"')
            line = status_line_for(ws)
            self.assertIsNotNone(line)
            text = line.render({'model': 'qwen', 'permission_mode': 'plan'})
            self.assertEqual(text, '[qwen] plan')
            self.assertIn('Status line: {"type": "command"', agent.run('/statusline').final_output)


class TodoAndRewindTests(unittest.TestCase):
    def test_todos_print_when_toggled(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('todo_write', {'items': [{'title': 'Write tests', 'status': 'pending'}]}),
                     text_reply('planned')]
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
                patch('sys.stdout', out):
            agent = make_agent(Path(tmp), allow_file_write=True)  # todo_write saves the list
            agent.show_todos = True
            agent.run('plan it')
        self.assertIn('Write tests', out.getvalue())

    def test_rewind_undoes_bash_change_to_tracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as ws:
            store = CheckpointStore(Path(ws) / 'cp', Path(ws))
            target = Path(ws) / 'app.py'
            target.write_text('v1', encoding='utf-8')
            first = store.begin()
            store.record_before('app.py')
            target.write_text('v2', encoding='utf-8')      # edit tool, prompt 1
            second = store.begin()
            store.record_tracked()                          # before bash, prompt 2
            target.write_text('v3 by bash', encoding='utf-8')
            only_bash = Path(ws) / 'untracked.txt'
            only_bash.write_text('bash only', encoding='utf-8')
            store.rewind(second)
            self.assertEqual(target.read_text(encoding='utf-8'), 'v2')
            self.assertTrue(only_bash.exists())  # files only bash touched stay untracked
            store.rewind(first)
            self.assertEqual(target.read_text(encoding='utf-8'), 'v1')


class NameAndWorktreeTests(unittest.TestCase):
    def test_name_flag_and_resume_by_name(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('hello'), text_reply('again')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            args = build_parser().parse_args(['agent-chat', 'first message', '--cwd', tmp, '-n', 'feature-x'])
            from src.main import _build_agent

            agent = _build_agent(args)
            outputs: list[str] = []
            _run_agent_chat_loop(agent, initial_prompt='first message', resume_session_id=None,
                                 show_transcript=False, input_func=lambda _l: (_ for _ in ()).throw(EOFError()),
                                 output_func=outputs.append, result_printer=lambda _r, show_transcript: None)
            resumed = _build_agent(build_parser().parse_args(['agent-chat', '--cwd', tmp]))
            prompts = iter(['second'])

            def _input(_label: str) -> str:
                try:
                    return next(prompts)
                except StopIteration:
                    raise EOFError

            outputs = []
            _run_agent_chat_loop(resumed, initial_prompt=None, resume_session_id='feature-x', show_transcript=False,
                                 input_func=_input, output_func=outputs.append,
                                 result_printer=lambda _r, show_transcript: None)
        self.assertTrue(any(line.startswith('resuming_session_id=') for line in outputs))
        self.assertIn('first message', [m.get('content') for m in recorded[1]['messages']])

    def test_worktree_flag(self) -> None:
        if subprocess.run(['git', '--version'], capture_output=True).returncode != 0:
            raise AssertionError('git is needed for this test')
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo'
            repo.mkdir()
            for command in (['git', 'init', '-q'], ['git', 'config', 'user.email', 't@t'],
                            ['git', 'config', 'user.name', 't'], ['git', 'commit', '--allow-empty', '-qm', 'init']):
                subprocess.run(command, cwd=repo, capture_output=True, check=True)
            args = build_parser().parse_args(['agent', 'x', '--cwd', str(repo), '-w', 'feat'])
            from src.main import _enter_worktree

            with patch('sys.stderr', io.StringIO()):
                _enter_worktree(args)
                first = args.cwd
                again = build_parser().parse_args(['agent', 'x', '--cwd', str(repo), '-w', 'feat'])
                _enter_worktree(again)  # an existing worktree is reused
            self.assertNotEqual(Path(first).resolve(), repo.resolve())
            self.assertTrue(Path(first).is_dir())
            self.assertEqual(Path(again.cwd).resolve(), Path(first).resolve())
            subprocess.run(['git', 'worktree', 'remove', '--force', first], cwd=repo, capture_output=True)


if __name__ == '__main__':
    unittest.main()
