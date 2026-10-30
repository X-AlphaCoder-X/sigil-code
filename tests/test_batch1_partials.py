"""Batch 1: background 'blocked: needs approval' note, agent maxTurns and
permissionMode, /tasks with background agents, --verbose."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime
from src.main import main


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


def make_agent(ws: Path, *, write: bool = False) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(allow_file_write=write),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def save_record(runtime: BackgroundSessionRuntime, background_id: str) -> None:
    runtime.save_record(BackgroundSessionRecord(
        background_id=background_id, pid=0, prompt='do things', workspace_cwd='.', model='m',
        mode='agent', status='running', log_path=str(runtime.log_path(background_id)),
        record_path=str(runtime.record_path(background_id)), started_at='now', command=()))


def write_agent(ws: Path, name: str, frontmatter: str) -> None:
    folder = ws / '.sigil' / 'agents'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{name}.md').write_text(f'---\nname: {name}\ndescription: test agent\n{frontmatter}---\nBody.',
                                       encoding='utf-8')


class BackgroundNoteTests(unittest.TestCase):
    def test_note_shows_in_ps_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = BackgroundSessionRuntime(Path(tmp))
            save_record(runtime, 'bg_000000000001')
            runtime.set_note('bg_000000000001', 'blocked: needs approval (Write)')
            self.assertIn('blocked: needs approval (Write)', runtime.render_ps())
            runtime.log_path('bg_000000000001').write_text('', encoding='utf-8')
            self.assertIn('blocked: needs approval', runtime.render_logs('bg_000000000001'))
            finished = runtime.mark_finished('bg_000000000001', exit_code=0)
            self.assertEqual(finished.note, 'blocked: needs approval (Write)')

    def test_blocked_call_reports_to_callback(self) -> None:
        recorded: list[dict[str, object]] = []
        notes: list[str] = []
        responses = [tool_reply('write_file', {'path': 'f.txt', 'content': 'x'}), text_reply('ok')]
        env = {k: v for k, v in os.environ.items() if k != 'SIGIL_BACKGROUND_ASK'}
        env['SIGIL_DISABLE_USER_SETTINGS'] = '1'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)), \
                redirect_stderr(io.StringIO()):
            ws = Path(tmp)
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'permissions': {'ask': ['Write']}}),
                                                          encoding='utf-8')
            agent = make_agent(ws, write=True)
            agent.on_permission_blocked = notes.append
            agent.run('write')
        self.assertEqual(notes, ['blocked: needs approval (Write)'])


class AgentFrontmatterTests(unittest.TestCase):
    def test_max_turns_from_frontmatter(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('delegate_agent', {'prompt': 'go', 'agent': 'short'}),
            tool_reply('list_dir', {'path': '.'}, call_id='c1'),
            text_reply('parent done'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            write_agent(ws, 'short', 'maxTurns: 1\n')
            result = make_agent(ws).run('delegate')
        self.assertEqual(result.final_output, 'parent done')
        self.assertEqual(len(recorded), 3)  # parent, one child turn, parent
        tool_text = [str(m.get('content')) for m in recorded[2]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any('max_turns' in t for t in tool_text), tool_text)

    def test_permission_mode_plan_and_accept_edits(self) -> None:
        for mode, expect_file, allow in (('plan', False, True), ('acceptEdits', True, False)):
            recorded: list[dict[str, object]] = []
            responses = [
                tool_reply('delegate_agent', {'prompt': 'write it', 'agent': 'writer', 'allow_write': allow}),
                tool_reply('write_file', {'path': 'out.txt', 'content': 'x'}, call_id='c1'),
                text_reply('child done'),
                text_reply('parent done'),
            ]
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1'}), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
                ws = Path(tmp)
                write_agent(ws, 'writer', f'permissionMode: {mode}\n')
                make_agent(ws, write=True).run('delegate')
                self.assertEqual((ws / 'out.txt').exists(), expect_file, mode)

    def test_accept_edits_never_exceeds_parent(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('delegate_agent', {'prompt': 'write it', 'agent': 'writer'}),
            tool_reply('write_file', {'path': 'out.txt', 'content': 'x'}, call_id='c1'),
            text_reply('child done'),
            text_reply('parent done'),
        ]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_DISABLE_USER_SETTINGS': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            write_agent(ws, 'writer', 'permissionMode: acceptEdits\n')
            make_agent(ws, write=False).run('delegate')
            self.assertFalse((ws / 'out.txt').exists())


class TasksAndVerboseTests(unittest.TestCase):
    def test_tasks_lists_background_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            save_record(BackgroundSessionRuntime(ws / '.port_sessions' / 'background'), 'bg_00000000abcd')
            output = make_agent(ws).run('/tasks').final_output
        self.assertIn('bg_00000000abcd', output)
        self.assertIn('# Background Sessions', output)

    def test_verbose_prints_tool_lines_to_stderr(self) -> None:
        recorded: list[dict[str, object]] = []

        class _Stdin(io.StringIO):
            def isatty(self) -> bool:
                return True

        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([tool_reply('list_dir', {'path': '.'}), text_reply('listed')],
                                              recorded)), \
                patch('sys.stdin', _Stdin('')), redirect_stdout(out), redirect_stderr(err):
            code = main(['-p', 'list', '--verbose', '--cwd', tmp])
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().strip(), 'listed')
        self.assertIn('[tool] list_dir', err.getvalue())
        self.assertIn('-> ok', err.getvalue())


if __name__ == '__main__':
    unittest.main()
