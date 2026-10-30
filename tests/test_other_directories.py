"""3.1 other-directories switch (src/path_access.py), through the real tools and
a fake model server."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.main import _run_agent_chat_loop
from src.path_access import PathAccess
from src.permission_modes import apply_mode
from src.rewind import rewind_session

SWITCH_VARS = ('SIGIL_OTHER_DIRECTORIES', 'OTHER_DIRECTORIES', 'SIGIL_ALLOWED_DIRECTORIES',
               'SIGIL_PROTECTED_PATHS')


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SWITCH_VARS}
    env['SIGIL_DISABLE_USER_SETTINGS'] = '1'
    env.update(extra)
    return env


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


def run_tool(workspace: Path, name: str, arguments: dict[str, object], *, write: bool = False,
             added: tuple[Path, ...] = ()):
    config = AgentRuntimeConfig(
        cwd=workspace,
        permissions=AgentPermissions(allow_file_write=write),
        additional_working_directories=added,
    )
    return execute_tool(default_tool_registry(), name, arguments, build_tool_context(config))


class ReadWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory()
        self._out = tempfile.TemporaryDirectory()
        self.ws = Path(self._ws.name)
        self.out = Path(self._out.name)
        (self.out / 'data.txt').write_text('outside data', encoding='utf-8')

    def tearDown(self) -> None:
        self._ws.cleanup()
        self._out.cleanup()

    def test_switch_off_refuses_outside_read(self) -> None:
        with patch.dict(os.environ, clean_env(), clear=True):
            result = run_tool(self.ws, 'read_file', {'path': str(self.out / 'data.txt')})
        self.assertFalse(result.ok)
        self.assertIn('SIGIL_OTHER_DIRECTORIES', result.content)

    def test_switch_on_reads_anywhere(self) -> None:
        for name in ('SIGIL_OTHER_DIRECTORIES', 'OTHER_DIRECTORIES'):
            for value in ('yes', 'true', '1'):
                with patch.dict(os.environ, clean_env(**{name: value}), clear=True):
                    result = run_tool(self.ws, 'read_file', {'path': str(self.out / 'data.txt')})
                self.assertTrue(result.ok, (name, value, result.content))
                self.assertIn('outside data', result.content)

    def test_write_outside_needs_write_permission(self) -> None:
        target = str(self.out / 'new.txt')
        with patch.dict(os.environ, clean_env(SIGIL_OTHER_DIRECTORIES='yes'), clear=True):
            denied = run_tool(self.ws, 'write_file', {'path': target, 'content': 'x'})
            allowed = run_tool(self.ws, 'write_file', {'path': target, 'content': 'saved'}, write=True)
        self.assertFalse(denied.ok)
        self.assertTrue(allowed.ok, allowed.content)
        self.assertEqual((self.out / 'new.txt').read_text(encoding='utf-8'), 'saved')

    def test_dotdot_allowed_when_resolved_path_is_allowed(self) -> None:
        rel = os.path.relpath(self.out / 'data.txt', self.ws)
        with patch.dict(os.environ, clean_env(), clear=True):
            inside = run_tool(self.ws, 'write_file', {'path': 'sub/../in.txt', 'content': 'x'}, write=True)
            outside_off = run_tool(self.ws, 'read_file', {'path': rel})
        with patch.dict(os.environ, clean_env(SIGIL_OTHER_DIRECTORIES='yes'), clear=True):
            outside_on = run_tool(self.ws, 'read_file', {'path': rel})
        self.assertTrue(inside.ok, inside.content)
        self.assertFalse(outside_off.ok)
        self.assertTrue(outside_on.ok, outside_on.content)

    def test_narrow_list_allows_only_named_folders(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            (Path(other) / 'x.txt').write_text('other', encoding='utf-8')
            env = clean_env(SIGIL_ALLOWED_DIRECTORIES=str(self.out))
            with patch.dict(os.environ, env, clear=True):
                named = run_tool(self.ws, 'read_file', {'path': str(self.out / 'data.txt')})
                saved = run_tool(self.ws, 'write_file', {'path': str(self.out / 'w.txt'), 'content': 'y'},
                                 write=True)
                unnamed = run_tool(self.ws, 'read_file', {'path': str(Path(other) / 'x.txt')})
        self.assertTrue(named.ok, named.content)
        self.assertTrue(saved.ok, saved.content)
        self.assertFalse(unnamed.ok)

    def test_protected_list_blocks_writes_not_reads(self) -> None:
        protected = self.out
        env = clean_env(SIGIL_OTHER_DIRECTORIES='yes', SIGIL_PROTECTED_PATHS=str(protected))
        with patch.dict(os.environ, env, clear=True):
            read = run_tool(self.ws, 'read_file', {'path': str(self.out / 'data.txt')})
            write = run_tool(self.ws, 'write_file', {'path': str(self.out / 'p.txt'), 'content': 'x'}, write=True)
        self.assertTrue(read.ok)
        self.assertFalse(write.ok)
        self.assertIn('write-protected', write.content)
        # An empty list turns protection off.
        env = clean_env(SIGIL_OTHER_DIRECTORIES='yes', SIGIL_PROTECTED_PATHS='')
        with patch.dict(os.environ, env, clear=True):
            write = run_tool(self.ws, 'write_file', {'path': str(self.out / 'p.txt'), 'content': 'x'}, write=True)
        self.assertTrue(write.ok, write.content)

    def test_protected_paths_from_settings(self) -> None:
        settings = {'permissions': {'protectedPaths': [str(self.out)]}}
        access = PathAccess.from_env(self.ws, settings=settings, env={'SIGIL_OTHER_DIRECTORIES': 'yes'})
        self.assertFalse(access.can_write(self.out / 'f.txt'))
        self.assertTrue(access.can_read(self.out / 'f.txt'))
        default = PathAccess.from_env(self.ws, env={'SIGIL_OTHER_DIRECTORIES': 'yes'})
        self.assertTrue(default.is_protected(Path.home() / '.ssh' / 'id_rsa'))
        # The workspace itself is never write-protected.
        inside = PathAccess.from_env(self.ws, settings={'permissions': {'protectedPaths': [str(self.ws)]}},
                                     env={})
        self.assertTrue(inside.can_write(self.ws / 'a.txt'))

    def test_glob_outside_with_switch(self) -> None:
        pattern = str(self.out / '*.txt')
        with patch.dict(os.environ, clean_env(), clear=True):
            off = run_tool(self.ws, 'glob_search', {'pattern': pattern})
        with patch.dict(os.environ, clean_env(SIGIL_OTHER_DIRECTORIES='yes'), clear=True):
            on = run_tool(self.ws, 'glob_search', {'pattern': pattern})
        self.assertFalse(off.ok)
        self.assertTrue(on.ok, on.content)
        self.assertIn('data.txt', on.content)


class AgentLevelTests(unittest.TestCase):
    def _agent(self, ws: Path, *, write: bool = True) -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model='m'),
            runtime_config=AgentRuntimeConfig(
                cwd=ws,
                permissions=AgentPermissions(allow_file_write=write),
                session_directory=ws / '.port_sessions' / 'agent',
                scratchpad_root=ws / '.port_sessions' / 'scratchpad',
                nudge_strategy='off',
            ),
        )

    def test_plan_mode_blocks_outside_write(self) -> None:
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as out:
            target = Path(out) / 'plan.txt'
            recorded: list[dict[str, object]] = []
            responses = [tool_reply('write_file', {'path': str(target), 'content': 'x'}), text_reply('ok')]
            with patch.dict(os.environ, clean_env(SIGIL_OTHER_DIRECTORIES='yes'), clear=True), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
                agent = self._agent(Path(ws))
                apply_mode(agent, 'plan')
                agent.run('save it')
            self.assertFalse(target.exists())

    def test_rewind_restores_outside_file(self) -> None:
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as out:
            target = Path(out) / 'keep.txt'
            target.write_text('before', encoding='utf-8')
            recorded: list[dict[str, object]] = []
            responses = [tool_reply('write_file', {'path': str(target), 'content': 'after'}), text_reply('done')]
            with patch.dict(os.environ, clean_env(SIGIL_OTHER_DIRECTORIES='yes'), clear=True), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
                agent = self._agent(Path(ws))
                result = agent.run('change it')
                self.assertEqual(target.read_text(encoding='utf-8'), 'after')
                rewind_session(agent, result.session_id, 1, 'code')
            self.assertEqual(target.read_text(encoding='utf-8'), 'before')

    def test_status_and_chat_start_show_switch(self) -> None:
        with tempfile.TemporaryDirectory() as ws, \
                patch.dict(os.environ, clean_env(SIGIL_OTHER_DIRECTORIES='yes'), clear=True):
            agent = self._agent(Path(ws))
            status = agent.run('/status').final_output
            outputs: list[str] = []

            def _eof(_prompt: str) -> str:
                raise EOFError

            _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                                 input_func=_eof, output_func=outputs.append)
        self.assertIn('Other directories: allowed', status)
        self.assertIn('Other directories: allowed', outputs)

    def test_background_agent_gets_added_folders(self) -> None:
        captured: dict[str, object] = {}

        class _Record:
            background_id = 'bg1'
            pid = 1
            log_path = 'log'

        def _fake_launch(self, command, **kwargs):  # noqa: ANN001
            captured['command'] = command
            return _Record()

        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as extra, \
                patch('src.background_runtime.BackgroundSessionRuntime.launch', _fake_launch):
            config = AgentRuntimeConfig(cwd=Path(ws), additional_working_directories=(Path(extra),))
            context = build_tool_context(config)
            context.model_config = ModelConfig(model='m')
            result = execute_tool(default_tool_registry(), 'background_agent_start', {'prompt': 'go'}, context)
            self.assertTrue(result.ok, result.content)
            command = captured['command']
            self.assertIn('--add-dir', command)
            self.assertIn(str(Path(extra)), command)


if __name__ == '__main__':
    unittest.main()
