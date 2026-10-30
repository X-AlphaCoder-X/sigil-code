"""Batch 4 (P1): images (read_file, --image, Alt+V), compaction live state,
background live-state forwarding, --effort."""
from __future__ import annotations

import dataclasses
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib import error

from prompt_toolkit.buffer import Buffer

from src.agent_bar import AgentBar, build_prompt_kwargs
from src.agent_runtime import LocalCodingAgent
from src.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.compact import _build_live_state_message
from src.main import _build_runtime_config, build_parser, main
from src.openai_compat import OpenAICompatClient, OpenAICompatError
from src.permission_modes import apply_mode
from src.plan_runtime import PlanStep

PNG = (b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
       b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82')


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


def image_parts(payload: dict[str, object]) -> list[dict[str, object]]:
    parts = []
    for message in payload['messages']:
        content = message.get('content')
        if isinstance(content, list):
            parts.extend(part for part in content if part.get('type') == 'image_url')
    return parts


def make_agent(ws: Path) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='vision-model'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


class ImageTests(unittest.TestCase):
    def test_read_file_image_goes_to_the_model(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('read_file', {'path': 'shot.png'}), text_reply('A red pixel.')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_VISION': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            (ws / 'shot.png').write_bytes(PNG)
            result = make_agent(ws).run('what is in shot.png?')
        self.assertEqual(result.final_output, 'A red pixel.')
        parts = image_parts(recorded[1])
        self.assertEqual(len(parts), 1)
        self.assertTrue(parts[0]['image_url']['url'].startswith('data:image/png;base64,'))
        tools = [m for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertTrue(all(isinstance(m['content'], str) for m in tools))

    def test_vision_off_and_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'shot.png').write_bytes(PNG)
            context = build_tool_context(AgentRuntimeConfig(cwd=ws))
            with patch.dict(os.environ, {'SIGIL_VISION': '0'}):
                off = execute_tool(default_tool_registry(), 'read_file', {'path': 'shot.png'}, context)
            with patch.dict(os.environ, {'SIGIL_VISION': '1', 'SIGIL_MAX_IMAGE_BYTES': '10'}):
                big = execute_tool(default_tool_registry(), 'read_file', {'path': 'shot.png'}, context)
        self.assertTrue(off.ok)
        self.assertFalse(off.metadata.get('image_attached'))
        self.assertFalse(big.ok)
        self.assertIn('SIGIL_MAX_IMAGE_BYTES', big.content)

    def test_image_flag_in_print_mode(self) -> None:
        recorded: list[dict[str, object]] = []

        class _Stdin(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_VISION': '1'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('ok')], recorded)), \
                patch('sys.stdin', _Stdin('')), redirect_stdout(io.StringIO()):
            (Path(tmp) / 'a.png').write_bytes(PNG)
            code = main(['-p', 'describe', '--image', 'a.png', '--cwd', tmp])
        self.assertEqual(code, 0)
        self.assertEqual(len(image_parts(recorded[0])), 1)

    def test_text_only_model_error_has_a_hint(self) -> None:
        def _reject(request_obj, timeout=None):  # noqa: ANN001
            raise error.HTTPError('http://x', 400, 'bad', {}, io.BytesIO(b'image input not supported'))

        client = OpenAICompatClient(ModelConfig(model='m'))
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'x'},
                                                  {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA'}}]}]
        with patch('src.openai_compat.request.urlopen', side_effect=_reject):
            with self.assertRaises(OpenAICompatError) as caught:
                client.complete(messages, [])
        self.assertIn('text-only', str(caught.exception))

    def test_alt_v_pastes_an_image_chip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / 'clip.png'
            png.write_bytes(PNG)
            bar = AgentBar(Path(tmp))
            bindings = build_prompt_kwargs(bar)['key_bindings']
            binding = next(b for b in bindings.bindings if b.handler.__name__ == '_paste_image')

            class _Event:
                current_buffer = Buffer()

            event = _Event()
            with patch('src.images.grab_clipboard_image', return_value=png):
                binding.handler(event)
            self.assertEqual(event.current_buffer.text, '[Image #1]')
            self.assertEqual(bar.pasted_images, [str(png)])


class CompactionStateTests(unittest.TestCase):
    def test_live_state_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(Path(tmp))
            agent.plan_runtime = dataclasses.replace(agent.plan_runtime, steps=(PlanStep(step='Write the parser'),))
            apply_mode(agent, 'plan')
            from src.agent_session import AgentMessage

            old = [AgentMessage(role='assistant', content='', tool_calls=({
                'id': 'c', 'type': 'function',
                'function': {'name': 'skill_get', 'arguments': json.dumps({'name': 'tdd-workflow'})}},))]
            message = _build_live_state_message(agent, old)
        self.assertIsNotNone(message)
        self.assertIn('Write the parser', message.content)
        self.assertIn('Permission mode: plan', message.content)
        self.assertIn('tdd-workflow', message.content)


class ForwardingTests(unittest.TestCase):
    def test_background_agent_gets_live_state(self) -> None:
        captured: dict[str, object] = {}

        class _Record:
            background_id = 'bg1'
            pid = 1
            log_path = 'log'

        def _fake_launch(self, command, **kwargs):  # noqa: ANN001
            captured['command'] = command
            return _Record()

        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.background_runtime.BackgroundSessionRuntime.launch', _fake_launch):
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m', effort='high', temperature=0.4),
                runtime_config=AgentRuntimeConfig(cwd=Path(tmp), permissions=AgentPermissions(
                    allow_file_write=True, allow_shell_commands=True, allow_destructive_shell_commands=True)),
            )
            apply_mode(agent, 'acceptEdits')
            context = agent.tool_context
            result = execute_tool(agent.tool_registry, 'background_agent_start',
                                  {'prompt': 'go', 'allow_destructive': True}, context)
            self.assertTrue(result.ok, result.content)
        command = captured['command']
        joined = ' '.join(command)
        self.assertIn('--effort high', joined)
        self.assertIn('--temperature 0.4', joined)
        self.assertIn('--permission-mode acceptEdits', joined)
        self.assertIn('--unsafe', command)

    def test_effort_flag_and_worker_session_folder(self) -> None:
        args = build_parser().parse_args(['agent', 'x', '--effort', 'low'])
        from src.main import _build_model_config

        self.assertEqual(_build_model_config(args).effort, 'low')
        with tempfile.TemporaryDirectory() as tmp:
            worker = build_parser().parse_args(['agent-bg-worker', 'bg1', 'p', '--background-root', '.', '--cwd', tmp])
            config = _build_runtime_config(worker)
            self.assertEqual(config.session_directory, Path(tmp).resolve() / '.port_sessions' / 'agent')
            self.assertEqual(config.scratchpad_root, Path(tmp).resolve() / '.port_sessions' / 'scratchpad')


class SessionStoreTests(unittest.TestCase):
    def test_project_store_gets_a_gitignore(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('hi')], recorded)):
            make_agent(Path(tmp)).run('hello')
            ignore = Path(tmp) / '.port_sessions' / '.gitignore'
            self.assertTrue(ignore.exists())
            self.assertIn('*', ignore.read_text(encoding='utf-8'))

    def test_home_store_outside_the_repo_with_migration(self) -> None:
        from src.main import _session_directory_for
        from src.session_store import list_agent_sessions

        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, \
                patch('src.settings._home', return_value=Path(home)), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('hi')], recorded)):
            old = make_agent(Path(tmp)).run('saved in the project first')
            with patch.dict(os.environ, {'SIGIL_SESSION_STORE': 'home'}):
                folder = _session_directory_for(tmp)
                self.assertTrue(str(folder).startswith(str(Path(home))))
                self.assertIn(old.session_id, [s[0] for s in list_agent_sessions(folder)])
                args = build_parser().parse_args(['agent', 'x', '--cwd', tmp])
                config = _build_runtime_config(args)
            self.assertEqual(config.session_directory, folder)
            self.assertTrue(str(config.scratchpad_root).startswith(str(Path(home))))


if __name__ == '__main__':
    unittest.main()
