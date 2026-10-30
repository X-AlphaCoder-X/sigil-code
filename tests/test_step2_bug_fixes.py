"""Step 2 bug fixes: temperature, ask_user, /effort, session folder, glob in
added folders, rewind of outside files, chat turn cap, background forwarding.

Model calls go through a fake urlopen (no network, no real server).
"""
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
from src.checkpoints import CheckpointStore
from src.main import (
    _append_agent_forwarded_args,
    _build_runtime_config,
    _launch_background_agent,
    _run_agent_chat_loop,
    _run_session_picker,
    build_parser,
)
from src.openai_compat import OpenAICompatClient
from src.providers import apply_effort, detect_provider, get_quirks


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
    return {
        'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
    }


def tool_reply(name: str, arguments: dict[str, object], call_id: str = 'call_1') -> dict[str, object]:
    return {
        'choices': [{
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': call_id,
                    'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)},
                }],
            },
            'finish_reason': 'tool_calls',
        }],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
    }


def make_agent(workspace: Path, *, model: str = 'test-model',
               base_url: str = 'http://127.0.0.1:8000/v1', **runtime) -> LocalCodingAgent:
    runtime.setdefault('session_directory', workspace / '.port_sessions' / 'agent')
    runtime.setdefault('scratchpad_root', workspace / '.port_sessions' / 'scratchpad')
    runtime.setdefault('nudge_strategy', 'off')
    return LocalCodingAgent(
        model_config=ModelConfig(model=model, base_url=base_url),
        runtime_config=AgentRuntimeConfig(cwd=workspace, **runtime),
    )


class TemperatureTests(unittest.TestCase):
    def test_default_payload_has_no_temperature(self) -> None:
        recorded: list[dict[str, object]] = []
        with patch('src.openai_compat.request.urlopen',
                   side_effect=fake_server([text_reply('ok')], recorded)):
            OpenAICompatClient(ModelConfig(model='m')).complete([{'role': 'user', 'content': 'hi'}], [])
        self.assertNotIn('temperature', recorded[0])

    def test_temperature_sent_when_set(self) -> None:
        recorded: list[dict[str, object]] = []
        with patch.dict(os.environ, {'SIGIL_EXCLUDE_PARAMS': ''}), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('ok')], recorded)):
            OpenAICompatClient(ModelConfig(model='m', temperature=0.3)).complete(
                [{'role': 'user', 'content': 'hi'}], [])
        self.assertEqual(recorded[0]['temperature'], 0.3)

    def test_cli_default_temperature_is_unset(self) -> None:
        args = build_parser().parse_args(['agent', 'hi'])
        self.assertIsNone(args.temperature)


class AskUserInChatTests(unittest.TestCase):
    def test_fake_input_answers_question(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('ask_user_question', {'question': 'Which color?'}),
            text_reply('You picked blue.'),
        ]
        asked: list[str] = []

        def _input(prompt: str) -> str:
            if 'Which color?' in prompt:
                asked.append(prompt)
                return 'blue'
            raise EOFError

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {'SIGIL_ASK_USER_INTERACTIVE': ''}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp))
            code = _run_agent_chat_loop(
                agent, initial_prompt='ask me', resume_session_id=None, show_transcript=False,
                input_func=_input, output_func=lambda _text: None,
                result_printer=lambda _result, show_transcript: None,
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(asked), 1)
        tool_messages = [m for m in recorded[1]['messages'] if m.get('role') == 'tool']
        self.assertTrue(any('blue' in str(m.get('content')) for m in tool_messages))


class EffortTests(unittest.TestCase):
    def test_provider_detection(self) -> None:
        self.assertEqual(detect_provider('https://api.anthropic.com/v1'), 'anthropic')
        self.assertEqual(detect_provider('https://openrouter.ai/api/v1'), 'openrouter')
        self.assertEqual(detect_provider('https://api.openai.com/v1'), 'openai')
        self.assertEqual(detect_provider('http://localhost:11434/v1'), 'ollama')
        self.assertEqual(detect_provider('http://localhost:1234/v1'), 'lmstudio')
        self.assertEqual(detect_provider('http://localhost:8080/v1'), 'llamacpp')
        self.assertEqual(detect_provider('http://localhost:8000/v1'), 'vllm')

    def test_effort_field_per_provider(self) -> None:
        with patch.dict(os.environ, {'SIGIL_EFFORT_FIELD': ''}):
            openai = apply_effort({}, 'high', get_quirks('https://api.openai.com/v1', 'gpt-5'))
            anthropic = apply_effort({'model': 'claude-sonnet-4-5'}, 'max',
                                     get_quirks('https://api.anthropic.com/v1', 'claude-sonnet-4-5'))
            anthropic_now = apply_effort({'model': 'claude-opus-5-5', 'temperature': 0.2}, 'max',
                                         get_quirks('https://api.anthropic.com/v1', 'claude-opus-5-5'))
            qwen = apply_effort({}, 'low', get_quirks('http://127.0.0.1:8000/v1', 'Qwen/Qwen3-32B'))
            gpt_oss = apply_effort({}, 'max', get_quirks('http://127.0.0.1:8000/v1', 'openai/gpt-oss-20b'))
            plain = apply_effort({}, 'high', get_quirks('http://127.0.0.1:8000/v1', 'mistral-small'))
            auto = apply_effort({}, 'auto', get_quirks('https://api.openai.com/v1', 'gpt-5'))
        self.assertEqual(openai, {'reasoning_effort': 'high'})
        self.assertEqual(anthropic, {'model': 'claude-sonnet-4-5',
                                     'thinking': {'type': 'enabled', 'budget_tokens': 32000}})
        # the OpenAI-compatible endpoint takes only `thinking`; 4.7+ rejects temperature below 1
        self.assertEqual(anthropic_now, {'model': 'claude-opus-5-5', 'thinking': {'type': 'adaptive'}})
        self.assertEqual(qwen, {'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual(gpt_oss, {'reasoning_effort': 'high'})
        self.assertEqual(plain, {})
        self.assertEqual(auto, {})

    def test_effort_field_override(self) -> None:
        with patch.dict(os.environ, {'SIGIL_EFFORT_FIELD': 'reasoning_effort'}):
            payload = apply_effort({}, 'medium', get_quirks('http://127.0.0.1:8000/v1', 'mistral'))
        self.assertEqual(payload, {'reasoning_effort': 'medium'})

    def test_slash_effort_reaches_the_request(self) -> None:
        recorded: list[dict[str, object]] = []
        env = {k: v for k, v in os.environ.items() if k not in ('SIGIL_EFFORT_LEVEL', 'SIGIL_EFFORT_FIELD')}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('done')], recorded)):
            agent = make_agent(Path(tmp), model='gpt-5', base_url='https://api.openai.com/v1')
            result = agent.run('/effort high')
            self.assertIn('Set effort level to: high', result.final_output)
            agent.run('think about it')
            shown = agent.run('/effort')
        self.assertEqual(recorded[0]['reasoning_effort'], 'high')
        self.assertIn('Current effort level: high', shown.final_output)


class SessionFolderTests(unittest.TestCase):
    def test_sessions_saved_under_cwd_and_found_by_picker(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(['agent', 'hello', '--cwd', tmp])
            config = _build_runtime_config(args)
            expected = Path(tmp).resolve() / '.port_sessions' / 'agent'
            self.assertEqual(config.session_directory, expected)
            self.assertEqual(config.scratchpad_root, Path(tmp).resolve() / '.port_sessions' / 'scratchpad')
            with patch('src.openai_compat.request.urlopen',
                       side_effect=fake_server([text_reply('hi there')], recorded)):
                agent = LocalCodingAgent(model_config=ModelConfig(model='m'), runtime_config=config)
                result = agent.run('hello')
            self.assertTrue((expected / f'{result.session_id}.json').exists())

            seen: dict[str, object] = {}

            def _fake_loop(agent, *, initial_prompt, resume_session_id, show_transcript, **_kw):
                seen['resume_session_id'] = resume_session_id
                return 0

            picker_args = build_parser().parse_args(['sessions', '--cwd', tmp])
            with patch('src.main._run_agent_chat_loop', side_effect=_fake_loop), \
                    patch('src.main._build_agent', return_value=object()), \
                    patch('builtins.input', return_value='0'):
                _run_session_picker(picker_args)
        self.assertEqual(seen['resume_session_id'], result.session_id)

    def test_chat_loop_resolves_latest(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen',
                      side_effect=fake_server([text_reply('first'), text_reply('second')], recorded)):
            agent = make_agent(Path(tmp))
            first = agent.run('remember me')
            resumed = make_agent(Path(tmp))
            prompts = iter(['continue'])

            def _input(_prompt: str) -> str:
                try:
                    return next(prompts)
                except StopIteration:
                    raise EOFError

            outputs: list[str] = []
            _run_agent_chat_loop(
                resumed, initial_prompt=None, resume_session_id='latest', show_transcript=False,
                input_func=_input, output_func=outputs.append,
                result_printer=lambda _result, show_transcript: None,
            )
        self.assertIn(f'resuming_session_id={first.session_id}', outputs)
        contents = [m.get('content') for m in recorded[1]['messages']]
        self.assertIn('remember me', contents)


class GlobAddedFolderTests(unittest.TestCase):
    def test_glob_finds_file_in_added_folder(self) -> None:
        with tempfile.TemporaryDirectory() as main_dir, tempfile.TemporaryDirectory() as extra_dir:
            (Path(extra_dir) / 'only_here.txt').write_text('x', encoding='utf-8')
            (Path(main_dir) / 'main_only.txt').write_text('x', encoding='utf-8')
            config = AgentRuntimeConfig(
                cwd=Path(main_dir), additional_working_directories=(Path(extra_dir),))
            context = build_tool_context(config)
            pattern = str(Path(extra_dir).resolve() / '*.txt')
            result = execute_tool(default_tool_registry(), 'glob_search', {'pattern': pattern}, context)
        self.assertTrue(result.ok, result.content)
        self.assertIn('only_here.txt', result.content)
        self.assertNotIn('main_only.txt', result.content)
        self.assertIn(str(Path(extra_dir).resolve()), result.content)


class RewindOutsideFileTests(unittest.TestCase):
    def test_rewind_restores_file_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'notes.txt'
            target.write_text('original', encoding='utf-8')
            store = CheckpointStore(Path(ws) / 'cp', Path(ws))
            cid = store.begin()
            store.record_before(target)
            target.write_text('changed', encoding='utf-8')
            created = Path(outside) / 'new.txt'
            store.record_before(created)
            created.write_text('new', encoding='utf-8')
            result = store.rewind(cid)
            self.assertEqual(target.read_text(encoding='utf-8'), 'original')
            self.assertFalse(created.exists())
        self.assertEqual(len(result.restored), 1)
        self.assertEqual(len(result.deleted), 1)


class ChatTurnCapTests(unittest.TestCase):
    def test_chat_default_is_unlimited(self) -> None:
        self.assertEqual(build_parser().parse_args(['agent-chat']).max_turns, 0)
        self.assertEqual(build_parser().parse_args(['sessions']).max_turns, 0)
        self.assertEqual(build_parser().parse_args(['agent', 'x']).max_turns, 12)

    def test_unlimited_run_goes_past_twelve_turns(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('list_dir', {'path': '.'}, call_id=f'call_{i}') for i in range(14)]
        responses.append(text_reply('finished after many turns'))
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            agent = make_agent(Path(tmp), max_turns=0)
            result = agent.run('keep going')
        self.assertEqual(result.final_output, 'finished after many turns')
        self.assertNotEqual(result.stop_reason, 'max_turns')
        self.assertEqual(len(recorded), 15)


class BackgroundForwardingTests(unittest.TestCase):
    def _launch(self, argv: list[str]) -> tuple[list[str], dict[str, str]]:
        captured: dict[str, object] = {}

        class _Record:
            background_id = 'bg1'
            pid = 1
            log_path = 'log'
            record_path = 'rec'

        def _fake_launch(self, command, **kwargs):  # noqa: ANN001
            captured['command'] = command
            captured['env'] = kwargs.get('env')
            return _Record()

        args = build_parser().parse_args(argv)
        with patch('src.main.BackgroundSessionRuntime.launch', _fake_launch), \
                patch('builtins.print'):
            _launch_background_agent(args)
        return captured['command'], captured['env']  # type: ignore[return-value]

    def test_api_key_not_in_command(self) -> None:
        command, env = self._launch(['agent-bg', 'do it', '--api-key', 'sk-secret-123'])
        self.assertNotIn('sk-secret-123', ' '.join(command))
        self.assertNotIn('--api-key', command)
        self.assertEqual(env['OPENAI_API_KEY'], 'sk-secret-123')

    def test_unset_values_are_skipped_and_worker_parses(self) -> None:
        args = build_parser().parse_args(['agent-bg', 'do it'])
        forwarded: list[str] = []
        _append_agent_forwarded_args(forwarded, args, include_backend=True)
        self.assertNotIn('--temperature', forwarded)
        self.assertNotIn('None', forwarded)
        worker = build_parser().parse_args(
            ['agent-bg-worker', 'bg1', 'do it', '--background-root', '.'] + forwarded)
        self.assertIsNone(worker.temperature)

        args = build_parser().parse_args(['agent-bg', 'do it', '--temperature', '0.4'])
        forwarded = []
        _append_agent_forwarded_args(forwarded, args, include_backend=True)
        self.assertEqual(forwarded[forwarded.index('--temperature') + 1], '0.4')


if __name__ == '__main__':
    unittest.main()
