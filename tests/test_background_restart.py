"""Restarting a stopped background agent by id: it continues its saved session (or starts the
prompt again), keeps its id, and refuses while it is still running."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import execute_tool
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime
from src.main import _run_background_worker, build_parser, main


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


BG = 'bg_0000000000aa'


def save_record(runtime: BackgroundSessionRuntime, status: str, cwd: str) -> None:
    runtime.root.mkdir(parents=True, exist_ok=True)
    runtime.log_path(BG).write_text('', encoding='utf-8')
    command = ('python', '-m', 'src.main', 'agent-bg-worker', BG, 'original task', '--cwd', cwd,
               '--background-root', str(runtime.root))
    runtime.save_record(BackgroundSessionRecord(
        background_id=BG, pid=0 if status != 'running' else os.getpid(), prompt='original task',
        workspace_cwd=cwd, model='m', mode='agent', status=status, log_path=str(runtime.log_path(BG)),
        record_path=str(runtime.record_path(BG)), started_at='now', command=command))


class _Launched:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, runtime, command, **kwargs):  # noqa: ANN001
        self.calls.append({'command': list(command), **kwargs})
        record = BackgroundSessionRecord(
            background_id=kwargs['background_id'], pid=os.getpid(), prompt=kwargs['prompt'],
            workspace_cwd=str(kwargs['workspace_cwd']), model=kwargs['model'], mode=kwargs.get('mode', 'agent'),
            status='running', log_path=str(runtime.log_path(kwargs['background_id'])),
            record_path=str(runtime.record_path(kwargs['background_id'])), started_at='now', command=tuple(command))
        runtime.save_record(record)
        return record


class RestartTests(unittest.TestCase):
    def test_restart_rebuilds_the_command(self) -> None:
        launched = _Launched()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(BackgroundSessionRuntime, 'launch', lambda self, cmd, **kw: launched(self, cmd, **kw)):
            runtime = BackgroundSessionRuntime(Path(tmp) / 'bg')
            save_record(runtime, 'killed', tmp)
            record = runtime.restart(BG, prompt='finish the docs too')
            self.assertEqual(record.background_id, BG)
            self.assertEqual(record.note, 'restarted')
            command = launched.calls[0]['command']
            self.assertEqual(command[command.index('agent-bg-worker') + 2], 'finish the docs too')
            self.assertEqual(command.count('--resume-session'), 1)

    def test_running_agent_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = BackgroundSessionRuntime(Path(tmp) / 'bg')
            save_record(runtime, 'running', tmp)
            with self.assertRaises(RuntimeError):
                runtime.restart(BG)

    def test_worker_continues_the_saved_session(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [text_reply('step one done'), text_reply('continued')]
        with tempfile.TemporaryDirectory() as tmp, patch('sys.stdout', io.StringIO()), \
                patch.dict(os.environ, {'SIGIL_BACKGROUND_LOG_MAX_BYTES': '0'}), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            root = Path(tmp) / 'bg'
            runtime = BackgroundSessionRuntime(root)
            save_record(runtime, 'running', tmp)
            first = build_parser().parse_args(['agent-bg-worker', BG, 'step one', '--background-root', str(root),
                                               '--cwd', tmp])
            _run_background_worker(first)
            again = build_parser().parse_args(['agent-bg-worker', BG, 'keep going', '--background-root', str(root),
                                               '--cwd', tmp, '--resume-session'])
            _run_background_worker(again)
        self.assertIn('step one', json.dumps(recorded[1]['messages']))  # the old conversation was continued
        self.assertIn('keep going', json.dumps(recorded[1]['messages']))

    def test_tool_and_cli(self) -> None:
        launched = _Launched()
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(BackgroundSessionRuntime, 'launch', lambda self, cmd, **kw: launched(self, cmd, **kw)):
            ws = Path(tmp)
            runtime = BackgroundSessionRuntime(ws / '.port_sessions' / 'background')
            save_record(runtime, 'completed', tmp)
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', api_key='sk-key'),
                                     runtime_config=AgentRuntimeConfig(cwd=ws))
            result = execute_tool(agent.tool_registry, 'background_agent_restart', {'background_id': BG},
                                  agent.tool_context)
            self.assertTrue(result.ok, result.content)
            self.assertEqual(launched.calls[0]['env']['OPENAI_API_KEY'], 'sk-key')
            unknown = execute_tool(agent.tool_registry, 'background_agent_restart',
                                   {'background_id': 'bg_0000000000ff'}, agent.tool_context)
            self.assertFalse(unknown.ok)
            save_record(runtime, 'killed', tmp)
            out = io.StringIO()
            with patch('src.main.BackgroundSessionRuntime', lambda: runtime), patch('sys.stdout', out):
                code = main(['agent-restart', BG, '--prompt', 'again'])
            self.assertEqual(code, 0)
            self.assertIn('status=running', out.getvalue())


if __name__ == '__main__':
    unittest.main()
