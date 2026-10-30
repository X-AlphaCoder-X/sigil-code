"""Batch 6 (P2): background agents - result file, limits, stall watchdog, log cap,
parent hooks, session approvals forwarded, worker cleanup."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import execute_tool
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime
from src.main import _CappedWriter, _run_background_worker, build_parser, build_permission_prompt

PY = sys.executable.replace('\\', '/')


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


def save_record(runtime: BackgroundSessionRuntime, background_id: str, status: str = 'running') -> None:
    runtime.log_path(background_id).parent.mkdir(parents=True, exist_ok=True)
    runtime.log_path(background_id).write_text('started\n', encoding='utf-8')
    runtime.save_record(BackgroundSessionRecord(
        background_id=background_id, pid=os.getpid(), prompt='p', workspace_cwd='.', model='m', mode='agent',
        status=status, log_path=str(runtime.log_path(background_id)),
        record_path=str(runtime.record_path(background_id)), started_at='now', command=()))


class _Record:
    background_id = 'bg_000000000abc'
    pid = 1
    log_path = 'log'


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


class WorkerTests(unittest.TestCase):
    def test_worker_writes_result_and_status_shows_it(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('All done.')], recorded)), \
                patch('sys.stdout', io.StringIO()), patch.dict(os.environ, {'SIGIL_BACKGROUND_LOG_MAX_BYTES': '0'}):
            ws = Path(tmp)
            root = ws / '.port_sessions' / 'background'
            runtime = BackgroundSessionRuntime(root)
            save_record(runtime, 'bg_000000000001')
            args = build_parser().parse_args(['agent-bg-worker', 'bg_000000000001', 'finish it',
                                              '--background-root', str(root), '--cwd', tmp])
            self.assertEqual(_run_background_worker(args), 0)
            result = runtime.read_result('bg_000000000001')
            self.assertEqual(result['final_output'], 'All done.')
            agent = make_agent(ws)
            status = execute_tool(agent.tool_registry, 'background_agent_status',
                                  {'background_id': 'bg_000000000001'}, agent.tool_context)
        self.assertIn('# Result', status.content)
        self.assertIn('All done.', status.content)

    def test_log_cap(self) -> None:
        stream = io.StringIO()
        writer = _CappedWriter(stream, {'written': 0}, limit=20)
        writer.write('0123456789')
        writer.write('0123456789')
        writer.write('more text')
        writer.write('even more')
        text = stream.getvalue()
        self.assertTrue(text.startswith('01234567890123456789'))
        self.assertEqual(text.count('[log size cap reached'), 1)
        self.assertNotIn('even more', text)


class LimitTests(unittest.TestCase):
    def test_max_running_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.background_runtime.BackgroundSessionRuntime.launch', return_value=_Record()), \
                patch.dict(os.environ, {'SIGIL_MAX_BACKGROUND_AGENTS': '2'}):
            ws = Path(tmp)
            runtime = BackgroundSessionRuntime(ws / '.port_sessions' / 'background')
            save_record(runtime, 'bg_000000000001')
            save_record(runtime, 'bg_000000000002')
            agent = make_agent(ws)
            result = execute_tool(agent.tool_registry, 'background_agent_start', {'prompt': 'x'}, agent.tool_context)
            self.assertFalse(result.ok)
            self.assertIn('SIGIL_MAX_BACKGROUND_AGENTS', result.content)
            with patch.dict(os.environ, {'SIGIL_MAX_BACKGROUND_AGENTS': '0'}):
                result = execute_tool(agent.tool_registry, 'background_agent_start', {'prompt': 'x'},
                                      agent.tool_context)
            self.assertTrue(result.ok, result.content)

    def test_turn_cap_setting(self) -> None:
        captured: dict[str, object] = {}

        def _launch(self, command, **kwargs):  # noqa: ANN001
            captured['command'] = command
            captured['env'] = kwargs.get('env')
            return _Record()

        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.background_runtime.BackgroundSessionRuntime.launch', _launch):
            agent = make_agent(Path(tmp))
            with patch.dict(os.environ, {'SIGIL_BACKGROUND_MAX_TURNS': '80'}):
                execute_tool(agent.tool_registry, 'background_agent_start', {'prompt': 'x', 'max_turns': 70},
                             agent.tool_context)
            high = captured['command'][captured['command'].index('--max-turns') + 1]
            execute_tool(agent.tool_registry, 'background_agent_start', {'prompt': 'x', 'max_turns': 70},
                         agent.tool_context)
            default = captured['command'][captured['command'].index('--max-turns') + 1]
        self.assertEqual(high, '70')
        self.assertEqual(default, '40')
        self.assertEqual(captured['env']['SIGIL_BACKGROUND'], '1')

    def test_stall_watchdog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = BackgroundSessionRuntime(Path(tmp))
            save_record(runtime, 'bg_000000000009')
            old = time.time() - 30 * 60
            os.utime(runtime.log_path('bg_000000000009'), (old, old))
            with patch.dict(os.environ, {'SIGIL_BACKGROUND_STALL_MINUTES': '10'}):
                self.assertIn('stalled: no output for 30 min', runtime.render_ps())
            with patch.dict(os.environ, {'SIGIL_BACKGROUND_STALL_MINUTES': '0'}):
                save_record(runtime, 'bg_00000000000a')
                os.utime(runtime.log_path('bg_00000000000a'), (old, old))
                self.assertNotIn('bg_00000000000a; status=running; pid=' + str(os.getpid()) + '; model=m; cwd=.; stalled',
                                 runtime.render_ps())
            with patch.dict(os.environ, {'SIGIL_BACKGROUND_STALL_MINUTES': '10', 'SIGIL_BACKGROUND_STALL_KILL': '1'}), \
                    patch.object(BackgroundSessionRuntime, 'kill', side_effect=lambda bid: runtime.load_record(bid)):
                runtime.render_ps()
                self.assertIn('(stopped)', runtime.load_record('bg_000000000009').note)


class ParentHookAndApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1', 'SIGIL_DISABLE_USER_SETTINGS': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_subagent_start_and_stop_for_background_agents(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('background_agent_start', {'prompt': 'work'}),
                     tool_reply('background_agent_status', {'background_id': _Record.background_id}),
                     tool_reply('background_agent_status', {'background_id': _Record.background_id}),
                     text_reply('ok')]
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.background_runtime.BackgroundSessionRuntime.launch', return_value=_Record()), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            runtime = BackgroundSessionRuntime(ws / '.port_sessions' / 'background')
            save_record(runtime, _Record.background_id, status='completed')
            log = ws / 'events.log'
            script = ws / 'hook.py'
            script.write_text('import json, sys\nd = json.load(sys.stdin)\n'
                              f'open({str(log)!r}, "a").write(d["hook_event_name"] + "\\n")\n', encoding='utf-8')
            command = f'"{PY}" "{script.as_posix()}"'
            (ws / '.sigil').mkdir()
            (ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': {
                event: [{'matcher': '*', 'hooks': [{'type': 'command', 'command': command}]}]
                for event in ('SubagentStart', 'SubagentStop')}}), encoding='utf-8')
            make_agent(ws).run('go')
            events = log.read_text(encoding='utf-8').split()
        self.assertEqual(events, ['SubagentStart', 'SubagentStop'])  # stop fires once

    def test_session_approval_is_reused_and_forwarded(self) -> None:
        recorded: list[dict[str, object]] = []
        captured: dict[str, object] = {}

        def _launch(self, command, **kwargs):  # noqa: ANN001
            captured['command'] = command
            return _Record()

        responses = [tool_reply('write_file', {'path': 'a.txt', 'content': '1'}),
                     tool_reply('write_file', {'path': 'a.txt', 'content': '2'}),
                     tool_reply('background_agent_start', {'prompt': 'more', 'allow_write': True}),
                     text_reply('done')]
        prompts: list[object] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.background_runtime.BackgroundSessionRuntime.launch', _launch), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agent = make_agent(ws)
            agent.permission_prompt = lambda request: (prompts.append(request), 'session')[1]
            agent.run('write twice')
            self.assertEqual((ws / 'a.txt').read_text(encoding='utf-8'), '2')
        self.assertEqual(len(prompts), 1)  # the second write used the session approval
        command = captured['command']
        self.assertIn('--allowed-tools', command)
        self.assertIn('Write(./a.txt)', command)

    def test_prompt_offers_session_choice(self) -> None:
        prompt = build_permission_prompt(lambda _label: 's', output_func=lambda _t: None)
        self.assertEqual(prompt({'tool': 'bash', 'arguments': {}, 'reason': 'r', 'rule': 'Bash(ls)'}), 'session')


class WorkerCleanupTests(unittest.TestCase):
    def test_worker_stops_its_background_shells(self) -> None:
        from src.background_shells import shell_manager

        stopped: list[int] = []
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server([text_reply('x')], recorded)), \
                patch('sys.stdout', io.StringIO()), patch.dict(os.environ, {'SIGIL_BACKGROUND_LOG_MAX_BYTES': '0'}), \
                patch.object(type(shell_manager()), 'kill_all', side_effect=lambda: stopped.append(1) or 0):
            root = Path(tmp) / 'bg'
            runtime = BackgroundSessionRuntime(root)
            save_record(runtime, 'bg_000000000002')
            args = build_parser().parse_args(['agent-bg-worker', 'bg_000000000002', 'x',
                                              '--background-root', str(root), '--cwd', tmp])
            _run_background_worker(args)
        self.assertEqual(stopped, [1])


if __name__ == '__main__':
    unittest.main()
