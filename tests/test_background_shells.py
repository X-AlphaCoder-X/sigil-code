"""3.6 background shells: bash run_in_background, bash_output, bash_kill, /bashes.

Runs real (short) shell processes; the model side goes through a fake server.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_tools import build_tool_context, default_tool_registry, execute_tool
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.background_shells import shell_manager


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


def cleanup_dir(tmp: tempfile.TemporaryDirectory) -> None:
    # Windows can hold a stopped process's log file open for a moment.
    for _ in range(50):
        try:
            tmp.cleanup()
            return
        except PermissionError:
            time.sleep(0.1)
    tmp.cleanup()


def wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


class BackgroundShellToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self._tmp.name)
        config = AgentRuntimeConfig(
            cwd=self.ws,
            permissions=AgentPermissions(allow_shell_commands=True),
            scratchpad_root=self.ws / '.port_sessions' / 'scratchpad',
        )
        self.context = build_tool_context(config)
        self.registry = default_tool_registry()

    def tearDown(self) -> None:
        shell_manager().kill_all()
        cleanup_dir(self._tmp)

    def run_tool(self, name: str, arguments: dict[str, object]):
        return execute_tool(self.registry, name, arguments, self.context)

    def test_start_read_kill(self) -> None:
        started = time.time()
        result = self.run_tool('bash', {'command': 'echo first-line; sleep 60; echo never', 'run_in_background': True})
        self.assertTrue(result.ok, result.content)
        self.assertLess(time.time() - started, 10, 'background start must return at once')
        job_id = result.metadata['shell_id']
        job = shell_manager().get(job_id)
        self.assertTrue(wait_for(lambda: 'first-line' in job.output_path.read_text(errors='replace')))
        output = self.run_tool('bash_output', {'id': job_id})
        self.assertIn('status=running', output.content)
        self.assertIn('first-line', output.content)
        again = self.run_tool('bash_output', {'id': job_id})
        self.assertIn('(no new output)', again.content)
        killed = self.run_tool('bash_kill', {'id': job_id})
        self.assertIn('killed', killed.content)
        self.assertTrue(wait_for(lambda: job.process.poll() is not None))
        self.assertNotIn('never', job.output_path.read_text(errors='replace'))

    def test_finished_job_reports_exit_code(self) -> None:
        result = self.run_tool('bash', {'command': 'echo done-now; exit 3', 'run_in_background': True})
        job = shell_manager().get(result.metadata['shell_id'])
        self.assertTrue(wait_for(lambda: job.process.poll() is not None))
        output = self.run_tool('bash_output', {'id': job.job_id})
        self.assertIn('status=failed exit_code=3', output.content)
        self.assertIn('done-now', output.content)

    def test_shell_permission_still_applies(self) -> None:
        config = AgentRuntimeConfig(cwd=self.ws)  # shell off
        context = build_tool_context(config)
        result = execute_tool(self.registry, 'bash', {'command': 'echo x', 'run_in_background': True}, context)
        self.assertFalse(result.ok)
        self.assertIn('--allow-shell', result.content)

    def test_job_limit(self) -> None:
        with patch.dict(os.environ, {'SIGIL_MAX_BACKGROUND_SHELLS': '1'}):
            first = self.run_tool('bash', {'command': 'sleep 60', 'run_in_background': True})
            second = self.run_tool('bash', {'command': 'sleep 60', 'run_in_background': True})
        self.assertTrue(first.ok, first.content)
        self.assertFalse(second.ok)
        self.assertIn('SIGIL_MAX_BACKGROUND_SHELLS', second.content)

    def test_unknown_id(self) -> None:
        result = self.run_tool('bash_output', {'id': 'bash_does_not_exist'})
        self.assertFalse(result.ok)
        self.assertIn('No background shell', result.content)


class BackgroundShellAgentTests(unittest.TestCase):
    def tearDown(self) -> None:
        shell_manager().kill_all()

    def test_agent_starts_and_lists_jobs(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [
            tool_reply('bash', {'command': 'echo serving; sleep 60', 'run_in_background': True}),
            text_reply('server started'),
        ]
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(cleanup_dir, holder)
        with patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(holder.name)
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m'),
                runtime_config=AgentRuntimeConfig(
                    cwd=ws, permissions=AgentPermissions(allow_shell_commands=True),
                    session_directory=ws / '.port_sessions' / 'agent',
                    scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
            )
            result = agent.run('start the server')
            self.assertEqual(result.final_output, 'server started')
            listing = agent.run('/bashes').final_output
            tasks = agent.run('/tasks').final_output
            self.assertIn('echo serving; sleep 60', listing)
            self.assertIn('[running', listing)
            self.assertIn('# Background shells', tasks)
            job_id = [j.job_id for j in shell_manager().jobs.values() if j.exit_code is None][0]
            stopped = agent.run(f'/bashes kill {job_id}').final_output
            self.assertIn('killed', stopped)
            shell_manager().kill_all()

    def test_forked_children_are_stopped(self) -> None:
        # 'sleep' is not the last command, so Git Bash forks it into a process
        # outside the plain process tree; stopping the job must still end it.
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(cleanup_dir, holder)
        manager = shell_manager()
        from src.agent_tools import _BASH_EXECUTABLE

        job = manager.start('echo go; sleep 60; echo never', cwd=Path(holder.name), env=None,
                            bash=_BASH_EXECUTABLE, output_dir=Path(holder.name))
        self.assertTrue(wait_for(lambda: 'go' in job.output_path.read_text(errors='replace')))
        manager.kill(job.job_id)
        released = wait_for(lambda: _can_remove(job.output_path), timeout=5)
        self.assertTrue(released, 'a child process still holds the output file')


def _can_remove(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except PermissionError:
        return False


if __name__ == '__main__':
    unittest.main()
