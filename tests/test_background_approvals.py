"""Approval questions from background agents, answered in the main chat or with
`sigil agent-approve`."""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.background_approvals import answer, answer_pending, ask, ask_path, pending
from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime
from src.main import _run_agent_chat_loop, build_parser, main


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason':
                                             'tool_calls' if message.get('tool_calls') else 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _bash(command: str) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'c1', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({'command': command})}}]}


def wait_for_request(root: Path, seconds: float = 15.0) -> list[dict]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = pending(root)
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError('no approval request appeared')


class RelayTests(unittest.TestCase):
    def test_round_trip_timeout_and_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            box: dict = {}
            worker = threading.Thread(target=lambda: box.update(answer=ask(
                root, 'bg_1', {'tool': 'bash', 'arguments': {'command': 'make'}, 'reason': 'shell is off'},
                timeout=15)))
            worker.start()
            request = wait_for_request(root)[0]
            self.assertEqual((request['background_id'], request['tool']), ('bg_1', 'bash'))
            self.assertFalse(answer(root, 'bg_2', 'y'))  # nobody else is waiting
            self.assertTrue(answer(root, 'bg_1', 's'))
            worker.join(10)
            self.assertEqual(box['answer'], 'session')
            self.assertFalse(ask_path(root, 'bg_1').exists())
            self.assertIsNone(ask(root, 'bg_1', {'tool': 'bash'}, timeout=0.3))
            self.assertEqual(pending(root), [])
            with patch.dict(os.environ, {'SIGIL_BACKGROUND_ASK_TIMEOUT': '0'}):
                self.assertIsNone(ask(root, 'bg_1', {'tool': 'bash'}))


class AgentTests(unittest.TestCase):
    def _agent(self, ws: Path, root: Path, replies: list[dict], recorded: list[dict]) -> LocalCodingAgent:
        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0))

        patcher = patch('src.openai_compat.request.urlopen', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        agent = LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
            runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s' / 'agent', nudge_strategy='off'))
        agent.background_ask = lambda request: ask(root, 'bg_t', request, timeout=15)
        return agent

    def test_answer_in_the_main_chat_lets_the_call_run(self) -> None:
        for reply, runs in (('y', True), ('n', False)):
            recorded: list[dict] = []
            with tempfile.TemporaryDirectory() as tmp:
                ws, root = Path(tmp), Path(tmp) / 'bg'
                root.mkdir()
                agent = self._agent(ws, root, [_bash('echo approved-run'),
                                               {'role': 'assistant', 'content': 'done'}], recorded)
                worker = threading.Thread(target=agent.run, args=('run it',))
                worker.start()
                wait_for_request(root)
                shown: list[str] = []
                self.assertEqual(answer_pending(root, lambda _prompt: reply, shown.append), 1)
                worker.join(30)
                self.assertIn('Background agent bg_t asks to use bash: echo approved-run', shown[0])
                tool = [m for m in recorded[-1]['messages'] if m.get('role') == 'tool'][-1]['content']
                self.assertEqual('approved-run' in tool, runs, tool)

    def test_chat_loop_asks_before_the_next_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, root = Path(tmp), Path(tmp) / 'bg'
            root.mkdir()
            box: dict = {}
            worker = threading.Thread(target=lambda: box.update(answer=ask(
                root, 'bg_c', {'tool': 'bash', 'arguments': {'command': 'git push'}}, timeout=15)))
            worker.start()
            wait_for_request(root)
            inputs = iter(['a'])

            def scripted(_label: str) -> str:
                try:
                    return next(inputs)
                except StopIteration:
                    raise EOFError from None

            output: list[str] = []
            agent = LocalCodingAgent(model_config=ModelConfig(model='m'),
                                     runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s'))
            with patch('src.background_runtime.DEFAULT_BACKGROUND_DIR', root), \
                    patch('src.main.HAS_PROMPT_TOOLKIT', False):
                _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                                     input_func=scripted, output_func=output.append)
            worker.join(10)
            self.assertEqual(box['answer'], 'always')
            self.assertTrue(any('asks to use bash: git push' in line for line in output), output)

    def test_cli_and_worker_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws, root = Path(tmp).resolve(), Path(tmp).resolve() / 'bg'
            runtime = BackgroundSessionRuntime(root)
            runtime.save_record(BackgroundSessionRecord(
                background_id='bg_w1', pid=os.getpid(), prompt='p', workspace_cwd=str(ws), model='m',
                mode='agent', status='running', log_path=str(root / 'bg_w1.log'),
                record_path=str(root / 'bg_w1.json'), started_at='now', command=()))
            replies = [_bash('echo from-worker'), {'role': 'assistant', 'content': 'worker done'}]
            recorded: list[dict] = []

            def fake(req, timeout=None):  # noqa: ANN001
                recorded.append(json.loads(req.data.decode('utf-8')))
                return _Resp(replies.pop(0))

            args = build_parser().parse_args(['agent-bg-worker', 'bg_w1', 'run it', '--background-root', str(root),
                                              '--cwd', str(ws), '--model', 'm',
                                              '--base-url', 'http://127.0.0.1:8000/v1'])
            from src.main import _run_background_worker

            with patch.dict(os.environ, {}), patch('src.main._cap_worker_output'), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake), redirect_stdout(io.StringIO()):
                worker = threading.Thread(target=_run_background_worker, args=(args,))
                worker.start()
                try:
                    wait_for_request(root)
                    # the worker posts the question first and sets its note just after: wait for it
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline and runtime.load_record('bg_w1').note is None:
                        time.sleep(0.05)
                    self.assertEqual(runtime.load_record('bg_w1').note, 'waiting for approval: bash')
                    with patch('src.background_runtime.DEFAULT_BACKGROUND_DIR', root):
                        out = io.StringIO()
                        with redirect_stdout(out):
                            self.assertEqual(main(['agent-approve']), 0)
                            self.assertEqual(main(['agent-approve', 'bg_w1', 'once']), 0)
                            self.assertEqual(main(['agent-approve', 'bg_nope', 'once']), 1)
                        self.assertIn('asks to use bash: echo from-worker', out.getvalue())
                finally:
                    worker.join(15)
                    if worker.is_alive():  # still waiting (an assertion failed first): release it
                        answer(root, 'bg_w1', 'n')
                        worker.join(30)
                self.assertFalse(worker.is_alive())
            self.assertEqual(runtime.read_result('bg_w1')['final_output'], 'worker done')
            tool = [m for m in recorded[-1]['messages'] if m.get('role') == 'tool'][-1]['content']
            self.assertIn('from-worker', tool)
            self.assertIsNone(runtime.load_record('bg_w1').note)


if __name__ == '__main__':
    unittest.main()
