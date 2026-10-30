from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.checkpoints import CheckpointStore
from src.permission_modes import apply_mode, next_mode, normalize_mode, plan_mode_block_message
from src.rewind import list_prompt_points, rewind_session
from src.session_store import load_agent_session


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


def _write_call(call_id: str, path: str, content: str) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function',
        'function': {'name': 'write_file', 'arguments': json.dumps({'path': path, 'content': content})},
    }]}


class CheckpointStoreTests(unittest.TestCase):
    def test_rewind_restores_and_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'a.txt').write_text('v1', encoding='utf-8')
            store = CheckpointStore(ws / '.cp', ws)
            first = store.begin()
            store.record_before('a.txt')
            (ws / 'a.txt').write_text('v2', encoding='utf-8')
            store.begin()
            store.record_before('a.txt')
            (ws / 'a.txt').write_text('v3', encoding='utf-8')
            store.record_before('new.txt')
            (ws / 'new.txt').write_text('x', encoding='utf-8')

            self.assertEqual(sorted(store.changed_files_since(first)), ['a.txt', 'new.txt'])
            result = store.rewind(first)
            self.assertEqual((ws / 'a.txt').read_text(encoding='utf-8'), 'v1')
            self.assertFalse((ws / 'new.txt').exists())
            self.assertEqual(result.deleted, ('new.txt',))
            self.assertEqual(store.ids(), [])


class RewindEndToEndTests(unittest.TestCase):
    def test_rewind_to_second_prompt_restores_files_and_conversation(self) -> None:
        replies = [
            _write_call('c1', 'notes.txt', 'first'), {'role': 'assistant', 'content': 'Wrote it.'},
            _write_call('c2', 'notes.txt', 'second'), {'role': 'assistant', 'content': 'Updated it.'},
        ]

        def fake_urlopen(req, timeout=None):  # noqa: ANN001
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            session_dir = ws / '.port_sessions' / 'agent'
            config = AgentRuntimeConfig(cwd=ws, session_directory=session_dir,
                                        permissions=AgentPermissions(allow_file_write=True))
            with patch('src.openai_compat.request.urlopen', side_effect=fake_urlopen):
                agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                         runtime_config=config)
                first = agent.run('create notes')
                stored = load_agent_session(first.session_id, directory=session_dir)
                agent.resume('change notes', stored)
            self.assertEqual((ws / 'notes.txt').read_text(encoding='utf-8'), 'second')

            stored = load_agent_session(first.session_id, directory=session_dir)
            points = list_prompt_points(stored.messages)
            self.assertEqual([p.text for p in points], ['create notes', 'change notes'])

            report, text = rewind_session(agent, first.session_id, 2, 'both')
            self.assertEqual(text, 'change notes')
            self.assertEqual((ws / 'notes.txt').read_text(encoding='utf-8'), 'first')
            after = load_agent_session(first.session_id, directory=session_dir)
            self.assertEqual([p.text for p in list_prompt_points(after.messages)], ['create notes'])
            self.assertIn('restored notes.txt', report)

            rewind_session(agent, first.session_id, 1, 'code')
            self.assertFalse((ws / 'notes.txt').exists())


class PermissionModeTests(unittest.TestCase):
    def test_cycle_and_aliases(self) -> None:
        self.assertEqual(next_mode('default'), 'acceptEdits')
        self.assertEqual(next_mode('acceptEdits'), 'plan')
        self.assertEqual(next_mode('plan'), 'default')
        self.assertEqual(normalize_mode('accept-edits'), 'acceptEdits')
        self.assertEqual(normalize_mode('yolo'), 'bypass')
        self.assertIsNone(normalize_mode('nope'))

    def test_plan_mode_blocks_writes_but_allows_reads(self) -> None:
        self.assertIsNotNone(plan_mode_block_message('write_file', {'path': 'x'}))
        self.assertIsNotNone(plan_mode_block_message('bash', {'command': 'rm -rf build'}))
        self.assertIsNone(plan_mode_block_message('bash', {'command': 'ls -la'}))
        self.assertIsNone(plan_mode_block_message('read_file', {'path': 'x'}))

    def test_apply_mode_switches_permissions_and_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = LocalCodingAgent(model_config=ModelConfig(model='m'),
                                     runtime_config=AgentRuntimeConfig(cwd=Path(tmp)))
            self.assertFalse(agent.runtime_config.permissions.allow_file_write)
            apply_mode(agent, 'acceptEdits')
            self.assertTrue(agent.runtime_config.permissions.allow_file_write)
            self.assertTrue(agent.tool_context.permissions.allow_file_write)
            apply_mode(agent, 'bypass')
            self.assertTrue(agent.runtime_config.permissions.allow_destructive_shell_commands)
            apply_mode(agent, 'default')
            self.assertFalse(agent.runtime_config.permissions.allow_file_write)

    def test_plan_mode_refuses_write_tool_in_a_run(self) -> None:
        replies = [_write_call('c1', 'x.txt', 'no'), {'role': 'assistant', 'content': 'Plan: 1. write x.txt'}]
        payloads: list[dict] = []

        def fake_urlopen(req, timeout=None):  # noqa: ANN001
            payloads.append(json.loads(req.data.decode()))
            return _Resp(replies.pop(0))

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            config = AgentRuntimeConfig(cwd=ws, session_directory=ws / 's',
                                        permissions=AgentPermissions(allow_file_write=True))
            with patch('src.openai_compat.request.urlopen', side_effect=fake_urlopen):
                agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                         runtime_config=config)
                apply_mode(agent, 'plan')
                agent.run('add x.txt')
            self.assertFalse((ws / 'x.txt').exists())
            self.assertIn('Plan mode is active', json.dumps(payloads[0]['messages']))
            self.assertIn('Plan mode is active: write_file is not allowed', json.dumps(payloads[1]['messages']))


if __name__ == '__main__':
    unittest.main()
