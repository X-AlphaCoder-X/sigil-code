from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent_session import AgentSessionState
from src.microcompact import MICROCOMPACT_KIND, microcompact_session


def _session(n_tools: int, size: int) -> AgentSessionState:
    state = AgentSessionState.create(['system'], 'start')
    for i in range(n_tools):
        call = {'id': f'call_{i}', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
        state.append_assistant('', (call,))
        state.append_tool('read_file', f'call_{i}', f'R{i}:' + 'x' * size)
    return state


class MicrocompactTests(unittest.TestCase):
    def test_offloads_old_large_results_keeps_hot_tail(self) -> None:
        state = _session(8, 10_000)
        with tempfile.TemporaryDirectory() as tmp:
            result = microcompact_session(state, Path(tmp), keep_recent=5, min_chars=8000)
            self.assertEqual(result.offloaded_count, 3)
            tools = [m for m in state.messages if m.role == 'tool']
            for m in tools[:3]:
                self.assertEqual(m.metadata['kind'], MICROCOMPACT_KIND)
                stored = Path(m.metadata['stored_path'])
                self.assertTrue(stored.read_text(encoding='utf-8').startswith(m.content.split('Preview:\n')[1][:3]))
                self.assertIn(str(stored), m.content)
                self.assertLess(len(m.content), 1500)
            for m in tools[3:]:
                self.assertEqual(len(m.content), 10_000 + len('R0:'))
            # Tool call / result pairing is untouched.
            api = state.to_openai_messages()
            self.assertEqual(sum(1 for m in api if m['role'] == 'tool'), 8)
            # Idempotent.
            self.assertEqual(microcompact_session(state, Path(tmp), keep_recent=5, min_chars=8000).offloaded_count, 0)

    def test_small_results_stay_inline(self) -> None:
        state = _session(8, 100)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(microcompact_session(state, Path(tmp), keep_recent=5, min_chars=8000).offloaded_count, 0)


if __name__ == '__main__':
    unittest.main()
