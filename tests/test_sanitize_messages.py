from __future__ import annotations

import json
import unittest

from src.agent_session import INTERRUPTED_TOOL_RESULT, sanitize_openai_messages


class SanitizeMessagesTests(unittest.TestCase):
    def test_truncated_arguments_and_missing_result_are_repaired(self) -> None:
        messages = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'c1', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path": "tests/test_compact.py"'}},
            ]},
            {'role': 'user', 'content': 'continue'},
        ]
        out = sanitize_openai_messages(messages)
        self.assertEqual([m['role'] for m in out], ['user', 'assistant', 'tool', 'user'])
        self.assertEqual(json.loads(out[1]['tool_calls'][0]['function']['arguments']), {})
        self.assertEqual(out[2]['tool_call_id'], 'c1')
        self.assertEqual(out[2]['content'], INTERRUPTED_TOOL_RESULT)
        # original is not mutated
        self.assertEqual(messages[1]['tool_calls'][0]['function']['arguments'], '{"path": "tests/test_compact.py"')

    def test_valid_history_is_unchanged_and_empty_assistant_dropped(self) -> None:
        good = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'x', 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"a": 1}'}},
            ]},
            {'role': 'tool', 'content': 'ok', 'tool_call_id': 'c1', 'name': 'f'},
            {'role': 'assistant', 'content': ''},
        ]
        out = sanitize_openai_messages(good)
        self.assertEqual(out, good[:3])


if __name__ == '__main__':
    unittest.main()
