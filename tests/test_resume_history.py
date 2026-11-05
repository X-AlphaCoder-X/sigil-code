"""What a resumed session prints as its earlier conversation: what the user typed and what the agent
answered, not the system prompt or the runtime's own <system-reminder> messages."""
from __future__ import annotations

import unittest

from src.main import _history_lines, _shown_in_history

# The shape of a real saved session: system prompt, session context, a prompt, an MCP status note,
# a resume hint, another prompt and the answer.
SESSION = [
    {'role': 'system', 'content': 'You are Sigil Code, an open-source Python coding agent. ...'},
    {'role': 'user', 'content': '<system-reminder>\nAs you answer the user\'s questions, you can use the following context:\n# currentDate'},
    {'role': 'user', 'content': 'Reply with exactly: GLM_OK'},
    {'role': 'assistant', 'content': 'GLM_OK', 'tool_calls': []},
    {'role': 'user', 'content': '<system-reminder>\nThe MCP servers changed since this conversation last described them.'},
    {'role': 'system', 'content': '<system-reminder>\nYOU HAVE ALREADY PLANNED.'},
    {'role': 'tool', 'content': 'file contents', 'tool_call_id': 't1'},
    {'role': 'user', 'content': '  Hi'},
    {'role': 'assistant', 'content': ''},
    {'role': 'user', 'content': [{'type': 'text', 'text': 'look at this'}, {'type': 'image_url', 'image_url': {'url': 'data:'}}]},
]


class ResumeHistoryTests(unittest.TestCase):
    def test_only_the_conversation_is_printed(self) -> None:
        lines = _history_lines(SESSION)
        self.assertEqual(lines[:3], ['user> Reply with exactly: GLM_OK', 'agent> GLM_OK', 'user>   Hi'])
        self.assertTrue(lines[3].startswith('user> [{'))            # a message with an image is still shown
        self.assertEqual(len(lines), 4)                             # the empty answer prints nothing
        printed = '\n'.join(lines)
        for hidden in ('You are Sigil Code', 'system-reminder', 'currentDate', 'MCP servers changed',
                       'ALREADY PLANNED', 'file contents'):
            self.assertNotIn(hidden, printed)

    def test_filter(self) -> None:
        self.assertEqual([_shown_in_history(m) for m in SESSION],
                         [False, False, True, True, False, False, False, True, True, True])
        self.assertFalse(_shown_in_history('not a message'))
        self.assertEqual(_history_lines(None), [])


if __name__ == '__main__':
    unittest.main()
