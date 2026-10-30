from __future__ import annotations

import unittest

from src.agent_runtime import LocalCodingAgent


def detect(text: str) -> bool:
    agent = LocalCodingAgent.__new__(LocalCodingAgent)
    return agent._detect_tool_omission(text)


class NudgeDetectionTests(unittest.TestCase):
    def test_final_reports_do_not_trigger(self) -> None:
        finished = [
            # Real replies that wrongly triggered the nudge before.
            "Nothing was left unfinished. All the tool calls from last turn ran, and the "
            "full test suite passed: 865 tests. Restart the agent to try it.\n\n"
            "Nothing is committed to git.",
            "You can now type to the agent while it's working. Type a line and press Enter; "
            "Ctrl+C still stops the task. Run it and test it when you restart.",
            "Done. I updated the config and all tests pass.",
            "Should I also add the rewind feature?",
            "Let me know if you want me to commit this.",
            "The heuristic checks the ending of the reply only.",
        ]
        for text in finished:
            with self.subTest(text=text[:40]):
                self.assertFalse(detect(text))

    def test_announced_but_not_executed_triggers(self) -> None:
        stalls = [
            "The bug is in the parser. Let me check the config file.",
            "Now I'll run the tests:",
            "I found the issue.\n\nNext, I will edit src/main.py to fix it.",
            "Let me look at how the interrupt is handled...",
        ]
        for text in stalls:
            with self.subTest(text=text[:40]):
                self.assertTrue(detect(text))


class EchoTests(unittest.TestCase):
    def test_typed_text_is_echoed_and_escape_interrupts(self) -> None:
        from src.message_queue import ConsoleInputReader, MessageQueue

        out: list[str] = []
        escapes: list[int] = []
        reader = ConsoleInputReader(MessageQueue(), echo=out.append, on_escape=lambda: escapes.append(1))
        for ch in 'hi\b!':
            reader.feed(ch)
        reader.feed('\x1b')  # clears the line, does not interrupt
        self.assertEqual(escapes, [])
        reader.feed('\x1b')  # empty line -> interrupt
        self.assertEqual(escapes, [1])
        echoed = ''.join(out)
        self.assertIn('h', echoed)
        self.assertIn('!', echoed)
        # backspace redraws the line as 'h' (cursor hidden during the redraw),
        # then '!' is typed
        self.assertIn('\033[?25l\r\033[J\u270e h\033[?25h!', echoed)


    def test_env_can_disable_or_choose_nudge_strategy(self) -> None:
        import os
        from unittest import mock
        from src.agent_runtime import _effective_nudge_strategy

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SIGIL_AUTO_NUDGE', None)
            os.environ.pop('SIGIL_NUDGE_STRATEGY', None)
            self.assertEqual(_effective_nudge_strategy('heuristic'), 'heuristic')  # default stays on
        for off in ('0', 'off', 'false', 'NO'):
            with mock.patch.dict(os.environ, {'SIGIL_AUTO_NUDGE': off}):
                self.assertEqual(_effective_nudge_strategy('heuristic'), 'off')
        with mock.patch.dict(os.environ, {'SIGIL_AUTO_NUDGE': '1', 'SIGIL_NUDGE_STRATEGY': 'structural'}):
            self.assertEqual(_effective_nudge_strategy('heuristic'), 'structural')
        with mock.patch.dict(os.environ, {'SIGIL_AUTO_NUDGE': '', 'SIGIL_NUDGE_STRATEGY': 'bogus'}):
            self.assertEqual(_effective_nudge_strategy('structural'), 'structural')


if __name__ == '__main__':
    unittest.main()
