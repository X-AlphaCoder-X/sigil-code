"""Text typed or pasted as a turn ends waits for Enter instead of sending itself."""

import sys
import threading
import types
import unittest
from unittest import mock

from src.message_queue import ConsoleInputReader, MessageQueue


def _fake_msvcrt(text: str) -> types.ModuleType:
    keys = list(text)
    module = types.ModuleType('msvcrt')
    module.kbhit = lambda: bool(keys)
    module.getwch = lambda: keys.pop(0)
    return module


def _stopped_thread() -> threading.Thread:
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    return thread


class TurnEndDrainTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in ('_win_cursor_col', '_win_cursor_row', '_win_pencil_row'):
            patcher = mock.patch(f'src.message_queue.{name}', return_value=0)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _reader(self, **kwargs):
        queue = MessageQueue()
        reader = ConsoleInputReader(queue, echo=lambda _text: None, **kwargs)
        reader._win_console = True
        reader._thread = _stopped_thread()
        return queue, reader

    def test_paste_still_arriving_at_turn_end_becomes_the_draft(self) -> None:
        queue, reader = self._reader()
        reader.feed_burst(list('pasted line one\r\n'))
        rest = 'pasted line two\r\npasted line three'
        with mock.patch.dict(sys.modules, {'msvcrt': _fake_msvcrt(rest)}):
            partial = reader.stop()
        self.assertEqual(queue.drain(), [])  # nothing was sent
        self.assertEqual(partial, 'pasted line one\npasted line two\npasted line three')

    def test_esc_and_ctrl_enter_do_nothing_after_the_turn(self) -> None:
        calls = []
        queue, reader = self._reader(on_escape=lambda: calls.append('esc'),
                                     on_send_now=lambda: calls.append('send'))
        with mock.patch.dict(sys.modules, {'msvcrt': _fake_msvcrt('\x1b')}):
            reader.stop()
        reader._thread = _stopped_thread()
        with mock.patch.dict(sys.modules, {'msvcrt': _fake_msvcrt('hi\n')}):
            reader.stop()
        self.assertEqual(calls, [])
        self.assertEqual(queue.drain(), ['hi'])  # kept for the draft, not sent now

    def test_paused_draft_is_returned_by_stop(self) -> None:
        _queue, reader = self._reader()
        reader.feed_burst(list('half typed'))
        reader.pause()
        with mock.patch.dict(sys.modules, {'msvcrt': _fake_msvcrt('')}):
            self.assertEqual(reader.stop(), 'half typed')


class OneReaderAtATimeTests(unittest.TestCase):
    def setUp(self) -> None:
        for name in ('_win_cursor_col', '_win_cursor_row', '_win_pencil_row'):
            patcher = mock.patch(f'src.message_queue.{name}', return_value=0)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_long_paste_is_echoed_in_a_few_writes(self) -> None:
        writes = []
        reader = ConsoleInputReader(MessageQueue(), echo=writes.append)
        reader.feed_burst(list(('\u2502 table row with text \u2502\r\n' * 200).rstrip()))
        self.assertLess(len(writes), 5)
        self.assertEqual(''.join(reader._buffer).count('\n'), 199)

    def test_reader_thread_stops_taking_keys_once_stopped(self) -> None:
        queue = MessageQueue()
        reader = ConsoleInputReader(queue, echo=lambda _text: None)
        taken = []
        module = types.ModuleType('msvcrt')
        module.kbhit = lambda: True  # a paste that never ends

        def getwch():
            taken.append('x')
            if len(taken) == 50:
                reader._stop.set()  # stop() is called mid-paste
            return 'x'

        module.getwch = getwch
        with mock.patch.dict(sys.modules, {'msvcrt': module}):
            reader._run_windows()
        self.assertEqual(len(taken), 50)

    def test_stop_does_not_read_keys_while_the_thread_still_reads_them(self) -> None:
        reader = ConsoleInputReader(MessageQueue(), echo=lambda _text: None)
        reader._win_console = True
        busy = threading.Event()
        reader._thread = threading.Thread(target=busy.wait, daemon=True)
        reader._thread.start()
        self.addCleanup(busy.set)
        with mock.patch.object(threading.Thread, 'join'), \
                mock.patch.object(reader, '_drain_console_keys') as drain:
            reader.stop()
        drain.assert_not_called()


class TypeaheadDraftTests(unittest.TestCase):
    def test_stored_lines_come_back_as_one_draft_not_one_per_prompt(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.input.typeahead import store_typeahead
        from prompt_toolkit.key_binding.key_processor import KeyPress
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.output import DummyOutput

        from src.main import _take_typeahead

        with create_pipe_input() as pipe:
            session = PromptSession(input=pipe, output=DummyOutput())
            store_typeahead(session.input, [
                KeyPress('a', 'a'), KeyPress(Keys.ControlM, '\r'),
                KeyPress('b', 'b'), KeyPress(Keys.ControlJ, '\n'),
                KeyPress(Keys.BracketedPaste, 'c\nd'), KeyPress(Keys.ControlM, '\r'),
            ])
            self.assertEqual(_take_typeahead(session), 'a\nb\nc\nd')
            self.assertEqual(_take_typeahead(session), '')  # cleared, so nothing replays


if __name__ == '__main__':
    unittest.main()
