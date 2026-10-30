from __future__ import annotations

import io
import sys
import unittest

from src.agent_runtime import INVALID_ARGUMENTS_KEY, _parse_tool_arguments
from src.message_queue import ConsoleInputReader, MessageQueue


class TypedLineStaysAtBottomTests(unittest.TestCase):
    def _reader(self):
        out: list[str] = []
        reader = ConsoleInputReader(MessageQueue(), echo=out.append)
        return reader, out

    def test_output_passes_straight_through_when_not_typing(self) -> None:
        reader, out = self._reader()
        reader.output('hello\n')
        self.assertEqual(''.join(out), 'hello\n')

    def test_agent_output_is_printed_above_the_typed_line(self) -> None:
        reader, out = self._reader()
        for ch in 'fix':
            reader.feed(ch)
        out.clear()
        reader.output('agent says hi')       # no newline yet: held back
        self.assertEqual(out, [])
        reader.output(' there\nnext')
        rendered = ''.join(out)
        # line cleared, agent text written, typed text redrawn below it
        # (cursor hidden during the clear/redraw so the line does not bounce)
        self.assertTrue(rendered.startswith('\033[?25l\r\033[Jagent says hi there\n'))
        self.assertTrue(rendered.endswith('\u270e fix\033[?25h'))
        out.clear()
        reader.feed('\r')                    # submit: typed line erased, held output flushed
        self.assertEqual(''.join(out), '\r\033[Jnext')
        self.assertEqual(reader.queue.drain(), ['fix'])

    def test_wrapped_typed_line_is_fully_cleared_before_redraw(self) -> None:
        """A typed line longer than the terminal wraps onto several rows; all of
        them must be erased, otherwise the old copy stays on screen each redraw."""
        from unittest import mock
        reader, out = self._reader()
        for ch in 'x' * 25:                   # '\u270e ' + 25 chars = 27 cols
            reader.feed(ch)
        out.clear()
        with mock.patch('src.message_queue.shutil.get_terminal_size',
                        return_value=__import__('os').terminal_size((10, 24))):
            reader.output('reply\n')
        rendered = ''.join(out)
        # 27 cols on a 10-col terminal = 3 rows: move up 2, clear to end of screen
        self.assertTrue(rendered.startswith('\033[?25l\r\033[2A\033[Jreply\n'))
        self.assertTrue(rendered.endswith('\u270e ' + 'x' * 25 + '\033[?25h'))

    def test_submitting_erases_the_typed_line(self) -> None:
        """Regression: Enter used to write '\n' under the typed line, leaving
        the '✎' copy in scrollback above the queued-message notice."""
        reader, out = self._reader()
        for ch in 'hello':
            reader.feed(ch)
        out.clear()
        reader.feed('\r')
        self.assertEqual(''.join(out), '\r\033[J')   # erased, no '\n' under it
        self.assertEqual(reader.queue.drain(), ['hello'])

    def test_stop_erases_partial_line_instead_of_stranding_it(self) -> None:
        """Regression: on completion the typed line used to be left stranded
        above the final output (stop() wrote '\n' under it)."""
        reader, out = self._reader()
        for ch in 'draft':
            reader.feed(ch)
        out.clear()
        partial = reader.stop()
        self.assertEqual(partial, 'draft')
        self.assertEqual(''.join(out), '\r\033[J')  # erased, nothing stranded

    def test_stop_without_partial_line_writes_nothing(self) -> None:
        reader, out = self._reader()
        out.clear()
        self.assertEqual(reader.stop(), '')
        self.assertEqual(''.join(out), '')

    def test_backspace_moves_back_up_across_a_wrapped_row(self) -> None:
        from unittest import mock
        reader, out = self._reader()
        with mock.patch('src.message_queue.shutil.get_terminal_size',
                        return_value=__import__('os').terminal_size((10, 24))):
            for ch in 'x' * 9:                # '\u270e ' + 9 = 11 cols: 2 rows
                reader.feed(ch)
            out.clear()
            reader.feed('\b')                 # back to 10 cols: fits on 1 row
        # erase both rows (up 1, clear down), redraw the shorter line on one row
        self.assertEqual(''.join(out), '\033[?25l\r\033[1A\033[J\u270e ' + 'x' * 8 + ' \b\033[?25h')
        self.assertEqual(reader.stop(), 'x' * 8)

    def test_deleting_a_wrapped_line_leaves_no_stuck_row(self) -> None:
        """Regression: a line exactly filling the row used to clear one row too
        few on consoles that wrap at the margin, leaving the top row on screen."""
        from unittest import mock
        reader, out = self._reader()
        with mock.patch('src.message_queue.shutil.get_terminal_size',
                        return_value=__import__('os').terminal_size((10, 24))):
            for ch in 'x' * 9:                # 11 cols: 2 rows
                reader.feed(ch)
            reader.feed('\b')                 # 10 cols: exactly one full row
            out.clear()
            reader.feed('\b')                 # cursor was parked on row 2: must go up 1
            self.assertEqual(''.join(out), '\033[?25l\r\033[1A\033[J\u270e ' + 'x' * 7 + '\033[?25h')
            for _ in range(7):
                reader.feed('\b')
        self.assertEqual(reader.stop(), '')

    def test_typing_to_exact_row_width_parks_cursor_on_next_row(self) -> None:
        from unittest import mock
        reader, out = self._reader()
        with mock.patch('src.message_queue.shutil.get_terminal_size',
                        return_value=__import__('os').terminal_size((10, 24))):
            for ch in 'x' * 8:                # 10 cols: fills the row
                reader.feed(ch)
            self.assertTrue(''.join(out).endswith('x \b'))
            out.clear()
            reader.output('reply\n')
        self.assertTrue(''.join(out).startswith('\033[?25l\r\033[1A\033[Jreply\n'))

    def test_redirect_and_restore_stdout(self) -> None:
        reader, out = self._reader()

        class Display:
            def __init__(self) -> None:
                self.out = sys.stdout

            def on_event(self, event) -> None:
                pass

        class Agent:
            pass

        original = sys.stdout
        try:
            sys.stdout = io.StringIO()
            display = Display()
            agent = Agent()
            agent.event_callback = display.on_event
            before = sys.stdout
            reader.redirect_output(agent)
            self.assertIsNot(sys.stdout, before)
            self.assertIs(display.out, sys.stdout)
            print('via print')
            self.assertIn('via print\n', ''.join(out))
            reader.restore_output()
            self.assertIs(sys.stdout, before)
            self.assertIs(display.out, before)
        finally:
            sys.stdout = original


class BadToolJsonTests(unittest.TestCase):
    def test_valid_json_is_parsed(self) -> None:
        self.assertEqual(_parse_tool_arguments('{"path": "a.py"}'), {'path': 'a.py'})

    def test_truncated_json_becomes_tool_error_not_exception(self) -> None:
        args = _parse_tool_arguments('{"path": "src/main.py"')
        self.assertIn(INVALID_ARGUMENTS_KEY, args)
        self.assertIn('Call the tool again', args[INVALID_ARGUMENTS_KEY])
        self.assertNotIn('path', args)  # never run a half-parsed call

    def test_non_object_json_becomes_tool_error(self) -> None:
        self.assertIn(INVALID_ARGUMENTS_KEY, _parse_tool_arguments('[1, 2]'))


if __name__ == '__main__':
    unittest.main()
