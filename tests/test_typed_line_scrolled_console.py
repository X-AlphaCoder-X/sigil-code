"""Regression: deleting the queued-input line must not leave it stuck on screen
when the console view is scrolled (Windows consoles clamp ESC[nA to the window)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from src.message_queue import ConsoleInputReader, MessageQueue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = textwrap.dedent(r'''
    import ctypes, ctypes.wintypes as w, sys
    sys.path.insert(0, sys.argv[2])
    from src.message_queue import ConsoleInputReader, MessageQueue

    k = ctypes.windll.kernel32
    h = k.GetStdHandle(-11)

    class COORD(ctypes.Structure):
        _fields_ = [("X", w.SHORT), ("Y", w.SHORT)]

    class SR(ctypes.Structure):
        _fields_ = [("L", w.SHORT), ("T", w.SHORT), ("R", w.SHORT), ("B", w.SHORT)]

    class CSBI(ctypes.Structure):
        _fields_ = [("size", COORD), ("cur", COORD), ("attr", w.WORD), ("win", SR), ("max", COORD)]

    def info():
        sys.stdout.flush()
        i = CSBI()
        k.GetConsoleScreenBufferInfo(h, ctypes.byref(i))
        return i

    def echo(t):
        sys.stdout.write(t)
        sys.stdout.flush()

    k.SetConsoleMode(h, 0x7)  # VT processing on, as for the agent's colours
    width = info().size.X
    if len(sys.argv) > 3 and sys.argv[3] == 'no-scrollback':
        # a buffer no taller than its window (ConPTY: Windows Terminal, VS Code; CI consoles):
        # every wrap at the last row scrolls the whole buffer
        w0 = info().win
        k.SetConsoleScreenBufferSize(h, COORD(width, w0.B - w0.T + 1))
    for _ in range(60):
        echo("filler\n")
    start = info().cur.Y
    reader = ConsoleInputReader(MessageQueue(), echo=echo)
    reader._win_console = True
    n = width + 10                      # wraps onto a second row
    for ch in "ok " + "s" * (n - 3):
        reader.feed(ch)
    i = info()                          # user scrolls: cursor row at top of view
    height = i.win.B - i.win.T
    k.SetConsoleWindowInfo(h, True, ctypes.byref(SR(0, i.cur.Y, i.win.R - i.win.L, i.cur.Y + height)))
    for _ in range(n):
        reader.feed("\b")
    reader.output("end\n")
    rows = []
    for y in range(start - 2, info().cur.Y + 1):
        buf = ctypes.create_unicode_buffer(width)
        got = w.DWORD()
        k.ReadConsoleOutputCharacterW(h, buf, width, COORD(0, y), ctypes.byref(got))
        rows.append(buf.value[:got.value].rstrip())
    open(sys.argv[1], "w", encoding="utf-8").write("\n".join(rows))
''')


@unittest.skipUnless(sys.platform == 'win32', 'needs a real Windows console')
class ScrolledConsoleTests(unittest.TestCase):
    def screen(self, *mode: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, 'child.py')
            result = os.path.join(tmp, 'screen.txt')
            with open(script, 'w', encoding='utf-8') as fh:
                fh.write(CHILD)
            subprocess.run(
                [sys.executable, script, result, ROOT, *mode],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                timeout=60,
            )
            with open(result, encoding='utf-8') as fh:
                return fh.read()

    def test_deleted_line_is_gone_on_a_console_without_scrollback(self) -> None:
        """Seen on GitHub's Windows runner (and true of ConPTY): the buffer scrolled under the typed line,
        its recorded row went stale, and the scrolled view hid the pencil from the row scan."""
        self.assert_clean(self.screen('no-scrollback'))

    def test_deleted_line_is_gone_even_when_view_is_scrolled(self) -> None:
        self.assert_clean(self.screen())

    def assert_clean(self, screen: str) -> None:
        self.assertIn('end', screen)
        self.assertNotIn('ok s', screen)
        self.assertNotIn('sss', screen)
        # The clear must write spaces, not a garbage fill character (a str
        # passed where a wchar is expected marshals as a pointer and fills
        # the rows with one junk code point).
        for row in screen.splitlines():
            self.assertTrue(
                row == '' or 'filler' in row or 'end' in row,
                'unexpected leftover text: %r' % row,
            )


class ClearFallbackTests(unittest.TestCase):
    def test_falls_back_to_escape_codes_when_console_clear_fails(self) -> None:
        out: list[str] = []
        reader = ConsoleInputReader(MessageQueue(), echo=out.append)
        reader._win_console = True
        for ch in 'abc':
            reader.feed(ch)
        out.clear()
        with mock.patch('src.message_queue._win_clear_rows', return_value=False):
            reader.feed('\b')
        self.assertEqual(''.join(out), '\033[?25l\r\033[J\u270e ab\033[?25h')

    def test_uses_console_clear_on_windows_console(self) -> None:
        out: list[str] = []
        reader = ConsoleInputReader(MessageQueue(), echo=out.append)
        reader._win_console = True
        for ch in 'abc':
            reader.feed(ch)
        out.clear()
        with mock.patch('src.message_queue._win_clear_rows', return_value=True) as clear:
            reader.feed('\b')
        clear.assert_called_once_with(reader._typed_start_row)
        self.assertEqual(''.join(out), '\033[?25l\u270e ab\033[?25h')

    def test_retyping_after_deleting_all_stays_on_the_same_row(self) -> None:
        """Regression: after deleting the whole line the cursor sits at column
        0 of the cleared row; the next keystroke used to write '\\n' first, so
        every delete-all + retype cycle marched the pencil one row lower."""
        out: list[str] = []
        reader = ConsoleInputReader(MessageQueue(), echo=out.append)
        reader._win_console = True
        for ch in 'ab':
            reader.feed(ch)
        reader.feed('\b')   # one char left
        out.clear()
        reader.feed('\b')   # delete-all: cursor parked at column 0
        out.clear()
        with mock.patch('src.message_queue._win_cursor_col', return_value=0):
            reader.feed('x')  # retype: must NOT write a newline first
        self.assertEqual(''.join(out), '\u270e x')
        # Mid-row cursor (agent output ended without a newline): newline back.
        out.clear()
        with mock.patch('src.message_queue._win_cursor_col', return_value=7):
            reader.feed('\b')
            out.clear()
            reader.feed('y')
        self.assertTrue(''.join(out).startswith('\n\u270e '))


if __name__ == '__main__':
    unittest.main()
