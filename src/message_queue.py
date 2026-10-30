"""Type messages to the agent while it is working (queued input).

While a run is in progress a background thread reads keystrokes from the
console. Every line the user finishes with Enter is queued; the agent runtime
drains the queue before each model call and injects the text as a user
message, so the model sees it at the next step instead of after the run ends.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Callable
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

QUEUED_MESSAGE_KIND = 'queued_user_message'


class MessageQueue:
    def __init__(self, notify: Callable[[str], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._messages: list[str] = []
        self._notify = notify

    def put(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._messages.append(text)
        if self._notify is not None:
            try:
                self._notify(text)
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)

    def drain(self) -> list[str]:
        with self._lock:
            messages, self._messages = self._messages, []
        return messages

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._messages)


def format_queued_messages(messages: list[str]) -> str:
    body = '\n\n'.join(messages)
    return (
        'The user sent the following message while you were working:\n'
        f'{body}\n\n'
        'Take it into account and adjust what you are doing if needed.'
    )


class ConsoleInputReader:
    """Collects typed lines from the console into a MessageQueue during a run.

    Windows uses msvcrt (non-blocking key polling). POSIX uses select() on a
    canonical-mode tty, so the terminal itself echoes what is typed.
    """

    def __init__(
        self,
        queue: MessageQueue,
        *,
        poll_interval: float = 0.05,
        echo: Callable[[str], None] | None = None,
        on_escape: Callable[[], None] | None = None,
        on_send_now: Callable[[], None] | None = None,
    ) -> None:
        self.queue = queue
        # Ctrl+Enter (LF on a Windows console; Enter is CR): queue the line and stop the
        # current turn, so the queued messages go to the agent right away.
        self._on_send_now = on_send_now
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        # Set while something else (an ask_user_question or approval prompt)
        # reads the console; the reader thread leaves the keys alone.
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffer: list[str] = []
        # Agent output that arrived while the user was mid-line and has not
        # reached a newline yet (see output()).
        self._pending_output = ''
        self._lock = threading.RLock()
        # echo receives raw terminal output so typed text is visible (Windows
        # msvcrt does not echo). on_escape is called when Esc is pressed on an
        # empty line (used to interrupt the agent).
        self._echo = echo
        self._on_escape = on_escape
        # Set by start() when reading a real Windows console (see _clear_typed_line).
        self._win_console = False
        # Buffer row where the current typed line starts. Updated on every draw
        # so the clear never has to guess from character counts (the pencil and
        # other glyphs can take 2 cells on East-Asian consoles, which throws
        # row arithmetic off and leaves the top row of a wrapped line stuck).
        self._typed_start_row: int | None = None

    def _write(self, text: str) -> None:
        if self._echo is not None and text:
            try:
                self._echo(text)
            except Exception as _ignored:
                from .debug_log import log_ignored_error

                log_ignored_error(_ignored)

    def _hide_cursor(self) -> None:
        """Hide the caret while the typed line is cleared and redrawn.

        ESC[?25l only works when VT processing is on; on Windows consoles
        where it is off (or cannot be enabled), fall back to the Win32 API.
        """
        if sys.platform != 'win32':
            return
        _win_set_cursor_visible(False)

    def _show_cursor(self) -> None:
        if sys.platform != 'win32':
            return
        _win_set_cursor_visible(True)

    def output(self, text: str) -> None:
        """Print agent output without disturbing the line the user is typing.

        While the user has a partial line, completed output lines are printed
        above it and the typed line is redrawn at the bottom, so it stays
        where the user is typing instead of being pushed up by replies.
        """
        with self._lock:
            if not self._buffer:
                self._write(text)
                return
            self._pending_output += text
            if '\n' not in self._pending_output:
                return
            done, self._pending_output = self._pending_output.rsplit('\n', 1)
            # Preferred path: scroll the whole buffer so the output lands in
            # the rows the typed line vacates. The typed line is never erased
            # or redrawn — the pencil slides with the buffer while staying on
            # the same on-screen row — so it cannot flicker or bounce.
            if self._win_console and self._typed_start_row is not None:
                if _win_scroll_output_above(self._typed_start_row, done):
                    # The typed line did not move: only the rows above it
                    # scrolled, so the recorded start row stays valid.
                    return
            # Fallback: clear, print the output, and redraw the typed line in
            # ONE write so at most one intermediate frame can flash.
            clear = self._clear_typed_line()
            if self._win_console:
                self._hide_cursor()
            self._write('\033[?25l' + clear + done + '\n'
                        + '\u270e ' + ''.join(self._buffer) + self._park_cursor() + '\033[?25h')
            if self._win_console:
                # The pencil glyph's actual screen row, not character math
                # (the pencil is 2 cells wide on some consoles).
                self._typed_start_row = _win_pencil_row() or _win_cursor_row()
                self._show_cursor()

    @staticmethod
    def _width() -> int:
        try:
            return max(shutil.get_terminal_size().columns, 1)
        except Exception as _ignored:
            _log_ignored(_ignored)
            return 80

    def _typed_cols(self) -> int:
        return len('\u270e ') + len(self._buffer)

    def _park_cursor(self) -> str:
        """Called after drawing the typed line. When it exactly fills the last
        row, some consoles wrap the cursor to the next row and others leave it
        at the margin; ' \\b' forces the wrap so the cursor row is always known."""
        return ' \b' if self._typed_cols() % self._width() == 0 else ''

    def _clear_typed_line(self) -> str:
        """Erase the typed line, including rows it wrapped onto.

        Returns the escape sequence to write. On a Windows console the rows are
        cleared directly instead (and '' is returned): there, cursor-up is
        relative to the visible window, so it lands on the wrong row whenever
        the user has scrolled, leaving the typed line stuck on screen.

        The escape path is preferred when the typed line is inside the visible
        window: it lets the caller merge clear, output, and redraw into ONE
        console write, so no intermediate frame without the typed line is
        ever painted (that read as the line bouncing while output streamed).
        The cursor-up count comes from ABSOLUTE buffer rows (cursor row minus
        the pencil's row), never from character counts: the pencil glyph is
        2 cells on East-Asian consoles, so counting characters undershoots
        by a row and the top row of a wrapped line survived deletion (it
        stayed frozen on screen even after the whole line was removed).
        """
        if self._win_console:
            if _win_vt_enabled() and _win_rows_visible(self._typed_start_row):
                cur = _win_cursor_row()
                start = self._typed_start_row
                if start is not None and cur is not None and cur >= start:
                    up = '\033[%dA' % (cur - start) if cur > start else ''
                    return '\r' + up + '\033[J'
                # rows unknown or stale: character-count fallback below
            elif _win_clear_rows(self._typed_start_row):
                return ''
        rows_above = self._typed_cols() // self._width()  # cursor parked by _park_cursor()
        up = '\033[%dA' % rows_above if rows_above else ''
        return '\r' + up + '\033[J'

    def _record_typed_start(self) -> None:
        """Remember the buffer row where the typed line begins (Windows only).

        Prefers the pencil glyph's actual screen row over the cursor row:
        the pencil can occupy 2 cells, which shifts every character-count
        based row estimate by one and left the top row of a wrapped line
        frozen on screen after deletion.
        """
        if self._win_console:
            self._typed_start_row = _win_pencil_row() or _win_cursor_row()

    def _flush_pending_output(self) -> None:
        pending, self._pending_output = self._pending_output, ''
        self._write(pending)

    def redirect_output(self, agent: object | None = None) -> None:
        """Send sys.stdout (and the live display, if it holds its own stream)
        through output() until restore_output() is called."""
        proxy = _ReaderStdout(self, sys.stdout)
        self._restore: list[tuple[object, str, object]] = [(sys, 'stdout', sys.stdout)]
        display = getattr(getattr(agent, 'event_callback', None), '__self__', None)
        if display is not None and getattr(display, 'out', None) is sys.stdout:
            self._restore.append((display, 'out', display.out))
            display.out = proxy
        sys.stdout = proxy

    def restore_output(self) -> None:
        for owner, attr, original in reversed(getattr(self, '_restore', [])):
            setattr(owner, attr, original)
        self._restore = []


    @staticmethod
    def supported() -> bool:
        try:
            if not sys.stdin.isatty():
                return False
        except Exception as _ignored:
            _log_ignored(_ignored)
            return False
        if sys.platform == 'win32':
            try:
                import msvcrt  # noqa: F401
            except ImportError:
                return False
            return True
        try:
            import select  # noqa: F401
        except ImportError:
            return False
        return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._win_console = sys.platform == 'win32'
        target = self._run_windows if sys.platform == 'win32' else self._run_posix
        self._thread = threading.Thread(target=target, name='sigil-input-reader', daemon=True)
        self._thread.start()

    def stop(self) -> str:
        """Stop reading. Returns any partially typed (not yet submitted) text."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            partial = ''.join(self._buffer)
            self._buffer = []
            if partial:
                # Erase the typed line instead of leaving it stranded above the
                # final output; the text is returned so the caller can keep it
                # as a draft for the next prompt.
                self._write(self._clear_typed_line())
            self._flush_pending_output()
            self._typed_start_row = None
        return partial

    def pause(self) -> None:
        """Stop taking keys and print straight to the real console.

        Used while the agent asks the user something, so the answer goes to
        the question instead of the queue. Pair with resume().
        """
        self._paused.set()
        with self._lock:
            partial = ''.join(self._buffer)
            if partial:
                self._write(self._clear_typed_line())
            self._flush_pending_output()
            self._typed_start_row = None
            self._paused_draft = partial
            self._buffer = []
        # Remember the proxies so resume() can put them back.
        restore = list(getattr(self, '_restore', []))
        self._paused_swap = [(owner, attr, getattr(owner, attr), original)
                             for owner, attr, original in restore]
        self.restore_output()

    def resume(self) -> None:
        """Undo pause(): route output through the reader and take keys again."""
        swap = getattr(self, '_paused_swap', [])
        self._paused_swap = []
        self._restore = []
        for owner, attr, proxy, original in swap:
            self._restore.append((owner, attr, original))
            setattr(owner, attr, proxy)
        with self._lock:
            draft = getattr(self, '_paused_draft', '')
            self._paused_draft = ''
            for ch in draft:
                self._feed(ch)
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def feed(self, ch: str) -> None:
        """Process one typed character (separated out for testing)."""
        with self._lock:
            self._feed(ch)

    def _feed(self, ch: str) -> None:
        if ch == '\n' and self._on_send_now is not None:
            line = ''.join(self._buffer)
            self._buffer = []
            if line:
                self._write(self._clear_typed_line())
                self._flush_pending_output()
                self._typed_start_row = None
                self.queue.put(line)
            if self.queue.has_pending():
                self._on_send_now()
            return
        if ch in ('\r', '\n'):
            line = ''.join(self._buffer)
            self._buffer = []
            if line:
                # Erase the typed line so submitting leaves no '✎' copy behind
                # in scrollback; the queued-message notice then prints on a
                # clean row instead of under the typed text.
                self._write(self._clear_typed_line())
            self._flush_pending_output()
            self._typed_start_row = None
            self.queue.put(line)
        elif ch in ('\b', '\x7f'):
            if self._buffer:
                clear = self._clear_typed_line()
                self._buffer.pop()
                if self._buffer:
                    # Erase just the last character cell with the console API:
                    # no clear, no redraw, so the line cannot flicker. Falls
                    # back to the escape-based clear+redraw when the console
                    # API is unavailable or the cursor is mid-line.
                    if self._win_console and _win_erase_last_cell():
                        pass
                    else:
                        if self._win_console:
                            self._hide_cursor()
                        self._write('\033[?25l' + clear
                                    + '\u270e ' + ''.join(self._buffer) + self._park_cursor() + '\033[?25h')
                        if self._win_console:
                            self._typed_start_row = _win_pencil_row() or _win_cursor_row()
                            self._show_cursor()
                else:
                    self._write(self._clear_typed_line())
                    self._flush_pending_output()
                    self._typed_start_row = None
        elif ch == '\x1b':
            if self._buffer:
                self._write(self._clear_typed_line())
                self._buffer = []
                self._typed_start_row = None
                self._flush_pending_output()
            elif self._on_escape is not None:
                self._on_escape()
        elif ch >= ' ' or ch == '\t':
            if not self._buffer:
                # Deleting the whole line parks the cursor at column 0 of the
                # cleared row; writing '\n' then would start the pencil one row
                # LOWER on every retype, marching the line down the screen.
                # Only write the newline when the cursor is mid-row (agent
                # output or a prompt ended there without a newline).
                at_col0 = (not self._win_console) or (_win_cursor_col() == 0)
                self._write(('\n' if not at_col0 else '') + '\u270e ')
                self._record_typed_start()  # after the newline: the row the pencil landed on
            self._buffer.append(ch)
            self._write(ch + self._park_cursor())

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(self.poll_interval)
                continue
            try:
                while not self._paused.is_set() and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ('\x00', '\xe0'):
                        msvcrt.getwch()  # swallow the second half of arrow/function keys
                        continue
                    self.feed(ch)
            except Exception as _ignored:
                _log_ignored(_ignored)
                return
            time.sleep(self.poll_interval)

    def _run_posix(self) -> None:
        import select

        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(self.poll_interval)
                continue
            try:
                ready, _, _ = select.select([sys.stdin], [], [], self.poll_interval)
                if ready and self._paused.is_set():
                    continue  # leave the line for the prompt that paused us
                if ready:
                    line = sys.stdin.readline()
                    if not line:
                        return
                    self.queue.put(line)
            except Exception as _ignored:
                _log_ignored(_ignored)
                return


def _win_console_out_handle():
    """A usable console output handle: the std handle if it is a console,
    otherwise a fresh CONOUT$ handle (the std handle can be a pipe when the
    process was launched with redirected stdio). Returns None off a console.
    The caller must CloseHandle() non-None results."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if handle and kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return handle, kernel32, False  # borrowed: do not close
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
        ]
        opened = kernel32.CreateFileW(
            'CONOUT$', 0x80000000 | 0x40000000, 0, None, 3, 0, None
        )
        if opened and opened != wintypes.HANDLE(-1).value:
            return opened, kernel32, True
        return None, kernel32, False
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None, None, False


def _win_set_cursor_visible(visible: bool) -> None:
    """Show/hide the caret via the Win32 API (works even without VT mode)."""
    handle, kernel32, close = _win_console_out_handle()
    if handle is None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _CONSOLE_CURSOR_INFO(ctypes.Structure):
            _fields_ = [('dwSize', wintypes.DWORD), ('bVisible', wintypes.BOOL)]

        info = _CONSOLE_CURSOR_INFO()
        if not kernel32.GetConsoleCursorInfo(handle, ctypes.byref(info)):
            return
        if bool(info.bVisible) == visible:
            return
        info.bVisible = visible
        kernel32.SetConsoleCursorInfo(handle, ctypes.byref(info))
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    finally:
        if close:
            kernel32.CloseHandle(handle)


def _win_pencil_row() -> int | None:
    """Buffer row of the pencil glyph on the visible screen, or None.

    Scanning for the glyph is immune to cell-width math: the pencil is
    East-Asian-ambiguous (2 cells on some consoles), so deriving the line's
    row from character counts undershoots and the top row of a wrapped line
    survived deletion (it stayed frozen on screen after the line was gone).
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _SMALL_RECT(ctypes.Structure):
            _fields_ = [('Left', wintypes.SHORT), ('Top', wintypes.SHORT),
                        ('Right', wintypes.SHORT), ('Bottom', wintypes.SHORT)]

        class _CHAR_INFO(ctypes.Structure):
            _fields_ = [('Char', wintypes.WCHAR), ('Attributes', wintypes.WORD)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', _SMALL_RECT), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return None
        cols = info.window.Right - info.window.Left + 1
        rows = info.window.Bottom - info.window.Top + 1
        if cols <= 0 or rows <= 0:
            return None
        buf = (_CHAR_INFO * (cols * rows))()
        rect = _SMALL_RECT(info.window.Left, info.window.Top,
                           info.window.Right, info.window.Bottom)
        if not kernel32.ReadConsoleOutputW(
                handle, buf, _COORD(cols, rows), _COORD(0, 0), ctypes.byref(rect)):
            return None
        for r in range(rows - 1, -1, -1):  # bottom-up: the live line, not scrollback
            for c in range(cols):
                if buf[r * cols + c].Char == '\u270e':
                    return info.window.Top + r
        return None
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None


def _win_erase_last_cell() -> bool:
    """Erase the character cell left of the cursor and step back one cell.

    Used for backspace when the cursor sits at the end of the typed line: no
    clear and no redraw are needed, so the line cannot flicker. Returns False
    when the cursor is not at a line start or the API is unavailable.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', wintypes.SHORT * 4), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return False
        x, y = info.cursor.X, info.cursor.Y
        if x <= 0:
            return False  # wrapped to a row start: fall back to clear+redraw
        written = wintypes.DWORD()
        if not kernel32.FillConsoleOutputCharacterW(
                handle, ctypes.c_wchar(' '), 1, _COORD(x - 1, y),
                ctypes.byref(written)) or written.value != 1:
            return False
        return bool(kernel32.SetConsoleCursorPosition(handle, _COORD(x - 1, y)))
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def _win_output_rows(text: str, width: int) -> int:
    """Rows the text will occupy when printed (each line wraps at width)."""
    return sum(max(1, -(-len(line) // width)) for line in text.split('\n'))


def _win_scroll_output_above(start_row: int | None, text: str) -> bool:
    """Print completed output lines ABOVE the typed line without touching it.

    Scrolls ONLY the rows above the typed line up by the number of rows the
    output needs (exactly like the console scrolling when output fills the
    last row), then writes the output into the rows that scroll vacated —
    directly above the typed line. The pencil is never erased, never redrawn,
    and never changes row, so it cannot flicker or bounce. Returns False when
    not possible (caller falls back to clear+redraw).
    """
    if start_row is None or start_row <= 0:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _SMALL_RECT(ctypes.Structure):
            _fields_ = [('Left', wintypes.SHORT), ('Top', wintypes.SHORT),
                        ('Right', wintypes.SHORT), ('Bottom', wintypes.SHORT)]

        class _CHAR_INFO(ctypes.Structure):
            _fields_ = [('Char', wintypes.WCHAR), ('Attributes', wintypes.WWORD if False else wintypes.WORD)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', _SMALL_RECT), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return False
        width = max(info.size.X, 1)
        # Rows the output will occupy (each line wraps at the buffer width).
        out_lines = text.split('\n')
        rows_needed = sum(max(1, -(-len(l) // width)) for l in out_lines)
        if rows_needed <= 0:
            return False
        # The output lands in the rows the scroll vacates, directly above the
        # typed line: there must be that many rows above it to scroll away.
        if start_row < rows_needed:
            return False
        # Scroll the rows ABOVE the typed line up by rows_needed. The typed
        # line (and anything below it) is outside the rectangle, so it stays
        # exactly where it is on screen. The fill character only lands in the
        # vacated rows, which the output overwrites right away.
        scroll_rect = _SMALL_RECT(0, 0, width - 1, start_row - 1)
        fill = _CHAR_INFO(' ', info.attr)
        if not kernel32.ScrollConsoleScreenBufferW(
                handle, ctypes.byref(scroll_rect), None,
                _COORD(0, -rows_needed), ctypes.byref(fill)):
            return False
        # Write the output into the vacated rows: one line per row with
        # explicit positioning (WriteConsoleW treats '\n' as a bare line feed
        # that keeps the column, which would stagger the lines).
        row = start_row - rows_needed
        for line in out_lines:
            if not kernel32.SetConsoleCursorPosition(handle, _COORD(0, row)):
                return False
            if line:
                written = wintypes.DWORD()
                # WriteConsoleW counts UTF-16 CODE UNITS: a glyph outside the
                # BMP (e.g. emoji) is one Python char but two units, so
                # len(line) truncates the write mid-escape-sequence. The VT
                # parser is then left half-way through an escape and eats the
                # next write's leading characters — observed as the output
                # line losing its first chars and the pencil row being wiped.
                units = len(line.encode('utf-16-le')) // 2
                kernel32.WriteConsoleW(handle, line, units,
                                       ctypes.byref(written), None)
                if written.value != units:
                    return False
            row += max(1, -(-len(line) // width))
        # Put the cursor back where the typed line had it (just after the
        # typed text) so the next keystroke appends instead of overwriting.
        kernel32.SetConsoleCursorPosition(handle, info.cursor)
        return True
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def _win_vt_enabled() -> bool:
    """Whether VT processing is on for the console output handle."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(mode.value & 0x4)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def _win_rows_visible(start_row: int | None) -> bool:
    """Whether the typed line's rows are inside the visible window.

    Escape-based cursor-up is clamped to the visible window, so it only
    clears the right rows when the line is on screen. When the user has
    scrolled the view away from the line, the caller must fall back to the
    absolute-coordinate console API clear.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _SMALL_RECT(ctypes.Structure):
            _fields_ = [('Left', wintypes.SHORT), ('Top', wintypes.SHORT),
                        ('Right', wintypes.SHORT), ('Bottom', wintypes.SHORT)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', _SMALL_RECT), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return False
        if start_row is None:
            return True  # no recorded row: assume the line is on screen
        return info.window.Top <= start_row <= info.window.Bottom
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


def _win_cursor_col() -> int | None:
    """Current buffer column of the cursor, or None off a Windows console."""
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', wintypes.SHORT * 4), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return None
        return info.cursor.X
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None


def _win_cursor_row() -> int | None:
    """Current buffer row of the cursor, or None off a Windows console."""
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', wintypes.SHORT * 4), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return None
        return info.cursor.Y
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None


def _win_clear_rows(start_row: int | None) -> bool:
    """Blank rows from an absolute buffer row down through the cursor row.

    The start row is recorded when the typed line is drawn, so no character
    counting is involved and glyph widths (the pencil is 2 cells on
    East-Asian consoles) cannot skew the range. Unlike ESC[nA (clamped to
    the visible window, which breaks when the user scrolls the view), these
    coordinates are absolute. Returns False if not possible.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

        class _CSBI(ctypes.Structure):
            _fields_ = [('size', _COORD), ('cursor', _COORD), ('attr', wintypes.WORD),
                        ('window', wintypes.SHORT * 4), ('max_size', _COORD)]

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        info = _CSBI()
        if not kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return False
        width = max(info.size.X, 1)
        top = start_row if start_row is not None else max(info.cursor.Y - 1, 0)
        if top > info.cursor.Y:              # stale (buffer shifted): fall back
            top = max(info.cursor.Y - 1, 0)
        # Through the row after the cursor, like ESC[J (covers a parked cursor).
        rows = min(info.cursor.Y + 2, info.size.Y) - top
        if rows <= 0:
            return False
        start = _COORD(0, top)
        written = wintypes.DWORD()
        kernel32.FillConsoleOutputCharacterW(handle, ctypes.c_wchar(' '), width * rows, start, ctypes.byref(written))
        kernel32.FillConsoleOutputAttribute(handle, info.attr, width * rows, start, ctypes.byref(written))
        return bool(kernel32.SetConsoleCursorPosition(handle, start))
    except Exception as _ignored:
        _log_ignored(_ignored)
        return False


class _ReaderStdout:
    """File-like stdout stand-in that routes writes through the reader."""

    def __init__(self, reader: 'ConsoleInputReader', real: object) -> None:
        self._reader = reader
        self._real = real

    def write(self, text: str) -> int:
        self._reader.output(text)
        return len(text)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)

    def __getattr__(self, name: str):
        return getattr(self._real, name)
