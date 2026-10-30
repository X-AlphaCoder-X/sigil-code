"""Small desktop helpers: clipboard copy and "done / needs you" notifications.

Clipboard: clip (Windows), pbcopy (macOS), wl-copy or xclip (Linux), else the
OSC 52 escape. SIGIL_CLIPBOARD=off turns it off (the copy still goes to a file).

Notifications (SIGIL_NOTIFY): bell (default: terminal bell plus an OSC 9
message, which Windows Terminal, iTerm2 and others show), toast (also a
Windows balloon / notify-send / osascript), or off. A finished run notifies
only when it took at least SIGIL_NOTIFY_AFTER seconds (default 30).
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from typing import Any, TextIO
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


def copy_to_clipboard(text: str, stream: TextIO | None = None) -> str | None:
    """Copy text; returns how ('clip', 'pbcopy', 'osc52', ...) or None."""
    if os.environ.get('SIGIL_CLIPBOARD', '').strip().lower() in {'off', '0', 'no', 'false'}:
        return None
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    candidates: list[tuple[list[str], bytes]] = []
    if sys.platform == 'win32':
        candidates.append((['clip'], text.encode('utf-16')))
    elif sys.platform == 'darwin':
        candidates.append((['pbcopy'], text.encode('utf-8')))
    else:
        candidates.append((['wl-copy'], text.encode('utf-8')))
        candidates.append((['xclip', '-selection', 'clipboard'], text.encode('utf-8')))
    for command, data in candidates:
        if shutil.which(command[0]) is None:
            continue
        try:
            completed = subprocess.run(command, input=data, capture_output=True, timeout=10, creationflags=flags)
        except (OSError, subprocess.TimeoutExpired) as _ignored:
            _log_ignored(_ignored)
            continue
        if completed.returncode == 0:
            return command[0]
    out = stream or sys.stdout
    try:
        out.write('\033]52;c;' + base64.b64encode(text.encode('utf-8')).decode('ascii') + '\a')
        out.flush()
        return 'osc52'
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None


def notify(title: str, message: str, stream: Any = None) -> str | None:
    """Tell the user something finished or needs them. Returns the method used."""
    mode = os.environ.get('SIGIL_NOTIFY', 'bell').strip().lower()
    if mode in ('off', '0', 'no', 'false', ''):
        return None
    out = stream or sys.stdout
    try:
        from .terminal import strip_terminal_controls

        # why (S07): tool names and backend error text reach here; an OSC/ESC inside them must not
        # ride out to the terminal (OSC 52 clipboard write, cursor moves)
        banner = strip_terminal_controls(f'{title}: {message}').replace('\a', ' ')
        out.write('\a\033]9;' + banner + '\a')
        out.flush()
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    if mode != 'toast':
        return 'bell'
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        if sys.platform == 'win32':
            # why (S07): the title/message go in through the environment, never the script text, so a
            # quote (U+2018/U+2019 also close a single-quoted PowerShell string) cannot start a command
            script = (
                'Add-Type -AssemblyName System.Windows.Forms; '
                '$n = New-Object System.Windows.Forms.NotifyIcon; '
                '$n.Icon = [System.Drawing.SystemIcons]::Information; $n.Visible = $true; '
                '$n.ShowBalloonTip(5000, $env:SIGIL_NOTIFY_TITLE, $env:SIGIL_NOTIFY_MESSAGE, '
                "'Info'); Start-Sleep -Seconds 6; $n.Dispose()"
            )
            child_env = dict(os.environ, SIGIL_NOTIFY_TITLE=title, SIGIL_NOTIFY_MESSAGE=message)
            subprocess.Popen(['powershell', '-NoProfile', '-Command', script], creationflags=flags,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env)
        elif sys.platform == 'darwin':
            # osascript reads the text from argv (on run argv), not from the script source
            script = ('on run argv\ndisplay notification (item 1 of argv) with title (item 2 of argv)\nend run')
            subprocess.Popen(['osascript', '-e', script, message, title])
        elif shutil.which('notify-send'):
            subprocess.Popen(['notify-send', '--', title, message])
    except OSError as _ignored:
        _log_ignored(_ignored)
        return 'bell'
    return 'toast'


def notify_after_seconds() -> float:
    try:
        return float(os.environ.get('SIGIL_NOTIFY_AFTER', '') or 30)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return 30.0
