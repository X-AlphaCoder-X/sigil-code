"""Short-lived Google Cloud access tokens for Vertex AI (OpenAI-compatible endpoint and the explicit
context cache calls).

Vertex takes an OAuth access token as the bearer key, and such a token expires (an hour for
`gcloud auth print-access-token`). Sigil used to send whatever was in OPENAI_API_KEY until it
expired. Now, when SIGIL_GOOGLE_TOKEN_COMMAND is set (for example `gcloud auth print-access-token`),
the token comes from that command and is fetched again before it gets old
(SIGIL_GOOGLE_TOKEN_REFRESH_SECONDS, default 2700 = 45 minutes). The command runs without a shell;
its output is the token. The token is registered with redact.register_secret, kept in memory only,
and never written to a log, a session file or a cache name. Without the setting the configured key
is used unchanged.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time

from .debug_log import get_logger

_LOCK = threading.Lock()
_TOKEN: dict[str, object] = {'value': '', 'at': 0.0, 'command': ''}
DEFAULT_REFRESH_SECONDS = 2700.0


def _refresh_seconds() -> float:
    raw = os.environ.get('SIGIL_GOOGLE_TOKEN_REFRESH_SECONDS', '').strip()
    if not raw:
        return DEFAULT_REFRESH_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        from .prompt_cache import _warn_once

        _warn_once('google-refresh', f'SIGIL_GOOGLE_TOKEN_REFRESH_SECONDS is not a number ({exc}); using 2700.')
        return DEFAULT_REFRESH_SECONDS
    return min(max(value, 60.0), 3300.0)  # a token from gcloud lives an hour: refresh before that


def token_command() -> str:
    return os.environ.get('SIGIL_GOOGLE_TOKEN_COMMAND', '').strip()


def _run(command: str) -> str:
    args = shlex.split(command, posix=(os.name != 'nt'))
    if os.name == 'nt':  # non-POSIX splitting keeps the quotes around a quoted argument
        args = [a[1:-1] if len(a) >= 2 and a[0] == a[-1] and a[0] in '"\'' else a for a in args]
    if os.name == 'nt' and args and not os.path.splitext(args[0])[1]:
        # gcloud is gcloud.cmd on Windows: let the system find the script without a shell string
        import shutil

        found = shutil.which(args[0])
        if found:
            args[0] = found
    result = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise OSError(f'token command exited with {result.returncode}')
    token = (result.stdout or '').strip().splitlines()[-1].strip() if (result.stdout or '').strip() else ''
    if not token:
        raise OSError('token command printed no token')
    return token


def access_token(fallback: str) -> str:
    """The bearer token for Vertex: from SIGIL_GOOGLE_TOKEN_COMMAND (cached, refreshed in time),
    else `fallback` (the configured key). A failed refresh keeps the last token while it is still
    young enough and reports the failure; with no token at all the configured key is used."""
    command = token_command()
    if not command:
        return fallback
    now = time.monotonic()
    with _LOCK:
        cached = str(_TOKEN['value'])
        fresh = cached and _TOKEN['command'] == command and now - float(_TOKEN['at']) < _refresh_seconds()
        if fresh:
            return cached
        try:
            token = _run(command)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            get_logger('google_auth').warning('SIGIL_GOOGLE_TOKEN_COMMAND failed (%s); %s', type(exc).__name__,
                                              'keeping the previous token' if cached else 'using the configured key')
            return cached or fallback
        from .redact import register_secret

        register_secret(token)
        _TOKEN.update(value=token, at=now, command=command)
        get_logger('google_auth').info('Google access token refreshed from SIGIL_GOOGLE_TOKEN_COMMAND')
        return token


def reset() -> None:
    """Forget the cached token (tests, or after a 401)."""
    with _LOCK:
        _TOKEN.update(value='', at=0.0, command='')
