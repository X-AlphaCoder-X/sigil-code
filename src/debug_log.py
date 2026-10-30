"""Log file for finding problems after the fact.

Always on: errors, tracebacks, backend failures, failed tool calls and session
start/end go to one file per day in ``~/.sigil/logs/`` (``sigil-YYYY-MM-DD.log``).
With ``--debug`` or ``SIGIL_DEBUG=1`` the raw request and response bodies are
logged too (truncated; secrets in them are replaced by [REDACTED], see redact.py).

Housekeeping (always on): each file is capped (``SIGIL_LOG_MAX_BYTES``, default
5 MB, 3 rollovers), and log files older than ``SIGIL_LOG_RETENTION_DAYS``
(default 2) are deleted when logging starts and then once an hour while it runs.

``SIGIL_LOG_DIR`` moves the folder; ``SIGIL_LOG_DIR=`` (empty) turns the file off.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import time
from datetime import date
from pathlib import Path

LOGGER_NAME = 'sigil'
_MAX_BODY_CHARS = 20_000

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
_state: dict[str, object] = {'path': None, 'debug': False}


def get_logger(name: str = '') -> logging.Logger:
    return logging.getLogger(f'{LOGGER_NAME}.{name}' if name else LOGGER_NAME)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, '') or default)
    except ValueError:
        return default


def log_dir() -> Path | None:
    raw = os.environ.get('SIGIL_LOG_DIR')
    if raw is not None and not raw.strip():
        return None
    return Path(raw).expanduser() if raw else Path.home() / '.sigil' / 'logs'


def remove_old_logs(folder: Path, days: int | None = None) -> int:
    """Delete sigil-*.log files older than `days` (default SIGIL_LOG_RETENTION_DAYS, 2)."""
    days = _env_int('SIGIL_LOG_RETENTION_DAYS', 2) if days is None else days
    cutoff = time.time() - max(days, 0) * 86400
    removed = 0
    try:
        entries = list(folder.glob('sigil-*.log*'))
    except OSError:
        return 0
    for path in entries:
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


class RedactingFormatter(logging.Formatter):
    """Every log line goes through redact.redact_text: environment secrets, keys Sigil was given
    (--api-key, profiles), common key formats, Bearer tokens and key=value assignments become
    [REDACTED]. --debug still logs request and response bodies, without the secrets in them."""

    def format(self, record: logging.LogRecord) -> str:
        from .redact import redact_text

        return redact_text(super().format(record))


class _RotatingFile(logging.handlers.RotatingFileHandler):
    """A rotating log that several processes share: when the rename fails (Windows, another
    Sigil process has the file open) it keeps writing to the same file and tries again a minute
    later, instead of printing a logging traceback."""

    _retry_at = 0.0
    _next_cleanup = 0.0  # time.monotonic() of the next removal of old logs

    def emit(self, record: logging.LogRecord) -> None:
        now = time.monotonic()
        if now >= self._next_cleanup:  # a session that runs for days still drops old logs
            self._next_cleanup = now + 3600
            remove_old_logs(Path(self.baseFilename).parent)
        super().emit(record)

    def _open(self):  # noqa: ANN202
        stream = super()._open()
        if os.name == 'posix':  # owner-only: the log can hold request bodies with --debug
            try:
                os.chmod(self.baseFilename, 0o600)
            except OSError:
                pass  # not our file to change (a shared folder): writing still works
        return stream

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        import time

        if time.monotonic() < self._retry_at:
            return 0
        return super().shouldRollover(record)

    def doRollover(self) -> None:  # noqa: N802
        import time

        try:
            super().doRollover()
        except OSError:
            self._retry_at = time.monotonic() + 60
            if self.stream is None:
                self.stream = self._open()


def setup_logging(*, debug: bool | None = None, debug_file: str | None = None) -> Path | None:
    """Start the log file once per process. Returns its path (None when off)."""
    if debug is None:
        debug = os.environ.get('SIGIL_DEBUG', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    logger = get_logger()
    level = logging.DEBUG if debug else logging.INFO
    if _state['path'] is not None and not debug_file:
        logger.setLevel(min(logger.level or level, level))
        _state['debug'] = bool(_state['debug']) or debug
        return _state['path']  # type: ignore[return-value]
    if debug_file:
        path = Path(debug_file).expanduser()
        folder = path.parent
    else:
        folder = log_dir()
        if folder is None:
            return None
        path = folder / f'sigil-{date.today().isoformat()}.log'
    try:
        folder.mkdir(parents=True, exist_ok=True)
        remove_old_logs(folder)
        handler = _RotatingFile(
            path, maxBytes=_env_int('SIGIL_LOG_MAX_BYTES', 5 * 1024 * 1024), backupCount=3,
            encoding='utf-8', delay=True,
        )
    except OSError:
        return None
    handler.setFormatter(RedactingFormatter('%(asctime)s %(levelname)s %(name)s [pid %(process)d] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _state['path'] = path
    _state['debug'] = debug
    logger.info('logging started (debug=%s)', debug)
    return path


def current_log_path() -> Path | None:
    return _state['path']  # type: ignore[return-value]


def debug_enabled() -> bool:
    return bool(_state['debug']) and get_logger().isEnabledFor(logging.DEBUG)


def sanitize_log(value: object, max_len: int = 200) -> str:
    """Text from the model, a tool or a file, made safe for one log line: CR/LF/tab become
    spaces (no forged log lines) and it is cut at max_len."""
    if value is None:
        return 'None'
    text = str(value).replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    return text if len(text) <= max_len else text[:max_len] + '...[truncated]'


def url_for_log(url: object) -> str:
    """A URL without its query and fragment (they can carry keys or tokens), for logs."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(str(url))
        bare = f'{parts.scheme}://{parts.hostname or ""}{parts.path}' if parts.scheme else parts.path
    except ValueError:
        bare = '<unparsable url>'
    return sanitize_log(bare)


# place -> [last time logged (monotonic), errors there since that line]
_IGNORED_STATE: dict[str, list[float]] = {}
_IGNORED_LOCK = __import__('threading').Lock()  # why: reader threads and subagents log at once
IGNORED_ERROR_INTERVAL = 60.0


def log_ignored_error(exc: BaseException) -> None:
    """An error the code carries on after is still written to the log file (~/.sigil/logs,
    always on; never shown in the terminal), so it can be found later. Each place logs at once
    the first time; after that at most one line a minute, with how many happened since, so every
    error shows up within a minute and a loop cannot flood the file."""
    import sys

    frame = sys._getframe(1)
    where = f'{Path(frame.f_code.co_filename).name}:{frame.f_lineno}'
    now = time.monotonic()
    forgotten = None
    with _IGNORED_LOCK:
        state = _IGNORED_STATE.get(where)
        if state is None:
            if len(_IGNORED_STATE) >= 512:  # bounded: forget the oldest place (its count is written)
                old_where = next(iter(_IGNORED_STATE))
                forgotten = (old_where, _IGNORED_STATE.pop(old_where))
            _IGNORED_STATE[where] = [now, 0, type(exc).__name__]
            repeats = 0
        else:
            state[1] += 1
            state[2] = type(exc).__name__
            if now - state[0] < IGNORED_ERROR_INTERVAL:
                return
            repeats, state[0], state[1] = int(state[1]), now, 0
    if forgotten is not None:
        _write_pending(*forgotten)
    extra = f' ({repeats} times since the last line)' if repeats else ''
    get_logger('ignored').info('error ignored at %s%s: %s: %s', where, extra,
                               type(exc).__name__, sanitize_log(exc))


def _write_pending(where: str, state: list) -> None:
    if len(state) > 1 and state[1]:
        get_logger('ignored').info('error ignored at %s: %d more time(s) since the last line (last %s)',
                                   where, state[1], state[2] if len(state) > 2 else 'error')


def flush_ignored_errors() -> None:
    """Write the counts that are still waiting for their once-a-minute line (at exit, so a burst
    that stopped is not lost)."""
    with _IGNORED_LOCK:
        pending = [(where, list(state)) for where, state in _IGNORED_STATE.items() if len(state) > 1 and state[1]]
        for _where, state in _IGNORED_STATE.items():
            if len(state) > 1:
                state[1] = 0
    for where, state in pending:
        _write_pending(where, state)


__import__('atexit').register(flush_ignored_errors)  # runs before logging's own shutdown (LIFO)


def clip(text: object, limit: int = _MAX_BODY_CHARS) -> str:
    value = str(text)
    return value if len(value) <= limit else value[:limit] + f'... [{len(value) - limit} more chars]'
