"""Log file: errors always, raw bodies with --debug, 3-day cleanup."""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src import debug_log
from src.main import main


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeStdin(io.StringIO):
    def isatty(self) -> bool:
        return True


def reset_logging() -> None:
    logger = debug_log.get_logger()
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            handler.close()
            logger.removeHandler(handler)
    debug_log._state['path'] = None
    debug_log._state['debug'] = False


class DebugLogTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_logging()
        self.addCleanup(reset_logging)

    def test_old_logs_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            old = folder / 'sigil-2020-01-01.log'
            new = folder / 'sigil-2026-10-28.log'
            other = folder / 'keep.txt'
            for path in (old, new, other):
                path.write_text('x', encoding='utf-8')
            four_days = time.time() - 4 * 86400
            os.utime(old, (four_days, four_days))
            os.utime(other, (four_days, four_days))
            removed = debug_log.remove_old_logs(folder)
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertTrue(other.exists())

    def test_setup_cleans_on_start_and_can_be_turned_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / 'sigil-2020-01-01.log'
            old.write_text('x', encoding='utf-8')
            os.utime(old, (time.time() - 5 * 86400,) * 2)
            with patch.dict(os.environ, {'SIGIL_LOG_DIR': tmp}):
                path = debug_log.setup_logging(debug=False)
                self.assertFalse(old.exists())
                self.assertEqual(path.parent, Path(tmp))
                reset_logging()
            with patch.dict(os.environ, {'SIGIL_LOG_DIR': ''}):
                self.assertIsNone(debug_log.setup_logging(debug=False))

    def test_errors_logged_and_bodies_only_with_debug(self) -> None:
        responses = [{'choices': [{'message': {'role': 'assistant', 'content': 'secret-answer-text'},
                                   'finish_reason': 'stop'}]}]

        def run(argv: list[str], tmp: str) -> str:
            queued = [FakeHTTPResponse(r) for r in responses]
            with patch.dict(os.environ, {'SIGIL_LOG_DIR': tmp}), \
                    patch('src.openai_compat.request.urlopen', side_effect=lambda *_a, **_k: queued.pop(0)), \
                    patch('sys.stdin', FakeStdin('')), redirect_stdout(io.StringIO()):
                main(argv)
            for handler in debug_log.get_logger().handlers:
                handler.flush()
            text = ''.join(p.read_text(encoding='utf-8') for p in Path(tmp).glob('sigil-*.log'))
            reset_logging()
            return text

        with tempfile.TemporaryDirectory() as plain, tempfile.TemporaryDirectory() as debug:
            plain_log = run(['-p', 'hi', '--cwd', plain, '--api-key', 'sk-never-logged'], plain)
            debug_log_text = run(['-p', 'hi', '--cwd', debug, '--debug', '--api-key', 'sk-never-logged'], debug)
        self.assertIn('start: print', plain_log)
        self.assertNotIn('secret-answer-text', plain_log)
        self.assertIn('secret-answer-text', debug_log_text)
        self.assertIn('request:', debug_log_text)
        self.assertNotIn('sk-never-logged', debug_log_text)

    def test_backend_error_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {'SIGIL_LOG_DIR': tmp}):
                debug_log.setup_logging(debug=False)
                from src.openai_compat import OpenAICompatError

                OpenAICompatError('HTTP 500 from local model backend: boom')
                for handler in debug_log.get_logger().handlers:
                    handler.flush()
                text = ''.join(p.read_text(encoding='utf-8') for p in Path(tmp).glob('sigil-*.log'))
                reset_logging()
        self.assertIn('backend error: HTTP 500', text)


if __name__ == '__main__':
    unittest.main()
