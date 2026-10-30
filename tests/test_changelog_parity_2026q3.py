"""Parity tests for features from the Jul-Oct 2026 reference changelog."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agent_context import _discover_memory_files_for_directory
from src.agent_runtime import LocalCodingAgent
from src.agent_session import AgentSessionState
from src.agent_tools import (
    ToolExecutionContext,
    ToolExecutionError,
    _edit_file,
    _read_file,
    _web_fetch,
)
from src.agent_types import AgentPermissions
from src.bash_security import SecurityBehavior, bash_command_is_safe


def _ctx(root: Path, **kwargs) -> ToolExecutionContext:
    return ToolExecutionContext(
        root=root.resolve(),
        command_timeout_seconds=5,
        max_output_chars=100,
        permissions=AgentPermissions(allow_file_write=True),
        **kwargs,
    )


class ReadAllowLargeTests(unittest.TestCase):
    def test_allow_large_skips_normal_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'big.txt').write_text('x' * 1000, encoding='utf-8')
            ctx = _ctx(root)
            self.assertIn('[truncated]', _read_file({'path': 'big.txt'}, ctx))
            self.assertEqual(_read_file({'path': 'big.txt', 'allow_large': True}, ctx), 'x' * 1000)

    def test_allow_large_must_be_bool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.txt').write_text('a', encoding='utf-8')
            with self.assertRaises(ToolExecutionError):
                _read_file({'path': 'a.txt', 'allow_large': 'yes'}, _ctx(root))


class NonUtf8EditTests(unittest.TestCase):
    def test_edit_refuses_non_utf8_and_leaves_file_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = 'caf\xe9 hello'.encode('cp1252')
            (root / 'legacy.txt').write_bytes(original)
            with self.assertRaises(ToolExecutionError) as cm:
                _edit_file({'path': 'legacy.txt', 'old_text': 'hello', 'new_text': 'bye'}, _ctx(root))
            self.assertIn('not valid UTF-8', str(cm.exception))
            self.assertEqual((root / 'legacy.txt').read_bytes(), original)

    def test_edit_utf8_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ok.txt').write_text('caf\xe9 hello', encoding='utf-8')
            _edit_file({'path': 'ok.txt', 'old_text': 'hello', 'new_text': 'bye'}, _ctx(root))
            self.assertEqual((root / 'ok.txt').read_text(encoding='utf-8'), 'caf\xe9 bye')


class AgentsMdFallbackTests(unittest.TestCase):
    def test_agents_md_used_without_sigil_md(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'AGENTS.md').write_text('agents', encoding='utf-8')
            files = _discover_memory_files_for_directory(root)
            self.assertEqual([p.name for p in files], ['AGENTS.md'])

    def test_sigil_md_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'AGENTS.md').write_text('agents', encoding='utf-8')
            (root / 'SIGIL.md').write_text('sigil', encoding='utf-8')
            files = _discover_memory_files_for_directory(root)
            self.assertEqual([p.name for p in files], ['SIGIL.md'])


class RetryBackoffTests(unittest.TestCase):
    def test_overloaded_uses_overloaded_env(self) -> None:
        env = {
            'SIGIL_OVERLOADED_RETRY_BASE_DELAY_MS': '100',
            'SIGIL_OVERLOADED_RETRY_MAX_DELAY_MS': '150',
        }
        with mock.patch.dict(os.environ, env):
            exc = Exception('HTTP 529 from local model backend: overloaded')
            first = LocalCodingAgent._backend_retry_delay_seconds(1, exc)
            third = LocalCodingAgent._backend_retry_delay_seconds(3, exc)
        self.assertGreaterEqual(first, 0.1)
        self.assertLessEqual(first, 0.15)
        self.assertLessEqual(third, 0.15)  # capped by max

    def test_zero_delay_allowed(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_RETRY_BASE_DELAY_MS': '0'}):
            self.assertEqual(LocalCodingAgent._backend_retry_delay_seconds(2, Exception('timed out')), 0.0)


class WebLimitsTests(unittest.TestCase):
    def test_disable_web_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'SIGIL_DISABLE_WEB_FETCH': '1'}):
            with self.assertRaises(ToolExecutionError):
                _web_fetch({'url': 'https://example.com'}, _ctx(Path(tmp)))

    def test_web_search_session_cap(self) -> None:
        from src.agent_tools import _web_search

        session = AgentSessionState.create(['sys'], 'hi')
        session.web_search_count = 2
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {'SIGIL_MAX_WEB_SEARCHES_PER_SESSION': '2'}
        ):
            ctx = _ctx(Path(tmp), session=session, search_runtime=mock.Mock())
            with self.assertRaises(ToolExecutionError) as cm:
                _web_search({'query': 'x'}, ctx)
            self.assertIn('limit reached', str(cm.exception))


class BashSecurityTests(unittest.TestCase):
    def assertAsks(self, command: str) -> None:
        self.assertEqual(bash_command_is_safe(command).behavior, SecurityBehavior.ASK, command)

    def assertPasses(self, command: str) -> None:
        self.assertEqual(bash_command_is_safe(command).behavior, SecurityBehavior.PASSTHROUGH, command)

    def test_bash_argv0(self) -> None:
        self.assertAsks('BASH_ARGV0=rm; echo hi')

    def test_docker_daemon_redirect(self) -> None:
        self.assertAsks('docker -H tcp://evil:2375 ps')
        self.assertAsks('docker --host ssh://x run alpine')
        self.assertAsks('podman --remote ps')
        self.assertAsks('podman --connection prod ps')
        self.assertAsks('DOCKER_HOST=tcp://x docker ps')

    def test_plain_docker_ok(self) -> None:
        self.assertPasses('docker ps')
        self.assertPasses('docker run --rm alpine echo hi')
        self.assertPasses('docker run -c 512 alpine sh -c true')
        self.assertPasses('docker logs -H')  # subcommand flag, not the global --host


if __name__ == '__main__':
    unittest.main()
