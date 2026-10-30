"""Shell guard is user-overridable: SIGIL_SHELL_GUARD=on|warn|off; bypass mode warns."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.bash_security import check_shell_security, shell_guard_mode
from src.permission_modes import apply_mode


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def fake_server(responses: list[dict[str, object]], recorded: list[dict[str, object]]):
    queued = [FakeHTTPResponse(payload) for payload in responses]

    def _fake_urlopen(request_obj, timeout=None):  # noqa: ANN001
        recorded.append(json.loads(request_obj.data.decode('utf-8')))
        return queued.pop(0)

    return _fake_urlopen


def tool_reply(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'call_1', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}]},
        'finish_reason': 'tool_calls'}]}


def text_reply(text: str) -> dict[str, object]:
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}


def env_without_guard(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != 'SIGIL_SHELL_GUARD'}
    env.update(extra)
    return env


BLOCKED = ': > outside.log'  # redirection outside the scratchpad: refused by default


class ShellGuardTests(unittest.TestCase):
    def test_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            with patch.dict(os.environ, env_without_guard(), clear=True):
                self.assertEqual(shell_guard_mode(), 'on')
                self.assertFalse(check_shell_security(BLOCKED, cwd=cwd)[0])
                allowed, warning = check_shell_security(BLOCKED, cwd=cwd, guard='warn')
                self.assertTrue(allowed)
                self.assertIn('Shell guard warning', warning)
            with patch.dict(os.environ, env_without_guard(SIGIL_SHELL_GUARD='off'), clear=True):
                self.assertEqual(check_shell_security(BLOCKED, cwd=cwd), (True, ''))
            with patch.dict(os.environ, env_without_guard(SIGIL_SHELL_GUARD='on'), clear=True):
                # The env setting wins over the caller (bypass mode passes 'warn').
                self.assertFalse(check_shell_security(BLOCKED, cwd=cwd, guard='warn')[0])
            with patch.dict(os.environ, env_without_guard(SIGIL_SHELL_GUARD='warn'), clear=True):
                self.assertTrue(check_shell_security(BLOCKED, cwd=cwd)[0])
                # The destructive guard is separate and still needs --unsafe.
                self.assertFalse(check_shell_security('rm -rf build', cwd=cwd)[0])

    def test_shell_off_still_refused(self) -> None:
        with patch.dict(os.environ, env_without_guard(SIGIL_SHELL_GUARD='off'), clear=True):
            allowed, message = check_shell_security('ls', allow_shell=False)
        self.assertFalse(allowed)
        self.assertIn('--allow-shell', message)

    def test_bypass_mode_runs_with_warning(self) -> None:
        recorded: list[dict[str, object]] = []
        responses = [tool_reply('bash', {'command': 'echo saved > out.txt'}), text_reply('done')]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env_without_guard(), clear=True), \
                patch('src.openai_compat.request.urlopen', side_effect=fake_server(responses, recorded)):
            ws = Path(tmp)
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='m'),
                runtime_config=AgentRuntimeConfig(
                    cwd=ws, permissions=AgentPermissions(allow_shell_commands=True),
                    session_directory=ws / '.port_sessions' / 'agent',
                    scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
            )
            apply_mode(agent, 'bypass')
            agent.run('save it')
            # Refused in default mode (redirect outside the scratchpad); bypass runs it.
            self.assertEqual((ws / 'out.txt').read_text(encoding='utf-8').strip(), 'saved')


if __name__ == '__main__':
    unittest.main()
