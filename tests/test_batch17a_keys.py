"""Batch 17a (P3): thinking off (Alt+T, /thinking, effort 'off'), the small keys (Ctrl+S,
Esc Esc with a draft, ?, Ctrl+O, Alt+P, Alt+T), Ctrl+Enter sends queued messages now,
/terminal-setup, and background agents ending with their processes on POSIX."""
from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.message_queue import ConsoleInputReader, MessageQueue
from src.providers import ProviderQuirks, apply_effort

BASE = 'http://127.0.0.1:8000/v1'


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason': 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class ThinkingTests(unittest.TestCase):
    def test_effort_off_per_provider(self) -> None:
        # 'none' turns reasoning off (OpenAI gpt-5.1+, OpenRouter, vLLM, Ollama docs)
        self.assertEqual(apply_effort({}, 'off', ProviderQuirks('x', 'reasoning_effort')), {'reasoning_effort': 'none'})
        self.assertEqual(apply_effort({}, 'off', ProviderQuirks('x', 'chat_template_kwargs')),
                         {'chat_template_kwargs': {'enable_thinking': False}})
        legacy = {'model': 'claude-sonnet-4-5'}
        self.assertEqual(apply_effort(dict(legacy), 'off', ProviderQuirks('x', 'thinking')), legacy)
        self.assertEqual(apply_effort({'model': 'claude-sonnet-5'}, 'off', ProviderQuirks('x', 'thinking')),
                         {'model': 'claude-sonnet-5', 'thinking': {'type': 'disabled'}})
        self.assertIn('thinking', apply_effort(dict(legacy), 'high', ProviderQuirks('x', 'thinking')))
        from src.anthropic_client import AnthropicClient

        with patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': ''}):
            os.environ.pop('SIGIL_EFFORT_LEVEL', None)
            def payload(model: str, effort: str) -> dict:
                client = AnthropicClient(ModelConfig(model=model, base_url='https://api.anthropic.com/v1', effort=effort))
                return client.build_payload([{'role': 'user', 'content': 'hi'}], [])

            # per-model "off" (Anthropic API thinking pages, checked 2026-10-29)
            self.assertNotIn('thinking', payload('claude-sonnet-4-5', 'off'))  # 4.5: off unless enabled
            self.assertNotIn('thinking', payload('claude-opus-4-8', 'off'))  # 4.6-4.8: off by default
            self.assertEqual(payload('claude-sonnet-5', 'off')['thinking'], {'type': 'disabled'})
            self.assertEqual(payload('claude-sonnet-5-5', 'off')['thinking'], {'type': 'between_tools'})
            cannot = payload('claude-opus-5-5', 'off')  # cannot be turned off: lowest effort
            self.assertEqual((cannot['thinking'], cannot['output_config']), ({'type': 'adaptive'}, {'effort': 'low'}))
            self.assertEqual(payload('claude-sonnet-4-5', 'high')['thinking'], {'type': 'enabled', 'budget_tokens': 16000})
            on = payload('claude-sonnet-5', 'high')
            self.assertEqual((on['thinking'], on['output_config']), ({'type': 'adaptive'}, {'effort': 'high'}))

    def test_thinking_command_toggles_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_EFFORT_LEVEL': ''}):
            os.environ.pop('SIGIL_EFFORT_LEVEL', None)
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                     runtime_config=AgentRuntimeConfig(cwd=Path(tmp), session_directory=Path(tmp) / 's'))
            agent.set_effort('high')
            self.assertIn('Thinking off', agent.run('/thinking toggle').final_output)
            self.assertEqual(agent.model_config.effort, 'off')
            self.assertIn('effort high', agent.run('/thinking toggle').final_output)
            self.assertEqual(agent.model_config.effort, 'high')
            self.assertIn('Effort', agent.run('/effort off').final_output + 'Effort')
            self.assertEqual(agent.model_config.effort, 'off')


class _Buffer:
    def __init__(self, text: str = '') -> None:
        from prompt_toolkit.history import InMemoryHistory

        self.text = text
        self.history = InMemoryHistory()

    def reset(self) -> None:
        self.text = ''

    def insert_text(self, text: str) -> None:
        self.text += text


class _App:
    def __init__(self, buffer: _Buffer) -> None:
        self.current_buffer = buffer
        self.exited = None

    def exit(self, result=None) -> None:
        self.exited = result

    def invalidate(self) -> None:
        return None


class KeyTests(unittest.TestCase):
    def handler(self, kwargs: dict, name: str):
        return next(b.handler for b in kwargs['key_bindings'].bindings if b.handler.__name__ == name)

    def press(self, kwargs: dict, name: str, text: str = '') -> _App:
        app = _App(_Buffer(text))
        self.handler(kwargs, name)(type('E', (), {'app': app})())
        return app

    def test_prompt_keys(self) -> None:
        from src.agent_bar import AgentBar, build_prompt_kwargs

        with tempfile.TemporaryDirectory() as tmp:
            bar = AgentBar(Path(tmp))
            kwargs = build_prompt_kwargs(bar)
            app = self.press(kwargs, '_stash', 'half-written idea')
            self.assertEqual((app.current_buffer.text, bar.stash), ('', 'half-written idea'))
            app = self.press(kwargs, '_stash', '')
            self.assertEqual((app.current_buffer.text, bar.stash), ('half-written idea', ''))
            app = self.press(kwargs, '_clear_draft', 'draft to drop')
            self.assertEqual(app.current_buffer.text, '')
            self.assertIn('draft to drop', list(app.current_buffer.history.get_strings()))
            self.assertEqual(self.press(kwargs, '_key_help').exited, '/keybindings')
            self.assertEqual(self.press(kwargs, '_models').exited, '/model list')
            self.assertEqual(self.press(kwargs, '_thinking').exited, '/thinking toggle')
            self.press(kwargs, '_verbose')
            self.assertTrue(bar.verbose)
            hints = ''.join(text for _style, text, *_ in kwargs['bottom_toolbar']())
            self.assertIn('verbose: on', hints)

    def test_ctrl_enter_sends_queued_messages_now(self) -> None:
        sent: list[bool] = []
        queue = MessageQueue()
        reader = ConsoleInputReader(queue, echo=lambda _t: None, on_send_now=lambda: sent.append(True))
        for ch in 'fix it':
            reader.feed(ch)
        reader.feed('\n')  # Ctrl+Enter on a Windows console
        self.assertEqual(queue.drain(), ['fix it'])
        self.assertEqual(sent, [True])
        plain = ConsoleInputReader(MessageQueue(), echo=lambda _t: None)
        for ch in 'later\r':
            plain.feed(ch)
        self.assertEqual(plain.queue.drain(), ['later'])

    def test_verbose_shows_the_transcript_after_answers(self) -> None:
        from src.agent_bar import AgentBar
        from src.main import _run_agent_chat_loop

        class VerboseBar(AgentBar):
            def __init__(self, cwd):
                super().__init__(cwd)
                self.verbose = True

        seen: list[bool] = []
        with tempfile.TemporaryDirectory() as tmp, patch('src.main.AgentBar', VerboseBar), \
                patch('src.openai_compat.request.urlopen',
                      side_effect=lambda req, timeout=None: _Resp({'role': 'assistant', 'content': 'ok'})):
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                     runtime_config=AgentRuntimeConfig(cwd=Path(tmp), session_directory=Path(tmp) / 's'))

            def no_more(_label: str = '') -> str:
                raise EOFError

            _run_agent_chat_loop(agent, initial_prompt='hi', resume_session_id=None, show_transcript=False,
                                 input_func=no_more, output_func=lambda _t: None,
                                 result_printer=lambda result, show_transcript: seen.append(show_transcript))
        self.assertEqual(seen, [True])


class TerminalSetupTests(unittest.TestCase):
    def test_vscode_and_windows_terminal(self) -> None:
        from src.terminal_setup import SEQUENCE, setup

        with tempfile.TemporaryDirectory() as tmp:
            keys = Path(tmp) / 'keybindings.json'
            keys.write_text('// my bindings\n[\n  {"key": "ctrl+k", "command": "x"},\n]\n', encoding='utf-8')
            wt = Path(tmp) / 'settings.json'
            wt.write_text('{"actions": []}', encoding='utf-8')
            with patch.dict(os.environ, {'SIGIL_VSCODE_KEYBINDINGS': str(keys), 'SIGIL_WT_SETTINGS': str(wt),
                                         'TERM_PROGRAM': 'vscode', 'WT_SESSION': ''}):
                self.assertIn('VS Code: Shift+Enter', setup(''))
                data = json.loads(keys.read_text(encoding='utf-8'))
                self.assertEqual(data[-1]['args']['text'], SEQUENCE)
                self.assertEqual(data[0]['key'], 'ctrl+k')  # existing bindings kept
                self.assertTrue((Path(tmp) / 'keybindings.json.bak').exists())
                self.assertIn('already bound', setup('vscode'))
                self.assertIn('Windows Terminal', setup('windows-terminal'))
                saved = json.loads(wt.read_text())  # current schema: action with an id + a separate key binding
                self.assertEqual(saved['actions'][0]['command'], {'action': 'sendInput', 'input': SEQUENCE})
                self.assertEqual(saved['keybindings'][0], {'keys': 'shift+enter', 'id': saved['actions'][0]['id']})
                self.assertNotIn('keys', saved['actions'][0])
                broken = Path(tmp) / 'broken.json'
                broken.write_text('{"actions": [ /* inline comment */ ]}', encoding='utf-8')
                with patch.dict(os.environ, {'SIGIL_WT_SETTINGS': str(broken)}):
                    self.assertIn('by hand', setup('wt'))
                self.assertIn('inline comment', broken.read_text())  # left alone
            with patch.dict(os.environ, {'TERM_PROGRAM': '', 'WT_SESSION': ''}):
                self.assertIn('No supported terminal', setup(''))


class PosixCleanupTests(unittest.TestCase):
    def test_kill_falls_back_to_sigkill(self) -> None:
        from src.background_runtime import BackgroundSessionRecord, BackgroundSessionRuntime

        with tempfile.TemporaryDirectory() as tmp:
            runtime = BackgroundSessionRuntime(Path(tmp))
            runtime.save_record(BackgroundSessionRecord(
                background_id='bg_k', pid=424242, prompt='p', workspace_cwd=tmp, model='m', mode='agent',
                status='running', log_path=str(Path(tmp) / 'l'), record_path=str(runtime.record_path('bg_k')),
                started_at='now', command=()))
            calls: list[tuple] = []
            with patch('src.background_runtime.sys.platform', 'linux'), \
                    patch('src.background_runtime.os.killpg', create=True,
                          side_effect=lambda pid, sig: calls.append((pid, sig))), \
                    patch('src.background_runtime.os.getpgid', create=True, side_effect=lambda pid: pid), \
                    patch('src.background_runtime.os.getpgrp', create=True, return_value=1000), \
                    patch('src.background_runtime.os.WNOHANG', 1, create=True), \
                    patch('src.background_runtime.os.waitpid', return_value=(0, 0)), \
                    patch('src.background_runtime.signal.SIGKILL', 9, create=True), \
                    patch('src.background_runtime._is_process_running', return_value=True), \
                    patch('src.background_runtime.time.sleep'):
                runtime.kill('bg_k')
            self.assertEqual(calls, [(424242, signal.SIGTERM), (424242, 9)])

    def test_worker_turns_sigterm_into_a_clean_exit(self) -> None:
        from src.main import _exit_cleanly_on_sigterm

        installed: dict = {}
        with patch.object(sys, 'platform', 'linux'), \
                patch('signal.signal', side_effect=lambda sig, handler: installed.update({sig: handler})):
            _exit_cleanly_on_sigterm()
        handler = installed[signal.SIGTERM]
        with self.assertRaises(SystemExit) as caught:
            handler(signal.SIGTERM, None)
        self.assertEqual(caught.exception.code, 128 + signal.SIGTERM)


if __name__ == '__main__':
    unittest.main()
