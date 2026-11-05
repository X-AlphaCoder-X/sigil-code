"""Every interactive start asks the folder trust question, a resumed chat (-c, -r, --resume-session-id) and
`sigil sessions` included. A chat that starts before the question was answered says its project items are
off instead of running without them quietly. Headless runs never ask."""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import main as main_module
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.workspace_trust import record, status


class TrustOnEveryStartTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('SIGIL_TRUST_ALL', None)  # conftest turns trust off for the other tests
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'srv': {'command': 'no-such-mcp-command-0'}}}),
                                           encoding='utf-8')
        self.assertEqual(status(self.ws)[0], 'new')

    def _start(self, argv: list[str]) -> list[str]:
        """Run main() as if on a terminal, answer [s]kip, and return the questions asked."""
        asked: list[str] = []
        with mock.patch('src.main._on_terminal', return_value=True), \
                mock.patch('builtins.input', side_effect=lambda prompt: asked.append(prompt) or 's'), \
                mock.patch('sys.stdout', io.StringIO()), mock.patch('sys.stderr', io.StringIO()), \
                mock.patch('src.main._run_agent_chat_loop', return_value=0) as chat, \
                mock.patch('src.main._run_session_picker', return_value=0) as picker:
            self.assertEqual(main_module.main(argv), 0)
        self.assertTrue(chat.called or picker.called)
        return asked

    def test_a_resumed_chat_asks_like_a_new_one(self) -> None:
        cwd = ['--cwd', str(self.ws)]
        for argv in ([*cwd], ['-c', *cwd], ['-r', *cwd], ['--resume-session-id', 'abc123', *cwd],
                     ['agent-chat', '--resume-session-id', 'abc123', *cwd], ['sessions', *cwd]):
            with self.subTest(argv=argv):
                record(self.ws, True)       # answered before: no question
                os.remove(main_module.Path.home() / '.sigil' / 'trusted_folders.json')   # undecided again
                self.assertEqual(status(self.ws)[0], 'new')
                asked = self._start(argv)
                self.assertEqual(len(asked), 1, asked)
                self.assertIn('Trust this folder?', asked[0])
                self.assertEqual(status(self.ws)[0], 'skipped')   # the answer is kept

    def test_interactive_starts_and_headless_runs(self) -> None:
        with mock.patch('src.main._on_terminal', return_value=True):
            for argv in ([], ['-c'], ['-r'], ['--resume-session-id', 'x'], ['agent-chat', '--resume'], ['sessions']):
                self.assertTrue(main_module._is_interactive_chat(argv), argv)
            for argv in (['-p', 'hi'], ['--print', 'hi'], ['print', 'hi'], ['agent', 'hi'], ['trust', 'status']):
                self.assertFalse(main_module._is_interactive_chat(argv), argv)
        with mock.patch('src.main._on_terminal', return_value=False):
            self.assertFalse(main_module._is_interactive_chat(['-c']))

    def test_a_launcher_asks_on_a_terminal_and_reports_otherwise(self) -> None:
        with mock.patch('builtins.input', return_value='t') as ask, mock.patch('sys.stdout', io.StringIO()):
            self.assertTrue(main_module.ask_folder_trust(self.ws, interactive=True))
        ask.assert_called_once()
        os.remove(main_module.Path.home() / '.sigil' / 'trusted_folders.json')
        err = io.StringIO()
        with mock.patch('builtins.input', side_effect=AssertionError('headless must not ask')), \
                mock.patch('sys.stderr', err):
            self.assertFalse(main_module.ask_folder_trust(self.ws, interactive=False))
        self.assertIn('srv', err.getvalue())                       # it names what it skipped
        self.assertEqual(status(self.ws)[0], 'new')                # and records no answer

    def test_a_chat_that_was_never_asked_says_its_project_items_are_off(self) -> None:
        def chat_output() -> list[str]:
            agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                     runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / 's'))
            out: list[str] = []
            main_module._run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                                             input_func=lambda prompt: '/exit', output_func=out.append)
            return [line for line in out if 'project item' in line]

        notes = chat_output()
        self.assertEqual(len(notes), 1)
        self.assertIn('was never asked about', notes[0])
        self.assertIn('/trust grant', notes[0])
        record(self.ws, False)                   # the user said skip: their choice, no note
        self.assertEqual(chat_output(), [])
        record(self.ws, True)
        self.assertEqual(chat_output(), [])
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'other': {'command': 'no-such-mcp-command-1'}}}),
                                           encoding='utf-8')
        notes = chat_output()                    # trusted, then the project items changed
        self.assertEqual(len(notes), 1)
        self.assertIn('changed since you trusted it', notes[0])
        with mock.patch.dict(os.environ, {'SIGIL_TRUST_ALL': '1'}):
            self.assertEqual(chat_output(), [])


if __name__ == '__main__':
    unittest.main()
