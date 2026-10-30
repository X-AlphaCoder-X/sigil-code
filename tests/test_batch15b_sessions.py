"""Batch 15b (P3): cleanupPeriodDays + `sigil purge`, session search (/resume <text>,
`sigil sessions --search`) and [bg] marks, /export --redact, /release-notes, attribution,
--plain output and the terminal title."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.main import _run_agent_chat_loop, _session_directory_for, main
from src.redact import redact_text
from src.session_store import list_agent_sessions, purge_agent_sessions, save_session_name, search_agent_sessions
from src.terminal import set_title, to_plain

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


class FakeModel:
    def __init__(self) -> None:
        self.replies: list[str] = []
        self.requests: list[dict] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001
        self.requests.append(json.loads(req.data.decode('utf-8')))
        return _Resp({'role': 'assistant', 'content': self.replies.pop(0) if self.replies else 'answer'})


def scripted(*answers: str):
    queue = list(answers)

    def _input(_label: str = '') -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return _input


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        self.dir = _session_directory_for(self.ws)
        self.model = FakeModel()
        for patcher in (patch('src.openai_compat.request.urlopen', side_effect=self.model),
                        patch.dict(os.environ, {'SIGIL_CLEANUP_PERIOD_DAYS': '', 'SIGIL_PLAIN': ''})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def print_mode(self, prompt: str, *extra: str) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            main(['-p', prompt, '--output-format', 'json', '--cwd', str(self.ws), '--model', 'm',
                  '--base-url', BASE, *extra])
        return json.loads(out.getvalue())['session_id']

    def agent(self) -> LocalCodingAgent:
        return LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.dir,
                                                                  nudge_strategy='off'))

    def age(self, session_id: str, days: float) -> None:
        stamp = time.time() - days * 86400
        os.utime(self.dir / f'{session_id}.json', (stamp, stamp))


class PurgeTests(Base):
    def test_purge_by_age_cli_and_auto_cleanup(self) -> None:
        old = self.print_mode('old work')
        new = self.print_mode('new work')
        self.age(old, 40)
        (self.dir / 'checkpoints' / old).mkdir(parents=True, exist_ok=True)
        (self.dir / 'notes').mkdir(exist_ok=True)
        (self.dir / 'notes' / f'{old}.md').write_text('notes', encoding='utf-8')
        save_session_name(self.dir, 'oldname', old)
        self.assertEqual(purge_agent_sessions(self.dir, older_than_days=30, dry_run=True), [old])
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(['purge', '--days', '30', '--cwd', str(self.ws)]), 0)
        self.assertIn(f'Deleted 1 of the sessions older than 30 day(s)', out.getvalue())
        self.assertEqual([s for s, _, _ in list_agent_sessions(self.dir)], [new])
        self.assertFalse((self.dir / 'checkpoints' / old).exists())
        self.assertFalse((self.dir / 'notes' / f'{old}.md').exists())
        self.assertNotIn('oldname', (self.dir / 'names.idx').read_text(encoding='utf-8'))

        self.age(new, 3)
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text('{"cleanupPeriodDays": 2}', encoding='utf-8')
        latest = self.print_mode('third')
        self.assertEqual([s for s, _, _ in list_agent_sessions(self.dir)], [latest])  # 3-day-old one went
        with redirect_stdout(io.StringIO()):
            main(['purge', '--all', '--cwd', str(self.ws)])
        self.assertEqual(list_agent_sessions(self.dir), [])


class SearchTests(Base):
    def test_search_resume_by_text_and_bg_marks(self) -> None:
        target = self.print_mode('fix the flaky widget test')
        self.print_mode('write docs')
        bg = self.print_mode('background flaky job', '--session-id', 'bg_0123456789ab')
        matches = search_agent_sessions(self.dir, 'FLAKY')
        self.assertEqual({m[0] for m in matches}, {target, bg})
        self.assertTrue(any('flaky widget' in m[3] for m in matches))
        output: list[str] = []
        _run_agent_chat_loop(self.agent(), initial_prompt=None, resume_session_id=None, show_transcript=False,
                             input_func=scripted('/resume widget', '0'), output_func=output.append)
        self.assertIn(f'resuming_session_id={target}', output)
        output.clear()
        _run_agent_chat_loop(self.agent(), initial_prompt=None, resume_session_id=None, show_transcript=False,
                             input_func=scripted('/resume'), output_func=output.append)
        self.assertTrue(any('bg_0123456789ab  [bg]' in line for line in output), output)
        out = io.StringIO()
        picked: dict = {}
        with redirect_stdout(out), patch('builtins.input', return_value='0'), \
                patch('src.main._run_agent_chat_loop', side_effect=lambda agent, **kw: picked.update(kw) or 0):
            main(['sessions', '--search', 'widget', '--cwd', str(self.ws)])
        self.assertIn(target, out.getvalue())
        self.assertNotIn('write docs', out.getvalue())
        self.assertEqual(picked['resume_session_id'], target)  # index 0 of the filtered list


class ExportAndNotesTests(Base):
    def test_redact_patterns(self) -> None:
        with patch.dict(os.environ, {'MY_SERVICE_TOKEN': 'supersecretvalue123'}):
            text = redact_text('key sk-abcdefghijklmnopqrstuvwx and ghp_abcdefghijklmnopqrstuvwxyz123456 '
                               'AKIAABCDEFGHIJKLMNOP Authorization: Bearer abc.def.ghijklmnop '
                               'password="hunter2pass" token: supersecretvalue123 plain words stay')
        for secret in ('sk-abcdef', 'ghp_abc', 'AKIAABCD', 'abc.def.ghij', 'hunter2pass', 'supersecretvalue123'):
            self.assertNotIn(secret, text)
        self.assertIn('plain words stay', text)
        self.assertIn('Bearer [REDACTED]', text)

    def test_export_redact_and_release_notes(self) -> None:
        self.model.replies = ['your key is sk-live1234567890abcdefghij']
        agent = self.agent()
        agent.run('show my key')
        agent.run('/export plain.txt')
        report = agent.run('/export --redact shared.txt').final_output
        self.assertIn('secrets redacted', report)
        self.assertIn('sk-live1234567890', (self.ws / 'plain.txt').read_text(encoding='utf-8'))
        shared = (self.ws / 'shared.txt').read_text(encoding='utf-8')
        self.assertNotIn('sk-live1234567890', shared)
        self.assertIn('[REDACTED]', shared)
        # /release-notes shows the newest section of CHANGELOG.md; `all` shows the whole file
        changelog = self.ws / 'CHANGELOG.md'
        changelog.write_text('# Changelog\n\n## [1.1.0] - 2026-11-01\n\n- newer\n\n'
                             '## [1.0.0] - 2026-10-30\n\n- First public release\n', encoding='utf-8')
        from src import agent_slash_commands

        with patch.object(agent_slash_commands, '_changelog_candidates', return_value=[changelog]):
            notes = agent.run('/release-notes').final_output
            everything = agent.run('/release-notes all').final_output
        self.assertTrue(notes.startswith('## [1.1.0]'), notes[:80])
        self.assertNotIn('## [1.0.0]', notes)
        self.assertIn('## [1.0.0]', everything)
        self.assertIn('## [1.1.0]', everything)

    def test_attribution_setting(self) -> None:
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text(json.dumps(
            {'attribution': {'commit': 'Co-Authored-By: Helper <helper@example.com>', 'pr': ''}}), encoding='utf-8')
        self.agent().run('hi')
        system = self.model.requests[-1]['messages'][0]['content']
        self.assertIn('Co-Authored-By: Helper <helper@example.com>', system)
        self.assertIn('Do not add co-author or "generated with" lines to a pull request description', system)
        (self.ws / '.sigil' / 'settings.json').write_text('{}', encoding='utf-8')
        self.agent().run('hi')
        self.assertNotIn('# Attribution', self.model.requests[-1]['messages'][0]['content'])


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class TerminalTests(Base):
    def test_plain_output_in_the_chat(self) -> None:
        self.assertEqual(to_plain('\x1b[32m↶ ok ✓\x1b[0m ╭─╮ \U0001F527'), '<- ok ok +-+ ')
        captured = io.StringIO()
        with patch.dict(os.environ, {'SIGIL_PLAIN': '1'}), patch.object(sys, 'stdout', captured):
            _run_agent_chat_loop(self.agent(), initial_prompt=None, resume_session_id=None,
                                 show_transcript=False, input_func=scripted('/mode plan'), output_func=print)
            self.assertEqual(os.environ.get('NO_COLOR'), '1')
        text = captured.getvalue()
        self.assertIn('Permission mode', text)
        self.assertNotIn('\x1b', text)
        self.assertNotIn('⏵', text)

    def test_terminal_title(self) -> None:
        tty = _TTY()
        with patch.object(sys, '__stdout__', tty), patch.dict(os.environ, {'SIGIL_TERMINAL_TITLE': ''}):
            _run_agent_chat_loop(self.agent(), initial_prompt='hello', resume_session_id=None,
                                 show_transcript=False, input_func=scripted(), output_func=lambda _t: None)
            titles = [part.split('\x07')[0] for part in tty.getvalue().split('\x1b]0;')[1:]]
            with patch.dict(os.environ, {'SIGIL_TERMINAL_TITLE': '0'}):
                self.assertFalse(set_title('x'))
            with patch.dict(os.environ, {'SIGIL_PLAIN': '1'}):
                self.assertFalse(set_title('x'))
        name = self.ws.name
        self.assertEqual(titles[:3], [f'Sigil: {name} - ready', f'Sigil: {name} - working', f'Sigil: {name} - ready'])


if __name__ == '__main__':
    unittest.main()
