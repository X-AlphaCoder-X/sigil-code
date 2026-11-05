"""Sessions record how they were started, and /resume, `sigil sessions` and a chat's -c show the user's own
chat sessions first. -p runs, background agents and subagents are hidden unless asked for."""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentRuntimeConfig, ModelConfig
from src.debug_log import get_logger
from src.main import _lookup_latest_session_id, _pick_session_in_chat
from src.session_store import (
    StoredAgentSession, is_user_session, load_agent_session, save_agent_session, session_origin,
)


def _session(session_id: str, origin: str, prompt: str) -> StoredAgentSession:
    return StoredAgentSession(
        session_id=session_id, model_config={}, runtime_config={}, system_prompt_parts=(),
        user_context={}, system_context={}, origin=origin,
        messages=({'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': prompt}),
        turns=1, tool_call_count=0, tool_execution_ledger=(), usage={}, total_cost_usd=0.0,
        file_history=(), budget_state={}, plugin_state={})


class SessionOriginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        # oldest to newest: an old session without an origin, the user's chat, then two -p runs and a
        # background agent that are newer than the chat
        for index, (sid, origin, prompt) in enumerate([
                ('aaaa0000old', '', 'old work'), ('bbbb1111chat', 'chat', 'my real work'),
                ('cccc2222print', 'print', 'Reply with exactly: GLM_OK'), ('bg_dddd3333', 'background', 'bg task'),
                ('eeee4444print', 'print', 'Reply with exactly: OK')]):
            path = save_agent_session(_session(sid, origin, prompt), self.dir)
            os.utime(path, (time.time() - 100 + index, time.time() - 100 + index))

    def test_origin_is_saved_near_the_top_and_read_back(self) -> None:
        head = (self.dir / 'bbbb1111chat.json').read_text(encoding='utf-8').splitlines()[:3]
        self.assertEqual(head[1:], ['  "session_id": "bbbb1111chat",', '  "origin": "chat",'])
        self.assertEqual(load_agent_session('bbbb1111chat', self.dir).origin, 'chat')
        self.assertEqual(session_origin(self.dir, 'cccc2222print'), 'print')

    def test_a_file_saved_before_origins_loads_and_counts_as_the_users(self) -> None:
        data = json.loads((self.dir / 'aaaa0000old.json').read_text(encoding='utf-8'))
        data.pop('origin')
        (self.dir / 'aaaa0000old.json').write_text(json.dumps(data), encoding='utf-8')
        self.assertEqual(load_agent_session('aaaa0000old', self.dir).origin, '')
        self.assertTrue(is_user_session(self.dir, 'aaaa0000old'))

    def test_which_sessions_are_the_users(self) -> None:
        self.assertEqual([sid for sid in ('aaaa0000old', 'bbbb1111chat', 'cccc2222print', 'bg_dddd3333',
                                          'eeee4444print', 'ffff--sub--1') if is_user_session(self.dir, sid)],
                         ['aaaa0000old', 'bbbb1111chat'])

    def test_an_unreadable_session_is_shown_and_logged(self) -> None:
        with self.assertLogs(get_logger('ignored'), level='DEBUG') as logs:
            self.assertTrue(is_user_session(self.dir, 'missing0000'))
        self.assertTrue(logs.output)
        (self.dir / 'notutf80000.json').write_bytes(b'{"session_id": "x", "origin": "\xff\xfe"}')
        with mock.patch('src.session_store._log_ignored') as logged:   # the log line is rate limited per place
            self.assertTrue(is_user_session(self.dir, 'notutf80000'))
        self.assertIsInstance(logged.call_args.args[0], UnicodeDecodeError)

    def test_chat_continue_picks_the_newest_of_the_users_sessions(self) -> None:
        self.assertEqual(_lookup_latest_session_id(str(self.dir), session_dir=self.dir), 'eeee4444print')
        self.assertEqual(_lookup_latest_session_id(str(self.dir), session_dir=self.dir, user_only=True), 'bbbb1111chat')

    def test_resume_picker_lists_the_users_sessions_and_all_lists_every_one(self) -> None:
        agent = SimpleNamespace(runtime_config=SimpleNamespace(session_directory=self.dir))
        out: list[str] = []
        self.assertIsNone(_pick_session_in_chat(agent, '', out.append, lambda prompt: ''))
        listing = '\n'.join(out)
        self.assertIn('bbbb1111chat', listing)
        self.assertIn('aaaa0000old', listing)
        self.assertNotIn('print', listing.split('\n', 1)[1])
        self.assertIn('3 -p or background session(s) hidden: /resume all shows them', out[0])
        out.clear()
        _pick_session_in_chat(agent, 'all', out.append, lambda prompt: '')
        self.assertIn('cccc2222print', '\n'.join(out))
        self.assertNotIn('hidden', out[0])

    def test_sessions_command_hides_and_all_shows(self) -> None:
        from src.main import _run_session_picker

        # --search looks through every session (like /resume <text>), so a -p run is found without --all
        for flag, search, expect_print, default_pick in ((False, None, False, 'bbbb1111chat'),
                                                         (True, None, True, 'eeee4444print'),
                                                         (False, 'GLM_OK', True, 'cccc2222print')):
            printed = io.StringIO()
            args = SimpleNamespace(cwd=str(self.dir), search=search, all=flag, max_turns=0)
            with mock.patch('src.main._session_directory_for', return_value=self.dir), \
                    mock.patch('builtins.input', return_value=''), mock.patch('sys.stdout', printed), \
                    mock.patch('src.main._build_agent', return_value=object()), \
                    mock.patch('src.main._run_agent_chat_loop', return_value=0) as chat:
                _run_session_picker(args)
            self.assertEqual('cccc2222print' in printed.getvalue(), expect_print)
            # the "N hidden" count only where something is hidden: not with --all, not over search results
            self.assertEqual('hidden' in printed.getvalue(), not flag and not search)
            # Enter (default 0) resumes the newest session in the list: the user's own chat by default
            self.assertEqual(chat.call_args.kwargs['resume_session_id'], default_pick)

    def test_a_chat_continues_the_users_newest_session_and_is_marked_chat(self) -> None:
        from src.main import _run_agent_chat_loop

        agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                 runtime_config=AgentRuntimeConfig(cwd=self.dir, session_directory=self.dir))
        out: list[str] = []
        _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id='latest', show_transcript=True,
                             input_func=lambda prompt: '/exit', output_func=out.append)
        self.assertIn('resuming_session_id=bbbb1111chat', out)   # not the newer -p run eeee4444print
        self.assertIn('user> my real work', out)
        self.assertEqual(agent.session_origin, 'chat')

    def test_origin_rules(self) -> None:
        agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                 runtime_config=AgentRuntimeConfig(cwd=self.dir, session_directory=self.dir))
        # an entry point that says nothing saves no origin, so its sessions are shown, never hidden
        self.assertEqual(agent._origin_for_save(), '')
        agent.new_session_origin = 'print'
        self.assertEqual(agent._origin_for_save(), 'print')      # a new -p, stream-json or one-shot session
        agent._resumed_origin = 'chat'
        self.assertEqual(agent._origin_for_save(), 'chat')       # -p -c on a chat session keeps it the user's
        agent._resumed_origin = 'print'
        agent.session_origin = 'chat'
        self.assertEqual(agent._origin_for_save(), 'chat')       # a -p session continued in a chat
        del agent.session_origin
        self.assertEqual(agent._origin_for_save(), 'print')      # a resumed session keeps its origin

    def test_each_entry_point_saves_its_origin(self) -> None:
        """-p, stream-json and `sigil agent` save print sessions; a chat saves chat; -p -c on a chat
        session keeps it chat. Driven through main() with a fake model server."""
        import contextlib

        from src.main import _session_directory_for, main

        ws = self.dir / 'ws'
        ws.mkdir()

        def reply(req, timeout=None):  # noqa: ANN001
            body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                                            'finish_reason': 'stop'}]}).encode()
            return mock.MagicMock(read=lambda: body, __enter__=lambda s: s, __exit__=lambda *a: None)

        def run(*argv: str) -> set[str]:
            sessions = _session_directory_for(ws)
            before = set(p.stem for p in sessions.glob('*.json')) if sessions.is_dir() else set()
            with mock.patch('src.openai_compat.request.urlopen', side_effect=reply), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch('sys.stdin', io.StringIO('{"type": "user", "message": {"role": "user", '
                                                        '"content": "hi"}}\n')):
                main([*argv, '--cwd', str(ws), '--model', 'm', '--base-url', 'http://127.0.0.1:8000/v1'])
            return set(p.stem for p in sessions.glob('*.json')) - before

        sessions = _session_directory_for(ws)
        for argv in (['-p', 'hi'], ['-p', '--input-format', 'stream-json', '--output-format', 'stream-json'],
                     ['agent', 'hi']):
            with self.subTest(argv=argv):
                new = run(*argv)
                self.assertEqual(len(new), 1, new)
                self.assertEqual(session_origin(sessions, new.pop()), 'print')
        agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                 runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=sessions))
        agent.session_origin = 'chat'
        with mock.patch('src.openai_compat.request.urlopen', side_effect=reply):
            chat_id = agent.run('my work').session_id
        self.assertEqual(session_origin(sessions, chat_id), 'chat')
        self.assertEqual(run('-p', 'more', '-c'), set())                # -c continued the newest: the chat
        self.assertEqual(session_origin(sessions, chat_id), 'chat')   # and it stays the user's


if __name__ == '__main__':
    unittest.main()
