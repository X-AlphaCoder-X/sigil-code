from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import build_parser, main, _run_session_picker
from src.session_store import (
    StoredAgentSession,
    list_agent_sessions,
    save_agent_session,
)


def _make_session(session_id: str, messages: int = 2) -> StoredAgentSession:
    return StoredAgentSession(
        session_id=session_id,
        model_config={'model': 'test-model'},
        runtime_config={},
        system_prompt_parts=(),
        user_context={},
        system_context={},
        messages=tuple(
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'm{i}'}
            for i in range(messages)
        ),
        turns=1,
        tool_call_count=0,
        tool_execution_ledger=(),
        usage={},
        total_cost_usd=0.0,
        file_history=(),
        budget_state={},
        plugin_state={},
    )


class ParserShortcutTests(unittest.TestCase):
    def test_bare_invocation_has_no_command(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.command)
        self.assertFalse(args.top_resume)
        self.assertIsNone(args.top_resume_session_id)

    def test_top_level_resume_flag(self) -> None:
        args = build_parser().parse_args(['-r'])
        self.assertIsNone(args.command)
        self.assertTrue(args.top_resume)
        args = build_parser().parse_args(['--resume'])
        self.assertTrue(args.top_resume)

    def test_top_level_resume_session_id(self) -> None:
        args = build_parser().parse_args(['--resume-session-id', 'abc123'])
        self.assertIsNone(args.command)
        self.assertEqual(args.top_resume_session_id, 'abc123')

    def test_agent_chat_resume_flags_are_not_clobbered(self) -> None:
        args = build_parser().parse_args(['agent-chat', '--resume'])
        self.assertEqual(args.command, 'agent-chat')
        self.assertTrue(args.resume)
        self.assertFalse(args.top_resume)
        args = build_parser().parse_args(['agent-chat', '--resume-session-id', 'xyz'])
        self.assertEqual(args.resume_session_id, 'xyz')
        self.assertIsNone(args.top_resume_session_id)

    def test_sessions_subcommand_parses(self) -> None:
        args = build_parser().parse_args(['sessions', '--cwd', '.'])
        self.assertEqual(args.command, 'sessions')


class ListSessionsTests(unittest.TestCase):
    def test_lists_sessions_newest_first_with_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_agent_session(_make_session('older'), directory=directory)
            save_agent_session(_make_session('newer', messages=5), directory=directory)
            sessions = list_agent_sessions(directory)
            self.assertEqual([s[0] for s in sessions], ['newer', 'older'])
            self.assertEqual(sessions[0][2], 5)
            self.assertEqual(sessions[1][2], 2)

    def test_empty_directory_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_agent_sessions(Path(tmp) / 'missing'), [])

    def test_corrupt_files_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_agent_session(_make_session('good'), directory=directory)
            (directory / 'bad.json').write_text('{not json', encoding='utf-8')
            sessions = list_agent_sessions(directory)
            self.assertEqual([s[0] for s in sessions], ['good'])


class BareEntryTranslationTests(unittest.TestCase):
    """main() with no subcommand must re-parse as agent-chat."""

    def _capture_chat(self, argv: list[str]) -> object:
        seen: dict[str, object] = {}

        def _fake_loop(agent, *, initial_prompt, resume_session_id, show_transcript,
                       input_func=input, output_func=print, result_printer=None):
            seen['resume_session_id'] = resume_session_id
            seen['initial_prompt'] = initial_prompt
            return 0

        with patch('src.main._run_agent_chat_loop', side_effect=_fake_loop), \
             patch('src.main._build_agent', return_value=object()):
            code = main(argv)
        self.assertEqual(code, 0)
        return seen

    def test_bare_main_starts_chat(self) -> None:
        seen = self._capture_chat([])
        self.assertIsNone(seen['resume_session_id'])

    def test_resume_shortcut_resumes_latest(self) -> None:
        seen = self._capture_chat(['-r'])
        self.assertEqual(seen['resume_session_id'], 'latest')

    def test_resume_session_id_shortcut(self) -> None:
        seen = self._capture_chat(['--resume-session-id', 'abc123'])
        self.assertEqual(seen['resume_session_id'], 'abc123')


class SessionPickerTests(unittest.TestCase):
    def _run_picker(self, choices: list[str], sessions: list[str]) -> tuple[int, object]:
        seen: dict[str, object] = {}
        prompts = iter(choices)

        def _input(prompt: str) -> str:
            return next(prompts)

        def _fake_loop(agent, *, initial_prompt, resume_session_id, show_transcript,
                       input_func=input, output_func=print, result_printer=None):
            seen['resume_session_id'] = resume_session_id
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / '.port_sessions' / 'agent'
            directory.mkdir(parents=True)
            for index, name in enumerate(sessions):
                path = save_agent_session(_make_session(name), directory=directory)
                # Pin distinct mtimes: same-second saves make newest-first
                # ordering unstable on coarse-timestamp filesystems.
                os.utime(path, (1_000_000_000 + index, 1_000_000_000 + index))
            args = build_parser().parse_args(['sessions', '--cwd', tmp])
            with patch('src.main._run_agent_chat_loop', side_effect=_fake_loop), \
                 patch('src.main._build_agent', return_value=object()), \
                 patch('builtins.input', side_effect=_input):
                code = _run_session_picker(args)
        return code, seen

    def test_pick_by_index(self) -> None:
        # Sessions are listed newest first, so index 1 is the older 'aaa'.
        code, seen = self._run_picker(['1'], ['aaa', 'bbb'])
        self.assertEqual(code, 0)
        self.assertEqual(seen['resume_session_id'], 'aaa')

    def test_default_choice_is_newest(self) -> None:
        code, seen = self._run_picker([''], ['aaa', 'bbb'])
        self.assertEqual(code, 0)
        self.assertEqual(seen['resume_session_id'], 'bbb')

    def test_new_session_choice(self) -> None:
        code, seen = self._run_picker(['n'], ['aaa'])
        self.assertEqual(code, 0)
        self.assertIsNone(seen['resume_session_id'])

    def test_paste_session_id(self) -> None:
        code, seen = self._run_picker(['aaa'], ['aaa', 'bbb'])
        self.assertEqual(code, 0)
        self.assertEqual(seen['resume_session_id'], 'aaa')

    def test_no_sessions_prints_and_exits(self) -> None:
        seen: dict[str, object] = {}

        def _fake_loop(agent, **kwargs):
            seen['called'] = True
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(['sessions', '--cwd', tmp])
            with patch('src.main._run_agent_chat_loop', side_effect=_fake_loop), \
                 patch('src.main._build_agent', return_value=object()):
                code = _run_session_picker(args)
        self.assertEqual(code, 0)
        self.assertNotIn('called', seen)


if __name__ == '__main__':
    unittest.main()
