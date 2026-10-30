"""Batch 13 (P2): rewind summarize-from / summarize-to, auto memory, session memory, repo map, architect/editor split."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_context import clear_context_caches
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.auto_memory import index_path, load_block, save_note
from src.compact import compact_conversation
from src.repo_map import build_repo_map, file_symbols
from src.session_memory import notes_path
from src.rewind import list_prompt_points, render_prompt_list, rewind_session
from src.session_store import load_agent_session


class _Resp:
    def __init__(self, message: dict) -> None:
        self.body = json.dumps({'choices': [{'message': message, 'finish_reason':
                                             'tool_calls' if message.get('tool_calls') else 'stop'}]}).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def _write_call(call_id: str, path: str, content: str) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function',
        'function': {'name': 'write_file', 'arguments': json.dumps({'path': path, 'content': content})},
    }]}


class RewindSummarizeTests(unittest.TestCase):
    def _session(self, ws: Path, recorded: list[dict]):
        replies = [
            _write_call('c1', 'notes.txt', 'first'), {'role': 'assistant', 'content': 'Wrote notes.'},
            {'role': 'assistant', 'content': 'The answer is 42.'},
            {'role': 'assistant', 'content': 'Third answer.'},
        ]

        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            if replies:
                return _Resp(replies.pop(0))
            return _Resp({'role': 'assistant', 'content': '<analysis>x</analysis><summary>SHORT SUMMARY</summary>'})

        session_dir = ws / '.port_sessions' / 'agent'
        config = AgentRuntimeConfig(cwd=ws, session_directory=session_dir, nudge_strategy='off',
                                    permissions=AgentPermissions(allow_file_write=True))
        patcher = patch('src.openai_compat.request.urlopen', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        agent = LocalCodingAgent(model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                                 runtime_config=config)
        first = agent.run('create notes')
        agent.resume('what is the answer', load_agent_session(first.session_id, directory=session_dir))
        agent.resume('third question', load_agent_session(first.session_id, directory=session_dir))
        return agent, first.session_id, session_dir

    def test_summarize_up_to_a_prompt_keeps_later_messages_and_files(self) -> None:
        recorded: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent, sid, session_dir = self._session(ws, recorded)
            before = load_agent_session(sid, directory=session_dir)
            self.assertEqual([p.text for p in list_prompt_points(before.messages)],
                             ['create notes', 'what is the answer', 'third question'])
            report, text = rewind_session(agent, sid, 3, 'summarize-to')
            self.assertEqual(text, 'third question')
            self.assertIn('Files were not changed', report)
            self.assertEqual((ws / 'notes.txt').read_text(encoding='utf-8'), 'first')
            after = load_agent_session(sid, directory=session_dir)
            summary = [m for m in after.messages if (m.get('metadata') or {}).get('summarized')]
            self.assertEqual(len(summary), 1)
            self.assertIn('SHORT SUMMARY', summary[0]['content'])
            self.assertNotIn('<analysis>', summary[0]['content'])
            self.assertEqual([p.text for p in list_prompt_points(after.messages)], ['third question'])
            self.assertEqual(after.messages[0]['role'], 'system')  # the system prompt is never summarized
            self.assertTrue(any(m.get('content') == 'Third answer.' for m in after.messages))
            self.assertFalse(any(m.get('content') == 'The answer is 42.' for m in after.messages))
            request = recorded[-1]['messages'][-1]['content']
            self.assertIn('[user]\ncreate notes', request)
            self.assertIn('The answer is 42.', request)
            self.assertNotIn('Third answer.', request)

    def test_summarize_from_a_prompt_and_aliases(self) -> None:
        recorded: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent, sid, session_dir = self._session(ws, recorded)
            report, _ = rewind_session(agent, sid, 2, 'from')
            self.assertIn('from prompt #2 onward', report)
            after = load_agent_session(sid, directory=session_dir)
            self.assertEqual([p.text for p in list_prompt_points(after.messages)], ['create notes'])
            self.assertIn('SHORT SUMMARY', after.messages[-1]['content'])
            self.assertTrue(any(m.get('content') == 'Wrote notes.' for m in after.messages))
            request = recorded[-1]['messages'][-1]['content']
            self.assertIn('what is the answer', request)
            self.assertNotIn('create notes', request)
            # the agent continues on the shortened conversation
            agent.resume('go on', after)
            sent = recorded[-1]['messages']
            self.assertTrue(any('SHORT SUMMARY' in str(m.get('content')) for m in sent))
            self.assertIn('summarize-from', render_prompt_list(list_prompt_points(after.messages), None))

    def test_bad_mode_and_empty_summary(self) -> None:
        recorded: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent, sid, session_dir = self._session(ws, recorded)
            with self.assertRaises(ValueError):
                rewind_session(agent, sid, 1, 'sideways')
            with self.assertRaises(ValueError):
                rewind_session(agent, sid, 1, 'summarize-to')  # nothing before the first prompt
            count = len(load_agent_session(sid, directory=session_dir).messages)
            with patch('src.openai_compat.request.urlopen',
                       return_value=_Resp({'role': 'assistant', 'content': '   '})):
                with self.assertRaises(ValueError):
                    rewind_session(agent, sid, 2, 'summarize-to')
            self.assertEqual(len(load_agent_session(sid, directory=session_dir).messages), count)


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class AutoMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name).resolve()
        self.ws = base / 'repo' / 'pkg'
        self.ws.mkdir(parents=True)
        (base / 'repo' / '.git').mkdir()
        self.home = base / 'home'
        self.home.mkdir()
        for target in ('src.settings._home',):
            patcher = patch(target, return_value=self.home)
            patcher.start()
            self.addCleanup(patcher.stop)
        clear_context_caches()
        self.addCleanup(clear_context_caches)

    def agent(self, replies: list[dict], recorded: list[dict]) -> LocalCodingAgent:
        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0) if replies else {'role': 'assistant', 'content': 'ok'})

        patcher = patch('src.openai_compat.request.urlopen', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s' / 'agent',
                                              scratchpad_root=self.ws / '.s' / 'scratch', nudge_strategy='off'))

    @staticmethod
    def sent_text(request: dict) -> str:
        return chr(10).join(str(m.get('content')) for m in request['messages'])

    def test_off_by_default(self) -> None:
        with patch.dict(os.environ, {'SIGIL_AUTO_MEMORY': ''}):
            recorded: list[dict] = []
            agent = self.agent([], recorded)
            self.assertNotIn('memory_save', agent.tool_registry)
            agent.run('hello')
            self.assertNotIn('Auto memory', self.sent_text(recorded[-1]))
            self.assertIsNone(load_block(self.ws))
            self.assertIn('Auto memory: off', agent.render_memory_report())

    def test_model_saves_a_note_and_the_next_session_sees_it(self) -> None:
        with patch.dict(os.environ, {'SIGIL_AUTO_MEMORY': '1'}):
            recorded: list[dict] = []
            agent = self.agent([_call('c1', 'memory_save', {'note': 'Run tests with  pytest -q'}),
                                {'role': 'assistant', 'content': 'noted'}], recorded)
            self.assertIn('memory_save', agent.tool_registry)
            agent.run('remember: tests use pytest -q')
            self.assertIn('memory_save', {tool['function']['name'] for tool in recorded[0]['tools']})
            self.assertIn('Auto memory', self.sent_text(recorded[0]))
            self.assertIn('(no notes yet)', self.sent_text(recorded[0]))
            path = index_path(self.ws)
            self.assertTrue(str(path).startswith(str(self.home / '.sigil' / 'projects')))
            self.assertIn('repo', path.parent.parent.name)  # keyed by the git root, not the sub-folder
            self.assertIn('- Run tests with pytest -q', path.read_text(encoding='utf-8'))
            tool_reply = [m for m in recorded[-1]['messages'] if m.get('role') == 'tool']
            self.assertIn('Saved to', str(tool_reply[-1].get('content')))

            clear_context_caches()
            recorded2: list[dict] = []
            self.agent([], recorded2).run('how do I run tests?')
            self.assertIn('- Run tests with pytest -q', self.sent_text(recorded2[-1]))
            self.assertIn(str(path), agent.render_memory_report())

    def test_duplicates_line_limit_and_settings_switch(self) -> None:
        with patch.dict(os.environ, {'SIGIL_AUTO_MEMORY': '1', 'SIGIL_AUTO_MEMORY_LINES': '3'}):
            self.assertTrue(save_note(self.ws, 'one')[1])
            self.assertFalse(save_note(self.ws, 'one')[1])
            save_note(self.ws, 'two')
            save_note(self.ws, 'three')
            block = load_block(self.ws)
            self.assertIn('- one', block)
            self.assertNotIn('- three', block)
            self.assertIn('more line(s)', block)
            with self.assertRaises(ValueError):
                save_note(self.ws, '   ')
        with patch.dict(os.environ, {'SIGIL_AUTO_MEMORY': ''}):
            (self.ws / '.sigil').mkdir()
            (self.ws / '.sigil' / 'settings.json').write_text('{"autoMemoryEnabled": true}', encoding='utf-8')
            self.assertIsNotNone(load_block(self.ws))
            with patch.dict(os.environ, {'SIGIL_AUTO_MEMORY': '0'}):
                self.assertIsNone(load_block(self.ws))


class SessionMemoryTests(unittest.TestCase):
    def run_session(self, ws: Path, env: dict, replies: list[dict], recorded: list[dict]) -> LocalCodingAgent:
        def fake(req, timeout=None):  # noqa: ANN001
            recorded.append(json.loads(req.data.decode('utf-8')))
            return _Resp(replies.pop(0))

        for patcher in (patch.dict(os.environ, env), patch('src.openai_compat.request.urlopen', side_effect=fake)):
            patcher.start()
            self.addCleanup(patcher.stop)
        return LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
            runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s' / 'agent',
                                              scratchpad_root=ws / '.s' / 'scratch', nudge_strategy='off',
                                              compact_preserve_messages=1))

    def test_notes_survive_two_compactions_word_for_word(self) -> None:
        recorded: list[dict] = []
        replies = [
            _call('c1', 'session_note', {'note': 'The flaky test is tests/test_io.py::test_retry (timeout 0.2s)'}),
            {'role': 'assistant', 'content': 'noted'},
            {'role': 'assistant', 'content': '<summary>FIRST SUMMARY with detail A</summary>'},
            {'role': 'assistant', 'content': 'more work done'},
            {'role': 'assistant', 'content': '<summary>SECOND SUMMARY</summary>'},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent = self.run_session(ws, {'SIGIL_SESSION_MEMORY': '1'}, replies, recorded)
            self.assertIn('session_note', agent.tool_registry)
            agent.run('find the flaky test')
            path = notes_path(agent)
            self.assertIn('test_retry (timeout 0.2s)', path.read_text(encoding='utf-8'))

            result = compact_conversation(agent, trigger='manual')
            self.assertIsNone(result.error)
            notes = [m for m in agent.last_session.messages if m.metadata.get('kind') == 'session_notes']
            self.assertEqual(len(notes), 1)
            self.assertIn('test_retry (timeout 0.2s)', notes[0].content)
            self.assertIn('FIRST SUMMARY', path.read_text(encoding='utf-8'))

            agent.resume('keep going', load_agent_session(agent.active_session_id,
                                                          directory=ws / '.s' / 'agent'))
            result = compact_conversation(agent, trigger='manual')
            self.assertIsNone(result.error)
            notes = [m for m in agent.last_session.messages if m.metadata.get('kind') == 'session_notes']
            self.assertEqual(len(notes), 1)
            # the first summary comes back as written, not re-summarized by the model
            self.assertIn('FIRST SUMMARY with detail A', notes[0].content)
            self.assertIn('test_retry (timeout 0.2s)', notes[0].content)
            text = path.read_text(encoding='utf-8')
            self.assertEqual(text.count('## Compaction summary'), 2)

    def test_off_by_default(self) -> None:
        recorded: list[dict] = []
        replies = [{'role': 'assistant', 'content': 'hi'},
                   {'role': 'assistant', 'content': '<summary>S</summary>'}]
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent = self.run_session(ws, {'SIGIL_SESSION_MEMORY': ''}, replies, recorded)
            self.assertNotIn('session_note', agent.tool_registry)
            agent.run('hello')
            compact_conversation(agent, trigger='manual')
            self.assertFalse(any(m.metadata.get('kind') == 'session_notes' for m in agent.last_session.messages))
            self.assertFalse(notes_path(agent).exists())


def make_repo(root: Path) -> None:
    (root / 'core').mkdir(parents=True)
    (root / 'core' / 'engine.py').write_text(
        'class Engine:\n    def run(self):\n        pass\n    def _private(self):\n        pass\n\n'
        'def make_engine():\n    return Engine()\n', encoding='utf-8')
    (root / 'app').mkdir()
    (root / 'app' / 'main.py').write_text(
        'from core.engine import Engine, make_engine\n\ndef start():\n'
        '    e = Engine()\n    f = make_engine()\n    return Engine, e, f\n', encoding='utf-8')
    (root / 'web').mkdir()
    (root / 'web' / 'view.ts').write_text('export function renderPage() {}\nexport class Widget {}\n',
                                          encoding='utf-8')
    (root / 'tools.go').write_text('package main\nfunc ParseArgs() {}\ntype Config struct {}\n', encoding='utf-8')
    (root / 'README.md').write_text('# Demo\n', encoding='utf-8')
    (root / 'node_modules' / 'x').mkdir(parents=True)
    (root / 'node_modules' / 'x' / 'index.js').write_text('function hidden() {}\n', encoding='utf-8')


class RepoMapTests(unittest.TestCase):
    def test_symbols_ranking_and_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_repo(root)
            self.assertEqual(file_symbols(root / 'core' / 'engine.py'), ['class Engine: run', 'def make_engine()'])
            self.assertEqual(file_symbols(root / 'web' / 'view.ts'), ['renderPage', 'Widget'])
            self.assertEqual(file_symbols(root / 'tools.go'), ['ParseArgs', 'Config'])
            text = build_repo_map(root)
            self.assertIn('core/engine.py\n  class Engine: run\n  def make_engine()', text)
            self.assertLess(text.index('core/engine.py'), text.index('app/main.py'))  # used most, first
            self.assertIn('README.md', text)
            self.assertNotIn('node_modules', text)
            small = build_repo_map(root, max_chars=120)
            self.assertLessEqual(len(small), 160)
            self.assertIn('more file(s)', small)

    def test_sent_to_the_model_only_when_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_repo(root)
            for value, expected in (('', False), ('1', True)):
                clear_context_caches()
                recorded: list[dict] = []

                def fake(req, timeout=None):  # noqa: ANN001
                    recorded.append(json.loads(req.data.decode('utf-8')))
                    return _Resp({'role': 'assistant', 'content': 'ok'})

                with patch.dict(os.environ, {'SIGIL_REPO_MAP': value}), \
                        patch('src.openai_compat.request.urlopen', side_effect=fake):
                    agent = LocalCodingAgent(
                        model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                        runtime_config=AgentRuntimeConfig(cwd=root, session_directory=root / '.s' / 'agent',
                                                          scratchpad_root=root / '.s' / 'scratch',
                                                          nudge_strategy='off'))
                    agent.run('where is the engine?')
                    shown = agent.run('/map').final_output
                sent = '\n'.join(str(m.get('content')) for m in recorded[0]['messages'])
                self.assertEqual('class Engine: run' in sent, expected, value)
                self.assertIn('core/engine.py', shown)
                self.assertEqual('Not sent to the model' in shown, not expected)
            clear_context_caches()


class ArchitectTests(unittest.TestCase):
    def test_architect_plans_and_editor_model_edits(self) -> None:
        recorded: list[dict] = []
        architect = [_call('a1', 'editor_edit', {'instructions': 'Create hello.txt containing: hi there',
                                                 'files': ['hello.txt']}),
                     {'role': 'assistant', 'content': 'Planned and checked.'}]
        editor = [_call('e1', 'write_file', {'path': 'hello.txt', 'content': 'hi there'}),
                  {'role': 'assistant', 'content': 'Changed hello.txt: created it.'}]

        def fake(req, timeout=None):  # noqa: ANN001
            body = json.loads(req.data.decode('utf-8'))
            recorded.append(body)
            return _Resp((editor if body['model'] == 'small' else architect).pop(0))

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            with patch.dict(os.environ, {'SIGIL_ARCHITECT': '1', 'SIGIL_EDITOR_MODEL': 'small'}), \
                    patch('src.openai_compat.request.urlopen', side_effect=fake):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='big', base_url='http://127.0.0.1:8000/v1'),
                    runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s' / 'agent',
                                                      scratchpad_root=ws / '.s' / 'scratch', nudge_strategy='off',
                                                      permissions=AgentPermissions(allow_file_write=True)))
                self.assertIn('editor_edit', agent.tool_registry)
                self.assertNotIn('write_file', agent.tool_registry)
                self.assertIn('read_file', agent.tool_registry)
                self.assertIn('small makes the edits', agent.run('/architect').final_output)
                result = agent.run('add a greeting file')
            self.assertEqual((ws / 'hello.txt').read_text(encoding='utf-8'), 'hi there')
            self.assertEqual(result.final_output, 'Planned and checked.')
            big = [r for r in recorded if r['model'] == 'big']
            small = [r for r in recorded if r['model'] == 'small']
            self.assertEqual((len(big), len(small)), (2, 2))
            big_tools = {tool['function']['name'] for tool in big[0]['tools']}
            small_tools = {tool['function']['name'] for tool in small[0]['tools']}
            self.assertNotIn('write_file', big_tools)
            self.assertIn('write_file', small_tools)
            self.assertNotIn('editor_edit', small_tools)
            self.assertIn('You are the editor', small[0]['messages'][0]['content'])
            self.assertTrue(any('Create hello.txt containing: hi there' in str(m.get('content'))
                                and 'Files: hello.txt' in str(m.get('content')) for m in small[0]['messages']))
            tool_reply = [m for m in big[1]['messages'] if m.get('role') == 'tool'][-1]['content']
            self.assertIn('Editor (small) finished', tool_reply)
            self.assertIn('Changed: hello.txt', tool_reply)
            self.assertIn('Changed hello.txt: created it.', tool_reply)

    def test_off_by_default_and_flags(self) -> None:
        from src.main import _apply_flag_environment, build_parser

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_ARCHITECT': '',
                                                                           'SIGIL_EDITOR_MODEL': ''}):
            ws = Path(tmp)
            agent = LocalCodingAgent(model_config=ModelConfig(model='m'),
                                     runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s'))
            self.assertNotIn('editor_edit', agent.tool_registry)
            self.assertIn('write_file', agent.tool_registry)
            self.assertIn('Architect mode is off', agent.run('/architect').final_output)
            _apply_flag_environment(build_parser().parse_args(
                ['agent', 'x', '--architect', '--editor-model', 'qwen@http://127.0.0.1:1234/v1']))
            self.assertEqual(os.environ['SIGIL_ARCHITECT'], '1')
            from src.architect import editor_model_config

            cfg = editor_model_config(ModelConfig(model='m'), ws)
            self.assertEqual((cfg.model, cfg.base_url), ('qwen', 'http://127.0.0.1:1234/v1'))


if __name__ == '__main__':
    unittest.main()
