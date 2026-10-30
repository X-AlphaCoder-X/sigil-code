"""prompt_constants is the single source for values the code uses: each test drives the agent
(fake model) or the UI and checks the constant reaches the request or the screen."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src import prompt_constants
from src.agent_context import clear_context_caches
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig

BASE = 'http://127.0.0.1:8000/v1'


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


def tool_call(name: str, **arguments) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'c1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name).resolve()
        self.replies: list[dict] = []
        self.requests: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            self.requests.append(json.loads(req.data.decode('utf-8')))
            return _Resp(self.replies.pop(0) if self.replies else {'role': 'assistant', 'content': 'ok'})

        patcher = patch('src.openai_compat.request.urlopen', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        clear_context_caches()
        self.addCleanup(clear_context_caches)
        prompt_constants.clear_system_prompt_sections()
        self.addCleanup(prompt_constants.clear_system_prompt_sections)

    def agent(self, **perms) -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url=BASE),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s',
                                              scratchpad_root=self.ws / '.scratch', nudge_strategy='off',
                                              permissions=AgentPermissions(**perms)))

    def system(self, index: int = -1) -> str:
        return str(self.requests[index]['messages'][0]['content'])

    def sent(self, index: int = -1) -> str:
        return json.dumps(self.requests[index]['messages'])


class PromptTextTests(Base):
    def test_boundary_sections_language_date_and_scratchpad(self) -> None:
        from src import agent_prompting

        self.assertIs(agent_prompting.SYSTEM_PROMPT_DYNAMIC_BOUNDARY, prompt_constants.SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        with patch.dict(os.environ, {'SIGIL_LANGUAGE': 'German', 'SIGIL_OVERRIDE_DATE': '2031-03-04',
                                     'SIGIL_DISABLE_MICROCOMPACT': ''}):
            self.agent().run('hi')
        system = self.system()
        self.assertIn(prompt_constants.SYSTEM_PROMPT_DYNAMIC_BOUNDARY, system)
        self.assertIn(prompt_constants.SYSTEM_REMINDERS_SECTION, system)
        self.assertIn(prompt_constants.SUMMARIZE_TOOL_RESULTS_SECTION, system)
        self.assertIn(prompt_constants.get_language_section('German'), system)
        self.assertIn("Today's date is 2031-03-04.", self.sent())
        note = prompt_constants.get_scratchpad_instructions(str(self.ws / '.scratch'))
        self.assertIn(json.dumps(note)[1:-1], self.sent())  # the whole note, as JSON text

    def test_static_sections_are_cached_until_clear_or_compaction(self) -> None:
        from src import agent_prompting

        original = agent_prompting.get_intro_section()
        self.agent().run('one')
        self.assertIn(original, self.system())
        with patch('src.agent_prompting.get_intro_section', return_value='PATCHED INTRO'):
            self.agent().run('two')
            self.assertNotIn('PATCHED INTRO', self.system())  # served from the cache
            agent = self.agent()
            agent.run('/clear')
            agent.run('three')
            self.assertIn('PATCHED INTRO', self.system())  # /clear started the cache again
        self.assertIn('intro', prompt_constants._section_cache)
        self.replies = [{'role': 'assistant', 'content': 'a'}, {'role': 'assistant', 'content': 'b'},
                        {'role': 'assistant', 'content': '<summary>S</summary>'}]
        from src.compact import compact_conversation

        agent = self.agent()
        agent.runtime_config = agent.runtime_config.__class__(**{**agent.runtime_config.__dict__,
                                                                  'compact_preserve_messages': 1})
        agent.run('first')
        agent.resume('second', __import__('src.session_store', fromlist=['x']).load_agent_session(
            agent.active_session_id, directory=self.ws / '.s'))
        self.assertIsNone(compact_conversation(agent, trigger='manual').error)
        self.assertEqual(prompt_constants._section_cache, {})

    def test_output_styles_come_from_the_constants(self) -> None:
        from src.output_styles import BUILT_IN_STYLES

        learning = prompt_constants.OUTPUT_STYLE_CONFIGS['Learning']
        self.assertEqual(BUILT_IN_STYLES['learning'], (learning.description, learning.prompt))
        agent = self.agent()
        agent.run('/output-style learning')
        agent.run('teach me')
        self.assertIn(learning.prompt, self.system())


class LimitTests(Base):
    def test_image_limit(self) -> None:
        image = self.ws / 'pic.png'
        image.write_bytes(b'\x89PNG\r\n\x1a\n' + b'0' * 200)
        agent = self.agent()
        with patch.object(prompt_constants, 'API_IMAGE_MAX_BASE64_SIZE', 100):
            problems = agent.attach_images([str(image)])
        self.assertTrue(problems)  # bigger than the (patched) limit
        self.assertEqual(agent.attach_images([str(image)]), [])
        agent.run('what is in the picture?')
        self.assertIn('image_url', self.sent())

    def test_pdf_pages_per_read(self) -> None:
        class Page:
            def __init__(self, number: int) -> None:
                self.number = number

            def extract_text(self) -> str:
                return f'PAGE-{self.number}-TEXT'

        fake_pypdf = types.SimpleNamespace(PdfReader=lambda path: types.SimpleNamespace(
            pages=[Page(n) for n in range(1, 31)]))
        (self.ws / 'doc.pdf').write_bytes(b'%PDF-1.4 fake')
        self.replies = [tool_call('read_file', path='doc.pdf'), {'role': 'assistant', 'content': 'read'},
                        tool_call('read_file', path='doc.pdf'), {'role': 'assistant', 'content': 'read'}]
        with patch.dict(sys.modules, {'pypdf': fake_pypdf}):
            agent = self.agent()
            agent.run('read it')
            first = self.sent()
            with patch.object(prompt_constants, 'PDF_MAX_PAGES_PER_READ', 3):
                agent.run('read it again')
            second = self.sent()
        default = prompt_constants.PDF_MAX_PAGES_PER_READ
        self.assertIn(f'PAGE-{default}-TEXT', first)
        self.assertNotIn(f'PAGE-{default + 1}-TEXT', first)
        self.assertIn('PAGE-3-TEXT', second.split('read it again')[-1])
        self.assertNotIn('PAGE-4-TEXT', second.split('read it again')[-1])

    def test_result_size_in_tool_schemas_and_read_only(self) -> None:
        agent = self.agent()
        agent.run('hi')
        limits = {tool['function']['name']: tool['function']['parameters']['properties']['max_chars']['maximum']
                  for tool in self.requests[-1]['tools']
                  if 'maximum' in tool['function']['parameters'].get('properties', {}).get('max_chars', {})}
        self.assertTrue(limits)
        self.assertEqual(set(limits.values()), {prompt_constants.DEFAULT_MAX_RESULT_SIZE_CHARS})
        (self.ws / 'spec.md').write_text('ABCDEFGHIJKLMNOP', encoding='utf-8')
        with patch.object(prompt_constants, 'DEFAULT_MAX_RESULT_SIZE_CHARS', 5):
            agent.run('/read-only spec.md')
        agent.run('go')
        self.assertIn('ABCDE\\n', self.sent())
        self.assertNotIn('ABCDEF', self.sent())


class UiTests(Base):
    def test_bug_url_effort_figure_tags_and_completion_line(self) -> None:
        agent = self.agent(allow_shell_commands=True)
        with patch.object(prompt_constants, 'PRODUCT_URL', 'https://example.test/repo'), \
                patch('webbrowser.open') as opened:
            agent.run('/bug crash')
        self.assertTrue(opened.call_args[0][0].startswith('https://example.test/repo/issues/new?'))
        self.assertIn(prompt_constants.EFFORT_HIGH, agent.run('/effort high').final_output)

        from src.main import _run_agent_chat_loop

        inputs = iter(['!echo tagged', 'and now?'])

        def scripted(_label: str = '') -> str:
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        output: list[str] = []
        with patch.dict(os.environ, {'SIGIL_COMPLETION_MIN_SECONDS': '0'}):
            _run_agent_chat_loop(self.agent(allow_shell_commands=True), initial_prompt=None, resume_session_id=None,
                                 show_transcript=False, input_func=scripted, output_func=output.append)
        sent = self.sent()
        self.assertIn(f'<{prompt_constants.BASH_INPUT_TAG}>echo tagged</{prompt_constants.BASH_INPUT_TAG}>', sent)
        self.assertIn(f'<{prompt_constants.BASH_STDOUT_TAG}>', sent)
        done = [line for line in output if prompt_constants.TEARDROP_ASTERISK in line]
        self.assertTrue(done, output)
        self.assertTrue(any(verb in done[-1] for verb in prompt_constants.TURN_COMPLETION_VERBS))

    def test_tui_shows_a_spinner_verb(self) -> None:
        from src.tui_display import SigilTUIDisplay

        tui = SigilTUIDisplay()
        tui.on_event({'type': 'model_query', 'turn_index': 0})
        verb = tui.status['last_action'].rstrip('.')
        self.assertIn(verb, prompt_constants.SPINNER_VERBS)


if __name__ == '__main__':
    unittest.main()
