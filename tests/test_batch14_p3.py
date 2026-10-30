"""Batch 14 (P3): memory-file rules (AGENTS.md / GEMINI.md / AGENTS.override.md, nested,
sigilMdExcludes), /memory refresh, language, --exclude-dynamic-system-prompt-sections,
extra request headers, disableSkillShellExecution, /reload-skills and /reload-plugins, context window detection."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_context import _sigil_md_at_or_above, clear_context_caches, memory_file_paths, nested_memory_for
from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.custom_commands import run_shell_injections


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
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001
        self.requests.append(json.loads(req.data.decode('utf-8')))
        self.headers.append({k.lower(): v for k, v in req.header_items()})
        return _Resp({'role': 'assistant', 'content': 'ok'})

    def text(self, index: int = -1) -> str:
        return '\n'.join(str(m.get('content')) for m in self.requests[index]['messages'])

    def system(self, index: int = -1) -> str:
        return str(self.requests[index]['messages'][0]['content'])


def make_agent(ws: Path, **perms) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
        runtime_config=AgentRuntimeConfig(cwd=ws, session_directory=ws / '.s' / 'agent',
                                          scratchpad_root=ws / '.s' / 'scratch', nudge_strategy='off',
                                          permissions=AgentPermissions(**perms)))


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ws = self.root / 'repo' / 'pkg'
        self.ws.mkdir(parents=True)
        home = self.root / 'home'
        home.mkdir()
        for patcher in (patch('src.settings._home', return_value=home),):
            patcher.start()
            self.addCleanup(patcher.stop)
        clear_context_caches()
        self.addCleanup(clear_context_caches)
        self.model = FakeModel()
        patcher = patch('src.openai_compat.request.urlopen', side_effect=self.model)
        patcher.start()
        self.addCleanup(patcher.stop)


class MemoryRuleTests(Base):
    def names(self) -> list[str]:
        # only files in the test folder: the real ~/.sigil/SIGIL.md (user memory) may load too
        return [p.relative_to(self.root).as_posix() for p in memory_file_paths(self.ws)
                if p.is_relative_to(self.root)]

    def test_sigil_local_md_above_turns_agents_md_off(self) -> None:
        self.assertFalse(_sigil_md_at_or_above(self.ws))  # ~/.sigil/SIGIL.md is user memory
        (self.ws / 'AGENTS.md').write_text('agents here', encoding='utf-8')
        self.assertEqual(self.names(), ['repo/pkg/AGENTS.md'])
        (self.root / 'repo' / 'SIGIL.local.md').write_text('local notes', encoding='utf-8')
        self.assertEqual(self.names(), ['repo/SIGIL.local.md'])

    def test_override_and_gemini_fallbacks_and_nesting(self) -> None:
        (self.root / 'repo' / 'GEMINI.md').write_text('gemini root', encoding='utf-8')
        (self.ws / 'AGENTS.md').write_text('agents pkg', encoding='utf-8')
        (self.ws / 'AGENTS.override.md').write_text('override pkg', encoding='utf-8')
        self.assertEqual(self.names(), ['repo/GEMINI.md', 'repo/pkg/AGENTS.override.md'])  # nearest last
        sub = self.ws / 'sub'
        sub.mkdir()
        (sub / 'AGENTS.md').write_text('agents sub', encoding='utf-8')
        found = nested_memory_for(sub / 'x.py', self.ws, set())
        self.assertEqual([(p.name, body) for p, body in found], [('AGENTS.md', 'agents sub')])
        (self.ws / 'SIGIL.md').write_text('sigil', encoding='utf-8')
        self.assertEqual(nested_memory_for(sub / 'x.py', self.ws, set()), [])  # SIGIL.md above: off

    def test_sigil_md_excludes(self) -> None:
        (self.root / 'repo' / 'SIGIL.md').write_text('root rules', encoding='utf-8')
        (self.ws / 'SIGIL.md').write_text('team rules', encoding='utf-8')
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text(
            json.dumps({'sigilMdExcludes': ['**/repo/SIGIL.md']}), encoding='utf-8')
        self.assertEqual(self.names(), ['repo/pkg/SIGIL.md'])
        make_agent(self.ws).run('hi')
        self.assertIn('team rules', self.model.text())
        self.assertNotIn('root rules', self.model.text())

    def test_memory_refresh_goes_with_the_next_prompt(self) -> None:
        (self.ws / 'SIGIL.md').write_text('first rules', encoding='utf-8')
        agent = make_agent(self.ws)
        first = agent.run('hi')
        (self.ws / 'SIGIL.md').write_text('second rules', encoding='utf-8')
        report = agent.run('/memory refresh').final_output
        self.assertIn('memory file(s)', report)
        self.assertIn(str(self.ws / 'SIGIL.md'), report)
        from src.session_store import load_agent_session

        agent.resume('again', load_agent_session(first.session_id, directory=self.ws / '.s' / 'agent'))
        last_user = [m for m in self.model.requests[-1]['messages'] if m.get('role') == 'user']
        self.assertTrue(any('second rules' in str(m.get('content')) and 'were reloaded' in str(m.get('content'))
                            for m in last_user), last_user)
        agent.resume('third', load_agent_session(first.session_id, directory=self.ws / '.s' / 'agent'))
        sent_once = sum('were reloaded' in str(m.get('content')) for m in self.model.requests[-1]['messages'])
        self.assertEqual(sent_once, 1)  # kept in the history, not added again


class PromptTests(Base):
    def test_language_setting(self) -> None:
        with patch.dict(os.environ, {'SIGIL_LANGUAGE': 'Spanish'}):
            make_agent(self.ws).run('hi')
        self.assertIn('Answer the user in Spanish', self.model.system())
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text('{"language": "French"}', encoding='utf-8')
        with patch.dict(os.environ, {'SIGIL_LANGUAGE': ''}):
            make_agent(self.ws).run('hi')
        self.assertIn('Answer the user in French', self.model.system())

    def test_exclude_dynamic_sections_moves_environment(self) -> None:
        with patch.dict(os.environ, {'SIGIL_EXCLUDE_DYNAMIC_SECTIONS': ''}):
            make_agent(self.ws).run('hi')
        self.assertIn('Primary working directory', self.model.system())
        with patch.dict(os.environ, {'SIGIL_EXCLUDE_DYNAMIC_SECTIONS': '1'}):
            make_agent(self.ws).run('hi')
        self.assertNotIn('Primary working directory', self.model.system())
        self.assertNotIn(str(self.ws), self.model.system())
        self.assertIn('Primary working directory', self.model.text())  # now in the first user message
        from src.main import _apply_flag_environment, build_parser

        with patch.dict(os.environ, {}):
            _apply_flag_environment(build_parser().parse_args(
                ['agent', 'x', '--exclude-dynamic-system-prompt-sections']))
            self.assertEqual(os.environ['SIGIL_EXCLUDE_DYNAMIC_SECTIONS'], '1')

    def test_extra_headers(self) -> None:
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text(
            json.dumps({'extraHeaders': {'X-Title': 'My App'}}), encoding='utf-8')
        with patch.dict(os.environ, {'SIGIL_EXTRA_HEADERS': 'anthropic-workspace-id: ${TEST_WS}; HTTP-Referer: x.dev',
                                     'TEST_WS': 'ws_123'}), patch('pathlib.Path.cwd', return_value=self.ws):
            make_agent(self.ws).run('hi')
        sent = self.model.headers[-1]
        self.assertEqual(sent.get('anthropic-workspace-id'), 'ws_123')
        self.assertEqual(sent.get('http-referer'), 'x.dev')
        self.assertEqual(sent.get('x-title'), 'My App')
        with patch.dict(os.environ, {'SIGIL_EXTRA_HEADERS': '{"X-A": "1"}'}):
            from src.openai_compat import extra_headers

            self.assertEqual(extra_headers().get('X-A'), '1')


class ExtensionTests(Base):
    def test_disable_skill_shell_execution(self) -> None:
        agent = make_agent(self.ws, allow_shell_commands=True)
        self.assertIn('hello-shell', run_shell_injections('out: !`echo hello-shell`', agent))
        with patch.dict(os.environ, {'SIGIL_DISABLE_SKILL_SHELL': '1'}):
            text = run_shell_injections('out: !`echo hello-shell`', agent)
        self.assertNotIn('hello-shell', text)
        self.assertIn('disabled by disableSkillShellExecution', text)
        (self.ws / '.sigil').mkdir()
        (self.ws / '.sigil' / 'settings.json').write_text('{"disableSkillShellExecution": true}',
                                                           encoding='utf-8')
        self.assertIn('disabled', run_shell_injections('!`echo x`', agent))

    def test_reload_skills_and_plugins(self) -> None:
        agent = make_agent(self.ws)
        before = agent.skills_runtime.list_skills()
        skill = self.ws / '.sigil' / 'skills' / 'fresh'
        skill.mkdir(parents=True)
        (skill / 'SKILL.md').write_text('---\nname: fresh\ndescription: A new skill\n---\nDo it.\n',
                                        encoding='utf-8')
        self.assertNotIn('fresh', before)
        self.assertIn('Reloaded', agent.run('/reload-skills').final_output)
        self.assertIn('fresh', agent.skills_runtime.list_skills())
        self.assertIs(agent.tool_context.skills_runtime, agent.skills_runtime)
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'later': {'command': 'nope-cmd'}}}),
                                           encoding='utf-8')
        report = agent.run('/reload-plugins').final_output
        self.assertIn('1 MCP server(s)', report)
        self.assertEqual([s.name for s in agent.mcp_runtime.servers], ['later'])


class _JSON:
    def __init__(self, payload) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class DetectTests(unittest.TestCase):
    def setUp(self) -> None:
        from src import model_info

        model_info._CACHE.clear()
        self.addCleanup(model_info._CACHE.clear)

    def serve(self, routes: dict):
        seen: list[str] = []

        def fake(req, timeout=None):  # noqa: ANN001
            url = req.full_url
            seen.append(url)
            for suffix, payload in routes.items():
                if url.endswith(suffix):
                    if isinstance(payload, Exception):
                        raise payload
                    return _JSON(payload)
            import urllib.error

            raise urllib.error.URLError('no route')

        return fake, seen

    def test_openai_style_models_list(self) -> None:
        from src.model_info import detect

        fake, seen = self.serve({'/v1/models': {'data': [
            {'id': 'other', 'max_model_len': 1}, {'id': 'qwen', 'max_model_len': 32768}]}})
        with patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}), patch('src.model_info.request.urlopen', side_effect=fake):
            info = detect('http://127.0.0.1:8000/v1', 'qwen')
        self.assertEqual(info.context_window, 32768)
        self.assertEqual(seen[0], 'http://127.0.0.1:8000/v1/models')

    def test_openrouter_and_ollama(self) -> None:
        from src.model_info import detect

        fake, _ = self.serve({'/models': {'data': [{'id': 'x/y', 'top_provider': {'context_length': 200000},
                                                    'architecture': {'input_modalities': ['text', 'image']}}]}})
        with patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}), patch('src.model_info.request.urlopen', side_effect=fake):
            info = detect('https://openrouter.ai/api/v1', 'x/y')
        self.assertEqual((info.context_window, info.vision), (200000, True))
        fake, seen = self.serve({'/api/show': {'model_info': {'llama.context_length': 8192},
                                               'capabilities': ['completion']}})
        with patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}), patch('src.model_info.request.urlopen', side_effect=fake):
            info = detect('http://localhost:11434/v1', 'llama3')
        self.assertEqual((info.context_window, info.vision, info.source), (8192, False, 'ollama'))
        self.assertEqual(seen[-1], 'http://localhost:11434/api/show')
        fake, seen = self.serve({})
        with patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}), patch('src.model_info.request.urlopen', side_effect=fake):
            detect('https://api.example.com/v1', 'm')
        self.assertEqual(seen, ['https://api.example.com/v1/models'])  # no Ollama probe on a cloud API

    def test_cli_uses_it_only_for_unset_values(self) -> None:
        from src.main import _build_agent, build_parser

        fake, _ = self.serve({'/api/show': {'parameters': 'num_ctx 16000', 'capabilities': ['completion']}})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '1'}), \
                patch('src.model_info.request.urlopen', side_effect=fake):
            os.environ.pop('SIGIL_VISION', None)
            args = build_parser().parse_args(['agent', 'x', '--cwd', tmp, '--model', 'small',
                                              '--base-url', 'http://127.0.0.1:11434/v1'])
            agent = _build_agent(args)
            self.assertEqual(agent.runtime_config.auto_compact_threshold_tokens, 12800)
            self.assertEqual(os.environ.get('SIGIL_VISION'), '0')
            from src import model_info

            model_info._CACHE.clear()
            args = build_parser().parse_args(['agent', 'x', '--cwd', tmp, '--model', 'small', '--base-url',
                                              'http://127.0.0.1:11434/v1', '--auto-compact-threshold', '5000'])
            self.assertEqual(_build_agent(args).runtime_config.auto_compact_threshold_tokens, 5000)
        with patch.dict(os.environ, {'SIGIL_DETECT_MODEL': '0'}):
            from src.model_info import detect

            self.assertIsNone(detect('http://127.0.0.1:1/v1', 'm').context_window)


if __name__ == '__main__':
    unittest.main()
