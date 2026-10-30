"""Tests for command hooks (settings.json "hooks")."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.agent_session import AgentSessionState
from src.command_hooks import CommandHookRuntime

PY = sys.executable.replace('\\', '/')


def _write_settings(root: Path, hooks: dict, name: str = 'settings.json') -> None:
    (root / '.sigil').mkdir(exist_ok=True)
    (root / '.sigil' / name).write_text(json.dumps({'hooks': hooks}), encoding='utf-8')


def _script(root: Path, name: str, body: str) -> str:
    path = root / name
    path.write_text(body, encoding='utf-8')
    return f'"{PY}" "{path.as_posix()}"'


class CommandHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        patcher = mock.patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_exit_2_blocks_with_stderr_reason(self) -> None:
        cmd = _script(self.root, 'deny.py', 'import sys; sys.stderr.write("no rm please"); sys.exit(2)')
        _write_settings(self.root, {'PreToolUse': [{'matcher': 'Bash', 'hooks': [{'type': 'command', 'command': cmd}]}]})
        runtime = CommandHookRuntime.from_workspace(self.root)
        outcome = runtime.run('PreToolUse', {'tool_input': {'command': 'rm -rf x'}}, tool_name='bash')
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.reason, 'no rm please')

    def test_matcher_uses_rule_tool_names_and_regex(self) -> None:
        cmd = _script(self.root, 'deny.py', 'import sys; sys.exit(2)')
        _write_settings(self.root, {'PreToolUse': [{'matcher': 'Edit|Write', 'hooks': [{'command': cmd}]}]})
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertTrue(runtime.run('PreToolUse', {}, tool_name='write_file').blocked)
        self.assertTrue(runtime.run('PreToolUse', {}, tool_name='edit_file').blocked)
        self.assertEqual(runtime.run('PreToolUse', {}, tool_name='bash').ran, 0)

    def test_stdin_payload_and_json_deny(self) -> None:
        body = (
            'import json,sys\n'
            'd=json.load(sys.stdin)\n'
            'ok = d["tool_name"]=="Bash" and d["tool_input"]["command"]=="ls" and d["hook_event_name"]=="PreToolUse"\n'
            'print(json.dumps({"hookSpecificOutput":{"permissionDecision":"deny","permissionDecisionReason":"payload ok" if ok else "bad payload"}}))\n'
        )
        cmd = _script(self.root, 'json_deny.py', body)
        _write_settings(self.root, {'PreToolUse': [{'matcher': '*', 'hooks': [{'command': cmd}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run(
            'PreToolUse', {'tool_input': {'command': 'ls'}}, tool_name='bash'
        )
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.reason, 'payload ok')

    def test_updated_input(self) -> None:
        body = 'import json; print(json.dumps({"hookSpecificOutput":{"permissionDecision":"allow","updatedInput":{"command":"ls -la"}}}))'
        cmd = _script(self.root, 'rewrite.py', body)
        _write_settings(self.root, {'PreToolUse': [{'hooks': [{'command': cmd}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertFalse(outcome.blocked)
        self.assertEqual(outcome.updated_input, {'command': 'ls -la'})

    def test_non_blocking_error_vs_on_failure_block(self) -> None:
        cmd = _script(self.root, 'crash.py', 'import sys; sys.exit(1)')
        _write_settings(self.root, {'PreToolUse': [{'hooks': [{'command': cmd}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertFalse(outcome.blocked)
        self.assertEqual(len(outcome.errors), 1)

        _write_settings(self.root, {'PreToolUse': [{'hooks': [{'command': cmd, 'onFailure': 'block'}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertTrue(outcome.blocked)

    def test_timeout_with_on_failure_block(self) -> None:
        cmd = _script(self.root, 'slow.py', 'import time; time.sleep(5)')
        _write_settings(
            self.root,
            {'PreToolUse': [{'hooks': [{'command': cmd, 'timeout': 0.5, 'onFailure': 'block'}]}]},
        )
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertTrue(outcome.blocked)
        self.assertIn('timed out', outcome.reason)

    def test_user_prompt_submit_plain_stdout_is_context(self) -> None:
        cmd = _script(self.root, 'ctx.py', 'print("Today is a release freeze.")')
        _write_settings(self.root, {'UserPromptSubmit': [{'hooks': [{'command': cmd}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('UserPromptSubmit', {'prompt': 'hi'})
        self.assertEqual(outcome.context_text(), 'Today is a release freeze.')

    def test_local_settings_merge_and_disable_all(self) -> None:
        cmd = _script(self.root, 'ok.py', 'pass')
        _write_settings(self.root, {'Stop': [{'hooks': [{'command': cmd}]}]})
        _write_settings(self.root, {'PostToolUse': [{'hooks': [{'command': cmd}]}]}, name='settings.local.json')
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertTrue(runtime.has('Stop') and runtime.has('PostToolUse'))

        (self.root / '.sigil' / 'settings.local.json').write_text(json.dumps({'disableAllHooks': True}), encoding='utf-8')
        self.assertEqual(CommandHookRuntime.from_workspace(self.root).hooks, ())

    def test_unsupported_event_reported(self) -> None:
        _write_settings(self.root, {'Bogus': [{'hooks': [{'command': 'x'}]}]})
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertEqual(runtime.hooks, ())
        self.assertTrue(any('Bogus' in e for e in runtime.load_errors))


class NewHookTypesAndEventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        patcher = mock.patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1', 'HOOK_TOKEN': 's3cret'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _serve(self, status: int, body: str):
        import http.server
        import threading

        received: dict = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get('Content-Length', 0))
                received['body'] = json.loads(self.rfile.read(length))
                received['auth'] = self.headers.get('Authorization')
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_address[1]}/hook', received

    def test_http_hook_json_deny_and_header_env_expansion(self) -> None:
        url, received = self._serve(200, json.dumps({'decision': 'block', 'reason': 'server says no'}))
        _write_settings(self.root, {'PreToolUse': [{'matcher': 'Bash', 'hooks': [
            # only variables listed in allowedEnvVars are resolved
            {'type': 'http', 'url': url, 'headers': {'Authorization': 'Bearer $HOOK_TOKEN'},
             'allowedEnvVars': ['HOOK_TOKEN']}
        ]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run(
            'PreToolUse', {'tool_input': {'command': 'ls'}}, tool_name='bash'
        )
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.reason, 'server says no')
        self.assertEqual(received['auth'], 'Bearer s3cret')
        self.assertEqual(received['body']['tool_name'], 'Bash')

    def test_http_403_blocks_and_500_is_non_blocking(self) -> None:
        # a non-2xx status is a non-blocking error; it blocks only with
        # "onFailure": "block". A server blocks with a 2xx JSON decision body (test above).
        url, _ = self._serve(403, 'forbidden by policy')
        _write_settings(self.root, {'PreToolUse': [{'hooks': [{'type': 'http', 'url': url}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertFalse(outcome.blocked)
        self.assertIn('forbidden by policy', outcome.errors[0])
        _write_settings(self.root, {'PreToolUse': [{'hooks': [{'type': 'http', 'url': url, 'onFailure': 'block'}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertTrue(outcome.blocked)
        self.assertIn('forbidden by policy', outcome.reason)

        url, _ = self._serve(500, 'oops')
        _write_settings(self.root, {'PreToolUse': [{'hooks': [{'type': 'http', 'url': url}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('PreToolUse', {}, tool_name='bash')
        self.assertFalse(outcome.blocked)
        self.assertEqual(len(outcome.errors), 1)

    def test_prompt_hook_uses_model_and_arguments(self) -> None:
        _write_settings(self.root, {'PreToolUse': [{'hooks': [
            {'type': 'prompt', 'prompt': 'Block anything touching prod. Input: $ARGUMENTS'}
        ]}]})
        runtime = CommandHookRuntime.from_workspace(self.root)
        seen: dict = {}

        def fake_model(system: str, prompt: str, timeout: float) -> str:
            seen['prompt'] = prompt
            return 'Sure. {"ok": false, "reason": "touches prod"}'

        runtime.model_complete = fake_model
        outcome = runtime.run('PreToolUse', {'tool_input': {'command': 'deploy prod'}}, tool_name='bash')
        self.assertTrue(outcome.blocked)
        self.assertEqual(outcome.reason, 'touches prod')
        self.assertIn('deploy prod', seen['prompt'])
        self.assertNotIn('$ARGUMENTS', seen['prompt'])

        runtime.model_complete = lambda s, p, t: '{"ok": true}'
        self.assertFalse(runtime.run('PreToolUse', {}, tool_name='bash').blocked)

    def test_prompt_hook_without_model_is_non_blocking(self) -> None:
        _write_settings(self.root, {'Stop': [{'hooks': [{'type': 'prompt', 'prompt': 'is the task done?'}]}]})
        outcome = CommandHookRuntime.from_workspace(self.root).run('Stop', {})
        self.assertFalse(outcome.blocked)
        self.assertTrue(outcome.errors)

    def test_missing_target_key_reported(self) -> None:
        _write_settings(self.root, {'Stop': [{'hooks': [{'type': 'http'}]}]})
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertEqual(runtime.hooks, ())
        self.assertTrue(any('"url"' in e for e in runtime.load_errors))

    def test_pre_compact_blocks_and_adds_instructions(self) -> None:
        from types import SimpleNamespace
        from src.compact import compact_conversation

        session = AgentSessionState.create(['sys'], 'hello')
        cmd = _script(self.root, 'block.py', 'import sys; sys.stderr.write("not now"); sys.exit(2)')
        _write_settings(self.root, {'PreCompact': [{'hooks': [{'command': cmd}]}]})
        agent = SimpleNamespace(
            last_session=session,
            command_hook_runtime=CommandHookRuntime.from_workspace(self.root),
            active_session_id='s1',
        )
        result = compact_conversation(agent, trigger='manual')
        self.assertIn('not now', result.error)

    def test_session_end_helper_runs_and_never_raises(self) -> None:
        from src.command_hooks import fire_session_end

        out = (self.root / 'ended.json').as_posix()
        cmd = _script(self.root, 'end.py', f'import sys,shutil; open({out!r},"w").write(sys.stdin.read())')
        _write_settings(self.root, {'SessionEnd': [{'hooks': [{'command': cmd}]}]})
        fire_session_end(CommandHookRuntime.from_workspace(self.root), 'abc', 'exit')
        payload = json.loads(Path(out).read_text(encoding='utf-8'))
        self.assertEqual((payload['session_id'], payload['reason']), ('abc', 'exit'))
        fire_session_end(None, 'abc', 'exit')  # no runtime: no-op


def _tool_call_response(name: str, args: dict) -> dict:
    return {'choices': [{'message': {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
    ]}, 'finish_reason': 'tool_calls'}]}


def _text_response(text: str) -> dict:
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}


class AgentLoopHookTests(unittest.TestCase):
    """End-to-end: hooks fire inside LocalCodingAgent's real tool loop (fake model backend)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        patcher = mock.patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _run_agent(self, responses: list[dict]):
        from tests.test_agent_runtime import make_urlopen_side_effect
        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig, ModelConfig

        with mock.patch('src.openai_compat.request.urlopen', side_effect=make_urlopen_side_effect(responses)):
            agent = LocalCodingAgent(
                model_config=ModelConfig(model='fake', base_url='http://127.0.0.1:8000/v1'),
                runtime_config=AgentRuntimeConfig(cwd=self.root),
            )
            return agent.run('do it')

    def test_pre_tool_use_hook_blocks_real_tool_call(self) -> None:
        (self.root / 'secret.txt').write_text('TOP SECRET', encoding='utf-8')
        cmd = _script(self.root, 'deny.py', 'import sys; sys.stderr.write("secrets are off limits"); sys.exit(2)')
        _write_settings(self.root, {'PreToolUse': [{'matcher': 'Read', 'hooks': [{'command': cmd}]}]})
        result = self._run_agent([_tool_call_response('read_file', {'path': 'secret.txt'}), _text_response('ok')])
        tool_messages = [m for m in result.transcript if m.get('role') == 'tool']
        self.assertTrue(tool_messages)
        self.assertIn('secrets are off limits', tool_messages[0]['content'])
        self.assertNotIn('TOP SECRET', tool_messages[0]['content'])

    def test_stop_hook_keeps_agent_working_once(self) -> None:
        marker = (self.root / 'stop_seen').as_posix()
        body = (
            'import os,sys\n'
            f'p={marker!r}\n'
            'if os.path.exists(p): sys.exit(0)\n'
            'open(p,"w").close(); sys.stderr.write("run the tests first"); sys.exit(2)\n'
        )
        cmd = _script(self.root, 'stop.py', body)
        _write_settings(self.root, {'Stop': [{'hooks': [{'command': cmd}]}]})
        result = self._run_agent([_text_response('done early'), _text_response('tests ran, done')])
        self.assertEqual(result.final_output, 'tests ran, done')
        self.assertTrue(any('run the tests first' in str(m.get('content')) for m in result.transcript))

    def test_notification_idle_and_session_start_context(self) -> None:
        log = (self.root / 'events.log').as_posix()
        logger = _script(
            self.root, 'log.py',
            f'import json,sys; d=json.load(sys.stdin); open({log!r},"a").write(d["hook_event_name"]+":"+d.get("notification_type","")+"\\n")',
        )
        ctx = _script(self.root, 'ctx.py', 'print("Branch policy: never push to main")')
        _write_settings(self.root, {
            'Notification': [{'hooks': [{'command': logger}]}],
            'SessionStart': [{'hooks': [{'command': ctx}]}],
        })
        result = self._run_agent([_text_response('hi')])
        self.assertIn('Notification:idle_prompt', Path(log).read_text(encoding='utf-8'))
        user_texts = [str(m.get('content')) for m in result.transcript if m.get('role') == 'user']
        self.assertTrue(any('do it' in t and 'Branch policy: never push to main' in t for t in user_texts))

    def test_subagent_stop_fires_for_delegated_child_not_stop(self) -> None:
        log = (self.root / 'events.log').as_posix()
        logger = _script(
            self.root, 'log.py',
            f'import json,sys; d=json.load(sys.stdin); open({log!r},"a").write(d["hook_event_name"]+"\\n")',
        )
        _write_settings(self.root, {
            'SubagentStop': [{'hooks': [{'command': logger}]}],
            'Stop': [{'hooks': [{'command': logger}]}],
        })
        responses = [
            _tool_call_response('delegate_agent', {'prompt': 'say hi'}),
            _text_response('child done'),   # child agent's only turn
            _text_response('parent done'),  # parent's final turn
        ]
        self._run_agent(responses)
        events = Path(log).read_text(encoding='utf-8').split()
        self.assertEqual(events, ['SubagentStop', 'Stop'])


if __name__ == '__main__':
    unittest.main()
