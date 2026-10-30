"""Fixes from checking the docs (2026-10-29): Elicitation hook answers + ElicitationResult, MCP
2026-07-28 input_required retries, hook input fields, telemetry endpoints, plugin agents, dontAsk."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.mcp_runtime import MCPRuntime, connection_manager, set_elicitation_handler

PY = sys.executable.replace('\\', '/')
BASE = 'http://127.0.0.1:8000/v1'

MRTR_SERVER = r'''
import json, sys
def send(obj):
    print(json.dumps(obj), flush=True)
for raw in sys.stdin:
    msg = json.loads(raw)
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2026-07-28", "capabilities": {"tools": {}}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [{"name": "book", "inputSchema": {"type": "object"}}]}})
    elif method == "tools/call":
        params = msg["params"]
        answers = params.get("inputResponses")
        if not answers:
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"resultType": "input_required", "requestState": "st-1",
                  "inputRequests": {"who": {"method": "elicitation/create", "params": {"message": "Name?",
                      "requestedSchema": {"type": "object", "properties": {"name": {"type": "string"}}}}}}}})
        else:
            text = "state=" + str(params.get("requestState")) + " " + json.dumps(answers["who"], sort_keys=True)
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": text}]}})
    elif method in ("resources/list", "resources/templates/list", "prompts/list"):
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {method.split("/")[0]: []}})
    elif method == "resources/read":
        params = msg["params"]
        if not params.get("inputResponses"):
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"resultType": "input_required", "requestState": "r-1",
                  "inputRequests": {"ok": {"method": "elicitation/create", "params": {"message": "Share it?",
                      "requestedSchema": {"type": "object", "properties": {}}}}}}})
        else:
            text = params["uri"] + " state=" + str(params.get("requestState")) + " " + params["inputResponses"]["ok"]["action"]
            send({"jsonrpc": "2.0", "id": msg["id"], "result": {"contents": [{"uri": params["uri"], "text": text}]}})
'''


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


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(connection_manager().close_all)
        self.addCleanup(set_elicitation_handler, None)
        self.ws = Path(self.tmp.name).resolve()
        self.replies: list[dict] = []
        self.requests: list[dict] = []

        def fake(req, timeout=None):  # noqa: ANN001
            self.requests.append(json.loads(req.data.decode('utf-8')))
            return _Resp(self.replies.pop(0) if self.replies else {'role': 'assistant', 'content': 'ok'})

        patcher = patch('src.openai_compat.request.urlopen', side_effect=fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def agent(self, **perms) -> LocalCodingAgent:
        return LocalCodingAgent(model_config=ModelConfig(model='m', base_url=BASE),
                                runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s',
                                                                  nudge_strategy='off',
                                                                  permissions=AgentPermissions(**perms)))

    def hook_script(self, name: str) -> tuple[str, Path]:
        out = self.ws / f'{name}.jsonl'
        script = self.ws / f'{name}.py'
        script.write_text('import sys, pathlib\n'
                          f'p = pathlib.Path(r"{out}")\n'
                          'p.open("a", encoding="utf-8").write(sys.stdin.read().replace("\\n", " ") + "\\n")\n',
                          encoding='utf-8')
        return f'"{PY}" "{script.as_posix()}"', out

    def settings(self, hooks: dict) -> None:
        (self.ws / '.sigil').mkdir(exist_ok=True)
        (self.ws / '.sigil' / 'settings.json').write_text(json.dumps({'hooks': hooks}), encoding='utf-8')


class ElicitationTests(Base):
    def test_hook_answer_and_result_event(self) -> None:
        answer = self.ws / 'answer.py'
        answer.write_text('import json\nprint(json.dumps({"hookSpecificOutput": {"hookEventName": "Elicitation", '
                          '"action": "accept", "content": {"name": "FromHook"}}}))\n', encoding='utf-8')
        record, out = self.hook_script('result')
        self.settings({'Elicitation': [{'hooks': [{'type': 'command', 'command': f'"{PY}" "{answer.as_posix()}"'}]}],
                       'ElicitationResult': [{'hooks': [{'type': 'command', 'command': record}]}]})
        agent = self.agent()
        reply = agent.answer_elicitation('srv', {'message': 'Name?', 'requestedSchema': {}},
                                         ask=lambda _l: self.fail('the hook answered; nobody should be asked'))
        self.assertEqual(reply, {'action': 'accept', 'content': {'name': 'FromHook'}})
        event = json.loads(out.read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual((event['hook_event_name'], event['action'], event['content']),
                         ('ElicitationResult', 'accept', {'name': 'FromHook'}))

    def test_2026_input_required_is_answered_and_retried(self) -> None:
        script = self.ws / 'mrtr.py'
        script.write_text(MRTR_SERVER, encoding='utf-8')
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'shop': {
            'command': sys.executable, 'args': ['-u', str(script)]}}}), encoding='utf-8')
        with patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'}):
            runtime = MCPRuntime.from_workspace(self.ws)
            agent = self.agent()
            set_elicitation_handler(lambda server, params: agent.answer_elicitation(
                server, params, ask=lambda _l: 'Ada', output=lambda _t: None))
            text, _ = runtime.call_tool('book', arguments={}, server_name='shop')
        self.assertEqual(text.strip(), 'state=st-1 {"action": "accept", "content": {"name": "Ada"}}')

    def test_2026_input_required_on_resources_read(self) -> None:
        script = self.ws / 'mrtr.py'
        script.write_text(MRTR_SERVER, encoding='utf-8')
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'shop': {
            'command': sys.executable, 'args': ['-u', str(script)]}}}), encoding='utf-8')
        asked: list[str] = []
        with patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'}):
            runtime = MCPRuntime.from_workspace(self.ws)
            set_elicitation_handler(lambda server, params: asked.append(params['message']) or {'action': 'accept',
                                                                                             'content': {}})
            text = runtime.read_resource('file:///notes.txt')
        self.assertEqual(asked, ['Share it?'])
        self.assertEqual(text.strip(), 'file:///notes.txt state=r-1 accept')


class HookFieldTests(Base):
    def test_documented_input_fields(self) -> None:
        record, out = self.hook_script('events')
        self.settings({event: [{'hooks': [{'type': 'command', 'command': record}]}]
                       for event in ('FileChanged', 'DirectoryAdded', 'MessageDisplay')})
        (self.ws / 'old.txt').write_text('x', encoding='utf-8')
        self.replies = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'a', 'type': 'function', 'function': {
                'name': 'write_file', 'arguments': json.dumps({'path': 'new.txt', 'content': 'n'})}}]},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'b', 'type': 'function', 'function': {
                'name': 'write_file', 'arguments': json.dumps({'path': 'old.txt', 'content': 'y'})}}]},
            {'role': 'assistant', 'content': 'done'}]
        agent = self.agent(allow_file_write=True)
        (self.ws / 'more').mkdir()
        from src.main import _run_agent_chat_loop

        inputs = iter([f'/add-dir {self.ws / "more"}', 'write files'])

        def scripted(_label: str = '') -> str:
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                             input_func=scripted, output_func=lambda _t: None)
        events = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
        changed = {e['file_path'].split('\\')[-1].split('/')[-1]: e['event'] for e in events
                   if e['hook_event_name'] == 'FileChanged'}
        self.assertEqual(changed, {'new.txt': 'add', 'old.txt': 'change'})
        added = next(e for e in events if e['hook_event_name'] == 'DirectoryAdded')
        self.assertEqual(added['source'], 'slash_command')
        shown = [e for e in events if e['hook_event_name'] == 'MessageDisplay']
        self.assertEqual((shown[-1]['final'], shown[-1]['delta']), (True, 'done'))


class HookDocsTests(Base):
    """Hook behaviour checked against the hooks reference (2026-10-29)."""

    def py_hook(self, name: str, code: str) -> dict:
        script = self.ws / f'{name}.py'
        script.write_text(code, encoding='utf-8')
        return {'hooks': [{'type': 'command', 'command': f'"{PY}" "{script.as_posix()}"'}]}

    def worktree_hooks(self) -> Path:
        root = self.ws / 'wt'
        create = ('import json, sys, pathlib\n'
                  'data = json.load(sys.stdin)\n'
                  f'root = pathlib.Path(r"{root}")\n'
                  'path = root / data["name"]\n'
                  'path.mkdir(parents=True)\n'
                  '(root / "create.json").write_text(json.dumps(data))\n'
                  'print("making it", file=sys.stderr)\n'
                  'print(path)\n')
        remove = ('import json, sys, shutil, pathlib\n'
                  'data = json.load(sys.stdin)\n'
                  f'(pathlib.Path(r"{root}") / "remove.json").write_text(json.dumps(data))\n'
                  'shutil.rmtree(data["worktree_path"])\n')
        self.settings({'WorktreeCreate': [self.py_hook('create', create)],
                       'WorktreeRemove': [self.py_hook('remove', remove)]})
        return root

    def test_worktree_create_hook_replaces_git_and_remove_hook_cleans_up(self) -> None:
        root = self.worktree_hooks()
        self.replies = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'w', 'type': 'function', 'function': {
                'name': 'worktree_enter', 'arguments': json.dumps({'name': 'feat'})}}]},
            {'role': 'assistant', 'content': 'in it'}]
        agent = self.agent(allow_file_write=True)
        agent.run('enter a worktree')
        made = (root / 'feat').resolve()
        self.assertTrue(made.is_dir())
        self.assertEqual(agent.runtime_config.cwd.resolve(), made)  # the session works in the hook's folder
        self.assertEqual(json.loads((root / 'create.json').read_text())['name'], 'feat')
        self.assertFalse((self.ws / '.git').exists())  # git was not used: the workspace is not even a repo
        self.assertIn('discard_changes', agent.render_worktree_exit_report(action='remove'))
        self.assertTrue(made.is_dir())  # git cannot see its changes, so removing needs discard_changes
        self.assertIn('Removed', agent.render_worktree_exit_report(action='remove', discard_changes=True))
        self.assertFalse(made.exists())
        self.assertEqual(Path(json.loads((root / 'remove.json').read_text())['worktree_path']).resolve(), made)
        self.assertEqual(agent.runtime_config.cwd.resolve(), self.ws)

    def test_worktree_flag_subagent_and_failed_hook(self) -> None:
        import argparse
        import io
        from contextlib import redirect_stderr

        from src.main import _enter_worktree

        self.settings({'WorktreeCreate': [self.py_hook('silent', 'import sys\nsys.stdin.read()\n')]})
        self.assertIn('gave no worktree path', self.agent().render_worktree_enter_report('x'))  # fails creation
        root = self.worktree_hooks()
        args = argparse.Namespace(cwd=str(self.ws), worktree='cli')
        with redirect_stderr(io.StringIO()):
            _enter_worktree(args)  # -w/--worktree
        self.assertEqual(Path(args.cwd).resolve(), (root / 'cli').resolve())
        agent = self.agent()
        path = agent._create_isolated_worktree('job')  # isolation: worktree
        self.assertEqual(path.parent, root.resolve())
        self.assertIn('kept', agent._finish_isolated_worktree(path))
        self.assertTrue(path.is_dir())  # a subagent's hook-made worktree is kept, WorktreeRemove not run
        self.assertFalse((root / 'remove.json').exists())

    def test_elicitation_result_hook_changes_blocks_or_cancels(self) -> None:
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}

        def answer(code: str) -> dict:
            self.settings({'ElicitationResult': [self.py_hook('result', code)]})
            return self.agent().answer_elicitation('srv', {'message': 'Name?', 'requestedSchema': schema},
                                                   ask=lambda _l: 'Ada', output=lambda _t: None)

        def out(action: str, extra: str = '') -> str:
            return ('import json, sys\nsys.stdin.read()\nprint(json.dumps({"hookSpecificOutput": '
                    '{"hookEventName": "ElicitationResult", "action": "%s"%s}}))\n' % (action, extra))

        self.assertEqual(answer('import sys\nsys.stdin.read()\n'), {'action': 'accept', 'content': {'name': 'Ada'}})
        self.assertEqual(answer(out('decline')), {'action': 'decline'})
        self.assertEqual(answer(out('cancel')), {'action': 'cancel', 'content': {'name': 'Ada'}})
        self.assertEqual(answer('import sys\nsys.stdin.read()\nsys.exit(2)\n'), {'action': 'decline'})
        self.assertEqual(answer(out('accept', ', "content": {"name": "Changed"}')),
                         {'action': 'accept', 'content': {'name': 'Changed'}})
        script = self.ws / 'mrtr.py'  # end to end: the MCP server receives the changed values
        script.write_text(MRTR_SERVER, encoding='utf-8')
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'shop': {
            'command': sys.executable, 'args': ['-u', str(script)]}}}), encoding='utf-8')
        with patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'}):
            runtime = MCPRuntime.from_workspace(self.ws)
            agent = self.agent()
            set_elicitation_handler(lambda server, params: agent.answer_elicitation(
                server, params, ask=lambda _l: 'Ada', output=lambda _t: None))
            text, _ = runtime.call_tool('book', arguments={}, server_name='shop')
        self.assertEqual(text.strip(), 'state=st-1 {"action": "accept", "content": {"name": "Changed"}}')

    def test_message_display_content_replaces_shown_text_only(self) -> None:
        import io
        from contextlib import redirect_stdout

        from src.session_store import load_agent_session
        from src.main import _run_agent_chat_loop

        record, out = self.hook_script('shown')
        self.settings({'MessageDisplay': [{'hooks': [{'type': 'command', 'command': record}]}, self.py_hook(
            'display', 'import json, sys\ndata = json.load(sys.stdin)\nprint(json.dumps({"hookSpecificOutput": '
                       '{"hookEventName": "MessageDisplay", "displayContent": '
                       'data["delta"].replace("secret", "[hidden]")}}))\n')]})
        self.replies = [{'role': 'assistant', 'content': 'the secret is 42'}]
        agent = self.agent()
        inputs = iter(['tell me'])

        def scripted(_label: str = '') -> str:
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        printed = io.StringIO()
        with redirect_stdout(printed):
            _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                                 input_func=scripted, output_func=lambda text: printed.write(text + '\n'))
        self.assertIn('the [hidden] is 42', printed.getvalue())
        self.assertNotIn('the secret is 42', printed.getvalue())
        event = json.loads(out.read_text(encoding='utf-8').splitlines()[-1])
        self.assertTrue(event['turn_id'] and event['message_id'])
        self.assertEqual((event['delta'], event['final'], event['index']), ('the secret is 42', True, 0))
        stored = load_agent_session(agent.active_session_id, directory=agent.runtime_config.session_directory)
        self.assertIn('the secret is 42', json.dumps(stored.messages, default=str))  # the transcript keeps it


class OtherTests(Base):
    def test_telemetry_endpoints(self) -> None:
        from src import telemetry

        with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT': 'http://c:9/exact',
                                     'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://c:4318'}):
            self.assertEqual(telemetry.endpoint(), 'http://c:9/exact')
        with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT': '', 'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://c:4318/'}):
            self.assertEqual(telemetry.endpoint(), 'http://c:4318/v1/logs')
        with patch.dict(os.environ, {'SIGIL_ENABLE_TELEMETRY': '1',
                                     'OTEL_EXPORTER_OTLP_ENDPOINT': '', 'OTEL_EXPORTER_OTLP_LOGS_ENDPOINT': ''}):
            self.assertTrue(telemetry.enabled())

    def test_plugin_agents_ignore_restricted_keys_and_dont_ask(self) -> None:
        from src.subagents import discover_subagents

        plugin = self.ws / 'plug'
        (plugin / '.sigil-plugin').mkdir(parents=True)
        (plugin / '.sigil-plugin' / 'plugin.json').write_text('{"name": "plug"}', encoding='utf-8')
        (plugin / 'agents').mkdir()
        (plugin / 'agents' / 'helper.md').write_text(
            '---\nname: helper\ndescription: h\npermissionMode: bypassPermissions\nmcpServers: [x]\n'
            'hooks:\n  Stop: []\n---\nHelp.\n', encoding='utf-8')
        with patch.dict(os.environ, {'SIGIL_PLUGIN_DIRS': str(plugin), 'SIGIL_SAFE_MODE': ''}):
            agent = discover_subagents(self.ws)['plug:helper']
        self.assertEqual((agent.permission_mode, agent.mcp_servers, agent.hooks), (None, (), None))
        agents = self.ws / '.sigil' / 'agents'
        agents.mkdir(parents=True)
        (agents / 'quiet.md').write_text('---\nname: quiet\ndescription: q\npermissionMode: dontAsk\n---\nGo.\n',
                                         encoding='utf-8')
        self.replies = [{'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'd', 'type': 'function', 'function': {
            'name': 'delegate_agent', 'arguments': json.dumps({'prompt': 'go', 'agent': 'quiet'})}}]},
            {'role': 'assistant', 'content': 'child'}, {'role': 'assistant', 'content': 'done'}]
        parent = self.agent()
        seen: list = []
        real_run = LocalCodingAgent.run

        def spy(self_, prompt, session_id=None):
            if self_.is_subagent:
                seen.append(self_.permission_mode)
            return real_run(self_, prompt, session_id=session_id)

        with patch.object(LocalCodingAgent, 'run', spy):
            parent.run('delegate')
        self.assertEqual(seen, ['dontAsk'])



if __name__ == '__main__':
    unittest.main()
