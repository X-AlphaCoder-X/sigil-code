"""Batch 16b (P3): agent frontmatter skills / mcpServers / hooks / memory / effort and
maxBudgetUsd, /subtask, mcp_tool and agent hook types, the new hook events, MCP prompts as
slash commands, MCP elicitation, /loop, /goal and the cron tools."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.main import _run_agent_chat_loop
from src.mcp_runtime import connection_manager, set_elicitation_handler
from src.scheduler import Scheduler, parse_cron, parse_interval
from src.subagents import discover_subagents

import urllib.request  # noqa: E402

REAL_URLOPEN = urllib.request.urlopen
PY = sys.executable.replace('\\', '/')
BASE = 'http://127.0.0.1:8000/v1'

MCP_SERVER = r'''
import json, sys
def send(obj):
    print(json.dumps(obj), flush=True)
for raw in sys.stdin:
    msg = json.loads(raw)
    method = msg.get("method")
    if method == "notifications/initialized" or "method" not in msg:
        continue
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2025-06-18",
              "capabilities": {"tools": {}, "prompts": {}}, "serverInfo": {"name": "p"},
              "client_caps": msg["params"]["capabilities"]}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
            {"name": "ask", "description": "asks the user", "inputSchema": {"type": "object"}},
            {"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]}})
    elif method == "prompts/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"prompts": [{"name": "review",
              "description": "Review a file", "arguments": [{"name": "file", "required": True},
                                                             {"name": "focus"}]}]}})
    elif method == "prompts/get":
        args = msg["params"]["arguments"]
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"messages": [{"role": "user", "content": {
              "type": "text", "text": "Review " + args["file"] + " for " + args.get("focus", "bugs")}}]}})
    elif method == "tools/call" and msg["params"]["name"] == "ask":
        # a server request that reuses the client's own id: it must not be taken as the reply
        send({"jsonrpc": "2.0", "id": msg["id"], "method": "elicitation/create", "params": {
              "message": "Who are you?", "requestedSchema": {"type": "object", "properties": {
                  "name": {"type": "string", "title": "Your name"}, "age": {"type": "integer"}}}}})
        reply = json.loads(sys.stdin.readline())
        result = reply.get("result", {})
        text = result.get("action", "?") + ":" + json.dumps(result.get("content", {}), sort_keys=True)
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": text}]}})
    elif method == "tools/call":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text",
              "text": "echo:" + json.dumps(msg["params"]["arguments"], sort_keys=True)}]}})
    else:
        send({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "nope"}})
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


def call(name: str, call_id: str = 'c1', **arguments) -> dict:
    return {'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def text(content: str) -> dict:
    return {'role': 'assistant', 'content': content}


def scripted(*answers):
    queue = list(answers)

    def _input(_label: str = '') -> str:
        if not queue:
            raise EOFError
        item = queue.pop(0)
        return item() if callable(item) else item

    return _input


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(connection_manager().close_all)
        self.addCleanup(set_elicitation_handler, None)
        self.ws = Path(self.tmp.name).resolve()
        self.requests: list[dict] = []
        self.route = lambda body: text('answer')

        def fake(req, timeout=None):  # noqa: ANN001
            if not req.full_url.endswith('/chat/completions'):
                return REAL_URLOPEN(req, timeout=timeout)  # MCP over HTTP is real; only the model is faked
            body = json.loads(req.data.decode('utf-8'))
            self.requests.append(body)
            return _Resp(self.route(body))

        for patcher in (patch('src.openai_compat.request.urlopen', side_effect=fake),
                        patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1'})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def agent(self, **perms) -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model='m', base_url=BASE),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s', nudge_strategy='off',
                                              permissions=AgentPermissions(**perms)))

    def queue(self, *replies: dict) -> None:
        pending = list(replies)
        self.route = lambda body: pending.pop(0) if pending else text('answer')

    def mcp_server(self) -> None:
        script = self.ws / 'server.py'
        script.write_text(MCP_SERVER, encoding='utf-8')
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'fake': {
            'command': sys.executable, 'args': ['-u', str(script)]}}}), encoding='utf-8')

    def settings(self, payload: dict) -> None:
        (self.ws / '.sigil').mkdir(exist_ok=True)
        (self.ws / '.sigil' / 'settings.json').write_text(json.dumps(payload), encoding='utf-8')


class FrontmatterTests(Base):
    def test_skills_mcp_servers_memory_effort_budget_and_hooks(self) -> None:
        self.mcp_server()
        skill = self.ws / '.sigil' / 'skills' / 'style'
        skill.mkdir(parents=True)
        (skill / 'SKILL.md').write_text('---\nname: style\ndescription: House style\n---\nALWAYS USE TABS.\n',
                                        encoding='utf-8')
        marker = self.ws / 'hook.json'
        hook_script = self.ws / 'hook.py'
        hook_script.write_text('import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(sys.stdin.read())\n',
                               encoding='utf-8')
        agents = self.ws / '.sigil' / 'agents'
        agents.mkdir(parents=True)
        (agents / 'helper.md').write_text('\n'.join([
            '---', 'name: helper', 'description: Helps', 'skills: [style]', 'mcpServers: [other]',
            'memory: project', 'effort: high', 'maxBudgetUsd: 0.5', 'hooks:', '  PreToolUse:',
            '    - hooks:', '        - type: command',
            f'          command: \'"{PY}" "{hook_script.as_posix()}" "{marker.as_posix()}"\'',
            '---', 'You help.', '']), encoding='utf-8')
        definition = discover_subagents(self.ws)['helper']
        self.assertEqual((definition.skills, definition.mcp_servers, definition.memory, definition.effort,
                          definition.max_budget_usd), (('style',), ('other',), 'project', 'high', 0.5))

        children: list[LocalCodingAgent] = []
        original = LocalCodingAgent.__post_init__

        def record(agent_self):
            original(agent_self)
            if agent_self.is_subagent:
                children.append(agent_self)

        def route(body):
            system = body['messages'][0]['content']
            if 'You help.' in system:
                done = any(m.get('role') == 'tool' for m in body['messages'])
                return text('child done') if done else call('memory_save', 'm1', note='prefer tabs')
            done = any(m.get('role') == 'tool' for m in body['messages'])
            return text('parent done') if done else call('delegate_agent', 'd1', prompt='do it', agent='helper')

        self.route = route
        with patch.object(LocalCodingAgent, '__post_init__', record):
            parent = self.agent()
            self.assertIn('mcp__fake__echo', parent.tool_registry)
            parent.run('delegate please')
        child = children[0]
        self.assertEqual(child.model_config.effort, 'high')
        self.assertEqual(child.runtime_config.budget_config.max_total_cost_usd, 0.5)
        child_request = next(r for r in self.requests if 'You help.' in r['messages'][0]['content'])
        self.assertIn('ALWAYS USE TABS.', child_request['messages'][0]['content'])
        names = {t['function']['name'] for t in child_request['tools']}
        self.assertIn('memory_save', names)
        self.assertNotIn('mcp__fake__echo', names)  # only the "other" server's tools
        memory = self.ws / '.sigil' / 'agent-memory' / 'helper' / 'MEMORY.md'
        self.assertIn('- prefer tabs', memory.read_text(encoding='utf-8'))
        self.assertEqual(json.loads(marker.read_text())['tool_name'], 'memory_save')  # the agent's own hook
        self.assertIsNone(parent.command_hook_runtime is not None and parent.command_hook_runtime.has('PreToolUse')
                          or None)  # the parent has no such hook


class SubtaskTests(Base):
    def test_subtask_sees_the_conversation(self) -> None:
        agent = self.agent()
        self.queue(text('noted'), text('The colour is teal.'), text('ok'))
        agent.run('my favourite colour is teal')
        report = agent.run('/subtask what is my favourite colour?').final_output
        self.assertIn('The colour is teal.', report)
        child = self.requests[1]
        self.assertIn('my favourite colour is teal', json.dumps(child['messages']))
        self.assertIn('what is my favourite colour?', json.dumps(child['messages']))
        from src.session_store import load_agent_session

        agent.resume('thanks', load_agent_session(agent.active_session_id, directory=self.ws / '.s'))
        self.assertIn('Result of /subtask', json.dumps(self.requests[-1]['messages']))


class HookTypeTests(Base):
    def test_mcp_tool_and_agent_hooks_and_file_events(self) -> None:
        self.mcp_server()
        marker = self.ws / 'changed.json'
        hook_script = self.ws / 'hook.py'
        hook_script.write_text('import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(sys.stdin.read())\n',
                               encoding='utf-8')
        self.settings({'hooks': {
            'UserPromptSubmit': [{'hooks': [{'type': 'mcp_tool', 'tool': 'mcp__fake__echo', 'input': {'k': 'v'}}]}],
            'PreToolUse': [{'matcher': 'write_file', 'hooks': [{'type': 'agent',
                                                                 'prompt': 'Block writes to secret.txt'}]}],
            'FileChanged': [{'hooks': [{'type': 'command',
                                        'command': f'"{PY}" "{hook_script.as_posix()}" "{marker.as_posix()}"'}]}],
        }})

        def route(body):
            system = body['messages'][0]['content']
            if 'You are a hook' in system:
                blocked = 'ok.txt' not in json.dumps(body['messages'])  # the hook input names the path
                return text('{"ok": false, "reason": "secret"}' if blocked else '{"ok": true}')
            done = any(m.get('role') == 'tool' for m in body['messages'])
            if done:
                return text('done')
            users = [str(m.get('content')) for m in body['messages'] if m.get('role') == 'user']
            target = 'secret.txt' if any('write secret' in u for u in users) else 'ok.txt'
            return call('write_file', 'w1', path=target, content='x')

        self.route = route
        agent = self.agent(allow_file_write=True)
        agent.run('write ok')
        main_request = next(r for r in self.requests if 'You are a hook' not in r['messages'][0]['content'])
        self.assertIn('echo:', json.dumps(main_request['messages']))  # mcp_tool hook output as context
        self.assertIn('"k": "v"', json.dumps(main_request['messages']).replace('\\"', '"'))
        self.assertTrue((self.ws / 'ok.txt').exists())
        self.assertEqual(json.loads(marker.read_text())['hook_event_name'], 'FileChanged')
        self.assertTrue(json.loads(marker.read_text())['file_path'].endswith('ok.txt'))
        agent.run('write secret')
        self.assertFalse((self.ws / 'secret.txt').exists())  # the agent hook blocked it

    def test_directory_worktree_and_message_events(self) -> None:
        fired: list[tuple[str, dict]] = []
        agent = self.agent()
        agent._run_hook_event = lambda event, payload, **kw: fired.append((event, payload)) or None
        extra = self.ws / 'extra'
        extra.mkdir()
        agent.run(f'/add-dir {extra}')
        self.assertIn('DirectoryAdded', [e for e, _ in fired])
        repo = self.ws / 'repo'
        repo.mkdir()
        for args in (['init', '-q'], ['-c', 'user.email=a@b', '-c', 'user.name=a', 'commit', '-q', '--allow-empty',
                                      '-m', 'init']):
            subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)
        agent.runtime_config = agent.runtime_config.__class__(**{**agent.runtime_config.__dict__, 'cwd': repo})
        path = agent._create_isolated_worktree('job')
        agent._finish_isolated_worktree(path)
        events = [e for e, _ in fired]
        self.assertIn('WorktreeCreate', events)
        self.assertNotIn('WorktreeRemove', events)  # docs: only for worktrees a WorktreeCreate hook made
        output: list[str] = []
        _run_agent_chat_loop(agent, initial_prompt='hello', resume_session_id=None, show_transcript=False,
                             input_func=scripted(), output_func=output.append)
        self.assertEqual([p['message'] for e, p in fired if e == 'MessageDisplay'], ['answer'])


class McpPromptAndElicitationTests(Base):
    def test_prompt_slash_command(self) -> None:
        self.mcp_server()
        agent = self.agent()
        agent.run('/mcp__fake__review app.py focus=security')
        self.assertIn('Review app.py for security', json.dumps(self.requests[-1]['messages']))
        self.assertIn('needs: file', agent.run('/mcp__fake__review').final_output)

    def test_elicitation_answered_declined_and_hook(self) -> None:
        self.mcp_server()
        agent = self.agent()
        runtime = agent.mcp_runtime
        answers = iter(['Ada', '36'])
        set_elicitation_handler(lambda server, params: agent.answer_elicitation(
            server, params, ask=lambda _label: next(answers), output=lambda _t: None))
        result, _ = runtime.call_tool('ask', arguments={}, server_name='fake')
        self.assertEqual(result.strip(), 'accept:{"age": 36, "name": "Ada"}')
        set_elicitation_handler(None)
        result, _ = runtime.call_tool('ask', arguments={}, server_name='fake')
        self.assertEqual(result.strip(), 'decline:{}')
        agent._run_hook_event = lambda event, payload, **kw: type('O', (), {'blocked': True, 'stop': False})()
        self.assertEqual(agent.answer_elicitation('fake', {'message': 'x'}, ask=lambda _l: 'y'), {'action': 'decline'})


class HttpElicitationTests(Base):
    def test_streamable_http_server_asks_mid_call(self) -> None:
        import queue
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        answers: queue.Queue = queue.Queue()
        seen_caps: list = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
                if 'method' not in body:  # our answer to the elicitation
                    answers.put(body)
                    self.send_response(202)
                    self.end_headers()
                    return
                if 'id' not in body:
                    self.send_response(202)
                    self.end_headers()
                    return
                method = body['method']
                if method == 'initialize':
                    seen_caps.append(body['params']['capabilities'])
                    result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}}, 'serverInfo': {}}
                elif method == 'tools/list':
                    result = {'tools': [{'name': 'ask', 'description': 'ask', 'inputSchema': {'type': 'object'}}]}
                else:
                    result = None
                if result is not None:
                    data = json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': result}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                ask = {'jsonrpc': '2.0', 'id': 'el-1', 'method': 'elicitation/create',
                       'params': {'message': 'Name?', 'requestedSchema': {'type': 'object', 'properties': {
                           'name': {'type': 'string'}}}}}
                self.wfile.write(f'event: message\ndata: {json.dumps(ask)}\n\n'.encode())
                self.wfile.flush()
                reply = answers.get(timeout=10)
                content = reply.get('result', {}).get('content', {})
                done = {'jsonrpc': '2.0', 'id': body['id'], 'result': {'content': [
                    {'type': 'text', 'text': f"{reply['id']}:{content.get('name')}"}]}}
                self.wfile.write(f'event: message\ndata: {json.dumps(done)}\n\n'.encode())
                self.wfile.flush()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'web': {
            'type': 'http', 'url': f'http://127.0.0.1:{server.server_address[1]}/mcp'}}}), encoding='utf-8')
        agent = self.agent()
        set_elicitation_handler(lambda name, params: agent.answer_elicitation(
            name, params, ask=lambda _label: 'Grace', output=lambda _t: None))
        result, _ = agent.mcp_runtime.call_tool('ask', arguments={}, server_name='web')
        self.assertEqual(result.strip(), 'el-1:Grace')
        self.assertEqual(seen_caps[0], {'elicitation': {}})


class ScheduleTests(Base):
    def test_intervals_and_cron(self) -> None:
        self.assertEqual(parse_interval('5m'), 300)
        self.assertEqual(parse_interval('every 2 hours'), 7200)
        self.assertIsNone(parse_interval('soon'))
        self.assertIsNotNone(parse_cron('*/15 9-17 * * 1-5'))
        self.assertIsNone(parse_cron('61 * * * *'))
        scheduler = Scheduler()
        job = scheduler.add('10s', 'check', now=1000.0)
        once = scheduler.add('5s', 'once', recurring=False, now=1000.0)
        self.assertIsNone(scheduler.pop_due(now=1004.0))
        self.assertEqual(scheduler.pop_due(now=1006.0).job_id, once.job_id)
        self.assertNotIn(once.job_id, scheduler.jobs)
        self.assertEqual(scheduler.pop_due(now=1011.0).job_id, job.job_id)
        self.assertEqual(scheduler.jobs[job.job_id].next_run, 1021.0)  # recurring: moved on
        with self.assertRaises(ValueError):
            scheduler.add('whenever', 'x')

    def test_loop_goal_and_cron_tools_in_the_chat(self) -> None:
        agent = self.agent()
        output: list[str] = []
        _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=None, show_transcript=False,
                             input_func=scripted('/loop 1s say tick', lambda: (time.sleep(1.2), '')[1],
                                                 '/loop stop'),
                             output_func=output.append)
        self.assertTrue(any(m.get('content') == 'say tick' for r in self.requests for m in r['messages']))
        self.assertIn('Stopped all.', output)
        self.assertIn('cron_create', agent.tool_registry)

        self.requests.clear()
        self.queue(text('step one'), text('all done\nGOAL MET'))
        output.clear()
        _run_agent_chat_loop(self.agent(), initial_prompt=None, resume_session_id=None, show_transcript=False,
                             input_func=scripted('/goal make the tests pass'), output_func=output.append)
        self.assertEqual(len(self.requests), 2)
        last_user = [m for m in self.requests[1]['messages'] if m.get('role') == 'user'][-1]['content']
        self.assertIn('Keep working towards the goal: make the tests pass', last_user)
        self.assertTrue(any('Goal met' in line for line in output))

        self.requests.clear()
        self.queue(call('cron_create', 'k1', schedule='5m', prompt='check CI'), text('scheduled'))
        agent = self.agent()
        _run_agent_chat_loop(agent, initial_prompt='watch CI', resume_session_id=None, show_transcript=False,
                             input_func=scripted(), output_func=lambda _t: None)
        self.assertEqual([job.prompt for job in agent.scheduler.jobs.values()], ['check CI'])


if __name__ == '__main__':
    unittest.main()
