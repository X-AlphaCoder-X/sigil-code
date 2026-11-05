"""/loop <prompt> without an interval is self-paced: the agent picks each next run with schedule_wakeup
(60 s to 1 h) or ends the loop, a run that does neither gets one fallback run, loops end after 7 days, and
the prompts of a session are saved with it and restored on resume. None of it changes the tools or the
system prompt, so every request of a loop extends the previous one and keeps the prompt cache, on every
provider, with real MCP servers, and across a model switch and a resume."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig
from src.main import _run_agent_chat_loop
from src.mcp_runtime import connection_manager
from src.scheduler import (
    LOOP_FALLBACK_DELAY, MAX_AGE, Scheduler, prompt_text, restore_state, save_state, state_path,
)

BASE = 'http://127.0.0.1:8000/v1'
START = 1_800_000_000.0

MCP_SERVER = r'''
import json, sys
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": "2025-06-18",
              "capabilities": {"tools": {}}, "serverInfo": {"name": "ci"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
            {"name": "build_status", "description": "status of the build",
             "inputSchema": {"type": "object", "properties": {"branch": {"type": "string"}}}},
            {"name": "logs", "description": "the build log", "inputSchema": {"type": "object"}}]}})
    else:
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": "green"}]}})
'''


class _Resp:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode()

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


def scripted(*answers):
    queue = list(answers)

    def _input(_label: str = '') -> str:
        if not queue:
            raise EOFError
        item = queue.pop(0)
        return item() if callable(item) else item

    return _input


def _no_cache_marks(value):
    """The request without cache_control: Anthropic breakpoints move to the newest messages by design."""
    if isinstance(value, dict):
        return {k: _no_cache_marks(v) for k, v in value.items() if k != 'cache_control'}
    if isinstance(value, list):
        return [_no_cache_marks(v) for v in value]
    return value


class SchedulerRuleTests(unittest.TestCase):
    def test_the_agent_paces_the_loop_and_it_falls_back_once_then_ends(self) -> None:
        s = Scheduler()
        loop = s.start_loop('check the build', now=START)
        self.assertEqual(loop.runs, 1)
        delay, _ = s.wakeup(30, 'soon', now=START)               # clamped up to one minute
        self.assertEqual(delay, 60)
        delay, _ = s.wakeup(99999, 'later', now=START)           # and down to one hour; replaces the pending run
        self.assertEqual(delay, 3600)
        self.assertEqual(len(s.jobs), 1)
        self.assertIn('next run in 1h', s.end_loop_run(now=START))
        due = s.pop_due(now=START + 3600)
        self.assertEqual((due.source, s.loop.runs, s.loop.armed), ('loop', 2, False))
        note = s.end_loop_run(now=START + 3601)                  # this run picked nothing
        self.assertIn('one fallback run follows in 20m', note)
        self.assertIsNotNone(s.pop_due(now=START + 3601 + LOOP_FALLBACK_DELAY))
        self.assertIn('ended: the fallback run did not schedule another', s.end_loop_run(now=START + 5000))
        self.assertIsNone(s.loop)
        self.assertEqual(s.jobs, {})

    def test_stop_interrupt_and_the_seven_day_limit(self) -> None:
        s = Scheduler()
        s.start_loop('watch', now=START)
        self.assertTrue(s.stop_loop('the build passed'))
        self.assertIn('ended: the build passed', s.end_loop_run(now=START))
        s.start_loop('watch', now=START)
        self.assertIn('stopped: the run was interrupted', s.end_loop_run(interrupted=True, now=START))
        s.start_loop('watch', now=START)
        s.wakeup(600, 'x', now=START)
        self.assertIn('ended after 7 days', s.end_loop_run(now=START + MAX_AGE))
        self.assertEqual(s.jobs, {})
        job = s.add('1h', 'daily check', now=START)               # a recurring prompt: one final run, then gone
        self.assertFalse(s.pop_due(now=START + 3600).final)
        final = s.pop_due(now=START + MAX_AGE + 3600)
        self.assertTrue(final.final)
        self.assertNotIn(job.job_id, s.jobs)

    def test_runs_due_when_the_cache_expires_go_fifteen_seconds_early_for_the_current_model(self) -> None:
        lifetime = {'value': (300, '5 minutes')}
        s = Scheduler()
        s.cache_lifetime = lambda: lifetime['value']
        job = s.add('5m', 'poll', now=START)
        self.assertEqual(job.next_run, START + 285)
        s.pop_due(now=START + 285)
        self.assertEqual(job.next_run, START + 570)
        lifetime['value'] = (3600, '1 hour')                       # /model to a provider with a 1-hour cache
        s.pop_due(now=START + 570)
        self.assertEqual(job.next_run, START + 870)               # no longer moved: 5m is well inside 1h
        s.start_loop('watch', now=START)
        delay, note = s.wakeup(3600, 'idle', now=START)
        self.assertEqual(delay, 3585)
        self.assertIn('still reads the prompt cache', note)
        lifetime['value'] = (300, '5 minutes')
        delay, note = s.wakeup(900, 'idle', now=START)
        self.assertEqual(delay, 900)
        self.assertIn('reads the whole prompt again at full price', note)
        lifetime['value'] = (None, 'an unpublished time')
        self.assertIn('does not say exactly how long', s.wakeup(120, 'x', now=START)[1])
        s.cache_lifetime = lambda: 1 / 0                          # a failing price lookup never stops a run
        with mock.patch('src.scheduler._log_ignored') as logged:
            self.assertEqual(s.wakeup(120, 'x', now=START)[0], 120)
        self.assertIsInstance(logged.call_args.args[0], ZeroDivisionError)

    def test_saved_prompts_come_back_by_the_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = Scheduler()
            s.add('10m', 'recurring check', now=START)
            s.add('5m', 'one shot', recurring=False, now=START)
            s.add('*/15 * * * *', 'cron check', now=START, source='model')
            s.start_loop('self paced', now=START)
            s.wakeup(1800, 'idle', now=START)
            save_state(s, tmp, 'sess1')
            self.assertTrue(state_path(tmp, 'sess1').is_file())

            back = Scheduler()                                    # an hour later: the one-shot was missed
            notes = restore_state(back, tmp, 'sess1', now=START + 3600)
            self.assertIn('Restored 2 scheduled prompt(s)', notes[0])
            self.assertTrue(any('"5m one shot" was due while the chat was closed' in n for n in notes))
            self.assertTrue(any('self paced' in n and 'came due while the chat was closed' in n for n in notes))
            recurring = next(j for j in back.jobs.values() if j.prompt == 'recurring check')
            self.assertGreater(recurring.next_run, START + 3600)  # no catch-up: the first run after now
            self.assertLessEqual(recurring.next_run, START + 3600 + 600)
            model_job = next(j for j in back.jobs.values() if j.source == 'model')
            self.assertIn('set up earlier by the assistant', prompt_text(model_job))  # still labelled

            soon = Scheduler()                                    # ten minutes later: the loop is still due later
            notes = restore_state(soon, tmp, 'sess1', now=START + 600)
            self.assertIsNotNone(soon.loop)
            self.assertIn('Restored 3 scheduled prompt(s)', notes[0])   # the 5m one-shot was due at +300
            late = Scheduler()                                    # eight days later: only the cron prompt is gone
            notes = restore_state(late, tmp, 'sess1', now=START + MAX_AGE + 86400)
            self.assertTrue(any('older than 7 days' in n for n in notes))

            with mock.patch.dict(os.environ, {'SIGIL_LOOP_RESTORE': '0'}):
                self.assertIn('SIGIL_LOOP_RESTORE=0', restore_state(Scheduler(), tmp, 'sess1')[0])
            s.clear()
            save_state(s, tmp, 'sess1')                           # nothing left: the file goes
            self.assertFalse(state_path(tmp, 'sess1').exists())

    def test_a_tampered_or_hostile_saved_file_is_reported_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                state_path(tmp, '../../escape')
            self.assertIn('could not be read', restore_state(Scheduler(), tmp, '../x')[0])
            path = state_path(tmp, 'sess2')
            path.parent.mkdir(parents=True)
            path.write_text('{not json', encoding='utf-8')
            self.assertIn('could not be read (JSONDecodeError)', restore_state(Scheduler(), tmp, 'sess2')[0])
            path.write_text(json.dumps({'version': 1, 'loop': {'prompt': 'x' * 9000, 'next_run': START + 99},
                                        'jobs': [{'prompt': 'y', 'schedule': '5m', 'source': 'root'},
                                                 'not an object',
                                                 {'prompt': 'z', 'schedule': 'whenever', 'source': 'user'},
                                                 {'prompt': 'ok', 'schedule': '5m', 'source': 'user',
                                                  'created_at': 'soon'}]}), encoding='utf-8')
            s = Scheduler()
            notes = restore_state(s, tmp, 'sess2', now=START)
            self.assertEqual(s.jobs, {})
            self.assertIsNone(s.loop)
            self.assertEqual(len(notes), 5)
            path.write_text(' ' * 600_000, encoding='utf-8')
            self.assertIn('could not be read (ValueError)', restore_state(Scheduler(), tmp, 'sess2')[0])


class LoopInTheChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(connection_manager().close_all)
        self.ws = Path(self.tmp.name).resolve()
        script = self.ws / 'ci_server.py'
        script.write_text(MCP_SERVER, encoding='utf-8')
        (self.ws / '.mcp.json').write_text(json.dumps({'mcpServers': {'ci': {
            'command': sys.executable, 'args': ['-u', str(script)]}}}), encoding='utf-8')
        self.clock = SimpleNamespace(now=START)
        self.requests: list[dict] = []
        self.plan: list[dict | None] = []
        # one urllib.request.urlopen for both clients (they share the module): routed by URL, and nothing
        # else leaves the machine
        for patcher in (mock.patch('src.scheduler.time', SimpleNamespace(time=lambda: self.clock.now)),
                        mock.patch('urllib.request.urlopen', side_effect=self._route),
                        mock.patch.dict(os.environ, {'SIGIL_MCP_PERSISTENT': '1', 'SIGIL_DISABLE_CRON': ''})):
            patcher.start()
            self.addCleanup(patcher.stop)

    # ------------------------------------------------------------ fake model servers

    def _next_step(self, last: dict) -> tuple[str, dict | None]:
        text = last.get('content') if isinstance(last.get('content'), str) else json.dumps(last.get('content'))
        if last.get('role') == 'user' and 'Self-paced /loop' in (text or '') and 'tool_result' not in (text or ''):
            args = self.plan.pop(0) if self.plan else None
            return ('call', args) if args is not None else ('text', None)
        return 'text', None

    def _route(self, req, timeout=None):  # noqa: ANN001
        url = getattr(req, 'full_url', str(req))
        if url.endswith('/chat/completions'):
            return self._openai(req, timeout)
        if url.endswith('/v1/messages'):
            return self._anthropic(req, timeout)
        from urllib.error import URLError

        raise URLError(f'offline in this test: {url}')

    def _openai(self, req, timeout=None):  # noqa: ANN001
        body = json.loads(req.data.decode('utf-8'))
        self.requests.append(body)
        kind, args = self._next_step(body['messages'][-1])
        if kind == 'call':
            message = {'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': f'c{len(self.requests)}', 'type': 'function',
                'function': {'name': 'schedule_wakeup', 'arguments': json.dumps(args)}}]}
            return _Resp({'choices': [{'message': message, 'finish_reason': 'tool_calls'}]})
        return _Resp({'choices': [{'message': {'role': 'assistant', 'content': 'checked'}, 'finish_reason': 'stop'}]})

    def _anthropic(self, req, timeout=None):  # noqa: ANN001
        body = json.loads(req.data.decode('utf-8'))
        self.requests.append(body)
        kind, args = self._next_step(body['messages'][-1])
        content = ([{'type': 'tool_use', 'id': f'toolu_{len(self.requests)}', 'name': 'schedule_wakeup',
                     'input': args}] if kind == 'call' else [{'type': 'text', 'text': 'checked'}])
        return _Resp({'id': f'msg_{len(self.requests)}', 'type': 'message', 'role': 'assistant', 'model': 'claude-x',
                      'content': content, 'stop_reason': 'tool_use' if kind == 'call' else 'end_turn',
                      'usage': {'input_tokens': 10, 'output_tokens': 2}})

    # ------------------------------------------------------------ helpers

    def _agent(self, model: str = 'm', base_url: str = BASE, api_key: str = 'k') -> LocalCodingAgent:
        return LocalCodingAgent(
            model_config=ModelConfig(model=model, base_url=base_url, api_key=api_key),
            runtime_config=AgentRuntimeConfig(cwd=self.ws, session_directory=self.ws / '.s', nudge_strategy='off',
                                              permissions=AgentPermissions()))

    def _wait(self, seconds: float):
        def _step() -> str:
            self.clock.now += seconds
            return ''
        return _step

    def _chat(self, agent: LocalCodingAgent, *inputs, resume: str | None = None) -> list[str]:
        out: list[str] = []
        _run_agent_chat_loop(agent, initial_prompt=None, resume_session_id=resume, show_transcript=False,
                             input_func=scripted(*inputs), output_func=out.append)
        return out

    def _assert_cache_kept(self, requests: list[dict], anthropic: bool = False) -> None:
        """Same tools and system prompt in every request, and each request's messages start with all of
        the previous request's messages: the cached prefix only ever grows."""
        clean = [_no_cache_marks(r) for r in requests] if anthropic else requests
        for previous, current in zip(clean, clean[1:]):
            self.assertEqual(json.dumps(current['tools']), json.dumps(previous['tools']))
            if anthropic:
                self.assertEqual(json.dumps(current['system']), json.dumps(previous['system']))
            self.assertEqual(current['messages'][:len(previous['messages'])], previous['messages'])

    # ------------------------------------------------------------ the tests

    def test_a_self_paced_loop_with_mcp_servers_keeps_the_cache_through_every_run(self) -> None:
        self.plan = [{'delay_seconds': 120, 'reason': 'build \x1b[31mstill\x1b]0;owned\x07 running'},
                     {'stop': True, 'reason': 'the build passed'}]
        agent = self._agent()
        out = self._chat(agent, '/loop check the build', self._wait(130), '/loop', '/exit')
        self.assertFalse(any('\x1b[31m' in line or '\x1b]0;' in line for line in out))  # the reason is plain text
        names = [t['function']['name'] for t in self.requests[0]['tools']]
        self.assertIn('mcp__ci__build_status', names)            # the real MCP server's tools are in every request
        self.assertIn('schedule_wakeup', names)
        runs = [m['content'] for r in self.requests for m in r['messages'][-1:] if m.get('role') == 'user']
        self.assertTrue(runs[0].startswith('[Self-paced /loop loop_') and 'run 1' in runs[0])
        self.assertIn('run 2', runs[1])
        self.assertTrue(any('next run in 2m' in line and 'build still running' in line for line in out), out)
        self.assertTrue(any('ended: the build passed' in line for line in out), out)
        self.assertIn('No scheduled prompts.', out)
        self.assertEqual(len(self.requests), 4)
        self._assert_cache_kept(self.requests)

    def test_the_loop_keeps_running_across_a_model_switch_and_follows_its_cache(self) -> None:
        self.plan = [{'delay_seconds': 300, 'reason': 'waiting'}, {'delay_seconds': 300, 'reason': 'waiting'},
                     {'stop': True}]
        agent = self._agent('gpt-5-mini', 'https://api.openai.com/v1')     # a 5-10 minute cache

        def switch() -> str:
            agent.set_model('gpt-5.5')                                     # 24 hours with extended retention
            self.clock.now += 290
            return ''

        # run 1 on gpt-5-mini, then the switch, then runs 2 and 3 on gpt-5.5
        out = self._chat(agent, '/loop poll the queue', switch, self._wait(400), '/exit')
        tool_results = [m['content'] for m in self.requests[-1]['messages'] if m.get('role') == 'tool']
        self.assertIn('Moved to 4m45s so the next run still reads the prompt cache', tool_results[0])
        self.assertIn('The next run reads the prompt cache (it lasts up to 24 hours', tool_results[1])
        self.assertIn('The loop is stopped', tool_results[2])
        self.assertEqual([r['model'] for r in self.requests][-1], 'gpt-5.5')
        self.assertEqual(sum('Self-paced /loop' in (r['messages'][-1].get('content') or '') for r in self.requests), 3)
        self.assertTrue(any('ended' in line for line in out), out)
        # the model changed, so its cache did too; up to the switch and after it, each prefix only grows
        switch_at = next(i for i, r in enumerate(self.requests) if r['model'] == 'gpt-5.5')
        self._assert_cache_kept(self.requests[:switch_at])
        self._assert_cache_kept(self.requests[switch_at:])

    def test_the_native_anthropic_client_keeps_its_cache_through_the_loop(self) -> None:
        self.plan = [{'delay_seconds': 600, 'reason': 'deploy in progress'}, {'stop': True, 'reason': 'deployed'}]
        agent = self._agent('claude-x', 'https://api.anthropic.com', 'test-key')
        out = self._chat(agent, '/loop watch the deploy', self._wait(700), '/exit')
        self.assertTrue(all('system' in r and 'tools' in r for r in self.requests))
        self.assertIn('schedule_wakeup', [t['name'] for t in self.requests[0]['tools']])
        self.assertTrue(any('ended: deployed' in line for line in out), out)
        self.assertEqual(len(self.requests), 4)
        self._assert_cache_kept(self.requests, anthropic=True)

    def test_a_resumed_session_brings_its_loop_back_with_the_same_cached_prefix(self) -> None:
        self.plan = [{'delay_seconds': 900, 'reason': 'nightly build'}]
        first = self._agent()
        self._chat(first, '/loop check the nightly build', '/exit')
        session_id = first.active_session_id or json.loads(
            next((self.ws / '.s').glob('*.json')).read_text(encoding='utf-8'))['session_id']
        self.assertTrue(state_path(self.ws / '.s', session_id).is_file())
        before = list(self.requests)

        self.plan = [{'stop': True, 'reason': 'done'}]
        self.clock.now += 60
        out = self._chat(self._agent(), self._wait(900), '/exit', resume=session_id)
        self.assertTrue(any('Restored 1 scheduled prompt(s)' in line for line in out), out)
        after = self.requests[len(before):]
        self.assertIn('run 2', after[0]['messages'][-1]['content'])
        self.assertTrue(any('ended: done' in line for line in out), out)
        self._assert_cache_kept([before[-1]] + after)          # the resumed run extends the saved conversation
        self.assertFalse(state_path(self.ws / '.s', session_id).exists())   # ended: nothing left to restore

    def test_the_tool_list_is_fixed_from_the_start_on_every_provider(self) -> None:
        from src.anthropic_client import AnthropicClient
        from src.bedrock_client import BedrockConverseClient
        from src.openai_compat import OpenAICompatClient

        self.plan = [{'delay_seconds': 120, 'reason': 'x'}, {'stop': True}]
        agent = self._agent()
        snapshots: list[list[dict]] = []

        def snap() -> str:
            snapshots.append([tool.to_openai_tool() for tool in agent.tool_registry.values()])
            return ''

        self._chat(agent, snap, '/loop check', snap, self._wait(130), snap, '/loop 5m poll', snap,
                   '/loop stop', snap, '/exit')
        clients = [OpenAICompatClient(ModelConfig(model='gpt-5.6', base_url='https://api.openai.com/v1')),
                   OpenAICompatClient(ModelConfig(model='gemini-3.6-pro',
                                                  base_url='https://generativelanguage.googleapis.com/v1beta/openai')),
                   OpenAICompatClient(ModelConfig(model='deepseek-chat', base_url='https://api.deepseek.com/v1')),
                   AnthropicClient(ModelConfig(model='claude-x', base_url='https://api.anthropic.com', api_key='k')),
                   BedrockConverseClient(ModelConfig(model='us.anthropic.claude-sonnet-5-v1:0', api_key='local-token',
                                                     base_url='https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1'))]
        messages = [{'role': 'system', 'content': 'You are a test agent. ' * 200}, {'role': 'user', 'content': 'hi'}]
        for client in clients:
            bodies = {json.dumps(client._build_payload(messages=messages, tools=tools, stream=False, output_schema=None)
                                 if isinstance(client, OpenAICompatClient) else client.build_payload(messages, tools))
                      for tools in snapshots}
            self.assertEqual(len(bodies), 1, type(client).__name__)

    def test_scheduling_can_be_turned_off(self) -> None:
        with mock.patch.dict(os.environ, {'SIGIL_DISABLE_CRON': '1'}):
            agent = self._agent()
            out = self._chat(agent, '/loop check', '/exit')
        self.assertNotIn('schedule_wakeup', agent.tool_registry)
        self.assertNotIn('cron_create', agent.tool_registry)
        self.assertIn('Scheduled prompts are off (SIGIL_DISABLE_CRON=1).', out)
        self.assertEqual(self.requests, [])


if __name__ == '__main__':
    unittest.main()
