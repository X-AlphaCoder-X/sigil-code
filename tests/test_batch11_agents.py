"""Batch 11 (P2): parallel delegate_agent, --agent/--agents, isolation: worktree, resuming a
subagent by name, /cost cache tokens, showing thinking, reminders as user text, schema warning."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent_runtime import LocalCodingAgent
from src.agent_types import AgentPermissions, AgentRuntimeConfig, ModelConfig, UsageStats
from src.main import build_parser
from src.openai_compat import OpenAICompatClient


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode('utf-8')

    def __enter__(self) -> 'FakeHTTPResponse':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def text_reply(text: str) -> dict[str, object]:
    return {'choices': [{'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}]}


def tool_calls(calls: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
    return {'choices': [{
        'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': f'call_{i}', 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}} for i, (name, args) in enumerate(calls)]},
        'finish_reason': 'tool_calls'}]}


def make_agent(ws: Path, **perms: bool) -> LocalCodingAgent:
    return LocalCodingAgent(
        model_config=ModelConfig(model='m'),
        runtime_config=AgentRuntimeConfig(
            cwd=ws, permissions=AgentPermissions(**perms),
            session_directory=ws / '.port_sessions' / 'agent',
            scratchpad_root=ws / '.port_sessions' / 'scratchpad', nudge_strategy='off'),
    )


def routing_server(route, recorded: list[dict[str, object]]):
    """Replies by looking at the request (children run in parallel, so order is not fixed)."""
    lock = threading.Lock()

    def _fake(request_obj, timeout=None):  # noqa: ANN001
        body = json.loads(request_obj.data.decode('utf-8'))
        with lock:
            recorded.append(body)
        return FakeHTTPResponse(route(body))

    return _fake


class ParallelDelegateTests(unittest.TestCase):
    def test_two_delegates_run_together(self) -> None:
        recorded: list[dict[str, object]] = []
        active = {'now': 0, 'peak': 0}
        lock = threading.Lock()

        def route(body: dict) -> dict:
            users = [m.get('content') for m in body['messages'] if m.get('role') == 'user']
            text = json.dumps(users)
            if 'task A' in text or 'task B' in text:
                if not any(m.get('role') == 'tool' for m in body['messages']):
                    with lock:
                        active['now'] += 1
                        active['peak'] = max(active['peak'], active['now'])
                    time.sleep(0.5)
                    with lock:
                        active['now'] -= 1
                    return text_reply('child finished ' + ('A' if 'task A' in text else 'B'))
            if any(m.get('role') == 'tool' for m in body['messages']):
                return text_reply('parent done')
            return tool_calls([('delegate_agent', {'prompt': 'task A', 'label': 'a'}),
                               ('delegate_agent', {'prompt': 'task B', 'label': 'b'})])

        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=routing_server(route, recorded)):
            result = make_agent(Path(tmp)).run('split the work')
        self.assertEqual(result.final_output, 'parent done')
        self.assertEqual(active['peak'], 2)


class SubagentFeatureTests(unittest.TestCase):
    def test_resume_subagent_by_name(self) -> None:
        recorded: list[dict[str, object]] = []
        replies = iter([
            tool_calls([('delegate_agent', {'prompt': 'remember 42', 'agent': 'Explore'})]),
            text_reply('noted 42'),
            tool_calls([('delegate_agent', {'prompt': 'what number?', 'resume_agent': 'Explore'})]),
            text_reply('it was 42'),
            text_reply('parent done'),
        ])
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen',
                      side_effect=routing_server(lambda _b: next(replies), recorded)):
            make_agent(Path(tmp)).run('use a helper twice')
        resumed_child = recorded[3]
        self.assertIn('remember 42', json.dumps(resumed_child['messages']))

    def test_isolation_worktree(self) -> None:
        recorded: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for command in (['git', 'init', '-q'], ['git', 'config', 'user.email', 't@t'],
                            ['git', 'config', 'user.name', 't'], ['git', 'commit', '--allow-empty', '-qm', 'i']):
                subprocess.run(command, cwd=repo, capture_output=True, check=True)
            replies = iter([
                tool_calls([('delegate_agent', {'prompt': 'write', 'isolation': 'worktree', 'allow_write': True,
                                                'label': 'writer'})]),
                tool_calls([('write_file', {'path': 'new.txt', 'content': 'from child'})]),
                text_reply('child wrote'),
                tool_calls([('delegate_agent', {'prompt': 'look only', 'isolation': 'worktree', 'label': 'reader'})]),
                text_reply('child looked'),
                text_reply('parent done'),
            ])
            with patch('src.openai_compat.request.urlopen',
                       side_effect=routing_server(lambda _b: next(replies), recorded)):
                make_agent(repo, allow_file_write=True).run('try isolation')
            self.assertFalse((repo / 'new.txt').exists())  # the child wrote in its own worktree
            # The writer's worktree is kept for review; the reader changed nothing, so its is removed.
            worktrees = [p for p in (repo / '.port_sessions' / 'worktrees').iterdir() if p.is_dir()]
            self.assertEqual(len(worktrees), 1)
            self.assertTrue((worktrees[0] / 'new.txt').exists())
            tool_text = json.dumps(recorded[3]['messages'])
            self.assertIn('has changes to review', tool_text)
            for path in worktrees:
                subprocess.run(['git', 'worktree', 'remove', '--force', str(path)], cwd=repo, capture_output=True)

    def test_session_agent_flags(self) -> None:
        from src.main import _build_agent

        definitions = json.dumps({'reviewer': {'description': 'Reviews', 'prompt': 'You only review.',
                                               'tools': ['Read', 'Grep']}})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}):
            agent = _build_agent(build_parser().parse_args(
                ['agent', 'x', '--cwd', tmp, '--agents', definitions, '--agent', 'reviewer']))
            explore = _build_agent(build_parser().parse_args(['agent', 'x', '--cwd', tmp, '--agent', 'Explore']))
        self.assertEqual(set(agent.tool_registry), {'read_file', 'grep_search'})
        self.assertIn('You only review.', agent.append_system_prompt)
        self.assertEqual(explore.permission_mode, 'plan')


class AnthropicExtraTests(unittest.TestCase):
    def test_cost_shows_cache_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(Path(tmp))
            agent.cumulative_usage = UsageStats(input_tokens=100, output_tokens=5, cache_read_input_tokens=80,
                                                cache_creation_input_tokens=20)
            output = agent.run('/cost').final_output
        self.assertIn('Cache read tokens:     80', output)
        self.assertIn('Cache creation tokens:  20', output)

    def test_thinking_is_shown(self) -> None:
        reply = {'choices': [{'message': {'role': 'assistant', 'content': '<think>step one</think>Answer'},
                              'finish_reason': 'stop'}]}
        events: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as tmp, \
                patch('src.openai_compat.request.urlopen', side_effect=lambda *_a, **_k: FakeHTTPResponse(reply)):
            agent = make_agent(Path(tmp))
            agent.event_callback = events.append
            agent.run('think')
            agent.event_callback = None
            out = io.StringIO()
            with patch.dict(os.environ, {'SIGIL_SHOW_THINKING': '1'}), patch('sys.stdout', out):
                agent.run('again')
        self.assertTrue(any(e.get('type') == 'reasoning' and e.get('content') == 'step one' for e in events))
        self.assertIn('[thinking] step one', out.getvalue())

    def test_reminders_become_user_text_for_anthropic(self) -> None:
        messages = [{'role': 'system', 'content': 'base'}, {'role': 'user', 'content': 'hi'},
                    {'role': 'system', 'content': 'mid-run note'}]
        anthropic = OpenAICompatClient(ModelConfig(model='claude-x', base_url='https://api.anthropic.com/v1'))
        local = OpenAICompatClient(ModelConfig(model='m'))
        with patch.dict(os.environ, {'SIGIL_REMINDERS_AS_USER': ''}):
            converted = anthropic._build_payload(messages=messages, tools=[], stream=False, output_schema=None)
            unchanged = local._build_payload(messages=messages, tools=[], stream=False, output_schema=None)
        self.assertEqual([m['role'] for m in converted['messages']], ['system', 'user', 'user'])
        self.assertIn('<system-reminder>', converted['messages'][2]['content'])
        self.assertEqual([m['role'] for m in unchanged['messages']], ['system', 'user', 'system'])

    def test_schema_warning_on_compat(self) -> None:
        from src.main import _build_agent

        with tempfile.TemporaryDirectory() as tmp:
            schema = Path(tmp) / 's.json'
            schema.write_text(json.dumps({'type': 'object'}), encoding='utf-8')
            err = io.StringIO()
            with patch('sys.stderr', err), patch.dict(os.environ, {'SIGIL_ANTHROPIC_NATIVE': ''}):
                _build_agent(build_parser().parse_args([
                    'agent', 'x', '--cwd', tmp, '--base-url', 'https://api.anthropic.com/v1',
                    '--model', 'claude-x', '--response-schema-file', str(schema)]))
        self.assertIn('ignored by the Anthropic OpenAI-compatible endpoint', err.getvalue())


if __name__ == '__main__':
    unittest.main()
