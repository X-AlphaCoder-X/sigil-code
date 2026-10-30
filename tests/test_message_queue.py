from __future__ import annotations

import unittest

from src.agent_session import AgentSessionState
from src.message_queue import (
    QUEUED_MESSAGE_KIND,
    ConsoleInputReader,
    MessageQueue,
    format_queued_messages,
)


class MessageQueueTests(unittest.TestCase):
    def test_reader_builds_lines_with_backspace_and_escape(self) -> None:
        notified: list[str] = []
        queue = MessageQueue(notify=notified.append)
        reader = ConsoleInputReader(queue)
        for ch in 'helo\bl\x7fo there\r':
            reader.feed(ch)
        for ch in 'junk\x1bok\n':
            reader.feed(ch)
        for ch in '   \r':  # blank lines are ignored
            reader.feed(ch)
        for ch in 'partial':
            reader.feed(ch)
        self.assertEqual(queue.drain(), ['helo there', 'ok'])
        self.assertEqual(notified, ['helo there', 'ok'])
        self.assertEqual(queue.drain(), [])
        self.assertEqual(reader.stop(), 'partial')

    def test_agent_injects_queued_messages_into_session(self) -> None:
        from src.agent_runtime import LocalCodingAgent

        agent = LocalCodingAgent.__new__(LocalCodingAgent)
        agent.message_queue = MessageQueue()
        session = AgentSessionState.create(['system'], 'do the task')
        events: list[dict] = []

        self.assertFalse(agent._inject_queued_messages(session, events, 1))
        agent.message_queue.put('use pytest not unittest')
        agent.message_queue.put('and be quick')
        self.assertTrue(agent._inject_queued_messages(session, events, 2))

        last = session.messages[-1]
        self.assertEqual(last.role, 'user')
        self.assertEqual(last.metadata.get('kind'), QUEUED_MESSAGE_KIND)
        self.assertEqual(
            last.content,
            format_queued_messages(['use pytest not unittest', 'and be quick']),
        )
        self.assertEqual(events[-1]['type'], 'queued_messages_injected')
        self.assertEqual(events[-1]['messages'], ['use pytest not unittest', 'and be quick'])
        self.assertFalse(agent.message_queue.has_pending())

    def test_message_typed_during_run_reaches_next_model_call(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from src.agent_runtime import LocalCodingAgent
        from src.agent_types import AgentRuntimeConfig, ModelConfig

        replies = ['First answer.', 'Switched to pytest as asked.']
        payloads: list[dict] = []
        agent_holder: list[LocalCodingAgent] = []

        class _Resp:
            def __init__(self, text: str) -> None:
                self.body = json.dumps({
                    'choices': [{'message': {'role': 'assistant', 'content': text},
                                 'finish_reason': 'stop'}]
                }).encode('utf-8')

            def read(self) -> bytes:
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *exc) -> None:
                return None

        def fake_urlopen(req, timeout=None):  # noqa: ANN001
            payloads.append(json.loads(req.data.decode('utf-8')))
            if len(payloads) == 1:
                # The user types while the first model call is in flight.
                agent_holder[0].message_queue.put('use pytest instead')
            return _Resp(replies[len(payloads) - 1])

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('src.openai_compat.request.urlopen', side_effect=fake_urlopen):
                agent = LocalCodingAgent(
                    model_config=ModelConfig(model='m', base_url='http://127.0.0.1:8000/v1'),
                    runtime_config=AgentRuntimeConfig(cwd=Path(tmp_dir)),
                )
                agent.message_queue = MessageQueue()
                agent_holder.append(agent)
                result = agent.run('write tests')

        self.assertEqual(len(payloads), 2)
        second_call_text = json.dumps(payloads[1]['messages'])
        self.assertIn('use pytest instead', second_call_text)
        self.assertIn('Switched to pytest', result.final_output)
        self.assertTrue(
            any(e.get('type') == 'queued_messages_injected' for e in result.events)
        )


if __name__ == '__main__':
    unittest.main()
