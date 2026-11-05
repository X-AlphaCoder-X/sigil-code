"""A multi-line paste typed while the agent works must stay one message."""

import unittest

from src.message_queue import ConsoleInputReader, MessageQueue


def _reader(**kwargs):
    queue = MessageQueue()
    return queue, ConsoleInputReader(queue, echo=lambda _text: None, **kwargs)


class PasteBurstTests(unittest.TestCase):
    def test_pasted_lines_are_one_message(self) -> None:
        queue, reader = _reader()
        reader.feed_burst(list('line one\r\nline two\rline three'))
        self.assertEqual(queue.drain(), [])  # nothing sent by the paste itself
        reader.feed_burst(['\r'])  # the user presses Enter
        self.assertEqual(queue.drain(), ['line one\nline two\nline three'])

    def test_trailing_line_break_in_a_long_paste_does_not_send(self) -> None:
        queue, reader = _reader()
        reader.feed_burst(list('a longer pasted sentence\r\n'))
        self.assertEqual(queue.drain(), [])
        reader.feed_burst(['\r'])
        self.assertEqual(queue.drain(), ['a longer pasted sentence'])

    def test_typed_enter_still_sends(self) -> None:
        queue, reader = _reader()
        for ch in 'hi':
            reader.feed_burst([ch])
        reader.feed_burst(['\r'])
        self.assertEqual(queue.drain(), ['hi'])
        reader.feed_burst(list('ok\r'))  # short fast burst ending in Enter
        self.assertEqual(queue.drain(), ['ok'])

    def test_ctrl_enter_alone_still_sends_now(self) -> None:
        sent = []
        queue, reader = _reader(on_send_now=lambda: sent.append(True))
        reader.feed_burst(list('go'))
        reader.feed_burst(['\n'])
        self.assertEqual(queue.drain(), ['go'])
        self.assertEqual(sent, [True])

    def test_pasted_line_break_shows_as_space(self) -> None:
        shown = []
        queue = MessageQueue()
        reader = ConsoleInputReader(queue, echo=shown.append)
        reader.feed_burst(list('first line\r\nsecond line'))
        self.assertNotIn('\n', ''.join(shown).replace('\r', ''))

    def test_paused_draft_with_line_breaks_is_not_sent_on_resume(self) -> None:
        queue, reader = _reader()
        reader.feed_burst(list('draft one\r\ndraft two'))
        reader.pause()
        reader.resume()
        self.assertEqual(queue.drain(), [])
        self.assertEqual(''.join(reader._buffer), 'draft one\ndraft two')


if __name__ == '__main__':
    unittest.main()
