"""Native Anthropic Messages API client (``/v1/messages``), stdlib HTTP only.

Used instead of the OpenAI-compatible endpoint when the base URL is on
anthropic.com (``SIGIL_ANTHROPIC_NATIVE=0`` keeps the compat endpoint, which does
not cache prompts). Compared with the compat layer it gives:
  - prompt caching: ``cache_control`` breakpoints on the tool list, the system
    prompt, the end of the previous request and the end of this one (at most 4;
    see prompt_cache.py); cache reads and 5-minute / 1-hour writes are counted;
  - streaming over server-sent events;
  - thinking via the effort level; thinking blocks go back unchanged with the tool results, and
    Claude 5.1+ requests ask the API to drop a replayed block whose earlier context changed
    (thinking_replay.py) instead of failing;
  - images as native image blocks;
  - mid-run system messages sent as ``<system-reminder>`` user text.
``temperature`` is sent only when set. ``SIGIL_MAX_TOKENS`` sets max_tokens
(default 8192; raised by the thinking budget).
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator
from urllib import error, request

from .agent_types import AssistantTurn, ModelConfig, OutputSchemaConfig, StreamEvent, ToolCall, UsageStats
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

ANTHROPIC_VERSION = '2023-06-01'
_STOP_REASONS = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length', 'stop_sequence': 'stop'}


def use_native_anthropic(config: ModelConfig) -> bool:
    """The native API for an anthropic.com base URL (exact host or subdomain), unless
    SIGIL_ANTHROPIC_NATIVE is 0/false/no/off. The compat endpoint cannot cache prompts."""
    from .prompt_cache import provider_of

    flag = os.environ.get('SIGIL_ANTHROPIC_NATIVE', '').strip().lower()
    if flag in {'0', 'false', 'no', 'off'}:
        return False
    return provider_of(str(getattr(config, 'base_url', '') or '')) == 'anthropic'


_UNMARKABLE = ('thinking', 'redacted_thinking')


def _markable_block(blocks: list[Any]) -> int | None:
    """Index of the last block a cache breakpoint may go on: not thinking, not empty text."""
    for index in range(len(blocks) - 1, -1, -1):
        block = blocks[index]
        if not isinstance(block, dict) or block.get('type') in _UNMARKABLE:
            continue
        if block.get('type') == 'text' and not str(block.get('text') or '').strip():
            continue
        return index
    return None


def add_breakpoints(payload: dict[str, Any], marker: dict[str, str], limit: int = 4) -> int:
    """cache_control on the last tool, the last system block, the end of the previous request (the
    message before the newest assistant reply) and the end of this request. The prefix is cached in
    the order tools, system, messages, so later breakpoints read what earlier requests wrote; the
    previous-request breakpoint keeps that working when a turn adds more than 20 blocks. Empty and
    thinking blocks are never marked; at most `limit` breakpoints. Returns how many were set."""
    spots: list[tuple[list[Any], int]] = []
    tools = payload.get('tools')
    if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        spots.append((tools, len(tools) - 1))
    system = payload.get('system')
    if isinstance(system, list):
        index = _markable_block(system)
        if index is not None:
            spots.append((system, index))
    messages = [m for m in payload.get('messages') or [] if isinstance(m, dict)]
    message_spots: list[tuple[list[Any], int]] = []
    last_assistant = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get('role') == 'assistant'), None)
    candidates = []
    if last_assistant is not None and last_assistant > 0:
        candidates.append(last_assistant - 1)
    candidates.append(len(messages) - 1)
    for position in candidates:
        # the nearest message at or before `position` with something to mark
        while position >= 0:
            content = messages[position].get('content')
            index = _markable_block(content) if isinstance(content, list) else None
            if index is not None:
                if not any(c is content and i == index for c, i in message_spots):
                    message_spots.append((content, index))
                break
            position -= 1
    spots.extend(message_spots)
    spots = spots[-limit:] if len(spots) > limit else spots
    for blocks, index in spots:
        blocks[index] = dict(blocks[index], cache_control=dict(marker))
    return len(spots)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
    return '' if content is None else str(content)


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{'type': 'text', 'text': content or ' '}]
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get('type') == 'text':
            blocks.append({'type': 'text', 'text': part.get('text') or ' '})
        elif part.get('type') == 'image_url':
            url = str((part.get('image_url') or {}).get('url', ''))
            if url.startswith('data:') and ';base64,' in url:
                media, data = url[5:].split(';base64,', 1)
                blocks.append({'type': 'image', 'source': {'type': 'base64', 'media_type': media, 'data': data}})
    return blocks or [{'type': 'text', 'text': ' '}]


def _original_content(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The Messages API content kept with an assistant message (anthropic_content), when it still
    matches the message's tool calls. Thinking blocks must go back complete and unmodified with
    the tool results (Anthropic Messages API: thinking with tool use)."""
    original = message.get('anthropic_content')
    if not isinstance(original, list) or not original:
        return None
    if not any(isinstance(b, dict) and b.get('type') in ('thinking', 'redacted_thinking') for b in original):
        return None
    wanted = [str(call.get('id')) for call in message.get('tool_calls') or [] if isinstance(call, dict)]
    have = [str(b.get('id')) for b in original if isinstance(b, dict) and b.get('type') == 'tool_use']
    if wanted != have:
        return None
    return [dict(block) for block in original if isinstance(block, dict)]


def convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """OpenAI chat messages -> (Anthropic system blocks, Anthropic messages)."""
    system: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    seen_conversation = False
    for message in messages:
        role = message.get('role')
        if role == 'system':
            text = _text_of(message.get('content'))
            if not text:
                continue
            if not seen_conversation:
                system.append({'type': 'text', 'text': text})
            else:  # mid-run reminder: as user text, so it stays where it was added
                converted.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': f'<system-reminder>\n{text}\n</system-reminder>'}]})
            continue
        seen_conversation = True
        if role == 'user':
            converted.append({'role': 'user', 'content': _user_blocks(message.get('content'))})
        elif role == 'assistant':
            original = _original_content(message)
            if original is not None:  # the reply as the API sent it: thinking blocks unchanged
                converted.append({'role': 'assistant', 'content': original})
                continue
            blocks: list[dict[str, Any]] = []
            text = _text_of(message.get('content'))
            if text.strip():
                blocks.append({'type': 'text', 'text': text})
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                raw = function.get('arguments') or '{}'
                try:
                    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except ValueError as _ignored:
                    _log_ignored(_ignored)
                    arguments = {}
                blocks.append({'type': 'tool_use', 'id': call.get('id') or 'call', 'name': function.get('name', ''),
                               'input': arguments if isinstance(arguments, dict) else {}})
            converted.append({'role': 'assistant', 'content': blocks or [{'type': 'text', 'text': ' '}]})
        elif role == 'tool':
            converted.append({'role': 'user', 'content': [{
                'type': 'tool_result', 'tool_use_id': message.get('tool_call_id') or 'call',
                'content': _text_of(message.get('content')) or ' '}]})
    # Anthropic needs alternating roles starting with the user: merge runs of one role.
    merged: list[dict[str, Any]] = []
    for item in converted:
        if merged and merged[-1]['role'] == item['role']:
            merged[-1]['content'].extend(item['content'])
        else:
            merged.append({'role': item['role'], 'content': list(item['content'])})
    if not merged or merged[0]['role'] != 'user':
        merged.insert(0, {'role': 'user', 'content': [{'type': 'text', 'text': '(continue)'}]})
    # Tool results must come first in a user turn.
    for item in merged:
        if item['role'] == 'user':
            item['content'].sort(key=lambda block: 0 if block.get('type') == 'tool_result' else 1)
    return system, merged


class AnthropicClient:
    """Same interface as OpenAICompatClient (complete / stream)."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        from .redact import register_secret

        register_secret(config.api_key)  # never written to the log, even inside a --debug body
        self._last_content: list[dict[str, Any]] | None = None
        # Reasoning blocks the API refused once as bound to another conversation (left out from
        # then on), and whether this endpoint refused thinking.block_binding (thinking_replay.py).
        self._dropped_thinking: set[str] = set()
        self._binding_refused = False

    def pop_original_content(self) -> list[dict[str, Any]] | None:
        """The last reply's content blocks when they held thinking (kept so they go back with the
        tool results); cleared once read."""
        content, self._last_content = self._last_content, None
        return content

    def build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                      output_schema: OutputSchemaConfig | None = None) -> dict[str, Any]:
        from .providers import effective_effort

        from . import thinking_replay

        system, converted = convert_messages(messages)
        thinking_replay.drop_known(converted, self._dropped_thinking)
        if output_schema is not None:
            system.append({'type': 'text', 'text': (
                'Reply with only a JSON object that matches this JSON schema:\n'
                + json.dumps(output_schema.schema))})
        try:
            max_tokens = int(getattr(self.config, 'max_tokens', None) or os.environ.get('SIGIL_MAX_TOKENS', '') or 8192)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            max_tokens = 8192
        payload: dict[str, Any] = {'model': self.config.model, 'max_tokens': max_tokens, 'messages': converted}
        from .prompt_cache import plan_for

        plan = plan_for(self.config.base_url, self.config.model, native=True)
        if system:
            payload['system'] = system
        if tools:
            converted_tools = []
            for tool in tools:
                function = tool.get('function', tool)
                converted_tools.append({
                    'name': function.get('name'),
                    'description': function.get('description', ''),
                    'input_schema': function.get('parameters') or {'type': 'object', 'properties': {}},
                })
            payload['tools'] = converted_tools
        if plan.markers:
            add_breakpoints(payload, plan.marker())
        from .providers import claude_thinking

        level = effective_effort(getattr(self.config, 'effort', None))
        thinking, effort, temperature_ok = claude_thinking(self.config.model, level)
        if thinking is not None:
            payload['thinking'] = thinking
            if thinking.get('type') == 'enabled':  # budget must stay below max_tokens
                payload['max_tokens'] = max(max_tokens, int(thinking['budget_tokens']) + 4096)
        if effort is not None:
            payload['output_config'] = {'effort': effort}
        if thinking is None and temperature_ok and self.config.temperature is not None:
            payload['temperature'] = self.config.temperature  # not allowed together with thinking
        if not self._binding_refused:
            # Claude 5.1+: a replayed thinking block whose earlier context changed is dropped by the
            # API instead of failing the request (SIGIL_THINKING_BINDING=error|off changes this).
            thinking_replay.add_block_binding(payload, self.config.model)
        return payload

    def _request(self, payload: dict[str, Any]) -> Any:
        from .openai_compat import _apply_excluded_params, extra_headers
        from .thinking_replay import beta_header

        payload = _apply_excluded_params(payload)
        base = self.config.base_url.rstrip('/')
        url = base + ('/messages' if base.endswith('/v1') else '/v1/messages')
        headers = {
            'x-api-key': str(self.config.api_key),
            'anthropic-version': ANTHROPIC_VERSION,
            'content-type': 'application/json',
            **extra_headers(),
        }
        thinking = payload.get('thinking')
        if isinstance(thinking, dict) and 'block_binding' in thinking:
            name = next((k for k in headers if k.lower() == 'anthropic-beta'), 'anthropic-beta')
            headers[name] = beta_header(headers.get(name))
        return url, request.Request(url, data=json.dumps(payload).encode('utf-8'), method='POST', headers=headers)

    def _open(self, payload: dict[str, Any]) -> Any:
        from .openai_compat import OpenAICompatError

        url, req = self._request(payload)
        try:
            return request.urlopen(req, timeout=self.config.timeout_seconds)
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise OpenAICompatError(f'HTTP {exc.code} from the Anthropic API: {detail}') from exc
        except (error.URLError, OSError) as exc:
            from .debug_log import url_for_log

            raise OpenAICompatError(f'Unable to reach the Anthropic API at {url_for_log(url)}: {exc}') from exc

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .openai_compat import OpenAICompatError

        with self._open(payload) as response:
            try:
                raw = response.read()
            except OSError as exc:
                raise OpenAICompatError(f'The Anthropic API connection failed while reading: {exc}') from exc
        try:
            data = json.loads(raw.decode('utf-8'))
        except ValueError as exc:
            raise OpenAICompatError('The Anthropic API returned invalid JSON') from exc
        if not isinstance(data, dict):
            raise OpenAICompatError('The Anthropic API returned a malformed payload')
        return data

    def _without_cache_fields(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
        from . import prompt_cache

        if not prompt_cache.should_retry_without_cache(exc, payload, f'the Anthropic API ({self.config.model})'):
            return None
        return prompt_cache.strip_cache_fields(payload)

    def _recover(self, payload: dict[str, Any], exc: Exception) -> tuple[dict[str, Any], str] | None:
        """The request to send instead after an error, and why, or None (raise the error):
        'binding' when the endpoint refused thinking.block_binding (a proxy that drops the beta),
        'strip' when a replayed thinking block was bound to a different conversation (the named
        block and every later one left out, as the Anthropic docs say), 'cache' for cache fields."""
        from . import thinking_replay

        text = str(exc)
        thinking = payload.get('thinking')
        if thinking_replay.binding_rejected(text) and isinstance(thinking, dict) and 'block_binding' in thinking:
            self._binding_refused = True
            thinking_replay.note('binding', self.config.model)
            return thinking_replay.without_block_binding(payload), 'binding'
        position = thinking_replay.signature_error(text)
        if position is not None:
            stripped, removed = thinking_replay.strip_from(payload, position)
            if removed:
                self._dropped_thinking |= removed
                thinking_replay.note('strip', self.config.model)
                return stripped, 'strip'
            return None
        cache_free = self._without_cache_fields(payload, exc)
        return (cache_free, 'cache') if cache_free is not None else None

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        from . import prompt_cache
        from .openai_compat import OpenAICompatError

        for _attempt in range(3):  # bounded: each recovery removes something
            try:
                data = self._post_once(payload)
            except OpenAICompatError as exc:
                recovered = self._recover(payload, exc)
                if recovered is None:
                    raise
                payload, reason = recovered
                if reason == 'cache':
                    data = self._post_once(payload)
                    prompt_cache.remember_rejected(self.config.base_url, self.config.model, str(exc))
                    return data
                continue
            from .thinking_replay import log_transformations

            log_transformations(data, self.config.model)
            return data
        return self._post_once(payload)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
                 output_schema: OutputSchemaConfig | None = None) -> AssistantTurn:
        return self._turn_from_data(self._post(self.build_payload(messages, tools, output_schema)))

    def _turn_from_data(self, data: dict[str, Any]) -> AssistantTurn:
        texts: list[str] = []
        thinking: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get('content') or []:
            kind = block.get('type')
            if kind == 'text':
                texts.append(block.get('text', ''))
            elif kind == 'thinking':
                thinking.append(block.get('thinking', ''))
            # redacted_thinking blocks carry no text; they are kept in anthropic_content
            elif kind == 'tool_use':
                calls.append(ToolCall(id=str(block.get('id') or f'call_{len(calls)}'),
                                      name=str(block.get('name')), arguments=dict(block.get('input') or {})))
        usage = anthropic_usage(data.get('usage'))
        content = ''.join(texts)
        raw_message: dict[str, Any] = {'role': 'assistant', 'content': content}
        if thinking:
            raw_message['reasoning_content'] = '\n\n'.join(thinking)
        blocks = [dict(block) for block in data.get('content') or [] if isinstance(block, dict)]
        self._last_content = blocks if any(
            block.get('type') in ('thinking', 'redacted_thinking') for block in blocks) else None
        if self._last_content is not None:
            raw_message['anthropic_content'] = self._last_content
        return AssistantTurn(
            content=content,
            tool_calls=tuple(calls),
            finish_reason=_STOP_REASONS.get(str(data.get('stop_reason')), data.get('stop_reason')),
            raw_message=raw_message,
            usage=usage,
        )

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
               output_schema: OutputSchemaConfig | None = None) -> Iterator[StreamEvent]:
        """Server-sent events: text and tool input as they arrive, one usage event at the end.
        SIGIL_ANTHROPIC_STREAM=0 sends one request and replays it as events instead."""
        if os.environ.get('SIGIL_ANTHROPIC_STREAM', '').strip().lower() in {'0', 'false', 'no', 'off'}:
            yield from self._replay(self.complete(messages, tools, output_schema=output_schema))
            return
        from . import prompt_cache
        from .openai_compat import OpenAICompatError

        payload = dict(self.build_payload(messages, tools, output_schema), stream=True)
        for _attempt in range(3):  # bounded: each recovery removes something
            started = False
            try:
                for event in self._stream_payload(payload):
                    started = True
                    yield event
                return
            except OpenAICompatError as exc:
                recovered = None if started else self._recover(payload, exc)
                if recovered is None:
                    raise
                payload, reason = recovered
                if reason == 'cache':
                    for index, event in enumerate(self._stream_payload(payload)):
                        if index == 0:
                            prompt_cache.remember_rejected(self.config.base_url, self.config.model, str(exc))
                        yield event
                    return
        yield from self._stream_payload(payload)

    @staticmethod
    def _replay(turn: AssistantTurn) -> Iterator[StreamEvent]:
        if turn.content:
            yield StreamEvent(type='content_delta', delta=turn.content)
        for index, call in enumerate(turn.tool_calls):
            yield StreamEvent(type='tool_call_delta', tool_call_index=index, tool_call_id=call.id,
                              tool_name=call.name, arguments_delta=json.dumps(call.arguments))
        yield StreamEvent(type='usage', usage=turn.usage)
        yield StreamEvent(type='message_stop', finish_reason=turn.finish_reason)

    def _stream_payload(self, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        from .openai_compat import OpenAICompatError

        self._last_content = None
        stopped = False
        blocks: dict[int, dict[str, Any]] = {}
        tool_slots: dict[int, int] = {}
        partial_json: dict[int, list[str]] = {}
        usage_raw: dict[str, Any] = {}
        stop_reason: Any = None
        with self._open(payload) as response:
            headers = getattr(response, 'headers', None)
            content_type = str(headers.get('Content-Type') or '') if headers is not None else ''
            if 'text/event-stream' not in content_type.lower():
                # A gateway or proxy that answers stream=true with one JSON message: use it whole.
                try:
                    data = json.loads(response.read().decode('utf-8'))
                except (OSError, ValueError) as exc:
                    raise OpenAICompatError('The Anthropic API returned neither a stream nor JSON') from exc
                if not isinstance(data, dict) or data.get('type') == 'error' or 'content' not in data:
                    from .debug_log import sanitize_log

                    raise OpenAICompatError(f'The Anthropic API returned an unexpected reply: {sanitize_log(data, 300)}')
                yield from self._replay(self._turn_from_data(data))
                return
            for name, data in _sse_events(response):
                kind = data.get('type') or name
                if kind == 'message_start':
                    usage_raw.update((data.get('message') or {}).get('usage') or {})
                    from .thinking_replay import log_transformations

                    log_transformations(data.get('message'), self.config.model)
                elif kind == 'content_block_start':
                    index = int(data.get('index') or 0)
                    block = dict(data.get('content_block') or {})
                    blocks[index] = block
                    if block.get('type') == 'tool_use':
                        slot = len(tool_slots)
                        tool_slots[index] = slot
                        partial_json[index] = []
                        yield StreamEvent(type='tool_call_delta', tool_call_index=slot,
                                          tool_call_id=str(block.get('id') or f'call_{slot}'),
                                          tool_name=str(block.get('name') or ''), arguments_delta='')
                    elif block.get('type') == 'text' and block.get('text'):
                        yield StreamEvent(type='content_delta', delta=str(block['text']))
                elif kind == 'content_block_delta':
                    index = int(data.get('index') or 0)
                    delta = data.get('delta') or {}
                    block = blocks.setdefault(index, {'type': 'text', 'text': ''})
                    dtype = delta.get('type')
                    if dtype == 'text_delta':
                        text = str(delta.get('text') or '')
                        block['text'] = str(block.get('text') or '') + text
                        if text:
                            yield StreamEvent(type='content_delta', delta=text)
                    elif dtype == 'input_json_delta':
                        piece = str(delta.get('partial_json') or '')
                        partial_json.setdefault(index, []).append(piece)
                        if piece:
                            yield StreamEvent(type='tool_call_delta', tool_call_index=tool_slots.get(index, 0),
                                              arguments_delta=piece)
                    elif dtype == 'thinking_delta':
                        block['thinking'] = str(block.get('thinking') or '') + str(delta.get('thinking') or '')
                    elif dtype == 'signature_delta':
                        block['signature'] = str(block.get('signature') or '') + str(delta.get('signature') or '')
                elif kind == 'content_block_stop':
                    index = int(data.get('index') or 0)
                    block = blocks.get(index) or {}
                    if block.get('type') == 'tool_use':
                        raw = ''.join(partial_json.get(index) or [])
                        try:
                            block['input'] = json.loads(raw) if raw.strip() else dict(block.get('input') or {})
                        except ValueError as exc:
                            _log_ignored(exc)  # the runtime reports the bad arguments to the model
                            block['input'] = {}
                elif kind == 'message_delta':
                    stop_reason = (data.get('delta') or {}).get('stop_reason') or stop_reason
                    for key, value in (data.get('usage') or {}).items():
                        if value not in (None, 0, {}):
                            usage_raw[key] = value
                elif kind == 'error':
                    err = data.get('error') or {}
                    from .debug_log import sanitize_log

                    raise OpenAICompatError(
                        f'Anthropic stream error: {sanitize_log(err.get("type"), 60)}: '
                        f'{sanitize_log(err.get("message"), 300)}')
                elif kind == 'message_stop':
                    stopped = True
                    break
        if not stopped:
            # a cut connection must not look like a finished reply
            raise OpenAICompatError('The Anthropic API stream ended before message_stop')
        ordered = [blocks[i] for i in sorted(blocks)]
        if any(b.get('type') in ('thinking', 'redacted_thinking') for b in ordered):
            self._last_content = [dict(b) for b in ordered]
        yield StreamEvent(type='usage', usage=anthropic_usage(usage_raw))
        yield StreamEvent(type='message_stop', finish_reason=_STOP_REASONS.get(str(stop_reason), stop_reason))


def _sse_events(response: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """(event name, JSON data) from a server-sent events body."""
    from .openai_compat import OpenAICompatError

    name = ''
    data_lines: list[str] = []
    while True:
        try:
            raw = response.readline()
        except OSError as exc:
            raise OpenAICompatError(f'The Anthropic API stream broke: {exc}') from exc
        if not raw:
            break
        line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
        if not line:
            if data_lines:
                text = '\n'.join(data_lines)
                try:
                    data = json.loads(text)
                except ValueError as exc:
                    raise OpenAICompatError('Malformed event in the Anthropic API stream') from exc
                if isinstance(data, dict):
                    yield name, data
            name, data_lines = '', []
            continue
        if line.startswith(':'):
            continue
        field_name, _, value = line.partition(':')
        value = value[1:] if value.startswith(' ') else value
        if field_name == 'event':
            name = value
        elif field_name == 'data':
            data_lines.append(value)
    if data_lines:
        try:
            data = json.loads('\n'.join(data_lines))
        except ValueError as exc:
            raise OpenAICompatError('Malformed trailing event in the Anthropic API stream') from exc
        if isinstance(data, dict):
            yield name, data


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError) as exc:
        _log_ignored(exc)
        return 0


def anthropic_usage(raw: Any) -> UsageStats:
    """Messages API usage: input_tokens excludes cache reads and writes; cache_creation splits the
    writes by TTL (ephemeral_5m_input_tokens / ephemeral_1h_input_tokens)."""
    raw = raw if isinstance(raw, dict) else {}
    split = raw.get('cache_creation') if isinstance(raw.get('cache_creation'), dict) else {}
    written = _int(raw.get('cache_creation_input_tokens'))
    written_1h = _int(split.get('ephemeral_1h_input_tokens'))
    if not written and split:
        written = _int(split.get('ephemeral_5m_input_tokens')) + written_1h
    return UsageStats(
        input_tokens=_int(raw.get('input_tokens')),
        output_tokens=_int(raw.get('output_tokens')),
        cache_creation_input_tokens=written,
        cache_read_input_tokens=_int(raw.get('cache_read_input_tokens')),
        cache_creation_1h_input_tokens=min(written_1h, written),
    )


def _single_client(config: ModelConfig) -> Any:
    if use_native_anthropic(config):
        return AnthropicClient(config)
    from .bedrock_client import BedrockConverseClient, use_bedrock_converse

    if use_bedrock_converse(config):
        try:
            return BedrockConverseClient(config)
        except ValueError as exc:  # no region: the OpenAI-compatible endpoint, as before (logged)
            from .debug_log import get_logger

            get_logger('model').warning('Bedrock Converse not used: %s', exc)
    from .openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)


class FallbackClient:
    """Tries the main model, then each fallback (``name`` or ``name@base_url``) after an error."""

    def __init__(self, config: ModelConfig) -> None:
        from dataclasses import replace

        self.config = config
        from .profiles import config_for_endpoint

        configs = [replace(config, fallback_models=())]
        for item in config.fallback_models:
            name, _, base = str(item).partition('@')
            # the main key only goes to its own server (config_for_endpoint)
            # (config_for_endpoint also derives the fallback's own cache prices)
            configs.append(config_for_endpoint(config, name.strip(), base.strip()))
        self.configs = configs
        self.clients = [_single_client(item) for item in configs]
        # The config of the endpoint that answered the last request: its model and prices are used
        # for that turn's cost (a fallback on another provider has other cache prices).
        self.last_config: ModelConfig | None = None
        self._last_client: Any = None

    def pop_original_content(self) -> list[dict[str, Any]] | None:
        """The answering client's kept reply blocks (native thinking), so they go back with the tool results."""
        pop = getattr(self._last_client, 'pop_original_content', None)
        return pop() if callable(pop) else None

    def pop_replay(self) -> dict[str, Any] | None:
        pop = getattr(self._last_client, 'pop_replay', None)
        return pop() if callable(pop) else None

    def complete(self, messages, tools, *, output_schema=None):  # noqa: ANN001
        from .openai_compat import OpenAICompatError

        last: Exception | None = None
        for config, client in zip(self.configs, self.clients):
            try:
                turn = client.complete(messages, tools, output_schema=output_schema)
            except OpenAICompatError as exc:
                last = exc
                continue
            self.last_config = config
            self._last_client = client
            return turn
        raise last  # type: ignore[misc]

    def stream(self, messages, tools, *, output_schema=None):  # noqa: ANN001
        from .openai_compat import OpenAICompatError

        last: Exception | None = None
        for config, client in zip(self.configs, self.clients):
            started = False
            try:
                for event in client.stream(messages, tools, output_schema=output_schema):
                    if not started:
                        self.last_config = config
                        self._last_client = client
                    started = True
                    yield event
                return
            except OpenAICompatError as exc:
                if started:
                    raise  # part of the answer was already shown
                last = exc
        raise last  # type: ignore[misc]


def fast_client(config: ModelConfig, default: Any = None) -> Any:
    """A client for side jobs (compaction summaries, the web_fetch prompt).

    SIGIL_FAST_MODEL or the `fastModel` setting: "model" or "model@base_url". Without one,
    the main model is used.
    """
    from dataclasses import replace

    value = os.environ.get('SIGIL_FAST_MODEL', '').strip()
    if not value:
        try:
            from .settings import load_settings, project_dir

            value = str(load_settings(project_dir()).get('fastModel') or '').strip()
        except Exception as _ignored:
            _log_ignored(_ignored)
            value = ''
    if not value:
        return default if default is not None else make_client(config)
    from .profiles import config_for_endpoint

    name, _, base = value.partition('@')
    return make_client(config_for_endpoint(config, name.strip(), base.strip()))  # key stays with its server


def make_client(config: ModelConfig) -> Any:
    """The native Anthropic client for an anthropic.com base URL (unless SIGIL_ANTHROPIC_NATIVE=0),
    else the compat client; wrapped in FallbackClient when fallback models are set."""
    if getattr(config, 'fallback_models', ()):
        return FallbackClient(config)
    return _single_client(config)
