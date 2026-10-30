"""Native Anthropic Messages API client (``/v1/messages``), stdlib HTTP only.

Used instead of the OpenAI-compatible endpoint when the base URL is
api.anthropic.com and ``SIGIL_ANTHROPIC_NATIVE=1``. Compared with the compat
layer it gives:
  - prompt caching: ``cache_control`` breakpoints on the system prompt, the
    tool list and the latest user turn (cache read/write tokens are counted);
  - thinking text (shown as reasoning, not sent back) via the effort level;
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
    from .providers import detect_provider

    flag = os.environ.get('SIGIL_ANTHROPIC_NATIVE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
    return flag and detect_provider(str(getattr(config, 'base_url', '') or ''),
                                    str(getattr(config, 'model', '') or '')) == 'anthropic'


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

    def pop_original_content(self) -> list[dict[str, Any]] | None:
        """The last reply's content blocks when they held thinking (kept so they go back with the
        tool results); cleared once read."""
        content, self._last_content = self._last_content, None
        return content

    def build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                      output_schema: OutputSchemaConfig | None = None) -> dict[str, Any]:
        from .providers import effective_effort

        system, converted = convert_messages(messages)
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
        if system:
            system[-1]['cache_control'] = {'type': 'ephemeral'}
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
            converted_tools[-1]['cache_control'] = {'type': 'ephemeral'}
            payload['tools'] = converted_tools
        last_user = next((m for m in reversed(converted) if m['role'] == 'user'), None)
        if last_user and last_user['content']:
            last_user['content'][-1] = dict(last_user['content'][-1], cache_control={'type': 'ephemeral'})
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
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .openai_compat import OpenAICompatError, _apply_excluded_params

        payload = _apply_excluded_params(payload)
        base = self.config.base_url.rstrip('/')
        url = base + ('/messages' if base.endswith('/v1') else '/v1/messages')
        from .openai_compat import extra_headers

        req = request.Request(url, data=json.dumps(payload).encode('utf-8'), method='POST', headers={
            'x-api-key': str(self.config.api_key),
            'anthropic-version': ANTHROPIC_VERSION,
            'content-type': 'application/json',
            **extra_headers(),
        })
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise OpenAICompatError(f'HTTP {exc.code} from the Anthropic API: {detail}') from exc
        except (error.URLError, OSError) as exc:
            raise OpenAICompatError(f'Unable to reach the Anthropic API at {url}: {exc}') from exc
        try:
            data = json.loads(raw.decode('utf-8'))
        except ValueError as exc:
            raise OpenAICompatError('The Anthropic API returned invalid JSON') from exc
        if not isinstance(data, dict):
            raise OpenAICompatError('The Anthropic API returned a malformed payload')
        return data

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
                 output_schema: OutputSchemaConfig | None = None) -> AssistantTurn:
        data = self._post(self.build_payload(messages, tools, output_schema))
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
        usage_raw = data.get('usage') or {}
        usage = UsageStats(
            input_tokens=int(usage_raw.get('input_tokens') or 0),
            output_tokens=int(usage_raw.get('output_tokens') or 0),
            cache_creation_input_tokens=int(usage_raw.get('cache_creation_input_tokens') or 0),
            cache_read_input_tokens=int(usage_raw.get('cache_read_input_tokens') or 0),
        )
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
        """One request, replayed as stream events (the runtime handles both the same way)."""
        turn = self.complete(messages, tools, output_schema=output_schema)
        if turn.content:
            yield StreamEvent(type='content_delta', delta=turn.content)
        for index, call in enumerate(turn.tool_calls):
            yield StreamEvent(type='tool_call_delta', tool_call_index=index, tool_call_id=call.id,
                              tool_name=call.name, arguments_delta=json.dumps(call.arguments))
        yield StreamEvent(type='usage', usage=turn.usage)
        yield StreamEvent(type='message_stop', finish_reason=turn.finish_reason)


def _single_client(config: ModelConfig) -> Any:
    if use_native_anthropic(config):
        return AnthropicClient(config)
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
            configs.append(config_for_endpoint(config, name.strip(), base.strip()))
        self.clients = [_single_client(item) for item in configs]

    def complete(self, messages, tools, *, output_schema=None):  # noqa: ANN001
        from .openai_compat import OpenAICompatError

        last: Exception | None = None
        for client in self.clients:
            try:
                return client.complete(messages, tools, output_schema=output_schema)
            except OpenAICompatError as exc:
                last = exc
        raise last  # type: ignore[misc]

    def stream(self, messages, tools, *, output_schema=None):  # noqa: ANN001
        from .openai_compat import OpenAICompatError

        last: Exception | None = None
        for client in self.clients:
            started = False
            try:
                for event in client.stream(messages, tools, output_schema=output_schema):
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
    """The native Anthropic client for api.anthropic.com when switched on, else the compat
    client; wrapped in FallbackClient when fallback models are set."""
    if getattr(config, 'fallback_models', ()):
        return FallbackClient(config)
    return _single_client(config)
