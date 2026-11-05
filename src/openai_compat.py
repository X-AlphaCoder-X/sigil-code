from __future__ import annotations

import http.client
import json
import os
import queue
import re
import threading
from typing import Any, Iterator
from urllib import error, request

from .agent_types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
    UsageStats,
)
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


class OpenAICompatError(RuntimeError):
    """Raised when the local OpenAI-compatible backend returns an invalid response."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        from .debug_log import clip, get_logger

        get_logger('model').warning('backend error: %s', clip(self, 2000))


_EXTRA_HEADERS_CACHE: dict[str, tuple[tuple, Any]] = {}
_EXTRA_HEADERS_LOCK = __import__('threading').Lock()  # why: parallel subagents send requests at once
# an error text about Google's cached content (CachedContent, cached_content, cachedContents/...)
_CACHED_CONTENT_RE = re.compile(r'cached[ _]?contents?', re.IGNORECASE)


def _configured_extra_headers() -> Any:
    """`extraHeaders` from the agent's settings, cached until a settings file changes."""
    import os

    from .settings import load_settings, project_dir, settings_paths

    folder = project_dir()
    key_parts: list[Any] = [os.environ.get('SIGIL_SETTINGS', ''), os.environ.get('SIGIL_SETTING_SOURCES', '')]
    for _label, path in settings_paths(folder):
        try:
            key_parts.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            key_parts.append((str(path), None))
    key = tuple(key_parts)
    with _EXTRA_HEADERS_LOCK:
        cached = _EXTRA_HEADERS_CACHE.get(str(folder))
    if cached is not None and cached[0] == key:
        return cached[1]
    value = load_settings(folder).get('extraHeaders')
    with _EXTRA_HEADERS_LOCK:
        if len(_EXTRA_HEADERS_CACHE) >= 32 and str(folder) not in _EXTRA_HEADERS_CACHE:
            _EXTRA_HEADERS_CACHE.pop(next(iter(_EXTRA_HEADERS_CACHE)))  # bounded: oldest folder out
        _EXTRA_HEADERS_CACHE[str(folder)] = (key, value)
    return value


def extra_headers() -> dict[str, str]:
    """Extra request headers: SIGIL_EXTRA_HEADERS (a JSON object, or "Name: value" pairs split by
    newlines or ';') and the `extraHeaders` setting. ${VAR} is replaced from the environment.
    For example anthropic-workspace-id, or OpenRouter's HTTP-Referer and X-Title."""
    import os

    headers: dict[str, str] = {}
    try:
        configured = _configured_extra_headers()
        if isinstance(configured, dict):
            headers.update({str(k): str(v) for k, v in configured.items()})
    except Exception as _ignored:
        from .debug_log import log_ignored_error

        log_ignored_error(_ignored)
    raw = os.environ.get('SIGIL_EXTRA_HEADERS', '').strip()
    if raw.startswith('{'):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                headers.update({str(k): str(v) for k, v in parsed.items()})
        except ValueError as _ignored:
            _log_ignored(_ignored)
            pass
    elif raw:
        for item in raw.replace(';', '\n').splitlines():
            name, sep, value = item.partition(':')
            if sep and name.strip():
                headers[name.strip()] = value.strip()
    return {name: os.path.expandvars(value) for name, value in headers.items()}


def _join_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip('/')
    return f'{base}/{suffix.lstrip("/")}'


def _parse_excluded_params() -> set[str]:
    """Parse SIGIL_EXCLUDE_PARAMS into a set of payload keys to strip.

    Some OpenAI-compatible providers reject sampling parameters such as
    temperature, top_p or top_k in the request payload. Set this env var
    (comma-separated) to remove those keys before the request is sent.

    Examples:
        SIGIL_EXCLUDE_PARAMS=temperature
        SIGIL_EXCLUDE_PARAMS=temperature,top_p,top_k
    """
    raw = os.environ.get('SIGIL_EXCLUDE_PARAMS', '')
    return {
        item.strip().lower()
        for item in raw.split(',')
        if item.strip()
    }


def _apply_excluded_params(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove user-excluded keys (SIGIL_EXCLUDE_PARAMS) from the payload."""
    excluded = _parse_excluded_params()
    if not excluded:
        return payload
    return {key: value for key, value in payload.items() if key.lower() not in excluded}


def _normalize_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get('type') == 'text' and isinstance(item.get('text'), str):
                parts.append(item['text'])
                continue
            if isinstance(item.get('text'), str):
                parts.append(item['text'])
                continue
            parts.append(json.dumps(item, ensure_ascii=True))
        return ''.join(parts)
    return str(content)


# Same key as agent_runtime.INVALID_ARGUMENTS_KEY: the tool loop answers such a call with an error.
INVALID_ARGUMENTS_KEY = '__invalid_arguments__'


def _tool_arguments_or_marker(raw_arguments: Any) -> dict[str, Any]:
    """Like _parse_tool_arguments, but a cut-off or non-object string becomes the
    INVALID_ARGUMENTS_KEY marker, so the tool loop answers that call with an error the model
    can retry from (as the streaming path does) instead of ending the run."""
    try:
        return _parse_tool_arguments(raw_arguments)
    except OpenAICompatError:
        if not isinstance(raw_arguments, str):
            raise
    raw = raw_arguments.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as _ignored:
        _log_ignored(_ignored)
        parsed = None
    detail = ('it is not valid JSON (probably cut off)' if parsed is None
              else f'it decoded to {type(parsed).__name__}, not an object')
    return {
        INVALID_ARGUMENTS_KEY: (
            f'The arguments for this tool call could not be used: {detail}. '
            f'Received: {raw[:300]!r}. Call the tool again with complete, valid JSON '
            'arguments. If the content is very large, split it into smaller calls '
            '(for example write the file in parts or use several edit_file calls).'
        )
    }


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        raw_arguments = raw_arguments.strip()
        if not raw_arguments:
            return {}
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise OpenAICompatError(
                f'Invalid tool arguments returned by model: {raw_arguments!r}'
            ) from exc
        if not isinstance(parsed, dict):
            raise OpenAICompatError(
                f'Tool arguments must decode to an object, got {type(parsed).__name__}'
            )
        return parsed
    raise OpenAICompatError(
        f'Unsupported tool arguments payload: {type(raw_arguments).__name__}'
    )


def _optional_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            return 0
    return 0


def _parse_usage(payload: Any) -> UsageStats:
    if not isinstance(payload, dict):
        return UsageStats()
    completion_details = payload.get('completion_tokens_details')
    if not isinstance(completion_details, dict):
        completion_details = {}
    input_tokens = (
        _optional_int(payload.get('input_tokens'))
        or _optional_int(payload.get('prompt_tokens'))
        or _optional_int(payload.get('prompt_eval_count'))
    )
    details = payload.get('prompt_tokens_details') or payload.get('input_tokens_details')
    if not isinstance(details, dict):
        details = {}
    split = payload.get('cache_creation') if isinstance(payload.get('cache_creation'), dict) else {}
    cache_write = _optional_int(payload.get('cache_creation_input_tokens'))
    cache_read = _optional_int(payload.get('cache_read_input_tokens'))
    anthropic_style = bool(cache_write or cache_read) and not _optional_int(payload.get('prompt_tokens'))
    if not anthropic_style:
        # Why: OpenAI-style APIs count cached tokens inside the prompt total; Anthropic-style
        # fields count them apart. Split them out so both read the same way.
        cache_read = (cache_read
                      or _optional_int(details.get('cached_tokens'))
                      or _optional_int(payload.get('cached_tokens'))              # Together (top level)
                      or _optional_int(payload.get('prompt_cache_hit_tokens')))   # DeepSeek
        cache_write = (cache_write
                       or _optional_int(details.get('cache_write_tokens'))           # OpenAI 5.6+, OpenRouter
                       or _optional_int(details.get('cache_creation_input_tokens'))   # DashScope
                       or _optional_int(details.get('cache_creation_tokens')))         # SambaNova
        cache_read = max(0, min(cache_read, input_tokens))
        cache_write = max(0, min(cache_write, input_tokens - cache_read))
        input_tokens -= cache_read + cache_write
    written_1h = min(_optional_int(split.get('ephemeral_1h_input_tokens')), cache_write)
    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=(
            _optional_int(payload.get('output_tokens'))
            or _optional_int(payload.get('completion_tokens'))
            or _optional_int(payload.get('eval_count'))
        ),
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
        reasoning_tokens=(
            _optional_int(payload.get('reasoning_tokens'))
            or _optional_int(completion_details.get('reasoning_tokens'))
        ),
        cache_creation_1h_input_tokens=max(0, written_1h),
    )


_CACHED_HEADER_KEY = '__sigil_cached_prompt_tokens'


def _header_cached_tokens(headers: Any) -> int | None:
    """Fireworks puts the cached prompt tokens of a non-streamed reply in a response header
    (docs.fireworks.ai/serverless/overview: fireworks-cached-prompt-tokens)."""
    for name, value in (headers or {}).items():
        if str(name).lower() == 'fireworks-cached-prompt-tokens':
            count = _optional_int(value)
            return count if count > 0 else None
    return None


def _perf_cached_tokens(payload: Any) -> int | None:
    """A streamed Fireworks reply has no per-request headers; the metrics come in the final chunk's
    perf_metrics (docs.fireworks.ai/api-reference/post-chatcompletions). The key spelling inside it
    is not documented, so the header name and its snake_case form are both read."""
    metrics = payload.get('perf_metrics') if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        return None
    for key, value in metrics.items():
        if str(key).lower().replace('_', '-') in ('fireworks-cached-prompt-tokens', 'cached-prompt-tokens'):
            count = _optional_int(value)
            return count if count > 0 else None
    return None


def _with_cached_count(usage: UsageStats, cached: int | None) -> UsageStats:
    """Move a cached count reported outside the usage object into cache reads (only when the usage
    object reports none, so nothing is counted twice)."""
    if not cached or usage.cache_read_input_tokens or usage.cache_creation_input_tokens:
        return usage
    moved = min(int(cached), usage.input_tokens)
    from dataclasses import replace as _replace

    return _replace(usage, input_tokens=usage.input_tokens - moved, cache_read_input_tokens=moved)


def _with_cache_breakpoints(messages: list[Any], cache_control: dict[str, str]) -> list[Any]:
    """Kept for callers of the old name: see prompt_cache.with_breakpoints."""
    from .prompt_cache import with_breakpoints

    return with_breakpoints(messages, cache_control)


def _normalize_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every tool has a valid JSON-schema parameters object.

    Anthropic's OpenAI compatibility layer requires tools.N.function.parameters.type
    to be present ("object"). OpenAI tolerates its absence, so malformed entries
    only surface on stricter providers. Normalize every entry defensively,
    regardless of where the tool came from (SigilTool, MCP, plugin, raw dict).
    """
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get('function')
        if not isinstance(function, dict):
            normalized.append(tool)
            continue
        parameters = function.get('parameters')
        if not isinstance(parameters, dict) or parameters.get('type') != 'object':
            function = {**function, 'parameters': {'type': 'object', 'properties': {}}}
        normalized.append({**tool, 'function': function})
    return normalized


def _build_response_format(
    schema: OutputSchemaConfig | None,
) -> dict[str, Any] | None:
    if schema is None:
        return None
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': schema.name,
            'schema': schema.schema,
            'strict': schema.strict,
        },
    }


def _log_body(label: str, payload: Any) -> None:
    """Raw bodies go to the log only with --debug / SIGIL_DEBUG=1 (no headers, so no key)."""
    from .debug_log import clip, debug_enabled, get_logger

    if debug_enabled():
        get_logger('model').debug('%s: %s', label, clip(json.dumps(payload, default=str)))


def _env_max_tokens() -> int | None:
    try:
        value = int(os.environ.get('SIGIL_MAX_TOKENS', '') or 0)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return None
    return value if value > 0 else None


def _reminders_as_user(config: Any) -> bool:
    """Mid-run system messages become user text for Anthropic's compat layer (it moves every
    system message to the top). SIGIL_REMINDERS_AS_USER=1/0 forces it on or off for any server."""
    flag = os.environ.get('SIGIL_REMINDERS_AS_USER', '').strip().lower()
    if flag in ('1', 'true', 'yes', 'on'):
        return True
    if flag in ('0', 'false', 'no', 'off'):
        return False
    from .providers import detect_provider

    return detect_provider(config.base_url, config.model) == 'anthropic'


def _system_reminders_to_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    started = False
    for message in messages:
        if message.get('role') == 'system' and started:
            text = message.get('content') if isinstance(message.get('content'), str) else str(message.get('content'))
            if '<system-reminder>' not in text:
                text = f'<system-reminder>\n{text}\n</system-reminder>'
            converted.append({'role': 'user', 'content': text})
            continue
        if message.get('role') != 'system':
            started = True
        converted.append(message)
    return converted


class OpenAICompatClient:
    """Minimal OpenAI-compatible chat client for local model servers."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        from .redact import register_secret

        register_secret(config.api_key)  # never written to the log, even inside a --debug body
        from .token_router import TokenRouter
        self._rate_limiter = TokenRouter.instance()
        import urllib.parse
        parsed = urllib.parse.urlparse(config.base_url)
        if (parsed.hostname not in ('127.0.0.1', 'localhost', '::1', None)
                and parsed.scheme != 'https'):
            if not getattr(config, 'allow_insecure_http', False):
                raise OpenAICompatError(
                    f'Refusing to send API key over unencrypted HTTP to {config.base_url}. '
                    'Use HTTPS for non-localhost endpoints, or set allow_insecure_http=True to override.'
                )
        # Reasoning sent back with later requests (thinking_replay.py): what the last reply needs
        # replayed, and what this endpoint has asked for or refused.
        self._last_replay: dict[str, Any] | None = None
        self._replay_echo_learned = False
        self._replay_fill = False
        self._replay_dummy_signatures = False
        self._replay_refused = False

    def pop_replay(self) -> dict[str, Any] | None:
        """reasoning_content and tool-call extra_content of the last reply (sent back with later
        requests to this endpoint); cleared once read."""
        replay, self._last_replay = self._last_replay, None
        return replay

    def _learn_replay(self, exc: Exception) -> bool:
        """A 400/422 that asks for reasoning to be sent back, or refuses it: change what is sent to
        this endpoint for the rest of the session and say whether to send the request again."""
        from . import thinking_replay

        kind = thinking_replay.classify_compat_error(str(exc))
        if kind == 'reasoning' and not (self._replay_fill or self._replay_refused):
            self._replay_echo_learned = self._replay_fill = True
        elif kind == 'signature' and not (self._replay_dummy_signatures or self._replay_refused):
            self._replay_dummy_signatures = True
        elif kind == 'reject' and not self._replay_refused:
            self._replay_refused = True
        else:
            return False
        thinking_replay.note(kind, self.config.model)
        return True

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn:
        self._rate_limiter.acquire()
        self._last_replay = None
        for attempt in range(3):  # bounded: each retry changes what is sent (_learn_replay)
            request_payload = self._build_payload(
                messages=messages,
                tools=tools,
                stream=False,
                output_schema=output_schema,
            )
            _log_body('request', request_payload)
            try:
                payload = self._post_with_fallbacks(request_payload)
                break
            except OpenAICompatError as exc:
                if attempt == 2 or not self._learn_replay(exc):
                    raise
        _log_body('response', payload)
        choices = payload.get('choices')
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatError('Local model backend returned no choices')
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenAICompatError('Local model backend returned malformed choice data')

        message = first_choice.get('message')
        if not isinstance(message, dict):
            raise OpenAICompatError('Local model backend returned no assistant message')

        content = _normalize_content(message.get('content'))
        tool_calls = self._parse_tool_calls_from_message(message)

        finish_reason = first_choice.get('finish_reason')
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        usage = _with_cached_count(_parse_usage(payload.get('usage')), payload.pop(_CACHED_HEADER_KEY, None))
        self._rate_limiter.record_usage(usage.total_tokens)
        from . import explicit_cache

        explicit_cache.observe(self.config.base_url, self.config.model, usage)
        from .thinking_replay import capture

        self._last_replay = capture(message, self.config.base_url, self.config.model)
        return AssistantTurn(
            content=content,
            tool_calls=tuple(tool_calls),
            finish_reason=finish_reason,
            raw_message=message,
            usage=usage,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> Iterator[StreamEvent]:
        self._rate_limiter.acquire()
        self._last_replay = None
        for attempt in range(3):  # bounded: each retry changes what is sent (_learn_replay)
            payload = self._build_payload(
                messages=messages,
                tools=tools,
                stream=True,
                output_schema=output_schema,
            )
            started = False
            try:
                for event in self._stream_with_fallbacks(payload):
                    started = True
                    yield event
                return
            except OpenAICompatError as exc:
                if started or attempt == 2 or not self._learn_replay(exc):
                    raise

    def _bearer(self) -> str:
        """The bearer key: on Vertex a fresh access token when SIGIL_GOOGLE_TOKEN_COMMAND is set
        (google_auth), else the configured key."""
        from .prompt_cache import provider_of

        if provider_of(self.config.base_url) == 'vertex':
            from .google_auth import access_token

            return access_token(str(self.config.api_key))
        return str(self.config.api_key)

    @staticmethod
    def _explicit_cache_failed(exc: Exception) -> bool:
        """An error to a request that named an explicit cache that is about that cache: send it again
        without the cache. Google answers a cache that expired, was deleted or belongs to another
        project with 403 or 404 ("CachedContent not found (or permission denied)"), and a 400/422 that
        names the cached content. A rate limit (429), an auth error (401) or a 400 about something else
        (a context that is too long) is not about the cache: it keeps the cache and goes to the normal
        retry. Why: every 4xx used to delete a valid cache and resend at once, without backoff."""
        text = str(exc)
        match = re.match(r'HTTP (\d{3})\b', text)
        if match is None:
            return False
        status = int(match.group(1))
        if status in (403, 404):
            return True
        return status in (400, 422) and _CACHED_CONTENT_RE.search(text) is not None

    def _post_with_fallbacks(self, payload: dict[str, Any]) -> dict[str, Any]:
        from . import explicit_cache

        full = payload.pop(explicit_cache.FULL_KEY, None)
        if full is None:
            return self._post_plain(payload)
        try:
            return self._post_plain(payload)
        except OpenAICompatError as exc:
            if not self._explicit_cache_failed(exc):
                raise
            explicit_cache.request_failed(payload, exc)
            return self._post_plain(full)

    def _post_plain(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST once; on a 400 for reasoning_effort "none" retry with "low"; on a 400/422 about the
        cache fields we sent retry once without them (remembered for the endpoint when that works)."""
        try:
            return self._request_json(payload)
        except OpenAICompatError as exc:
            if self._retry_without_effort_none(payload, exc):
                return self._post_plain(payload)  # bounded: effort is now "low"
            stripped = self._without_cache_fields(payload, exc)
            if stripped is None:
                raise
            result = self._request_json(stripped)
            self._remember_cache_rejection(exc)
            return result

    def _stream_with_fallbacks(self, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        from . import explicit_cache

        full = payload.pop(explicit_cache.FULL_KEY, None)
        if full is None:
            yield from self._stream_plain(payload)
            return
        started = False
        try:
            for event in self._stream_plain(payload):
                started = True
                yield event
        except OpenAICompatError as exc:
            if started or not self._explicit_cache_failed(exc):
                raise
            explicit_cache.request_failed(payload, exc)
            yield from self._stream_plain(full)

    def _stream_plain(self, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        started = False
        try:
            for event in self._stream_payload(payload):
                started = True
                yield event
        except OpenAICompatError as exc:
            if started:
                raise
            if self._retry_without_effort_none(payload, exc):
                yield from self._stream_plain(payload)
                return
            stripped = self._without_cache_fields(payload, exc)
            if stripped is None:
                raise
            remembered = False
            for event in self._stream_payload(stripped):
                if not remembered:
                    remembered = True
                    self._remember_cache_rejection(exc)
                yield event
            if not remembered:
                self._remember_cache_rejection(exc)

    def _without_cache_fields(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
        """The payload without cache fields when the 400/422 is about them, else None (logged either way)."""
        from . import prompt_cache

        if not prompt_cache.should_retry_without_cache(exc, payload, self.config.model):
            return None
        return prompt_cache.strip_cache_fields(payload)

    def _remember_cache_rejection(self, exc: Exception) -> None:
        from . import prompt_cache

        prompt_cache.remember_rejected(self.config.base_url, self.config.model, str(exc))

    def _cache_headers(self, payload: dict[str, Any]) -> dict[str, str]:
        """Session headers that keep a conversation on the server holding its cache (xAI, Fireworks,
        OpenRouter). Headers are never rejected as unknown, so they stay on a retry."""
        from . import prompt_cache

        plan = prompt_cache.plan_for(self.config.base_url, self.config.model)
        headers: dict[str, str] = {}
        if plan.active and plan.key_headers:
            key = prompt_cache.cache_key(self.config.model, payload.get('messages') or [])
            headers = {name: key for name in plan.key_headers}
        return headers

    def _stream_payload(self, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        _log_body('request (stream)', payload)
        req = request.Request(
            _join_url(self.config.base_url, '/chat/completions'),
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {self._bearer()}',
                'Content-Type': 'application/json',
                **self._cache_headers(payload),
                **extra_headers(),
            },
            method='POST',
        )

        q: queue.Queue[Any] = queue.Queue()

        def _worker():
            try:
                # [BLOCKING] Handshake
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    q.put(StreamEvent(type='message_start'))
                    state: dict[str, Any] = {}  # per stream: a cached count seen in perf_metrics
                    # [BLOCKING] Payload iteration (line-by-line reading)
                    for event_payload in self._iter_sse_payloads(response):
                        for event in self._parse_stream_payload(event_payload, state):
                            q.put(event)
                    self._last_replay = self._replay_from_stream(state)
            except BaseException as e:  # noqa: BLE001 - re-raised in the calling thread
                q.put(e)
            finally:
                q.put(None) # SENTINEL

        # Start background worker to decouple I/O from signal handling
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            try:
                # Poll queue every 100ms to allow main-thread SIGINT processing
                item = q.get(timeout=0.1)
                if item is None:
                    break
                if isinstance(item, BaseException) and not isinstance(item, Exception):
                    raise item
                if isinstance(item, Exception):
                    # Handle errors encountered in the background thread
                    if isinstance(item, error.HTTPError):
                        detail = item.read().decode('utf-8', errors='replace')
                        raise OpenAICompatError(f'HTTP {item.code} from local model backend: {detail}') from item
                    if isinstance(item, error.URLError):
                        raise OpenAICompatError(f'Unable to reach local model backend at {self.config.base_url}: {item.reason}') from item
                    if isinstance(item, TimeoutError):
                        raise OpenAICompatError(f'Model API request timed out after {self.config.timeout_seconds}s. Consider increasing timeout_seconds in config.') from item
                    if isinstance(item, (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionError, OSError)):
                        raise OpenAICompatError(f'Backend closed the connection without a response ({type(item).__name__}: {item}). This is usually a transient backend issue; retrying.') from item
                    raise item
                yield item
            except queue.Empty as _ignored:
                _log_ignored(_ignored)
                continue

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode('utf-8')
        req = request.Request(
            _join_url(self.config.base_url, '/chat/completions'),
            data=body,
            headers={
                'Authorization': f'Bearer {self._bearer()}',
                'Content-Type': 'application/json',
                **self._cache_headers(payload),
                **extra_headers(),
            },
            method='POST',
        )

        q: queue.Queue[Any] = queue.Queue()

        def _worker():
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    q.put((response.read(), dict(getattr(response, 'headers', None) or {})))
            except BaseException as e:  # noqa: BLE001 - re-raised in the calling thread
                q.put(e)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        raw = None
        while True:
            try:
                item = q.get(timeout=0.1)
                raw = item
                break
            except queue.Empty as _ignored:
                _log_ignored(_ignored)
                continue

        if isinstance(raw, BaseException) and not isinstance(raw, Exception):
            raise raw
        if isinstance(raw, Exception):
            if isinstance(raw, error.HTTPError):
                detail = raw.read().decode('utf-8', errors='replace')
                if '"image_url"' in json.dumps(payload) and raw.code in (400, 415, 422):
                    detail += (' (this request included an image; the model may be text-only. '
                               'Use a vision model, or set SIGIL_VISION=0)')
                raise OpenAICompatError(f'HTTP {raw.code} from local model backend: {detail}') from raw
            if isinstance(raw, error.URLError):
                raise OpenAICompatError(f'Unable to reach local model backend at {self.config.base_url}: {raw.reason}') from raw
            if isinstance(raw, TimeoutError):
                raise OpenAICompatError(f'Model API request timed out after {self.config.timeout_seconds}s. Consider increasing timeout_seconds in config.') from raw
            if isinstance(raw, (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionError, OSError)):
                raise OpenAICompatError(f'Backend closed the connection without a response ({type(raw).__name__}: {raw}). This is usually a transient backend issue; retrying.') from raw
            raise raw

        raw, response_headers = raw if isinstance(raw, tuple) else (raw, {})
        try:
            payload = json.loads(raw.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise OpenAICompatError('Local model backend returned invalid JSON') from exc
        if not isinstance(payload, dict):
            raise OpenAICompatError('Local model backend returned malformed JSON payload')
        cached = _header_cached_tokens(response_headers)
        if cached is not None:
            payload[_CACHED_HEADER_KEY] = cached
        return payload

    def _build_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        stream: bool,
        output_schema: OutputSchemaConfig | None,
    ) -> dict[str, Any]:
        from . import prompt_cache as _prompt_cache
        from . import thinking_replay

        provider = _prompt_cache.cache_provider(self.config.base_url, self.config.model)
        payload: dict[str, Any] = {
            'model': self.config.model,
            # anthropic_content and provider_replay are internal and never sent as such; a reply's
            # reasoning_content / tool-call extra_content go back to the endpoint that wrote them
            'messages': thinking_replay.prepare(
                messages, base_url=self.config.base_url, model=self.config.model, provider=provider,
                echo=not self._replay_refused and (
                    self._replay_echo_learned or thinking_replay.echoes_reasoning(provider, self.config.model)),
                fill=self._replay_fill, dummy_signatures=self._replay_dummy_signatures,
                send_fields=not self._replay_refused),
            'stream': stream,
        }
        # Sent only when set: Claude 4.7+ returns HTTP 400 for values below 1.
        if self.config.temperature is not None:
            payload['temperature'] = self.config.temperature
        top_p_raw = os.environ.get('SIGIL_TOP_P')
        if top_p_raw:
            try:
                payload['top_p'] = float(top_p_raw)
            except ValueError as _ignored:
                _log_ignored(_ignored)
                pass
        top_k_raw = os.environ.get('SIGIL_TOP_K')
        if top_k_raw:
            try:
                payload['top_k'] = int(float(top_k_raw))
            except ValueError as _ignored:
                _log_ignored(_ignored)
                pass
        if tools:
            payload['tools'] = _normalize_tool_schemas(tools)
            payload['tool_choice'] = 'auto'
        if stream:
            payload['stream_options'] = {'include_usage': True}
        response_format = _build_response_format(output_schema)
        if response_format is not None:
            payload['response_format'] = response_format
        max_tokens = getattr(self.config, 'max_tokens', None) or _env_max_tokens()
        if max_tokens:
            payload['max_tokens'] = max_tokens
        if _reminders_as_user(self.config):
            payload['messages'] = _system_reminders_to_user(payload['messages'])
        from .providers import apply_effort, effective_effort, get_quirks

        apply_effort(
            payload,
            effective_effort(getattr(self.config, 'effort', None)),
            get_quirks(self.config.base_url, self.config.model),
        )
        if getattr(self, '_effort_none_rejected', False) and payload.get('reasoning_effort') == 'none':
            payload['reasoning_effort'] = 'low'
        from . import prompt_cache

        plan = prompt_cache.plan_for(self.config.base_url, self.config.model)
        if plan.markers:
            payload['messages'] = prompt_cache.with_breakpoints(payload['messages'], plan.marker())
        prompt_cache.apply_hints(plan, payload, {})  # headers: _cache_headers, per request
        from . import explicit_cache

        # Gemini / Vertex: the stable prefix from a cachedContents resource when the price rule says so
        return explicit_cache.apply(self.config, _apply_excluded_params(payload), self._bearer)

    def _retry_without_effort_none(self, payload: dict[str, Any], exc: Exception) -> bool:
        """A 400 for reasoning_effort 'none' (models before gpt-5.1, some newer ones): use 'low' from now on."""
        if payload.get('reasoning_effort') != 'none' or 'HTTP 400' not in str(exc):
            return False
        self._effort_none_rejected = True
        payload['reasoning_effort'] = 'low'
        from .debug_log import get_logger

        get_logger('model').info('reasoning_effort "none" was rejected by %s; using "low" instead',
                                 self.config.model)
        return True

    def _parse_tool_calls_from_message(self, message: dict[str, Any]) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get('tool_calls')
        if isinstance(raw_tool_calls, list):
            for idx, raw_call in enumerate(raw_tool_calls):
                if not isinstance(raw_call, dict):
                    raise OpenAICompatError('Malformed tool call payload from model')
                function_block = raw_call.get('function') or {}
                if not isinstance(function_block, dict):
                    raise OpenAICompatError('Malformed tool call function payload from model')
                name = function_block.get('name')
                if not isinstance(name, str) or not name:
                    raise OpenAICompatError('Tool call missing function name')
                call_id = raw_call.get('id')
                if not isinstance(call_id, str) or not call_id:
                    call_id = f'call_{idx}'
                arguments = _tool_arguments_or_marker(function_block.get('arguments'))
                tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        elif isinstance(message.get('function_call'), dict):
            function_call = message['function_call']
            name = function_call.get('name')
            if not isinstance(name, str) or not name:
                raise OpenAICompatError('Function call missing name')
            arguments = _tool_arguments_or_marker(function_call.get('arguments'))
            tool_calls.append(ToolCall(id='call_0', name=name, arguments=arguments))
        return tool_calls

    @staticmethod
    def _collect_replay(delta: dict[str, Any], state: dict[str, Any]) -> None:
        """Streamed reasoning_content and each tool call's id and extra_content, by call index."""
        reasoning = delta.get('reasoning_content')
        if isinstance(reasoning, str):
            state.setdefault('reasoning', []).append(reasoning)
        for call in delta.get('tool_calls') or []:
            if not isinstance(call, dict):
                continue
            index = call.get('index') if isinstance(call.get('index'), int) else 0
            if isinstance(call.get('id'), str) and call['id']:
                state.setdefault('call_ids', {})[index] = call['id']
            if isinstance(call.get('extra_content'), dict):
                state.setdefault('call_extras', {})[index] = call['extra_content']

    def _replay_from_stream(self, state: dict[str, Any]) -> dict[str, Any] | None:
        from .thinking_replay import capture

        ids = state.get('call_ids') or {}
        extras = state.get('call_extras') or {}
        message: dict[str, Any] = {'tool_calls': [
            {'id': ids.get(index) or f'call_{index}', 'extra_content': extra} for index, extra in extras.items()]}
        if 'reasoning' in state:
            message['reasoning_content'] = ''.join(state['reasoning'])
        return capture(message, self.config.base_url, self.config.model)

    def _iter_sse_payloads(self, response: Any) -> Iterator[dict[str, Any]]:
        buffer: list[str] = []
        while True:
            line = response.readline()
            if not line:
                break
            if isinstance(line, bytes):
                text = line.decode('utf-8', errors='replace')
            else:
                text = str(line)
            stripped = text.strip()
            if not stripped:
                if not buffer:
                    continue
                joined = '\n'.join(buffer)
                buffer.clear()
                if joined == '[DONE]':
                    break
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f'Invalid JSON in streaming response: {joined!r}'
                    ) from exc
                if not isinstance(payload, dict):
                    raise OpenAICompatError('Malformed SSE payload from model backend')
                yield payload
                continue
            if stripped.startswith('data:'):
                buffer.append(stripped[5:].strip())

        if buffer:
            joined = '\n'.join(buffer)
            if joined != '[DONE]':
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError as exc:
                    raise OpenAICompatError(
                        f'Invalid trailing JSON in streaming response: {joined!r}'
                    ) from exc
                if not isinstance(payload, dict):
                    raise OpenAICompatError('Malformed trailing SSE payload from model backend')
                yield payload

    def _parse_stream_payload(
        self,
        payload: dict[str, Any],
        state: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        state = state if state is not None else {}
        cached = _perf_cached_tokens(payload)
        if cached is not None:
            state['cached'] = cached
        usage = _with_cached_count(_parse_usage(payload.get('usage')), state.get('cached'))
        if usage.total_tokens:
            self._rate_limiter.record_usage(usage.total_tokens)
            from . import explicit_cache

            explicit_cache.observe(self.config.base_url, self.config.model, usage)
            yield StreamEvent(
                type='usage',
                usage=usage,
                raw_event=payload,
            )

        choices = payload.get('choices')
        if not isinstance(choices, list):
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                delta = {}
            self._collect_replay(delta, state)
            content = delta.get('content')
            if isinstance(content, str) and content:
                yield StreamEvent(
                    type='content_delta',
                    delta=content,
                    raw_event=choice,
                )
            tool_calls = delta.get('tool_calls')
            if isinstance(tool_calls, list):
                for raw_tool_call in tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        continue
                    function_block = raw_tool_call.get('function')
                    if not isinstance(function_block, dict):
                        function_block = {}
                    yield StreamEvent(
                        type='tool_call_delta',
                        tool_call_index=(
                            raw_tool_call.get('index')
                            if isinstance(raw_tool_call.get('index'), int)
                            else 0
                        ),
                        tool_call_id=(
                            raw_tool_call.get('id')
                            if isinstance(raw_tool_call.get('id'), str)
                            else None
                        ),
                        tool_name=(
                            function_block.get('name')
                            if isinstance(function_block.get('name'), str)
                            else None
                        ),
                        arguments_delta=(
                            function_block.get('arguments')
                            if isinstance(function_block.get('arguments'), str)
                            else ''
                        ),
                        raw_event=raw_tool_call,
                    )
            finish_reason = choice.get('finish_reason')
            if finish_reason is not None:
                if not isinstance(finish_reason, str):
                    finish_reason = str(finish_reason)
                yield StreamEvent(
                    type='message_stop',
                    finish_reason=finish_reason,
                    raw_event=choice,
                )
