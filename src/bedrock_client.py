"""Amazon Bedrock Converse API client (POST /model/{modelId}/converse and /converse-stream),
standard library only. Used for Claude and Amazon Nova models on a bedrock-runtime base URL
(SIGIL_BEDROCK_CONVERSE=0 keeps the OpenAI-compatible endpoint), because Bedrock caches prompts
only on Converse / InvokeModel (cachePoint blocks) and its docs mark Chat Completions as not
supported for current Claude and Nova models.

- Prompt caching: `{"cachePoint": {"type": "default"}}` after the tool list, the system prompt,
  the end of the previous request and the end of this one (at most 4, as on the native Anthropic
  API). `"ttl": "1h"` on Claude when the TTL is 1h; Nova takes 5 minutes only and no tools cache
  point. A 400 that names cachePoint is retried once without cache points (prompt_cache rules).
  Nova caches at most 20K tokens: no point is placed after more than that (moved back or left out).
- Usage: inputTokens (uncached), cacheReadInputTokens, cacheWriteInputTokens, cacheDetails (1h).
- Auth: a Bedrock API key as `Authorization: Bearer` (AWS_BEARER_TOKEN_BEDROCK, else the configured
  key when it is not the local placeholder), otherwise AWS Signature Version 4 from
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN or the shared credentials file
  (AWS_PROFILE, default "default"). Keys never reach a log, an error or a session file.
- Streaming: the application/vnd.amazon.eventstream frames, both CRCs checked; an exception frame or
  a stream without messageStop raises.
- Thinking (Claude): additionalModelRequestFields.thinking / output_config from the effort level;
  reasoningContent blocks go back unchanged with the tool results.
"""
from __future__ import annotations

import configparser
import datetime
import hashlib
import hmac
import json
import os
import re
import struct
import urllib.parse
import zlib
from pathlib import Path
from typing import Any, Iterator
from urllib import error, request

from .agent_types import AssistantTurn, ModelConfig, OutputSchemaConfig, StreamEvent, ToolCall, UsageStats
from .debug_log import log_ignored_error as _log_ignored

_STOP_REASONS = {'end_turn': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length', 'stop_sequence': 'stop',
                 'model_context_window_exceeded': 'length', 'content_filtered': 'content_filter',
                 'guardrail_intervened': 'content_filter'}
_REGION_RE = re.compile(r'^[a-z]{2}(-[a-z]+)+-\d$|^[a-z0-9-]{3,30}$')
_IMAGE_FORMATS = {'image/png': 'png', 'image/jpeg': 'jpeg', 'image/jpg': 'jpeg', 'image/gif': 'gif',
                  'image/webp': 'webp'}


# ---------------------------------------------------------------- routing

def model_kind(model: str) -> str:
    """'claude', 'nova' or '' for a Bedrock model id or inference profile (us.anthropic.claude-..., amazon.nova-...)."""
    name = (model or '').lower()
    if 'anthropic.' in name or 'claude' in name:
        return 'claude'
    if 'nova' in name:
        return 'nova'
    return ''


_HOST_RE = re.compile(r'^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com$')


def converse_host(base_url: str) -> str | None:
    """The Bedrock runtime host Converse goes to (bedrock-runtime.<region>.amazonaws.com or its -fips
    form), checked against a fixed pattern; None for any other host. Why: the -fips host was accepted for
    the region but the request went to the non-FIPS host, and other amazonaws hosts were planned for
    cache markers that only Converse understands."""
    host = (urllib.parse.urlparse(base_url or '').hostname or '').lower()
    match = _HOST_RE.match(host)
    return host if match and _REGION_RE.match(match.group(1)) else None


def use_bedrock_converse(config: Any) -> bool:
    """Converse for Claude and Nova on a bedrock-runtime(-fips) URL, unless SIGIL_BEDROCK_CONVERSE=0."""
    if os.environ.get('SIGIL_BEDROCK_CONVERSE', '').strip().lower() in {'0', 'false', 'no', 'off'}:
        return False
    return (converse_host(str(getattr(config, 'base_url', '') or '')) is not None
            and model_kind(str(getattr(config, 'model', '') or '')) != '')


def region_of(base_url: str) -> str:
    host = (urllib.parse.urlparse(base_url or '').hostname or '').lower()
    match = re.match(r'^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com$', host)
    region = match.group(1) if match else (os.environ.get('AWS_REGION') or os.environ.get('AWS_DEFAULT_REGION') or '')
    region = region.strip().lower()
    if not region or not _REGION_RE.match(region):
        raise ValueError('no valid AWS region: use a bedrock-runtime.<region>.amazonaws.com base URL or AWS_REGION')
    return region


# ---------------------------------------------------------------- credentials and signing

def _shared_credentials() -> dict[str, str]:
    path = Path(os.environ.get('AWS_SHARED_CREDENTIALS_FILE') or Path.home() / '.aws' / 'credentials').expanduser()
    profile = os.environ.get('AWS_PROFILE', '').strip() or 'default'
    parser = configparser.RawConfigParser()
    try:
        parser.read(path, encoding='utf-8')
    except (OSError, configparser.Error) as exc:
        _log_ignored(exc)
        return {}
    if not parser.has_section(profile):
        return {}
    section = parser[profile]
    return {'key': section.get('aws_access_key_id', '').strip(), 'secret': section.get('aws_secret_access_key', '').strip(),
            'token': section.get('aws_session_token', '').strip()}


def credentials(config: ModelConfig) -> dict[str, str]:
    """{'bearer': token} or {'key', 'secret', 'token'}; registered as secrets, never logged."""
    from .redact import register_secret

    bearer = os.environ.get('AWS_BEARER_TOKEN_BEDROCK', '').strip()
    if bearer:
        register_secret(bearer)
        return {'bearer': bearer}
    key, secret = os.environ.get('AWS_ACCESS_KEY_ID', '').strip(), os.environ.get('AWS_SECRET_ACCESS_KEY', '').strip()
    found = ({'key': key, 'secret': secret, 'token': os.environ.get('AWS_SESSION_TOKEN', '').strip()}
             if key and secret else _shared_credentials())
    if found.get('key') and found.get('secret'):
        for value in (found['secret'], found.get('token', '')):
            if value:
                register_secret(value)
        return found
    configured = str(getattr(config, 'api_key', '') or '').strip()
    if configured and configured != 'local-token':
        register_secret(configured)
        return {'bearer': configured}  # a Bedrock API key in the usual key setting
    raise ValueError('no AWS credentials: set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key) or AWS_ACCESS_KEY_ID / '
                     'AWS_SECRET_ACCESS_KEY (or a profile in ~/.aws/credentials)')


def canonical_path(path: str) -> str:
    """SigV4 canonical URI for services other than S3: the request path (already encoded once)
    encoded again (AWS SDK signers: 'uri_escape_path ... is required for every service, except S3')."""
    return urllib.parse.quote(path, safe='/~')


def sign(method: str, url: str, headers: dict[str, str], body: bytes, creds: dict[str, str], region: str,
         service: str = 'bedrock', now: datetime.datetime | None = None) -> dict[str, str]:
    """Headers with an AWS Signature Version 4 (HMAC-SHA256) Authorization added."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    amz_date, day = now.strftime('%Y%m%dT%H%M%SZ'), now.strftime('%Y%m%d')
    parsed = urllib.parse.urlparse(url)
    # SigV4: header values trimmed with inner runs of spaces (and folded lines) made one space, and
    # query pairs sorted by their encoded form (AWS test suite get-header-value-trim, -multiline,
    # get-vanilla-query-order-encoded)
    signed = {k.lower(): ' '.join(str(v).split()) for k, v in headers.items()}
    signed['host'] = parsed.netloc
    signed['x-amz-date'] = amz_date
    if creds.get('token'):
        signed['x-amz-security-token'] = creds['token']
    names = sorted(signed)
    pairs = sorted((urllib.parse.quote(k, safe='-_.~'), urllib.parse.quote(v, safe='-_.~'))
                   for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query = '&'.join(f'{k}={v}' for k, v in pairs)
    canonical = '\n'.join([method.upper(), canonical_path(parsed.path or '/'), query,
                           ''.join(f'{n}:{signed[n]}\n' for n in names), ';'.join(names),
                           hashlib.sha256(body).hexdigest()])
    scope = f'{day}/{region}/{service}/aws4_request'
    to_sign = '\n'.join(['AWS4-HMAC-SHA256', amz_date, scope, hashlib.sha256(canonical.encode('utf-8')).hexdigest()])
    key = ('AWS4' + creds['secret']).encode('utf-8')
    for part in (day, region, service, 'aws4_request'):
        key = hmac.new(key, part.encode('utf-8'), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
    out = {k: v for k, v in headers.items()}
    out['x-amz-date'] = amz_date
    if creds.get('token'):
        out['x-amz-security-token'] = creds['token']
    out['Authorization'] = (f'AWS4-HMAC-SHA256 Credential={creds["key"]}/{scope}, '
                            f'SignedHeaders={";".join(names)}, Signature={signature}')
    return out


# ---------------------------------------------------------------- messages

def _text(content: Any) -> str:
    from .anthropic_client import _text_of

    return _text_of(content)


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{'text': content or ' '}]
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        if part.get('type') == 'text':
            blocks.append({'text': part.get('text') or ' '})
        elif part.get('type') == 'image_url':
            url = str((part.get('image_url') or {}).get('url', ''))
            if url.startswith('data:') and ';base64,' in url:
                media, data = url[5:].split(';base64,', 1)
                fmt = _IMAGE_FORMATS.get(media.lower())
                if fmt:
                    blocks.append({'image': {'format': fmt, 'source': {'bytes': data}}})
    return blocks or [{'text': ' '}]


def _original(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Converse content kept with an assistant reply (reasoningContent with its signature) when it
    still matches the reply's tool calls: it goes back unchanged."""
    original = message.get('anthropic_content')
    if not isinstance(original, list) or not any(isinstance(b, dict) and 'reasoningContent' in b for b in original):
        return None
    wanted = [str(c.get('id')) for c in message.get('tool_calls') or [] if isinstance(c, dict)]
    have = [str((b.get('toolUse') or {}).get('toolUseId')) for b in original if isinstance(b, dict) and 'toolUse' in b]
    return [dict(b) for b in original if isinstance(b, dict)] if wanted == have else None


def convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    started = False
    for message in messages:
        role = message.get('role')
        if role == 'system':
            text = _text(message.get('content'))
            if not text:
                continue
            if not started:
                system.append({'text': text})
            else:
                converted.append({'role': 'user', 'content': [{'text': f'<system-reminder>\n{text}\n</system-reminder>'}]})
            continue
        started = True
        if role == 'user':
            converted.append({'role': 'user', 'content': _user_blocks(message.get('content'))})
        elif role == 'assistant':
            original = _original(message)
            if original is not None:
                converted.append({'role': 'assistant', 'content': original})
                continue
            blocks: list[dict[str, Any]] = []
            text = _text(message.get('content'))
            if text.strip():
                blocks.append({'text': text})
            for call in message.get('tool_calls') or []:
                function = call.get('function') or {}
                raw = function.get('arguments') or '{}'
                try:
                    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except ValueError as exc:
                    _log_ignored(exc)
                    arguments = {}
                blocks.append({'toolUse': {'toolUseId': str(call.get('id') or 'call'), 'name': function.get('name', ''),
                                           'input': arguments if isinstance(arguments, dict) else {}}})
            converted.append({'role': 'assistant', 'content': blocks or [{'text': ' '}]})
        elif role == 'tool':
            converted.append({'role': 'user', 'content': [{'toolResult': {
                'toolUseId': str(message.get('tool_call_id') or 'call'),
                'content': [{'text': _text(message.get('content')) or ' '}]}}]})
    merged: list[dict[str, Any]] = []
    for item in converted:
        if merged and merged[-1]['role'] == item['role']:
            merged[-1]['content'].extend(item['content'])
        else:
            merged.append({'role': item['role'], 'content': list(item['content'])})
    if not merged or merged[0]['role'] != 'user':
        merged.insert(0, {'role': 'user', 'content': [{'text': '(continue)'}]})
    for item in merged:
        if item['role'] == 'user':
            item['content'].sort(key=lambda block: 0 if 'toolResult' in block else 1)
    return system, merged


def _cacheable(block: Any) -> bool:
    if not isinstance(block, dict) or 'reasoningContent' in block or 'cachePoint' in block:
        return False
    return not ('text' in block and not str(block.get('text') or '').strip())


# "Amazon Nova models support a maximum of 20K tokens for prompt caching" (docs.aws.amazon.com/
# bedrock/latest/userguide/prompt-caching.html, every Nova row; no Claude row has a maximum). AWS does
# not say what happens past it, so a cache point is never placed after more than this.
NOVA_CACHE_MAX_TOKENS = 20_000
_LIMIT_NOTED: set[str] = set()


def estimate_tokens(block: Any) -> int:
    """A high estimate of a content block's tokens (its JSON text / 3): a cache point placed by it
    never sits past a token limit (real tokens are about characters / 4 for text)."""
    try:
        text = json.dumps(block, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        _log_ignored(exc)
        text = str(block)
    return (len(text) + 2) // 3


def _within_token_limit(payload: dict[str, Any], spots: list[tuple[list[Any], int]],
                        max_tokens: int) -> tuple[list[tuple[list[Any], int]], bool]:
    """The spots whose prefix (tools, then system, then messages) stays within max_tokens. A spot
    past it moves back to the last block boundary that fits and is not taken yet, or is left out.
    Returns (spots in prefix order, whether any moved or was left out)."""
    flat: list[tuple[list[Any], int, int, bool]] = []   # (list, index, tokens through it, may hold a point)
    total = 0
    tools = (payload.get('toolConfig') or {}).get('tools')
    for blocks, holds in ((tools, False), (payload.get('system'), True)):
        if isinstance(blocks, list):
            for index, block in enumerate(blocks):
                total += estimate_tokens(block)
                flat.append((blocks, index, total, holds and _cacheable(block)))
    for message in payload.get('messages') or []:
        content = message.get('content')
        if isinstance(content, list):
            for index, block in enumerate(content):
                total += estimate_tokens(block)
                flat.append((content, index, total, _cacheable(block)))
    where = {(id(blocks), index): n for n, (blocks, index, _t, _h) in enumerate(flat)}
    used: list[int] = []
    changed = False
    for blocks, index in spots:
        n = where.get((id(blocks), index))
        if n is None:
            continue
        if flat[n][2] <= max_tokens and n not in used:
            used.append(n)
            continue
        changed = True
        for m in range(n - 1, -1, -1):
            if flat[m][3] and flat[m][2] <= max_tokens and m not in used:
                used.append(m)
                break
    return [(flat[n][0], flat[n][1]) for n in sorted(used)], changed


def add_cache_points(payload: dict[str, Any], point: dict[str, Any], with_tools: bool, limit: int = 4,
                     max_prefix_tokens: int | None = None, label: str = '') -> int:
    """cachePoint after the tool list, the system prompt, the end of the previous request and the
    end of this one (same places as the native Anthropic breakpoints); at most `limit`. With
    max_prefix_tokens (Amazon Nova) no point sits after more tokens than that: it moves back to the
    last block boundary that fits, or is left out (logged once per model)."""
    spots: list[tuple[list[Any], int]] = []   # (list, index to insert after)
    tools = (payload.get('toolConfig') or {}).get('tools')
    if with_tools and isinstance(tools, list) and tools:
        spots.append((tools, len(tools) - 1))
    system = payload.get('system')
    if isinstance(system, list) and system and any(_cacheable(b) for b in system):
        spots.append((system, len(system) - 1))
    messages = payload.get('messages') or []
    last_assistant = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get('role') == 'assistant'), None)
    candidates = ([last_assistant - 1] if last_assistant is not None and last_assistant > 0 else []) + [len(messages) - 1]
    message_spots: list[tuple[list[Any], int]] = []
    for position in candidates:
        while position >= 0:
            content = messages[position].get('content')
            if isinstance(content, list) and any(_cacheable(b) for b in content):
                if not any(c is content for c, _ in message_spots):
                    message_spots.append((content, len(content) - 1))
                break
            position -= 1
    spots = (spots + message_spots)[-limit:]
    if max_prefix_tokens is not None:
        spots, changed = _within_token_limit(payload, spots, max_prefix_tokens)
        if changed and label not in _LIMIT_NOTED and len(_LIMIT_NOTED) < 64:
            _LIMIT_NOTED.add(label)
            from .debug_log import get_logger

            get_logger('bedrock').info('%s caches at most %s tokens of prompt; cache points after that were moved '
                                       'back or left out (logged once)', label or 'this model', f'{max_prefix_tokens:,}')
    for blocks, index in reversed(spots):  # the last first: an earlier insert never shifts a later spot
        blocks.insert(index + 1, {'cachePoint': dict(point)})
    return len(spots)


def strip_cache_points(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(payload))
    for key in ('system',):
        if isinstance(cleaned.get(key), list):
            cleaned[key] = [b for b in cleaned[key] if not (isinstance(b, dict) and 'cachePoint' in b)]
    tools = (cleaned.get('toolConfig') or {}).get('tools')
    if isinstance(tools, list):
        cleaned['toolConfig']['tools'] = [t for t in tools if not (isinstance(t, dict) and 'cachePoint' in t)]
    for message in cleaned.get('messages') or []:
        if isinstance(message.get('content'), list):
            message['content'] = [b for b in message['content'] if not (isinstance(b, dict) and 'cachePoint' in b)]
    return cleaned


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError) as exc:
        _log_ignored(exc)
        return 0


def bedrock_usage(raw: Any) -> UsageStats:
    """Converse usage: inputTokens are the uncached tokens; cacheDetails splits writes by TTL."""
    raw = raw if isinstance(raw, dict) else {}
    written = _int(raw.get('cacheWriteInputTokens'))
    one_hour = sum(_int(d.get('inputTokens')) for d in raw.get('cacheDetails') or []
                   if isinstance(d, dict) and str(d.get('ttl')) == '1h')
    return UsageStats(input_tokens=_int(raw.get('inputTokens')), output_tokens=_int(raw.get('outputTokens')),
                      cache_creation_input_tokens=written, cache_read_input_tokens=_int(raw.get('cacheReadInputTokens')),
                      cache_creation_1h_input_tokens=min(one_hour, written))


# ---------------------------------------------------------------- reasoning and structured output

# Nova 2 effort levels (docs.aws.amazon.com/nova/latest/nova2-userguide/using-converse-api.html):
# reasoningConfig {"type": "enabled", "maxReasoningEffort": low | medium | high}; off by default.
_NOVA_EFFORT = {'low': 'low', 'medium': 'medium', 'high': 'high', 'xhigh': 'high', 'max': 'high'}


def nova_reasoning_effort(model: str, effort: str | None) -> str | None:
    """The Nova 2 maxReasoningEffort for the effort setting, or None (no reasoningConfig: reasoning
    stays off, as for effort off/auto and for Nova models before Nova 2)."""
    from .providers import effective_effort

    if 'nova-2' not in (model or '').lower():
        return None
    return _NOVA_EFFORT.get(str(effective_effort(effort) or '').lower())


def _closed_schema(schema: Any) -> Any:
    """The schema with additionalProperties false on every object that does not set it: Bedrock
    structured output answers 400 for an object without it. An explicit value is left alone."""
    if isinstance(schema, list):
        return [_closed_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out = {key: _closed_schema(value) for key, value in schema.items()}
    kind = out.get('type')
    if kind == 'object' or (isinstance(kind, list) and 'object' in kind):
        out.setdefault('additionalProperties', False)
    return out


def output_config(schema: OutputSchemaConfig) -> dict[str, Any]:
    """Converse outputConfig for a JSON schema (docs.aws.amazon.com/bedrock/latest/userguide/
    structured-output.html): textFormat json_schema, the schema as a JSON string."""
    name = re.sub(r'[^A-Za-z0-9_-]', '_', str(schema.name or ''))[:64] or 'response'
    return {'textFormat': {'type': 'json_schema', 'structure': {'jsonSchema': {
        'schema': json.dumps(_closed_schema(schema.schema)), 'name': name}}}}


_OUTPUT_CONFIG_REFUSED: dict[tuple[str, str], str] = {}   # (base URL, model) -> the error, for this run
_OUTPUT_CONFIG_LOCK = __import__('threading').Lock()
_OUTPUT_CONFIG_ERROR_RE = re.compile(r'outputconfig|textformat|json[ _]?schema|structured output|schema is too complex'
                                     r'|additionalproperties|invalid regex in pattern', re.IGNORECASE)


def _output_config_refused(config: Any) -> bool:
    with _OUTPUT_CONFIG_LOCK:
        return (str(config.base_url), str(config.model)) in _OUTPUT_CONFIG_REFUSED


def _refuse_output_config(config: Any, reason: str) -> None:
    with _OUTPUT_CONFIG_LOCK:
        if len(_OUTPUT_CONFIG_REFUSED) >= 64:                 # bounded: the oldest model is dropped first
            _OUTPUT_CONFIG_REFUSED.pop(next(iter(_OUTPUT_CONFIG_REFUSED)))
        _OUTPUT_CONFIG_REFUSED[(str(config.base_url), str(config.model))] = reason


# ---------------------------------------------------------------- eventstream

def eventstream_frames(response: Any) -> Iterator[tuple[dict[str, Any], bytes]]:
    """(headers, payload) for each application/vnd.amazon.eventstream message; CRCs checked."""
    from .openai_compat import OpenAICompatError

    while True:
        prelude = response.read(12)
        if not prelude:
            return
        if len(prelude) < 12:
            raise OpenAICompatError('Amazon Bedrock stream ended inside a frame')
        total, header_length, prelude_crc = struct.unpack('>III', prelude)
        if zlib.crc32(prelude[:8]) & 0xffffffff != prelude_crc or total < 16 or total > 16 * 1024 * 1024:
            raise OpenAICompatError('Amazon Bedrock stream frame failed its prelude check')
        rest = response.read(total - 12)
        if len(rest) != total - 12:
            raise OpenAICompatError('Amazon Bedrock stream ended inside a frame')
        if zlib.crc32(prelude + rest[:-4]) & 0xffffffff != struct.unpack('>I', rest[-4:])[0]:
            raise OpenAICompatError('Amazon Bedrock stream frame failed its message check')
        if header_length > total - 16:
            raise OpenAICompatError('Amazon Bedrock stream frame has a header length past its end')
        yield _frame_headers(rest[:header_length]), rest[header_length:-4]


def _frame_headers(raw: bytes) -> dict[str, Any]:
    """The headers of one eventstream message. Every length is checked against the bytes there, so a
    malformed frame is a clear error. Why: a length past the end raised IndexError / struct.error, or a
    value was cut short without a word."""
    from .openai_compat import OpenAICompatError

    def take(start: int, size: int) -> bytes:
        if size < 0 or start + size > len(raw):
            raise OpenAICompatError('Amazon Bedrock stream frame has a header that runs past its end')
        return raw[start:start + size]

    headers: dict[str, Any] = {}
    at = 0
    while at < len(raw):
        name_length = take(at, 1)[0]
        name = take(at + 1, name_length).decode('utf-8', errors='replace')
        at += 1 + name_length
        kind = take(at, 1)[0]
        at += 1
        if kind in (0, 1):
            headers[name] = kind == 0
        elif kind in (6, 7):
            length = struct.unpack('>H', take(at, 2))[0]
            value = take(at + 2, length)
            headers[name] = value.decode('utf-8', errors='replace') if kind == 7 else value
            at += 2 + length
        else:
            size = {2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}.get(kind)
            if size is None:
                raise OpenAICompatError('Amazon Bedrock stream frame has an unknown header type')
            headers[name] = take(at, size)
            at += size
    return headers


# ---------------------------------------------------------------- the client

class BedrockConverseClient:
    """Same interface as OpenAICompatClient (complete / stream / pop_original_content)."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.region = region_of(str(config.base_url))
        self.host = converse_host(str(config.base_url)) or f'bedrock-runtime.{self.region}.amazonaws.com'
        self.kind = model_kind(str(config.model))
        self._last_content: list[dict[str, Any]] | None = None
        # reasoningContent blocks Bedrock refused once as bound to another conversation
        self._dropped_thinking: set[str] = set()

    def pop_original_content(self) -> list[dict[str, Any]] | None:
        content, self._last_content = self._last_content, None
        return content

    def build_payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                      output_schema: OutputSchemaConfig | None = None) -> dict[str, Any]:
        from .prompt_cache import plan_for
        from .providers import claude_thinking, effective_effort

        from .thinking_replay import drop_known

        system, converted = convert_messages(messages)
        drop_known(converted, self._dropped_thinking)
        use_output_config = output_schema is not None and not _output_config_refused(self.config)
        if output_schema is not None and not use_output_config:
            system.append({'text': 'Reply with only a JSON object that matches this JSON schema:\n'
                                   + json.dumps(output_schema.schema)})
        try:
            max_tokens = int(getattr(self.config, 'max_tokens', None) or os.environ.get('SIGIL_MAX_TOKENS', '') or 8192)
        except ValueError as exc:
            _log_ignored(exc)
            max_tokens = 8192
        payload: dict[str, Any] = {'messages': converted, 'inferenceConfig': {'maxTokens': max_tokens}}
        if system:
            payload['system'] = system
        if tools:
            specs = []
            for tool in tools:
                function = tool.get('function', tool)
                specs.append({'toolSpec': {'name': function.get('name'), 'description': function.get('description') or ' ',
                                           'inputSchema': {'json': function.get('parameters')
                                                           or {'type': 'object', 'properties': {}}}}})
            payload['toolConfig'] = {'tools': specs}
        plan = plan_for(self.config.base_url, self.config.model)
        if plan.active and plan.markers:
            point: dict[str, Any] = {'type': 'default'}
            if self.kind == 'claude' and plan.ttl in ('1h', '24h'):
                point['ttl'] = '1h'   # Nova takes 5 minutes only ("can result in a ValidationException")
            add_cache_points(payload, point, with_tools=self.kind == 'claude',
                             max_prefix_tokens=NOVA_CACHE_MAX_TOKENS if self.kind == 'nova' else None,
                             label=str(self.config.model))
        if self.kind == 'claude':
            thinking, effort, temperature_ok = claude_thinking(self.config.model,
                                                               effective_effort(getattr(self.config, 'effort', None)))
            extra: dict[str, Any] = {}
            if thinking is not None:
                extra['thinking'] = thinking
                if thinking.get('type') == 'enabled':
                    payload['inferenceConfig']['maxTokens'] = max(max_tokens, int(thinking['budget_tokens']) + 4096)
            if effort is not None:
                extra['output_config'] = {'effort': effort}
            if extra:
                payload['additionalModelRequestFields'] = extra
            if thinking is None and temperature_ok and self.config.temperature is not None:
                payload['inferenceConfig']['temperature'] = self.config.temperature
        else:
            reasoning = nova_reasoning_effort(str(self.config.model), getattr(self.config, 'effort', None))
            if reasoning is not None:
                payload['additionalModelRequestFields'] = {
                    'reasoningConfig': {'type': 'enabled', 'maxReasoningEffort': reasoning}}
            # "Temperature, topP and topK cannot be used with maxReasoningEffort set to high"
            if self.config.temperature is not None and reasoning != 'high':
                payload['inferenceConfig']['temperature'] = self.config.temperature
        if use_output_config and output_schema is not None:
            payload['outputConfig'] = output_config(output_schema)
        return payload

    def _open(self, payload: dict[str, Any], stream: bool) -> Any:
        from .debug_log import sanitize_log
        from .openai_compat import OpenAICompatError, extra_headers

        model = urllib.parse.quote(str(self.config.model), safe='')
        url = f'https://{self.host}/model/{model}/' + ('converse-stream' if stream else 'converse')
        body = json.dumps(payload).encode('utf-8')
        # SIGIL_EXTRA_HEADERS go in before signing, so they are signed with the rest. Why: they were
        # added after SigV4, so one that named a signed header (Content-Type) broke the signature.
        reserved = {'authorization', 'host', 'x-amz-date', 'x-amz-security-token', 'content-length'}
        headers = {k: v for k, v in extra_headers().items() if k.lower() not in reserved}
        for name in [k for k in headers if k.lower() in ('content-type', 'accept')]:
            headers.pop(name)  # the body and the reply format are fixed by the API
        headers.update({'Content-Type': 'application/json', 'Accept': 'application/vnd.amazon.eventstream' if stream
                        else 'application/json'})
        try:
            creds = credentials(self.config)
        except ValueError as exc:
            raise OpenAICompatError(str(exc)) from exc
        if 'bearer' in creds:
            headers['Authorization'] = f'Bearer {creds["bearer"]}'
        else:
            headers = sign('POST', url, headers, body, creds, self.region)
        req = request.Request(url, data=body, method='POST', headers=headers)
        try:
            return request.urlopen(req, timeout=self.config.timeout_seconds)
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            kind = (exc.headers.get('x-amzn-ErrorType') if exc.headers is not None else '') or ''
            try:
                parsed = json.loads(detail)
                message = parsed.get('message') or parsed.get('Message') or detail
                kind = kind or str(parsed.get('__type') or '')
            except ValueError as parse_error:
                _log_ignored(parse_error)
                message = detail
            kind = kind.split(':', 1)[0]
            raise OpenAICompatError(f'HTTP {exc.code} from Amazon Bedrock: {sanitize_log(kind, 80)}: '
                                    f'{sanitize_log(message, 500)}') from exc
        except (error.URLError, OSError) as exc:
            raise OpenAICompatError(f'Unable to reach Amazon Bedrock in {self.region}: {type(exc).__name__}') from exc

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        from .openai_compat import OpenAICompatError

        with self._open(payload, stream=False) as response:
            raw = response.read()
        try:
            data = json.loads(raw.decode('utf-8'))
        except ValueError as exc:
            raise OpenAICompatError('Amazon Bedrock returned invalid JSON') from exc
        if not isinstance(data, dict):
            raise OpenAICompatError('Amazon Bedrock returned a malformed reply')
        return data

    def _without_cache_points(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
        """A 400 that names cachePoint (a model or size the cache does not take): once without them."""
        from .debug_log import get_logger, sanitize_log

        text = str(exc)
        if not text.startswith('HTTP 400') or 'cachepoint' not in text.lower() or strip_cache_points(payload) == payload:
            return None
        get_logger('prompt_cache').info(
            'HTTP 400 about cachePoint from Amazon Bedrock; retrying once without cache points: %s', sanitize_log(text, 200))
        return strip_cache_points(payload)

    def _without_bound_reasoning(self, payload: dict[str, Any], exc: Exception) -> dict[str, Any] | None:
        """A 400 that a replayed Claude reasoning block is bound to a different conversation (something
        before it changed): the named block and every later one left out, then sent once more."""
        from . import thinking_replay

        position = thinking_replay.signature_error(str(exc))
        if position is None:
            return None
        stripped, removed = thinking_replay.strip_from(payload, position)
        if not removed:
            return None
        self._dropped_thinking |= removed
        thinking_replay.note('strip', self.config.model)
        return stripped

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
                 output_schema: OutputSchemaConfig | None = None) -> AssistantTurn:
        from . import prompt_cache
        from .openai_compat import OpenAICompatError

        payload = self.build_payload(messages, tools, output_schema)
        for attempt in range(2):  # a second time only after leaving out bound reasoning
            try:
                return self._turn(self._post(payload))
            except OpenAICompatError as exc:
                unbound = self._without_bound_reasoning(payload, exc) if attempt == 0 else None
                if unbound is not None:
                    payload = unbound
                    continue
                stripped = self._without_cache_points(payload, exc)
                if stripped is not None:
                    data = self._post(stripped)
                    prompt_cache.remember_rejected(self.config.base_url, self.config.model, str(exc))
                    return self._turn(data)
                plain = self._without_output_config(payload, exc, messages, tools, output_schema)
                if plain is None:
                    raise
                return self._turn(self._post(plain))
        raise AssertionError('unreachable')

    def _without_output_config(self, payload: dict[str, Any], exc: Exception, messages: list[dict[str, Any]],
                               tools: list[dict[str, Any]], output_schema: OutputSchemaConfig | None
                               ) -> dict[str, Any] | None:
        """A 400 about outputConfig (a model without structured output, a schema it does not take):
        once more with the schema as a text instruction, and so for this model for the rest of the run."""
        from .debug_log import get_logger, sanitize_log

        text = str(exc)
        if ('outputConfig' not in payload or not text.startswith('HTTP 400')
                or _OUTPUT_CONFIG_ERROR_RE.search(text) is None):
            return None
        _refuse_output_config(self.config, text)
        get_logger('bedrock').info('HTTP 400 about structured output from Amazon Bedrock; the schema goes as an '
                                   'instruction for %s from now on: %s', self.config.model, sanitize_log(text, 200))
        return self.build_payload(messages, tools, output_schema)

    def _turn(self, data: dict[str, Any]) -> AssistantTurn:
        content = ((data.get('output') or {}).get('message') or {}).get('content') or []
        texts, thinking, calls = [], [], []
        for block in content:
            if not isinstance(block, dict):
                continue
            if 'text' in block:
                texts.append(str(block.get('text') or ''))
            elif 'toolUse' in block:
                use = block['toolUse']
                if not isinstance(use, dict):
                    _log_ignored(ValueError('Bedrock toolUse block is not an object'))
                    continue
                arguments = use.get('input')  # an input that is not an object is sent as no arguments
                calls.append(ToolCall(id=str(use.get('toolUseId') or f'call_{len(calls)}'), name=str(use.get('name')),
                                      arguments=dict(arguments) if isinstance(arguments, dict) else {}))
            elif 'reasoningContent' in block:
                thinking.append(str(((block['reasoningContent'] or {}).get('reasoningText') or {}).get('text') or ''))
        blocks = [dict(b) for b in content if isinstance(b, dict)]
        self._last_content = blocks if any('reasoningContent' in b for b in blocks) else None
        raw_message: dict[str, Any] = {'role': 'assistant', 'content': ''.join(texts)}
        if thinking:
            raw_message['reasoning_content'] = '\n\n'.join(thinking)
        if self._last_content is not None:
            raw_message['anthropic_content'] = self._last_content
        stop = data.get('stopReason')
        return AssistantTurn(content=''.join(texts), tool_calls=tuple(calls),
                             finish_reason=_STOP_REASONS.get(str(stop), stop), raw_message=raw_message,
                             usage=bedrock_usage(data.get('usage')))

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
               output_schema: OutputSchemaConfig | None = None) -> Iterator[StreamEvent]:
        from . import prompt_cache
        from .openai_compat import OpenAICompatError

        payload = self.build_payload(messages, tools, output_schema)
        for attempt in range(2):  # a second time only after leaving out bound reasoning
            started = False
            try:
                for event in self._stream_payload(payload):
                    started = True
                    yield event
                return
            except OpenAICompatError as exc:
                if started:
                    raise
                unbound = self._without_bound_reasoning(payload, exc) if attempt == 0 else None
                if unbound is not None:
                    payload = unbound
                    continue
                stripped = self._without_cache_points(payload, exc)
                if stripped is None:
                    plain = self._without_output_config(payload, exc, messages, tools, output_schema)
                    if plain is None:
                        raise
                    yield from self._stream_payload(plain)
                    return
                first = True
                for event in self._stream_payload(stripped):
                    if first:
                        first = False
                        prompt_cache.remember_rejected(self.config.base_url, self.config.model, str(exc))
                    yield event
                return

    def _stream_payload(self, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        from .debug_log import sanitize_log
        from .openai_compat import OpenAICompatError

        self._last_content = None
        blocks: dict[int, dict[str, Any]] = {}
        tool_slots: dict[int, int] = {}
        pieces: dict[int, list[str]] = {}
        usage: dict[str, Any] = {}
        stop: Any = None
        stopped = False
        with self._open(payload, stream=True) as response:
            for headers, body in eventstream_frames(response):
                try:
                    data = json.loads(body.decode('utf-8')) if body else {}
                except ValueError as exc:
                    raise OpenAICompatError('Amazon Bedrock sent a malformed stream event') from exc
                if headers.get(':message-type') == 'exception' or headers.get(':message-type') == 'error':
                    kind = headers.get(':exception-type') or headers.get(':error-code') or 'error'
                    raise OpenAICompatError(f'Amazon Bedrock stream error: {sanitize_log(kind, 60)}: '
                                            f'{sanitize_log((data or {}).get("message"), 300)}')
                event = headers.get(':event-type')
                index = int(data.get('contentBlockIndex') or 0) if isinstance(data, dict) else 0
                if event == 'contentBlockStart':
                    start = (data.get('start') or {})
                    if 'toolUse' in start:
                        use = start['toolUse'] or {}
                        slot = len(tool_slots)
                        tool_slots[index] = slot
                        pieces[index] = []
                        blocks[index] = {'toolUse': {'toolUseId': str(use.get('toolUseId') or f'call_{slot}'),
                                                     'name': str(use.get('name') or ''), 'input': {}}}
                        yield StreamEvent(type='tool_call_delta', tool_call_index=slot,
                                          tool_call_id=blocks[index]['toolUse']['toolUseId'],
                                          tool_name=blocks[index]['toolUse']['name'], arguments_delta='')
                elif event == 'contentBlockDelta':
                    delta = data.get('delta') or {}
                    if 'text' in delta:
                        text = str(delta.get('text') or '')
                        block = blocks.setdefault(index, {'text': ''})
                        block['text'] = str(block.get('text') or '') + text
                        if text:
                            yield StreamEvent(type='content_delta', delta=text)
                    elif 'toolUse' in delta:
                        piece = str((delta.get('toolUse') or {}).get('input') or '')
                        pieces.setdefault(index, []).append(piece)
                        if piece:
                            yield StreamEvent(type='tool_call_delta', tool_call_index=tool_slots.get(index, 0),
                                              arguments_delta=piece)
                    elif 'reasoningContent' in delta:
                        part = delta.get('reasoningContent') or {}
                        block = blocks.setdefault(index, {'reasoningContent': {'reasoningText': {'text': ''}}})
                        reasoning = block.setdefault('reasoningContent', {})
                        if 'redactedContent' in part:
                            reasoning['redactedContent'] = part['redactedContent']
                        else:
                            text_block = reasoning.setdefault('reasoningText', {'text': ''})
                            if 'text' in part:
                                text_block['text'] = str(text_block.get('text') or '') + str(part.get('text') or '')
                            if 'signature' in part:
                                text_block['signature'] = str(text_block.get('signature') or '') + str(part['signature'])
                elif event == 'contentBlockStop':
                    block = blocks.get(index) or {}
                    if 'toolUse' in block:
                        raw = ''.join(pieces.get(index) or [])
                        try:
                            block['toolUse']['input'] = json.loads(raw) if raw.strip() else {}
                        except ValueError as exc:
                            _log_ignored(exc)
                            block['toolUse']['input'] = {}
                elif event == 'messageStop':
                    stopped = True
                    stop = data.get('stopReason')
                elif event == 'metadata':
                    usage = data.get('usage') or {}
        if not stopped:
            raise OpenAICompatError('The Amazon Bedrock stream ended before messageStop')
        ordered = [blocks[i] for i in sorted(blocks)]
        if any('reasoningContent' in b for b in ordered):
            self._last_content = [dict(b) for b in ordered]
        yield StreamEvent(type='usage', usage=bedrock_usage(usage))
        yield StreamEvent(type='message_stop', finish_reason=_STOP_REASONS.get(str(stop), stop))


def encode_frame(headers: dict[str, str], payload: bytes) -> bytes:
    """One eventstream message (string headers); used by tests and kept beside the decoder."""
    raw = b''
    for name, value in headers.items():
        name_b, value_b = name.encode('utf-8'), str(value).encode('utf-8')
        raw += bytes([len(name_b)]) + name_b + bytes([7]) + struct.pack('>H', len(value_b)) + value_b
    total = 12 + len(raw) + len(payload) + 4
    prelude = struct.pack('>II', total, len(raw))
    prelude += struct.pack('>I', zlib.crc32(prelude) & 0xffffffff)
    message = prelude + raw + payload
    return message + struct.pack('>I', zlib.crc32(message) & 0xffffffff)
