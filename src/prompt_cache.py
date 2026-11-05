"""Prompt caching for every model call: what to send, per provider, and how long the cache lives.

On by default. Each request goes through `plan_for(base_url, model)`, which picks the best way the
provider behind the URL caches a repeated prompt prefix:

- markers: `cache_control` breakpoints on content blocks (Anthropic, OpenRouter for Claude and
  Gemini, Alibaba DashScope, NanoGPT for Claude). At most 4 per request: the system prompt (which
  also covers the tools before it), the end of the previous request and the end of this one.
- hints: request fields or headers that keep a conversation on the server that holds its cache
  (`prompt_cache_key` for OpenAI, Mistral, Together and servers with no documented scheme;
  `x-grok-conv-id` for xAI; `x-session-affinity` for Fireworks; `x-session-id` for OpenRouter;
  `prompt_cache_options` for Kimi's 1-hour TTL; `prompt_cache_retention` for OpenAI's long TTL;
  `keep_alive` for Ollama).
- nothing extra where caching is automatic (DeepSeek, Gemini, Groq, Z.ai, vLLM, llama.cpp); the
  usage is still read and shown.

Settings (later wins): the `promptCache` setting ({"mode": ..., "ttl": ...}), then the environment
(SIGIL_PROMPT_CACHE = auto | on | off, SIGIL_PROMPT_CACHE_TTL = 5m | 1h | 24h), then `/cache` for
this session. `off` sends no cache field, marker or header at all. `on` also puts markers on
servers that are not known to take them. SIGIL_PROMPT_CACHE_PROVIDER names the provider of a server
Sigil does not recognise by its host (a local server, a proxy or a custom domain).

A cache field never breaks a request: a 400 or 422 answer that names a cache field we sent (or says
an unknown field was sent, without naming another one) is retried once without the cache fields,
and when that works the endpoint is remembered for the session. Any other 400/422 is not retried.
Both decisions are logged. The cache key is a hash of the model and the system prompt, never a
path, user name or key.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

from .debug_log import get_logger, sanitize_log

MODES = ('auto', 'on', 'off')
TTLS = ('5m', '1h', '24h')
DEFAULT_MODE = 'auto'
DEFAULT_TTL = '1h'  # user decision 2026-11-04; SIGIL_PROMPT_CACHE_TTL=5m for the short cache
TTL_SECONDS = {'5m': 300, '1h': 3600, '24h': 86400}
MAX_BREAKPOINTS = 4

_MODE_ALIASES = {'1': 'on', 'true': 'on', 'yes': 'on', 'on': 'on', 'force': 'on',
                 '0': 'off', 'false': 'off', 'no': 'off', 'off': 'off', 'none': 'off',
                 'auto': 'auto', 'default': 'auto', '': 'auto'}
_TTL_ALIASES = {'5m': '5m', '5min': '5m', '300': '5m', '300s': '5m', 'short': '5m', 'default': '1h',
                '1h': '1h', '60m': '1h', '3600': '1h', '3600s': '1h', 'long': '1h', 'hour': '1h',
                '24h': '24h', '1d': '24h', 'day': '24h', '86400': '24h', '86400s': '24h'}
TTL_WORDS = frozenset(_TTL_ALIASES)

_LOCK = threading.Lock()
_SESSION: dict[str, str] = {}            # /cache overrides for this process (mode, ttl)
_REJECTED: 'OrderedDict[str, str]' = OrderedDict()  # endpoint -> why its cache fields were dropped
_REJECTED_MAX = 64
_WARNED: set[str] = set()
_WARNED_MAX = 64


# ---------------------------------------------------------------- settings

def _warn_once(key: str, message: str) -> None:
    """An invalid value is never dropped silently: one line on stderr and in the log."""
    with _LOCK:
        if key in _WARNED:
            return
        if len(_WARNED) < _WARNED_MAX:
            _WARNED.add(key)
    get_logger('prompt_cache').warning(message)
    try:
        print(f'warning: {message}', file=sys.stderr)
    except (OSError, ValueError) as exc:  # stderr closed (a detached worker): the log line above stays
        from .debug_log import log_ignored_error

        log_ignored_error(exc)


def parse_mode(value: Any, source: str = 'value') -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    mode = _MODE_ALIASES.get(text)
    if mode is None:
        _warn_once(f'mode:{source}:{text}', f'{source}={sanitize_log(value, 40)} is not a prompt cache mode '
                   f'(auto, on, off); using the next setting or "{DEFAULT_MODE}".')
    return mode


def parse_ttl(value: Any, source: str = 'value') -> str | None:
    if value is None or str(value).strip() == '':
        return None
    text = str(value).strip().lower()
    ttl = _TTL_ALIASES.get(text)
    if ttl is None:
        _warn_once(f'ttl:{source}:{text}', f'{source}={sanitize_log(value, 40)} is not a prompt cache TTL '
                   f'(5m, 1h, 24h); using the next setting or "{DEFAULT_TTL}".')
    return ttl


def _settings_value(cwd: Any = None) -> dict[str, Any]:
    try:
        from .settings import load_settings, project_dir

        value = load_settings(cwd if cwd is not None else project_dir()).get('promptCache')
    except Exception as exc:  # broken settings are reported by settings.py itself
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
        return {}
    if isinstance(value, str):
        return {'mode': value}
    if isinstance(value, bool):
        return {'mode': 'on' if value else 'off'}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class CacheSettings:
    mode: str = DEFAULT_MODE
    ttl: str = DEFAULT_TTL
    mode_source: str = 'default'
    ttl_source: str = 'default'


def current_settings(cwd: Any = None) -> CacheSettings:
    """Effective mode and TTL: /cache for this session, then the environment, then settings."""
    with _LOCK:
        session = dict(_SESSION)
    mode, mode_source = None, 'default'
    ttl, ttl_source = None, 'default'
    if session.get('mode'):
        mode, mode_source = session['mode'], '/cache'
    if session.get('ttl'):
        ttl, ttl_source = session['ttl'], '/cache'
    if mode is None and os.environ.get('SIGIL_PROMPT_CACHE', '').strip():
        mode = parse_mode(os.environ['SIGIL_PROMPT_CACHE'], 'SIGIL_PROMPT_CACHE')
        mode_source = 'SIGIL_PROMPT_CACHE' if mode else mode_source
    if ttl is None and os.environ.get('SIGIL_PROMPT_CACHE_TTL', '').strip():
        ttl = parse_ttl(os.environ['SIGIL_PROMPT_CACHE_TTL'], 'SIGIL_PROMPT_CACHE_TTL')
        ttl_source = 'SIGIL_PROMPT_CACHE_TTL' if ttl else ttl_source
    if mode is None or ttl is None:
        configured = _settings_value(cwd)
        if mode is None and configured.get('mode') is not None:
            mode = parse_mode(configured.get('mode'), 'promptCache.mode')
            mode_source = 'settings' if mode else mode_source
        if ttl is None and configured.get('ttl') is not None:
            ttl = parse_ttl(configured.get('ttl'), 'promptCache.ttl')
            ttl_source = 'settings' if ttl else ttl_source
    return CacheSettings(mode or DEFAULT_MODE, ttl or DEFAULT_TTL, mode_source, ttl_source)


def set_session(*, mode: str | None = None, ttl: str | None = None, clear: bool = False) -> None:
    """/cache: change the mode or TTL for this process (forwarded to background agents)."""
    with _LOCK:
        if clear:
            _SESSION.clear()
            _REJECTED.clear()
        if mode is not None:
            _SESSION['mode'] = mode
        if ttl is not None:
            _SESSION['ttl'] = ttl


def session_overrides() -> dict[str, str]:
    with _LOCK:
        return dict(_SESSION)


def forward_env(env: dict[str, str]) -> dict[str, str]:
    """Put this session's /cache choices into a background agent's environment. Only values that
    are set go in; nothing is ever written as the text "None"."""
    overrides = session_overrides()
    if overrides.get('mode') in MODES:
        env['SIGIL_PROMPT_CACHE'] = overrides['mode']
    if overrides.get('ttl') in TTLS:
        env['SIGIL_PROMPT_CACHE_TTL'] = overrides['ttl']
    return env


# ---------------------------------------------------------------- provider detection

def _host_is(host: str, *domains: str) -> bool:
    """host equals a domain or is a subdomain of it (never a substring match: evilanthropic.com)."""
    return any(host == d or host.endswith('.' + d) for d in domains)


def provider_of(base_url: str, model: str = '') -> str:
    """The server behind a base URL, by host (exact or subdomain) and port."""
    parsed = urllib.parse.urlparse(base_url or '')
    host = (parsed.hostname or '').lower()
    try:
        port = parsed.port
    except ValueError as exc:  # a malformed port: detection falls back to the host
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
        port = None
    table = (
        (('anthropic.com',), 'anthropic'),
        (('openrouter.ai',), 'openrouter'),
        (('openai.azure.com', 'cognitiveservices.azure.com', 'services.ai.azure.com'), 'azure'),
        (('openai.com',), 'openai'),
        (('deepseek.com',), 'deepseek'),
        (('x.ai',), 'xai'),
        (('aliyuncs.com',), 'dashscope'),
        (('moonshot.ai', 'moonshot.cn', 'kimi.ai', 'kimi.com'), 'moonshot'),
        (('z.ai', 'bigmodel.cn'), 'zai'),
        (('mistral.ai',), 'mistral'),
        (('groq.com',), 'groq'),
        (('fireworks.ai',), 'fireworks'),
        (('together.xyz', 'together.ai'), 'together'),
        (('nano-gpt.com',), 'nanogpt'),
        (('googleapis.com',), 'gemini'),
        (('chutes.ai',), 'chutes'),
        (('cerebras.ai',), 'cerebras'),
        (('deepinfra.com',), 'deepinfra'),
        (('sambanova.ai',), 'sambanova'),
        (('amazonaws.com',), 'bedrock'),
    )
    for domains, name in table:
        if _host_is(host, *domains):
            if name == 'gemini' and 'aiplatform' in host:
                return 'vertex'
            return name
    if 'ollama' in host or port == 11434:
        return 'ollama'
    if 'lmstudio' in host or port == 1234:
        return 'lmstudio'
    if 'llama' in host or port == 8080:
        return 'llamacpp'
    if 'vllm' in host or port == 8000:
        return 'vllm'
    return 'openai-compatible'


PROVIDERS = ('anthropic', 'openrouter', 'azure', 'openai', 'deepseek', 'xai', 'dashscope', 'moonshot', 'zai',
             'mistral', 'groq', 'fireworks', 'together', 'nanogpt', 'gemini', 'vertex', 'chutes', 'bedrock',
             'cerebras', 'deepinfra', 'sambanova',
             'ollama', 'lmstudio', 'llamacpp', 'vllm', 'openai-compatible')
# Found by port or name only, or not found at all: SIGIL_PROMPT_CACHE_PROVIDER may name these.
_NOT_BY_HOST = frozenset(('ollama', 'lmstudio', 'llamacpp', 'vllm', 'openai-compatible'))


def cache_provider(base_url: str, model: str = '') -> str:
    """The provider whose cache rules apply to base_url. SIGIL_PROMPT_CACHE_PROVIDER overrides the
    guess for a server not recognised by its host (a known host such as api.openai.com keeps its
    own rules); an unknown value is reported once and ignored. Only the cache plan and the cache
    prices use this; which client talks to the server is decided by provider_of alone."""
    detected = provider_of(base_url, model)
    override = os.environ.get('SIGIL_PROMPT_CACHE_PROVIDER', '').strip().lower()
    if not override or detected not in _NOT_BY_HOST:
        return detected
    if override not in PROVIDERS:
        _warn_once(f'provider:{override}', f'SIGIL_PROMPT_CACHE_PROVIDER={sanitize_log(override, 40)} is not a '
                   f'known provider ({", ".join(PROVIDERS)}); detecting it from the URL instead.')
        return detected
    return override


def _model_family(model: str) -> str:
    lowered = (model or '').lower()
    name = lowered.rsplit('/', 1)[-1]
    if lowered.startswith('anthropic/') or 'claude' in lowered:
        return 'claude'
    if lowered.startswith('google/') or 'gemini' in lowered:
        return 'gemini'
    if lowered.startswith('openai/') or re.match(r'(gpt-|o\d)', name):
        return 'openai'
    if 'qwen' in lowered:
        return 'qwen'
    if 'kimi' in lowered:
        return 'kimi'
    return ''


# ---------------------------------------------------------------- the plan for one request

@dataclass(frozen=True)
class CachePlan:
    provider: str
    mode: str
    ttl: str
    markers: bool = False           # cache_control on content blocks
    marker_ttl: bool = False        # the marker carries "ttl": "1h" when the TTL is 1h
    key_field: str | None = None    # body field for the hashed cache key
    key_headers: tuple[str, ...] = ()
    body: dict[str, Any] = field(default_factory=dict)  # other body hints
    note: str = ''

    @property
    def active(self) -> bool:
        return self.mode != 'off'

    def marker(self) -> dict[str, str]:
        if self.marker_ttl and self.ttl in ('1h', '24h'):
            return {'type': 'ephemeral', 'ttl': '1h'}
        return {'type': 'ephemeral'}


# The models OpenAI lists for extended retention, exactly (developers.openai.com/api/docs/guides/
# prompt-caching, checked 2026-11-05). Not a prefix list: gpt-5-mini, gpt-5-nano, gpt-5.3 and
# gpt-4.1-mini are not on it, and a model that rejects the field loses every cache field for the
# session (the 400 retry), prompt_cache_key included. A dated snapshot of a listed model counts.
_OPENAI_EXTENDED_RETENTION = frozenset((
    'gpt-5.5', 'gpt-5.5-pro', 'gpt-5.4', 'gpt-5.2', 'gpt-5.1-codex-max', 'gpt-5.1', 'gpt-5.1-codex',
    'gpt-5.1-codex-mini', 'gpt-5.1-chat-latest', 'gpt-5', 'gpt-5-codex', 'gpt-4.1'))


def _openai_retention(model: str, ttl: str) -> dict[str, Any]:
    """prompt_cache_retention="24h" for a long TTL on the models OpenAI lists for extended retention;
    nothing for any other model (gpt-5.6+ use prompt_cache_options instead)."""
    name = (model or '').lower().rsplit('/', 1)[-1]
    name = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', name)  # gpt-4.1-2025-04-14 -> gpt-4.1
    if ttl in ('1h', '24h') and name in _OPENAI_EXTENDED_RETENTION:
        return {'prompt_cache_retention': '24h'}
    return {}


def plan_for(base_url: str, model: str = '', *, native: bool = False, cwd: Any = None) -> CachePlan:
    """What to send for one request to base_url/model. Thread-safe; reads the settings each time."""
    settings = current_settings(cwd)
    provider = 'anthropic' if native else cache_provider(base_url, model)
    family = _model_family(model)
    base = CachePlan(provider=provider, mode=settings.mode, ttl=settings.ttl)
    if settings.mode == 'off':
        return replace(base, note='prompt caching is off')
    rejected = rejected_reason(base_url, model)
    if rejected:
        return replace(base, mode='off', note=f'cache fields dropped for this endpoint: {rejected}')
    if provider == 'anthropic':
        if native:
            return replace(base, markers=True, marker_ttl=True, note='cache_control breakpoints (native API)')
        # The OpenAI-compatible endpoint does not cache (A13); make_client picks the native client.
        return replace(base, note="Anthropic's OpenAI-compatible endpoint does not cache; "
                                  'use the native API (default, SIGIL_ANTHROPIC_NATIVE=0 opts out)')
    if provider == 'openrouter':
        plan = replace(base, key_headers=('x-session-id',))
        if family == 'claude':
            return replace(plan, markers=True, marker_ttl=True, note='cache_control on content blocks (Claude)')
        if family == 'gemini':
            return replace(plan, markers=True, note='cache_control on content blocks (Gemini)')
        return replace(plan, note='automatic caching upstream; sticky routing by x-session-id')
    if provider == 'dashscope':
        return replace(base, markers=True, note='explicit cache_control on content blocks (5-minute cache)')
    if provider == 'nanogpt':
        if family == 'claude':
            return replace(base, markers=True, marker_ttl=True, note='cache_control on content blocks (Claude)')
        return replace(base, note='automatic caching upstream')
    if provider in ('openai', 'azure'):
        if provider == 'azure' and not re.search(r'gpt-(5\.[6-9]|[6-9])', (model or '').lower()):
            return replace(base, note='automatic caching (Azure documents prompt_cache_key for gpt-5.6+ only)')
        return replace(base, key_field='prompt_cache_key', body=_openai_retention(model, settings.ttl),
                       note='automatic caching; prompt_cache_key keeps a conversation together')
    if provider in ('mistral', 'together', 'cerebras', 'deepinfra'):
        # Cerebras (inference-docs.cerebras.ai/capabilities/prompt-caching) and DeepInfra
        # (deepinfra.com/blog/prompt-cache-retention) document prompt_cache_key too
        return replace(base, key_field='prompt_cache_key', note='automatic caching; prompt_cache_key')
    if provider == 'xai':
        return replace(base, key_headers=('x-grok-conv-id',), note='automatic caching; x-grok-conv-id header')
    if provider == 'fireworks':
        return replace(base, key_headers=('x-session-affinity',), note='automatic caching; x-session-affinity')
    if provider == 'moonshot':
        body: dict[str, Any] = {}
        if settings.ttl in ('1h', '24h') and 'kimi-k3' in (model or '').lower():
            body = {'prompt_cache_options': {'mode': 'implicit', 'ttl': '1h'}}
        # prompt_cache_key is in Kimi's chat schema (platform.kimi.ai/docs/api/chat)
        return replace(base, key_field='prompt_cache_key', body=body,
                       note='automatic caching; prompt_cache_key' + ('; 1h TTL' if body else ''))
    if provider == 'ollama':
        body = {} if settings.ttl == '5m' else {'keep_alive': settings.ttl}
        return replace(base, body=body, note='KV cache kept while the model stays loaded (keep_alive)')
    if provider in ('deepseek', 'gemini', 'vertex', 'groq', 'zai', 'vllm', 'llamacpp', 'sambanova'):
        note = ('implicit caching; explicit cachedContents when the price rule says so (explicit_cache.py)'
                if provider in ('gemini', 'vertex') else 'automatic caching; usage is read')
        plan = replace(base, note=note)
        if settings.mode == 'on':
            return replace(plan, markers=True)
        return plan
    if provider == 'bedrock':
        from types import SimpleNamespace

        from .bedrock_client import model_kind, use_bedrock_converse

        kind = model_kind(model)
        # only where Converse is really used (bedrock-runtime[-fips] host); another amazonaws host goes
        # to the OpenAI-compatible endpoint, which does not take cache markers
        if kind and use_bedrock_converse(SimpleNamespace(base_url=base_url, model=model)):
            # docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html: cachePoint on Converse;
            # 1h only for Claude, Nova 5 minutes and no tools cache point (bedrock_client.py)
            return replace(base, markers=True, marker_ttl=kind == 'claude',
                           note='cachePoint blocks on the Converse API (Claude: tools, system, messages; Nova: '
                                'system, messages)')
    # chutes, lmstudio, bedrock and anything unknown: nothing is documented, so the OpenAI approach
    # (user decision 2026-11-04); a 400/422 drops it for the session.
    plan = replace(base, key_field='prompt_cache_key', note='no documented scheme: OpenAI-style prompt_cache_key')
    if settings.mode == 'on':
        return replace(plan, markers=True)
    return plan


def cache_lifetime(base_url: str, model: str = '', *, native: bool = False, cwd: Any = None) -> tuple[int | None, str]:
    """About how long an unused cached prefix lasts at this endpoint, in seconds, and how to say it; None
    when the provider does not publish a lifetime. Used to time /loop runs (scheduler.py), never sent.
    Sources, checked 2026-11-05: Anthropic and Bedrock 5 minutes or 1 hour with the ttl marker; OpenAI
    "5 to 10 minutes of inactivity, up to a maximum of one hour", 24 hours with extended retention
    (developers.openai.com/api/docs/guides/prompt-caching); DeepSeek "a few hours to a few days", best
    effort (api-docs.deepseek.com/guides/kv_cache); Gemini implicit at most 24 hours, most evicted sooner
    (docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview)."""
    plan = plan_for(base_url, model, native=native, cwd=cwd)
    if not plan.active:
        return None, 'no time at all (prompt caching is off)'
    long_ttl = plan.ttl in ('1h', '24h')
    if plan.markers and plan.marker_ttl:  # Anthropic-style cache_control / Bedrock Claude cachePoint
        return (3600, '1 hour') if long_ttl else (300, '5 minutes')
    if plan.provider == 'bedrock' and plan.markers:
        return 300, '5 minutes (Amazon Nova)'
    if plan.provider == 'dashscope':
        return 300, '5 minutes'
    if plan.provider in ('openai', 'azure'):
        if plan.body.get('prompt_cache_retention') == '24h':
            return 86400, 'up to 24 hours (extended retention)'
        return 300, '5 to 10 minutes without use'
    if plan.provider == 'moonshot' and plan.body.get('prompt_cache_options'):
        return 3600, '1 hour'
    if plan.provider == 'deepseek':
        return None, 'a few hours to a few days, best effort'
    if plan.provider in ('gemini', 'vertex'):
        return None, 'up to 24 hours, often less'
    return None, 'an unpublished time'


# ---------------------------------------------------------------- the cache key

def cache_key(model: str, messages: list[Any]) -> str:
    """A stable key for one conversation prefix: a hash of the model and the first system message
    (what the provider receives anyway). Never a path, user name or secret."""
    system = ''
    for message in messages or ():
        if isinstance(message, dict) and message.get('role') == 'system':
            content = message.get('content')
            system = content if isinstance(content, str) else repr(content)
            break
    digest = hashlib.sha256(f'{model}\n{system}'.encode('utf-8', errors='replace')).hexdigest()
    return f'sigil-{digest[:32]}'


def apply_hints(plan: CachePlan, payload: dict[str, Any], headers: dict[str, str]) -> bool:
    """Add the plan's body hints and headers. Returns whether anything was added."""
    if not plan.active:
        return False
    added = False
    key = None
    if plan.key_field or plan.key_headers:
        key = cache_key(str(payload.get('model', '')), payload.get('messages') or [])
    if plan.key_field and plan.key_field not in payload:
        payload[plan.key_field] = key
        added = True
    for name in plan.key_headers:
        headers.setdefault(name, key)  # headers are never rejected as unknown fields
    for name, value in plan.body.items():
        if name not in payload:
            payload[name] = value
            added = True
    return added


# ---------------------------------------------------------------- breakpoints (OpenAI message format)

def _markable_part(content: Any) -> int | None:
    """Index of the last non-empty text part in a content list, or None."""
    if not isinstance(content, list):
        return None
    for index in range(len(content) - 1, -1, -1):
        part = content[index]
        if isinstance(part, dict) and part.get('type') == 'text' and str(part.get('text') or '').strip():
            return index
    return None


def _can_mark(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    content = message.get('content')
    if isinstance(content, str):
        return bool(content.strip())
    return _markable_part(content) is not None


def _mark(message: dict[str, Any], marker: dict[str, str]) -> dict[str, Any]:
    content = message.get('content')
    if isinstance(content, str):
        parts = [{'type': 'text', 'text': content, 'cache_control': dict(marker)}]
    else:
        parts = [dict(p) if isinstance(p, dict) else p for p in content]
        index = _markable_part(parts)
        parts[index] = dict(parts[index], cache_control=dict(marker))
    return dict(message, content=parts)


def breakpoint_indexes(messages: list[Any], limit: int = MAX_BREAKPOINTS) -> list[int]:
    """Which messages get a breakpoint: the last system message before the conversation (it also
    covers the tools before it), the end of the previous request (the message before the newest
    assistant reply) and the end of this one. Messages with nothing to mark are skipped (the
    nearest earlier markable message is used); never more than `limit`, never duplicates."""
    def nearest(index: int) -> int | None:
        while index >= 0:
            if _can_mark(messages[index]):
                return index
            index -= 1
        return None

    chosen: list[int] = []
    system_end = None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get('role') == 'system':
            system_end = index
            continue
        break
    if system_end is not None and _can_mark(messages[system_end]):
        chosen.append(system_end)
    last_assistant = next((i for i in range(len(messages) - 1, -1, -1)
                           if isinstance(messages[i], dict) and messages[i].get('role') == 'assistant'), None)
    if last_assistant is not None and last_assistant > 0:
        previous = nearest(last_assistant - 1)
        if previous is not None and previous not in chosen:
            chosen.append(previous)
    final = nearest(len(messages) - 1)
    if final is not None and final not in chosen:
        chosen.append(final)
    return sorted(chosen)[-limit:] if len(chosen) > limit else sorted(chosen)


def _as_parts(message: Any) -> Any:
    """A user, tool or system message with string text, as the one text part a breakpoint uses."""
    if not isinstance(message, dict) or message.get('role') not in ('user', 'tool', 'system'):
        return message
    content = message.get('content')
    if isinstance(content, str) and content.strip():
        return dict(message, content=[{'type': 'text', 'text': content}])
    return message


def with_breakpoints(messages: list[Any], marker: dict[str, str], limit: int = MAX_BREAKPOINTS) -> list[Any]:
    """A copy of OpenAI-format messages with cache_control on the chosen text parts.

    Every user, tool and system text goes as one text part, marked or not. Why: a message sent as a
    string while unmarked and as a part list while it carries a breakpoint changed its bytes when
    the breakpoint moved on, so a later request (or a summary request) did not repeat the earlier
    prefix byte for byte."""
    marked = [_as_parts(message) for message in messages]
    for index in breakpoint_indexes(marked, limit):
        marked[index] = _mark(marked[index], marker)
    return marked


def strip_cache_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload without any cache field: markers, cache key, retention/keep_alive hints."""
    cleaned = {k: v for k, v in payload.items()
               if k not in ('prompt_cache_key', 'prompt_cache_retention', 'prompt_cache_options', 'keep_alive')}

    def clean_blocks(blocks: Any) -> Any:
        if not isinstance(blocks, list):
            return blocks
        return [{k: v for k, v in b.items() if k != 'cache_control'} if isinstance(b, dict) else b for b in blocks]

    if isinstance(cleaned.get('messages'), list):
        cleaned['messages'] = [
            dict(m, content=clean_blocks(m.get('content'))) if isinstance(m, dict) and isinstance(m.get('content'), list)
            else m for m in cleaned['messages']]
    for key in ('system', 'tools'):
        if isinstance(cleaned.get(key), list):
            cleaned[key] = clean_blocks(cleaned[key])
    cleaned.pop('cache_control', None)
    return cleaned


def has_cache_fields(payload: dict[str, Any]) -> bool:
    return strip_cache_fields(payload) != payload


# ---------------------------------------------------------------- rejected endpoints

def _endpoint(base_url: str, model: str) -> str:
    parsed = urllib.parse.urlparse(base_url or '')
    return f'{(parsed.hostname or "").lower()}:{parsed.port or ""}{parsed.path.rstrip("/")}|{model}'


def rejected_reason(base_url: str, model: str) -> str | None:
    with _LOCK:
        return _REJECTED.get(_endpoint(base_url, model))


def remember_rejected(base_url: str, model: str, reason: str) -> None:
    """The endpoint answered 400/422 to cache fields and accepted the request without them."""
    key = _endpoint(base_url, model)
    with _LOCK:
        _REJECTED[key] = reason[:200]
        _REJECTED.move_to_end(key)
        while len(_REJECTED) > _REJECTED_MAX:  # bounded: the oldest endpoint is forgotten
            _REJECTED.popitem(last=False)
    from .debug_log import url_for_log

    get_logger('prompt_cache').warning('cache fields rejected by %s (model %s); sent without them from now on: %s',
                                       url_for_log(base_url), sanitize_log(model, 80), sanitize_log(reason, 200))


_CACHE_FIELDS = ('prompt_cache_key', 'prompt_cache_retention', 'prompt_cache_options', 'keep_alive', 'cache_control')
# "unknown field" wording from OpenAI, pydantic (vLLM, LiteLLM), Anthropic, Go and Rust servers
_UNKNOWN_FIELD = re.compile(
    r'unrecogni[sz]ed (request )?(argument|field|param|parameter|key|property)|'
    r'unknown (argument|field|param|parameter|key|property|name)|'
    r'unexpected (argument|field|param|parameter|key|keyword|property)|'
    r'extra[ _](fields?|inputs?|forbidden|arguments?)|additional ?propert', re.IGNORECASE)


def _status_400_or_422(text: str) -> bool:
    return text.startswith(('HTTP 400', 'HTTP 422')) or ' 400 ' in text[:40] or ' 422 ' in text[:40]


def _names(text: str, name: str) -> bool:
    return re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', text, re.IGNORECASE) is not None


def sent_cache_fields(payload: dict[str, Any]) -> list[str]:
    """The cache fields this payload carries (cache_control anywhere counts once)."""
    sent = [name for name in _CACHE_FIELDS if name != 'cache_control' and name in payload]
    rest = {key: value for key, value in payload.items() if key not in sent}
    if strip_cache_fields(rest) != rest:  # what is left differs only by cache_control markers
        sent.append('cache_control')
    return sent


def is_cache_rejection(exc: Exception, payload: dict[str, Any] | None = None) -> bool:
    """Is this 400/422 about the cache fields in `payload`? Yes when the error names one of them, or
    says an unknown field was sent without naming another field of the payload. A 400 about
    anything else (context too long, a bad message) is not, so it is not retried.
    Without a payload, any cache field name counts."""
    text = str(exc)
    if not _status_400_or_422(text):
        return False
    sent = sent_cache_fields(payload) if payload is not None else list(_CACHE_FIELDS)
    if any(_names(text, name) for name in sent):
        return True
    if not _UNKNOWN_FIELD.search(text):
        return False
    others = [key for key in (payload or {}) if key not in _CACHE_FIELDS and key not in ('model', 'messages')]
    return not any(_names(text, key) for key in others)


def should_retry_without_cache(exc: Exception, payload: dict[str, Any], source: str) -> bool:
    """The one-time retry decision for a failed request, logged either way when cache fields were sent."""
    text = str(exc)
    if not _status_400_or_422(text) or not has_cache_fields(payload):
        return False
    retry = is_cache_rejection(exc, payload)
    if retry:
        get_logger('prompt_cache').info('HTTP 400/422 about cache fields from %s; retrying once without them: %s',
                                        sanitize_log(source, 80), sanitize_log(text, 200))
    else:
        get_logger('prompt_cache').info('HTTP 400/422 from %s does not name a cache field we sent (%s); not '
                                        'retried: %s', sanitize_log(source, 80), ', '.join(sent_cache_fields(payload)),
                                        sanitize_log(text, 200))
    return retry


# ---------------------------------------------------------------- prices

def _anthropic_read_ratio(model: str) -> float:
    name = (model or '').lower()
    if re.search(r'(fable|mythos)-5-1', name):
        return 0.025   # A10e
    if re.search(r'(opus|sonnet)-5-5', name):
        return 0.05    # A10d
    return 0.1         # A10c


def _openai_read_ratio(model: str) -> float | None:
    name = (model or '').lower().rsplit('/', 1)[-1]
    if name.startswith('gpt-6.1'):
        return 0.05    # O6d
    if name.startswith(('gpt-5', 'gpt-6')):
        return 0.1     # O6c
    if name.startswith(('gpt-4.1', 'o3')):
        return 0.25    # O6b
    if name.startswith('gpt-4o'):
        return 0.5     # O6
    return None


# Cache-hit price / cache-miss price, from the providers' price pages (checked 2026-11-05). Writes
# carry no surcharge on either (DeepSeek bills a miss at the input price; Z.ai lists cache storage
# as free). Models not listed here are charged at the input price.
_DEEPSEEK_READ = {'deepseek-flash': 0.003 / 0.15,       # api-docs.deepseek.com/quick_start/pricing
                  'deepseek-v4-pro': 0.022 / 0.66}
_ZAI_READ = {'glm-5.3-flashx-legal': 0.075 / 0.37,     # docs.z.ai/guides/overview/pricing
             'glm-5.2': 0.26 / 1.4, 'glm-5.1': 0.26 / 1.4, 'glm-5': 0.2 / 1.0,
             'glm-4.7': 0.11 / 0.6, 'glm-4.6': 0.11 / 0.6}


# A price the provider lists as $0.00. ModelPricing reads 0 as "not set" (those tokens then cost the
# input price), so a free price is kept as a ratio too small to show in any total.
_FREE = 1e-9


@dataclass(frozen=True)
class CacheRatios:
    """Cache prices as multiples of the input price (None = unknown: charged at the input price)."""
    read: float | None = None
    write_5m: float | None = None
    write_1h: float | None = None


def price_ratios(base_url: str, model: str = '') -> CacheRatios:
    provider = cache_provider(base_url, model)
    family = _model_family(model)
    if provider == 'anthropic' or (provider in ('openrouter', 'nanogpt', 'bedrock') and family == 'claude'):
        return CacheRatios(_anthropic_read_ratio(model), 1.25, 2.0)          # A10-A10e
    if provider in ('openai', 'azure') or (provider == 'openrouter' and family == 'openai'):
        read = _openai_read_ratio(model)
        new = re.search(r'gpt-(5\.[6-9]|[6-9])', (model or '').lower()) is not None
        return CacheRatios(read, 1.25 if new else 1.0, 1.25 if new else 1.0)  # O6e: writes 1.25x on 5.6+
    if provider == 'dashscope':
        return CacheRatios(0.1, 1.25, 1.25)                                   # Q1c
    if provider == 'moonshot':
        return CacheRatios(0.1, 1.0, 2.0)                                     # K1b: write 5m = input, 1h = 2x
    if provider == 'mistral':
        return CacheRatios(0.1, 1.0, 1.0)                                     # M1b
    if provider == 'groq':
        return CacheRatios(0.5, 1.0, 1.0)                                     # GQ1
    if provider == 'fireworks':
        return CacheRatios(0.5, 1.0, 1.0)                                     # F1 default discount
    if provider == 'cerebras':
        return CacheRatios(1.0, 1.0, 1.0)                                     # "no additional fee": no discount
    if provider in ('gemini', 'vertex'):
        # aistudio.google.com/docs/pricing (2026-11-05): the context caching price is 10% of the input
        # price for every listed model, implicit hits included; storage is charged apart (explicit_cache)
        return CacheRatios(0.1, 1.0, 1.0)
    if provider == 'bedrock' and 'nova' in (model or '').lower():
        # AWS Price List (pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock, us-east-1,
        # 2026-11-02): Nova cache reads 25% of input (Pro, Lite, Micro, Premier, Nova 2 Lite) and cache
        # writes $0.00. Implicit Nova reuse is not reported in the cache fields, so never priced as a read.
        return CacheRatios(0.25, _FREE, _FREE)
    if provider == 'sambanova':
        # docs.sambanova.ai/docs/en/features/prompt-caching: cached tokens $0.06 vs $0.60 (MiniMax-M2.7, the
        # only caching model); cache_creation_tokens are informational, no extra charge
        return CacheRatios(0.1, 1.0, 1.0)
    if provider in ('deepseek', 'zai'):
        name = (model or '').lower().rsplit('/', 1)[-1]
        read = (_DEEPSEEK_READ if provider == 'deepseek' else _ZAI_READ).get(name)
        return CacheRatios(read, 1.0, 1.0) if read is not None else CacheRatios()
    return CacheRatios()


_PRICE_FIELDS = (('cache_read_input_cost_per_million_tokens_usd', 'read', 'cache_read'),
                 ('cache_creation_input_cost_per_million_tokens_usd', 'write_5m', 'cache_write'),
                 ('cache_creation_1h_input_cost_per_million_tokens_usd', 'write_1h', 'cache_write_1h'))
_BASE_FIELDS = (('input_cost_per_million_tokens_usd', 'input'), ('output_cost_per_million_tokens_usd', 'output'))


def _same_price(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def known_prices(base_url: str, model: str, api_key: str = '') -> dict[str, float]:
    """Published or user-set prices for model at base_url (model_info.model_prices); {} on any error."""
    try:
        from .model_info import model_prices

        return model_prices(base_url, model, api_key)
    except Exception as exc:  # a price lookup never stops a model switch
        from .debug_log import log_ignored_error

        log_ignored_error(exc)
        return {}


def derived_cache_price(input_price: float, ratio: float | None, published: float | None) -> float:
    """A cache price: the published one, else the input price times the provider's ratio, else 0
    (unset: cache tokens are then charged at the input price)."""
    if published is not None:
        return published
    return input_price * ratio if (input_price and ratio is not None) else 0.0


def reprice(pricing: Any, old_base_url: str, old_model: str, new_base_url: str, new_model: str, *,
            old_input: float | None = None, api_key: str = '') -> Any:
    """Prices (a ModelPricing) after a switch of model or endpoint (/model, /profile, --profile,
    --agent, a fallback, a subagent or background model, agent-resume --model / --base-url).

    A price that came from the old model's known prices (its server's list or the user's
    modelPrices table; for cache prices also its input price x the provider's ratio), or is unset,
    is derived again for the new model. A price that differs from that was given with a flag and is
    kept. Input and output prices change only when the model or endpoint changes; `old_input` is
    the input price the current cache prices came from when the caller changed the input price."""
    base_now = float(getattr(pricing, 'input_cost_per_million_tokens_usd', 0.0) or 0.0)
    input_before = base_now if old_input is None else float(old_input or 0.0)
    model_changed = (old_base_url, old_model) != (new_base_url, new_model)
    if not model_changed and _same_price(input_before, base_now):
        return pricing
    old_known = known_prices(old_base_url, old_model, api_key) if model_changed else {}
    new_known = known_prices(new_base_url, new_model, api_key) if model_changed else {}
    if not model_changed:
        old_known = new_known = known_prices(new_base_url, new_model, api_key)
    changes: dict[str, float] = {}
    if model_changed:
        for name, key in _BASE_FIELDS:
            current = float(getattr(pricing, name, 0.0) or 0.0)
            before = old_known.get(key)
            if not current or (before is not None and _same_price(current, before)):
                changes[name] = float(new_known.get(key, 0.0))
    new_input = changes.get('input_cost_per_million_tokens_usd', base_now)
    old_ratios = price_ratios(old_base_url, old_model)
    new_ratios = price_ratios(new_base_url, new_model)
    for name, ratio_name, key in _PRICE_FIELDS:
        current = float(getattr(pricing, name, 0.0) or 0.0)
        derived_before = derived_cache_price(input_before, getattr(old_ratios, ratio_name), old_known.get(key))
        if current and not _same_price(current, derived_before):
            continue  # set by the user
        changes[name] = derived_cache_price(new_input, getattr(new_ratios, ratio_name), new_known.get(key))
    return replace(pricing, **changes)
