"""Explicit context caches: a named `cachedContents` resource on the Gemini API or Vertex AI that
holds the session's stable prefix (system instruction and tool list), referenced by each request.

When it is used (SIGIL_EXPLICIT_CACHE = auto, the default; `on` forces it, `off` never creates one;
SIGIL_PROMPT_CACHE=off turns it off too):
- auto: only where the documented prices say it saves money over implicit caching. Implicit hits on
  Gemini cost the same as explicit reads, so an explicit cache only pays when implicit caching misses.
  Sigil measures this per endpoint and model from the last requests (at least 3 within 15 minutes)
  and creates a cache when
      requests per hour x implicit miss rate x (input price - cached price) > storage price per hour
  (all per million tokens). The prefix must also reach the model's minimum cache size, and the
  model must be one whose storage price is known (gemini-4-argon has no explicit caching).
- The cache lives for the session's TTL (SIGIL_PROMPT_CACHE_TTL, promptCache, /cache ttl). Its TTL
  is renewed (PATCH) when half of it is left and the cache is still in use.
- A request that uses the cache leaves out what is already in it (system messages, tools,
  tool_choice; the forum-reported rule) and names it in extra_body.google.cached_content.
- A changed prefix (another system prompt or tool list) gets its own cache; one of the same
  endpoint and model that has not been used for 2 minutes is then deleted.

Never breaking a request: a failed create, lookup or renewal is logged and the request goes out as
before. A request that names the cache and fails is sent again once without it, the cache is
dropped, and after two failed creates the endpoint gets no explicit cache for the rest of the run.

Never leaking a cache: every cache is named `sigil-<install id>-<pid>-<random>` (no user name, path
or key) and listed in ~/.sigil/explicit-caches.json. This process deletes its caches at exit and when
a session ends; caches of this install whose process is gone (a crash) are deleted at start-up in the
background (every endpoint with a usable key, within a time limit) and at the first use of an
endpoint. The storage cost of the active session is saved in its session file and restored on resume. At most SIGIL_EXPLICIT_CACHE_MAX (4) live per process; the least recently used
goes first. Background agents are other processes and get their own caches.
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from .debug_log import get_logger, log_ignored_error, sanitize_log, url_for_log

FULL_KEY = '__sigil_full_payload'   # the payload without the cache, kept for the one fallback
MODES = ('auto', 'on', 'off')
DEFAULT_MAX = 4
STALE_SECONDS = 120.0
SAMPLE_WINDOW = 900.0               # 15 minutes of requests for the price rule
MIN_SAMPLES = 3
_TIMEOUT = 30.0

# USD per million tokens: (input, cached input, storage per hour). Gemini API price page
# (aistudio.google.com/docs/pricing, Standard tab, checked 2026-11-05; prompts <= 200k tokens).
# A row whose price changes on a known day is a tuple of (first UTC day 'YYYY-MM-DD', prices) steps
# in date order ('' = from the start); prices_for picks the step for the day of the request.
_FLASH_3_6_3_8 = (('', (0.75, 0.075, 0.50)),            # "$0.75 through Dec 31, 2026"
                  ('2027-01-01', (1.50, 0.15, 1.00)))    # "$1.50 starting Jan 1, 2027"
GEMINI_PRICES: dict[str, Any] = {
    'gemini-3.8-flash': _FLASH_3_6_3_8,
    'gemini-3.6-flash': _FLASH_3_6_3_8,
    'gemini-3.5-flash-lite': (0.30, 0.03, 1.00),
    'gemini-3.1-flash-lite': (0.25, 0.025, 1.00),
    'gemini-3.1-pro-preview': (2.00, 0.20, 4.50),
    'gemini-3-flash-preview': (0.50, 0.05, 1.00),
    'gemini-2.5-pro': (1.25, 0.125, 4.50),
    'gemini-2.5-flash': (0.30, 0.03, 1.00),
    'gemini-2.5-flash-lite': (0.10, 0.01, 1.00),
}
# Vertex (cloud.google.com/model-garden/pricing, where .../vertex-ai/generative-ai/pricing redirects,
# 2026-11-05): cached input per 1M and "Context Cache Storage price for Explicit Caching" per token-hour
# (x 1e6 = per million per hour). Input = cached x 10: Vertex's caching overview gives a 90% discount
# for explicit caching on Gemini 2.5 and later. Only models the explicit-caching table lists.
VERTEX_PRICES: dict[str, tuple[float, float, float]] = {
    'gemini-2.5-pro': (1.25, 0.125, 4.50),
    'gemini-2.5-flash': (0.30, 0.03, 1.00),
    'gemini-2.5-flash-lite': (0.10, 0.01, 1.00),
    'gemini-3.1-pro-preview': (2.00, 0.20, 4.50),
}
# "Implicit caching only (explicit caching is not supported)" (same page)
NO_EXPLICIT = frozenset(('gemini-4-argon',))
# Minimum cached size (aistudio.google.com/docs/caching): 2,048 for 2.5 Flash / Pro, 4,096 for the rest
MIN_TOKENS = {'gemini-2.5-flash': 2048, 'gemini-2.5-pro': 2048, 'gemini-2.5-flash-lite': 2048}
DEFAULT_MIN_TOKENS = 4096
# Vertex (docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview):
# 2,048 for the Gemini 2 family, 6,144 for these, 4,096 for the other Gemini 3 and 4 models
VERTEX_MIN_TOKENS = {'gemini-3-flash-preview': 6144, 'gemini-3.0-flash-preview': 6144, 'gemini-3.1-pro-preview': 6144,
                     'gemini-3.7-flash': 6144, 'gemini-3.8-flash': 6144}

_LOCK = threading.RLock()
_REG_LOCK = threading.Lock()
_SESSION: dict[str, str] = {}
_LIVE: dict[str, 'LiveCache'] = {}
_CREATING: set[str] = set()
_FAILED_CREATES: dict[str, int] = {}
_DEAD: dict[str, str] = {}                      # endpoint id -> why explicit caches are off there
_SAMPLES: dict[str, list[tuple[float, int, int]]] = {}
# Endpoints that have reported a cached-token count above 0 at least once. Google does not document
# the cached-token field on its OpenAI-compatible endpoints (a forum report says it was missing), so
# zeros only count as implicit misses once the field is known to be sent.
_REPORTED: set[str] = set()
_SWEPT: set[str] = set()
_NOTED: set[str] = set()
_SPENT = [0.0]                                  # storage cost of caches already deleted
# The session the storage cost is counted for: its saved cost (base) and the process total when it
# became active (mark). Saved in the session file (cache_storage_usd) and restored on resume.
_ACTIVE: dict[str, Any] = {'session': None, 'base': 0.0, 'mark': 0.0}
_ATEXIT = [False]
_START_SWEEP: dict[str, Any] = {'thread': None}  # the start-up crash sweep (once per process)
_NAME_RE = re.compile(r'^(projects/[A-Za-z0-9_.-]+/locations/[A-Za-z0-9_-]+/)?cachedContents/[A-Za-z0-9_-]+$')
_SEGMENT_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


class ExplicitCacheError(RuntimeError):
    pass


# ---------------------------------------------------------------- settings

def mode() -> str:
    """auto | on | off. /cache explicit for this session, then SIGIL_EXPLICIT_CACHE; off when prompt
    caching as a whole is off."""
    from .prompt_cache import current_settings, parse_mode

    if current_settings().mode == 'off':
        return 'off'
    with _LOCK:
        chosen = _SESSION.get('mode')
    if chosen in MODES:
        return chosen
    raw = os.environ.get('SIGIL_EXPLICIT_CACHE', '').strip()
    return (parse_mode(raw, 'SIGIL_EXPLICIT_CACHE') or 'auto') if raw else 'auto'


def set_session_mode(value: str | None) -> None:
    with _LOCK:
        if value is None:
            _SESSION.pop('mode', None)
        else:
            _SESSION['mode'] = value


def forward_env(env: dict[str, str]) -> dict[str, str]:
    """A background agent gets this session's /cache explicit choice (only when one was made)."""
    with _LOCK:
        chosen = _SESSION.get('mode')
    if chosen in MODES:
        env['SIGIL_EXPLICIT_CACHE'] = chosen
    return env


def ttl_seconds() -> int:
    from .prompt_cache import TTL_SECONDS, current_settings

    return int(TTL_SECONDS.get(current_settings().ttl, 3600))


def max_caches() -> int:
    raw = os.environ.get('SIGIL_EXPLICIT_CACHE_MAX', '').strip()
    if not raw:
        return DEFAULT_MAX
    try:
        return min(max(int(raw), 1), 64)
    except ValueError as exc:
        from .prompt_cache import _warn_once

        _warn_once('explicit-max', f'SIGIL_EXPLICIT_CACHE_MAX is not a whole number ({exc}); using {DEFAULT_MAX}.')
        return DEFAULT_MAX


# ---------------------------------------------------------------- endpoints

@dataclass(frozen=True)
class Endpoint:
    kind: str           # 'gemini' or 'vertex'
    version_root: str   # https://host/v1beta (resource names are relative to this)
    parent: str         # where cachedContents live: '' (Gemini) or projects/p/locations/l (Vertex)
    model_name: str     # models/<m> or projects/p/locations/l/publishers/google/models/<m>
    short_model: str

    @property
    def collection_url(self) -> str:
        return f'{self.version_root}/{self.parent + "/" if self.parent else ""}cachedContents'

    @property
    def id(self) -> str:
        return f'{self.version_root}|{self.parent}'


def endpoint_for(base_url: str, model: str) -> Endpoint | None:
    """The cache endpoint behind an OpenAI-compatible Gemini or Vertex base URL, or None."""
    parsed = urllib.parse.urlparse(base_url or '')
    host = (parsed.hostname or '').lower()
    short = (model or '').strip().split('/')[-1]
    if not short or '..' in (model or '') or not _SEGMENT_RE.match(short) or parsed.scheme != 'https':
        return None
    if host == 'generativelanguage.googleapis.com':
        match = re.match(r'^/(v1(?:beta|alpha)?)(/|$)', parsed.path or '')
        version = match.group(1) if match else 'v1beta'
        return Endpoint('gemini', f'https://{host}/{version}', '', f'models/{short}', short)
    if host == 'aiplatform.googleapis.com' or host.endswith('-aiplatform.googleapis.com'):
        match = re.match(r'^/(v1(?:beta1)?)/projects/([^/]+)/locations/([^/]+)(/|$)', parsed.path or '')
        if not match or not (_SEGMENT_RE.match(match.group(2)) and _SEGMENT_RE.match(match.group(3))):
            return None
        version, project, location = match.group(1), match.group(2), match.group(3)
        parent = f'projects/{project}/locations/{location}'
        return Endpoint('vertex', f'https://{host}/{version}', parent,
                        f'{parent}/publishers/google/models/{short}', short)
    return None


def prices_for(endpoint: Endpoint, when: float | None = None) -> tuple[float, float, float] | None:
    """(input, cached, storage per hour) for the endpoint's model on the day of `when` (default now)."""
    table = VERTEX_PRICES if endpoint.kind == 'vertex' else GEMINI_PRICES
    row = table.get(endpoint.short_model)
    if row is None or not row or not isinstance(row[0], tuple):
        return row
    day = time.strftime('%Y-%m-%d', time.gmtime(time.time() if when is None else when))
    chosen = None
    for start, prices in row:   # dated steps, in date order
        if not start or day >= start:
            chosen = prices
    return chosen


def min_tokens(model: str, kind: str = 'gemini') -> int:
    if kind == 'vertex':
        if model in VERTEX_MIN_TOKENS:
            return VERTEX_MIN_TOKENS[model]
        return 2048 if model.startswith('gemini-2') else DEFAULT_MIN_TOKENS
    return MIN_TOKENS.get(model, DEFAULT_MIN_TOKENS)


def worth_it(requests_per_hour: float, miss_rate: float, input_price: float, cached_price: float,
             storage_per_hour: float) -> bool:
    """The price rule: an explicit cache saves money when the hourly saving on implicit misses is
    more than the hourly storage (per million prefix tokens on both sides)."""
    return requests_per_hour * max(0.0, min(1.0, miss_rate)) * max(0.0, input_price - cached_price) > storage_per_hour


# ---------------------------------------------------------------- live caches

@dataclass
class LiveCache:
    key: str
    name: str
    endpoint: Endpoint
    display_name: str
    tokens: int
    created: float
    expire: float
    ttl: int
    storage_per_hour: float | None
    auth: Callable[[], str] = field(repr=False, compare=False)
    last_used: float = 0.0
    uses: int = 0

    def storage_cost(self, now: float | None = None) -> float:
        if not self.storage_per_hour:
            return 0.0
        end = min(time.time() if now is None else now, self.expire)
        hours = max(0.0, end - self.created) / 3600.0
        return self.tokens / 1e6 * self.storage_per_hour * hours


def _install_id() -> str:
    path = Path.home() / '.sigil' / 'install-id'
    try:
        value = path.read_text(encoding='utf-8').strip()
        if re.fullmatch(r'[0-9a-f]{12}', value):
            return value
    except OSError as exc:
        log_ignored_error(exc)
    value = secrets.token_hex(6)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from .settings import write_text_atomic

        write_text_atomic(path, value + '\n')
    except OSError as exc:
        log_ignored_error(exc)
    return value


def _display_name() -> str:
    return f'sigil-{_install_id()}-{os.getpid()}-{secrets.token_hex(3)}'


def _owner_pid(display_name: str) -> int | None:
    match = re.fullmatch(r'sigil-([0-9a-f]{12})-(\d+)-[0-9a-f]+', display_name or '')
    if not match or match.group(1) != _install_id():
        return None
    return int(match.group(2))


def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == 'nt':
        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            kernel.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError as exc:
        log_ignored_error(exc)
        return False
    except PermissionError as exc:  # someone else's process: alive, and not ours to judge
        log_ignored_error(exc)
        return True
    return True


# ---------------------------------------------------------------- registry (crash recovery)

def _registry_path() -> Path:
    return Path.home() / '.sigil' / 'explicit-caches.json'


def _registry() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_registry_path().read_text(encoding='utf-8'))
    except FileNotFoundError as exc:  # no cache was ever created here
        log_ignored_error(exc)
        return {}
    except (OSError, ValueError) as exc:
        get_logger('explicit_cache').warning('cache registry unreadable, starting a new one: %s', type(exc).__name__)
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _registry_update(add: dict[str, Any] | None = None, remove: str | None = None) -> None:
    from .settings import write_text_atomic

    with _REG_LOCK:
        data = _registry()
        if add is not None:
            data[add['name']] = add
        if remove is not None:
            data.pop(remove, None)
        try:
            _registry_path().parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(_registry_path(), json.dumps(data, indent=1, sort_keys=True))
        except OSError as exc:
            get_logger('explicit_cache').warning('cache registry not written: %s', type(exc).__name__)


# ---------------------------------------------------------------- HTTP

def delete_timeout() -> float:
    """Seconds an exit or session-end delete may take (SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT, default 5):
    exit must not hang without a network; an undeleted cache still expires by its TTL."""
    raw = os.environ.get('SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT', '').strip()
    try:
        return min(max(float(raw), 0.5), 60.0) if raw else 5.0
    except ValueError as exc:
        log_ignored_error(exc)
        return 5.0


def _call(endpoint: Endpoint, method: str, url: str, body: Any, key: str, timeout: float = _TIMEOUT) -> Any:
    headers = {'Content-Type': 'application/json'}
    if endpoint.kind == 'gemini':
        headers['x-goog-api-key'] = key
    else:
        headers['Authorization'] = f'Bearer {key}'
    data = json.dumps(body).encode('utf-8') if body is not None else None
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace') if exc.fp else ''
        raise ExplicitCacheError(f'HTTP {exc.code} from {url_for_log(url)}: {sanitize_log(detail, 300)}') from exc
    except (error.URLError, OSError) as exc:
        raise ExplicitCacheError(f'cannot reach {url_for_log(url)}: {type(exc).__name__}') from exc
    if not raw:
        return {}
    try:
        return json.loads(raw.decode('utf-8'))
    except ValueError as exc:
        raise ExplicitCacheError(f'{url_for_log(url)} returned invalid JSON') from exc


def _resource_url(endpoint: Endpoint, name: str) -> str:
    if not _NAME_RE.match(name or ''):
        raise ExplicitCacheError(f'unexpected cache name {sanitize_log(name, 80)}')
    return f'{endpoint.version_root}/{name}'


def _parse_time(text: Any) -> float | None:
    if not isinstance(text, str) or not text:
        return None
    from datetime import datetime

    try:
        cleaned = re.sub(r'(\.\d{6})\d+', r'\1', text.replace('Z', '+00:00'))
        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError as exc:
        log_ignored_error(exc)
        return None


# ---------------------------------------------------------------- the request side

def _split_prefix(payload: dict[str, Any]) -> tuple[str, list[Any], list[Any]] | None:
    messages = payload.get('messages') or []
    system_parts: list[str] = []
    index = 0
    while index < len(messages) and isinstance(messages[index], dict) and messages[index].get('role') == 'system':
        content = messages[index].get('content')
        if isinstance(content, list):
            content = ''.join(str(p.get('text') or '') for p in content if isinstance(p, dict))
        system_parts.append(str(content or ''))
        index += 1
    rest = list(messages[index:])
    if not rest:
        return None
    return '\n\n'.join(part for part in system_parts if part), list(payload.get('tools') or []), rest


def _gemini_tools(tools: list[Any]) -> list[dict[str, Any]]:
    declarations = []
    for tool in tools:
        function = tool.get('function', tool) if isinstance(tool, dict) else {}
        if not isinstance(function, dict) or not function.get('name'):
            continue
        declaration = {'name': str(function['name']), 'description': str(function.get('description') or '')}
        parameters = function.get('parameters')
        if isinstance(parameters, dict) and parameters.get('properties'):
            declaration['parametersJsonSchema'] = parameters
        declarations.append(declaration)
    return [{'functionDeclarations': declarations}] if declarations else []


def observe(base_url: str, model: str, usage: Any) -> None:
    """Record one reply's prompt and cached tokens (the price rule's evidence)."""
    endpoint = endpoint_for(base_url, model)
    if endpoint is None or usage is None:
        return
    prompt = int(getattr(usage, 'prompt_tokens', 0) or 0)
    if prompt <= 0:
        return
    cached = int(getattr(usage, 'cache_read_input_tokens', 0) or 0)
    key = f'{endpoint.id}|{endpoint.short_model}'
    now = time.time()
    with _LOCK:
        if cached > 0 and len(_REPORTED) < 256:
            _REPORTED.add(key)
        samples = [s for s in _SAMPLES.get(key, []) if now - s[0] <= SAMPLE_WINDOW][-19:]
        samples.append((now, prompt, cached))
        _SAMPLES[key] = samples


def observed(endpoint: Endpoint, prefix_tokens: int) -> tuple[float, float] | None:
    """(requests per hour, implicit miss rate on the prefix) from recent replies, or None."""
    now = time.time()
    key = f'{endpoint.id}|{endpoint.short_model}'
    with _LOCK:
        samples = [s for s in _SAMPLES.get(key, []) if now - s[0] <= SAMPLE_WINDOW]
        reported = key in _REPORTED
    if len(samples) < MIN_SAMPLES:
        return None
    if not reported:
        _note_once(f'unreported:{key}', 'no cached-token count reported yet by %s; implicit misses cannot be '
                   'measured, so implicit caching only (SIGIL_EXPLICIT_CACHE=on forces explicit caches)',
                   endpoint.short_model)
        return None
    span = max(samples[-1][0] - samples[0][0], 60.0)
    rate = (len(samples) - 1) / (span / 3600.0)
    later = samples[1:]  # the first request of a prefix can never hit
    average_cached = sum(s[2] for s in later) / len(later)
    miss = 1.0 - min(1.0, average_cached / max(prefix_tokens, 1))
    return rate, max(0.0, miss)


def _note_once(key: str, message: str, *args: Any) -> None:
    with _LOCK:
        if key in _NOTED or len(_NOTED) > 256:
            return
        _NOTED.add(key)
    get_logger('explicit_cache').info(message, *args)


def _should_create(endpoint: Endpoint, prefix_tokens: int, chosen: str) -> bool:
    if endpoint.short_model in NO_EXPLICIT:
        _note_once(f'none:{endpoint.short_model}', '%s supports implicit caching only', endpoint.short_model)
        return False
    if prefix_tokens < min_tokens(endpoint.short_model, endpoint.kind):
        _note_once(f'small:{endpoint.id}:{prefix_tokens // 1024}', 'prefix of about %d tokens is below the %d-token '
                   'minimum for %s; implicit caching only', prefix_tokens,
                   min_tokens(endpoint.short_model, endpoint.kind), endpoint.short_model)
        return False
    if chosen == 'on':
        return True
    prices = prices_for(endpoint)
    if prices is None:
        _note_once(f'price:{endpoint.short_model}', 'no storage price known for %s; implicit caching only '
                   '(SIGIL_EXPLICIT_CACHE=on forces explicit caches)', endpoint.short_model)
        return False
    seen = observed(endpoint, prefix_tokens)
    if seen is None:
        return False
    rate, miss = seen
    decision = worth_it(rate, miss, *prices)
    _note_once(f'rule:{endpoint.id}:{endpoint.short_model}:{decision}', 'explicit cache for %s: %s (%.1f requests/h, '
               'implicit miss rate %.0f%%, input %.3f, cached %.3f, storage %.2f per million per hour)',
               endpoint.short_model, 'worth it' if decision else 'not worth it', rate, miss * 100, *prices)
    return decision


def _create(endpoint: Endpoint, key: str, system: str, tools: list[Any], auth: Callable[[], str],
            prefix_tokens: int) -> LiveCache | None:
    ttl = ttl_seconds()
    display = _display_name()
    body: dict[str, Any] = {'model': endpoint.model_name, 'displayName': display, 'ttl': f'{ttl}s'}
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    gemini_tools = _gemini_tools(tools)
    if gemini_tools:
        body['tools'] = gemini_tools
    try:
        answer = _call(endpoint, 'POST', endpoint.collection_url, body, auth())
        name = str(answer.get('name') or '')
        _resource_url(endpoint, name)
    except ExplicitCacheError as exc:
        with _LOCK:
            failures = _FAILED_CREATES.get(endpoint.id, 0) + 1
            _FAILED_CREATES[endpoint.id] = failures
            if failures >= 2:
                _DEAD[endpoint.id] = str(exc)[:200]
        get_logger('explicit_cache').warning('explicit cache not created (%s); the request goes out without it%s',
                                             exc, '; explicit caches are off for this endpoint now' if failures >= 2 else '')
        return None
    now = time.time()
    tokens = int((answer.get('usageMetadata') or {}).get('totalTokenCount') or prefix_tokens)
    expire = _parse_time(answer.get('expireTime')) or now + ttl
    prices = prices_for(endpoint)
    live = LiveCache(key, name, endpoint, display, tokens, now, expire, ttl, prices[2] if prices else None, auth,
                     last_used=time.monotonic())
    _registry_update(add={'name': name, 'version_root': endpoint.version_root, 'parent': endpoint.parent,
                          'kind': endpoint.kind, 'model': endpoint.short_model, 'display_name': display,
                          'pid': os.getpid(), 'created': now, 'expire': expire, 'tokens': tokens})
    get_logger('explicit_cache').info('explicit cache %s created for %s: %d tokens, ttl %ds', name,
                                      endpoint.short_model, tokens, ttl)
    _telemetry('create', live)
    with _LOCK:
        if not _ATEXIT[0]:
            _ATEXIT[0] = True
            atexit.register(release_all, 'exit')
    return live


def _telemetry(action: str, live: LiveCache) -> None:
    try:
        from .telemetry import record

        record({'type': 'explicit_cache', 'action': action, 'model': live.endpoint.short_model,
                'tokens': live.tokens, 'storage_cost_usd': round(live.storage_cost(), 6)})
    except Exception as exc:  # telemetry never stops a request
        log_ignored_error(exc)


def _delete(live: LiveCache, reason: str, timeout: float = _TIMEOUT) -> bool:
    ok = True
    try:
        _call(live.endpoint, 'DELETE', _resource_url(live.endpoint, live.name), None, live.auth(), timeout)
    except ExplicitCacheError as exc:
        ok = '404' in str(exc)[:12]  # already gone (expired) counts as deleted
        if not ok:
            get_logger('explicit_cache').warning('explicit cache %s not deleted (%s): %s; it expires on its own',
                                                 live.name, reason, exc)
    with _LOCK:
        _SPENT[0] += live.storage_cost()
    _registry_update(remove=live.name)
    _telemetry('delete', live)
    if ok:
        get_logger('explicit_cache').info('explicit cache %s deleted (%s)', live.name, reason)
    return ok


def _renew(live: LiveCache) -> None:
    try:
        url = _resource_url(live.endpoint, live.name) + '?updateMask=ttl'
        answer = _call(live.endpoint, 'PATCH', url, {'ttl': f'{live.ttl}s'}, live.auth())
        with _LOCK:
            live.expire = _parse_time((answer or {}).get('expireTime')) or time.time() + live.ttl
        _registry_update(add={**_registry().get(live.name, {'name': live.name}), 'expire': live.expire})
    except ExplicitCacheError as exc:
        get_logger('explicit_cache').warning('explicit cache %s TTL not renewed: %s', live.name, exc)


def sweep(endpoint: Endpoint, auth: Callable[[], str], timeout: float = _TIMEOUT) -> int:
    """Delete this install's caches at `endpoint` whose process is gone (crash recovery). Returns
    how many were deleted. Caches of live processes and of other installs are never touched: a
    registry entry counts only when its display name carries this install's id."""
    with _LOCK:
        if endpoint.id in _SWEPT:
            return 0
        _SWEPT.add(endpoint.id)
    names: dict[str, str] = {}
    for entry in _registry().values():
        if entry.get('version_root') == endpoint.version_root and entry.get('parent') == endpoint.parent:
            pid = _owner_pid(str(entry.get('display_name') or ''))
            if pid is not None and not _pid_alive(pid):
                names[str(entry.get('name'))] = 'registry'
    try:
        token = ''
        for _page in range(10):
            url = endpoint.collection_url + '?pageSize=100' + (f'&pageToken={urllib.parse.quote(token)}' if token else '')
            answer = _call(endpoint, 'GET', url, None, auth(), timeout)
            for item in answer.get('cachedContents') or []:
                pid = _owner_pid(str(item.get('displayName') or ''))
                if pid is not None and not _pid_alive(pid):
                    names[str(item.get('name'))] = 'list'
            token = str(answer.get('nextPageToken') or '')
            if not token:
                break
    except ExplicitCacheError as exc:
        get_logger('explicit_cache').warning('could not list explicit caches for cleanup: %s', exc)
    deleted = 0
    for name in names:
        try:
            _call(endpoint, 'DELETE', _resource_url(endpoint, name), None, auth(), timeout)
            deleted += 1
        except ExplicitCacheError as exc:
            if '404' not in str(exc)[:12]:
                get_logger('explicit_cache').warning('orphan cache %s not deleted: %s', name, exc)
        _registry_update(remove=name)
    if deleted:
        get_logger('explicit_cache').info('deleted %d explicit cache(s) left by an earlier run', deleted)
    return deleted


def _base_url(endpoint: Endpoint) -> str:
    """The OpenAI-compatible base URL an endpoint belongs to (for its key)."""
    if endpoint.kind == 'vertex':
        return f'{endpoint.version_root}/{endpoint.parent}/endpoints/openapi'
    return f'{endpoint.version_root}/openai'


def _registry_endpoint(entry: dict[str, Any]) -> Endpoint | None:
    """The endpoint of a registry entry, checked like a base URL (Google hosts, https, plain path
    segments), so an edited registry file cannot send a request anywhere else."""
    kind, root, parent = entry.get('kind'), entry.get('version_root'), entry.get('parent') or ''
    if kind not in ('gemini', 'vertex') or not isinstance(root, str) or not isinstance(parent, str):
        return None
    base = f'{root}/{parent}/endpoints/openapi' if kind == 'vertex' else f'{root}/openai'
    endpoint = endpoint_for(base, str(entry.get('model') or 'model'))
    if endpoint is None or (endpoint.kind, endpoint.version_root, endpoint.parent) != (kind, root, parent):
        return None
    return endpoint


def sweep_at_start(key_for: Callable[[str], str | None], *, time_limit: float | None = None,
                   wait: bool = False) -> None:
    """At start-up, in the background: delete this install's caches left by a process that is gone
    (a crash) at every endpoint in the registry that has a usable key or token (`key_for(base_url)`;
    Vertex also takes SIGIL_GOOGLE_TOKEN_COMMAND). Bounded: each request waits at most
    SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT seconds and the whole sweep stops after `time_limit` (ten
    times that by default); failures are logged. An endpoint without a key keeps the first-use
    clean-up. Once per process."""
    with _LOCK:
        if _START_SWEEP['thread'] is not None:
            return
        limit = 10 * delete_timeout() if time_limit is None else max(0.0, float(time_limit))
        thread = threading.Thread(target=_sweep_registry, args=(key_for, limit), name='cache-start-sweep',
                                  daemon=True)
        _START_SWEEP['thread'] = thread
    thread.start()
    if wait:
        thread.join(limit + 30)


def wait_for_start_sweep(timeout: float) -> bool:
    """Wait for the start-up sweep (tests, a clean exit); True when it is done or never started."""
    with _LOCK:
        thread = _START_SWEEP['thread']
    if thread is not None:
        thread.join(timeout)
        return not thread.is_alive()
    return True


def _sweep_registry(key_for: Callable[[str], str | None], limit: float) -> None:
    deadline = time.monotonic() + limit
    log = get_logger('explicit_cache')
    endpoints: dict[str, Endpoint] = {}
    for entry in _registry().values():
        pid = _owner_pid(str(entry.get('display_name') or ''))   # this install's caches only
        if pid is None or _pid_alive(pid):
            continue
        endpoint = _registry_endpoint(entry)
        if endpoint is not None:
            endpoints.setdefault(endpoint.id, endpoint)
    for endpoint in endpoints.values():
        if time.monotonic() >= deadline:
            log.warning('start-up clean-up of explicit caches stopped at its time limit (%.1f s); the rest is '
                        'cleaned up at first use of the endpoint or expires on its own', limit)
            return
        base = _base_url(endpoint)
        try:
            key = key_for(base)
        except Exception as exc:  # a key lookup never stops the start-up
            log_ignored_error(exc)
            key = None
        if endpoint.kind == 'vertex' and (key or os.environ.get('SIGIL_GOOGLE_TOKEN_COMMAND', '').strip()):
            from .google_auth import access_token

            auth: Callable[[], str] = lambda fallback=str(key or ''): access_token(fallback)  # noqa: E731
        elif endpoint.kind == 'gemini' and key:
            auth = lambda value=str(key): value  # noqa: E731
        else:
            log.info('start-up clean-up skipped %s: no usable key (SIGIL_API_KEY_<HOST>, or '
                     'SIGIL_GOOGLE_TOKEN_COMMAND for Vertex); it runs at the first use of that endpoint',
                     url_for_log(base))
            continue
        try:
            sweep(endpoint, auth, timeout=max(0.5, min(delete_timeout(), deadline - time.monotonic())))
        except Exception as exc:  # logged; the first-use clean-up does not run again for it
            log.warning('start-up clean-up of %s failed: %s', url_for_log(base), sanitize_log(str(exc), 200))


def apply(config: Any, payload: dict[str, Any], auth: Callable[[], str]) -> dict[str, Any]:
    """The payload to send: unchanged, or using an explicit cache (FULL_KEY holds the original)."""
    if FULL_KEY in payload:
        return payload
    endpoint = endpoint_for(str(getattr(config, 'base_url', '')), str(getattr(config, 'model', '')))
    if endpoint is None:
        return payload
    chosen = mode()
    if chosen == 'off':
        return payload
    with _LOCK:
        if endpoint.id in _DEAD:
            return payload
    split = _split_prefix(payload)
    if split is None:
        return payload
    system, tools, rest = split
    if not system and not tools:
        return payload
    key = hashlib.sha256(json.dumps([endpoint.id, endpoint.short_model, system, tools], sort_keys=True,
                                    default=str).encode('utf-8')).hexdigest()[:24]
    prefix_tokens = (len(system) + len(json.dumps(tools))) // 4
    with _LOCK:
        first_use = endpoint.id not in _SWEPT
    if first_use:  # crash recovery once per endpoint, off the request path
        threading.Thread(target=sweep, args=(endpoint, auth), name='cache-sweep', daemon=True).start()
    with _LOCK:
        live = _LIVE.get(key)
        if live is not None and live.expire <= time.time() + 5:
            _LIVE.pop(key, None)
            live = None
        if live is None:
            if key in _CREATING:
                return payload  # another thread is creating it; this request goes without
            _CREATING.add(key)
    if live is None:
        try:
            if not _should_create(endpoint, prefix_tokens, chosen):
                return payload
            live = _create(endpoint, key, system, tools, auth, prefix_tokens)
        finally:
            with _LOCK:
                _CREATING.discard(key)
        if live is None:
            return payload
        _admit(live)
    with _LOCK:
        live.last_used = time.monotonic()
        live.uses += 1
        renew = live.expire - time.time() < live.ttl / 2
    if renew:
        threading.Thread(target=_renew, args=(live,), name='cache-renew', daemon=True).start()
    sent = {k: v for k, v in payload.items() if k not in ('tools', 'tool_choice')}
    sent['messages'] = rest
    extra = dict(sent.get('extra_body') or {})
    extra['google'] = {**dict(extra.get('google') or {}), 'cached_content': live.name}
    sent['extra_body'] = extra
    sent[FULL_KEY] = payload
    return sent


def _admit(live: LiveCache) -> None:
    """Add a new cache; drop stale caches of the same endpoint and model, and keep within the cap."""
    evict: list[LiveCache] = []
    now = time.monotonic()
    with _LOCK:
        for other in list(_LIVE.values()):
            same = other.endpoint == live.endpoint
            if same and now - other.last_used > STALE_SECONDS:
                evict.append(other)
                _LIVE.pop(other.key, None)
        _LIVE[live.key] = live
        while len(_LIVE) > max_caches():
            oldest = min(_LIVE.values(), key=lambda c: c.last_used)
            _LIVE.pop(oldest.key, None)
            evict.append(oldest)
    for other in evict:
        _delete(other, 'replaced' if other.endpoint == live.endpoint else 'cache limit')


def cached_name(payload: dict[str, Any]) -> str | None:
    google = (payload.get('extra_body') or {}).get('google') if isinstance(payload.get('extra_body'), dict) else None
    return str(google.get('cached_content')) if isinstance(google, dict) and google.get('cached_content') else None


def request_failed(payload: dict[str, Any], exc: Exception) -> None:
    """A request that named a cache failed: drop that cache (it may have expired or been deleted)."""
    name = cached_name(payload)
    if not name:
        return
    with _LOCK:
        found = next((c for c in _LIVE.values() if c.name == name), None)
        if found is not None:
            _LIVE.pop(found.key, None)
    get_logger('explicit_cache').warning('request with explicit cache %s failed (%s); sent again without it',
                                         name, sanitize_log(str(exc), 200))
    if found is not None:
        threading.Thread(target=_delete, args=(found, 'request failed'), name='cache-drop', daemon=True).start()


# ---------------------------------------------------------------- display and clean-up

def live_caches() -> list[dict[str, Any]]:
    now = time.time()
    with _LOCK:
        items = list(_LIVE.values())
    return [{'name': c.name, 'model': c.endpoint.short_model, 'tokens': c.tokens,
             'expires_in_seconds': max(0, int(c.expire - now)), 'uses': c.uses,
             'storage_cost_usd': round(c.storage_cost(now), 6)} for c in items]


def _process_storage(now: float) -> float:
    """Storage cost of every cache this process created so far (call with _LOCK held)."""
    return _SPENT[0] + sum(c.storage_cost(now) for c in _LIVE.values())


def storage_cost_usd() -> float:
    """The active session's explicit cache storage cost: what its session file had when it was
    resumed (`base`) plus what this process spent since (`mark` is the process total then)."""
    now = time.time()
    with _LOCK:
        return _ACTIVE['base'] + max(0.0, _process_storage(now) - _ACTIVE['mark'])


def adopt_session_storage(session_id: str | None, stored: Any) -> None:
    """A resumed session (/resume, -c, -r, agent-resume): its saved storage cost becomes the base.
    Resuming the session that is already active changes nothing, so nothing is counted twice."""
    try:
        base = max(0.0, float(stored or 0.0))
    except (TypeError, ValueError) as exc:
        log_ignored_error(exc)
        base = 0.0
    now = time.time()
    with _LOCK:
        if not session_id or _ACTIVE['session'] == session_id:
            return
        _ACTIVE.update(session=session_id, base=base, mark=_process_storage(now))


def session_storage_usd(session_id: str | None) -> float:
    """The storage cost to save with `session_id`. A new session takes over this process's count
    when it is first saved; saving another session than the active one starts its own count."""
    now = time.time()
    with _LOCK:
        if session_id and _ACTIVE['session'] is None:
            _ACTIVE['session'] = session_id
        elif session_id and _ACTIVE['session'] != session_id:
            _ACTIVE.update(session=session_id, base=0.0, mark=_process_storage(now))
    return storage_cost_usd()


def reset_session_storage() -> None:
    """/clear: the next session starts its storage count at 0 (deleted caches stay paid for)."""
    now = time.time()
    with _LOCK:
        _ACTIVE.update(session=None, base=0.0, mark=_process_storage(now))


def delete(which: str) -> list[str]:
    """/cache delete <name|all>: delete this process's caches; returns the names deleted."""
    with _LOCK:
        chosen = [c for c in _LIVE.values() if which == 'all' or c.name == which or c.name.endswith('/' + which)]
        for c in chosen:
            _LIVE.pop(c.key, None)
    return [c.name for c in chosen if _delete(c, '/cache delete')]


def release_all(reason: str = 'session end') -> int:
    """Delete every cache this process created (exit, session end). Never raises."""
    with _LOCK:
        items = list(_LIVE.values())
        _LIVE.clear()
    count = 0
    for live in items:
        try:
            count += bool(_delete(live, reason, delete_timeout()))
        except Exception as exc:  # exit must not fail
            log_ignored_error(exc)
    return count


def reset_for_tests() -> None:
    with _LOCK:
        _LIVE.clear()
        _CREATING.clear()
        _FAILED_CREATES.clear()
        _DEAD.clear()
        _SAMPLES.clear()
        _REPORTED.clear()
        _SWEPT.clear()
        _NOTED.clear()
        _SESSION.clear()
        _SPENT[0] = 0.0
        _ACTIVE.update(session=None, base=0.0, mark=0.0)
        _START_SWEEP['thread'] = None
