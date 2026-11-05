"""Ask the model server for the context window and vision support (fewer env vars to set).

Tried in order, with a short timeout, once per server and model:
  - <base>/models (OpenAI style): context_length, max_context_length (LM Studio),
    max_model_len (vLLM), context_window, top_provider.context_length (OpenRouter);
    input_modalities / architecture.input_modalities containing "image" means vision.
  - Ollama <root>/api/show: model_info "<arch>.context_length", a num_ctx parameter,
    and "vision" in capabilities.
At start-up the CLI uses the result only for what was not set by hand: auto compaction
at SIGIL_AUTO_COMPACT_RATIO (0.8) of the window when --auto-compact-threshold is not
given, and SIGIL_VISION=0 for a model the server reports as text-only.
SIGIL_DETECT_MODEL=0 turns the lookup off.

Prices: the same list gives each model's prices where the server publishes them (OpenRouter:
`pricing` in USD per token; NanoGPT with ?detailed=true: per million tokens). The user's own
table (`modelPrices` setting or SIGIL_MODEL_PRICES) wins over the server. model_prices() is what
a model switch uses to price the new model.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

# (base URL, model) -> ModelInfo, and (base URL, _LIST) -> that server's /models answer (one request
# per server; a failed request is not kept, so it is tried again for the next model)
_CACHE: dict[tuple[str, str], Any] = {}
_LIST = '\x00models'
PRICE_KEYS = ('input', 'output', 'cache_read', 'cache_write', 'cache_write_1h')


@dataclass(frozen=True)
class ModelInfo:
    context_window: int | None = None
    vision: bool | None = None
    source: str = ''
    prices: dict[str, float] | None = None  # USD per million tokens, keys from PRICE_KEYS


def _get(url: str, payload: dict | None, api_key: str, timeout: float) -> Any:
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = request.Request(url, data=data, headers=headers, method='POST' if data else 'GET')
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def _int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None
    return number if number > 0 else None


def _price(value: Any, scale: float) -> float | None:
    """A non-negative price as USD per million tokens, or None (missing, negative, not a number).
    OpenRouter uses "-1" for models whose price depends on the route."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None
    if number < 0 or number != number or number == float('inf'):
        return None
    return number * scale


def prices_from_entry(entry: Any) -> dict[str, float] | None:
    """Prices from one /models entry: OpenRouter (strings, USD per token) or NanoGPT
    (unit "per_million_tokens"; cache prices per 1k tokens)."""
    pricing = entry.get('pricing') if isinstance(entry, dict) else None
    if not isinstance(pricing, dict):
        return None
    if str(pricing.get('unit') or '').lower() == 'per_million_tokens':
        found = {'input': _price(pricing.get('prompt'), 1.0), 'output': _price(pricing.get('completion'), 1.0),
                 'cache_read': _price(pricing.get('cacheReadInputPer1kTokens'), 1000.0),
                 'cache_write': _price(pricing.get('cacheWriteInputPer1kTokens'), 1000.0)}
    else:
        found = {'input': _price(pricing.get('prompt'), 1e6), 'output': _price(pricing.get('completion'), 1e6),
                 'cache_read': _price(pricing.get('input_cache_read'), 1e6),
                 'cache_write': _price(pricing.get('input_cache_write'), 1e6),
                 'cache_write_1h': _price(pricing.get('input_cache_write_1h'), 1e6)}
    found = {key: value for key, value in found.items() if value is not None}
    return found if ('input' in found or 'output' in found) else None


def _from_models_list(payload: Any, model: str) -> ModelInfo | None:
    items = payload.get('data') if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return None
    entry = next((m for m in items if isinstance(m, dict) and m.get('id') == model), None)
    if entry is None:
        return None
    prices = prices_from_entry(entry)
    window = None
    for key in ('context_length', 'max_context_length', 'max_model_len', 'context_window', 'max_input_tokens'):
        window = window or _int(entry.get(key))
    provider = entry.get('top_provider')
    if window is None and isinstance(provider, dict):
        window = _int(provider.get('context_length'))
    modalities = entry.get('input_modalities') or (entry.get('architecture') or {}).get('input_modalities')
    vision = None
    if isinstance(modalities, list):
        vision = 'image' in modalities
    elif entry.get('type') in ('vlm', 'llm'):  # LM Studio
        vision = entry.get('type') == 'vlm'
    if window is None and vision is None and prices is None:
        return None
    return ModelInfo(window, vision, 'models', prices)


def _from_ollama(payload: Any) -> ModelInfo | None:
    if not isinstance(payload, dict):
        return None
    window = None
    match = re.search(r'num_ctx\s+(\d+)', str(payload.get('parameters') or ''))
    if match:
        window = _int(match.group(1))
    info = payload.get('model_info')
    if window is None and isinstance(info, dict):
        window = next((_int(v) for k, v in info.items() if str(k).endswith('.context_length') and _int(v)), None)
    capabilities = payload.get('capabilities')
    vision = 'vision' in capabilities if isinstance(capabilities, list) else None
    if window is None and vision is None:
        return None
    return ModelInfo(window, vision, 'ollama')


def _looks_local(base_url: str) -> bool:
    """Ollama's /api/show is only tried on local or private servers (not on cloud APIs)."""
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or '').lower()
    return (host in ('localhost', '0.0.0.0', '::1') or host.startswith(('127.', '10.', '192.168.'))
            or re.match(r'^172\.(1[6-9]|2\d|3[01])\.', host) is not None or host.endswith('.local')
            or ':11434' in base_url)


def detect(base_url: str, model: str, api_key: str = '', timeout: float = 3.0) -> ModelInfo:
    if os.environ.get('SIGIL_DETECT_MODEL', '1').strip().lower() in ('0', 'false', 'no', 'off'):
        return ModelInfo()
    key = (base_url.rstrip('/'), model)
    if key in _CACHE:
        return _CACHE[key]
    found = ModelInfo()
    base = base_url.rstrip('/')
    listing = _models_list(base, api_key, timeout)
    if listing is not None:
        found = _from_models_list(listing, model) or found
    if (found.context_window is None or found.vision is None) and _looks_local(base):
        root = base[:-3] if base.endswith('/v1') else base
        try:
            ollama = _from_ollama(_get(root + '/api/show', {'model': model}, api_key, timeout))
        except (error.URLError, OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            ollama = None
        if ollama is not None:
            found = ModelInfo(found.context_window or ollama.context_window,
                              found.vision if found.vision is not None else ollama.vision,
                              found.source or ollama.source, found.prices)
    _CACHE[key] = found
    return found


def _models_list(base: str, api_key: str, timeout: float) -> Any:
    """The server's /models answer, fetched once per server (NanoGPT lists prices only with
    ?detailed=true). None when the server cannot be asked."""
    if (base, _LIST) in _CACHE:
        return _CACHE[(base, _LIST)]
    from .prompt_cache import provider_of

    url = base + '/models' + ('?detailed=true' if provider_of(base) == 'nanogpt' else '')
    try:
        listing = _get(url, None, api_key, timeout)
    except (error.URLError, OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None
    _CACHE[(base, _LIST)] = listing
    return listing


def _price_table() -> dict[str, Any]:
    """The user's prices: the `modelPrices` setting, then SIGIL_MODEL_PRICES (a JSON object or the
    path of a JSON file), later entries winning. {"model" or "model@base_url": {"input": 3,
    "output": 15, "cache_read": 0.3, "cache_write": 3.75, "cache_write_1h": 6}}, USD per million."""
    table: dict[str, Any] = {}
    try:
        from .settings import load_settings, project_dir

        configured = load_settings(project_dir()).get('modelPrices')
        if isinstance(configured, dict):
            table.update(configured)
    except Exception as _ignored:  # a broken settings file is reported by settings.py
        _log_ignored(_ignored)
    raw = os.environ.get('SIGIL_MODEL_PRICES', '').strip()
    if raw:
        try:
            text = raw if raw.startswith('{') else Path(raw).expanduser().read_text(encoding='utf-8')
            parsed = json.loads(text)
        except (OSError, ValueError) as exc:
            from .prompt_cache import _warn_once

            _warn_once(f'prices:{raw[:40]}', f'SIGIL_MODEL_PRICES could not be read ({type(exc).__name__}); '
                       'it is ignored.')
            parsed = None
        if isinstance(parsed, dict):
            table.update(parsed)
    return table


def user_prices(base_url: str, model: str) -> dict[str, float]:
    """The user's table entry for model@base_url, else for the model name."""
    table = _price_table()
    entry = table.get(f'{model}@{str(base_url).rstrip("/")}', table.get(model))
    if not isinstance(entry, dict):
        return {}
    return {key: value for key in PRICE_KEYS
            if (value := _price(entry.get(key), 1.0)) is not None}


def model_prices(base_url: str, model: str, api_key: str = '') -> dict[str, float]:
    """Known prices for model at base_url (USD per million tokens): the server's published prices
    (when SIGIL_DETECT_MODEL is on) with the user's table on top. Empty when nothing is known."""
    server = detect(str(base_url), str(model), api_key).prices or {}
    return {**server, **user_prices(base_url, model)}


def apply_detected(agent: Any, *, threshold_given: bool) -> ModelInfo:
    """Use the server's answer for settings the user did not give."""
    from dataclasses import replace

    config = agent.model_config
    info = detect(str(config.base_url), str(config.model), str(getattr(config, 'api_key', '') or ''))
    if info.context_window and not threshold_given and agent.runtime_config.auto_compact_threshold_tokens is None:
        try:
            ratio = float(os.environ.get('SIGIL_AUTO_COMPACT_RATIO', '0.8'))
        except ValueError as _ignored:
            _log_ignored(_ignored)
            ratio = 0.8
        agent.runtime_config = replace(agent.runtime_config,
                                       auto_compact_threshold_tokens=int(info.context_window * ratio))
    if info.vision is False and 'SIGIL_VISION' not in os.environ:
        os.environ['SIGIL_VISION'] = '0'
    if info.source:
        try:
            from .debug_log import get_logger

            get_logger('model_info').info('server says %s: context window %s, vision %s', config.model,
                                          info.context_window, info.vision)
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
    return info
