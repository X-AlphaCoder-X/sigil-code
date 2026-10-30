"""Provider detection and per-provider request quirks.

OpenAI-compatible servers differ in small ways: which field controls
reasoning effort, whether they emit <think> blocks, and whether they honor
``response_format``. This module detects the provider from the base URL and
model name and returns those quirks, so ``openai_compat`` can build the right
payload.

Effort levels are ``low``, ``medium``, ``high``, ``max`` and ``auto``
(``auto`` sends nothing and lets the server decide).

Overrides:
    SIGIL_EFFORT_FIELD=reasoning_effort|chat_template_kwargs|thinking|none
        forces the field used for effort, for servers this module guesses wrong.
    SIGIL_EXCLUDE_PARAMS still strips any key from the final payload.
"""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import Any

EFFORT_LEVELS = ('off', 'low', 'medium', 'high', 'xhigh', 'max', 'auto')  # off = no thinking (Alt+T)
EFFORT_FIELDS = ('reasoning_effort', 'chat_template_kwargs', 'thinking', 'none')

# Anthropic extended-thinking budgets per level (the API minimum is 1024).
_THINKING_BUDGETS = {'low': 1024, 'medium': 4096, 'high': 16000, 'xhigh': 24000, 'max': 32000}


def claude_version(model: str) -> tuple[str, int, int] | None:
    """('opus'|'sonnet'|'haiku'|'fable'|'mythos', major, minor) from a Claude model id, or None."""
    import re

    name = str(model).lower()
    match = re.search(r'claude-(opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d+))?', name)
    if match:
        family, major, minor = match.group(1), int(match.group(2)), match.group(3)
    else:
        match = re.search(r'claude-(\d+)(?:-(\d+))?-(opus|sonnet|haiku)', name)  # claude-3-7-sonnet
        if not match:
            return ('mythos', 9, 0) if 'mythos' in name else None
        major, minor, family = int(match.group(1)), match.group(2), match.group(3)
    minor_value = int(minor) if minor and len(minor) <= 2 else 0  # claude-sonnet-4-20250514: a date
    return family, major, minor_value


def claude_thinking(model: str, level: str | None) -> tuple[dict | None, str | None, bool]:
    """(thinking parameter, output_config.effort, temperature allowed) for a Claude model.

    Claude 4.5 and earlier: {"type": "enabled", "budget_tokens": N}. Claude 4.6 and later:
    {"type": "adaptive"} plus output_config.effort; "enabled" is rejected from 4.7 on. Off:
    4.6-4.8 omit thinking; Sonnet 5, Opus 5 and Haiku 5.5 take {"type": "disabled"}; Sonnet 5.5
    goes down to {"type": "between_tools"}; Opus 5.5, Fable and Mythos cannot turn it off (lowest
    effort instead). From 4.7 (and Mythos) a temperature below 1 is rejected, so it is not sent.
    """
    version = claude_version(model)
    family, major, minor = version if version else ('', 99, 0)
    modern = family == 'mythos' or major > 4 or (major == 4 and minor >= 6)
    temperature_ok = not (family == 'mythos' or major > 4 or (major == 4 and minor >= 7))
    if not modern:  # 4.5 and earlier: manual budget only
        if level and level != 'off':
            return {'type': 'enabled', 'budget_tokens': _THINKING_BUDGETS[level]}, None, False
        return None, None, True
    if level == 'off':
        if major == 4:
            return None, None, temperature_ok
        if (family, major, minor) in (('sonnet', 5, 0), ('opus', 5, 0), ('haiku', 5, 5)):
            return {'type': 'disabled'}, None, temperature_ok
        if (family, major, minor) == ('sonnet', 5, 5):
            return {'type': 'between_tools'}, None, temperature_ok
        return {'type': 'adaptive'}, 'low', temperature_ok
    if level:
        return {'type': 'adaptive'}, level, temperature_ok
    return None, None, temperature_ok


@dataclass(frozen=True)
class ProviderQuirks:
    name: str
    effort_field: str = 'none'
    # True when the model may write reasoning inside <think>...</think>.
    think_tags: bool = False
    supports_response_format: bool = True


def detect_provider(base_url: str, model: str = '') -> str:
    """Best guess at the server behind an OpenAI-compatible base URL."""
    parsed = urllib.parse.urlparse(base_url or '')
    host = (parsed.hostname or '').lower()
    port = parsed.port
    if host.endswith('anthropic.com'):
        return 'anthropic'
    if host.endswith('openrouter.ai'):
        return 'openrouter'
    if host.endswith('openai.com'):
        return 'openai'
    if 'ollama' in host or port == 11434:
        return 'ollama'
    if 'lmstudio' in host or port == 1234:
        return 'lmstudio'
    if 'llama' in host or port == 8080:
        return 'llamacpp'
    if 'vllm' in host or port == 8000:
        return 'vllm'
    return 'openai-compatible'


def _model_family(model: str) -> str:
    name = (model or '').lower()
    if 'gpt-oss' in name:
        return 'gpt-oss'
    if 'qwen3' in name:
        return 'qwen3'
    if any(tag in name for tag in ('deepseek-r1', 'qwq', 'r1-distill')):
        return 'reasoner'
    return ''


def get_quirks(base_url: str, model: str = '') -> ProviderQuirks:
    provider = detect_provider(base_url, model)
    family = _model_family(model)
    think_tags = family in ('qwen3', 'reasoner')
    if provider == 'anthropic':
        # The compat layer ignores reasoning_effort and response_format.
        quirks = ProviderQuirks(provider, 'thinking', False, False)
    elif provider in ('openai', 'openrouter'):
        quirks = ProviderQuirks(provider, 'reasoning_effort', think_tags, True)
    elif family == 'gpt-oss' or provider == 'ollama':
        quirks = ProviderQuirks(provider, 'reasoning_effort', think_tags, True)
    elif family == 'qwen3':
        quirks = ProviderQuirks(provider, 'chat_template_kwargs', True, True)
    else:
        quirks = ProviderQuirks(provider, 'none', think_tags, True)
    forced = os.environ.get('SIGIL_EFFORT_FIELD', '').strip().lower()
    if forced in EFFORT_FIELDS:
        quirks = ProviderQuirks(quirks.name, forced, quirks.think_tags, quirks.supports_response_format)
    return quirks


def normalize_effort(level: str | None) -> str | None:
    """Lower-case a level and drop 'auto' / unknown values (None = send nothing)."""
    if not level:
        return None
    value = str(level).strip().lower()
    value = 'off' if value == 'none' else value  # OpenAI's word for off
    if value not in EFFORT_LEVELS or value == 'auto':
        return None
    return value


def apply_effort(payload: dict[str, Any], level: str | None, quirks: ProviderQuirks) -> dict[str, Any]:
    """Add the provider's effort field to a chat payload (in place, also returned)."""
    level = normalize_effort(level)
    if level is None or quirks.effort_field == 'none':
        return payload
    if quirks.effort_field == 'reasoning_effort':
        # 'none' turns reasoning off (OpenAI gpt-5.1+, OpenRouter, vLLM, Ollama); a model that
        # rejects it gets 'low' instead (openai_compat retries once).
        payload['reasoning_effort'] = {'max': 'high', 'xhigh': 'high', 'off': 'none'}.get(level, level)
    elif quirks.effort_field == 'chat_template_kwargs':
        kwargs = dict(payload.get('chat_template_kwargs') or {})
        kwargs['enable_thinking'] = level not in ('low', 'off')
        payload['chat_template_kwargs'] = kwargs
    elif quirks.effort_field == 'thinking':
        # Anthropic's OpenAI-compatible endpoint: only `thinking` (reasoning_effort is ignored there).
        thinking, _effort, temperature_ok = claude_thinking(str(payload.get('model', '')), level)
        if thinking is not None:
            payload['thinking'] = thinking
        if not temperature_ok:
            payload.pop('temperature', None)
            payload.pop('top_p', None)
    return payload


CLAUDE_MODEL_ALIASES = ('sonnet', 'opus', 'haiku', 'fable')


def resolve_model_alias(value: str | None) -> str | None:
    """A model name from command or agent frontmatter, or None to keep the current one.

    'inherit' keeps the session model. The Claude aliases (sonnet, opus, haiku,
    fable) only mean something on Anthropic's API, so they map through
    SIGIL_MODEL_<ALIAS> (for example SIGIL_MODEL_HAIKU=qwen3-4b) and otherwise
    keep the session model. Any other value is used as given.
    """
    if not value:
        return None
    name = str(value).strip()
    if not name or name.lower() == 'inherit':
        return None
    if name.lower() in CLAUDE_MODEL_ALIASES:
        return os.environ.get(f'SIGIL_MODEL_{name.upper()}') or None
    return name


def effective_effort(configured: str | None) -> str | None:
    """SIGIL_EFFORT_LEVEL wins over the session setting (same rule as /effort)."""
    env_value = os.environ.get('SIGIL_EFFORT_LEVEL')
    return normalize_effort(env_value) if env_value else normalize_effort(configured)
