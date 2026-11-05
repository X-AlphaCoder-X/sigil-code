"""Sending a model's own reasoning back, so a later request is never rejected for it.

Providers bind reasoning to the conversation and check it when it comes back:

- Anthropic, Claude 5.1 and later (platform.claude.com/docs/en/build-with-claude/preserved-thinking):
  a replayed ``thinking`` / ``redacted_thinking`` block is accepted only while the ``system``
  prompt, ``tools`` and messages before it are unchanged. Microcompaction, compaction that keeps
  recent turns, removed reminders and changed tools all edit that prefix, and the API answers
  "Invalid `signature` in `thinking` block. The block is bound to a different conversation".
  The documented fix is the ``thinking-binding-controls-2026-08-01`` beta with
  ``thinking.block_binding.prefix_mismatch_behavior: "drop_block"``; where that can't be sent,
  strip the named block and every later thinking block, then retry once.
- Amazon Bedrock (Converse): the same Claude blocks as ``reasoningContent``; the same strip.
- DeepSeek thinking mode with tools (api-docs.deepseek.com/guides/thinking_mode) and Kimi K3,
  K3.1 Pro and k2.7-code (platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model) need
  ``reasoning_content`` sent back on assistant messages, or answer HTTP 400.
- Gemini 3 (docs.cloud.google.com/vertex-ai/generative-ai/docs/thought-signatures) needs each
  tool call's ``extra_content.google.thought_signature`` back, or answers 400. A call the API did
  not write (history from another model) takes ``skip_thought_signature_validator``.

SIGIL_THINKING_BINDING = drop (default) | error | off chooses what Anthropic is asked to do.
"""
from __future__ import annotations

import copy
import os
import re
import threading
from typing import Any

BINDING_BETA = 'thinking-binding-controls-2026-08-01'
SKIP_SIGNATURE = 'skip_thought_signature_validator'
_THINKING_TYPES = ('thinking', 'redacted_thinking')

_SIGNATURE_ERROR_RE = re.compile(r'Invalid `?signature`? in `?thinking`? block', re.IGNORECASE)
_BLOCK_PATH_RE = re.compile(r'messages\.(\d+)\.content\.(\d+)')
_BINDING_REJECTED_RE = re.compile(r'block_binding|' + re.escape(BINDING_BETA), re.IGNORECASE)
_UNKNOWN_FIELD_RE = re.compile(r'extra inputs|not permitted|unknown|unrecognized|unexpected|not allowed|'
                               r'additional propert', re.IGNORECASE)
_REASONING_WANTED_RE = re.compile(r'reasoning_content', re.IGNORECASE)
_WANTED_WORDS_RE = re.compile(r'missing|must|required|passed back|pass back|need', re.IGNORECASE)
_THOUGHT_SIGNATURE_RE = re.compile(r'thought[_ ]signature', re.IGNORECASE)

_lock = threading.Lock()
_warned: set[str] = set()


def _status(text: str) -> int | None:
    match = re.match(r'HTTP (\d{3})\b', text)
    return int(match.group(1)) if match else None


def _log(message: str, *args: Any, once: str | None = None) -> None:
    if once is not None:
        with _lock:
            if once in _warned:
                return
            _warned.add(once)
    from .debug_log import get_logger

    get_logger('model').warning(message, *args)


# ---------------------------------------------------------------- Anthropic block binding ---

def binding_mode() -> str:
    """drop (default), error or off, from SIGIL_THINKING_BINDING."""
    value = os.environ.get('SIGIL_THINKING_BINDING', '').strip().lower()
    if value in ('off', '0', 'false', 'no', 'none'):
        return 'off'
    if value == 'error':
        return 'error'
    return 'drop'


def checks_prefix(model: str) -> bool:
    """Claude 5.1 and later check a replayed thinking block's prefix; earlier models don't."""
    from .providers import claude_version

    version = claude_version(model)
    if version is None:
        return False
    _family, major, minor = version
    return (major, minor) >= (5, 1)


def add_block_binding(payload: dict[str, Any], model: str) -> bool:
    """thinking.block_binding on a request to a model that checks the prefix. Sent on every request
    (not only once thinking is replayed) so the thinking setting, which is part of the cached prefix,
    stays the same for the whole conversation. Returns True when the beta header is needed.

    block_binding goes with thinking "adaptive" or "enabled". Without a thinking setting, Claude 5.1+
    models think adaptively by default, so {"type": "adaptive"} is the same request. Sonnet 5.5
    ("between_tools") and Haiku 5.5 ("disabled") can't take it; the strip-and-retry covers them."""
    mode = binding_mode()
    if mode == 'off' or not checks_prefix(model):
        return False
    thinking = payload.get('thinking')
    if thinking is None:
        thinking = {'type': 'adaptive'}
    if not isinstance(thinking, dict) or thinking.get('type') not in ('adaptive', 'enabled'):
        return False
    behavior = 'drop_block' if mode == 'drop' else 'error'
    payload['thinking'] = dict(thinking, block_binding={'prefix_mismatch_behavior': behavior})
    return True


def without_block_binding(payload: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(payload)
    thinking = stripped.get('thinking')
    if isinstance(thinking, dict) and 'block_binding' in thinking:
        thinking = {k: v for k, v in thinking.items() if k != 'block_binding'}
        stripped['thinking'] = thinking
    return stripped


def beta_header(existing: str | None) -> str:
    """The anthropic-beta value with the binding beta added to any the user set."""
    parts = [p.strip() for p in str(existing or '').split(',') if p.strip()]
    if BINDING_BETA not in parts:
        parts.append(BINDING_BETA)
    return ','.join(parts)


def binding_rejected(text: str) -> bool:
    """A 400 about the block_binding field or its beta (a proxy that drops the beta header)."""
    return _status(text) in (400, 422) and _BINDING_REJECTED_RE.search(text) is not None \
        and _SIGNATURE_ERROR_RE.search(text) is None


def signature_error(text: str) -> tuple[int, int] | None:
    """(message index, block index) of the thinking block a 400 names, (-1, -1) when it names none,
    or None when the error is not about a replayed thinking signature."""
    if _status(text) not in (400, 422) or _SIGNATURE_ERROR_RE.search(text) is None:
        return None
    match = _BLOCK_PATH_RE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)


def _block_key(block: dict[str, Any]) -> str | None:
    """What identifies a replayed reasoning block: its signature (thinking, Bedrock reasoningText),
    its data (redacted_thinking) or Bedrock's redactedContent."""
    if block.get('type') == 'thinking':
        return str(block.get('signature') or '') or None
    if block.get('type') == 'redacted_thinking':
        return str(block.get('data') or '') or None
    reasoning = block.get('reasoningContent')
    if isinstance(reasoning, dict):
        text = reasoning.get('reasoningText')
        if isinstance(text, dict) and text.get('signature'):
            return str(text['signature'])
        if reasoning.get('redactedContent'):
            return str(reasoning['redactedContent'])
    return None


def is_reasoning_block(block: Any) -> bool:
    return isinstance(block, dict) and (block.get('type') in _THINKING_TYPES or 'reasoningContent' in block)


def strip_from(payload: dict[str, Any], position: tuple[int, int]) -> tuple[dict[str, Any], set[str]]:
    """A copy of the payload without the named reasoning block and every later one (all of them
    when the error named none), and the keys of what was removed. Each turn's other blocks stay;
    an assistant turn left empty gets a one-space text block."""
    out = dict(payload)
    messages = copy.deepcopy(payload.get('messages') or [])
    msg_index, block_index = position
    removed: set[str] = set()
    for i, message in enumerate(messages):
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        kept = []
        for j, block in enumerate(content):
            later = msg_index < 0 or i > msg_index or (i == msg_index and j >= block_index)
            if later and is_reasoning_block(block):
                key = _block_key(block)
                if key:
                    removed.add(key)
                continue
            kept.append(block)
        if not kept:
            kept = [{'type': 'text', 'text': ' '}] if 'type' in (content[0] if content else {}) else [{'text': ' '}]
        message['content'] = kept
    out['messages'] = messages
    return out, removed


def drop_known(messages: list[dict[str, Any]], dropped: set[str]) -> None:
    """Remove (in place) reasoning blocks already stripped once in this session, so the same
    rejected block is not sent again on every later request."""
    if not dropped:
        return
    for message in messages:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        kept = [b for b in content if not (is_reasoning_block(b) and _block_key(b) in dropped)]
        if len(kept) != len(content):
            message['content'] = kept or [{'type': 'text', 'text': ' '}]


def log_transformations(data: Any, model: str) -> None:
    """input_transformations from a binding-beta response: a prefix_binding_mismatch means something
    before a replayed block changed (logged once per model; the request itself succeeded)."""
    if not isinstance(data, dict):
        return
    entries = data.get('input_transformations')
    if not isinstance(entries, list):
        return
    for entry in entries:
        if isinstance(entry, dict) and entry.get('reason') == 'prefix_binding_mismatch':
            _log('Anthropic dropped replayed thinking for %s (%s at %s): an earlier part of the conversation '
                 'changed, so that reasoning was not reused', model, entry.get('type'), entry.get('path'),
                 once=f'transform:{model}')
            return


# ------------------------------------------------- OpenAI-compatible reasoning round trip ---

def endpoint_key(base_url: str, model: str) -> str:
    return f'{str(base_url or "").rstrip("/").lower()}|{str(model or "").lower()}'


def capture(message: dict[str, Any], base_url: str, model: str) -> dict[str, Any] | None:
    """What a reply needs sent back later: reasoning_content and each tool call's extra_content."""
    if not isinstance(message, dict):
        return None
    reasoning = message.get('reasoning_content')
    extras: dict[str, Any] = {}
    for call in message.get('tool_calls') or []:
        if isinstance(call, dict) and call.get('id') and isinstance(call.get('extra_content'), dict):
            extras[str(call['id'])] = copy.deepcopy(call['extra_content'])
    if not isinstance(reasoning, str):
        reasoning = None
    if reasoning is None and not extras:
        return None
    replay: dict[str, Any] = {'endpoint': endpoint_key(base_url, model)}
    if reasoning is not None:
        replay['reasoning_content'] = reasoning
    if extras:
        replay['tool_extras'] = extras
    return replay


def echoes_reasoning(provider: str, model: str) -> bool:
    """Providers whose docs say reasoning_content must go back (DeepSeek with tools, Kimi K3,
    K3.1 Pro and k2.7-code); SIGIL_REASONING_REPLAY=on sends it to every server, off to none."""
    value = os.environ.get('SIGIL_REASONING_REPLAY', '').strip().lower()
    if value in ('off', '0', 'false', 'no'):
        return False
    if value in ('on', '1', 'true', 'yes'):
        return True
    name = (model or '').lower()
    if provider == 'deepseek':
        return True
    return provider == 'moonshot' and any(tag in name for tag in ('kimi-k3', 'k2.7'))


def is_gemini3(provider: str, model: str) -> bool:
    name = (model or '').lower()
    return provider in ('gemini', 'vertex') and re.search(r'gemini-(\d+)', name) is not None \
        and int(re.search(r'gemini-(\d+)', name).group(1)) >= 3


def prepare(messages: list[Any], *, base_url: str, model: str, provider: str, echo: bool,
            fill: bool, dummy_signatures: bool, send_fields: bool = True) -> list[Any]:
    """Messages for an OpenAI-compatible request. Internal keys (anthropic_content, provider_replay)
    never leave Sigil. A reply's reasoning_content and tool-call extra_content go back only to the
    endpoint that wrote them. fill sends reasoning_content "" on assistant messages without any
    (after a server asked for it); dummy_signatures gives Gemini 3 tool calls the API didn't write
    the documented skip value."""
    key = endpoint_key(base_url, model)
    gemini3 = dummy_signatures or is_gemini3(provider, model)
    out: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        replay = message.get('provider_replay')
        clean = {k: v for k, v in message.items() if k not in ('anthropic_content', 'provider_replay')}
        if clean.get('role') != 'assistant' or not send_fields:
            out.append(clean)
            continue
        own = isinstance(replay, dict) and replay.get('endpoint') == key
        if echo:
            reasoning = replay.get('reasoning_content') if own else None
            if isinstance(reasoning, str):
                clean['reasoning_content'] = reasoning
            elif fill:
                clean['reasoning_content'] = ''
        calls = clean.get('tool_calls')
        if isinstance(calls, list) and calls:
            extras = replay.get('tool_extras') if own and isinstance(replay.get('tool_extras'), dict) else {}
            new_calls = []
            for call in calls:
                if not isinstance(call, dict):
                    new_calls.append(call)
                    continue
                extra = extras.get(str(call.get('id')))
                if isinstance(extra, dict):
                    call = dict(call, extra_content=copy.deepcopy(extra))
                new_calls.append(call)
            # Gemini 3 signs only the FIRST call of a step (later parallel calls carry none), and the
            # skip value is for calls the API did not write. So it goes on the first call, and only when
            # no call of this step has a real signature (history from another model).
            signed = any(isinstance(c, dict) and 'extra_content' in c for c in new_calls)
            if gemini3 and not signed and isinstance(new_calls[0], dict):
                new_calls[0] = dict(new_calls[0], extra_content={'google': {'thought_signature': SKIP_SIGNATURE}})
            clean['tool_calls'] = new_calls
        out.append(clean)
    return out


def classify_compat_error(text: str) -> str | None:
    """What a 400/422 asks for: 'reasoning' (send reasoning_content back), 'signature' (thought
    signatures on tool calls), 'reject' (the server refuses those fields), or None."""
    if _status(text) not in (400, 422):
        return None
    names_reasoning = _REASONING_WANTED_RE.search(text) is not None
    names_extra = 'extra_content' in text
    if (names_reasoning or names_extra) and _UNKNOWN_FIELD_RE.search(text):
        return 'reject'
    if names_reasoning and _WANTED_WORDS_RE.search(text):
        return 'reasoning'
    if _THOUGHT_SIGNATURE_RE.search(text):
        return 'signature'
    return None


def note(kind: str, model: str) -> None:
    messages = {
        'reasoning': 'the server asked for reasoning_content to be sent back; Sigil now sends it '
                     '(empty where a reply had none) for %s',
        'signature': 'the server asked for thought signatures on tool calls; calls without one now carry '
                     'Google\'s skip value for %s',
        'reject': 'the server refused reasoning_content / extra_content; they are no longer sent to %s',
        'binding': 'the server refused thinking.block_binding; it is no longer sent to %s',
        'strip': 'a replayed thinking block was bound to a different conversation; it and the later '
                 'thinking blocks were left out and the request sent again (%s)',
    }
    _log('Reasoning replay: ' + messages[kind], model, once=None if kind == 'strip' else f'{kind}:{model}')
