"""Reasoning output and text-form tool calls from local models.

Reasoning: ``<think>...</think>`` blocks (Qwen3, DeepSeek-R1 on llama.cpp or
Ollama) and ``reasoning_content`` are taken out of the reply, so old reasoning
is not sent back to the model and does not fill the context.
``SIGIL_STRIP_THINK=0`` keeps them in the text.

Tool calls written as text, for servers without a tool-call parser:
    <tool_call>{"name": "read_file", "arguments": {...}}</tool_call>   (Hermes)
    <function=read_file><parameter=path>a.py</parameter></function>   (Qwen3-Coder)
    {"name": "read_file", "arguments": {...}}                         (bare JSON reply)
Only names of real tools are accepted. ``SIGIL_TEXT_TOOL_CALLS=0`` turns this off.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.S | re.I)
_HERMES_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
_QWEN_FUNC_RE = re.compile(r'<function=([\w.\-]+)>(.*?)</function>', re.S)
_QWEN_PARAM_RE = re.compile(r'<parameter=([\w.\-]+)>(.*?)</parameter>', re.S)
_FENCE_RE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.S)


def _flag(name: str) -> bool:
    return os.environ.get(name, '1').strip().lower() not in {'0', 'false', 'no', 'off'}


def strip_think_enabled() -> bool:
    return _flag('SIGIL_STRIP_THINK')


def text_tool_calls_enabled() -> bool:
    return _flag('SIGIL_TEXT_TOOL_CALLS')


def split_reasoning(content: str, reasoning_field: Any = None) -> tuple[str, str]:
    """(visible text, reasoning). Handles a missing opening <think> tag too."""
    reasoning_parts: list[str] = []
    if isinstance(reasoning_field, str) and reasoning_field.strip():
        reasoning_parts.append(reasoning_field.strip())
    text = content or ''
    if not strip_think_enabled() or 'think>' not in text.lower():
        # Nothing to take out: keep the text exactly (a cut-off reply may end in a space).
        return text, '\n\n'.join(reasoning_parts)
    reasoning_parts.extend(part.strip() for part in _THINK_RE.findall(text) if part.strip())
    text = _THINK_RE.sub('', text)
    if '</think>' in text.lower():
        # Some chat templates add the opening tag themselves: "reasoning</think>answer".
        index = text.lower().index('</think>')
        if text[:index].strip():
            reasoning_parts.append(text[:index].strip())
        text = text[index + len('</think>'):]
    if text.lower().lstrip().startswith('<think>'):
        # Unclosed think block (cut off): everything is reasoning.
        reasoning_parts.append(text.lstrip()[len('<think>'):].strip())
        text = ''
    return text.strip(), '\n\n'.join(part for part in reasoning_parts if part)


def _call_from_json(raw: Any, tool_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(raw, dict):
        return None
    function = raw.get('function') if isinstance(raw.get('function'), dict) else {}
    name = raw.get('name') or function.get('name')
    if not isinstance(name, str) or name not in tool_names:
        return None
    arguments = function.get('arguments', raw.get('arguments', raw.get('parameters', {})))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError as _ignored:
            _log_ignored(_ignored)
            return None
    if not isinstance(arguments, dict):
        return None
    return name, arguments


def _coerce_param(value: str) -> Any:
    stripped = value.strip()
    if stripped[:1] in '[{' or stripped in ('true', 'false', 'null') or re.fullmatch(r'-?\d+(\.\d+)?', stripped):
        try:
            return json.loads(stripped)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            pass
    return value.strip('\n')


def parse_text_tool_calls(content: str, tool_names: Iterable[str]) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Tool calls written as text. Returns (text left over, [(name, arguments)])."""
    names = set(tool_names)
    calls: list[tuple[str, dict[str, Any]]] = []
    text = content or ''
    if not names or not text.strip():
        return text, calls
    if '<tool_call>' in text:
        for raw in _HERMES_RE.findall(text):
            try:
                call = _call_from_json(json.loads(raw), names)
            except ValueError as _ignored:
                _log_ignored(_ignored)
                call = None
            if call is not None:
                calls.append(call)
        if calls:
            return _HERMES_RE.sub('', text).strip(), calls
    if '<function=' in text:
        for name, body in _QWEN_FUNC_RE.findall(text):
            if name in names:
                arguments = {key: _coerce_param(value) for key, value in _QWEN_PARAM_RE.findall(body)}
                calls.append((name, arguments))
        if calls:
            leftover = _QWEN_FUNC_RE.sub('', text).replace('<tool_call>', '').replace('</tool_call>', '')
            return leftover.strip(), calls
    candidate = text.strip()
    fence = _FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    if candidate[:1] in '[{':
        try:
            payload = json.loads(candidate)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            return text, calls
        items = payload if isinstance(payload, list) else [payload]
        parsed = [_call_from_json(item, names) for item in items]
        if parsed and all(item is not None for item in parsed):
            return '', [item for item in parsed if item is not None]
    return text, calls
