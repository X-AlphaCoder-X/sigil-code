from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .agent_prompting import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from .agent_session import AgentMessage, AgentSessionState
from .tokenizer_runtime import describe_token_counter, count_tokens
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_PATH_HEADER_RE = re.compile(r'^## ((?:/|[A-Za-z]:[\\/]).+)$', re.MULTILINE)


@dataclass(frozen=True)
class UsageEntry:
    name: str
    tokens: int


@dataclass(frozen=True)
class ToolUsageEntry:
    name: str
    call_tokens: int
    result_tokens: int


@dataclass(frozen=True)
class MessageBreakdown:
    user_message_tokens: int
    assistant_message_tokens: int
    tool_call_tokens: int
    tool_result_tokens: int
    user_context_tokens: int
    tool_calls_by_type: tuple[ToolUsageEntry, ...]


@dataclass(frozen=True)
class ContextUsageReport:
    model: str
    total_tokens: int
    raw_max_tokens: int
    percentage: float
    strategy: str
    message_count: int
    categories: tuple[UsageEntry, ...]
    system_prompt_sections: tuple[UsageEntry, ...]
    user_context_entries: tuple[UsageEntry, ...]
    system_context_entries: tuple[UsageEntry, ...]
    memory_files: tuple[UsageEntry, ...]
    message_breakdown: MessageBreakdown
    token_counter_backend: str
    token_counter_source: str
    token_counter_accurate: bool


def estimate_tokens(text: str, model: str | None = None) -> int:
    return count_tokens(text, model)


def infer_context_window(model: str) -> int:
    lowered = model.lower()

    # --- 256K+ models ---
    if 'qwen3-coder' in lowered:
        return 256_000
    if 'devstral' in lowered:
        return 256_000
    if 'gemini-2.5' in lowered or 'gemini-2.0' in lowered:
        return 1_048_576
    if 'gemini-1.5-pro' in lowered:
        return 2_097_152
    if 'gemini-1.5-flash' in lowered:
        return 1_048_576
    if 'gemini' in lowered:
        return 1_048_576

    # --- 200K models ---
    if 'claude' in lowered:
        return 200_000

    # --- 128K models ---
    if 'glm-4' in lowered or 'glm-5' in lowered or 'glm4' in lowered or 'chatglm' in lowered:
        return 128_000
    if 'qwen2.5' in lowered or 'qwen2' in lowered:
        return 131_072
    if 'qwen' in lowered:
        return 131_072
    if 'gpt-4.1' in lowered or 'gpt-4o' in lowered:
        return 128_000
    if 'gpt-4-turbo' in lowered or 'gpt-4-0125' in lowered or 'gpt-4-1106' in lowered:
        return 128_000
    if 'gpt-5' in lowered:
        return 128_000
    if 'deepseek-v3' in lowered or 'deepseek-r1' in lowered or 'deepseek-coder-v2' in lowered:
        return 128_000
    if 'deepseek' in lowered:
        return 64_000
    if 'yi-large' in lowered or 'yi-1.5' in lowered:
        return 200_000
    if 'yi' in lowered:
        return 32_000
    if 'codestral' in lowered:
        return 256_000
    if 'mistral-large' in lowered or 'mistral-medium' in lowered:
        return 128_000
    if 'mistral' in lowered or 'mixtral' in lowered:
        return 32_000
    if 'llama-3.3' in lowered or 'llama-3.1' in lowered:
        return 131_072
    if 'llama' in lowered:
        return 8_192
    if 'phi-4' in lowered or 'phi-3' in lowered:
        return 128_000
    if 'command-r-plus' in lowered or 'command-r' in lowered:
        return 128_000

    # --- 32K models ---
    if 'gpt-4-32k' in lowered:
        return 32_768
    if 'gpt-4' in lowered:
        return 8_192

    # --- Fallback ---
    env_override = os.environ.get('SIGIL_CONTEXT_WINDOW')
    if env_override:
        try:
            return int(env_override)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            pass
    return 128_000


def collect_context_usage(
    *,
    session: AgentSessionState,
    model: str,
    strategy: str,
    tool_specs: list[dict] | None = None,
    max_tokens_override: int | None = None,
) -> ContextUsageReport:
    raw_max_tokens = max_tokens_override if max_tokens_override is not None else infer_context_window(model)
    token_counter = describe_token_counter(model)
    count = lambda text: estimate_tokens(text, model)  # noqa: E731
    
    system_prompt_sections = tuple(
        UsageEntry(name=_section_name(part, idx), tokens=count(part))
        for idx, part in enumerate(session.system_prompt_parts, start=1)
    )
    system_context_entries = tuple(
        UsageEntry(name=key, tokens=count(f'{key}: {value}'))
        for key, value in session.system_context.items()
        if value
    )
    user_context_entries = tuple(
        UsageEntry(name=key, tokens=count(_render_user_context_chunk(key, value)))
        for key, value in session.user_context.items()
        if value
    )
    memory_files = tuple(_parse_memory_usage(session.user_context.get('sigilMd'), model=model))

    user_context_tokens = sum(entry.tokens for entry in user_context_entries)
    system_prompt_tokens = (
        sum(entry.tokens for entry in system_prompt_sections)
        + sum(entry.tokens for entry in system_context_entries)
    )

    tool_definitions_tokens = 0
    if tool_specs:
        for tool in tool_specs:
            tool_definitions_tokens += count(json.dumps(tool))

    conversation_user_tokens = 0
    assistant_tokens = 0
    tool_call_tokens = 0
    tool_result_tokens = 0
    tool_usage: dict[str, list[int]] = {}

    for index, message in enumerate(session.messages):
        if index == 0 and message.role == 'system':
            continue
        if _is_user_context_message(session, index, message):
            continue
        if message.role == 'user':
            conversation_user_tokens += count(message.content)
            continue
        if message.role == 'assistant':
            assistant_tokens += count(message.content)
            for tool_call in message.tool_calls:
                serialized = json.dumps(tool_call, ensure_ascii=True)
                tokens = count(serialized)
                tool_call_tokens += tokens
                tool_name = _extract_tool_call_name(tool_call)
                call_totals = tool_usage.setdefault(tool_name, [0, 0])
                call_totals[0] += tokens
            continue
        if message.role == 'tool':
            tokens = count(message.content)
            tool_result_tokens += tokens
            result_totals = tool_usage.setdefault(message.name or 'tool', [0, 0])
            result_totals[1] += tokens

    categories = [
        UsageEntry('System prompt', system_prompt_tokens),
        UsageEntry('Tool definitions', tool_definitions_tokens),
        UsageEntry('User context', user_context_tokens),
        UsageEntry('User messages', conversation_user_tokens),
        UsageEntry('Assistant messages', assistant_tokens),
        UsageEntry('Tool calls', tool_call_tokens),
        UsageEntry('Tool results', tool_result_tokens),
    ]
    total_tokens = sum(entry.tokens for entry in categories)
    free_space = max(raw_max_tokens - total_tokens, 0)
    categories.append(UsageEntry('Free space', free_space))

    tool_calls_by_type = tuple(
        ToolUsageEntry(
            name=name,
            call_tokens=values[0],
            result_tokens=values[1],
        )
        for name, values in sorted(
            tool_usage.items(),
            key=lambda item: (item[1][0] + item[1][1], item[0]),
            reverse=True,
        )
        if values[0] or values[1]
    )
    percentage = (total_tokens / raw_max_tokens * 100) if raw_max_tokens else 0.0
    return ContextUsageReport(
        model=model,
        total_tokens=total_tokens,
        raw_max_tokens=raw_max_tokens,
        percentage=percentage,
        strategy=strategy,
        message_count=len(session.messages),
        categories=tuple(categories),
        system_prompt_sections=system_prompt_sections,
        user_context_entries=user_context_entries,
        system_context_entries=system_context_entries,
        memory_files=memory_files,
        message_breakdown=MessageBreakdown(
            user_message_tokens=conversation_user_tokens,
            assistant_message_tokens=assistant_tokens,
            tool_call_tokens=tool_call_tokens,
            tool_result_tokens=tool_result_tokens,
            user_context_tokens=user_context_tokens,
            tool_calls_by_type=tool_calls_by_type,
        ),
        token_counter_backend=token_counter.backend,
        token_counter_source=token_counter.source,
        token_counter_accurate=token_counter.accurate,
    )


def format_context_usage(report: ContextUsageReport) -> str:
    lines = [
        '## Context Usage',
        '',
        f'**Model:** {report.model}  ',
        f'**Estimated tokens:** {_format_tokens(report.total_tokens)} / {_format_tokens(report.raw_max_tokens)} ({report.percentage:.1f}%)  ',
        f'**Token counter:** {report.token_counter_backend} ({report.token_counter_source}){" [accurate]" if report.token_counter_accurate else " [fallback]"}  ',
        f'**Context strategy:** {report.strategy}  ',
        f'**Messages in session:** {report.message_count}',
        '',
    ]

    visible_categories = [entry for entry in report.categories if entry.tokens > 0]
    if visible_categories:
        lines.extend(
            [
                '### Estimated usage by category',
                '',
                '| Category | Tokens | Percentage |',
                '|----------|--------|------------|',
            ]
        )
        for entry in visible_categories:
            percent = (entry.tokens / report.raw_max_tokens * 100) if report.raw_max_tokens else 0.0
            lines.append(f'| {entry.name} | {_format_tokens(entry.tokens)} | {percent:.1f}% |')
        lines.append('')

    if report.system_prompt_sections:
        lines.extend(
            [
                '### System Prompt Sections',
                '',
                '| Section | Tokens |',
                '|---------|--------|',
            ]
        )
        for entry in report.system_prompt_sections:
            lines.append(f'| {entry.name} | {_format_tokens(entry.tokens)} |')
        lines.append('')

    if report.user_context_entries:
        lines.extend(
            [
                '### User Context',
                '',
                '| Entry | Tokens |',
                '|-------|--------|',
            ]
        )
        for entry in report.user_context_entries:
            lines.append(f'| {entry.name} | {_format_tokens(entry.tokens)} |')
        lines.append('')

    if report.system_context_entries:
        lines.extend(
            [
                '### System Context',
                '',
                '| Entry | Tokens |',
                '|-------|--------|',
            ]
        )
        for entry in report.system_context_entries:
            lines.append(f'| {entry.name} | {_format_tokens(entry.tokens)} |')
        lines.append('')

    if report.memory_files:
        lines.extend(
            [
                '### Memory Files',
                '',
                '| Path | Tokens |',
                '|------|--------|',
            ]
        )
        for entry in report.memory_files:
            lines.append(f'| {entry.name} | {_format_tokens(entry.tokens)} |')
        lines.append('')

    breakdown = report.message_breakdown
    lines.extend(
        [
            '### Message Breakdown',
            '',
            '| Category | Tokens |',
            '|----------|--------|',
            f'| User context reminder | {_format_tokens(breakdown.user_context_tokens)} |',
            f'| User messages | {_format_tokens(breakdown.user_message_tokens)} |',
            f'| Assistant messages | {_format_tokens(breakdown.assistant_message_tokens)} |',
            f'| Tool calls | {_format_tokens(breakdown.tool_call_tokens)} |',
            f'| Tool results | {_format_tokens(breakdown.tool_result_tokens)} |',
            '',
        ]
    )

    if breakdown.tool_calls_by_type:
        lines.extend(
            [
                '#### Top Tools',
                '',
                '| Tool | Call Tokens | Result Tokens |',
                '|------|-------------|---------------|',
            ]
        )
        for entry in breakdown.tool_calls_by_type:
            lines.append(
                f'| {entry.name} | {_format_tokens(entry.call_tokens)} | {_format_tokens(entry.result_tokens)} |'
            )
        lines.append('')

    while lines and lines[-1] == '':
        lines.pop()
    return '\n'.join(lines)


def _section_name(part: str, index: int) -> str:
    stripped = part.strip()
    if stripped == SYSTEM_PROMPT_DYNAMIC_BOUNDARY:
        return 'Dynamic boundary'
    first_line = stripped.splitlines()[0] if stripped else ''
    if first_line.startswith('#'):
        return first_line.lstrip('#').strip() or f'Section {index}'
    return f'Section {index}'


def _render_user_context_chunk(key: str, value: str) -> str:
    return f'# {key}\n{value}'


def _extract_tool_call_name(tool_call: dict[str, object]) -> str:
    function_block = tool_call.get('function')
    if isinstance(function_block, dict):
        name = function_block.get('name')
        if isinstance(name, str) and name:
            return name
    return 'unknown'


def _is_user_context_message(
    session: AgentSessionState,
    index: int,
    message: AgentMessage,
) -> bool:
    if not session.user_context:
        return False
    # Search first few messages for the context reminder
    # Compaction or other injections can shift the index, so we check the signature.
    return (
        message.role == 'user'
        and message.content.startswith('<system-reminder>')
        and ("As you answer the user's questions" in message.content 
             or message.message_id == 'user_context_0')
    )


def _parse_memory_usage(sigil_md: str | None, *, model: str | None = None) -> list[UsageEntry]:
    if not sigil_md:
        return []
    matches = list(_PATH_HEADER_RE.finditer(sigil_md))
    if not matches:
        return []
    entries: list[UsageEntry] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(sigil_md)
        content = sigil_md[start:end].strip()
        entries.append(UsageEntry(name=match.group(1), tokens=estimate_tokens(content, model)))
    return entries


def _format_tokens(value: int) -> str:
    return f'{value:,}'
