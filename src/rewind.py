"""/rewind: go back to an earlier prompt (conversation and/or files).

Used by the /rewind command and by Esc Esc at an empty prompt. Lists past
user prompts; the chosen prompt and everything after it are removed from the
conversation, and files changed by write/edit tools since then are restored.

The summarize modes keep every file as it is and only shrink the conversation:
  summarize-from  the chosen prompt and everything after it become one summary
  summarize-to    everything before the chosen prompt becomes one summary
                  (the chosen prompt and what follows stay word for word)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .checkpoints import CheckpointStore
from .session_store import load_agent_session, save_agent_session

REWIND_MODES = ('both', 'conversation', 'code', 'summarize-from', 'summarize-to')
SUMMARIZE_ALIASES = {'from': 'summarize-from', 'summarize': 'summarize-from', 'to': 'summarize-to',
                     'up-to': 'summarize-to', 'upto': 'summarize-to'}


@dataclass(frozen=True)
class PromptPoint:
    number: int            # 1-based, oldest first
    message_index: int     # index in the stored message list
    text: str
    checkpoint_id: str | None


def list_prompt_points(messages: tuple[dict, ...] | list[dict]) -> list[PromptPoint]:
    points: list[PromptPoint] = []
    for index, message in enumerate(messages):
        if message.get('role') != 'user':
            continue
        metadata = message.get('metadata') or {}
        if metadata.get('kind'):  # system reminders, replays, continuations, queued, ...
            continue
        text = str(message.get('content') or '').strip()
        if not text or text.startswith('<system-reminder>'):
            continue
        points.append(PromptPoint(
            number=len(points) + 1,
            message_index=index,
            text=text,
            checkpoint_id=metadata.get('checkpoint_id'),
        ))
    return points


def render_prompt_list(points: list[PromptPoint], store: CheckpointStore | None, limit: int = 20) -> str:
    if not points:
        return 'No earlier prompts to rewind to.'
    lines = ['Rewind: pick the prompt to go back to (it and everything after it are undone):', '']
    for point in points[-limit:]:
        preview = point.text.replace('\n', ' ')
        preview = preview[:70] + ('...' if len(preview) > 70 else '')
        files = ''
        if store is not None and point.checkpoint_id:
            changed = store.changed_files_since(point.checkpoint_id)
            if changed:
                files = f'  [{len(changed)} file(s) changed since]'
        lines.append(f'  {point.number:>3}. {preview}{files}')
    lines.append('')
    lines.append('Usage: /rewind <number> [both|conversation|code]   (default: both)')
    lines.append('       /rewind <number> summarize-from|summarize-to   (shrink the conversation, keep files)')
    return '\n'.join(lines)


def rewind_session(agent: Any, session_id: str, number: int, mode: str = 'both') -> tuple[str, str]:
    """Rewind to prompt `number`. Returns (report, original prompt text)."""
    mode = SUMMARIZE_ALIASES.get(mode, mode)
    if mode not in REWIND_MODES:
        raise ValueError(f'mode must be one of {", ".join(REWIND_MODES)}')
    directory = agent.runtime_config.session_directory
    stored = load_agent_session(session_id, directory=directory)
    points = list_prompt_points(stored.messages)
    match = next((p for p in points if p.number == number), None)
    if match is None:
        raise ValueError(f'No prompt #{number}. Run /rewind to see the list.')
    if mode.startswith('summarize-'):
        return summarize_session(agent, stored, match, mode), match.text

    report: list[str] = []
    if mode in ('both', 'code'):
        if match.checkpoint_id:
            store = agent._checkpoints_for(session_id)
            result = store.rewind(match.checkpoint_id)
            if result.restored or result.deleted:
                report.append(
                    f'Files restored: {len(result.restored)}'
                    + (f', removed (created after that prompt): {len(result.deleted)}' if result.deleted else '')
                )
                for rel in result.restored:
                    report.append(f'  restored {rel}')
                for rel in result.deleted:
                    report.append(f'  removed  {rel}')
            else:
                report.append('No file changes to undo since that prompt.')
        else:
            report.append('That prompt predates checkpoints, so files were left as they are.')

    if mode in ('both', 'conversation'):
        kept = tuple(stored.messages[:match.message_index])
        save_agent_session(replace(stored, messages=kept), directory=directory)
        agent.last_session = None
        report.append(f'Conversation rewound to before prompt #{number} ({len(stored.messages) - len(kept)} message(s) removed).')
    return '\n'.join(report), match.text


def _conversation_start(messages: tuple[dict, ...] | list[dict]) -> int:
    """Index of the first real message (after the system prompt, the context
    reminder and earlier compaction markers)."""
    start = 0
    for index, message in enumerate(messages):
        metadata = message.get('metadata') or {}
        if index == 0 and message.get('role') == 'system':
            start = 1
            continue
        if index <= 1 and message.get('role') == 'user' and (
                message.get('message_id') == 'user_context_0' or '<system-reminder>' in str(message.get('content'))):
            start = index + 1
            continue
        if metadata.get('kind') == 'compact_boundary':
            start = index + 1
            continue
        break
    return start


def summarize_session(agent: Any, stored: Any, match: PromptPoint, mode: str,
                      instructions: str | None = None) -> str:
    """Replace part of the conversation with one model-written summary. Files stay as they are."""
    from .agent_session import AgentMessage
    from .anthropic_client import fast_client
    from .compact import _render_transcript, format_compact_summary, get_compact_prompt
    from .agent_context_usage import estimate_tokens

    messages = list(stored.messages)
    if mode == 'summarize-from':
        start, end = match.message_index, len(messages)
    else:
        start, end = _conversation_start(messages), match.message_index
    chunk = messages[start:end]
    if not chunk:
        raise ValueError('Nothing to summarize there. Pick another prompt.')
    rendered = _render_transcript([AgentMessage.from_openai_message(m) for m in chunk])
    request = [{'role': 'user', 'content': (
        'Below is part of a conversation to summarize.\n\n<transcript>\n' + rendered + '\n</transcript>\n\n'
        + get_compact_prompt(instructions))}]
    turn = fast_client(agent.model_config, default=agent.client).complete(request, tools=[])
    summary = format_compact_summary(getattr(turn, 'content', None) or '')
    if not summary.strip():
        raise ValueError('The model returned an empty summary; nothing was changed.')
    where = f'from prompt #{match.number} onward' if mode == 'summarize-from' else f'before prompt #{match.number}'
    summary_message = {
        'role': 'user',
        'content': f'Summary of the conversation {where} (shortened at the user\'s request to save context):\n\n'
                   + summary,
        'message_id': f'rewind_summary_{match.number}',
        'metadata': {'kind': 'compact_summary', 'is_compact_summary': True, 'summarized': mode},
    }
    kept = messages[:start] + [summary_message] + messages[end:]
    save_agent_session(replace(stored, messages=tuple(kept)), directory=agent.runtime_config.session_directory)
    agent.last_session = None
    model = agent.model_config.model
    before = sum(estimate_tokens(str(m.get('content') or ''), model) for m in chunk)
    after = estimate_tokens(summary_message['content'], model)
    return (f'Summarized {len(chunk)} message(s) {where} into one (~{before} -> ~{after} tokens). '
            'Files were not changed.')
