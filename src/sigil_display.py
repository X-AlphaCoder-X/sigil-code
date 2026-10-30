"""
Live terminal display renderer for Sigil Code.

Prints colorized, structured output to the terminal in real time
as events flow from the agent runtime:

  ╭─ Turn 1 ─────────────────────────╮
  │ 🤖 Thinking...                   │
  ╰──────────────────────────────────╯

  I'll check the files for you.

    ╭─ 🔧 bash ──────────────────────╮
    │ ls -la ~/project              │
    ╰────────────────────────────────╯
    ✓ exit_code=0 (truncated)

  Here is what I found: ...
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, TextIO

# ── ANSI color helpers ──────────────────────────────────────────────
_NO_COLOR = bool(os.environ.get('NO_COLOR'))

def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f'\033[{code}m{text}\033[0m'

def _bold(t: str) -> str:    return _c('1', t)
def _dim(t: str) -> str:     return _c('2', t)
def _cyan(t: str) -> str:    return _c('36', t)
def _green(t: str) -> str:   return _c('32', t)
def _yellow(t: str) -> str:  return _c('33', t)
def _red(t: str) -> str:     return _c('31', t)
def _magenta(t: str) -> str: return _c('35', t)
def _blue(t: str) -> str:    return _c('34', t)
def _gray(t: str) -> str:    return _c('90', t)


# ── Box drawing ─────────────────────────────────────────────────────
def _box_top(label: str, width: int = 60) -> str:
    inner = f' {label} '
    pad = width - 2 - len(inner)
    if pad < 0:
        pad = 0
    return _dim('╭─') + _bold(_cyan(inner)) + _dim('─' * pad + '╮')


def _box_line(text: str, width: int = 60) -> str:
    visible_len = len(text)
    pad = width - 4 - visible_len
    if pad < 0:
        pad = 0
    return _dim('│ ') + text + ' ' * pad + _dim(' │')


def _box_bottom(width: int = 60) -> str:
    return _dim('╰' + '─' * (width - 2) + '╯')


# ── Main display class ──────────────────────────────────────────────
class SigilDisplay:
    """Receives stream events from the agent runtime and renders them."""

    def __init__(self, out: TextIO | None = None):
        self.out = out or sys.stdout
        self._current_turn = 0
        self._in_content = False
        self._tool_depth = 0
        self._last_event_type = ''
        self._start_time = time.monotonic()
        self._tool_start_time = 0.0
        self._thinking_start_time = 0.0

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _writeln(self, text: str = '') -> None:
        self.out.write(text + '\n')
        self.out.flush()

    # ── Public entry point ──────────────────────────────────────────
    def on_event(self, event: dict[str, Any]) -> None:
        """Called by the agent runtime for each stream event."""
        from .terminal import clean_event

        event = clean_event(event)  # model/tool text never carries escape sequences to the terminal
        etype = event.get('type', '')

        if etype == 'turn_start':
            self._on_turn_start(event)
        elif etype == 'model_query':
            self._on_model_query(event)
        elif etype == 'content_delta':
            self._on_content_delta(event)
        elif etype == 'assistant_response':
            self._on_assistant_response(event)
        elif etype == 'tool_start':
            self._on_tool_start(event)
        elif etype == 'tool_delta':
            self._on_tool_delta(event)
        elif etype == 'tool_result':
            self._on_tool_result(event)
        elif etype == 'turn_complete':
            self._on_turn_complete(event)
        elif etype == 'task_budget_exceeded':
            self._on_budget_exceeded(event)
        elif etype == 'prompt_length_check':
            self._on_prompt_length(event)
        elif etype == 'auto_compact_summary':
            self._on_compact(event)
        elif etype == 'microcompact':
            self._writeln(_gray(
                f"  ↳ microcompact: moved {event.get('offloaded_count', 0)} old tool result(s) "
                f"to disk (~{int(event.get('chars_removed', 0)):,} chars freed)"
            ))
        elif etype == 'queued_messages_injected':
            for text in event.get('messages', []):
                self._writeln(_gray(f"  ↳ delivered to agent: {text}"))
        elif etype == 'auto_nudge':
            self._on_auto_nudge(event)
        elif etype == 'content_filter_stop':
            self._on_content_filter_stop(event)
        elif etype == 'compaction_evaluated':
            self._on_compaction_evaluated(event)

        self._last_event_type = etype

    # ── Event handlers ──────────────────────────────────────────────
    def _on_turn_start(self, event: dict) -> None:
        self._current_turn = event.get('turn_index', self._current_turn + 1)
        self._in_content = False
        strategy = event.get('nudge_strategy', 'off')
        
        label = f'Turn {self._current_turn + 1}'
        if strategy != 'off':
            label += f' [{strategy}]'
            
        self._writeln(_box_top(label))
        self._writeln(_box_line(_yellow('🤖 Thinking...')))
        self._writeln(_box_bottom())
        self._writeln()

    def _on_model_query(self, event: dict) -> None:
        self._thinking_start_time = time.monotonic()
        if self._last_event_type != 'turn_start':
            self._writeln()
            self._writeln(_gray(f'  ⏳ Querying model...'))

    def _on_content_delta(self, event: dict) -> None:
        if self._thinking_start_time > 0:
            elapsed = time.monotonic() - self._thinking_start_time
            self._writeln(_dim(f'  🧠 Thought for {elapsed:.1f}s'))
            self._thinking_start_time = 0

        delta = event.get('delta', '')
        if not self._in_content:
            self._in_content = True
            self._write('  ')
        self._write(delta)

    def _on_assistant_response(self, event: dict) -> None:
        if self._thinking_start_time > 0:
            # If no content delta but we got a response
            elapsed = time.monotonic() - self._thinking_start_time
            self._writeln(_dim(f'  🧠 Thought for {elapsed:.1f}s'))
            self._thinking_start_time = 0

        content = event.get('content', '')
        if self._in_content:
            self._writeln()
            self._in_content = False
        elif content:
            self._writeln('  ' + content)

    def _on_tool_start(self, event: dict) -> None:
        if self._thinking_start_time > 0:
            elapsed = time.monotonic() - self._thinking_start_time
            self._writeln(_dim(f'  🧠 Thought for {elapsed:.1f}s'))
            self._thinking_start_time = 0

        if self._in_content:
            self._writeln()
            self._writeln()
            self._in_content = False
        tool_name = event.get('tool_name', '?')
        arguments = event.get('arguments', {})

        self._tool_start_time = time.monotonic()
        self._tool_depth += 1

        # Show tool name in a box
        icon = self._tool_icon(tool_name)
        self._writeln(f'    {_box_top(f"{icon} {tool_name}")}')

        # Show arguments preview
        if isinstance(arguments, dict):
            cmd = arguments.get('command', '')
            path = arguments.get('file_path', arguments.get('path', ''))
            pattern = arguments.get('pattern', arguments.get('query', ''))
            if cmd:
                for line in cmd.split('\n')[:3]:
                    self._writeln(f'    {_box_line(_gray(line[:52]))}')
            elif path:
                self._writeln(f'    {_box_line(_gray(str(path)[:52]))}')
            elif pattern:
                self._writeln(f'    {_box_line(_gray(str(pattern)[:52]))}')

        self._writeln(f'    {_box_bottom()}')

    def _on_tool_delta(self, event: dict) -> None:
        # Stream tool output chunks
        delta = event.get('delta', '')
        if delta:
            for line in delta.split('\n')[:2]:
                if line.strip():
                    self._writeln(_gray(f'    │ {line[:70]}'))

    def _on_tool_result(self, event: dict) -> None:
        ok = event.get('ok', True)
        metadata = event.get('metadata', {})
        self._tool_depth = max(0, self._tool_depth - 1)

        elapsed = time.monotonic() - self._tool_start_time
        elapsed_str = f'{elapsed:.1f}s' if elapsed > 0.1 else ''

        if ok:
            exit_code = metadata.get('exit_code', '')
            preview = metadata.get('output_preview', metadata.get('stdout_preview', ''))
            if isinstance(preview, str) and len(preview) > 100:
                preview = preview[:100] + '...'
            status = _green('✓')
            detail_parts = []
            if exit_code != '':
                detail_parts.append(f'exit_code={exit_code}')
            if elapsed_str:
                detail_parts.append(elapsed_str)
            detail = ' '.join(detail_parts)
            self._writeln(f'    {status} {_gray(detail)}')
            if preview:
                for line in str(preview).split('\n')[:3]:
                    self._writeln(_gray(f'    {line[:72]}'))
        else:
            error = metadata.get('error', metadata.get('stderr_preview', ''))
            self._writeln(f'    {_red("✗ FAILED")} {_gray(elapsed_str)}')
            if error:
                self._writeln(f'    {_red(str(error)[:72])}')

        self._writeln()

    def _on_turn_complete(self, event: dict) -> None:
        if self._in_content:
            self._writeln()
            self._in_content = False
        
        elapsed = time.monotonic() - self._start_time
        usage = event.get('usage', {})
        total = usage.get('total_tokens', 0)
        input_t = usage.get('input_tokens', 0)
        output_t = usage.get('output_tokens', 0)
        
        self._writeln()
        self._writeln(_dim(f'  ╭── Turn {self._current_turn + 1} Statistics ──'))
        self._writeln(_dim(f'  │ ') + _gray(f'Duration: {elapsed:.1f}s'))
        self._writeln(_dim(f'  │ ') + _gray(f'Tokens:   {total:,} total ({input_t:,} in / {output_t:,} out)'))
        self._writeln(_dim(f'  ╰──────────────────────────────'))
        self._writeln()

    def _on_budget_exceeded(self, event: dict) -> None:
        reason = event.get('reason', 'budget exceeded')
        self._writeln()
        self._writeln(f'  {_red("⚠ BUDGET EXCEEDED")}: {_yellow(reason)}')
        self._writeln()

    def _on_prompt_length(self, event: dict) -> None:
        projected = event.get('projected_input_tokens', 0)
        limit = event.get('soft_input_limit_tokens', 0)
        self._writeln(_gray(f'  📏 Context: {projected:,} tokens (limit: {limit:,})'))

    def _on_compact(self, event: dict) -> None:
        pre = event.get('pre_compact_token_count', 0)
        post = event.get('post_compact_token_count', 0)
        self._writeln(_gray(f'  🗜️  Compacted: {pre:,} → {post:,} tokens'))

    def _on_compaction_evaluated(self, event: dict) -> None:
        usage = event.get('known_usage')
        threshold = event.get('threshold', 0)
        if usage:
            self._writeln(_gray(f'  [Context: {usage:,} / {threshold:,}]'))

    def _on_auto_nudge(self, event: dict) -> None:
        idx = event.get('nudge_index', 1)
        self._writeln()
        self._writeln(f'  {_yellow("⚠️  Auto-Nudge")} #{idx}: The model stalled, retrying...')
        self._writeln()

    def _on_content_filter_stop(self, event: dict) -> None:
        msg = event.get('message', 'Provider refused the request (content_filter).')
        self._writeln()
        self._writeln(f'  {_yellow("🛑 Content filter")}: provider refused the request.')
        self._writeln(_gray(f'  {msg}'))
        self._writeln(_gray('  Nudges skipped. Edit the prompt/context and try again.'))
        self._writeln()

    # ── Helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _tool_icon(name: str) -> str:
        icons = {
            'bash': '🔧',
            'read_file': '📄',
            'write_file': '✏️',
            'edit_file': '✏️',
            'list_dir': '📁',
            'glob_search': '🔍',
            'grep_search': '🔍',
            'web_search': '🌐',
            'web_fetch': '🌐',
            'delegate_agent': '🤖',
            'notebook_edit': '📓',
        }
        return icons.get(name, '🔧')
