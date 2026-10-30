"""Agent bar shown under the chat input (prompt_toolkit only).

Lists background agents: press Down on an empty
input to select one, Enter to open its status/log,
Esc or selecting "main" to return to the normal chat.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .background_runtime import BackgroundSessionRuntime

VIEW_COMMAND = '/agent-view'
KILL_COMMAND = '/agent-kill'
LIST_COMMAND = '/agents'
MAX_SHOWN = 6
_ID_RE = re.compile(r'bg_[0-9a-f]{12}')


REWIND_COMMAND = '/rewind'
MODE_COMMAND = '/mode'


class AgentBar:
    def __init__(self, cwd: Path) -> None:
        self.stash = ''  # Ctrl+S
        self.verbose = False  # Ctrl+O
        self.root = cwd.resolve() / '.port_sessions' / 'background'
        # Permission mode shown in the toolbar; Shift+Tab cycles it.
        self.mode = 'default'
        self.active = False
        self.selected: str | None = None  # None means "main"
        self._items: list[tuple[str, str, str]] = []  # (id, status, prompt)
        self._loaded_at = 0.0
        # Images pasted with Alt+V since the last prompt (shown as [Image #n]).
        self.pasted_images: list[str] = []
        # Ctrl+T: print the todo list as the agent updates it.
        self.show_todos = False
        # Text from the statusLine command (set by the chat loop), shown in the toolbar.
        self.status_provider: Any = None

    # ---- data -------------------------------------------------------------
    def runtime(self) -> BackgroundSessionRuntime:
        return BackgroundSessionRuntime(self.root)

    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._loaded_at < 1.0:
            return
        self._loaded_at = now
        if not self.root.is_dir():
            self._items = []
            return
        try:
            records = self.runtime().list_records()[:MAX_SHOWN]
        except Exception:
            from .debug_log import get_logger

            get_logger('agent_bar').warning('agent bar refresh failed', exc_info=True)
            return
        self._items = [(r.background_id, r.status, r.prompt) for r in records]
        if self.selected is not None and self.selected not in self.ids():
            self.selected = None

    def ids(self) -> list[str]:
        return [item[0] for item in self._items]

    # ---- navigation -------------------------------------------------------
    def activate(self) -> None:
        self.refresh(force=True)
        ids = self.ids()
        if ids:
            self.active = True
            self.selected = ids[0]

    def reset(self) -> None:
        self.active = False
        self.selected = None

    def move(self, step: int) -> None:
        options: list[str | None] = [None, *self.ids()]
        index = options.index(self.selected) if self.selected in options else 0
        self.selected = options[max(0, min(len(options) - 1, index + step))]

    # ---- rendering --------------------------------------------------------
    def fragments(self, on_click: Any = None) -> list[tuple]:
        self.refresh()
        if not self._items:
            return []

        def frag(style: str, text: str, target: str | None) -> tuple:
            if on_click is None:
                return (style, text)
            return (style, text, lambda event, t=target: on_click(event, t))

        def style_for(target: str | None) -> str:
            if self.active and self.selected == target:
                return 'class:agentbar.selected'
            if not self.active and target is None:
                return 'class:agentbar.main'
            return 'class:agentbar.item'

        out: list[tuple] = [('class:agentbar.label', ' agents: '), frag(style_for(None), ' main ', None)]
        for background_id, status, prompt in self._items:
            label = prompt.strip().replace('\n', ' ')
            label = label[:18] + ('...' if len(label) > 18 else '')
            out.append(('', ' '))
            out.append(frag(style_for(background_id), f' {background_id[3:9]} [{status}] {label} ', background_id))
        hint = (
            '  Left/Right move - Enter open - Esc back'
            if self.active
            else '  Down (empty input) or click to select'
        )
        out.append(('class:agentbar.hint', hint))
        return out

    # ---- chat commands ----------------------------------------------------
    def handle_command(self, text: str) -> str | None:
        """Return output for /agents, /agent-view <id>, /agent-kill <id>; None if not ours."""
        parts = text.split()
        if not parts or parts[0] not in {LIST_COMMAND, VIEW_COMMAND, KILL_COMMAND}:
            return None
        runtime = self.runtime()
        if parts[0] == LIST_COMMAND:
            from .subagents import discover_subagents

            lines = ['# Agents (use with delegate_agent agent=<name>)', '']
            for name, definition in discover_subagents(self.root.parent.parent).items():
                lines.append(f'- {name} [{definition.source}]: {definition.description[:100]}')
            return '\n'.join(lines) + '\n\n' + runtime.render_ps()
        if len(parts) != 2 or not _ID_RE.fullmatch(parts[1]):
            return f'Usage: {parts[0]} bg_<12 hex chars>'
        background_id = parts[1]
        if not runtime.record_path(background_id).exists():
            return f'Unknown background agent: {background_id}'
        self._loaded_at = 0.0
        if parts[0] == KILL_COMMAND:
            record = runtime.kill(background_id)
            return f'Background agent {background_id}: status={record.status}'
        text_out = runtime.render_logs(background_id, tail=60)
        if runtime.load_record(background_id).status == 'running':
            text_out += f'\n\n(still running - stop it with: {KILL_COMMAND} {background_id})'
        return text_out


def build_prompt_kwargs(bar: AgentBar) -> dict[str, Any]:
    """Extra PromptSession.prompt() kwargs: toolbar, key bindings, mouse, style."""
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.styles import Style

    def open_selected(app: Any) -> None:
        target = bar.selected
        bar.reset()
        if target is None:
            app.invalidate()
        else:
            app.exit(result=f'{VIEW_COMMAND} {target}')

    def on_click(event: Any, target: str | None) -> Any:
        if event.event_type != MouseEventType.MOUSE_UP:
            return NotImplemented
        bar.active = True
        bar.selected = target
        open_selected(get_app())
        return None

    kb = KeyBindings()
    empty = Condition(lambda: get_app().current_buffer.text == '')
    has_agents = Condition(lambda: bool(bar.ids()))
    active = Condition(lambda: bar.active)

    @kb.add('down', filter=empty & has_agents & ~active)
    def _enter_bar(event: Any) -> None:
        bar.activate()

    @kb.add('right', filter=active & empty)
    @kb.add('down', filter=active & empty)
    def _next(event: Any) -> None:
        bar.move(1)

    @kb.add('left', filter=active & empty)
    @kb.add('up', filter=active & empty)
    def _prev(event: Any) -> None:
        bar.move(-1)

    @kb.add('enter', filter=active & empty)
    def _open(event: Any) -> None:
        open_selected(event.app)

    @kb.add('escape', filter=active, eager=True)
    def _back(event: Any) -> None:
        bar.reset()

    # Newlines: Ctrl+J and Alt+Enter insert one (most terminals, the Windows
    # console included, send the same key for Shift+Enter and Enter), and so
    # does Enter right after a trailing backslash.
    @kb.add('c-j', filter=~active)
    @kb.add('escape', 'enter', filter=~active)
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text('\n')

    # Ctrl+G (or Ctrl+X Ctrl+E) edits the prompt in $VISUAL / $EDITOR.
    @kb.add('c-g', filter=~active)
    @kb.add('c-x', 'c-e', filter=~active)
    def _edit_in_editor(event: Any) -> None:
        event.current_buffer.open_in_editor(validate_and_handle=False)

    # Ctrl+T shows or hides the todo list while the agent works.
    @kb.add('c-t', filter=~active)
    def _toggle_todos(event: Any) -> None:
        bar.show_todos = not bar.show_todos
        event.app.invalidate()

    # Alt+V pastes an image from the clipboard as an [Image #n] chip.
    @kb.add('escape', 'v', filter=~active)
    def _paste_image(event: Any) -> None:
        from .images import grab_clipboard_image

        path = grab_clipboard_image(bar.root.parent / 'pasted')
        if path is None:
            return
        bar.pasted_images.append(str(path))
        event.current_buffer.insert_text(f'[Image #{len(bar.pasted_images)}]')

    ends_with_backslash = Condition(
        lambda: get_app().current_buffer.document.text_before_cursor.endswith('\\'))

    @kb.add('enter', filter=ends_with_backslash & ~active)
    def _continue_line(event: Any) -> None:
        buffer = event.current_buffer
        buffer.delete_before_cursor(1)
        buffer.insert_text('\n')

    # Shift+Tab cycles permission mode (default -> accept edits -> plan).
    @kb.add('s-tab')
    def _cycle_mode(event: Any) -> None:
        from .permission_modes import next_mode

        bar.mode = next_mode(bar.mode)
        event.app.invalidate()

    # Esc Esc on an empty prompt opens /rewind.
    @kb.add('escape', 'escape', filter=empty & ~active)
    def _rewind(event: Any) -> None:
        event.app.exit(result=REWIND_COMMAND)

    # Esc Esc with a draft: clear it, but keep it in the history (Up brings it back).
    @kb.add('escape', 'escape', filter=~empty & ~active)
    def _clear_draft(event: Any) -> None:
        buffer = event.app.current_buffer
        if buffer.text.strip():
            buffer.history.append_string(buffer.text)
        buffer.reset()

    # Ctrl+S stashes the draft (and puts a stashed one back on an empty prompt).
    @kb.add('c-s', filter=~active)
    def _stash(event: Any) -> None:
        buffer = event.app.current_buffer
        if buffer.text:
            bar.stash = buffer.text
            buffer.reset()
        elif bar.stash:
            buffer.insert_text(bar.stash)
            bar.stash = ''

    # ? on an empty prompt: key help.
    @kb.add('?', filter=empty & ~active)
    def _key_help(event: Any) -> None:
        event.app.exit(result='/keybindings')

    # Ctrl+O: verbose transcript after each answer (on/off).
    @kb.add('c-o', filter=~active)
    def _verbose(event: Any) -> None:
        bar.verbose = not bar.verbose
        event.app.invalidate()

    # Alt+P: model list (then /model <name>); Alt+T: thinking on/off.
    @kb.add('escape', 'p', filter=~active)
    def _models(event: Any) -> None:
        event.app.exit(result='/model list')

    @kb.add('escape', 't', filter=~active)
    def _thinking(event: Any) -> None:
        event.app.exit(result='/thinking toggle')

    def toolbar() -> list[tuple]:
        from .permission_modes import MODE_LABELS

        mode_frag: list[tuple] = []
        if bar.mode != 'default':
            style = 'class:mode.plan' if bar.mode == 'plan' else 'class:mode.on'
            mode_frag = [(style, f' \u23f5\u23f5 {MODE_LABELS[bar.mode]} '), ('class:agentbar.hint', ' (shift+tab to cycle)  ')]
        else:
            mode_frag = [('class:agentbar.hint', ' shift+tab: mode  esc esc: rewind  ')]
        status: list[tuple] = []
        if bar.status_provider is not None:
            try:
                text = bar.status_provider()
            except Exception:
                from .debug_log import get_logger

                get_logger('agent_bar').warning('status line provider failed', exc_info=True)
                text = ''
            if text:
                status = [('class:agentbar.hint', f' {text}  ')]
        todos = [('class:agentbar.hint', ' todos: on ')] if bar.show_todos else []
        if bar.verbose:
            todos.append(('class:agentbar.hint', ' verbose: on '))
        if bar.stash:
            todos.append(('class:agentbar.hint', ' draft stashed (ctrl+s) '))
        return status + mode_frag + todos + bar.fragments(on_click)

    return {
        'bottom_toolbar': toolbar,
        'key_bindings': kb,
        # No mouse capture and no periodic redraw: both hijack the terminal
        # scrollback (wheel events / snapping the view back to the prompt),
        # so the user could not scroll up through history. The bar still
        # updates on every keypress and is navigable with the arrow keys.
        'mouse_support': False,
        'style': Style.from_dict({
            'bottom-toolbar': 'noreverse',
            'agentbar.label': 'fg:ansibrightblack',
            'agentbar.main': 'fg:ansicyan bold',
            'agentbar.item': 'fg:ansiwhite',
            'agentbar.selected': 'reverse bold',
            'agentbar.hint': 'fg:ansibrightblack italic',
            'mode.on': 'fg:ansimagenta bold',
            'mode.plan': 'fg:ansicyan bold',
        }),
    }
