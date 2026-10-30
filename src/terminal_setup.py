"""/terminal-setup: make Shift+Enter insert a new line in VS Code's terminal or Windows Terminal.

Both get a key binding that sends Esc+Enter (the same as Alt+Enter, which the prompt
already treats as a new line). The settings file is backed up (.bak) before it is
changed; a file with comments that cannot be read safely is left alone and the
snippet to add by hand is shown instead. SIGIL_VSCODE_KEYBINDINGS and SIGIL_WT_SETTINGS
point at other files.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

SEQUENCE = '\u001b\r'
VSCODE_BINDING = {'key': 'shift+enter', 'command': 'workbench.action.terminal.sendSequence',
                  'args': {'text': SEQUENCE}, 'when': 'terminalFocus'}
WT_ACTION_ID = 'User.SigilShiftEnterNewline'
WT_ACTION = {'command': {'action': 'sendInput', 'input': SEQUENCE}, 'id': WT_ACTION_ID}
WT_KEYBINDING = {'keys': 'shift+enter', 'id': WT_ACTION_ID}


def detect() -> str | None:
    if os.environ.get('TERM_PROGRAM', '').lower() == 'vscode':
        return 'vscode'
    if os.environ.get('WT_SESSION'):
        return 'windows-terminal'
    return None


def vscode_path() -> Path:
    custom = os.environ.get('SIGIL_VSCODE_KEYBINDINGS')
    if custom:
        return Path(custom)
    if sys.platform == 'win32':
        return Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'Code' / 'User' / 'keybindings.json'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'Code' / 'User' / 'keybindings.json'
    return Path.home() / '.config' / 'Code' / 'User' / 'keybindings.json'


def wt_path() -> Path:
    custom = os.environ.get('SIGIL_WT_SETTINGS')
    if custom:
        return Path(custom)
    local = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    for package in ('Microsoft.WindowsTerminal_8wekyb3d8bbwe', 'Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe'):
        candidate = local / 'Packages' / package / 'LocalState' / 'settings.json'
        if candidate.exists():
            return candidate
    return local / 'Packages' / 'Microsoft.WindowsTerminal_8wekyb3d8bbwe' / 'LocalState' / 'settings.json'


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    text = path.read_text(encoding='utf-8-sig')
    if not text.strip():
        return default
    try:
        return json.loads(text)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        pass
    stripped = re.sub(r'^\s*//.*$', '', text, flags=re.M)  # whole-line comments only
    stripped = re.sub(r',(\s*[\]}])', r'\1', stripped)  # trailing commas
    return json.loads(stripped)  # still bad: the caller leaves the file alone


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copyfile(path, path.with_name(path.name + '.bak'))
    path.write_text(json.dumps(data, indent=4) + '\n', encoding='utf-8')


def setup(target: str | None = None) -> str:
    target = (target or '').strip().lower() or detect()
    if target in ('vscode', 'code'):
        path = vscode_path()
        try:
            bindings = _load_json(path, [])
        except (ValueError, OSError) as _ignored:
            _log_ignored(_ignored)
            return (f'Could not read {path} safely. Add this entry to it by hand:\n'
                    + json.dumps(VSCODE_BINDING, indent=4))
        if not isinstance(bindings, list):
            return f'{path} is not a list of key bindings; left unchanged.'
        if any(isinstance(b, dict) and b.get('key') == 'shift+enter' and b.get('when') == 'terminalFocus'
               for b in bindings):
            return f'Shift+Enter is already bound in {path}.'
        _write(path, [*bindings, VSCODE_BINDING])
        return f'VS Code: Shift+Enter now inserts a new line in the terminal ({path}; backup .bak).'
    if target in ('windows-terminal', 'wt'):
        path = wt_path()
        try:
            settings = _load_json(path, {})
        except (ValueError, OSError) as _ignored:
            _log_ignored(_ignored)
            return (f'Could not read {path} safely. Add these by hand, to "actions" and "keybindings":\n'
                    + json.dumps(WT_ACTION, indent=4) + '\n' + json.dumps(WT_KEYBINDING, indent=4))
        if not isinstance(settings, dict):
            return f'{path} is not a settings object; left unchanged.'
        actions = settings.get('actions') if isinstance(settings.get('actions'), list) else []
        bindings = settings.get('keybindings') if isinstance(settings.get('keybindings'), list) else []
        if any(isinstance(b, dict) and b.get('keys') == 'shift+enter' for b in [*actions, *bindings]):
            return f'Shift+Enter is already bound in {path}.'
        # current schema: the action (with an id) and the key binding live in separate lists
        settings['actions'] = [*actions, WT_ACTION]
        settings['keybindings'] = [*bindings, WT_KEYBINDING]
        _write(path, settings)
        return f'Windows Terminal: Shift+Enter now inserts a new line ({path}; backup .bak).'
    return ('No supported terminal found. Run /terminal-setup vscode or /terminal-setup windows-terminal. '
            'Alt+Enter and Ctrl+J insert a new line in every terminal.')
