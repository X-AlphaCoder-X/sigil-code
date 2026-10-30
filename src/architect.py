"""Architect / editor split (opt-in), after Aider's architect mode.

The main model (the architect) plans; an editor model makes the file edits.
Turn it on with --architect, SIGIL_ARCHITECT=1 or "architect": true. The editor
model is --editor-model, SIGIL_EDITOR_MODEL or "editorModel": "model" or
"model@base_url" (without one the main model edits too, in its own short
context). In architect mode the main agent keeps its read, search and shell
tools; its file-edit tools are replaced by editor_edit, which hands precise
instructions to an editor agent that has the edit tools and reports back.
SIGIL_EDITOR_MAX_TURNS (default 20) limits one editor run.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

EDIT_TOOLS = ('write_file', 'edit_file', 'notebook_edit')

EDITOR_PROMPT = (
    'You are the editor. Another model planned the change and wrote the instructions in the '
    'user message. Make exactly those file edits with your edit tools, reading files first '
    'where you need the exact text. When you are done, reply with the files you changed and '
    'one line per change.'
)


def _setting(cwd: Path, key: str) -> Any:
    try:
        from .settings import load_settings

        return load_settings(cwd).get(key)
    except Exception as _ignored:
        _log_ignored(_ignored)
        return None


def enabled(cwd: Path) -> bool:
    raw = os.environ.get('SIGIL_ARCHITECT', '').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    return _setting(cwd, 'architect') is True


def editor_model_config(config: Any, cwd: Path) -> Any:
    value = os.environ.get('SIGIL_EDITOR_MODEL', '').strip() or str(_setting(cwd, 'editorModel') or '').strip()
    if not value:
        return config
    from .profiles import config_for_endpoint

    name, _, base = value.partition('@')
    return config_for_endpoint(config, name.strip(), base.strip())  # the main key stays with its server


def _max_turns() -> int:
    try:
        return max(1, int(os.environ.get('SIGIL_EDITOR_MAX_TURNS', '20')))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return 20


def run_editor(agent: Any, instructions: str) -> tuple[str, dict[str, Any]]:
    """Run one editor agent on the instructions; returns its report and metadata."""
    from .agent_runtime import LocalCodingAgent
    from .agent_tools import default_tool_registry

    model_config = editor_model_config(agent.model_config, agent.runtime_config.cwd)
    editor = LocalCodingAgent(
        model_config=model_config,
        runtime_config=replace(agent.runtime_config, max_turns=_max_turns()),
        append_system_prompt=EDITOR_PROMPT,
        tool_registry=default_tool_registry(),
        command_hook_runtime=getattr(agent, 'command_hook_runtime', None),
        is_subagent=True,
    )
    editor.permission_prompt = getattr(agent, 'permission_prompt', None)
    if getattr(agent, 'permission_mode', 'default') not in ('plan',):
        editor.permission_mode = getattr(agent, 'permission_mode', 'default')
    result = editor.run(instructions)
    changed = sorted({str((entry.get('arguments') or {}).get('path'))
                      for entry in (result.tool_execution_ledger or ())
                      if isinstance(entry, dict) and entry.get('tool_name') in EDIT_TOOLS and entry.get('ok')
                      and isinstance(entry.get('arguments'), dict) and entry['arguments'].get('path')})
    report = result.final_output or '(the editor gave no report)'
    header = f'Editor ({model_config.model}) finished: {result.turns} turn(s), {result.tool_call_count} tool call(s).'
    if changed:
        header += ' Changed: ' + ', '.join(changed)
    return f'{header}\n{report}', {'editor_model': model_config.model, 'changed_files': changed,
                                   'stop_reason': result.stop_reason}


def register(agent: Any, registry: dict[str, Any]) -> dict[str, Any]:
    """In architect mode, swap the main agent's edit tools for editor_edit."""
    from .agent_tools import SigilTool

    if getattr(agent, 'is_subagent', False) or not enabled(agent.runtime_config.cwd):
        return registry

    def _edit(arguments: dict[str, Any], context: Any) -> Any:
        from .agent_tools import ToolExecutionError

        instructions = str(arguments.get('instructions', '')).strip()
        if not instructions:
            raise ToolExecutionError('editor_edit: instructions are empty')
        files = arguments.get('files')
        if isinstance(files, list) and files:
            instructions += '\n\nFiles: ' + ', '.join(str(f) for f in files)
        return run_editor(agent, instructions)

    trimmed = {name: tool for name, tool in registry.items() if name not in EDIT_TOOLS}
    trimmed['editor_edit'] = SigilTool(
        name='editor_edit',
        description=('Architect mode: you plan, an editor model makes the file edits. Give complete, exact '
                     'instructions: each file, what to change, and the new code. The editor applies them and '
                     'reports which files it changed; read the files afterwards to check the result.'),
        parameters={'type': 'object', 'properties': {
            'instructions': {'type': 'string', 'description': 'Exact edits to make, with the new code.'},
            'files': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Files to edit.'}},
            'required': ['instructions']},
        handler=_edit,
    )
    return trimmed
