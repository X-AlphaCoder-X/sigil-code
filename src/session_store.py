from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent_types import (
    AgentPermissions,
    AgentRuntimeConfig,
    BudgetConfig,
    ModelConfig,
    ModelPricing,
    OutputSchemaConfig,
    UsageStats,
)


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    messages: tuple[str, ...]
    input_tokens: int
    output_tokens: int


DEFAULT_SESSION_DIR = Path('.port_sessions')
DEFAULT_AGENT_SESSION_DIR = DEFAULT_SESSION_DIR / 'agent'

import re as _re
from .settings import write_text_atomic  # replaces a file in one step: readers never see half of it
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file
_SAFE_SESSION_ID = _re.compile(r'^[a-zA-Z0-9_-]+$')


def _validate_session_id(session_id: str) -> str:
    if not _SAFE_SESSION_ID.match(session_id) or '..' in session_id:
        raise ValueError(f'Invalid session ID: {session_id!r}')
    return session_id


def save_session(session: StoredSession, directory: Path | None = None) -> Path:
    target_dir = directory or DEFAULT_SESSION_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f'{session.session_id}.json'
    write_text_atomic(path, json.dumps(asdict(session), indent=2))
    return path


def load_session(session_id: str, directory: Path | None = None) -> StoredSession:
    _validate_session_id(session_id)
    target_dir = directory or DEFAULT_SESSION_DIR
    data = json.loads((target_dir / f'{session_id}.json').read_text())
    return StoredSession(
        session_id=data['session_id'],
        messages=tuple(data['messages']),
        input_tokens=data['input_tokens'],
        output_tokens=data['output_tokens'],
    )


JSONDict = dict[str, Any]


@dataclass(frozen=True)
class StoredAgentSession:
    session_id: str
    model_config: JSONDict
    runtime_config: JSONDict
    system_prompt_parts: tuple[str, ...]
    user_context: dict[str, str]
    system_context: dict[str, str]
    messages: tuple[JSONDict, ...]
    turns: int
    tool_call_count: int
    tool_execution_ledger: tuple[JSONDict, ...]
    usage: JSONDict
    total_cost_usd: float
    file_history: tuple[JSONDict, ...]
    budget_state: JSONDict
    plugin_state: JSONDict
    file_hashes: JSONDict = field(default_factory=dict)
    scratchpad_directory: str | None = None
    current_turn: int = 0
    file_access_history: JSONDict = field(default_factory=dict)


def _ignore_in_git(folder: Path) -> None:
    """A .gitignore in <cwd>/.port_sessions keeps session files out of git status."""
    if folder.name != '.port_sessions':
        return
    try:
        if not (folder / '.gitignore').exists():
            (folder / '.gitignore').write_text('# Sigil Code session data\n*\n', encoding='utf-8')
    except OSError as _ignored:
        _log_ignored(_ignored)
        pass


def save_agent_session(session: StoredAgentSession, directory: Path | None = None) -> Path:
    target_dir = directory or DEFAULT_AGENT_SESSION_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    _ignore_in_git(target_dir.parent)
    path = target_dir / f'{session.session_id}.json'

    # [ATOMIC-SAVE] a unique temp file, then os.replace (write_text_atomic): an interrupted save
    # leaves the old file, and two processes saving one session no longer share one temp name
    def path_serializer(obj):
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    write_text_atomic(path, json.dumps(asdict(session), indent=2, default=path_serializer))
    return path


def load_agent_session(session_id: str, directory: Path | None = None) -> StoredAgentSession:
    _validate_session_id(session_id)
    target_dir = directory or DEFAULT_AGENT_SESSION_DIR
    session_file = target_dir / f'{session_id}.json'
    
    # [RETRY-LOGIC] Handle race conditions where a session might be written 
    # but not yet fully synced or renamed (atomic-save).
    import time
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            data = json.loads(session_file.read_text(encoding='utf-8'))
            break
        except FileNotFoundError:
            raise  # no such session: retrying cannot help
        except (OSError, json.JSONDecodeError) as e:
            if attempt < max_attempts - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise OSError(f"Unable to load session {session_id} after {max_attempts} attempts: {e}") from e

    # [SCHEMA-DRIFT SAFETY] Use .get() with defaults for all fields so old
    # session files missing newer fields don't crash on load.
    return StoredAgentSession(
        session_id=data.get('session_id', session_id),
        model_config=dict(data.get('model_config', {})),
        runtime_config=dict(data.get('runtime_config', {})),
        system_prompt_parts=tuple(data.get('system_prompt_parts', [])),
        user_context=dict(data.get('user_context', {})),
        system_context=dict(data.get('system_context', {})),
        messages=tuple(
            message for message in data.get('messages', []) if isinstance(message, dict)
        ),
        turns=(
            len(data.get('turns', 0)) if isinstance(data.get('turns'), (list, tuple))
            else int(data.get('turns', 0))
        ),
        tool_call_count=(
            len(data.get('tool_call_count', [])) if isinstance(data.get('tool_call_count'), (list, tuple))
            else (
                int(data.get('tool_call_count', 0)) if data.get('tool_call_count') is not None
                else int(data.get('tool_calls', 0)) # Fallback for old sessions
            )
        ),
        tool_execution_ledger=tuple(
            entry for entry in data.get('tool_execution_ledger', 
                data.get('tool_calls', []) if isinstance(data.get('tool_calls'), list) else []
            ) if isinstance(entry, dict)
        ),
        usage=dict(data.get('usage', {})),
        total_cost_usd=float(data.get('total_cost_usd', 0.0)),
        file_history=tuple(
            entry for entry in data.get('file_history', []) if isinstance(entry, dict)
        ),
        budget_state=(
            dict(data.get('budget_state', {}))
            if isinstance(data.get('budget_state'), dict)
            else {}
        ),
        plugin_state=(
            dict(data.get('plugin_state', {}))
            if isinstance(data.get('plugin_state'), dict)
            else {}
        ),
        scratchpad_directory=(
            str(data['scratchpad_directory'])
            if isinstance(data.get('scratchpad_directory'), str)
            else None
        ),
        file_hashes=dict(data.get('file_hashes', {})),
        current_turn=int(data.get('current_turn', 0)),
        file_access_history=dict(data.get('file_access_history', {})),
    )


def list_agent_sessions(directory: Path | None = None) -> list[tuple[str, float, int]]:
    """Saved agent sessions as (session_id, mtime, message_count), newest first.

    Only readable *.json session files are included; corrupt or partially
    written files are skipped rather than crashing the picker.
    """
    target_dir = directory or DEFAULT_AGENT_SESSION_DIR
    if not target_dir.exists():
        return []
    results: list[tuple[str, float, int]] = []
    for path in target_dir.glob('*.json'):
        try:
            mtime = path.stat().st_mtime
            data = json.loads(path.read_text(encoding='utf-8'))
            messages = data.get('messages', [])
            count = len(messages) if isinstance(messages, (list, tuple)) else 0
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            continue
        results.append((path.stem, mtime, count))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def search_agent_sessions(directory: Path, text: str, limit: int = 50) -> list[tuple[str, float, int, str]]:
    """Sessions whose user or assistant messages contain text (any case), newest first,
    as (session_id, mtime, message_count, snippet)."""
    needle = text.strip().lower()
    found: list[tuple[str, float, int, str]] = []
    if not needle:
        return found
    for session_id, mtime, count in list_agent_sessions(directory):
        try:
            data = json.loads((Path(directory) / f'{session_id}.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            continue
        for message in data.get('messages', []) or []:
            if not isinstance(message, dict) or message.get('role') not in ('user', 'assistant'):
                continue
            content = ' '.join(str(message.get('content') or '').split())
            at = content.lower().find(needle)
            if at >= 0:
                start = max(0, at - 30)
                snippet = ('...' if start else '') + content[start:at + len(needle) + 40]
                found.append((session_id, mtime, count, snippet))
                break
        if len(found) >= limit:
            break
    return found


def cleanup_period_days(cwd: Path) -> float | None:
    """cleanupPeriodDays (or SIGIL_CLEANUP_PERIOD_DAYS): delete sessions older than this at start-up."""
    import os

    raw = os.environ.get('SIGIL_CLEANUP_PERIOD_DAYS', '').strip()
    if not raw:
        try:
            from .settings import load_settings

            raw = str(load_settings(cwd).get('cleanupPeriodDays') or '')
        except Exception as _ignored:
            _log_ignored(_ignored)
            raw = ''
    try:
        days = float(raw)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return None
    return days if days > 0 else None


def purge_agent_sessions(directory: Path, *, older_than_days: float | None, dry_run: bool = False) -> list[str]:
    """Delete saved sessions (with their checkpoints and notes) older than N days; None = all."""
    import shutil
    import time

    directory = Path(directory)
    cutoff = None if older_than_days is None else time.time() - older_than_days * 86400
    removed: list[str] = []
    for session_id, mtime, _count in list_agent_sessions(directory):
        if cutoff is not None and mtime >= cutoff:
            continue
        removed.append(session_id)
        if dry_run:
            continue
        (directory / f'{session_id}.json').unlink(missing_ok=True)
        (directory / 'notes' / f'{session_id}.md').unlink(missing_ok=True)
        shutil.rmtree(directory / 'checkpoints' / session_id, ignore_errors=True)
    if removed and not dry_run:
        try:
            from .settings import read_text_retry

            names = json.loads(read_text_retry(_names_path(directory)))
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            names = None
        if isinstance(names, dict):
            kept = {name: sid for name, sid in names.items() if sid not in removed}
            from .settings import write_text_atomic

            write_text_atomic(_names_path(directory), json.dumps(kept, indent=2))
    return removed


def fork_agent_session(directory: Path, session_id: str, new_id: str | None = None) -> str:
    """Copy a saved session under a new id (--fork-session); returns the new id."""
    from dataclasses import replace as _replace
    from uuid import uuid4

    stored = load_agent_session(session_id, directory=directory)
    new_id = _validate_session_id(new_id) if new_id else uuid4().hex
    save_agent_session(_replace(stored, session_id=new_id), directory=directory)
    return new_id


def _names_path(directory: Path) -> Path:
    return Path(directory) / 'names.idx'  # not *.json, so session listings skip it


def save_session_name(directory: Path, name: str, session_id: str) -> None:
    """Remember a name for a session (-n / --name, /rename)."""
    path = _names_path(directory)
    try:
        from .settings import read_text_retry

        names = json.loads(read_text_retry(path)) if path.exists() else {}
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        names = {}
    if not isinstance(names, dict):
        names = {}
    names[name] = session_id
    from .settings import write_text_atomic

    write_text_atomic(path, json.dumps(names, indent=2))  # several chats can name sessions


def resolve_session_name(directory: Path, name_or_id: str) -> str:
    """A session id for a saved name; anything else is returned unchanged."""
    try:
        from .settings import read_text_retry

        names = json.loads(read_text_retry(_names_path(directory)))
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return name_or_id
    if isinstance(names, dict) and isinstance(names.get(name_or_id), str):
        return names[name_or_id]
    return name_or_id


def session_first_prompt(directory: Path, session_id: str, limit: int = 70) -> str:
    """The first user prompt of a saved session (for pickers), shortened."""
    try:
        data = json.loads((Path(directory) / f'{session_id}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return ''
    for message in data.get('messages', []) or []:
        if isinstance(message, dict) and message.get('role') == 'user':
            text = ' '.join(str(message.get('content', '')).split())
            if text and not text.startswith('<system-reminder>'):
                return text if len(text) <= limit else text[:limit - 3] + '...'
    return ''


def serialize_model_config(model_config: ModelConfig) -> JSONDict:
    return {
        'model': model_config.model,
        'base_url': model_config.base_url,
        'api_key': '***REDACTED***',
        'temperature': model_config.temperature,
        'timeout_seconds': model_config.timeout_seconds,
        'effort': model_config.effort,
        'pricing': {
            'input_cost_per_million_tokens_usd': model_config.pricing.input_cost_per_million_tokens_usd,
            'output_cost_per_million_tokens_usd': model_config.pricing.output_cost_per_million_tokens_usd,
            'cache_creation_input_cost_per_million_tokens_usd': model_config.pricing.cache_creation_input_cost_per_million_tokens_usd,
            'cache_read_input_cost_per_million_tokens_usd': model_config.pricing.cache_read_input_cost_per_million_tokens_usd,
        },
    }


def deserialize_model_config(payload: JSONDict) -> ModelConfig:
    from .profiles import resolve_api_key

    base_url = str(payload.get('base_url', 'http://127.0.0.1:8000/v1'))
    return ModelConfig(
        model=str(payload['model']),
        base_url=base_url,
        # keys are never saved; read again in the same order as a new session, for this server
        api_key=resolve_api_key(base_url),
        temperature=(
            None if payload.get('temperature', 0.0) is None
            else float(payload.get('temperature', 0.0))
        ),
        timeout_seconds=float(payload.get('timeout_seconds', 120.0)),
        pricing=_deserialize_pricing(payload.get('pricing')),
        effort=(str(payload['effort']) if payload.get('effort') else None),
    )


def serialize_runtime_config(runtime_config: AgentRuntimeConfig) -> JSONDict:
    return {
        'cwd': str(runtime_config.cwd),
        'max_turns': runtime_config.max_turns,
        'command_timeout_seconds': runtime_config.command_timeout_seconds,
        'max_output_chars': runtime_config.max_output_chars,
        'stream_model_responses': runtime_config.stream_model_responses,
        'auto_snip_threshold_tokens': runtime_config.auto_snip_threshold_tokens,
        'auto_compact_threshold_tokens': runtime_config.auto_compact_threshold_tokens,
        'compact_preserve_messages': runtime_config.compact_preserve_messages,
        'permissions': {
            'allow_file_write': runtime_config.permissions.allow_file_write,
            'allow_shell_commands': runtime_config.permissions.allow_shell_commands,
            'allow_destructive_shell_commands': runtime_config.permissions.allow_destructive_shell_commands,
        },
        'additional_working_directories': [str(path) for path in runtime_config.additional_working_directories],
        'disable_sigil_md_discovery': runtime_config.disable_sigil_md_discovery,
        'budget_config': {
            'max_total_tokens': runtime_config.budget_config.max_total_tokens,
            'max_input_tokens': runtime_config.budget_config.max_input_tokens,
            'max_output_tokens': runtime_config.budget_config.max_output_tokens,
            'max_reasoning_tokens': runtime_config.budget_config.max_reasoning_tokens,
            'max_total_cost_usd': runtime_config.budget_config.max_total_cost_usd,
            'max_tool_call_count': runtime_config.budget_config.max_tool_call_count,
            'max_delegated_tasks': runtime_config.budget_config.max_delegated_tasks,
            'max_model_calls': runtime_config.budget_config.max_model_calls,
            'max_session_turns': runtime_config.budget_config.max_session_turns,
        },
        'output_schema': (
            {
                'name': runtime_config.output_schema.name,
                'schema': runtime_config.output_schema.schema,
                'strict': runtime_config.output_schema.strict,
            }
            if runtime_config.output_schema is not None
            else None
        ),
        'nudge_strategy': runtime_config.nudge_strategy,
        'nudge_max_consecutive': runtime_config.nudge_max_consecutive,
        'nudge_max_total': runtime_config.nudge_max_total,
        'session_directory': str(runtime_config.session_directory),
        'scratchpad_root': str(runtime_config.scratchpad_root),
    }


def deserialize_runtime_config(payload: JSONDict) -> AgentRuntimeConfig:
    permissions_payload = payload.get('permissions')
    if not isinstance(permissions_payload, dict):
        permissions_payload = {}
    budget_payload = payload.get('budget_config')
    if not isinstance(budget_payload, dict):
        budget_payload = {}
    output_schema_payload = payload.get('output_schema')
    return AgentRuntimeConfig(
        cwd=Path(str(payload['cwd'])).resolve(),
        max_turns=int(payload.get('max_turns', 12)),
        command_timeout_seconds=float(payload.get('command_timeout_seconds', 30.0)),
        max_output_chars=int(payload.get('max_output_chars', 12000)),
        stream_model_responses=bool(payload.get('stream_model_responses', False)),
        auto_snip_threshold_tokens=_optional_int(payload.get('auto_snip_threshold_tokens')),
        auto_compact_threshold_tokens=_optional_int(payload.get('auto_compact_threshold_tokens')),
        compact_preserve_messages=int(payload.get('compact_preserve_messages', 4)),
        permissions=AgentPermissions(
            allow_file_write=bool(permissions_payload.get('allow_file_write', False)),
            allow_shell_commands=bool(permissions_payload.get('allow_shell_commands', False)),
            allow_destructive_shell_commands=bool(permissions_payload.get('allow_destructive_shell_commands', False)),
        ),
        additional_working_directories=tuple(
            Path(str(path)).resolve()
            for path in payload.get('additional_working_directories', [])
        ),
        disable_sigil_md_discovery=bool(payload.get('disable_sigil_md_discovery', False)),
        budget_config=BudgetConfig(
            max_total_tokens=_optional_int(budget_payload.get('max_total_tokens')),
            max_input_tokens=_optional_int(budget_payload.get('max_input_tokens')),
            max_output_tokens=_optional_int(budget_payload.get('max_output_tokens')),
            max_reasoning_tokens=_optional_int(budget_payload.get('max_reasoning_tokens')),
            max_total_cost_usd=_optional_float(budget_payload.get('max_total_cost_usd')),
            max_tool_call_count=_optional_int(budget_payload.get('max_tool_call_count')),
            max_delegated_tasks=_optional_int(budget_payload.get('max_delegated_tasks')),
            max_model_calls=_optional_int(budget_payload.get('max_model_calls')),
            max_session_turns=_optional_int(budget_payload.get('max_session_turns')),
        ),
        output_schema=_deserialize_output_schema(output_schema_payload),
        nudge_strategy=_validate_nudge_strategy(payload.get('nudge_strategy', 'structural')),
        nudge_max_consecutive=int(payload.get('nudge_max_consecutive', 3)),
        nudge_max_total=int(payload.get('nudge_max_total', 20)),
        session_directory=Path(str(payload.get('session_directory', DEFAULT_AGENT_SESSION_DIR))).resolve(),
        scratchpad_root=Path(str(payload.get('scratchpad_root', DEFAULT_SESSION_DIR / 'scratchpad'))).resolve(),
    )


def usage_from_payload(payload: JSONDict | None) -> UsageStats:
    if not isinstance(payload, dict):
        return UsageStats()
    return UsageStats(
        input_tokens=_optional_int(payload.get('input_tokens')) or 0,
        output_tokens=_optional_int(payload.get('output_tokens')) or 0,
        cache_creation_input_tokens=_optional_int(payload.get('cache_creation_input_tokens')) or 0,
        cache_read_input_tokens=_optional_int(payload.get('cache_read_input_tokens')) or 0,
        reasoning_tokens=_optional_int(payload.get('reasoning_tokens')) or 0,
    )


def _deserialize_pricing(payload: Any) -> ModelPricing:
    if not isinstance(payload, dict):
        return ModelPricing()
    return ModelPricing(
        input_cost_per_million_tokens_usd=_optional_float(payload.get('input_cost_per_million_tokens_usd')) or 0.0,
        output_cost_per_million_tokens_usd=_optional_float(payload.get('output_cost_per_million_tokens_usd')) or 0.0,
        cache_creation_input_cost_per_million_tokens_usd=(
            _optional_float(payload.get('cache_creation_input_cost_per_million_tokens_usd'))
            or 0.0
        ),
        cache_read_input_cost_per_million_tokens_usd=(
            _optional_float(payload.get('cache_read_input_cost_per_million_tokens_usd'))
            or 0.0
        ),
    )


def _deserialize_output_schema(payload: Any) -> OutputSchemaConfig | None:
    if not isinstance(payload, dict):
        return None
    schema = payload.get('schema')
    if not isinstance(schema, dict):
        return None
    name = payload.get('name')
    if not isinstance(name, str) or not name:
        return None
    return OutputSchemaConfig(
        name=name,
        schema=dict(schema),
        strict=bool(payload.get('strict', False)),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None


_VALID_NUDGE_STRATEGIES = {'off', 'structural', 'heuristic'}


def _validate_nudge_strategy(value: Any) -> str:
    s = str(value) if value is not None else 'structural'
    return s if s in _VALID_NUDGE_STRATEGIES else 'structural'


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return None
