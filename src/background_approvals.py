"""Approval prompts from background agents, answered in the main chat.

A background agent that needs approval (and has no SIGIL_BACKGROUND_ASK policy
answer) writes <background folder>/<id>.ask.json and waits for <id>.answer.json.
The main chat shows the request before the next prompt (a notice appears while
you sit at the prompt: press Enter) and asks once / session / always / deny,
like a normal approval. From another terminal: `sigil agent-approve <id> <answer>`.
The agent waits SIGIL_BACKGROUND_ASK_TIMEOUT seconds (300); after that, or with
0, the call is refused as before.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

ANSWERS = {'y': 'once', 'yes': 'once', 'once': 'once', 's': 'session', 'session': 'session',
           'a': 'always', 'always': 'always', 'n': 'deny', 'no': 'deny', 'deny': 'deny', '': 'deny'}


def _timeout() -> float:
    try:
        return max(0.0, float(os.environ.get('SIGIL_BACKGROUND_ASK_TIMEOUT', '300')))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return 300.0


def _write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(tmp, path)


def ask_path(root: Path, background_id: str) -> Path:
    return Path(root) / f'{background_id}.ask.json'


def answer_path(root: Path, background_id: str) -> Path:
    return Path(root) / f'{background_id}.answer.json'


def ask(root: Path, background_id: str, request: dict[str, Any], *, timeout: float | None = None,
        on_wait: Callable[[str | None], None] | None = None) -> str | None:
    """Worker side: post the request and wait for the main chat's answer (None on timeout)."""
    timeout = _timeout() if timeout is None else timeout
    if timeout <= 0:
        return None
    ask_id = uuid4().hex[:8]
    payload = {'ask_id': ask_id, 'background_id': background_id, 'created': time.time(),
               'tool': request.get('tool'), 'arguments': request.get('arguments'),
               'reason': request.get('reason'), 'rule': request.get('rule')}
    question, reply = ask_path(root, background_id), answer_path(root, background_id)
    reply.unlink(missing_ok=True)
    _write(question, payload)
    if on_wait is not None:
        on_wait(f'waiting for approval: {request.get("tool")}')
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if reply.exists():
                try:
                    data = json.loads(reply.read_text(encoding='utf-8'))
                except (OSError, ValueError) as _ignored:
                    _log_ignored(_ignored)
                    data = {}
                if data.get('ask_id') == ask_id:
                    return ANSWERS.get(str(data.get('answer', '')).strip().lower(), 'deny')
            time.sleep(0.25)
        return None
    finally:
        question.unlink(missing_ok=True)
        reply.unlink(missing_ok=True)
        if on_wait is not None:
            on_wait(None)


def pending(root: Path) -> list[dict[str, Any]]:
    found = []
    for path in sorted(Path(root).glob('bg_*.ask.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            continue
        if isinstance(data, dict) and data.get('ask_id'):
            found.append(data)
    return found


def answer(root: Path, background_id: str, reply: str, ask_id: str | None = None) -> bool:
    """Main side: answer the open request of one background agent."""
    open_requests = [p for p in pending(root) if p.get('background_id') == background_id]
    if not open_requests:
        return False
    target = open_requests[0]
    if ask_id and target.get('ask_id') != ask_id:
        return False
    _write(answer_path(root, background_id), {'ask_id': target['ask_id'],
                                              'answer': ANSWERS.get(reply.strip().lower(), 'deny')})
    return True


def render(request: dict[str, Any]) -> str:
    from .terminal import clean_event

    request = clean_event(request)  # shown to the user who approves it: plain text only
    arguments = request.get('arguments') or {}
    detail = arguments.get('command') or arguments.get('path') or json.dumps(arguments)[:200]
    lines = [f'Background agent {request.get("background_id")} asks to use {request.get("tool")}: {detail}']
    if request.get('reason'):
        lines.append(f'  why: {request["reason"]}')
    if request.get('rule'):
        lines.append(f'  rule for "always": {request["rule"]}')
    return '\n'.join(lines)


def answer_pending(root: Path, ask_func: Callable[[str], str], output: Callable[[str], None]) -> int:
    """Chat side: show each open request and send the answer. Returns how many were answered."""
    count = 0
    for request in pending(root):
        output(render(request))
        try:
            reply = ask_func('Allow? [y] once / [s] this session / [a] always / [n] no > ')
        except (EOFError, KeyboardInterrupt):
            reply = 'n'
        if answer(root, str(request.get('background_id')), reply, str(request.get('ask_id'))):
            count += 1
            output(f'  -> {ANSWERS.get(reply.strip().lower(), "deny")}')
    return count


def start_watcher(root: Path, output: Callable[[str], None], interval: float = 2.0) -> threading.Event:
    """Print a one-line notice when a new request arrives while the user is at the prompt."""
    stop = threading.Event()
    seen: set[str] = set()

    def _watch() -> None:
        while not stop.wait(interval):
            for request in pending(root):
                if request['ask_id'] not in seen:
                    seen.add(request['ask_id'])
                    output(f'\n[background agent {request.get("background_id")} is waiting for approval of '
                           f'{request.get("tool")}; press Enter to answer]')

    threading.Thread(target=_watch, name='bg-approvals', daemon=True).start()
    return stop
