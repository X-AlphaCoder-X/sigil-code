"""Session-scoped scheduled prompts: /loop, the cron_create / cron_list / cron_delete tools,
and /goal.

A schedule is an interval ("30s", "5m", "2h", "1d", "every 10 minutes") or a five-field cron
expression ("*/15 * * * *", "0 9 * * 1-5"; numbers, *, */n, ranges and lists). Jobs live only
while the chat runs. A due job is sent as the next prompt when the agent is idle; while you sit
at the prompt the chat wakes up for it. SIGIL_MAX_SCHEDULED_JOBS (20) limits how many exist.

/goal <goal> keeps working on a goal: after each answer that does not end with GOAL MET, the
chat sends "continue" again, at most SIGIL_GOAL_MAX_ROUNDS (5) times.

/loop <prompt> with no interval is self-paced: the prompt runs now, and at the end of each run the
agent picks the next one with schedule_wakeup (60 s to 1 h, with a reason shown to the user) or
ends the loop (stop=true). A run that does neither gets one fallback run 20 minutes later; the loop
ends if that run does neither too. Recurring prompts and loops end after 7 days (one final run).

The prompt cache: a run due exactly when the cache would expire (a 5m loop on a 5-minute cache)
goes 15 s early, so it still reads the cache; the lifetime is worked out for the current model at
each run, so a /model switch is followed. Nothing here changes the system prompt or the tools: a
run is a new message at the end of the conversation, as if typed.

The prompts of a session are saved next to it (loops/<session id>.json) and restored when it is
resumed, except recurring ones past 7 days, one-shots whose time passed while the chat was closed,
and a self-paced loop whose next run came due while it was closed (SIGIL_LOOP_RESTORE=0 restores
nothing). SIGIL_DISABLE_CRON=1 turns all of this off.
"""
from __future__ import annotations

import itertools
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

GOAL_DONE = 'GOAL MET'
LOOP_MIN_DELAY, LOOP_MAX_DELAY = 60, 3600   # the self-paced range (Claude Code: one minute to one hour)
LOOP_FALLBACK_DELAY = 1200                  # one fallback run when a run neither reschedules nor stops
MAX_AGE = 7 * 86400                         # recurring prompts and loops end after 7 days
WARM_LEAD = 15                              # seconds early for a run due when the cache would expire
_MAX_PROMPT = 8000
_STATE_VERSION = 1
_UNITS = {'s': 1, 'sec': 1, 'second': 1, 'm': 60, 'min': 60, 'minute': 60, 'h': 3600, 'hour': 3600,
          'd': 86400, 'day': 86400}
_IDS = itertools.count(1)


def parse_interval(text: str) -> float | None:
    match = re.fullmatch(r'\s*(?:every\s+)?(\d+(?:\.\d+)?)\s*([a-z]+?)s?\s*', text.lower())
    if not match or match.group(2) not in _UNITS:
        return None
    seconds = float(match.group(1)) * _UNITS[match.group(2)]
    return seconds if seconds >= 1 else None


def _field_values(spec: str, low: int, high: int) -> set[int]:
    values: set[int] = set()
    for part in spec.split(','):
        step = 1
        if '/' in part:
            part, raw_step = part.split('/', 1)
            step = int(raw_step)
        if part in ('*', ''):
            start, end = low, high
        elif '-' in part:
            start, end = (int(x) for x in part.split('-', 1))
        else:
            start = end = int(part)
        if start < low or end > high or step < 1:
            raise ValueError(f'{spec} is out of range {low}-{high}')
        values.update(range(start, end + 1, step))
    return values


def parse_cron(text: str) -> tuple[set[int], ...] | None:
    fields = text.split()
    if len(fields) != 5:
        return None
    try:
        return (_field_values(fields[0], 0, 59), _field_values(fields[1], 0, 23), _field_values(fields[2], 1, 31),
                _field_values(fields[3], 1, 12), _field_values(fields[4], 0, 6))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return None


def next_cron_time(spec: tuple[set[int], ...], after: datetime) -> datetime:
    minutes, hours, days, months, weekdays = spec
    moment = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        cron_weekday = (moment.weekday() + 1) % 7  # cron: 0 = Sunday
        if (moment.minute in minutes and moment.hour in hours and moment.day in days
                and moment.month in months and cron_weekday in weekdays):
            return moment
        moment += timedelta(minutes=1)
    raise ValueError('the cron expression never matches')


@dataclass
class Job:
    job_id: str
    schedule: str
    prompt: str
    recurring: bool
    next_run: float
    interval: float | None = None
    cron: tuple | None = field(default=None, repr=False)
    runs: int = 0
    source: str = 'user'  # 'user' (/loop), 'model' (the cron_create tool) or 'loop' (a self-paced run)
    created_at: float = 0.0
    final: bool = False   # the last run of a recurring prompt (7 days)

    def advance(self, now: float, lead: float = 0.0) -> None:
        self.runs += 1
        if self.interval:
            self.next_run = now + max(1.0, self.interval - lead)
        elif self.cron:
            self.next_run = next_cron_time(self.cron, datetime.fromtimestamp(now)).timestamp()


@dataclass
class SelfPacedLoop:
    loop_id: str
    prompt: str
    created_at: float
    runs: int = 0
    armed: bool = False             # this run picked the next one (or stopped the loop)
    fallback_pending: bool = False  # the next run is the fallback: the loop ends unless it reschedules
    reason: str = ''
    ended: str = ''                 # why schedule_wakeup ended it during a run


def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds % 3600 == 0:
        return f'{seconds // 3600}h'
    if seconds >= 60 and seconds % 60 == 0:
        return f'{seconds // 60}m'
    return f'{seconds // 60}m{seconds % 60:02d}s' if seconds >= 60 else f'{seconds}s'


def _model_min_interval() -> float:
    try:
        value = float(os.environ.get('SIGIL_CRON_MIN_INTERVAL', '') or 60)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        value = 60.0
    return max(value, 0.0)


def prompt_text(job: Job) -> str:
    """What the chat sends when a job is due. A job the assistant scheduled is labelled as such,
    so text it picked up from a file or a page cannot come back later looking like the user."""
    if job.source != 'model':
        return job.prompt
    return (f'[Scheduled prompt {job.job_id} ({job.schedule}), set up earlier by the assistant with '
            f'cron_create; not typed by the user now]\n{job.prompt}')


def loop_prompt_text(loop: SelfPacedLoop) -> str:
    """One run of a self-paced loop: the user's prompt with what to do at the end of the run."""
    return (f'[Self-paced /loop {loop.loop_id}, run {loop.runs}, started by the user. At the end of this run '
            'call schedule_wakeup: delay_seconds (60 to 3600) for when the next run should happen and a short '
            'reason, or stop=true when the task is finished.]\n' + loop.prompt)


class Scheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.loop: SelfPacedLoop | None = None
        # () -> (seconds or None, description): how long the prompt cache lasts for the current model
        self.cache_lifetime: Any = None
        self._last_ended: SelfPacedLoop | None = None
        self._lock = threading.Lock()

    def _lifetime(self) -> tuple[int | None, str]:
        if self.cache_lifetime is None:
            return None, 'unknown'
        try:
            return self.cache_lifetime()
        except Exception as _ignored:  # never let a pricing question stop a scheduled prompt
            _log_ignored(_ignored)
            return None, 'unknown'

    def warm_lead(self, interval: float | None) -> float:
        """15 s early for a run due exactly when the cache would expire (Claude Code does the same for
        a 5m loop), so the run still reads the cache instead of writing it again."""
        lifetime, _ = self._lifetime()
        return float(WARM_LEAD) if interval and lifetime and abs(interval - lifetime) < 1 else 0.0

    def add(self, schedule: str, prompt: str, *, recurring: bool = True, now: float | None = None,
            source: str = 'user') -> Job:
        now = time.time() if now is None else now
        try:
            limit = int(os.environ.get('SIGIL_MAX_SCHEDULED_JOBS', '20'))
        except ValueError as _ignored:
            _log_ignored(_ignored)
            limit = 20
        if not prompt.strip():
            raise ValueError('the prompt is empty')
        interval = parse_interval(schedule)
        cron = None if interval else parse_cron(schedule)
        if interval is None and cron is None:
            raise ValueError(f'not an interval (5m, 2h) or a cron expression (*/15 * * * *): {schedule!r}')
        minimum = _model_min_interval()
        if source == 'model' and interval is not None and interval < minimum:
            raise ValueError(f'the shortest interval the assistant may schedule is {minimum:g} s '
                             '(SIGIL_CRON_MIN_INTERVAL)')
        if len(prompt) > _MAX_PROMPT:
            raise ValueError(f'the prompt is longer than {_MAX_PROMPT} characters')
        lead = self.warm_lead(interval) if recurring else 0.0
        next_run = (now + max(1.0, interval - lead) if interval
                    else next_cron_time(cron, datetime.fromtimestamp(now)).timestamp())
        job = Job(f'job_{next(_IDS)}', schedule.strip(), prompt.strip(), recurring, next_run, interval, cron,
                  source=source, created_at=now)
        self._insert(job, limit)
        return job

    def _insert(self, job: Job, limit: int | None = None) -> None:
        if limit is None:
            try:
                limit = int(os.environ.get('SIGIL_MAX_SCHEDULED_JOBS', '20'))
            except ValueError as _ignored:
                _log_ignored(_ignored)
                limit = 20
        with self._lock:  # check and add in one step: parallel calls cannot pass the limit together
            if limit > 0 and len(self.jobs) >= limit:
                raise ValueError(f'{limit} scheduled prompts already exist (SIGIL_MAX_SCHEDULED_JOBS)')
            self.jobs[job.job_id] = job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id == 'all':
                found = bool(self.jobs) or self.loop is not None
                self.jobs.clear()
                self.loop = None
                return found
            if self.loop is not None and job_id == self.loop.loop_id:
                self._drop_loop_jobs()
                self.loop = None
                return True
            job = self.jobs.pop(job_id, None)
            if job is not None and job.source == 'loop':
                self.loop = None  # deleting the pending run ends the loop
            return job is not None

    def _drop_loop_jobs(self) -> None:
        for job_id in [j.job_id for j in self.jobs.values() if j.source == 'loop']:
            del self.jobs[job_id]

    def pop_due(self, now: float | None = None) -> Job | None:
        """The next job whose time has come (recurring jobs are moved to their next time; a recurring
        job 7 days old runs one final time and is removed). A missed run is not caught up: a job that
        came due several times while the agent was busy runs once."""
        now = time.time() if now is None else now
        lead = {job_id: self.warm_lead(job.interval) for job_id, job in list(self.jobs.items()) if job.recurring}
        with self._lock:
            due = sorted((job for job in self.jobs.values() if job.next_run <= now), key=lambda j: j.next_run)
            if not due:
                return None
            job = due[0]
            if job.recurring and job.created_at and now - job.created_at >= MAX_AGE:
                job.final = True
                del self.jobs[job.job_id]
            elif job.recurring:
                job.advance(now, lead.get(job.job_id, 0.0))
            else:
                del self.jobs[job.job_id]
            if job.source == 'loop' and self.loop is not None:
                self.loop.runs += 1
                self.loop.armed = False
            return job

    # ---------------------------------------------------------------- self-paced /loop

    def start_loop(self, prompt: str, now: float | None = None) -> SelfPacedLoop:
        """Start a self-paced loop; its first run is the caller's next prompt."""
        now = time.time() if now is None else now
        prompt = prompt.strip()
        if not prompt:
            raise ValueError('the prompt is empty')
        if len(prompt) > _MAX_PROMPT:
            raise ValueError(f'the prompt is longer than {_MAX_PROMPT} characters')
        with self._lock:
            if self.loop is not None:
                raise ValueError(f'self-paced loop {self.loop.loop_id} is already running '
                                 f'(/loop stop {self.loop.loop_id} first)')
            self.loop = SelfPacedLoop(f'loop_{next(_IDS)}', prompt, now, runs=1)
            return self.loop

    def wakeup(self, delay: float, reason: str, now: float | None = None) -> tuple[float, str]:
        """schedule_wakeup: the loop's next run in delay seconds (60 to 3600). Returns the delay used
        and a note about the prompt cache for the agent."""
        now = time.time() if now is None else now
        loop = self.loop
        if loop is None:
            raise ValueError('no self-paced /loop is running (it is started with /loop <prompt>)')
        delay = min(max(float(delay), LOOP_MIN_DELAY), LOOP_MAX_DELAY)
        lifetime, about = self._lifetime()
        note = f'The prompt cache here lasts {about}; the provider does not say exactly how long.'
        if lifetime:
            if lifetime - WARM_LEAD < delay <= lifetime + 60:
                delay = float(lifetime - WARM_LEAD)
                note = (f'Moved to {_duration(delay)} so the next run still reads the prompt cache '
                        f'(it lasts {about}).')
            elif delay > lifetime:
                note = (f'The prompt cache here lasts {about}, so the next run reads the whole prompt again at '
                        f'full price. {_duration(lifetime - WARM_LEAD)} or less keeps it warm.')
            else:
                note = f'The next run reads the prompt cache (it lasts {about}).'
        job = Job(f'job_{next(_IDS)}', _duration(delay), loop.prompt, False, now + delay, delay,
                  source='loop', created_at=now)
        with self._lock:
            self._drop_loop_jobs()
        self._insert(job)
        loop.armed, loop.fallback_pending, loop.reason = True, False, ' '.join(str(reason).split())[:200]
        return delay, note

    def stop_loop(self, reason: str = '') -> bool:
        with self._lock:
            if self.loop is None:
                return False
            self._drop_loop_jobs()
            self.loop.armed = True
            self.loop.ended = ' '.join(str(reason).split())[:200] or 'the agent ended it'
            ended, self.loop = self.loop, None
            self._last_ended = ended
            return True

    def end_loop_run(self, *, interrupted: bool = False, now: float | None = None) -> str:
        """After a run of the loop: what happens next, in one line for the user."""
        now = time.time() if now is None else now
        ended, self._last_ended = self._last_ended, None
        loop = self.loop
        if loop is None:
            return f'Loop {ended.loop_id} ended: {ended.ended}.' if ended else ''
        if interrupted:
            self.delete(loop.loop_id)
            return f'Loop {loop.loop_id} stopped: the run was interrupted.'
        if now - loop.created_at >= MAX_AGE:
            self.delete(loop.loop_id)
            return f'Loop {loop.loop_id} ended after 7 days (/loop {loop.prompt[:40]} starts it again).'
        pending = next((job for job in self.jobs.values() if job.source == 'loop'), None)
        if loop.armed and pending is not None:
            when = datetime.fromtimestamp(pending.next_run).strftime('%H:%M:%S')
            return f'Loop {loop.loop_id}: next run in {pending.schedule} at {when}: {loop.reason or "no reason given"}'
        if loop.fallback_pending:
            self.delete(loop.loop_id)
            return f'Loop {loop.loop_id} ended: the fallback run did not schedule another.'
        job = Job(f'job_{next(_IDS)}', _duration(LOOP_FALLBACK_DELAY), loop.prompt, False,
                  now + LOOP_FALLBACK_DELAY, float(LOOP_FALLBACK_DELAY), source='loop', created_at=now)
        try:
            self._insert(job)
        except ValueError as exc:
            self.delete(loop.loop_id)
            return f'Loop {loop.loop_id} ended: no room for its next run ({exc}).'
        loop.fallback_pending = True
        return (f'Loop {loop.loop_id}: this run did not pick a next one, so one fallback run follows in '
                f'{_duration(LOOP_FALLBACK_DELAY)}; the loop ends if that run does not schedule another.')

    def clear(self) -> None:
        with self._lock:
            self.jobs.clear()
            self.loop = None

    def seconds_until_next(self, now: float | None = None) -> float | None:
        now = time.time() if now is None else now
        with self._lock:
            return min((job.next_run - now for job in self.jobs.values()), default=None)

    def render(self) -> str:
        if not self.jobs and self.loop is None:
            return 'No scheduled prompts.'
        lines = ['Scheduled prompts:']
        if self.loop is not None:
            pending = next((job for job in self.jobs.values() if job.source == 'loop'), None)
            when = datetime.fromtimestamp(pending.next_run).strftime('%H:%M:%S') if pending else 'after this run'
            lines.append(f'  {self.loop.loop_id}  self-paced  next {when}  runs {self.loop.runs}: '
                         f'{self.loop.prompt[:80]}' + (f'  ({self.loop.reason})' if self.loop.reason else ''))
        for job in self.jobs.values():
            if job.source == 'loop':
                continue
            when = datetime.fromtimestamp(job.next_run).strftime('%H:%M:%S')
            kind = 'every ' + job.schedule if job.recurring else 'once at ' + job.schedule
            lines.append(f'  {job.job_id}  {kind}  next {when}  runs {job.runs}: {job.prompt[:80]}')
        return '\n'.join(lines)

    # ---------------------------------------------------------------- saved with the session

    def to_state(self) -> dict[str, Any]:
        with self._lock:
            jobs = [{'schedule': j.schedule, 'prompt': j.prompt, 'recurring': j.recurring, 'next_run': j.next_run,
                     'created_at': j.created_at, 'source': j.source, 'runs': j.runs}
                    for j in self.jobs.values() if j.source in ('user', 'model')]
            loop = None
            if self.loop is not None:
                pending = next((j for j in self.jobs.values() if j.source == 'loop'), None)
                loop = {'prompt': self.loop.prompt, 'created_at': self.loop.created_at, 'runs': self.loop.runs,
                        'reason': self.loop.reason, 'fallback_pending': self.loop.fallback_pending,
                        'next_run': pending.next_run if pending else None}
        return {'version': _STATE_VERSION, 'jobs': jobs, 'loop': loop}

    def restore(self, state: Any, now: float | None = None) -> list[str]:
        """Bring back the prompts of a resumed session. Returns one line per thing worth telling the
        user: what came back, and what did not and why. Saved data is checked like any other input."""
        now = time.time() if now is None else now
        if not isinstance(state, dict) or state.get('version') != _STATE_VERSION:
            raise ValueError('not a saved schedule of this version')
        restored, notes = 0, []
        for item in state.get('jobs') or []:
            try:
                job = self._restored_job(item, now)
                if job is not None:
                    self._insert(job)
                    restored += 1
            except (TypeError, ValueError) as exc:
                _log_ignored(exc)
                notes.append(f'A saved scheduled prompt was not restored: {exc}.')
        loop = state.get('loop')
        if isinstance(loop, dict):
            try:
                note = self._restore_loop(loop, now)
            except (TypeError, ValueError) as exc:
                _log_ignored(exc)
                note = f'The saved self-paced loop was not restored: {exc}.'
            if note:
                notes.append(note)
            if self.loop is not None:
                restored += 1
        if restored:
            notes.insert(0, f'Restored {restored} scheduled prompt(s) from this session (/loop lists them, '
                            '/loop stop ends them).')
        return notes

    def _restored_job(self, item: Any, now: float) -> Job | None:
        if not isinstance(item, dict):
            raise ValueError('not an object')
        prompt, schedule, source = item.get('prompt'), item.get('schedule'), item.get('source')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > _MAX_PROMPT:
            raise ValueError('a missing or oversized prompt')
        if source not in ('user', 'model') or not isinstance(schedule, str):
            raise ValueError('an unknown kind of prompt')
        recurring = item.get('recurring') is not False
        created_at = float(item.get('created_at') or now)
        next_run = float(item.get('next_run') or now)
        interval = parse_interval(schedule)
        cron = None if interval else parse_cron(schedule)
        if interval is None and cron is None:
            raise ValueError(f'an unreadable schedule {schedule[:30]!r}')
        label = f'{schedule} {prompt[:40]}'
        if recurring and now - created_at >= MAX_AGE:
            raise ValueError(f'"{label}" is older than 7 days')
        if not recurring and next_run <= now:
            raise ValueError(f'"{label}" was due while the chat was closed')
        if recurring and next_run <= now:  # no catch-up: the first run due after now
            if interval:
                next_run += ((now - next_run) // interval + 1) * interval
            else:
                next_run = next_cron_time(cron, datetime.fromtimestamp(now)).timestamp()
        return Job(f'job_{next(_IDS)}', schedule, prompt, recurring, next_run, interval, cron,
                   runs=int(item.get('runs') or 0), source=source, created_at=created_at)

    def _restore_loop(self, item: dict[str, Any], now: float) -> str:
        prompt = item.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > _MAX_PROMPT:
            return 'The saved self-paced loop was not restored: its prompt is missing or too long.'
        created_at = float(item.get('created_at') or now)
        next_run = item.get('next_run')
        if now - created_at >= MAX_AGE:
            return f'The self-paced loop "{prompt[:40]}" was not restored: it is older than 7 days.'
        if not isinstance(next_run, (int, float)) or next_run <= now:
            return (f'The self-paced loop "{prompt[:40]}" was not restored: its next run came due while the chat '
                    'was closed. /loop <prompt> starts it again.')
        loop = SelfPacedLoop(f'loop_{next(_IDS)}', prompt.strip(), created_at, runs=int(item.get('runs') or 0),
                             armed=True, fallback_pending=item.get('fallback_pending') is True,
                             reason=str(item.get('reason') or '')[:200])
        job = Job(f'job_{next(_IDS)}', _duration(next_run - now), loop.prompt, False, float(next_run),
                  None, source='loop', created_at=now)
        self._insert(job)
        self.loop = loop
        return ''


_STATE_ID = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


def state_path(session_directory: Any, session_id: str) -> Any:
    from pathlib import Path

    if not isinstance(session_id, str) or not _STATE_ID.match(session_id):
        raise ValueError('not a session id')
    return Path(session_directory) / 'loops' / f'{session_id}.json'


def save_state(scheduler: Scheduler, session_directory: Any, session_id: str) -> None:
    """Write (or remove, when nothing is scheduled) the session's saved prompts. Raises OSError/ValueError."""
    import json

    from .settings import write_text_atomic

    path = state_path(session_directory, session_id)
    state = scheduler.to_state()
    if not state['jobs'] and state['loop'] is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(state, indent=2) + '\n')


def restore_state(scheduler: Scheduler, session_directory: Any, session_id: str,
                  now: float | None = None) -> list[str]:
    """Restore a resumed session's prompts; lines for the user. A broken file is reported, not fatal."""
    import json

    try:
        path = state_path(session_directory, session_id)
        if not path.is_file():
            return []
        if os.environ.get('SIGIL_LOOP_RESTORE', '1').strip().lower() in ('0', 'false', 'no', 'off'):
            return ['Saved scheduled prompts were not restored (SIGIL_LOOP_RESTORE=0).']
        if path.stat().st_size > 512_000:
            raise ValueError('the file is larger than 512 KB')
        return scheduler.restore(json.loads(path.read_text(encoding='utf-8')), now)
    except (OSError, ValueError) as exc:
        _log_ignored(exc)
        return [f'The saved scheduled prompts of this session could not be read ({type(exc).__name__}); '
                'none were restored.']


def register_tools(agent: Any, scheduler: Scheduler) -> None:
    """cron_create / cron_list / cron_delete for this chat (the model can schedule prompts too)."""
    from .agent_tools import SigilTool, ToolExecutionError

    def _create(arguments: dict[str, Any], context: Any) -> Any:
        try:
            job = scheduler.add(str(arguments.get('schedule', '')), str(arguments.get('prompt', '')),
                                recurring=arguments.get('recurring', True) is not False, source='model')
        except ValueError as exc:
            raise ToolExecutionError(f'cron_create: {exc}') from exc
        return f'Scheduled {job.job_id} ({job.schedule}).', {'job_id': job.job_id}

    def _list(arguments: dict[str, Any], context: Any) -> Any:
        return scheduler.render()

    def _delete(arguments: dict[str, Any], context: Any) -> Any:
        job_id = str(arguments.get('id', ''))
        return f'Deleted {job_id}.' if scheduler.delete(job_id) else f'No scheduled prompt {job_id}.'

    tools = {
        'cron_create': SigilTool('cron_create', 'Schedule a prompt in this chat: an interval (5m, 2h) or a '
                                 'cron expression; recurring unless recurring=false.',
                                 {'type': 'object', 'properties': {'schedule': {'type': 'string'},
                                                                   'prompt': {'type': 'string'},
                                                                   'recurring': {'type': 'boolean'}},
                                  'required': ['schedule', 'prompt']}, _create),
        'cron_list': SigilTool('cron_list', 'List the prompts scheduled in this chat.',
                               {'type': 'object', 'properties': {}}, _list),
        'cron_delete': SigilTool('cron_delete', 'Delete a scheduled prompt by id (or "all").',
                                 {'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']},
                                 _delete),
    }
    def _wakeup(arguments: dict[str, Any], context: Any) -> Any:
        if arguments.get('stop') is True:
            if not scheduler.stop_loop(str(arguments.get('reason', ''))):
                raise ToolExecutionError('schedule_wakeup: no self-paced /loop is running')
            return 'The loop is stopped; it will not run again.', {'stopped': True}
        try:
            delay, note = scheduler.wakeup(float(arguments.get('delay_seconds', 0) or 0),
                                           str(arguments.get('reason', '')))
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(f'schedule_wakeup: {exc}') from exc
        return f'The next run starts in {_duration(delay)}. {note}', {'delay_seconds': delay}

    # The description is fixed text, so the tool list (the start of every cached prompt) never changes
    # with the cache lifetime or the model: those go in the result, at the end of the conversation.
    tools['schedule_wakeup'] = SigilTool(
        'schedule_wakeup',
        'Self-paced /loop only: at the end of a run, schedule the next run in delay_seconds (60 to 3600) with a '
        'short reason shown to the user, or stop=true when the task is finished. Short waits while something '
        'is about to change, longer ones when nothing is pending. A run that does neither gets one fallback '
        'run 20 minutes later. The result says how long the prompt cache lasts here.',
        {'type': 'object', 'properties': {'delay_seconds': {'type': 'number'}, 'reason': {'type': 'string'},
                                          'stop': {'type': 'boolean'}}},
        _wakeup)
    agent.tool_registry = {**agent.tool_registry, **tools}


def goal_rounds() -> int:
    try:
        return max(1, int(os.environ.get('SIGIL_GOAL_MAX_ROUNDS', '5')))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return 5


def goal_prompt(goal: str, first: bool) -> str:
    lead = f'Goal: {goal}' if first else f'Keep working towards the goal: {goal}'
    return (f'{lead}\nWork on it step by step. When the goal is fully met, end your reply with the line '
            f'{GOAL_DONE}.')
