"""Session-scoped scheduled prompts: /loop, the cron_create / cron_list / cron_delete tools,
and /goal.

A schedule is an interval ("30s", "5m", "2h", "1d", "every 10 minutes") or a five-field cron
expression ("*/15 * * * *", "0 9 * * 1-5"; numbers, *, */n, ranges and lists). Jobs live only
while the chat runs. A due job is sent as the next prompt when the agent is idle; while you sit
at the prompt the chat wakes up for it. SIGIL_MAX_SCHEDULED_JOBS (20) limits how many exist.

/goal <goal> keeps working on a goal: after each answer that does not end with GOAL MET, the
chat sends "continue" again, at most SIGIL_GOAL_MAX_ROUNDS (5) times.
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
    source: str = 'user'  # 'user' (/loop) or 'model' (the cron_create tool)

    def advance(self, now: float) -> None:
        self.runs += 1
        if self.interval:
            self.next_run = now + self.interval
        elif self.cron:
            self.next_run = next_cron_time(self.cron, datetime.fromtimestamp(now)).timestamp()


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


class Scheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

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
        next_run = now + interval if interval else next_cron_time(cron, datetime.fromtimestamp(now)).timestamp()
        job = Job(f'job_{next(_IDS)}', schedule.strip(), prompt.strip(), recurring, next_run, interval, cron,
                  source=source)
        with self._lock:  # check and add in one step: parallel calls cannot pass the limit together
            if limit > 0 and len(self.jobs) >= limit:
                raise ValueError(f'{limit} scheduled prompts already exist (SIGIL_MAX_SCHEDULED_JOBS)')
            self.jobs[job.job_id] = job
        return job

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id == 'all':
                found = bool(self.jobs)
                self.jobs.clear()
                return found
            return self.jobs.pop(job_id, None) is not None

    def pop_due(self, now: float | None = None) -> Job | None:
        """The next job whose time has come (recurring jobs are moved to their next time)."""
        now = time.time() if now is None else now
        with self._lock:
            due = sorted((job for job in self.jobs.values() if job.next_run <= now), key=lambda j: j.next_run)
            if not due:
                return None
            job = due[0]
            if job.recurring:
                job.advance(now)
            else:
                del self.jobs[job.job_id]
            return job

    def seconds_until_next(self, now: float | None = None) -> float | None:
        now = time.time() if now is None else now
        with self._lock:
            return min((job.next_run - now for job in self.jobs.values()), default=None)

    def render(self) -> str:
        if not self.jobs:
            return 'No scheduled prompts.'
        lines = ['Scheduled prompts:']
        for job in self.jobs.values():
            when = datetime.fromtimestamp(job.next_run).strftime('%H:%M:%S')
            kind = 'every ' + job.schedule if job.recurring else 'once at ' + job.schedule
            lines.append(f'  {job.job_id}  {kind}  next {when}  runs {job.runs}: {job.prompt[:80]}')
        return '\n'.join(lines)


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
