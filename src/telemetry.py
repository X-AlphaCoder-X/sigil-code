"""OpenTelemetry export (opt-in) to a local collector: tokens, cost and tool calls.

Turn it on with SIGIL_ENABLE_TELEMETRY=1 (or an OTLP endpoint
variable). OTEL_EXPORTER_OTLP_LOGS_ENDPOINT is used as given. Records
go as OTLP/HTTP JSON log records to <endpoint>/v1/logs (default http://localhost:4318), with
OTEL_EXPORTER_OTLP_HEADERS ("key=value,key2=value2") as extra headers and OTEL_SERVICE_NAME
(default "sigil-code") as the service. Sent in the background every SIGIL_OTEL_INTERVAL
seconds (5) and at exit; a collector that is down only costs a log line. No prompt or file
text is sent: model name, token counts, cost, tool name and whether it worked.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any
from urllib import error, request
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_LOCK = threading.Lock()
_PENDING: list[dict[str, Any]] = []
_STARTED = False


def enabled() -> bool:
    flag = os.environ.get('SIGIL_ENABLE_TELEMETRY', '').strip().lower()
    if flag in ('0', 'false', 'no', 'off'):
        return False
    return (flag in ('1', 'true', 'yes', 'on') or bool(os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'))
            or bool(os.environ.get('OTEL_EXPORTER_OTLP_LOGS_ENDPOINT')))


_EXPORT_FAILURES = {'next': 0.0, 'count': 0}


def endpoint() -> str:
    """OTEL_EXPORTER_OTLP_LOGS_ENDPOINT as given, else OTEL_EXPORTER_OTLP_ENDPOINT + /v1/logs."""
    exact = os.environ.get('OTEL_EXPORTER_OTLP_LOGS_ENDPOINT', '').strip()
    if exact:
        return exact
    return (os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT') or 'http://localhost:4318').rstrip('/') + '/v1/logs'


def _headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    for item in os.environ.get('OTEL_EXPORTER_OTLP_HEADERS', '').split(','):
        key, sep, value = item.partition('=')
        if sep and key.strip():
            headers[key.strip()] = value.strip()
    return headers


def _attribute(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {'key': key, 'value': {'boolValue': value}}
    if isinstance(value, int):
        return {'key': key, 'value': {'intValue': str(value)}}
    if isinstance(value, float):
        return {'key': key, 'value': {'doubleValue': value}}
    return {'key': key, 'value': {'stringValue': str(value)}}


def record(event: dict[str, Any]) -> None:
    """Called for every runtime event; keeps model_usage and tool_result ones."""
    if not enabled():
        return
    kind = event.get('type')
    if kind == 'model_usage':
        name = 'sigil.model_request'
        attrs = {key: event.get(key) for key in ('model', 'input_tokens', 'output_tokens',
                                                  'cache_read_input_tokens', 'cost_usd', 'turn_index')}
    elif kind == 'tool_result':
        name = 'sigil.tool_result'
        attrs = {'tool_name': event.get('tool_name'), 'success': bool(event.get('ok'))}
    else:
        return
    entry = {'timeUnixNano': str(time.time_ns()), 'severityText': 'INFO', 'body': {'stringValue': name},
             'attributes': [_attribute('event.name', name)]
             + [_attribute(k, v) for k, v in attrs.items() if v is not None]}
    with _LOCK:
        _PENDING.append(entry)
    _start()


def _start() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    atexit.register(flush)
    threading.Thread(target=_loop, name='otel-export', daemon=True).start()


def _loop() -> None:
    try:
        interval = max(0.2, float(os.environ.get('SIGIL_OTEL_INTERVAL', '5')))
    except ValueError as _ignored:
        _log_ignored(_ignored)
        interval = 5.0
    while True:
        time.sleep(interval)
        flush()


def flush() -> int:
    """Send what is pending; returns the number of records sent (0 when the collector is down)."""
    with _LOCK:
        batch = list(_PENDING)
        _PENDING.clear()
    if not batch:
        return 0
    payload = {'resourceLogs': [{
        'resource': {'attributes': [_attribute('service.name', os.environ.get('OTEL_SERVICE_NAME', 'sigil-code'))]},
        'scopeLogs': [{'scope': {'name': 'sigil-code'}, 'logRecords': batch}],
    }]}
    req = request.Request(endpoint(), data=json.dumps(payload).encode('utf-8'), headers=_headers(), method='POST')
    try:
        with request.urlopen(req, timeout=5) as response:
            response.read()
    except (error.URLError, OSError) as exc:
        try:
            from .debug_log import get_logger

            now = time.monotonic()
            _EXPORT_FAILURES['count'] += 1
            if now >= _EXPORT_FAILURES['next']:  # a collector that is down does not flood the log
                from .debug_log import url_for_log

                get_logger('telemetry').warning('OTLP export to %s failed (%d time(s)): %s', url_for_log(endpoint()),
                                                _EXPORT_FAILURES['count'], exc)
                _EXPORT_FAILURES.update(next=now + 600.0, count=0)
        except Exception as _ignored:
            from .debug_log import log_ignored_error

            log_ignored_error(_ignored)
        return 0
    return len(batch)
