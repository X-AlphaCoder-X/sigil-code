from __future__ import annotations
import logging
import threading
import time
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

_log = logging.getLogger('sigil.token_router')  # under the sigil logger: reaches the log file


def limits_from_env(env=None) -> tuple[int, int]:
    """(rpm, tpm) from SIGIL_RPM_LIMIT / SIGIL_TPM_LIMIT. Unset, 0, negative or not a number
    means no limit: a local server has no rate limit, and a cloud API answers 429 (retried)."""
    import os

    source = os.environ if env is None else env
    values = []
    for name in ('SIGIL_RPM_LIMIT', 'SIGIL_TPM_LIMIT'):
        try:
            value = int(str(source.get(name, '') or '0').strip())
        except ValueError as _ignored:
            _log_ignored(_ignored)
            value = 0
        values.append(max(value, 0))
    return values[0], values[1]


class TokenRouter:
    """
    Optional client-side rate limiter for API calls (RPM/TPM), off by default.
    A limit of 0 means no limit. Set SIGIL_RPM_LIMIT / SIGIL_TPM_LIMIT to stay under
    a provider's quota during parallel execution (e.g. delegated sub-agents).
    """
    _instance: TokenRouter | None = None
    _lock = threading.Lock()

    def __init__(self, rpm_limit: int = 0, tpm_limit: int = 0):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_timestamps: list[float] = []
        self.token_ledger: list[tuple[float, int]] = []
        self._access_lock = threading.Lock()

    @classmethod
    def instance(cls) -> TokenRouter:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def configure(cls, rpm_limit: int = 0, tpm_limit: int = 0) -> TokenRouter:
        with cls._lock:
            cls._instance = cls(rpm_limit=rpm_limit, tpm_limit=tpm_limit)
            return cls._instance

    def acquire(self) -> None:
        max_wait = 65.0
        deadline = time.time() + max_wait
        while True:
            with self._access_lock:
                wait = self._calculate_wait()
                if wait <= 0:
                    self.request_timestamps.append(time.time())
                    return
            capped_wait = min(wait, deadline - time.time())
            if capped_wait <= 0:
                with self._access_lock:
                    self.request_timestamps.append(time.time())
                return
            (_log.info if capped_wait > 2 else _log.debug)('Rate limit (SIGIL_RPM_LIMIT/SIGIL_TPM_LIMIT): '
                                                           'waiting %.1fs', capped_wait)
            time.sleep(capped_wait)

    def record_usage(self, total_tokens: int) -> None:
        with self._access_lock:
            self.token_ledger.append((time.time(), total_tokens))

    def _calculate_wait(self) -> float:
        now = time.time()
        cutoff = now - 60.0
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff]
        self.token_ledger = [e for e in self.token_ledger if e[0] > cutoff]

        wait = 0.0
        if self.rpm_limit > 0 and len(self.request_timestamps) >= self.rpm_limit and self.request_timestamps:
            wait = max(wait, 60.0 - (now - self.request_timestamps[0]) + 0.1)

        current_tpm = sum(tokens for _, tokens in self.token_ledger)
        if self.tpm_limit > 0 and current_tpm >= self.tpm_limit and self.token_ledger:
            wait = max(wait, 60.0 - (now - self.token_ledger[0][0]) + 0.1)

        return wait
