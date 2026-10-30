"""Hide secrets in text before it is shared (/export --redact).

Replaced with [REDACTED]: values of environment variables whose names look secret
(KEY, TOKEN, SECRET, PASSWORD, ...), common key formats (sk-..., ghp_..., AKIA...,
xox?-..., AIza..., JWTs, PEM private keys), Bearer / Basic credentials, and
password=..., api_key: ... style assignments. Not perfect: read the file before
you share it.
"""
from __future__ import annotations

import os
import re
import threading

MASK = '[REDACTED]'
# Keys Sigil knows that are not in the environment (--api-key, a literal profile key): kept in
# memory only, so log lines and exports can hide them too. Bounded.
_KNOWN: set[str] = set()
_KNOWN_LOCK = threading.Lock()
_PLACEHOLDERS = {'local-token', 'none', 'null', 'dummy', 'test'}


def register_secret(value: object) -> None:
    """Remember a key that is not in the environment so redact_text hides it."""
    text = str(value or '').strip()
    if len(text) < 8 or text.lower() in _PLACEHOLDERS:
        return
    with _KNOWN_LOCK:
        if len(_KNOWN) < 256:
            _KNOWN.add(text)
_SECRET_NAME = re.compile(r'(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION)', re.I)
_PATTERNS = (
    re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----'),
    re.compile(r'\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}'),
    re.compile(r'\bsk-ant-[A-Za-z0-9_\-]{16,}'),
    re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'\bxox[abposr]-[A-Za-z0-9\-]{10,}'),
    re.compile(r'\bAIza[0-9A-Za-z_\-]{35}\b'),
    re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}'),
)
_AUTH = re.compile(r'\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=\-]{12,}', re.I)
_ASSIGN = re.compile(
    r'(?P<name>["\']?[A-Za-z0-9_\-]*(?:api[_-]?key|token|secret|password|passwd|pwd)["\']?\s*[:=]\s*)'
    r'(?P<quote>["\']?)(?P<value>[^\s"\',;]{4,})', re.I)


def _env_secrets() -> list[str]:
    values = []
    for name, value in os.environ.items():
        if _SECRET_NAME.search(name) and value and len(value) >= 8:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def redact_text(text: str) -> str:
    with _KNOWN_LOCK:
        known = list(_KNOWN)
    for value in sorted(set(_env_secrets()) | set(known), key=len, reverse=True):
        text = text.replace(value, MASK)
    for pattern in _PATTERNS:
        text = pattern.sub(MASK, text)
    text = _AUTH.sub(lambda m: f'{m.group(1)} {MASK}', text)
    text = _ASSIGN.sub(lambda m: f"{m.group('name')}{m.group('quote')}{MASK}", text)
    return text
