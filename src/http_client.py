"""A process-wide urllib opener that drops credentials on a cross-origin redirect.

urllib's default HTTPRedirectHandler carries every non-content request header (Authorization,
Cookie, custom headers) to the redirect target, even on another scheme/host/port. A server Sigil
talks to (an MCP endpoint, a model backend, an HTTP hook, a web_fetch target) could therefore 302
the request to a host it chooses and collect the bearer token or API key. This installs an opener
whose redirect handler keeps only a small allow-list of safe headers when the origin changes.

Override: SIGIL_HTTP_KEEP_AUTH_ON_REDIRECT=1 restores urllib's default behaviour.
"""
from __future__ import annotations

import os
import urllib.request
from urllib.parse import urlsplit

from .debug_log import log_ignored_error as _log_ignored

# Headers that are safe to resend to another origin; everything else is dropped on a cross-origin hop.
_SAFE_REDIRECT_HEADERS = frozenset({
    'accept', 'accept-encoding', 'accept-language', 'accept-charset', 'content-type', 'user-agent',
})


def _origin(url: str) -> tuple[str, str, int | str]:
    parts = urlsplit(url)
    return (parts.scheme.lower(), (parts.hostname or '').lower(), parts.port or '')


def _keep_auth_on_redirect() -> bool:
    return os.environ.get('SIGIL_HTTP_KEEP_AUTH_ON_REDIRECT', '').strip().lower() in ('1', 'true', 'yes', 'on')


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None or _keep_auth_on_redirect():
            return new
        try:
            if _origin(req.full_url) == _origin(new.full_url):
                return new  # same origin: nothing to strip
            for store in (getattr(new, 'headers', {}), getattr(new, 'unredirected_hdrs', {})):
                for name in [key for key in store if key.lower() not in _SAFE_REDIRECT_HEADERS]:
                    store.pop(name, None)
        except Exception as _ignored:  # never let header hygiene break a request
            _log_ignored(_ignored)
        return new


_INSTALLED = False


def install() -> None:
    """Install the safe-redirect opener as urllib's default (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    # Keep urllib's usual default handlers, only swapping the redirect handler, so proxies, cookies
    # and HTTPS still work as before. urlopen() with no opener uses this installed one.
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    urllib.request.install_opener(opener)
    _INSTALLED = True
