"""Opt-in sandbox for bash on Linux (bubblewrap) and macOS (sandbox-exec).

Off by default. Turn it on with SIGIL_SANDBOX=1 or settings
"sandbox": {"enabled": true}. Inside the sandbox:
  - the whole disk is read-only, except the workspace, the extra working
    folders and a private /tmp (sandbox.writePaths adds more);
  - the network is off, unless sandbox.network is true (all hosts) or
    sandbox.allowedDomains lists hosts: then traffic goes through a small local
    proxy (HTTP_PROXY / HTTPS_PROXY / ALL_PROXY) that only lets those hosts
    through. On Linux the command gets its own network namespace and reaches only
    that proxy, through a socket bound into the sandbox and a small relay inside it
    (sandbox_relay.py), so programs that ignore the proxy settings get no network.
    On macOS only the proxy port can be reached. The settings
    sandbox.network.allowedDomains and sandbox.filesystem.allowWrite work too.
While the sandbox is really active, shell commands run without the
--allow-shell question (sandbox.autoAllow, default true); the destructive
command check stays as it is. Windows has no sandbox here (bash_security
stays in charge). If the sandbox is on but bwrap or sandbox-exec is missing,
commands do not run and the error says how to install it or turn it off.
SIGIL_SANDBOX_BWRAP overrides the bubblewrap command (a command line).
"""
from __future__ import annotations

import os
import select
import shlex
import shutil
import socket
import sys
import threading
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file


class SandboxUnavailable(RuntimeError):
    pass


def _settings() -> dict[str, Any]:
    try:
        from .settings import load_settings

        value = load_settings(Path.cwd()).get('sandbox')
    except Exception as _ignored:
        _log_ignored(_ignored)
        value = None
    return value if isinstance(value, dict) else {}


def enabled() -> bool:
    raw = os.environ.get('SIGIL_SANDBOX', '').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    return _settings().get('enabled') is True


def _platform() -> str:
    return sys.platform


def _bwrap() -> list[str] | None:
    custom = os.environ.get('SIGIL_SANDBOX_BWRAP', '').strip()
    if custom:
        return [part.strip('"') for part in shlex.split(custom, posix=os.name != 'nt')]
    found = shutil.which('bwrap')
    return [found] if found else None


def active() -> bool:
    """On, on a supported system, with the sandbox tool installed."""
    if not enabled():
        return False
    platform = _platform()
    if platform.startswith('linux'):
        return _bwrap() is not None
    if platform == 'darwin':
        return shutil.which('sandbox-exec') is not None
    return False


def windows_limits() -> dict[str, int] | None:
    """Windows: the sandbox is a Job Object with limits (SIGIL_SANDBOX_MEMORY_MB, default 4096;
    SIGIL_SANDBOX_MAX_PROCESSES, default 64; no clipboard, desktop or system settings access).
    It does not keep writes inside the workspace or block the network (bash_security still
    checks commands), so shell commands still need --allow-shell."""
    if not enabled() or _platform() != 'win32':
        return None

    def _int(name: str, default: int) -> int:
        try:
            return max(0, int(os.environ.get(name, default)))
        except ValueError as _ignored:
            _log_ignored(_ignored)
            return default

    return {'memory_bytes': _int('SIGIL_SANDBOX_MEMORY_MB', 4096) * 1024 * 1024,
            'max_processes': _int('SIGIL_SANDBOX_MAX_PROCESSES', 64),
            # desktop, display settings, exit Windows, global atoms, USER handles, clipboard, system params
            'ui_restrictions': 0x40 | 0x10 | 0x80 | 0x20 | 0x01 | 0x02 | 0x04 | 0x08}


def shell_allowed(flag: bool) -> bool:
    """The --allow-shell flag, or true while the sandbox contains the commands."""
    if flag:
        return True
    return active() and _settings().get('autoAllow', True) is not False


def _write_paths(cwd: Path, extra: tuple[str, ...]) -> list[str]:
    paths = [str(Path(cwd).resolve())] + [str(Path(p).resolve()) for p in extra]
    settings = _settings()
    filesystem = settings.get('filesystem') if isinstance(settings.get('filesystem'), dict) else {}
    for raw in [*(settings.get('writePaths') or ()), *(filesystem.get('allowWrite') or ())]:
        if isinstance(raw, str) and raw.strip():
            paths.append(str(Path(raw).expanduser().resolve()))
    return list(dict.fromkeys(paths))


def _network() -> tuple[bool, list[str]]:
    settings = _settings()
    network = settings.get('network') if isinstance(settings.get('network'), dict) else {}
    listed = [*(settings.get('allowedDomains') or ()), *(network.get('allowedDomains') or ())]
    domains = [str(d).strip().lower() for d in listed if str(d).strip()]
    env = os.environ.get('SIGIL_SANDBOX_DOMAINS', '').strip()
    if env:
        domains += [d.strip().lower() for d in env.split(',') if d.strip()]
    allow_all = settings.get('network') is True or os.environ.get('SIGIL_SANDBOX_NETWORK') == '1'
    return allow_all, list(dict.fromkeys(domains))


def build_argv(bash: str, command: str, cwd: Path, extra_dirs: tuple[str, ...] = (), *,
               platform: str | None = None, proxy_port: int | None = None,
               proxy_socket: str | None = None) -> list[str]:
    platform = platform or _platform()
    writable = _write_paths(cwd, extra_dirs)
    allow_all, domains = _network()
    if platform.startswith('linux'):
        tool = _bwrap()
        if tool is None:
            raise SandboxUnavailable('The sandbox is on but bubblewrap (bwrap) is not installed. '
                                     'Install it (apt install bubblewrap) or set SIGIL_SANDBOX=0.')
        argv = [*tool, '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp']
        for path in writable:
            argv += ['--bind', path, path]
        argv += ['--unshare-all', '--die-with-parent', '--new-session']
        if allow_all:
            argv.append('--share-net')
            argv += ['--chdir', str(Path(cwd).resolve()), '--', bash, '-c', command]
            return argv
        if domains and proxy_port and proxy_socket:
            # Own network namespace (loopback only): the relay inside forwards 127.0.0.1:<port>
            # to the proxy outside through the bound socket, so nothing else can be reached.
            import posixpath

            folder = posixpath.dirname(proxy_socket)  # Linux paths, even when built elsewhere
            relay = str(Path(__file__).resolve().with_name('sandbox_relay.py'))
            argv += ['--bind', folder, folder, '--chdir', str(Path(cwd).resolve()), '--',
                     sys.executable, relay, 'unix:' + proxy_socket, str(proxy_port), '--', bash, '-c', command]
            return argv
        argv += ['--chdir', str(Path(cwd).resolve()), '--', bash, '-c', command]
        return argv
    if platform == 'darwin':
        if shutil.which('sandbox-exec') is None:
            raise SandboxUnavailable('The sandbox is on but sandbox-exec was not found; set SIGIL_SANDBOX=0.')
        rules = ['(version 1)', '(allow default)', '(deny file-write*)',
                 '(allow file-write* (literal "/dev/null") (literal "/dev/tty") (subpath "/private/tmp")'
                 ' (subpath "/private/var/folders")'
                 + ''.join(f' (subpath "{p}")' for p in writable) + ')']
        if not allow_all:
            rules.append('(deny network*)')
            rules.append('(allow network* (remote unix-socket))')
            if proxy_port:
                rules.append(f'(allow network-outbound (remote ip "localhost:{proxy_port}"))')
        return ['sandbox-exec', '-p', ' '.join(rules), bash, '-c', command]
    raise SandboxUnavailable(f'No sandbox on {platform}; set SIGIL_SANDBOX=0.')


def sandbox_argv(bash: str, command: str, cwd: Path, extra_dirs: tuple[str, ...] = ()) -> list[str] | None:
    """The wrapped command, or None when the sandbox is off or this is Windows."""
    if not enabled() or _platform() == 'win32':
        return None
    allow_all, domains = _network()
    port = start_proxy(domains) if domains and not allow_all else None
    socket_path = _PROXY.unix_path if (port and _PROXY is not None) else None
    return build_argv(bash, command, cwd, extra_dirs, proxy_port=port, proxy_socket=socket_path)


def proxy_env(env: dict[str, str]) -> dict[str, str]:
    """Point HTTP(S)_PROXY at the allow-list proxy when one is running."""
    if _PROXY is None:
        return env
    url = f'http://127.0.0.1:{_PROXY.port}'
    return {**env, 'HTTP_PROXY': url, 'HTTPS_PROXY': url, 'http_proxy': url, 'https_proxy': url,
            'ALL_PROXY': url, 'all_proxy': url, 'NO_PROXY': '', 'no_proxy': ''}


# ---- allow-list proxy ----------------------------------------------------

def host_allowed(host: str, domains: list[str]) -> bool:
    host = host.strip().lower().rstrip('.')
    return any(host == d or (d.startswith('*.') and host.endswith(d[1:])) or host.endswith('.' + d)
               for d in domains)


class _Proxy:
    def __init__(self, domains: list[str]) -> None:
        self.domains = domains
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', 0))
        self.server.listen(32)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._serve, args=(self.server,), name='sandbox-proxy', daemon=True).start()
        # Linux: the same proxy on a Unix socket, which the sandbox reaches through its relay.
        self.unix_path: str | None = None
        self.unix_server: socket.socket | None = None
        if _platform().startswith('linux') and hasattr(socket, 'AF_UNIX'):
            import tempfile

            folder = tempfile.mkdtemp(prefix='sigil-proxy-')
            __import__('atexit').register(__import__('shutil').rmtree, folder, True)  # no leftover folder
            path = os.path.join(folder, 'proxy.sock')
            self.unix_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
            self.unix_server.bind(path)
            self.unix_server.listen(32)
            self.unix_path = path
            threading.Thread(target=self._serve, args=(self.unix_server,), name='sandbox-proxy-unix',
                             daemon=True).start()

    def close(self) -> None:
        for listener in (self.server, self.unix_server):
            if listener is not None:
                try:
                    listener.close()
                except OSError as _ignored:
                    _log_ignored(_ignored)
                    pass

    def _serve(self, listener: socket.socket) -> None:
        while True:
            try:
                client, _ = listener.accept()
            except OSError as _ignored:
                _log_ignored(_ignored)
                return
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        try:
            client.settimeout(30)
            head = b''
            while b'\r\n\r\n' not in head and len(head) < 65536:
                chunk = client.recv(4096)
                if not chunk:
                    return
                head += chunk
            line = head.split(b'\r\n', 1)[0].decode('latin-1')
            method, target, _ = (line.split(' ') + ['', ''])[:3]
            if method.upper() == 'CONNECT':
                host, _, port = target.rpartition(':')
                rest = b''
            else:
                from urllib.parse import urlsplit

                parts = urlsplit(target)
                host, port = parts.hostname or '', str(parts.port or 80)
                path = (parts.path or '/') + (f'?{parts.query}' if parts.query else '')
                rest = head.replace(target.encode('latin-1'), path.encode('latin-1'), 1)
            if not host_allowed(host, self.domains):
                client.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
                return
            upstream = socket.create_connection((host, int(port or 443)), timeout=30)
            if method.upper() == 'CONNECT':
                client.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
            else:
                upstream.sendall(rest)
            self._pipe(client, upstream)
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            pass
        finally:
            try:
                client.close()
            except OSError as _ignored:
                _log_ignored(_ignored)
                pass

    @staticmethod
    def _pipe(a: socket.socket, b: socket.socket) -> None:
        try:
            while True:
                ready, _, _ = select.select([a, b], [], [], 60)
                if not ready:
                    return
                for sock in ready:
                    data = sock.recv(65536)
                    if not data:
                        return
                    (b if sock is a else a).sendall(data)
        finally:
            b.close()


_PROXY: _Proxy | None = None
_PROXY_LOCK = threading.Lock()


def start_proxy(domains: list[str]) -> int:
    global _PROXY
    with _PROXY_LOCK:
        if _PROXY is None or _PROXY.domains != domains:
            if _PROXY is not None:
                _PROXY.close()
            _PROXY = _Proxy(domains)
        return _PROXY.port
