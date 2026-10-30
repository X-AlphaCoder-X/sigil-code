"""Relay that runs inside the Linux sandbox (standalone: no package imports).

The sandboxed command has its own network namespace with only a loopback device, so it cannot
reach any host. This relay listens on 127.0.0.1:<port> inside that namespace and forwards every
connection to the allow-list proxy outside, through a Unix socket bound into the sandbox (or a
TCP address, for tests). Then it runs the command and exits with its exit code.

    python sandbox_relay.py unix:/path/proxy.sock 18080 -- bash -c "pip install x"
    python sandbox_relay.py tcp:127.0.0.1:9000 18080 -- python -c "..."
"""
from __future__ import annotations

import select
import socket
import subprocess
import sys
import threading


def _connect(target: str) -> socket.socket:
    if target.startswith('unix:'):
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined]
        conn.connect(target[len('unix:'):])
        return conn
    host, _, port = target[len('tcp:'):].rpartition(':')
    return socket.create_connection((host, int(port)), timeout=30)


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            ready, _, _ = select.select([a, b], [], [], 120)
            if not ready:
                return
            for sock in ready:
                data = sock.recv(65536)
                if not data:
                    return
                (b if sock is a else a).sendall(data)
    except OSError as _ignored:
        sys.stderr.write(f'relay: {_ignored!r}\n')
        return
    finally:
        for sock in (a, b):
            try:
                sock.close()
            except OSError as _ignored:
                sys.stderr.write(f'relay: {_ignored!r}\n')
                pass


def _serve(listener: socket.socket, target: str) -> None:
    while True:
        try:
            client, _ = listener.accept()
        except OSError as _ignored:
            sys.stderr.write(f'relay: {_ignored!r}\n')
            return
        try:
            upstream = _connect(target)
        except OSError as _ignored:
            sys.stderr.write(f'relay: {_ignored!r}\n')
            client.close()
            continue
        threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != '--':
        print('usage: sandbox_relay.py unix:<socket>|tcp:<host>:<port> <listen-port> -- <command...>', file=sys.stderr)
        return 2
    target, port, command = argv[0], int(argv[1]), argv[3:]
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('127.0.0.1', port))
    listener.listen(64)
    threading.Thread(target=_serve, args=(listener, target), daemon=True).start()
    return subprocess.run(command).returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
