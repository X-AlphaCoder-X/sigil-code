"""Images for vision models: read_file on an image, --image, and Alt+V paste.

Images are sent as OpenAI ``image_url`` parts with a base64 data URL, in a
user message (most OpenAI-compatible servers do not accept images inside tool
results). ``SIGIL_VISION=0`` turns images off; ``SIGIL_MAX_IMAGE_BYTES``
(default 5 MB) caps each file.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

IMAGE_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
}


def vision_enabled() -> bool:
    return os.environ.get('SIGIL_VISION', '1').strip().lower() not in {'0', 'false', 'no', 'off'}


def max_image_bytes() -> int:
    from . import prompt_constants

    try:
        return int(os.environ.get('SIGIL_MAX_IMAGE_BYTES', '') or prompt_constants.API_IMAGE_MAX_BASE64_SIZE)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        return prompt_constants.API_IMAGE_MAX_BASE64_SIZE


def is_image(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_TYPES


def image_data_url(path: Path) -> str:
    """A data URL for an image file; raises ValueError when it cannot be sent."""
    path = Path(path)
    mime = IMAGE_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f'{path.name} is not a png, jpg, gif or webp image')
    size = path.stat().st_size
    encoded = 4 * ((size + 2) // 3)  # the limit is on the base64 text that is sent
    if encoded > max_image_bytes():
        raise ValueError(f'{path.name} is {size:,} bytes ({encoded:,} as base64); the limit is '
                         f'{max_image_bytes():,} base64 bytes (SIGIL_MAX_IMAGE_BYTES)')
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def _remove_old_pastes(folder: Path, max_age: float = 86400.0) -> int:
    """Pasted screenshots older than a day are removed (they were only sent to the model)."""
    removed = 0
    cutoff = time.time() - max_age
    for old in folder.glob('paste_*.png'):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
                removed += 1
        except OSError as exc:
            from .debug_log import log_ignored_error

            log_ignored_error(exc)
    return removed


def grab_clipboard_image(folder: Path | None = None) -> Path | None:
    """Save an image from the clipboard to a PNG file, or return None."""
    folder = Path(folder or tempfile.gettempdir()) / 'sigil-pasted-images'
    folder.mkdir(parents=True, exist_ok=True)
    _remove_old_pastes(folder)
    target = folder / f'paste_{int(time.time() * 1000)}.png'
    if sys.platform == 'win32':
        script = (
            'Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; '
            '$img = [System.Windows.Forms.Clipboard]::GetImage(); '
            f"if ($img) {{ $img.Save('{target}', [System.Drawing.Imaging.ImageFormat]::Png); 'ok' }}"
        )
        commands = [['powershell', '-NoProfile', '-STA', '-Command', script]]
    elif sys.platform == 'darwin':
        commands = [['pngpaste', str(target)]]
    else:
        commands = [['wl-paste', '--type', 'image/png'], ['xclip', '-selection', 'clipboard', '-t', 'image/png', '-o']]
    for command in commands:
        try:
            completed = subprocess.run(command, capture_output=True, timeout=10,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except (OSError, subprocess.TimeoutExpired) as _ignored:
            _log_ignored(_ignored)
            continue
        if command[0] in ('wl-paste', 'xclip'):
            if completed.returncode == 0 and completed.stdout:
                target.write_bytes(completed.stdout)
                return target
            continue
        if completed.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            return target
    return None
