"""Per-prompt file checkpoints for /rewind (and Esc Esc).

Every user prompt starts a checkpoint. Before a file tool (write_file,
edit_file, notebook_edit) changes a file for the first time within that
checkpoint, the file's current contents are saved. Rewinding to a prompt
restores every file changed since then to its state just before that prompt,
and the conversation can be cut back to just before the prompt as well.

Before a bash command, files the edit tools already touched are saved too (record_tracked),
so rewind also undoes what bash did to them; files only bash touches are not tracked.

Layout: <session_dir>/checkpoints/<session_id>/
    index.json            ordered list of checkpoint ids
    <cid>/manifest.json   {relative_path: {"existed": bool, "backup": name}}
    <cid>/<n>.bak         saved file contents
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from .settings import write_text_atomic  # replaces a file in one step: readers never see half of it
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

FILE_TOOLS = frozenset({'write_file', 'edit_file', 'notebook_edit'})


@dataclass(frozen=True)
class RewindResult:
    restored: tuple[str, ...]
    deleted: tuple[str, ...]


class CheckpointStore:
    def __init__(self, root: Path, workspace: Path) -> None:
        self.root = root
        self.workspace = workspace.resolve()
        self.current: str | None = None

    # ---- index -------------------------------------------------------------
    def _index_path(self) -> Path:
        return self.root / 'index.json'

    def ids(self) -> list[str]:
        try:
            data = json.loads(self._index_path().read_text(encoding='utf-8'))
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            return []
        return [str(x) for x in data] if isinstance(data, list) else []

    def _save_ids(self, ids: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self._index_path(), json.dumps(ids), encoding='utf-8')

    def _manifest(self, cid: str) -> dict[str, dict]:
        try:
            data = json.loads((self.root / cid / 'manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as _ignored:
            _log_ignored(_ignored)
            return {}
        return data if isinstance(data, dict) else {}

    # ---- recording ---------------------------------------------------------
    def begin(self) -> str:
        """Start a checkpoint for a new prompt. Nothing is written until a file changes."""
        stamp = int(time.time() * 1000)
        taken = set(self.ids())
        if self.current:
            taken.add(self.current)
        while f'cp_{stamp}' in taken:  # two prompts in the same millisecond still differ
            stamp += 1
        self.current = f'cp_{stamp}'
        return self.current

    def record_before(self, path: str | Path) -> None:
        """Save a file's contents before its first change in the current checkpoint."""
        if self.current is None:
            return
        target = Path(path)
        if not target.is_absolute():
            target = self.workspace / target
        try:
            target = target.resolve()
        except OSError as _ignored:
            _log_ignored(_ignored)
            return
        try:
            rel = target.relative_to(self.workspace).as_posix()
        except ValueError as _ignored:
            # Outside the workspace (an added folder, or anywhere when other
            # folders are allowed): keyed by absolute path. rewind() joins it
            # with the workspace, and joining an absolute path gives it back.
            _log_ignored(_ignored)
            rel = target.as_posix()
        cid = self.current
        manifest = self._manifest(cid)
        if rel in manifest:
            return
        folder = self.root / cid
        folder.mkdir(parents=True, exist_ok=True)
        entry: dict[str, object] = {'existed': target.is_file()}
        if target.is_file():
            backup = f'{len(manifest)}.bak'
            shutil.copy2(target, folder / backup)
            entry['backup'] = backup
        manifest[rel] = entry
        write_text_atomic((folder / 'manifest.json'), json.dumps(manifest, indent=2), encoding='utf-8')
        ids = self.ids()
        if cid not in ids:
            ids.append(cid)
            self._save_ids(ids)

    def tracked_paths(self) -> list[str]:
        """Every file the edit tools changed in this session (for bash snapshots)."""
        seen: list[str] = []
        for cid in self.ids():
            for rel in self._manifest(cid):
                if rel not in seen:
                    seen.append(rel)
        return seen

    def record_tracked(self) -> None:
        """Before a bash command: save files the edit tools touched, so rewind also
        undoes what bash does to them (files only bash touches stay untracked)."""
        for rel in self.tracked_paths():
            self.record_before(rel)

    # ---- inspection / rewind -----------------------------------------------
    def changed_files_since(self, cid: str) -> list[str]:
        ids = self.ids()
        later = self._from(ids, cid)
        seen: list[str] = []
        for later_id in later:
            for rel in self._manifest(later_id):
                if rel not in seen:
                    seen.append(rel)
        return seen

    @staticmethod
    def _from(ids: list[str], cid: str) -> list[str]:
        # Checkpoint ids are time ordered, so a prompt without file changes
        # (no entry in the index) still rewinds everything after it.
        return [x for x in ids if x >= cid]

    def rewind(self, cid: str) -> RewindResult:
        ids = self.ids()
        later = self._from(ids, cid)
        original: dict[str, tuple[str, dict]] = {}
        for later_id in later:  # oldest first: the earliest backup is the real "before"
            for rel, entry in self._manifest(later_id).items():
                original.setdefault(rel, (later_id, entry))
        restored: list[str] = []
        deleted: list[str] = []
        for rel, (owner, entry) in original.items():
            target = self.workspace / rel
            if entry.get('existed') and entry.get('backup'):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.root / owner / str(entry['backup']), target)
                restored.append(rel)
            elif target.exists():
                target.unlink()
                deleted.append(rel)
        for later_id in later:
            shutil.rmtree(self.root / later_id, ignore_errors=True)
        self._save_ids([x for x in ids if x not in later])
        self.current = None
        return RewindResult(tuple(restored), tuple(deleted))
