"""Hooks are read again on every call: an edit to a settings file applies without a restart,
a half-written file keeps the last good hooks, and two sessions on one folder do not disturb each other."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from src.command_hooks import CommandHookRuntime, _parse_hooks


def _settings(root: Path, hooks: dict, name: str = 'settings.json', extra: dict | None = None) -> Path:
    (root / '.sigil').mkdir(exist_ok=True)
    path = root / '.sigil' / name
    path.write_text(json.dumps({'hooks': hooks, **(extra or {})}), encoding='utf-8')
    return path


def _stop_hook(command: str = 'echo hi') -> dict:
    return {'Stop': [{'hooks': [{'type': 'command', 'command': command}]}]}


def _pre_hook(command: str = 'echo pre') -> dict:
    return {'PreToolUse': [{'matcher': 'Bash', 'hooks': [{'type': 'command', 'command': command}]}]}


class HookReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        patcher = mock.patch.dict(os.environ, {'SIGIL_DISABLE_USER_HOOKS': '1'})
        patcher.start()
        os.environ.pop('SIGIL_HOOKS_RELOAD', None)
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def test_added_hook_applies_on_next_call(self) -> None:
        _settings(self.root, _stop_hook())
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertTrue(runtime.has('Stop'))
        self.assertFalse(runtime.has('PreToolUse'))
        _settings(self.root, {**_stop_hook(), **_pre_hook()})
        self.assertTrue(runtime.has('PreToolUse'))  # no restart, no new runtime

    def test_removed_hook_and_new_local_file_and_disable_all(self) -> None:
        settings = _settings(self.root, {**_stop_hook(), **_pre_hook()})
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertTrue(runtime.has('PreToolUse'))
        _settings(self.root, _stop_hook())
        self.assertFalse(runtime.has('PreToolUse'))
        _settings(self.root, {'PostToolUse': [{'hooks': [{'command': 'echo post'}]}]}, name='settings.local.json')
        self.assertTrue(runtime.has('PostToolUse'))  # a file created after start is seen
        settings.write_text(json.dumps({'disableAllHooks': True, 'hooks': _stop_hook()}), encoding='utf-8')
        self.assertFalse(runtime.has('Stop'))
        self.assertEqual(runtime.hooks, ())

    def test_run_uses_the_new_hooks(self) -> None:
        _settings(self.root, {})
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertEqual(runtime.run('Stop', {}).ran, 0)
        _settings(self.root, _stop_hook('exit 0'))
        self.assertEqual(runtime.run('Stop', {}).ran, 1)

    def test_half_written_file_keeps_last_good_hooks(self) -> None:
        settings = _settings(self.root, _stop_hook())
        runtime = CommandHookRuntime.from_workspace(self.root)
        self.assertTrue(runtime.has('Stop'))
        settings.write_text('{"hooks": {"Stop": [', encoding='utf-8')  # an editor mid-save
        with mock.patch('sys.stderr') as stderr:
            self.assertTrue(runtime.has('Stop'))
            self.assertTrue(runtime.has('Stop'))
        warnings = [c.args[0] for c in stderr.write.call_args_list if 'could not be read' in c.args[0]]
        self.assertEqual(len(warnings), 1)  # reported once, not on every call
        _settings(self.root, _pre_hook())  # the save finishes
        self.assertTrue(runtime.has('PreToolUse'))
        self.assertFalse(runtime.has('Stop'))

    def test_same_size_rewrite_within_one_tick_is_seen(self) -> None:
        settings = _settings(self.root, _stop_hook('echo aa'))
        runtime = CommandHookRuntime.from_workspace(self.root)
        stat = settings.stat()
        settings.write_text(json.dumps({'hooks': {'Stop': [{'hooks': [{'type': 'command', 'command': 'echo bb'}]}]}}),
                            encoding='utf-8')
        os.utime(settings, ns=(stat.st_atime_ns, stat.st_mtime_ns))  # same size, same mtime
        self.assertEqual(settings.stat().st_size, stat.st_size)
        runtime.has('Stop')
        self.assertEqual(runtime.hooks[0].command, 'echo bb')

    def test_reload_can_be_turned_off(self) -> None:
        _settings(self.root, _stop_hook())
        runtime = CommandHookRuntime.from_workspace(self.root)
        with mock.patch.dict(os.environ, {'SIGIL_HOOKS_RELOAD': '0'}):
            _settings(self.root, _pre_hook())
            self.assertFalse(runtime.has('PreToolUse'))
        self.assertTrue(runtime.has('PreToolUse'))

    def test_hooks_built_in_code_are_never_reloaded(self) -> None:
        errors: list[str] = []
        runtime = CommandHookRuntime(cwd=self.root, hooks=tuple(_parse_hooks(_stop_hook(), 'test', errors)))
        _settings(self.root, _pre_hook())
        self.assertTrue(runtime.has('Stop'))
        self.assertFalse(runtime.has('PreToolUse'))

    def test_agent_hooks_survive_a_reload(self) -> None:
        _settings(self.root, _stop_hook())
        base = CommandHookRuntime.from_workspace(self.root)
        errors: list[str] = []
        own = tuple(_parse_hooks({'SubagentStop': [{'hooks': [{'command': 'echo own'}]}]}, 'agent:x', errors))
        child = replace(base, hooks=(*base.hooks, *own), fixed_hooks=(*base.fixed_hooks, *own))
        _settings(self.root, _pre_hook())
        self.assertTrue(child.has('PreToolUse'))
        self.assertTrue(child.has('SubagentStop'))
        self.assertFalse(child.has('Stop'))

    def test_two_sessions_on_one_folder(self) -> None:
        """Two runtimes (two sessions) see the same edit, never write the settings file, and threads
        reading while a third rewrites it only ever see a whole hook set."""
        settings = _settings(self.root, _stop_hook())
        first = CommandHookRuntime.from_workspace(self.root)
        second = CommandHookRuntime.from_workspace(self.root)
        before = settings.read_bytes()
        first.has('Stop')
        second.has('Stop')
        self.assertEqual(settings.read_bytes(), before)  # reading never writes
        _settings(self.root, _pre_hook())
        self.assertTrue(first.has('PreToolUse'))
        self.assertTrue(second.has('PreToolUse'))

        seen: list[tuple[str, ...]] = []
        failures: list[BaseException] = []
        stop = threading.Event()

        def reader(runtime: CommandHookRuntime) -> None:
            try:
                while not stop.is_set():
                    runtime.has('Stop')
                    seen.append(tuple(sorted({h.event for h in runtime.hooks})))
            except BaseException as exc:  # recorded and asserted below
                failures.append(exc)

        threads = [threading.Thread(target=reader, args=(r,)) for r in (first, second, first)]
        for thread in threads:
            thread.start()
        with mock.patch('sys.stderr'):
            for index in range(40):
                _settings(self.root, _stop_hook() if index % 2 else _pre_hook())
        stop.set()
        for thread in threads:
            thread.join(10)
        self.assertEqual(failures, [])
        self.assertTrue(set(seen) <= {('Stop',), ('PreToolUse',)}, set(seen))


if __name__ == '__main__':
    unittest.main()
