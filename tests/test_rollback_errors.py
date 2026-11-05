"""rollback_workspace reports git failures instead of claiming success."""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import agent_tools


def _context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(root=root, permissions=SimpleNamespace(allow_destructive_shell_commands=True))


class RollbackErrorTests(unittest.TestCase):
    def test_outside_a_git_repo_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent_tools, '_ensure_write_allowed'):
            with self.assertRaises(agent_tools.ToolExecutionError) as caught:
                agent_tools._rollback_workspace_tool({}, _context(Path(tmp)))
        self.assertIn('git reset --hard HEAD exited', str(caught.exception))

    def test_resets_and_cleans_a_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(agent_tools, '_ensure_write_allowed'):
            root = Path(tmp)
            git = ['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com']
            subprocess.run([*git, 'init', '-q'], cwd=root, check=True)
            (root / 'a.txt').write_text('1', encoding='utf-8')
            subprocess.run([*git, 'add', '.'], cwd=root, check=True)
            subprocess.run([*git, 'commit', '-qm', 'base'], cwd=root, check=True)
            (root / 'a.txt').write_text('changed', encoding='utf-8')
            (root / 'junk.txt').write_text('x', encoding='utf-8')
            result = agent_tools._rollback_workspace_tool({}, _context(root))
            self.assertIn('rolled back', result)
            self.assertEqual((root / 'a.txt').read_text(encoding='utf-8'), '1')
            self.assertFalse((root / 'junk.txt').exists())


if __name__ == '__main__':
    unittest.main()
