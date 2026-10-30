"""The core needs only prompt_toolkit and rich; tree-sitter and PyYAML are extras."""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import patch

from src import shredder

ROOT = Path(__file__).resolve().parent.parent


class OptionalDependencyTests(unittest.TestCase):
    def test_two_core_dependencies(self) -> None:
        text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8')
        block = re.search(r'^dependencies = \[(.*?)\]', text, re.S | re.M).group(1)
        deps = sorted(re.findall(r'"([A-Za-z0-9_\-]+)', block))
        self.assertEqual([d.lower() for d in deps], ['prompt_toolkit', 'rich'])
        self.assertIn('[project.optional-dependencies]', text)
        self.assertRegex(text, r'(?m)^ast = ')
        self.assertRegex(text, r'(?m)^yaml = ')

    def test_shredder_works_without_tree_sitter(self) -> None:
        source = 'def hello():\n    return 1\n' + '\n'.join(f'x{i} = {i}' for i in range(200))
        with patch.object(shredder, 'PY_LANGUAGE', None), patch.object(shredder, 'CPP_LANGUAGE', None):
            outline = shredder.get_file_outline(source, '.py')
            cpp = shredder.get_file_outline('int main() { return 0; }\n', '.cpp')
        self.assertIsInstance(outline, str)
        self.assertTrue(outline)
        self.assertIsInstance(cpp, str)


if __name__ == '__main__':
    unittest.main()
