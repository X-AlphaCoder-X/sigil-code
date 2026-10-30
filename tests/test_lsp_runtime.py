from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.lsp_runtime import LSPRuntime


SAMPLE_SOURCE = '''def helper(value):
    """Double a numeric value."""
    return value * 2


def orchestrate(item):
    return helper(item)


class Greeter:
    def greet(self, name):
        return helper(len(name))
'''


class LSPRuntimeTests(unittest.TestCase):
    def test_runtime_renders_symbols_definitions_references_and_hover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'sample.py').write_text(SAMPLE_SOURCE, encoding='utf-8')
            runtime = LSPRuntime.from_workspace(workspace)

            summary = runtime.render_summary()
            symbols = runtime.render_document_symbols('sample.py')
            workspace_symbols = runtime.render_workspace_symbols('helper')
            definition = runtime.render_definition('sample.py', 7, 12)
            references = runtime.render_references('sample.py', 7, 12)
            hover = runtime.render_hover('sample.py', 1, 5)

        self.assertIn('Indexed candidate files: 1', summary)
        self.assertIn('# LSP Document Symbols', symbols)
        self.assertIn('function helper', symbols)
        self.assertIn('function orchestrate', symbols)
        self.assertIn('class Greeter', symbols)
        self.assertIn('# LSP Workspace Symbols', workspace_symbols)
        self.assertIn('helper', workspace_symbols)
        self.assertIn('# LSP Definition', definition)
        self.assertIn('function helper', definition)
        self.assertIn('# LSP References', references)
        self.assertIn('helper(item)', references)
        self.assertIn('# LSP Hover', hover)
        self.assertIn('signature=helper(value)', hover)
        self.assertIn('Double a numeric value.', hover)

    def test_runtime_renders_call_hierarchy_and_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            (workspace / 'sample.py').write_text(SAMPLE_SOURCE, encoding='utf-8')
            (workspace / 'broken.py').write_text('def broken(:\n    pass\n', encoding='utf-8')
            runtime = LSPRuntime.from_workspace(workspace)

            hierarchy = runtime.render_prepare_call_hierarchy('sample.py', 6, 12)
            incoming = runtime.render_incoming_calls('sample.py', 1, 5)
            outgoing = runtime.render_outgoing_calls('sample.py', 6, 12)
            diagnostics = runtime.render_diagnostics('broken.py')

        self.assertIn('# LSP Call Hierarchy', hierarchy)
        self.assertIn('symbol=orchestrate', hierarchy)
        self.assertIn('# LSP Incoming Calls', incoming)
        self.assertIn('orchestrate', incoming)
        self.assertIn('greet', incoming)
        self.assertIn('# LSP Outgoing Calls', outgoing)
        self.assertIn('helper', outgoing)
        self.assertIn('# LSP Diagnostics', diagnostics)
        self.assertIn('syntax-error', diagnostics)
