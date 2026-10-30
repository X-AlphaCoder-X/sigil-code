import unittest
from unittest.mock import patch

from src.command_graph import CommandGraph, build_command_graph
from src.models import PortingModule


def _module(name: str, source_hint: str) -> PortingModule:
    return PortingModule(name=name, responsibility="stub", source_hint=source_hint)


class CommandGraphTests(unittest.TestCase):
    # -- construction & immutability ------------------------------------------

    def test_empty_graph(self) -> None:
        graph = CommandGraph(builtins=(), plugin_like=(), skill_like=())
        self.assertEqual(graph.builtins, ())
        self.assertEqual(graph.plugin_like, ())
        self.assertEqual(graph.skill_like, ())

    def test_fields_are_tuples(self) -> None:
        b = (_module("b1", "core/b1.py"),)
        p = (_module("p1", "plugin/p1.py"),)
        s = (_module("s1", "skills/s1.py"),)
        graph = CommandGraph(builtins=b, plugin_like=p, skill_like=s)
        self.assertIsInstance(graph.builtins, tuple)
        self.assertIsInstance(graph.plugin_like, tuple)
        self.assertIsInstance(graph.skill_like, tuple)

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        graph = CommandGraph(builtins=(), plugin_like=(), skill_like=())
        with self.assertRaises(AttributeError):
            graph.builtins = ()  # type: ignore[misc]

    # -- flattened -------------------------------------------------------------

    def test_flattened_combines_all_categories(self) -> None:
        b = (_module("b1", "core/b1.py"),)
        p = (_module("p1", "plugin/p1.py"),)
        s = (_module("s1", "skills/s1.py"),)
        graph = CommandGraph(builtins=b, plugin_like=p, skill_like=s)
        self.assertEqual(graph.flattened(), b + p + s)

    def test_flattened_preserves_order(self) -> None:
        b1 = _module("b1", "core/b1.py")
        b2 = _module("b2", "core/b2.py")
        p1 = _module("p1", "plugin/p1.py")
        s1 = _module("s1", "skills/s1.py")
        graph = CommandGraph(builtins=(b1, b2), plugin_like=(p1,), skill_like=(s1,))
        self.assertEqual(graph.flattened(), (b1, b2, p1, s1))

    def test_flattened_of_empty_graph_returns_empty_tuple(self) -> None:
        graph = CommandGraph(builtins=(), plugin_like=(), skill_like=())
        self.assertEqual(graph.flattened(), ())

    def test_flattened_length_is_sum_of_categories(self) -> None:
        b = (_module("b1", "core/b1.py"), _module("b2", "core/b2.py"))
        p = (_module("p1", "plugin/p1.py"),)
        s = ()
        graph = CommandGraph(builtins=b, plugin_like=p, skill_like=s)
        self.assertEqual(len(graph.flattened()), 3)

    # -- as_markdown -----------------------------------------------------------

    def test_as_markdown_includes_header(self) -> None:
        graph = CommandGraph(builtins=(), plugin_like=(), skill_like=())
        md = graph.as_markdown()
        self.assertIn("# Command Graph", md)

    def test_as_markdown_includes_counts(self) -> None:
        b = (_module("b1", "core/b1.py"), _module("b2", "core/b2.py"))
        p = (_module("p1", "plugin/p1.py"),)
        s = (_module("s1", "skills/s1.py"), _module("s2", "skills/s2.py"), _module("s3", "skills/s3.py"))
        graph = CommandGraph(builtins=b, plugin_like=p, skill_like=s)
        md = graph.as_markdown()
        self.assertIn("Builtins: 2", md)
        self.assertIn("Plugin-like commands: 1", md)
        self.assertIn("Skill-like commands: 3", md)

    def test_as_markdown_empty_counts(self) -> None:
        graph = CommandGraph(builtins=(), plugin_like=(), skill_like=())
        md = graph.as_markdown()
        self.assertIn("Builtins: 0", md)
        self.assertIn("Plugin-like commands: 0", md)
        self.assertIn("Skill-like commands: 0", md)

    def test_as_markdown_returns_string(self) -> None:
        graph = CommandGraph(builtins=(), plugin_like=(), skill_like=())
        self.assertIsInstance(graph.as_markdown(), str)


class BuildCommandGraphTests(unittest.TestCase):
    # -- return type -----------------------------------------------------------

    @patch("src.command_graph.get_commands")
    def test_returns_command_graph(self, mock_get: unittest.mock.MagicMock) -> None:
        mock_get.return_value = ()
        result = build_command_graph()
        self.assertIsInstance(result, CommandGraph)

    # -- categorization --------------------------------------------------------

    @patch("src.command_graph.get_commands")
    def test_plugin_source_goes_to_plugin_like(self, mock_get: unittest.mock.MagicMock) -> None:
        p = _module("p1", "plugin/p1.py")
        mock_get.return_value = (p,)
        graph = build_command_graph()
        self.assertIn(p, graph.plugin_like)
        self.assertNotIn(p, graph.builtins)
        self.assertNotIn(p, graph.skill_like)

    @patch("src.command_graph.get_commands")
    def test_skills_source_goes_to_skill_like(self, mock_get: unittest.mock.MagicMock) -> None:
        s = _module("s1", "skills/s1.py")
        mock_get.return_value = (s,)
        graph = build_command_graph()
        self.assertIn(s, graph.skill_like)
        self.assertNotIn(s, graph.builtins)
        self.assertNotIn(s, graph.plugin_like)

    @patch("src.command_graph.get_commands")
    def test_plain_source_goes_to_builtins(self, mock_get: unittest.mock.MagicMock) -> None:
        b = _module("b1", "core/b1.py")
        mock_get.return_value = (b,)
        graph = build_command_graph()
        self.assertIn(b, graph.builtins)
        self.assertNotIn(b, graph.plugin_like)
        self.assertNotIn(b, graph.skill_like)

    @patch("src.command_graph.get_commands")
    def test_case_insensitive_plugin_match(self, mock_get: unittest.mock.MagicMock) -> None:
        p = _module("p1", "Plugin/p1.py")
        mock_get.return_value = (p,)
        graph = build_command_graph()
        self.assertIn(p, graph.plugin_like)

    @patch("src.command_graph.get_commands")
    def test_case_insensitive_skills_match(self, mock_get: unittest.mock.MagicMock) -> None:
        s = _module("s1", "Skills/s1.py")
        mock_get.return_value = (s,)
        graph = build_command_graph()
        self.assertIn(s, graph.skill_like)

    @patch("src.command_graph.get_commands")
    def test_mixed_commands_are_sorted_correctly(self, mock_get: unittest.mock.MagicMock) -> None:
        b = _module("b1", "core/b1.py")
        p = _module("p1", "plugin/p1.py")
        s = _module("s1", "skills/s1.py")
        mock_get.return_value = (b, p, s)
        graph = build_command_graph()
        self.assertEqual(graph.builtins, (b,))
        self.assertEqual(graph.plugin_like, (p,))
        self.assertEqual(graph.skill_like, (s,))

    @patch("src.command_graph.get_commands")
    def test_empty_commands_yields_empty_graph(self, mock_get: unittest.mock.MagicMock) -> None:
        mock_get.return_value = ()
        graph = build_command_graph()
        self.assertEqual(graph.builtins, ())
        self.assertEqual(graph.plugin_like, ())
        self.assertEqual(graph.skill_like, ())

    @patch("src.command_graph.get_commands")
    def test_all_builtins(self, mock_get: unittest.mock.MagicMock) -> None:
        b1 = _module("b1", "core/b1.py")
        b2 = _module("b2", "commands/b2.py")
        mock_get.return_value = (b1, b2)
        graph = build_command_graph()
        self.assertEqual(len(graph.builtins), 2)
        self.assertEqual(graph.plugin_like, ())
        self.assertEqual(graph.skill_like, ())

    @patch("src.command_graph.get_commands")
    def test_builtins_plugin_skill_are_tuples_of_porting_module(self, mock_get: unittest.mock.MagicMock) -> None:
        b = _module("b1", "core/b1.py")
        p = _module("p1", "plugin/p1.py")
        s = _module("s1", "skills/s1.py")
        mock_get.return_value = (b, p, s)
        graph = build_command_graph()
        for category in (graph.builtins, graph.plugin_like, graph.skill_like):
            self.assertIsInstance(category, tuple)
            for item in category:
                self.assertIsInstance(item, PortingModule)

    @patch("src.command_graph.get_commands")
    def test_flattened_matches_original_commands(self, mock_get: unittest.mock.MagicMock) -> None:
        modules = (
            _module("b1", "core/b1.py"),
            _module("p1", "plugin/p1.py"),
            _module("s1", "skills/s1.py"),
        )
        mock_get.return_value = modules
        graph = build_command_graph()
        self.assertEqual(set(graph.flattened()), set(modules))


if __name__ == "__main__":
    unittest.main()
