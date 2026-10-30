from __future__ import annotations

import unittest

from src.permissions import ToolPermissionContext


class TestToolPermissionContext(unittest.TestCase):
    # 1. Empty context blocks nothing
    def test_empty_context_blocks_nothing(self) -> None:
        ctx = ToolPermissionContext()
        self.assertFalse(ctx.blocks("anything"))
        self.assertFalse(ctx.blocks(""))

    # 2. Exact name blocking (case insensitive)
    def test_exact_name_blocking(self) -> None:
        ctx = ToolPermissionContext.from_iterables(deny_names=["dangerous_tool"])
        self.assertTrue(ctx.blocks("dangerous_tool"))
        self.assertTrue(ctx.blocks("Dangerous_Tool"))
        self.assertTrue(ctx.blocks("DANGEROUS_TOOL"))
        self.assertFalse(ctx.blocks("safe_tool"))

    # 3. Prefix blocking (case insensitive)
    def test_prefix_blocking(self) -> None:
        ctx = ToolPermissionContext.from_iterables(deny_prefixes=["admin_"])
        self.assertTrue(ctx.blocks("admin_delete"))
        self.assertTrue(ctx.blocks("Admin_Delete"))
        self.assertTrue(ctx.blocks("ADMIN_CREATE"))
        self.assertFalse(ctx.blocks("user_admin"))

    # 4. Combined name + prefix blocking
    def test_combined_name_and_prefix_blocking(self) -> None:
        ctx = ToolPermissionContext.from_iterables(
            deny_names=["rm"],
            deny_prefixes=["sudo_"],
        )
        self.assertTrue(ctx.blocks("rm"))
        self.assertTrue(ctx.blocks("sudo_restart"))
        self.assertFalse(ctx.blocks("ls"))

    # 5. Non-matching names are allowed
    def test_non_matching_names_allowed(self) -> None:
        ctx = ToolPermissionContext.from_iterables(
            deny_names=["blocked"],
            deny_prefixes=["bad_"],
        )
        self.assertFalse(ctx.blocks("allowed"))
        self.assertFalse(ctx.blocks("good_tool"))
        self.assertFalse(ctx.blocks("not_bad"))

    # 6. from_iterables with None args
    def test_from_iterables_none_args(self) -> None:
        ctx = ToolPermissionContext.from_iterables(deny_names=None, deny_prefixes=None)
        self.assertEqual(ctx.deny_names, frozenset())
        self.assertEqual(ctx.deny_prefixes, ())
        self.assertFalse(ctx.blocks("anything"))

    def test_from_iterables_default_args(self) -> None:
        ctx = ToolPermissionContext.from_iterables()
        self.assertEqual(ctx.deny_names, frozenset())
        self.assertEqual(ctx.deny_prefixes, ())

    # 7. from_iterables normalizes to lowercase
    def test_from_iterables_normalizes_to_lowercase(self) -> None:
        ctx = ToolPermissionContext.from_iterables(
            deny_names=["FooBar"],
            deny_prefixes=["PFX_"],
        )
        self.assertIn("foobar", ctx.deny_names)
        self.assertNotIn("FooBar", ctx.deny_names)
        self.assertEqual(ctx.deny_prefixes, ("pfx_",))
        self.assertTrue(ctx.blocks("FOOBAR"))
        self.assertTrue(ctx.blocks("pfx_something"))

    # 8. Multiple deny_names
    def test_multiple_deny_names(self) -> None:
        ctx = ToolPermissionContext.from_iterables(
            deny_names=["tool_a", "tool_b", "tool_c"],
        )
        self.assertTrue(ctx.blocks("tool_a"))
        self.assertTrue(ctx.blocks("tool_b"))
        self.assertTrue(ctx.blocks("tool_c"))
        self.assertFalse(ctx.blocks("tool_d"))


if __name__ == "__main__":
    unittest.main()
