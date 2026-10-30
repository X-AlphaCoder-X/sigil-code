"""Tests for prompt_constants module.

Validates that the shared constants are present,
correctly typed, and that helper functions behave as expected.
"""

from __future__ import annotations

import os
import platform
from unittest.mock import patch

import pytest

from src.prompt_constants import (
    # Product metadata
    PRODUCT_URL,
    # API limits
    API_IMAGE_MAX_BASE64_SIZE,
    IMAGE_TARGET_RAW_SIZE,
    IMAGE_MAX_WIDTH,
    IMAGE_MAX_HEIGHT,
    PDF_TARGET_RAW_SIZE,
    API_PDF_MAX_PAGES,
    PDF_EXTRACT_SIZE_THRESHOLD,
    PDF_MAX_EXTRACT_SIZE,
    PDF_MAX_PAGES_PER_READ,
    PDF_AT_MENTION_INLINE_THRESHOLD,
    API_MAX_MEDIA_PER_REQUEST,
    # Tool limits
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    MAX_TOOL_RESULT_TOKENS,
    BYTES_PER_TOKEN,
    MAX_TOOL_RESULT_BYTES,
    MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
    TOOL_SUMMARY_MAX_LENGTH,
    # Spinner verbs
    SPINNER_VERBS,
    # Turn completion verbs
    TURN_COMPLETION_VERBS,
    # Figures
    BLACK_CIRCLE,
    BULLET_OPERATOR,
    TEARDROP_ASTERISK,
    UP_ARROW,
    DOWN_ARROW,
    LIGHTNING_BOLT,
    EFFORT_LOW,
    EFFORT_MEDIUM,
    EFFORT_HIGH,
    EFFORT_MAX,
    PLAY_ICON,
    PAUSE_ICON,
    REFRESH_ARROW,
    CHANNEL_ARROW,
    INJECTED_ARROW,
    FORK_GLYPH,
    DIAMOND_OPEN,
    DIAMOND_FILLED,
    REFERENCE_MARK,
    FLAG_ICON,
    BLOCKQUOTE_BAR,
    HEAVY_HORIZONTAL,
    BRIDGE_SPINNER_FRAMES,
    BRIDGE_READY_INDICATOR,
    BRIDGE_FAILED_INDICATOR,
    # XML tags
    COMMAND_NAME_TAG,
    COMMAND_MESSAGE_TAG,
    COMMAND_ARGS_TAG,
    BASH_INPUT_TAG,
    BASH_STDOUT_TAG,
    BASH_STDERR_TAG,
    LOCAL_COMMAND_STDOUT_TAG,
    LOCAL_COMMAND_STDERR_TAG,
    LOCAL_COMMAND_CAVEAT_TAG,
    TERMINAL_OUTPUT_TAGS,
    TICK_TAG,
    TASK_NOTIFICATION_TAG,
    TASK_ID_TAG,
    TOOL_USE_ID_TAG,
    TASK_TYPE_TAG,
    OUTPUT_FILE_TAG,
    STATUS_TAG,
    SUMMARY_TAG,
    REASON_TAG,
    WORKTREE_TAG,
    WORKTREE_PATH_TAG,
    WORKTREE_BRANCH_TAG,
    ULTRAPLAN_TAG,
    REMOTE_REVIEW_TAG,
    REMOTE_REVIEW_PROGRESS_TAG,
    TEAMMATE_MESSAGE_TAG,
    CHANNEL_MESSAGE_TAG,
    CHANNEL_TAG,
    CROSS_SESSION_MESSAGE_TAG,
    FORK_BOILERPLATE_TAG,
    FORK_DIRECTIVE_PREFIX,
    COMMON_HELP_ARGS,
    COMMON_INFO_ARGS,
    # Messages
    NO_CONTENT_MESSAGE,
    # Date utilities
    get_local_iso_date,
    get_session_start_date,
    reset_session_start_date,
    get_local_month_year,
    # System prompt section caching
    SystemPromptSection,
    system_prompt_section,
    dangerous_uncached_system_prompt_section,
    resolve_system_prompt_sections,
    clear_system_prompt_sections,
    # Output styles
    DEFAULT_OUTPUT_STYLE_NAME,
    OutputStyleConfig,
    OUTPUT_STYLE_CONFIGS,
    # Knowledge cutoff
    FRONTIER_MODEL_NAME,
    get_knowledge_cutoff,
    CLAUDE_MODEL_IDS,
    # Prompt sections
    HOOKS_SECTION,
    SYSTEM_REMINDERS_SECTION,
    SUMMARIZE_TOOL_RESULTS_SECTION,
    DEFAULT_AGENT_PROMPT,
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    get_language_section,
    get_output_style_section,
    get_scratchpad_instructions,
    # Error IDs
    E_TOOL_USE_SUMMARY_GENERATION_FAILED,
    # UI helpers
    EFFORT_FIGURES,
    random_completion_verb,
    random_spinner_verb,
)


# =========================================================================
# Product metadata
# =========================================================================

class TestProductMetadata:
    def test_product_url(self):
        # the project's own URL (pyproject.toml)
        import re
        from pathlib import Path

        pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        assert PRODUCT_URL == re.search(r'^Repository\s*=\s*"([^"]+)"', pyproject, re.M).group(1)


# =========================================================================
# API limits
# =========================================================================

class TestAPILimits:
    def test_image_base64_size(self):
        assert API_IMAGE_MAX_BASE64_SIZE == 5 * 1024 * 1024

    def test_image_target_raw_size(self):
        assert IMAGE_TARGET_RAW_SIZE == (API_IMAGE_MAX_BASE64_SIZE * 3) // 4

    def test_image_dimensions(self):
        assert IMAGE_MAX_WIDTH == 2000
        assert IMAGE_MAX_HEIGHT == 2000

    def test_pdf_target_raw_size(self):
        assert PDF_TARGET_RAW_SIZE == 20 * 1024 * 1024

    def test_pdf_max_pages(self):
        assert API_PDF_MAX_PAGES == 100

    def test_pdf_extract_threshold(self):
        assert PDF_EXTRACT_SIZE_THRESHOLD == 3 * 1024 * 1024

    def test_pdf_max_extract_size(self):
        assert PDF_MAX_EXTRACT_SIZE == 100 * 1024 * 1024

    def test_pdf_pages_per_read(self):
        assert PDF_MAX_PAGES_PER_READ == 20

    def test_pdf_inline_threshold(self):
        assert PDF_AT_MENTION_INLINE_THRESHOLD == 10

    def test_media_per_request(self):
        assert API_MAX_MEDIA_PER_REQUEST == 100


# =========================================================================
# Tool limits
# =========================================================================

class TestToolLimits:
    def test_default_max_result_size(self):
        assert DEFAULT_MAX_RESULT_SIZE_CHARS == 50_000

    def test_max_tool_result_tokens(self):
        assert MAX_TOOL_RESULT_TOKENS == 100_000

    def test_bytes_per_token(self):
        assert BYTES_PER_TOKEN == 4

    def test_max_tool_result_bytes_derived(self):
        assert MAX_TOOL_RESULT_BYTES == MAX_TOOL_RESULT_TOKENS * BYTES_PER_TOKEN
        assert MAX_TOOL_RESULT_BYTES == 400_000

    def test_max_per_message_chars(self):
        assert MAX_TOOL_RESULTS_PER_MESSAGE_CHARS == 200_000

    def test_tool_summary_max_length(self):
        assert TOOL_SUMMARY_MAX_LENGTH == 50


# =========================================================================
# Spinner verbs
# =========================================================================

class TestSpinnerVerbs:
    def test_is_tuple(self):
        assert isinstance(SPINNER_VERBS, tuple)

    def test_count(self):
        assert len(SPINNER_VERBS) == 187

    def test_first_verb(self):
        assert SPINNER_VERBS[0] == "Accomplishing"

    def test_last_verb(self):
        assert SPINNER_VERBS[-1] == "Zigzagging"

    def test_all_strings(self):
        for verb in SPINNER_VERBS:
            assert isinstance(verb, str)

    def test_contains_sigiling(self):
        assert "Sigiling" in SPINNER_VERBS
        assert "Clauding" not in SPINNER_VERBS  # no upstream product names in the word list

    def test_contains_thinking(self):
        assert "Thinking" in SPINNER_VERBS

    def test_no_duplicates(self):
        assert len(SPINNER_VERBS) == len(set(SPINNER_VERBS))


# =========================================================================
# Turn completion verbs
# =========================================================================

class TestTurnCompletionVerbs:
    def test_is_tuple(self):
        assert isinstance(TURN_COMPLETION_VERBS, tuple)

    def test_count(self):
        assert len(TURN_COMPLETION_VERBS) == 8

    def test_contains_worked(self):
        assert "Worked" in TURN_COMPLETION_VERBS

    def test_contains_baked(self):
        assert "Baked" in TURN_COMPLETION_VERBS

    def test_all_past_tense(self):
        # All end in 'd' (past tense)
        for verb in TURN_COMPLETION_VERBS:
            assert verb[-1] == "d", f"{verb} doesn't end with 'd'"


# =========================================================================
# Figures / UI symbols
# =========================================================================

class TestFigures:
    def test_black_circle_is_string(self):
        assert isinstance(BLACK_CIRCLE, str)
        assert len(BLACK_CIRCLE) == 1

    def test_effort_symbols_are_distinct(self):
        symbols = {EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX}
        assert len(symbols) == 4

    def test_arrows(self):
        assert UP_ARROW == "\u2191"
        assert DOWN_ARROW == "\u2193"

    def test_bridge_spinner_frames(self):
        assert isinstance(BRIDGE_SPINNER_FRAMES, tuple)
        assert len(BRIDGE_SPINNER_FRAMES) == 4

    def test_play_pause_icons(self):
        assert PLAY_ICON == "\u25b6"
        assert PAUSE_ICON == "\u23f8"

    def test_diamond_symbols(self):
        assert DIAMOND_OPEN == "\u25c7"
        assert DIAMOND_FILLED == "\u25c6"


# =========================================================================
# XML tag constants
# =========================================================================

class TestXMLTags:
    def test_command_tags(self):
        assert COMMAND_NAME_TAG == "command-name"
        assert COMMAND_MESSAGE_TAG == "command-message"
        assert COMMAND_ARGS_TAG == "command-args"

    def test_bash_tags(self):
        assert BASH_INPUT_TAG == "bash-input"
        assert BASH_STDOUT_TAG == "bash-stdout"
        assert BASH_STDERR_TAG == "bash-stderr"

    def test_terminal_output_tags_tuple(self):
        assert isinstance(TERMINAL_OUTPUT_TAGS, tuple)
        assert len(TERMINAL_OUTPUT_TAGS) == 6
        assert BASH_INPUT_TAG in TERMINAL_OUTPUT_TAGS
        assert LOCAL_COMMAND_STDOUT_TAG in TERMINAL_OUTPUT_TAGS

    def test_tick_tag(self):
        assert TICK_TAG == "tick"

    def test_task_tags(self):
        assert TASK_NOTIFICATION_TAG == "task-notification"
        assert TASK_ID_TAG == "task-id"
        assert TOOL_USE_ID_TAG == "tool-use-id"

    def test_worktree_tags(self):
        assert WORKTREE_TAG == "worktree"
        assert WORKTREE_PATH_TAG == "worktreePath"

    def test_fork_tags(self):
        assert FORK_BOILERPLATE_TAG == "fork-boilerplate"
        assert FORK_DIRECTIVE_PREFIX == "Your directive: "

    def test_common_help_args(self):
        assert isinstance(COMMON_HELP_ARGS, tuple)
        assert "help" in COMMON_HELP_ARGS
        assert "-h" in COMMON_HELP_ARGS
        assert "--help" in COMMON_HELP_ARGS

    def test_common_info_args(self):
        assert isinstance(COMMON_INFO_ARGS, tuple)
        assert "list" in COMMON_INFO_ARGS
        assert "status" in COMMON_INFO_ARGS
        assert "?" in COMMON_INFO_ARGS


# =========================================================================
# Message constants
# =========================================================================

class TestMessages:
    def test_no_content_message(self):
        assert NO_CONTENT_MESSAGE == "(no content)"


# =========================================================================
# Date utilities
# =========================================================================

class TestDateUtilities:
    def test_get_local_iso_date_format(self):
        d = get_local_iso_date()
        parts = d.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4  # year
        assert len(parts[1]) == 2  # month
        assert len(parts[2]) == 2  # day

    def test_get_local_iso_date_override(self):
        with patch.dict(os.environ, {"SIGIL_OVERRIDE_DATE": "2025-01-15"}):
            assert get_local_iso_date() == "2025-01-15"

    def test_get_local_iso_date_own_override_wins(self):
        with patch.dict(os.environ, {"SIGIL_OVERRIDE_DATE": "2031-03-04"}):
            assert get_local_iso_date() == "2031-03-04"

    def test_get_session_start_date_memoised(self):
        reset_session_start_date()
        d1 = get_session_start_date()
        d2 = get_session_start_date()
        assert d1 == d2

    def test_reset_session_start_date(self):
        reset_session_start_date()
        d = get_session_start_date()
        assert isinstance(d, str)
        reset_session_start_date()
        # After reset, should still return valid date
        d2 = get_session_start_date()
        assert isinstance(d2, str)

    def test_get_local_month_year_format(self):
        result = get_local_month_year()
        parts = result.split()
        assert len(parts) == 2
        assert parts[1].isdigit()
        assert len(parts[1]) == 4

    def test_get_local_month_year_override(self):
        with patch.dict(os.environ, {"SIGIL_OVERRIDE_DATE": "2026-02-15"}):
            assert get_local_month_year() == "February 2026"


# =========================================================================
# System prompt section caching
# =========================================================================

class TestSystemPromptSections:
    def setup_method(self):
        clear_system_prompt_sections()

    def test_system_prompt_section_creates_cached(self):
        s = system_prompt_section("test", lambda: "hello")
        assert s.name == "test"
        assert s.cache_break is False

    def test_dangerous_uncached_creates_volatile(self):
        s = dangerous_uncached_system_prompt_section("test", lambda: "hello", "reason")
        assert s.name == "test"
        assert s.cache_break is True

    def test_resolve_caches_sections(self):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return f"value-{call_count}"

        sections = [system_prompt_section("s1", compute)]
        r1 = resolve_system_prompt_sections(sections)
        r2 = resolve_system_prompt_sections(sections)
        assert r1 == ["value-1"]
        assert r2 == ["value-1"]  # cached
        assert call_count == 1

    def test_uncached_recomputes(self):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return f"value-{call_count}"

        sections = [dangerous_uncached_system_prompt_section("s2", compute, "test")]
        r1 = resolve_system_prompt_sections(sections)
        r2 = resolve_system_prompt_sections(sections)
        assert r1 == ["value-1"]
        assert r2 == ["value-2"]  # recomputed
        assert call_count == 2

    def test_clear_resets_cache(self):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return f"value-{call_count}"

        sections = [system_prompt_section("s3", compute)]
        resolve_system_prompt_sections(sections)
        clear_system_prompt_sections()
        r = resolve_system_prompt_sections(sections)
        assert r == ["value-2"]
        assert call_count == 2

    def test_resolve_handles_none(self):
        sections = [system_prompt_section("nil", lambda: None)]
        r = resolve_system_prompt_sections(sections)
        assert r == [None]

    def test_multiple_sections(self):
        sections = [
            system_prompt_section("a", lambda: "alpha"),
            system_prompt_section("b", lambda: "beta"),
            system_prompt_section("c", lambda: None),
        ]
        r = resolve_system_prompt_sections(sections)
        assert r == ["alpha", "beta", None]


# =========================================================================
# Output styles
# =========================================================================

class TestOutputStyles:
    def test_default_style_name(self):
        assert DEFAULT_OUTPUT_STYLE_NAME == "default"

    def test_default_style_is_none(self):
        assert OUTPUT_STYLE_CONFIGS[DEFAULT_OUTPUT_STYLE_NAME] is None

    def test_explanatory_exists(self):
        style = OUTPUT_STYLE_CONFIGS["Explanatory"]
        assert style is not None
        assert style.name == "Explanatory"
        assert "explains" in style.description

    def test_learning_exists(self):
        style = OUTPUT_STYLE_CONFIGS["Learning"]
        assert style is not None
        assert style.name == "Learning"
        assert "hands-on" in style.description

    def test_output_style_config_frozen(self):
        style = OutputStyleConfig(
            name="Test", description="desc", prompt="prompt"
        )
        with pytest.raises(Exception):
            style.name = "other"  # type: ignore[misc]


# =========================================================================
# Knowledge cutoff
# =========================================================================

class TestKnowledgeCutoff:
    def test_frontier_model_name(self):
        assert FRONTIER_MODEL_NAME == "Claude Opus 4.6"

    def test_opus_46_cutoff(self):
        assert get_knowledge_cutoff("claude-opus-4-6-20250601") == "May 2025"

    def test_sonnet_46_cutoff(self):
        assert get_knowledge_cutoff("claude-sonnet-4-6-20250801") == "August 2025"

    def test_opus_45_cutoff(self):
        assert get_knowledge_cutoff("claude-opus-4-5-20250601") == "May 2025"

    def test_haiku_cutoff(self):
        assert get_knowledge_cutoff("claude-haiku-4-20250201") == "February 2025"

    def test_sonnet_4_cutoff(self):
        assert get_knowledge_cutoff("claude-sonnet-4-20250114") == "January 2025"

    def test_unknown_model_returns_none(self):
        assert get_knowledge_cutoff("gpt-4-turbo") is None

    def test_claude_model_ids(self):
        assert "opus" in CLAUDE_MODEL_IDS
        assert "sonnet" in CLAUDE_MODEL_IDS
        assert "haiku" in CLAUDE_MODEL_IDS


# =========================================================================
# Prompt section helpers
# =========================================================================

class TestPromptSectionHelpers:
    def test_hooks_section_content(self):
        assert "hooks" in HOOKS_SECTION
        assert "user-prompt-submit-hook" in HOOKS_SECTION

    def test_system_reminders_section(self):
        assert "system-reminder" in SYSTEM_REMINDERS_SECTION

    def test_summarize_tool_results(self):
        assert "tool results" in SUMMARIZE_TOOL_RESULTS_SECTION

    def test_default_agent_prompt(self):
        assert "agent for Sigil Code" in DEFAULT_AGENT_PROMPT

    def test_dynamic_boundary(self):
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY == "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

    def test_language_section_none_when_no_preference(self):
        assert get_language_section(None) is None
        assert get_language_section("") is None

    def test_language_section_with_preference(self):
        result = get_language_section("Spanish")
        assert result is not None
        assert "Spanish" in result
        assert "# Language" in result

    def test_output_style_section_none_when_no_config(self):
        assert get_output_style_section(None) is None

    def test_output_style_section_with_config(self):
        config = OutputStyleConfig(
            name="TestStyle",
            description="A test style",
            prompt="Be concise.",
        )
        result = get_output_style_section(config)
        assert result is not None
        assert "# Output Style: TestStyle" in result
        assert "Be concise." in result

    def test_scratchpad_none_when_no_dir(self):
        assert get_scratchpad_instructions(None) is None
        assert get_scratchpad_instructions("") is None

    def test_scratchpad_with_dir(self):
        result = get_scratchpad_instructions("/tmp/session-123")
        assert result is not None
        assert "/tmp/session-123" in result
        assert "# Scratchpad Directory" in result
        assert "temporary files" in result


# =========================================================================
# Error IDs
# =========================================================================

class TestErrorIDs:
    def test_tool_use_summary_error(self):
        assert E_TOOL_USE_SUMMARY_GENERATION_FAILED == 344


# =========================================================================
# UI helpers
# =========================================================================

class TestUIHelpers:
    def test_random_verbs_come_from_the_lists(self):
        assert random_spinner_verb() in SPINNER_VERBS
        assert random_completion_verb() in TURN_COMPLETION_VERBS

    def test_effort_figures(self):
        assert EFFORT_FIGURES == {"low": EFFORT_LOW, "medium": EFFORT_MEDIUM, "high": EFFORT_HIGH,
                                  "xhigh": EFFORT_MAX, "max": EFFORT_MAX}
