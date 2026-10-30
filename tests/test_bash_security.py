"""
Tests for bash_security module.

Tests are organized by validator function.
"""

import pytest

from src.bash_security import (
    SecurityBehavior,
    SecurityResult,
    ValidationContext,
    bash_command_is_safe,
    check_shell_security,
    extract_quoted_content,
    get_destructive_command_warning,
    has_unescaped_char,
    interpret_command_result,
    is_command_read_only,
    split_command,
    strip_safe_redirections,
    validate_backslash_escaped_operators,
    validate_backslash_escaped_whitespace,
    validate_brace_expansion,
    validate_carriage_return,
    validate_comment_quote_desync,
    validate_control_characters,
    validate_dangerous_patterns,
    validate_dangerous_variables,
    validate_empty,
    validate_git_commit,
    validate_ifs_injection,
    validate_incomplete_commands,
    validate_jq_command,
    validate_mid_word_hash,
    validate_newlines,
    validate_obfuscated_flags,
    validate_proc_environ_access,
    validate_quoted_newline,
    validate_redirections,
    validate_shell_metacharacters,
    validate_unicode_whitespace,
    validate_zsh_dangerous_commands,
)


# ---- Helper to build a context ----

def _ctx(cmd: str) -> ValidationContext:
    """Build a ValidationContext for the given command."""
    base = cmd.strip().split()[0] if cmd.strip() else ''
    with_dq, fully_unq, keep_qc = extract_quoted_content(cmd)
    return ValidationContext(
        original_command=cmd,
        base_command=base,
        unquoted_content=with_dq,
        fully_unquoted_content=strip_safe_redirections(fully_unq),
        fully_unquoted_pre_strip=fully_unq,
        unquoted_keep_quote_chars=keep_qc,
    )


# ===========================================================================
# extract_quoted_content
# ===========================================================================

class TestExtractQuotedContent:
    def test_no_quotes(self):
        dq, full, kqc = extract_quoted_content('echo hello')
        assert dq == 'echo hello'
        assert full == 'echo hello'

    def test_single_quotes_stripped(self):
        dq, full, kqc = extract_quoted_content("echo 'hello world'")
        assert 'hello world' not in full
        assert 'echo' in full

    def test_double_quotes_in_dq_output(self):
        dq, full, kqc = extract_quoted_content('echo "hello world"')
        assert 'hello world' in dq  # double-quoted content preserved in dq
        assert 'hello world' not in full  # but stripped in fully_unquoted

    def test_escape_handling(self):
        dq, full, kqc = extract_quoted_content('echo \\$HOME')
        assert '$HOME' in full

    def test_keep_quote_chars(self):
        _, _, kqc = extract_quoted_content("echo 'x'#")
        assert "'" in kqc  # quote chars preserved


# ===========================================================================
# strip_safe_redirections
# ===========================================================================

class TestStripSafeRedirections:
    def test_dev_null_output(self):
        assert '>/dev/null' not in strip_safe_redirections('cmd > /dev/null')

    def test_stderr_redirect(self):
        assert '2>&1' not in strip_safe_redirections('cmd 2>&1')

    def test_dev_null_input(self):
        assert '</dev/null' not in strip_safe_redirections('cmd < /dev/null')

    def test_preserves_other_redirections(self):
        result = strip_safe_redirections('cmd > output.txt')
        assert '> output.txt' in result


# ===========================================================================
# has_unescaped_char
# ===========================================================================

class TestHasUnescapedChar:
    def test_unescaped_backtick(self):
        assert has_unescaped_char('echo `date`', '`') is True

    def test_escaped_backtick(self):
        assert has_unescaped_char('echo \\`safe\\`', '`') is False

    def test_double_backslash_then_backtick(self):
        # \\\` → \\ (literal backslash) + ` (unescaped)
        assert has_unescaped_char('test\\\\`date`', '`') is True

    def test_no_match(self):
        assert has_unescaped_char('echo hello', '`') is False


# ===========================================================================
# split_command
# ===========================================================================

class TestSplitCommand:
    def test_simple(self):
        assert split_command('echo hello') == ['echo hello']

    def test_semicolon(self):
        assert split_command('echo a; echo b') == ['echo a', 'echo b']

    def test_and_and(self):
        assert split_command('cmd1 && cmd2') == ['cmd1', 'cmd2']

    def test_pipe(self):
        assert split_command('cat file | grep pattern') == ['cat file', 'grep pattern']

    def test_or_or(self):
        assert split_command('cmd1 || cmd2') == ['cmd1', 'cmd2']

    def test_quotes_preserved(self):
        result = split_command("echo 'a; b'")
        assert len(result) == 1  # semicolon inside quotes not split

    def test_complex(self):
        result = split_command('cd /tmp && echo hi; ls | head')
        assert len(result) == 4


# ===========================================================================
# validate_empty
# ===========================================================================

class TestValidateEmpty:
    def test_empty(self):
        assert validate_empty(_ctx('')).behavior == SecurityBehavior.ALLOW

    def test_whitespace_only(self):
        assert validate_empty(_ctx('   ')).behavior == SecurityBehavior.ALLOW

    def test_non_empty(self):
        assert validate_empty(_ctx('ls')).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_control_characters
# ===========================================================================

class TestValidateControlCharacters:
    def test_null_byte(self):
        result = validate_control_characters(_ctx('echo\x00hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_bell(self):
        result = validate_control_characters(_ctx('echo\x07hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean_command(self):
        result = validate_control_characters(_ctx('echo hello'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_tab_allowed(self):
        result = validate_control_characters(_ctx('echo\thello'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_newline_allowed(self):
        result = validate_control_characters(_ctx('echo\nhello'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_incomplete_commands
# ===========================================================================

class TestValidateIncompleteCommands:
    def test_starts_with_tab(self):
        result = validate_incomplete_commands(_ctx('\techo hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_starts_with_dash(self):
        result = validate_incomplete_commands(_ctx('-rf /'))
        assert result.behavior == SecurityBehavior.ASK

    def test_starts_with_operator(self):
        result = validate_incomplete_commands(_ctx('&& echo hello'))
        assert result.behavior == SecurityBehavior.ASK
        result = validate_incomplete_commands(_ctx('; echo hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_normal_command(self):
        result = validate_incomplete_commands(_ctx('ls -la'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_git_commit
# ===========================================================================

class TestValidateGitCommit:
    def test_not_git(self):
        result = validate_git_commit(_ctx('echo hello'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_simple_commit(self):
        result = validate_git_commit(_ctx("git commit -m 'initial commit'"))
        assert result.behavior == SecurityBehavior.ALLOW

    def test_double_quoted_commit(self):
        result = validate_git_commit(_ctx('git commit -m "fix bug"'))
        assert result.behavior == SecurityBehavior.ALLOW

    def test_commit_with_substitution(self):
        result = validate_git_commit(_ctx('git commit -m "$(date)"'))
        assert result.behavior == SecurityBehavior.ASK

    def test_commit_with_backtick(self):
        result = validate_git_commit(_ctx('git commit -m "`date`"'))
        assert result.behavior == SecurityBehavior.ASK

    def test_commit_with_chained_commands(self):
        result = validate_git_commit(_ctx("git commit -m 'msg'; rm -rf /"))
        # Should passthrough (not early-allow) due to ; in remainder
        assert result.behavior != SecurityBehavior.ALLOW

    def test_commit_with_backslash(self):
        result = validate_git_commit(_ctx('git commit -m "test\\"msg"'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_jq_command
# ===========================================================================

class TestValidateJqCommand:
    def test_not_jq(self):
        assert validate_jq_command(_ctx('echo hi')).behavior == SecurityBehavior.PASSTHROUGH

    def test_jq_system(self):
        result = validate_jq_command(_ctx('jq "system(\"rm -rf /\")"'))
        assert result.behavior == SecurityBehavior.ASK

    def test_jq_from_file(self):
        result = validate_jq_command(_ctx('jq -f evil.jq'))
        assert result.behavior == SecurityBehavior.ASK

    def test_jq_slurpfile(self):
        result = validate_jq_command(_ctx('jq --slurpfile x data.json'))
        assert result.behavior == SecurityBehavior.ASK

    def test_safe_jq(self):
        result = validate_jq_command(_ctx('jq ".name" data.json'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_obfuscated_flags
# ===========================================================================

class TestValidateObfuscatedFlags:
    def test_ansi_c_quoting(self):
        result = validate_obfuscated_flags(_ctx("find . $'-exec' evil"))
        assert result.behavior == SecurityBehavior.ASK

    def test_locale_quoting(self):
        result = validate_obfuscated_flags(_ctx('find . $"-exec" evil'))
        assert result.behavior == SecurityBehavior.ASK

    def test_echo_safe(self):
        result = validate_obfuscated_flags(_ctx("echo $'hello'"))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_empty_quotes_before_dash(self):
        result = validate_obfuscated_flags(_ctx("find . '' -exec evil"))
        assert result.behavior == SecurityBehavior.ASK

    def test_quoted_flag(self):
        result = validate_obfuscated_flags(_ctx('find . "-exec" rm {} ;'))
        assert result.behavior == SecurityBehavior.ASK

    def test_normal_command(self):
        result = validate_obfuscated_flags(_ctx('ls -la'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_three_consecutive_quotes(self):
        result = validate_obfuscated_flags(_ctx("find . '''exec"))
        assert result.behavior == SecurityBehavior.ASK


# ===========================================================================
# validate_shell_metacharacters
# ===========================================================================

class TestValidateShellMetacharacters:
    def test_semicolon_in_quotes(self):
        result = validate_shell_metacharacters(_ctx('echo "a;b"'))
        # unquoted_content (with_double_quotes) has the ; inside
        assert result.behavior == SecurityBehavior.PASSTHROUGH or result.behavior == SecurityBehavior.ASK

    def test_find_name_with_pipe(self):
        # Single-quoted pipe is stripped entirely from unquoted content → safe
        ctx = _ctx("find . -name '|evil'")
        result = validate_shell_metacharacters(ctx)
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_double_quoted_metachar(self):
        # Check that we catch metacharacters in unquoted positions
        # Quote characters are stripped before the check, so quoted
        # metacharacters are not flagged. So we test
        # the actual dangerous case: unquoted semicolon
        ctx = _ctx('find . -name evil; rm -rf /')
        # This won't be caught by this specific validator (it looks for
        # metacharacters INSIDE quoted args, not command separators)
        result = validate_shell_metacharacters(ctx)
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_clean_command(self):
        assert validate_shell_metacharacters(_ctx('ls -la')).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_dangerous_variables
# ===========================================================================

class TestValidateDangerousVariables:
    def test_variable_in_pipe(self):
        result = validate_dangerous_variables(_ctx('$CMD | grep x'))
        assert result.behavior == SecurityBehavior.ASK

    def test_variable_in_redirect(self):
        result = validate_dangerous_variables(_ctx('echo x > $FILE'))
        assert result.behavior == SecurityBehavior.ASK

    def test_safe_variable(self):
        result = validate_dangerous_variables(_ctx('echo $HOME'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_dangerous_patterns
# ===========================================================================

class TestValidateDangerousPatterns:
    def test_backtick(self):
        result = validate_dangerous_patterns(_ctx('echo `date`'))
        assert result.behavior == SecurityBehavior.ASK

    def test_dollar_paren(self):
        result = validate_dangerous_patterns(_ctx('echo $(date)'))
        assert result.behavior == SecurityBehavior.ASK

    def test_dollar_brace(self):
        result = validate_dangerous_patterns(_ctx('echo ${PATH}'))
        assert result.behavior == SecurityBehavior.ASK

    def test_process_substitution(self):
        result = validate_dangerous_patterns(_ctx('diff <(cmd1) <(cmd2)'))
        assert result.behavior == SecurityBehavior.ASK

    def test_safe_echo(self):
        result = validate_dangerous_patterns(_ctx('echo hello'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_escaped_backtick(self):
        result = validate_dangerous_patterns(_ctx('echo \\`safe\\`'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_redirections
# ===========================================================================

class TestValidateRedirections:
    def test_output_redirect(self):
        result = validate_redirections(_ctx('echo x > file.txt'))
        assert result.behavior == SecurityBehavior.ASK

    def test_input_redirect(self):
        result = validate_redirections(_ctx('cat < /etc/passwd'))
        assert result.behavior == SecurityBehavior.ASK

    def test_dev_null_stripped(self):
        # >/dev/null is stripped by strip_safe_redirections
        result = validate_redirections(_ctx('cmd > /dev/null'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_no_redirect(self):
        result = validate_redirections(_ctx('echo hello'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_newlines
# ===========================================================================

class TestValidateNewlines:
    def test_no_newlines(self):
        assert validate_newlines(_ctx('echo hello')).behavior == SecurityBehavior.PASSTHROUGH

    def test_newline_with_command(self):
        result = validate_newlines(_ctx('echo hello\nrm -rf /'))
        assert result.behavior == SecurityBehavior.ASK

    def test_backslash_continuation(self):
        result = validate_newlines(_ctx('cmd \\\n--flag'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_carriage_return
# ===========================================================================

class TestValidateCarriageReturn:
    def test_no_cr(self):
        assert validate_carriage_return(_ctx('echo hello')).behavior == SecurityBehavior.PASSTHROUGH

    def test_cr_in_command(self):
        result = validate_carriage_return(_ctx('echo hello\rworld'))
        assert result.behavior == SecurityBehavior.ASK
        assert result.is_misparsing is True

    def test_cr_in_double_quotes_safe(self):
        result = validate_carriage_return(_ctx('echo "hello\rworld"'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_ifs_injection
# ===========================================================================

class TestValidateIFSInjection:
    def test_ifs_variable(self):
        result = validate_ifs_injection(_ctx('echo$IFS/etc/passwd'))
        assert result.behavior == SecurityBehavior.ASK

    def test_ifs_expansion(self):
        result = validate_ifs_injection(_ctx('echo ${IFS:0:1}'))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean(self):
        assert validate_ifs_injection(_ctx('echo hello')).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_proc_environ_access
# ===========================================================================

class TestValidateProcEnvironAccess:
    def test_proc_environ(self):
        result = validate_proc_environ_access(_ctx('cat /proc/self/environ'))
        assert result.behavior == SecurityBehavior.ASK

    def test_proc_pid_environ(self):
        result = validate_proc_environ_access(_ctx('cat /proc/1/environ'))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean(self):
        assert validate_proc_environ_access(_ctx('cat /etc/hosts')).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_backslash_escaped_whitespace
# ===========================================================================

class TestValidateBackslashEscapedWhitespace:
    def test_escaped_space(self):
        result = validate_backslash_escaped_whitespace(_ctx('echo\\ hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_escaped_tab(self):
        result = validate_backslash_escaped_whitespace(_ctx('echo\\\thello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean(self):
        assert validate_backslash_escaped_whitespace(
            _ctx('echo hello')
        ).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_backslash_escaped_operators
# ===========================================================================

class TestValidateBackslashEscapedOperators:
    def test_escaped_semicolon(self):
        result = validate_backslash_escaped_operators(_ctx('cat safe.txt \\; echo /etc/passwd'))
        assert result.behavior == SecurityBehavior.ASK

    def test_escaped_pipe(self):
        result = validate_backslash_escaped_operators(_ctx('cmd \\| evil'))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean(self):
        assert validate_backslash_escaped_operators(
            _ctx('ls -la')
        ).behavior == SecurityBehavior.PASSTHROUGH

    def test_inside_quotes_safe(self):
        result = validate_backslash_escaped_operators(_ctx("echo '\\;'"))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_brace_expansion
# ===========================================================================

class TestValidateBraceExpansion:
    def test_comma_expansion(self):
        result = validate_brace_expansion(_ctx('echo {a,b,c}'))
        assert result.behavior == SecurityBehavior.ASK

    def test_sequence_expansion(self):
        result = validate_brace_expansion(_ctx('echo {1..5}'))
        assert result.behavior == SecurityBehavior.ASK

    def test_no_expansion(self):
        result = validate_brace_expansion(_ctx('echo {hello}'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_escaped_brace(self):
        result = validate_brace_expansion(_ctx('echo \\{a,b\\}'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_excess_closing_braces(self):
        result = validate_brace_expansion(_ctx("git diff {@'{'0},--output=/tmp/pwned}"))
        assert result.behavior == SecurityBehavior.ASK


# ===========================================================================
# validate_unicode_whitespace
# ===========================================================================

class TestValidateUnicodeWhitespace:
    def test_nbsp(self):
        result = validate_unicode_whitespace(_ctx('echo\u00a0hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_em_space(self):
        result = validate_unicode_whitespace(_ctx('echo\u2003hello'))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean(self):
        assert validate_unicode_whitespace(_ctx('echo hello')).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_mid_word_hash
# ===========================================================================

class TestValidateMidWordHash:
    def test_mid_word_hash(self):
        result = validate_mid_word_hash(_ctx('echotest#comment'))
        assert result.behavior == SecurityBehavior.ASK

    def test_word_start_hash(self):
        result = validate_mid_word_hash(_ctx('echo # comment'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_dollar_brace_hash_safe(self):
        # ${#var} is bash string length, should be safe
        result = validate_mid_word_hash(_ctx('echo ${#var}'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_comment_quote_desync
# ===========================================================================

class TestValidateCommentQuoteDesync:
    def test_quote_in_comment(self):
        result = validate_comment_quote_desync(_ctx("echo hello # it's a comment"))
        assert result.behavior == SecurityBehavior.ASK

    def test_clean_comment(self):
        result = validate_comment_quote_desync(_ctx('echo hello # clean comment'))
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_no_comment(self):
        assert validate_comment_quote_desync(_ctx('echo hello')).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_quoted_newline
# ===========================================================================

class TestValidateQuotedNewline:
    def test_quoted_newline_with_hash(self):
        result = validate_quoted_newline(_ctx("echo 'hello\n# dangerous line'"))
        assert result.behavior == SecurityBehavior.ASK

    def test_no_newline(self):
        assert validate_quoted_newline(_ctx('echo hello')).behavior == SecurityBehavior.PASSTHROUGH

    def test_newline_without_hash(self):
        assert validate_quoted_newline(_ctx("echo 'hello\nworld'")).behavior == SecurityBehavior.PASSTHROUGH


# ===========================================================================
# validate_zsh_dangerous_commands
# ===========================================================================

class TestValidateZshDangerousCommands:
    def test_zmodload(self):
        result = validate_zsh_dangerous_commands(_ctx('zmodload zsh/system'))
        assert result.behavior == SecurityBehavior.ASK

    def test_zpty(self):
        result = validate_zsh_dangerous_commands(_ctx('zpty cmd echo'))
        assert result.behavior == SecurityBehavior.ASK

    def test_emulate(self):
        result = validate_zsh_dangerous_commands(_ctx('emulate -c evil'))
        assert result.behavior == SecurityBehavior.ASK

    def test_fc_e(self):
        result = validate_zsh_dangerous_commands(_ctx('fc -e vim'))
        assert result.behavior == SecurityBehavior.ASK

    def test_normal_command(self):
        assert validate_zsh_dangerous_commands(_ctx('ls -la')).behavior == SecurityBehavior.PASSTHROUGH

    def test_env_var_prefix(self):
        result = validate_zsh_dangerous_commands(_ctx('FOO=bar zmodload zsh/system'))
        assert result.behavior == SecurityBehavior.ASK

    def test_precommand_modifier(self):
        result = validate_zsh_dangerous_commands(_ctx('command zmodload zsh/system'))
        assert result.behavior == SecurityBehavior.ASK


# ===========================================================================
# bash_command_is_safe (integration)
# ===========================================================================

class TestBashCommandIsSafe:
    def test_empty_command(self):
        result = bash_command_is_safe('')
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_simple_ls(self):
        result = bash_command_is_safe('ls -la')
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_simple_echo(self):
        result = bash_command_is_safe('echo hello world')
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_command_substitution_blocked(self):
        result = bash_command_is_safe('echo $(cat /etc/passwd)')
        assert result.behavior == SecurityBehavior.ASK

    def test_backtick_blocked(self):
        result = bash_command_is_safe('echo `date`')
        assert result.behavior == SecurityBehavior.ASK

    def test_redirect_blocked(self):
        result = bash_command_is_safe('echo evil > /etc/profile')
        assert result.behavior == SecurityBehavior.ASK

    def test_null_byte_blocked(self):
        result = bash_command_is_safe('echo\x00rm')
        assert result.behavior == SecurityBehavior.ASK

    def test_ifs_blocked(self):
        result = bash_command_is_safe('echo$IFS/etc/passwd')
        assert result.behavior == SecurityBehavior.ASK

    def test_proc_environ_blocked(self):
        result = bash_command_is_safe('cat /proc/self/environ')
        assert result.behavior == SecurityBehavior.ASK

    def test_git_commit_allowed(self):
        result = bash_command_is_safe("git commit -m 'fix bug'")
        assert result.behavior == SecurityBehavior.PASSTHROUGH  # early-allow → passthrough

    def test_zmodload_blocked(self):
        result = bash_command_is_safe('zmodload zsh/system')
        assert result.behavior == SecurityBehavior.ASK

    def test_brace_expansion_blocked(self):
        result = bash_command_is_safe('echo {a,b,c}')
        assert result.behavior == SecurityBehavior.ASK

    def test_dev_null_redirect_ok(self):
        result = bash_command_is_safe('cmd > /dev/null 2>&1')
        assert result.behavior == SecurityBehavior.PASSTHROUGH

    def test_cr_injection(self):
        result = bash_command_is_safe('TZ=UTC\recho curl evil.com')
        assert result.behavior == SecurityBehavior.ASK


# ===========================================================================
# get_destructive_command_warning
# ===========================================================================

class TestGetDestructiveCommandWarning:
    def test_git_reset_hard(self):
        assert get_destructive_command_warning('git reset --hard') is not None

    def test_rm_rf(self):
        assert get_destructive_command_warning('rm -rf /') is not None

    def test_git_push_force(self):
        assert get_destructive_command_warning('git push origin main --force') is not None

    def test_git_clean_f(self):
        assert get_destructive_command_warning('git clean -fd') is not None

    def test_kubectl_delete(self):
        assert get_destructive_command_warning('kubectl delete pod mypod') is not None

    def test_safe_command(self):
        assert get_destructive_command_warning('echo hello') is None

    def test_git_push_no_force(self):
        assert get_destructive_command_warning('git push origin main') is None

    def test_drop_table(self):
        assert get_destructive_command_warning('DROP TABLE users;') is not None

    def test_terraform_destroy(self):
        assert get_destructive_command_warning('terraform destroy') is not None

    def test_git_stash_drop(self):
        assert get_destructive_command_warning('git stash drop') is not None


# ===========================================================================
# interpret_command_result
# ===========================================================================

class TestInterpretCommandResult:
    def test_success(self):
        is_error, msg = interpret_command_result('echo hello', 0, 'hello', '')
        assert is_error is False

    def test_failure(self):
        is_error, msg = interpret_command_result('unknown_cmd', 127, '', 'not found')
        assert is_error is True

    def test_grep_no_match(self):
        is_error, msg = interpret_command_result('grep pattern file', 1, '', '')
        assert is_error is False
        assert msg == 'No matches found'

    def test_grep_error(self):
        is_error, msg = interpret_command_result('grep pattern file', 2, '', 'error')
        assert is_error is True

    def test_diff_files_differ(self):
        is_error, msg = interpret_command_result('diff a b', 1, 'output', '')
        assert is_error is False
        assert msg == 'Files differ'

    def test_find_partial(self):
        is_error, msg = interpret_command_result('find / -name x', 1, '', '')
        assert is_error is False


# ===========================================================================
# is_command_read_only
# ===========================================================================

class TestIsCommandReadOnly:
    def test_ls(self):
        assert is_command_read_only('ls -la') is True

    def test_cat(self):
        assert is_command_read_only('cat file.txt') is True

    def test_grep(self):
        assert is_command_read_only('grep -r pattern .') is True

    def test_git_status(self):
        assert is_command_read_only('git status') is True

    def test_git_log(self):
        assert is_command_read_only('git log --oneline') is True

    def test_git_push(self):
        assert is_command_read_only('git push') is False

    def test_rm(self):
        assert is_command_read_only('rm file.txt') is False

    def test_sed_read_only(self):
        assert is_command_read_only("sed -n '1,5p' file") is True

    def test_sed_in_place(self):
        assert is_command_read_only("sed -i 's/old/new/' file") is False

    def test_find_safe(self):
        assert is_command_read_only('find . -name "*.py"') is True

    def test_find_exec(self):
        assert is_command_read_only('find . -exec rm {} ;') is False

    def test_find_delete(self):
        assert is_command_read_only('find . -name "*.tmp" -delete') is False

    def test_echo(self):
        assert is_command_read_only('echo hello') is True

    def test_python_version(self):
        assert is_command_read_only('python --version') is True

    def test_unknown_command(self):
        assert is_command_read_only('some_random_command') is False


# ===========================================================================
# check_shell_security (integration)
# ===========================================================================

class TestCheckShellSecurity:
    def test_shell_disabled(self):
        allowed, msg = check_shell_security('ls', allow_shell=False)
        assert allowed is False
        assert 'disabled' in msg.lower()

    def test_safe_command_allowed(self):
        allowed, msg = check_shell_security('ls -la')
        assert allowed is True

    def test_destructive_blocked(self):
        allowed, msg = check_shell_security('rm -rf /', allow_destructive=False)
        assert allowed is False
        assert 'destructive' in msg.lower()

    def test_destructive_allowed_when_enabled(self):
        allowed, msg = check_shell_security('rm -rf /tmp/test', allow_destructive=True)
        # rm -rf still triggers destructive check, but allow_destructive=True skips it
        # However rm -rf may also trigger the security check for force-remove
        # Let's check: the main security check should pass (no injection)
        # and destructive should be allowed
        assert allowed is True

    def test_injection_blocked(self):
        allowed, msg = check_shell_security('echo `evil`')
        assert allowed is False
        assert 'backtick' in msg.lower() or 'security' in msg.lower()

    def test_misparsing_always_blocked(self):
        allowed, msg = check_shell_security('echo\x00rm')
        assert allowed is False

    def test_safe_git_commit(self):
        allowed, msg = check_shell_security("git commit -m 'fix'")
        assert allowed is True
