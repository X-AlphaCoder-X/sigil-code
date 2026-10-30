"""Parity audit: which features of a reference list of coding-agent features Sigil Code has.

The lists below are feature names (checked on 2026-10-29): built-in commands, hook events and
tool names used in permission rules and hook matchers. Sigil's side is read from its own code:
its slash commands, the hook events it handles and the tool names its tools answer to.
`sigil parity-audit` prints the coverage and what is missing.
"""
from __future__ import annotations

from dataclasses import dataclass

DOCS_CHECKED = '2026-10-29'

DOCUMENTED_COMMANDS = (
    'add-dir', 'advisor', 'agents', 'artifact-capabilities', 'artifact-diagramming', 'artifacts',
    'auto-mode-setup', 'autocompact', 'autofix-pr', 'background', 'batch', 'branch', 'btw', 'bug', 'cd',
    'chrome', 'clear', 'code-review', 'color', 'compact', 'config',
    'context', 'copy', 'cost', 'dataviz', 'debug', 'deep-research', 'design', 'design-login', 'design-sync',
    'desktop', 'diff', 'doctor', 'effort', 'exit', 'export', 'fast', 'feedback', 'fewer-permission-prompts',
    'focus', 'fork', 'goal', 'heapdump', 'help', 'hooks', 'ide', 'import', 'init', 'insights',
    'install-github-app', 'install-slack-app', 'keybindings', 'list-agents', 'login', 'logout', 'loop', 'mcp',
    'memory', 'mobile', 'model', 'output-style', 'passes', 'permissions', 'plan', 'plugin', 'plugin-authoring',
    'powerup', 'pr-comments', 'privacy-settings', 'radio', 'rate-limit-options', 'recap', 'release-notes',
    'reload-plugins', 'reload-skills', 'remote-control', 'remote-env', 'rename', 'resume', 'review', 'rewind',
    'run', 'run-skill-generator', 'sandbox', 'schedule', 'scroll-speed', 'security-review', 'setup-bedrock',
    'setup-vertex', 'simplify', 'skill-doctor', 'skills', 'slides', 'stats', 'status', 'statusline', 'stickers',
    'stop', 'subtask', 'tasks', 'team-onboarding', 'teleport', 'terminal-setup', 'theme', 'tui', 'ultraplan',
    'ultrareview', 'update-config', 'upgrade', 'usage', 'usage-credits', 'verify', 'vim', 'voice', 'web-setup',
    'workflow-authoring', 'workflows',
)

DOCUMENTED_HOOK_EVENTS = (
    'SessionStart', 'Setup', 'UserPromptSubmit', 'UserPromptExpansion', 'PreToolUse', 'PermissionRequest',
    'PermissionDenied', 'PostToolUse', 'PostToolUseFailure', 'PostToolBatch', 'Notification', 'MessageDisplay',
    'SubagentStart', 'SubagentStop', 'TaskCreated', 'TaskCompleted', 'Stop', 'StopFailure', 'TeammateIdle',
    'InstructionsLoaded', 'ConfigChange', 'CwdChanged', 'DirectoryAdded', 'FileChanged', 'WorktreeCreate',
    'WorktreeRemove', 'PreCompact', 'PostCompact', 'PreModelSwitch', 'PostModelSwitch', 'Elicitation',
    'ElicitationResult', 'SessionEnd',
)

DOCUMENTED_TOOLS = (
    'Agent', 'Artifact', 'AskUserQuestion', 'Bash', 'CronCreate', 'CronDelete', 'CronList', 'Edit',
    'EndConversation', 'EnterPlanMode', 'EnterWorktree', 'ExitPlanMode', 'ExitWorktree', 'Glob', 'Grep',
    'ListAgents', 'ListMcpResourcesTool', 'LSP', 'Monitor', 'NotebookEdit', 'PowerShell', 'PushNotification',
    'Read', 'ReadMcpResourceTool', 'RemoteTrigger', 'ReportFindings', 'ScheduleWakeup', 'SendFeedback',
    'SendMessage', 'SendUserFile', 'ShareOnboardingGuide', 'Skill', 'SubagentHandback', 'TaskCreate', 'TaskGet',
    'TaskList', 'TaskOutput', 'TaskStop', 'TaskUpdate', 'TodoWrite', 'ToolSearch', 'WaitForMcpServers',
    'WebFetch', 'WebSearch', 'Workflow', 'Write',
)


@dataclass(frozen=True)
class ParityGroup:
    title: str
    covered: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ratio(self) -> tuple[int, int]:
        return len(self.covered), len(self.covered) + len(self.missing)


@dataclass(frozen=True)
class ParityAuditResult:
    groups: tuple[ParityGroup, ...]

    def group(self, title: str) -> ParityGroup:
        return next(g for g in self.groups if g.title == title)

    def to_markdown(self) -> str:
        lines = ['# Parity Audit', '',
                 f"Reference feature names (checked {DOCS_CHECKED}) that Sigil Code has.", '']
        for group in self.groups:
            have, total = group.ratio
            lines.append(f'{group.title}: **{have}/{total}**')
        for group in self.groups:
            lines.extend(['', f'Missing {group.title.lower()}:'])
            if group.missing:
                lines.extend(f'- {name}' for name in group.missing)
            else:
                lines.append('- none')
        return '\n'.join(lines)


def _split(documented: tuple[str, ...], have: set[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(n for n in documented if n in have), tuple(n for n in documented if n not in have)


def sigil_command_names() -> set[str]:
    """Every name Sigil answers to: slash command names and aliases, the built-in prompt
    commands and the chat loop's own commands."""
    from .agent_slash_commands import get_slash_command_specs
    from .commands import command_names

    names = {name.lstrip('/') for spec in get_slash_command_specs() for name in spec.names}
    return names | set(command_names())


def sigil_hook_events() -> set[str]:
    from .command_hooks import SUPPORTED_EVENTS

    return set(SUPPORTED_EVENTS)


def sigil_tool_names() -> set[str]:
    """Documented tool names Sigil's own tools answer to (permission rules, hook matchers)."""
    from .command_hooks import HOOK_TOOL_ALIASES

    return set(HOOK_TOOL_ALIASES.values())


def run_parity_audit() -> ParityAuditResult:
    groups = []
    for title, documented, have in (
        ('Commands', DOCUMENTED_COMMANDS, sigil_command_names()),
        ('Hook events', DOCUMENTED_HOOK_EVENTS, sigil_hook_events()),
        ('Tools', DOCUMENTED_TOOLS, sigil_tool_names()),
    ):
        covered, missing = _split(documented, have)
        groups.append(ParityGroup(title=title, covered=covered, missing=missing))
    return ParityAuditResult(groups=tuple(groups))
