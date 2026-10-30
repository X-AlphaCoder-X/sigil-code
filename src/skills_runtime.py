"""
Skills and Agents Runtime for Sigil Code.

This module provides auto-triggering of skills and agents based on task context,
similar to how MCP servers are called when needed.

Skills: Markdown files with YAML frontmatter providing workflow procedures, checklists, code patterns.
Agents: Markdown persona files defining specialist roles for delegate_agent.

Three-layer system:
- Rules: Always in context (not handled here)
- Agents: Spawned via delegate_agent with personas loaded into prompts
- Skills: Lazy-loaded when triggered by task context matching descriptions
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

try:
    import yaml
except ImportError:  # PyYAML is optional; fall back to a simple key: value parser
    yaml = None


DEFAULT_SKILL_REPOS: tuple[str, ...] = ()  # opt in with SIGIL_SKILL_REPOS


def _home() -> Path:
    return Path.home()


def _user_skills_disabled() -> bool:
    import os

    return os.environ.get('SIGIL_DISABLE_USER_SKILLS', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _skills_within_budget(descriptions: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """As many skills as fit in about 1% of the context window (SIGIL_SKILL_LIST_TOKENS)."""
    import os

    try:
        budget = int(os.environ.get('SIGIL_SKILL_LIST_TOKENS', '') or 0)
    except ValueError as _ignored:
        _log_ignored(_ignored)
        budget = 0
    if budget <= 0:
        try:
            window = int(os.environ.get('SIGIL_CONTEXT_WINDOW', '') or 128000)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            window = 128000
        budget = max(200, window // 100)
    chars = budget * 4
    picked: list[tuple[str, str]] = []
    used = 0
    for name, description in descriptions:
        cost = len(name) + min(len(description), 150) + 8
        if used + cost > chars:
            break
        picked.append((name, description))
        used += cost
    return picked


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {'true', 'yes', '1', 'on'}


def _falsy(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {'false', 'no', '0', 'off'}


def _tool_names(value: Any) -> tuple[str, ...]:
    from .custom_commands import tool_list

    return tool_list(value)


_SKIP_SCAN_DIRS = frozenset({'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.port_sessions', '.tox'})


def _glob_regex(pattern: str) -> 're.Pattern[str]':
    import re

    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith('**/', i):
            out.append('(?:.*/)?')
            i += 3
        elif pattern.startswith('**', i):
            out.append('.*')
            i += 2
        elif pattern[i] == '*':
            out.append('[^/]*')
            i += 1
        elif pattern[i] == '?':
            out.append('[^/]')
            i += 1
        elif pattern[i] == '[' and ']' in pattern[i + 1:]:
            end = pattern.index(']', i + 1)
            body = pattern[i + 1:end]
            negate = body.startswith('!')
            body = (body[1:] if negate else body).replace('\\', '\\\\').replace('[', '\\[')
            if body.startswith('^'):  # literal in a glob class (fnmatch does the same)
                body = '\\' + body
            out.append('[' + ('^' if negate else '') + (body or '\\]') + ']')
            i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile(''.join(out) + r'\Z')


def _any_match(cwd: Path, pattern: str, limit: int | None = None) -> bool:
    """Whether any file or folder under cwd matches the glob (relative to cwd). Looks at no more
    than SIGIL_SKILL_PATHS_SCAN_LIMIT entries (default 20000; 0 = no limit)."""
    import os

    if limit is None:
        try:
            limit = int(os.environ.get('SIGIL_SKILL_PATHS_SCAN_LIMIT', '') or 20000)
        except ValueError as _ignored:
            _log_ignored(_ignored)
            limit = 20000
    pattern = pattern.strip().lstrip('/').replace('\\', '/')
    if not pattern:
        return False
    try:
        regex = _glob_regex(pattern)
    except re.error:
        from .debug_log import get_logger, sanitize_log

        get_logger('skills').warning('skill paths pattern %s is not a valid glob', sanitize_log(pattern))
        return False
    # start below the pattern's fixed folders (src/*.py walks src/ only, not the whole tree)
    fixed: list[str] = []
    for part in pattern.split('/')[:-1]:
        if any(ch in part for ch in '*?['):
            break
        fixed.append(part)
    start = Path(cwd).joinpath(*fixed) if fixed else Path(cwd)
    if not start.is_dir():
        return False
    seen = 0
    try:
        for folder, dirs, files in os.walk(start):
            dirs[:] = sorted(d for d in dirs if d not in _SKIP_SCAN_DIRS)
            rel_folder = Path(folder).relative_to(cwd).as_posix()
            prefix = '' if rel_folder == '.' else rel_folder + '/'
            for name in (*dirs, *sorted(files)):
                seen += 1
                if limit and seen > limit:
                    return False
                if regex.match(prefix + name):
                    return True
    except (OSError, ValueError) as _ignored:
        _log_ignored(_ignored)
        return False
    return False


def _skill_repo_names() -> tuple[str, ...]:
    """Folder names scanned as skills repos (SIGIL_SKILL_REPOS overrides the default)."""
    import os

    raw = os.environ.get('SIGIL_SKILL_REPOS')
    if raw is None:
        return DEFAULT_SKILL_REPOS
    return tuple(part.strip() for part in raw.split(os.pathsep) if part.strip())


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata extracted from a SKILL.md YAML frontmatter."""
    name: str
    description: str
    origin: str = ''
    version: str = ''
    author: str = ''
    tags: list[str] = field(default_factory=list)
    context: str = ''
    compatible_tools: list[str] = field(default_factory=list)
    # skill frontmatter
    disable_model_invocation: bool = False  # only the user runs it (/name); hidden from the model
    user_invocable: bool = True  # False: the model may load it, but it is not a /command
    allowed_tools: list[str] = field(default_factory=list)  # granted while the skill's turn runs
    paths: list[str] = field(default_factory=list)  # globs: offered only when such files exist
    agent: str = ''  # with context: fork, the subagent type to run it in


@dataclass(frozen=True)
class AgentMetadata:
    """Metadata extracted from an agent MD file YAML frontmatter."""
    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    model: str = 'sonnet'
    origin: str = ''


@dataclass(frozen=True)
class SkillEntry:
    """A fully loaded skill with metadata and content."""
    metadata: SkillMetadata
    content: str
    file_path: Path


@dataclass(frozen=True)
class AgentEntry:
    """A fully loaded agent with metadata and content."""
    metadata: AgentMetadata
    content: str
    file_path: Path


class SkillsRuntime:
    """
    Runtime for skills and agents discovery and context injection.
    
    Follows the same pattern as other runtimes (MCPRuntime, SearchRuntime, etc.)
    with from_workspace() and render_summary() methods.
    """
    
    def __init__(self, registry: SkillsRegistry | None = None):
        self._registry = registry or SkillsRegistry()
    
    @classmethod
    def from_workspace(
        cls,
        cwd: Path,
        additional_working_directories: tuple[str, ...] = (),
    ) -> 'SkillsRuntime':
        """
        Discover skills and agents from workspace directories.
        
        Looks for:
        - skills repos named in SIGIL_SKILL_REPOS (skills + agents)
        - .sigil/skills/ local directory
        """
        registry = SkillsRegistry()
        import os

        if os.environ.get('SIGIL_SAFE_MODE') == '1':
            return cls(registry)  # --safe-mode: no skills or agents
        
        # Skills directories
        skills_dirs: list[Path] = []
        agents_dirs: list[Path] = []
        
        # Skills repos checked out in the workspace. The folder names are a
        # setting: SIGIL_SKILL_REPOS (os.pathsep list); each repo's agents/
        # folder is read too.
        repo_names = _skill_repo_names()
        for repo_name in repo_names:
            repo = cwd / repo_name
            if repo.exists():
                skills_dirs.append(repo)
                if (repo / 'agents').is_dir():
                    agents_dirs.append(repo / 'agents')

        # Check for local skills directory (and .agents/skills, shared by other agent CLIs)
        local_skills = cwd / '.sigil' / 'skills'
        if local_skills.exists():
            skills_dirs.append(local_skills)
        if (cwd / '.agents' / 'skills').is_dir():
            skills_dirs.append(cwd / '.agents' / 'skills')
        
        # Check additional working directories
        for raw_path in additional_working_directories:
            path = Path(raw_path)
            for repo_name in repo_names:
                repo = path / repo_name
                if repo.exists():
                    skills_dirs.append(repo)
                    if (repo / 'agents').is_dir():
                        agents_dirs.append(repo / 'agents')

            local_skills = path / '.sigil' / 'skills'
            if local_skills.exists():
                skills_dirs.append(local_skills)
        
        # Parent folders' .sigil/skills (monorepos), then personal ~/.sigil/skills.
        home = _home().resolve()
        for parent in cwd.resolve().parents:
            if parent == home:
                break  # home is covered by the personal folder below
            candidate = parent / '.sigil' / 'skills'
            if candidate.is_dir() and candidate not in skills_dirs:
                skills_dirs.append(candidate)
        user_skills = home / '.sigil' / 'skills'
        if not _user_skills_disabled() and user_skills.is_dir() and user_skills not in skills_dirs:
            skills_dirs.append(user_skills)
        from .sigil_plugins import loaded_plugins

        for plugin in loaded_plugins():  # --plugin-dir
            skills_dirs.extend(folder for folder in plugin.skill_dirs if folder not in skills_dirs)

        # Discover skills and agents
        registry.discover_skills(skills_dirs)
        registry.discover_agents(agents_dirs)
        registry.filter_by_paths(cwd)
        
        return cls(registry)
    
    def has_skills_or_agents(self) -> bool:
        """Check if any skills or agents are registered."""
        return bool(self._registry._skills or self._registry._agents)
    
    def render_summary(self) -> dict[str, Any]:
        """Render a summary for context injection."""
        return self._registry.get_registry_summary()
    
    @property
    def registry(self) -> SkillsRegistry:
        """Access the underlying registry."""
        return self._registry
    
    # Convenience delegation methods for tool handlers
    
    def match_skills(self, task_context: str, threshold: float = 0.3) -> list[str]:
        """Match skills based on task context. Delegates to registry."""
        return self._registry.match_skills(task_context, threshold)
    
    def match_agents(self, task_context: str, threshold: float = 0.3) -> list[str]:
        """Match agents based on task context. Delegates to registry."""
        return self._registry.match_agents(task_context, threshold)
    
    def get_skill(self, name: str) -> SkillEntry | None:
        """Get a skill by name. Delegates to registry."""
        return self._registry.get_skill(name)
    
    def get_agent(self, name: str) -> AgentEntry | None:
        """Get an agent by name. Delegates to registry."""
        return self._registry.get_agent(name)
    
    def list_skills(self) -> list[str]:
        """List all skill names. Delegates to registry."""
        return self._registry.list_skills()
    
    def list_agents(self) -> list[str]:
        """List all agent names. Delegates to registry."""
        return self._registry.list_agents()


class SkillsRegistry:
    """
    Registry for skills and agents with auto-triggering support.
    
    Skills are indexed by name and description for context matching.
    Full content is loaded lazily when triggered.
    """
    
    def __init__(self):
        self._skills: dict[str, SkillEntry] = {}
        self._agents: dict[str, AgentEntry] = {}
        self._skill_descriptions: list[tuple[str, str]] = []  # (name, description) for matching
        self._agent_descriptions: list[tuple[str, str]] = []  # (name, description) for matching
        self._loaded = False
    
    def discover_skills(self, skills_dirs: list[Path]) -> int:
        """
        Discover all SKILL.md files in the given directories.
        
        Returns the number of skills discovered.
        """
        count = 0
        for skills_dir in skills_dirs:
            if not skills_dir.exists():
                continue
            for skill_file in skills_dir.rglob('SKILL.md'):
                try:
                    entry = self._load_skill(skill_file)
                    if entry:
                        name = entry.metadata.name
                        # Remove old description entry if name already exists (dedup)
                        if name in self._skills:
                            self._skill_descriptions = [
                                (n, d) for n, d in self._skill_descriptions if n != name
                            ]
                        self._skills[name] = entry
                        if not entry.metadata.disable_model_invocation:
                            self._skill_descriptions.append(
                                (name, entry.metadata.description)
                            )
                        count += 1
                except Exception as e:
                    # Log but continue
                    import logging
                    logging.getLogger('sigil.skills').warning(
                        f'Failed to load skill {skill_file}: {e}'
                    )
        return count
    
    def discover_agents(self, agents_dirs: list[Path]) -> int:
        """
        Discover all agent MD files in the given directories.
        
        Returns the number of agents discovered.
        """
        count = 0
        for agents_dir in agents_dirs:
            if not agents_dir.exists():
                continue
            for agent_file in agents_dir.glob('*.md'):
                try:
                    entry = self._load_agent(agent_file)
                    if entry:
                        name = entry.metadata.name
                        # Remove old description entry if name already exists (dedup)
                        if name in self._agents:
                            self._agent_descriptions = [
                                (n, d) for n, d in self._agent_descriptions if n != name
                            ]
                        self._agents[name] = entry
                        self._agent_descriptions.append(
                            (name, entry.metadata.description)
                        )
                        count += 1
                except Exception as e:
                    import logging
                    logging.getLogger('sigil.skills').warning(
                        f'Failed to load agent {agent_file}: {e}'
                    )
        return count
    
    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from markdown content."""
        if not content.startswith('---'):
            return {}, content
        
        # Find the closing ---
        end_match = re.search(r'\n---\s*\n', content[3:])
        if not end_match:
            return {}, content
        
        frontmatter_str = content[3:end_match.start() + 3]
        body = content[end_match.end() + 3:]
        
        if yaml is None:
            from .custom_commands import _fallback_frontmatter

            frontmatter = _fallback_frontmatter(frontmatter_str)
        else:
            try:
                frontmatter = yaml.safe_load(frontmatter_str) or {}
            except yaml.YAMLError as _ignored:
                _log_ignored(_ignored)
                frontmatter = {}
        
        return frontmatter, body
    
    def _load_skill(self, file_path: Path) -> SkillEntry | None:
        """Load a skill from a SKILL.md file."""
        content = file_path.read_text(encoding='utf-8')
        frontmatter, body = self._parse_frontmatter(content)
        
        name = frontmatter.get('name', file_path.parent.name)
        description = str(frontmatter.get('description', '') or '')
        if not description and file_path.name == 'SKILL.md':
            # description is recommended, not required; without it the first
            # paragraph of the body is used.
            description = next(
                (line.strip().lstrip('#').strip() for line in body.splitlines() if line.strip().lstrip('#').strip()),
                '',
            )
        
        if not description:
            return None
        
        metadata = SkillMetadata(
            name=name,
            description=description,
            origin=frontmatter.get('origin', ''),
            version=str(frontmatter.get('version', '')),
            author=frontmatter.get('author', ''),
            tags=frontmatter.get('tags', []),
            context=str(frontmatter.get('context', '') or ''),
            compatible_tools=frontmatter.get('compatible_tools', []),
            disable_model_invocation=_truthy(frontmatter.get('disable-model-invocation')),
            user_invocable=not _falsy(frontmatter.get('user-invocable')),
            allowed_tools=list(_tool_names(frontmatter.get('allowed-tools'))),
            paths=list(_tool_names(frontmatter.get('paths'))),
            agent=str(frontmatter.get('agent', '') or ''),
        )
        
        return SkillEntry(
            metadata=metadata,
            content=body.strip(),
            file_path=file_path,
        )
    
    def _load_agent(self, file_path: Path) -> AgentEntry | None:
        """Load an agent from an MD file."""
        content = file_path.read_text(encoding='utf-8')
        frontmatter, body = self._parse_frontmatter(content)
        
        name = frontmatter.get('name', file_path.stem)
        description = frontmatter.get('description', '')
        
        if not description:
            return None
        
        metadata = AgentMetadata(
            name=name,
            description=description,
            tools=frontmatter.get('tools', []),
            model=frontmatter.get('model', 'sonnet'),
            origin=frontmatter.get('origin', ''),
        )
        
        return AgentEntry(
            metadata=metadata,
            content=body.strip(),
            file_path=file_path,
        )
    
    def get_skill(self, name: str) -> SkillEntry | None:
        """Get a skill by name."""
        return self._skills.get(name)
    
    def get_agent(self, name: str) -> AgentEntry | None:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def list_skills(self) -> list[str]:
        """List all skill names."""
        return list(self._skills.keys())
    
    def list_agents(self) -> list[str]:
        """List all agent names."""
        return list(self._agents.keys())
    
    def filter_by_paths(self, cwd: Path) -> None:
        """Skills with `paths:` are offered to the model only when such files exist."""
        kept = []
        for name, description in self._skill_descriptions:
            entry = self._skills.get(name)
            patterns = entry.metadata.paths if entry is not None else []
            if not patterns or any(_any_match(cwd, pattern) for pattern in patterns):
                kept.append((name, description))
        self._skill_descriptions = kept

    def get_skill_descriptions(self) -> list[tuple[str, str]]:
        """Get all (name, description) pairs for skills."""
        return self._skill_descriptions.copy()
    
    def get_agent_descriptions(self) -> list[tuple[str, str]]:
        """Get all (name, description) pairs for agents."""
        return self._agent_descriptions.copy()
    
    def match_skills(self, task_context: str, threshold: float = 0.3) -> list[str]:
        """
        Match skills based on task context.
        
        Returns a list of skill names that match the context.
        Simple keyword matching for now; can be enhanced with embeddings.
        """
        task_lower = task_context.lower()
        task_words = set(re.findall(r'\b\w+\b', task_lower))
        
        matched = []
        for name, description in self._skill_descriptions:
            desc_lower = description.lower()
            desc_words = set(re.findall(r'\b\w+\b', desc_lower))
            
            # Simple overlap score
            overlap = len(task_words & desc_words)
            if overlap > 0:
                score = overlap / max(len(desc_words), 1)
                if score >= threshold or overlap >= 3:
                    matched.append(name)
        
        return matched
    
    def match_agents(self, task_context: str, threshold: float = 0.3) -> list[str]:
        """
        Match agents based on task context.
        
        Returns a list of agent names that match the context.
        """
        task_lower = task_context.lower()
        task_words = set(re.findall(r'\b\w+\b', task_lower))
        
        matched = []
        for name, description in self._agent_descriptions:
            desc_lower = description.lower()
            desc_words = set(re.findall(r'\b\w+\b', desc_lower))
            
            overlap = len(task_words & desc_words)
            if overlap > 0:
                score = overlap / max(len(desc_words), 1)
                if score >= threshold or overlap >= 3:
                    matched.append(name)
        
        return matched
    
    def get_registry_summary(self) -> dict[str, Any]:
        """Get a summary of the registry for runtime context."""
        return {
            'skills_count': len(self._skills),
            'agents_count': len(self._agents),
            'skills_sample': _skills_within_budget(self._skill_descriptions),
            'agents_sample': self._agent_descriptions[:20],
        }


# Global registry instance
_global_registry: SkillsRegistry | None = None


def get_skills_registry() -> SkillsRegistry:
    """Get or create the global skills registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillsRegistry()
    return _global_registry


def initialize_skills_registry(
    skills_dirs: list[Path] | None = None,
    agents_dirs: list[Path] | None = None,
) -> SkillsRegistry:
    """
    Initialize the global skills registry with the given directories.
    
    Default directories are relative to the workspace root.
    """
    global _global_registry
    
    registry = SkillsRegistry()
    
    if skills_dirs:
        registry.discover_skills(skills_dirs)
    
    if agents_dirs:
        registry.discover_agents(agents_dirs)
    
    _global_registry = registry
    return registry


def format_skills_for_prompt(registry: SkillsRegistry, max_skills: int = 50) -> str:
    """
    Format skills registry for inclusion in system prompt.
    
    Returns a markdown section describing available skills and agents.
    """
    skills = registry.get_skill_descriptions()[:max_skills]
    agents = registry.get_agent_descriptions()[:max_skills]
    
    lines = ['# Skills & Agents', '']
    
    if skills:
        lines.append('## Available Skills')
        lines.append('Skills are workflow procedures, checklists, and code patterns that are auto-loaded when task context matches.')
        lines.append('')
        for name, desc in skills:
            # Truncate long descriptions
            short_desc = desc[:150] + '...' if len(desc) > 150 else desc
            lines.append(f'- **{name}**: {short_desc}')
        lines.append('')
    
    if agents:
        lines.append('## Available Agents')
        lines.append('Agents are specialist personas that can be spawned via delegate_agent for focused tasks.')
        lines.append('')
        for name, desc in agents:
            short_desc = desc[:150] + '...' if len(desc) > 150 else desc
            lines.append(f'- **{name}**: {short_desc}')
        lines.append('')
    
    lines.append('When a task matches a skill or agent description, consider using it for specialized guidance.')
    
    return '\n'.join(lines)


def format_agent_persona(agent: AgentEntry) -> str:
    """
    Format an agent's content as a persona prompt for delegate_agent.
    """
    return f"""You are {agent.metadata.name}, a specialist agent.

{agent.content}

Your available tools: {', '.join(agent.metadata.tools) if agent.metadata.tools else 'all default tools'}
Recommended model: {agent.metadata.model}
"""
