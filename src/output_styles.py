"""Output styles: a short system-prompt section that changes how answers are written.

Built in: default, concise, explanatory, learning. Custom styles are Markdown
files in .sigil/output-styles/ or ~/.sigil/output-styles/ (frontmatter
`name` and `description`; the body is the instruction). Pick one with
/output-style <name> (saved to .sigil/settings.local.json) or the
`outputStyle` setting.
"""
from __future__ import annotations

from pathlib import Path

from .custom_commands import parse_frontmatter

from .prompt_constants import DEFAULT_OUTPUT_STYLE_NAME, OUTPUT_STYLE_CONFIGS
from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

# Built-in styles are defined once, in prompt_constants.OUTPUT_STYLE_CONFIGS.
BUILT_IN_STYLES = {DEFAULT_OUTPUT_STYLE_NAME: ('Normal answers.', '')}
BUILT_IN_STYLES.update({config.name.lower(): (config.description, config.prompt)
                        for config in OUTPUT_STYLE_CONFIGS.values() if config is not None})


def _home() -> Path:
    return Path.home()


def discover_styles(cwd: Path) -> dict[str, tuple[str, str]]:
    styles = dict(BUILT_IN_STYLES)
    for folder in (Path(cwd) / '.sigil' / 'output-styles', _home() / '.sigil' / 'output-styles'):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.md')):
            try:
                meta, body = parse_frontmatter(path.read_text(encoding='utf-8'))
            except (OSError, UnicodeDecodeError) as _ignored:
                _log_ignored(_ignored)
                continue
            name = str(meta.get('name') or path.stem).strip()
            styles.setdefault(name, (str(meta.get('description') or ''), body.strip()))
    return styles


def style_prompt(cwd: Path, name: str | None) -> str:
    if not name:
        return ''
    styles = discover_styles(cwd)
    for key, (_description, prompt) in styles.items():
        if key.lower() == str(name).strip().lower():
            return prompt
    return ''
