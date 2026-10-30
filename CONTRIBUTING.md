# Contributing to Sigil Code

Thanks for helping out. This file covers what you need to get a change merged.

## Setup and tests

```bash
git clone https://github.com/X-AlphaCoder-X/sigil-code.git
cd sigil-code
pip install -e ".[dev]"      # or ".[all,dev]" to pull in the optional extras too
python -m pytest tests/
```

The suite has 1,476 tests in 102 files. It runs fully offline against a fake local model server, so you don't need an API key, a GPU or a network connection. The same tests run in CI (`.github/workflows/tests.yml`).

To try your change by hand without reinstalling:

```bash
python -m src.main agent-chat --allow-write --allow-shell
```

## Project layout

| Path | What's there |
|---|---|
| `src/main.py` | CLI entry point (`sigil` / `sigilcode`) |
| `src/agent_runtime.py` | The agent loop |
| `src/agent_tools.py` | Built-in tools and the tool registry |
| `src/agent_slash_commands.py` | Slash command parsing and handlers |
| `src/agent_prompting.py` | System prompt |
| `src/openai_compat.py` | OpenAI-compatible client |
| `src/compact.py`, `src/microcompact.py` | Context management |
| `src/bash_security.py` | Shell command guard |
| `src/checkpoints.py`, `src/rewind.py` | Per-prompt checkpoints and `/rewind` |
| `src/command_hooks.py` | `settings.json` hooks |
| `src/mcp_runtime.py` | MCP client |
| `src/custom_tools/` | Drop-in custom tools |
| `tests/` | Test suite |
| `benchmarks/` | Optional eval harness, not part of the test run |
| `examples/` | Sample `.sigil-mcp.json`, `.sigil-search.json`, `settings.json` |

## Adding a tool

**Built-in tool.** Add a `SigilTool` entry to the `tools` list in `default_tool_registry() -> dict[str, SigilTool]` in `src/agent_tools.py`:

```python
SigilTool(
    name='my_tool',
    description='One sentence the model will read to decide when to call this.',
    parameters={
        'type': 'object',
        'properties': {
            'path': {'type': 'string', 'description': 'Relative path from workspace root.'},
        },
        'required': ['path'],
    },
    handler=_my_tool,
),
```

The handler takes `(arguments, context)` and returns the result text, or `(text, metadata)`. Keep `parameters` a JSON schema with `'type': 'object'`, because strict providers reject anything else. If the tool writes files or runs commands, make sure it respects the write/shell permissions and plan mode, the same way the existing tools do.

**Custom tool (no core changes).** Put a Python file in `src/custom_tools/`. The file name becomes the tool name. You can optionally set module-level `name`, `description` and `parameters`. The handler is the `handler` function if there is one, and otherwise the first public callable in the file. A tool in a repository's `src/custom_tools/` is imported (and so runs) only after you trust the folder (see Folder trust in the README); your own `~/.sigil/custom_tools/` always load. A custom tool may not take a built-in tool's name or shadow one of your own tools; such a file is refused with a note on stderr:

```python
# src/custom_tools/word_count.py
description = 'Count words in a string.'
parameters = {
    'type': 'object',
    'properties': {'text': {'type': 'string'}},
    'required': ['text'],
}

def handler(arguments, context):
    return str(len(arguments.get('text', '').split()))
```

## Adding a slash command

Add a `SlashCommandSpec` to the tuple returned by `get_slash_command_specs()` in `src/agent_slash_commands.py`:

```python
SlashCommandSpec(
    names=('mycmd', 'mc'),          # first name is the main one, the rest are aliases
    description='What /mycmd does.',
    handler=_handle_mycmd,
),
```

The handler gets `(agent, args, input_text)` and returns a `SlashCommandResult`. Look at the existing handlers in that file for the usual patterns (local output only vs. sending a prompt to the model). `/help` and tab completion read from the same list, so new commands show up there automatically.

## Pull requests

- Keep each PR focused on one fix or one feature.
- Add or update tests in `tests/`, and run `python -m pytest tests/` before you push.
- Tests must stay offline. Use the fake model server, not real endpoints.
- Don't add new core dependencies. Right now there are two (`prompt_toolkit`, `rich`). If something is optional, put it behind an extra in `pyproject.toml`.
- Remember Windows. If you touch shell, paths or keyboard input, think about Git-Bash and `msvcrt` too.
- If your change affects users, update `README.md` and add a line to `CHANGELOG.md`. If it closes a roadmap item, update `MISSING_FEATURES.md`.
- Don't submit proprietary code, including anything copied from Claude Code. By contributing, you agree your work is licensed under AGPL-3.0.
