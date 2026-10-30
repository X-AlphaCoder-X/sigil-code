# Contributing to Sigil Code

Thanks for helping out. Bug reports, fixes, new tools, docs and tests are all welcome. This guide covers how to report a problem, how to get a change merged, and what the code expects from you.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- Report a bug with the [bug report form](https://github.com/X-AlphaCoder-X/sigil-code/issues/new?template=bug_report.yml).
- Suggest a feature with the [feature request form](https://github.com/X-AlphaCoder-X/sigil-code/issues/new?template=feature_request.yml). [MISSING_FEATURES.md](MISSING_FEATURES.md) lists what's already planned.
- Fix a bug or build a feature and open a pull request.
- Improve the docs: the README, [docs/WINDOWS.md](docs/WINDOWS.md) or this file.

Found a security problem? Don't open a public issue. Report it privately as described in [SECURITY.md](SECURITY.md).

## Reporting a bug

Search the [open issues](https://github.com/X-AlphaCoder-X/sigil-code/issues) first. If it's new, the bug report form asks for:

- what happened and what you expected,
- the steps to reproduce it, ideally with the smallest prompt or command that shows it,
- your Sigil version (`sigil --version`), OS, Python version, and the model and provider you used,
- the relevant log lines. Errors are always logged to `~/.sigil/logs`; run with `--debug` to also log the request and response bodies.

Remove API keys and anything private from logs before you paste them. Sigil redacts known key formats, but check anyway.

## Setup and tests

You need Python 3.10 or later and git. On Windows, install Git for Windows, since the `bash` tool uses its `bash.exe` (see [docs/WINDOWS.md](docs/WINDOWS.md)).

```bash
git clone https://github.com/X-AlphaCoder-X/sigil-code.git
cd sigil-code
pip install -e ".[dev]"      # or ".[all,dev]" to pull in the optional extras too
python -m pytest tests/
```

The suite has 1,476 tests in 102 files. It runs fully offline against a fake local model server, so you don't need an API key, a GPU or a network connection. The same tests run in CI (`.github/workflows/tests.yml`) on Linux and Windows with Python 3.10 and 3.12, on every pull request.

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
| `src/process_group.py` | Safe stopping of child process groups on Linux and macOS |
| `src/custom_tools/` | Drop-in custom tools |
| `tests/` | Test suite |
| `benchmarks/` | Optional eval harness, not part of the test run |
| `examples/` | Sample `.sigil-mcp.json`, `.sigil-search.json`, `settings.json` |

## Code guidelines

- Match the code around your change: its naming, comment density and structure. Small, readable functions with type hints are the norm.
- Use the standard library where it does the job. Don't add new core dependencies. Right now there are two (`prompt_toolkit`, `rich`); anything optional goes behind an extra in `pyproject.toml`.
- Every fix or feature gets a test, and a bug fix gets a test that fails without the fix. Tests must stay offline: use the fake model server, temporary folders and localhost, never real endpoints or your own files.
- Never swallow an error silently. Log it, or report it to the user plainly.
- Respect the user's settings. Anything that writes files or runs commands must go through the write and shell permissions, plan mode and the permission rules, the same way the existing tools do. A safety limit you add needs a documented env var or setting to change it.
- Remember Windows, Linux and macOS. If you touch the shell, paths, processes or keyboard input, think about Git Bash and `msvcrt` on Windows and process groups on POSIX. CI runs both.
- Keep each file's line endings as they are.

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

## Commits and pull requests

For a large change, such as a new subsystem or a change to how permissions work, open an issue first so we can agree on the approach before you spend time on it. Small fixes can go straight to a pull request.

1. Fork the repo and create a branch from `main` with a short descriptive name, such as `fix-rewind-windows` or `add-tool-word-count`.
2. Make your change, with tests, and run `python -m pytest tests/` until it passes.
3. Write commit messages with a short summary line in the imperative ("Fix rewind on Windows paths"), and a body that says why when it isn't obvious.
4. Open a pull request against `main` and fill in the template. Link the issue it closes, if there is one.
5. CI has to pass on all four jobs. A maintainer will review it, and may ask for changes.

Before you open the pull request, check that:

- each PR is focused on one fix or one feature,
- tests were added or updated and the full suite passes offline,
- user-facing changes are in `README.md` and have a line in `CHANGELOG.md`, and a closed roadmap item is updated in `MISSING_FEATURES.md`,
- no keys, tokens, personal data or local paths are in the diff.

## License

Sigil Code is released under the [GNU AGPL-3.0](LICENSE). By contributing, you agree that your work is licensed under AGPL-3.0 too. Only submit code you wrote yourself or have the right to share under that license. Don't submit proprietary code, including anything copied from Claude Code.
