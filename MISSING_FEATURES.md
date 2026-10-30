# Roadmap

What Sigil Code doesn't do yet, and what it won't do. Ideas and bug reports are welcome as
[GitHub issues](https://github.com/X-AlphaCoder-X/sigil-code/issues).

## Open

### Sandbox
- A bash sandbox on native Windows. Linux (bubblewrap) and macOS (sandbox-exec) isolate files and
  network; on Windows `SIGIL_SANDBOX=1` only sets Job Object limits (memory, processes). Use WSL2
  for full isolation.

### MCP
- The stateless HTTP transport from the 2026-07-28 MCP revision (stdio, streamable HTTP and SSE work).
- OAuth Client ID Metadata Documents. Dynamic client registration and a pre-set `client_id` work.

### Skills
- `paths:` in a skill's frontmatter loads the skill when matching files exist in the workspace,
  not only while the agent works on those files.

### Sessions
- Asking whether to keep or remove a `-w` worktree when the session ends. It is kept today;
  `sigil worktree-exit --action remove` removes it.

### Packaging
- A PyPI package. The top-level package is named `src`, which clashes with other projects, so it
  needs a new name first. Install from a clone with `pip install -e .` until then.

## Not planned

- Features that need a hosted service or an Anthropic account (subscription usage pages, cloud and
  remote sessions, hosted apps, artifacts, the browser extension, voice, server-side tools, fast
  mode, sending feedback to a vendor, managed fleet settings). Sigil is a local tool that uses API
  keys.
- A model that approves or denies actions on the user's behalf. Your permission rules and modes
  decide.
- Prompt suggestions or tips that cost an extra model call every turn. They're slow on local models.
- IDE diff views and a fullscreen, mouse-driven interface. Sigil keeps your terminal scrollback,
  and the `LSP` tool gives code intelligence.
- tmux or iTerm2 panes for agent teams. Neither runs on native Windows.
