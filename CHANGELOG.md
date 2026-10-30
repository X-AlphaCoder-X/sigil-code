# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). `/release-notes` shows the newest section.

## [1.0.0] - 2026-10-30

First public release.

### Added

**Models and providers**
- Any OpenAI-compatible endpoint: OpenAI, OpenRouter, DeepSeek, Groq, Together, Anthropic's
  compatible endpoint, and local servers (vLLM, Ollama, LM Studio, llama.cpp).
- Effort levels (`/effort`, `/thinking`, Alt+T, `SIGIL_EFFORT_LEVEL`), sent in the field each
  provider reads (`reasoning_effort`, `chat_template_kwargs.enable_thinking`, `thinking`).
- Optional native Anthropic Messages API (`SIGIL_ANTHROPIC_NATIVE=1`) with prompt caching, image
  blocks and thinking blocks kept across tool calls.
- Context window and vision support read from the server; `<think>` blocks removed; tool calls
  written as text (Hermes, Qwen3-Coder, JSON) run as real calls.
- Provider profiles (`--profile`, `/profile`), fallback models (`--fallback-model`), an
  architect/editor split (`--architect`, `--editor-model`), and image input.

**Agent loop and context**
- Tool loop with parallel tool calls and budgets for turns, tokens, cost and tool calls.
- Auto-compaction, microcompaction of bulky tool results, AST outlines of untouched files,
  snipping, and `/context` to see token use.
- Auto-nudge for stalled turns from weak and local models, and a message sanitizer for strict
  providers.
- Structured output with `--response-schema-file`.

**Tools**
- 77 built-in tools: files, shell (with background shells), web fetch and search, subagents,
  background agents, plans, tasks and todos, teams, LSP code intelligence, MCP, skills, worktrees,
  workflows, triggers, config, and `create_new_tool`.
- Opt-in tools: scheduled prompts (`cron_*`, `/loop`), `memory_save`, `session_note`, and
  `editor_edit` in architect mode.

**Subagents and background work**
- Subagent types (`general-purpose`, `Explore`, `Plan`, and agents in `.sigil/agents/` and
  `~/.sigil/agents/`), parallel subagents, `isolation: worktree`, `--agent` and `--agents`.
- Background agents (`sigil agent-bg`, `agent-ps`, `agent-restart`, the agent bar) whose approval
  questions appear in the main chat, and background shells (`run_in_background`, `/bashes`).

**Permissions and folders**
- Start read-only; `--allow-write`, `--allow-shell`, `--unsafe`; permission modes (default,
  acceptEdits, plan, bypass, dontAsk) with Shift+Tab; permission rules (`allow` / `ask` / `deny`).
- Folder access switches (`SIGIL_OTHER_DIRECTORIES`, `SIGIL_ALLOWED_DIRECTORIES`, `--add-dir`) and
  a write-protected list (`SIGIL_PROTECTED_PATHS`).
- Folder trust for project items a cloned repository can carry (see README, Folder trust).
- An opt-in bash sandbox on Linux (bubblewrap) and macOS (sandbox-exec).

**Hooks, MCP and extensions**
- Hooks in `settings.json`: command, http, prompt, agent and MCP-tool hooks for 31 events.
- MCP over stdio, streamable HTTP and SSE: one process per server per session, tools as
  `mcp__server__tool`, prompts as slash commands, elicitation, and OAuth login (`/mcp login`).
- Custom slash commands and skills (`.sigil/commands/`, `.sigil/skills/`), plugin folders
  (`--plugin-dir`), plugin marketplaces with sha / sha256 pins, output styles, custom tools.

**Sessions, memory and CLI**
- Sessions saved per workspace: resume (`-c`, `-r`, `sessions`), names, search, fork, export,
  `sigil purge`, and `/rewind` for the conversation, the files, or both.
- Memory files (`SIGIL.md`, `SIGIL.local.md`, `AGENTS.md`, `.sigil/rules/`), plus opt-in auto
  memory, session memory and a repo map.
- Print mode (`-p`) with text, JSON or stream-JSON output; `sigil doctor`; shell completion;
  91 slash commands; type-while-it-works input; Esc to stop; multiline input; prompt history.
- Windows support without WSL: Git-Bash path translation, `msvcrt` keyboard input, and a
  PowerShell tool.
- An always-on error log (`~/.sigil/logs`, kept 2 days), `--debug` request logging with secrets
  redacted, and opt-in OpenTelemetry logs.

### Security

- Folder trust: hooks, MCP servers, `env`, a `.env`, settings that grant more, commands and skills
  with `allowed-tools`, agent hooks, custom tools in `src/custom_tools/`, plugin manifests,
  pre-set answer files and `@imports` outside the folder only apply after you trust the folder
  once. `deny` and `ask` rules always apply, and a project file never starts bypass mode.
- A pre-set answer from `.sigil-ask-user.json` or `.sigil/ask-user.json` reaches the model
  labelled as a pre-set answer from that file, never as something you typed.
- Credentials are dropped when an HTTP request is redirected to another origin; OAuth endpoints
  must use HTTPS (http only on localhost); the OAuth callback page is not cached and sends no
  referrer.
- The main API key is sent only to its own server; other hosts use `SIGIL_API_KEY_<HOST>`. MCP
  servers get neither Sigil's keys nor, by default, the SSH agent socket.
- Plan mode checks every command on a shell line (and PowerShell commands) before treating it as
  read-only. Permission rules match bash and PowerShell commands alike.
- A plugin source listed by a remote marketplace must use HTTPS or SSH; git runs with protocol
  limits.
- Restarting a background agent rebuilds Sigil's own worker command; a resumed session takes its
  folders and permissions from the command line.
- Escape sequences in model and tool text are removed before they reach the terminal; tool-result
  text is framed as data for the model.
- Every security limit can be changed by you: see the security overrides in `.env.example`.

### Notes

- Requires Python 3.10 or later. Core dependencies: `prompt_toolkit` and `rich`. Optional extras:
  `pdf`, `yaml`, `ast`, `tokenizers`, `dev`, `all`.
