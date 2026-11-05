# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). `/release-notes` shows the newest section.

## [1.1.0] - 2026-11-05

### Added

- Prompt caching for every model call, on by default. Sigil picks the method from the base URL
  and model: `cache_control` breakpoints (Anthropic, OpenRouter Claude and Gemini, DashScope,
  NanoGPT Claude), `prompt_cache_key` (OpenAI, Mistral, Together, Cerebras, DeepInfra, Kimi, and
  servers with no documented scheme), session headers (xAI, Fireworks, OpenRouter), or nothing
  where the provider caches on its own. `SIGIL_PROMPT_CACHE` (`auto`, `on`, `off`),
  `SIGIL_PROMPT_CACHE_TTL` (`1h` by default, `5m`, `24h`) and the `promptCache` setting choose the
  mode and lifetime for the main agent and all subagents. OpenAI's `prompt_cache_retention` is sent
  only to the models OpenAI lists for it. See "Prompt caching" in the README.
- `SIGIL_PROMPT_CACHE_PROVIDER` names the provider whose cache rules and prices apply to a server
  Sigil does not recognise by its host, such as a local server on another port, a proxy or a
  custom domain. A known host keeps its own rules. An unknown name is reported once and ignored.
- Explicit context caches on the Gemini API and Vertex AI. When the published prices say it saves
  money over Google's implicit caching, measured from recent replies, Sigil puts the system prompt
  and tool list in a `cachedContents` resource, renews it while it is in use, and deletes it at
  exit and at the end of a session. Caches left by a crashed run are deleted in the background at
  the next start-up and at the first use of an endpoint, and only caches named with this install's
  id are touched. A failed create, or a request that names a dead cache, goes out without the
  cache. Only an error about the cache itself (403, 404, or a 400 that names it) drops it.
  `SIGIL_EXPLICIT_CACHE` and `SIGIL_EXPLICIT_CACHE_MAX` control it, and `/cache list`,
  `/cache delete` and `/cache explicit` manage it. Storage cost appears in `/cost`, `/stats`, the
  status line and OpenTelemetry, counts toward `--max-budget-usd` and background agents' cost, and
  is saved with the session.
- Vertex AI access tokens from `SIGIL_GOOGLE_TOKEN_COMMAND` (for example
  `gcloud auth print-access-token`), fetched again before they expire. The token used to be read
  once from the key setting and stopped working after an hour.
- Amazon Bedrock prompt caching. Claude and Nova models on a bedrock-runtime base URL, FIPS
  endpoints (`bedrock-runtime-fips.<region>.amazonaws.com`) included, use the Converse API with
  `cachePoint` blocks on tools, system and messages, with 1h for Claude when the TTL is 1h. Nova
  caches at most 20,000 tokens, so no Nova cache point is placed past that. Replies stream over
  Bedrock's event stream and thinking blocks are kept for tool loops. The effort level reaches
  Nova 2 models as `reasoningConfig`, and a JSON schema goes as `outputConfig` structured output.
  A model that answers 400 to it gets the schema as an instruction instead. Auth is a Bedrock API
  key (`AWS_BEARER_TOKEN_BEDROCK`) or AWS Signature Version 4 from the standard AWS variables or
  `~/.aws/credentials`, using only the standard library. `SIGIL_BEDROCK_CONVERSE=0` keeps the
  OpenAI-compatible endpoint.
- Cache usage is read from each provider's own fields, among them DeepSeek's
  `prompt_cache_hit_tokens`, DashScope cache writes, Together's top-level `cached_tokens`,
  SambaNova's `cache_creation_tokens`, and Fireworks' `fireworks-cached-prompt-tokens` header and
  `perf_metrics` in streamed replies.
- `/cache` shows the hit rate and cache tokens and sets the mode or TTL for the session.
  `/cost`, `/usage`, `/stats`, the status line JSON and OpenTelemetry records include cache reads,
  writes, 1-hour writes and the hit rate. Sessions store 1-hour writes, and older sessions still
  load.
- Cache reads and 5-minute and 1-hour writes are priced with each provider's published ratio to
  the input price: for example reads at 10% on Gemini, Vertex AI and SambaNova and 25% on Bedrock
  Nova, and Nova writes at no charge. DeepSeek and Z.ai reads use the cache-hit prices on their
  price pages. Gemini 3.6 and 3.8 Flash use the price Google lists for the day of the request (it
  doubles on 2027-01-01). `--cache-read-cost-per-million`, `--cache-write-cost-per-million` and
  `--cache-write-1h-cost-per-million` override these prices and are passed to background agents.
- Model prices without flags. Sigil reads them from the server's model list where the server
  publishes one (OpenRouter, NanoGPT), or from your own table in the `modelPrices` setting or
  `SIGIL_MODEL_PRICES` (USD per million tokens: `input`, `output`, `cache_read`, `cache_write`,
  `cache_write_1h`). Prices are looked up again after `/model`, `/profile`, a fallback model, a
  subagent or background model, and `agent-resume --model` / `--base-url`, so each model is priced
  as itself. A price set with a flag is kept.
- `/cost` and `/stats` have a "Side calls" line, and the status line JSON has
  `cost.side_cost_usd`. Side calls are model calls made outside the main loop, and their cost is
  already part of the totals.
- `background_agent_status` shows a finished background agent's input and output tokens and its
  cache reads and writes next to its cost.
- Hooks in the settings files are re-read on the next hook call after an edit. If a settings file
  can't be read, the last good hooks stay in use. `SIGIL_HOOKS_RELOAD=0` turns this off.
- Background agents can run on a different OpenAI-compatible endpoint from the main session. Pass
  `model: "name@base_url"` or a separate `base_url`. The key comes from `SIGIL_API_KEY_<HOST>`
  (for example `SIGIL_API_KEY_NANO_GPT_COM`). The main API key is never sent to another host, and
  `background_agent_start` prints the endpoint it used. A `base_url` must start with `http://` or
  `https://`, and a model id that contains `@` but no URL is passed on unchanged.
- `/mcp disable` and `/mcp enable` take effect from the next message, with no restart. A large server
  stops costing its whole tool list at once, and the prompt cache is written again once. A refresh
  that changes nothing sends the same tools in the same order, so the cache stays on every provider.
  A server name that isn't configured, or one already in that state, says so and changes nothing.
  Each change is logged with the server name and tool count only.
- `/loop <prompt>` with no interval runs the prompt now and lets the agent pick when each next run
  happens, from 1 minute to 1 hour, with the reason shown. The agent ends the loop with
  `schedule_wakeup` when the task is done. A run that picks nothing gets one fallback run 20 minutes
  later, and the loop ends if that run picks nothing too. `/loop every 10 minutes <prompt>` works as
  well as `/loop 10m <prompt>`.
- Scheduled prompts are saved with their session and come back when it is resumed. Recurring prompts
  older than 7 days, one-time prompts whose time passed while Sigil was closed, and a self-paced loop
  whose next run came due while it was closed are not brought back, and Sigil says which ones and
  why. `SIGIL_LOOP_RESTORE=0` restores nothing, and `SIGIL_DISABLE_CRON=1` turns scheduling off.
- Scheduled prompts work with the prompt cache. A run due just as the cache would expire, like a
  `/loop 5m` on a 5-minute cache, starts 15 seconds early so it still reads the cache. The cache
  lifetime is worked out for the model in use at each run, so `/model` and `/profile` are followed.
  The agent is told how long the cache lasts after it picks a time, and the tool list never changes
  while a loop runs.
- Recurring prompts and loops end after 7 days, with one final run. A run that came due several
  times while the agent was busy runs once.

### Changed

- With an anthropic.com base URL, Sigil now uses the native Messages API by default because the
  OpenAI-compatible endpoint does not cache prompts. `SIGIL_ANTHROPIC_NATIVE=0` keeps the old
  behaviour. The native client now streams over server-sent events, and `SIGIL_ANTHROPIC_STREAM=0`
  waits for the whole reply.
- When a request gets HTTP 400 or 422 and the error names a cache field it carried, or is an
  unknown-field error that names no other field, Sigil sends it once more without the cache
  fields and sends that endpoint no cache fields for the rest of the session. Any other 400 or 422
  is shown at once and not resent. Both cases are logged.
- `/btw`, compaction and rewind summaries send the main loop's own tools, system prompt and
  messages plus one request message, so they read the conversation's prompt cache and don't write
  a new entry. `tool_choice` stays as it is, since changing it would invalidate the cached
  messages on Anthropic. The model is told not to call tools, a tool call in the reply is never
  run, and a reply without text is asked for once more before Sigil falls back to the plain-text
  transcript.
- On servers that take `cache_control` markers in the OpenAI format, user, tool and system text is
  always sent as one text part, so a message keeps the same bytes after its breakpoint moves on.
- Microcompaction moved one old tool result per turn, which rewrote the cached prompt every turn.
  While the cache is warm it now moves them in groups of `SIGIL_MICROCOMPACT_BATCH` (default 3).
- MCP tools are listed in a fixed order (server, then name), so the tool list at the start of
  every prompt stays the same whatever order the servers answer in.
- Each model call is priced when it is made, at the prices of the endpoint that answered. The cost
  used to be recomputed every turn from the session's total tokens at the current prices.
- `read_file` with `allow_large` returns at most 240,000 characters (about 60k tokens), down
  from 500,000. One full read of a large file could fill most of the context window.
- The output one bash command keeps is still 10 times the tool output limit, but now stops at
  240,000 characters when that limit is raised. `SIGIL_BASH_STREAM_MAX_CHARS` still overrides it.
- Sessions now record how they were started: in a chat, with `-p`, or as a background agent.
  `/resume`, `sigil sessions` and a chat's `-c` / `-r` list only your chat sessions, so a quick
  `-p` test no longer takes the place of the conversation you were working in. The list says how
  many it hid, and `/resume all` or `sigil sessions --all` shows them. A text search
  (`/resume <text>`, `sigil sessions --search`) still looks through every session. Sessions saved
  before this change count as yours. A session file that can't be read stays in the list and the error is logged.

### Fixed

- On Windows, a paste of several lines made while the agent is working stays one message with its
  line breaks, and Enter sends it. It used to be sent as one message per line. A long paste also
  shows up at once instead of one character at a time.
- Text typed or pasted as a turn ends no longer starts a new turn by itself. Unsent text, the rest
  of a paste that was still arriving, and messages queued too late for the turn wait in the input
  box until you press Enter, and no characters are lost when a turn ends in the middle of a paste.
  Ctrl+Enter still sends queued messages at once. Before, the start of such a paste was lost and
  the rest was sent one line per turn.
- Resuming a session printed the whole system prompt, the hidden session context and the runtime's own
  notes as if you had typed them. The earlier conversation now shows only what you typed and what the
  agent answered.
- A comment after a value in a `.env` file (`SIGIL_TRUST_ALL=1  # note`) was read as part of the
  value, so the setting did nothing. The three `.env` readers now share one parser that follows
  python-dotenv's rules. A `#` after a space starts a comment, a `#` with no space before it stays
  part of the value (`abc#def`), quoted values keep everything inside the quotes, and `export` is
  allowed. Quotes around a value in the install folder's `.env` are now removed as well, as they
  already were in a project's `.env`.
- When folder trust left a folder's MCP servers out, the MCP tools said to add a `.sigil-mcp.json`
  although the file was there. They now name the servers that wait for folder trust and say to run
  `/trust grant` and restart Sigil. The startup note and `/status` say the same.
- MCP servers from a folder that isn't trusted yet were left out without a word in what the model
  was told, and a resumed session kept the MCP summary from when it started, so it could still list
  servers that folder trust had since left out. The MCP summary, `/mcp` and `/status` now say how many
  servers wait for folder trust (`/trust grant` loads them), and a resumed session gets a note with
  the current MCP status when it changed. Server names from an untrusted project file reach the
  model only as short plain labels.
- `sigil sessions` opened the chosen session without asking the folder trust question, so a folder
  that was never trusted ran with its hooks, MCP servers and env turned off and nothing said why. It
  now asks like every other interactive start, `-c` and `-r` included. A launcher that builds the
  agent itself can ask the same question with `ask_folder_trust()`. A chat that starts before
  the question was answered, or after the project items changed, now says its project items are off
  and how to turn them on.
- A long session on Claude 5.1 or later could stop with HTTP 400 "Invalid `signature` in `thinking` block. The block
  is bound to a different conversation" and fail on every turn after it. Claude checks that nothing before a
  replayed thinking block changed, and microcompaction, compaction and changed tools edit that part. Requests now
  send the `thinking-binding-controls-2026-08-01` beta with `block_binding` set to `drop_block`, so the API leaves
  such a block out. Where that can't be sent, Sigil removes the block the error names and every later one, sends
  the request again once, and doesn't send those blocks again. Saved sessions that already hit the error continue
  when resumed. Amazon Bedrock gets the same removal. `SIGIL_THINKING_BINDING` (`drop`, `error`, `off`) changes it.
- Reasoning that other providers need back is now kept and sent back to the endpoint that wrote it. DeepSeek
  (thinking mode with tools) and Kimi K3, K3.1 Pro and k2.7-code get `reasoning_content` on assistant messages, and
  Gemini 3 gets each tool call's thought signature in `extra_content`. When a step of tool calls came from another
  model and has no signature, its first call carries Google's `skip_thought_signature_validator` value. If a server answers 400 asking for
  `reasoning_content` or thought signatures, the request is sent again with them, and if it refuses those fields,
  again without them. Either choice holds for the rest of the session. `SIGIL_REASONING_REPLAY` (`on`, `off`)
  overrides the automatic choice.
- Provider detection matched host names by suffix, so `evilanthropic.com` counted as Anthropic.
  It now needs the exact host or a subdomain.
- After the first prompt of a REPL session, and after `-c` / `-r`, `/cost`, `/stats` and the
  status line counted the session's earlier usage and cost twice.
- `agent-resume` without `--input-cost-per-million` / `--output-cost-per-million` set the
  session's prices to 0, so resumed turns cost nothing. The stored prices are now kept.
- Model calls outside the main loop were missing from `/cost`, `/stats`, the status line and the
  budgets: `/btw`, compaction, rewind summaries, the `web_fetch` prompt, `prompt` and `agent`
  hooks, subagents, the architect's editor and the fast model. Each is now counted once, priced by
  the model that served it, and kept in the session file, including in the backup a Ctrl+C writes.
- `/compact` replaced the session's saved usage with a context estimate, so the token, cache and
  cost totals reset after `/compact` and after resuming a compacted session. They are kept now.
- Automatic compaction of a prompt that was too long counted the summary's tokens inside the
  turn. It is now counted once, as a side call, at the price of the model that wrote it.
- Usage from proxies that report `prompt_tokens` together with `cache_read_input_tokens` counted
  cached tokens twice.
- `/reload-plugins` did not connect `prompt`, `mcp_tool` and `agent` hooks to the model, so those
  hooks failed after a reload.
- `skill_get` and `/reload-skills` now find skills in `~/.sigil/skills`, parent folders,
  `.agents/skills` and `--add-dir` folders. These skills were listed in the prompt but could
  not be loaded.
- `rollback_workspace` returns an error when `git reset --hard` or `git clean -fd` fails. It used
  to report success.
- A subagent run with `isolation: worktree` that made no changes now reports a failed
  `git worktree remove` instead of saying the worktree was removed.
- On Windows, a hung `taskkill` can no longer block stopping a background agent or shell. Each
  call times out after 10 seconds.
- Output from git and other subprocesses is decoded as UTF-8 with replacement characters. A
  non-ASCII file name in `git ls-files` used to crash the repo map on Windows code pages.
- Saved sessions are read as UTF-8, so a session with non-ASCII text loads on Windows.
- `harbor_adapter.py`, which runs Sigil in Harbor and Terminal-Bench, could not install Sigil on
  current Debian, Ubuntu or Alpine images, which refuse a plain `pip install`, or where the image's
  Python was older than 3.10. It now installs Sigil with uv in the agent user's folders, with its
  own Python 3.12 when the image has none. A system prompt with spaces reached Sigil as separate
  words, a task that started with `-` was read as a flag, `--cwd` broke on a folder with spaces in
  its name, and runs stopped after 12 turns. Runs now default to 100 turns, a version such as
  `1.1.0` installs the `v1.1.0` tag, the same version already in the image is not installed again,
  and the token totals and cost reach Harbor. Checked against Harbor 0.25.0.
- `benchmarks/run.py` and `benchmarks/run_suite.py` crashed on Windows when their output went to a
  file or a pipe. They write UTF-8 now.

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
- Context window and vision support read from the server, `<think>` blocks removed, and tool calls
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
- Starts read-only, with `--allow-write`, `--allow-shell` and `--unsafe`. Permission modes (default,
  acceptEdits, plan, bypass, dontAsk) with Shift+Tab, and permission rules (`allow` / `ask` / `deny`).
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
- Print mode (`-p`) with text, JSON or stream-JSON output, `sigil doctor` and shell completion.
  91 slash commands, type-while-it-works input, Esc to stop, multiline input and prompt history.
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
- Credentials are dropped when an HTTP request is redirected to another origin. OAuth endpoints
  must use HTTPS (http only on localhost), and the OAuth callback page is not cached and sends no
  referrer.
- The main API key is sent only to its own server, and other hosts use `SIGIL_API_KEY_<HOST>`. MCP
  servers get neither Sigil's keys nor, by default, the SSH agent socket.
- Plan mode checks every command on a shell line (and PowerShell commands) before treating it as
  read-only. Permission rules match bash and PowerShell commands alike.
- A plugin source listed by a remote marketplace must use HTTPS or SSH, and git runs with protocol
  limits.
- Restarting a background agent rebuilds Sigil's own worker command, and a resumed session takes its
  folders and permissions from the command line.
- Escape sequences in model and tool text are removed before they reach the terminal, and tool-result
  text is framed as data for the model.
- Every security limit can be changed by you: see the security overrides in `.env.example`.

### Notes

- Requires Python 3.10 or later. Core dependencies: `prompt_toolkit` and `rich`. Optional extras:
  `pdf`, `yaml`, `ast`, `tokenizers`, `dev`, `all`.
