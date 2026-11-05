<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.png">
    <img src="docs/images/logo-light.png" alt="Sigil Code" width="420">
  </picture>
</p>

<p align="center">
  <b>An open-source AI coding agent for your terminal.</b><br>
  Plain Python. Any OpenAI-compatible model, including the one running on your own GPU.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <a href="https://github.com/X-AlphaCoder-X/sigil-code/actions/workflows/tests.yml"><img src="https://github.com/X-AlphaCoder-X/sigil-code/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platforms: Windows, Linux, macOS">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#use-any-model">Models</a> ·
  <a href="#features">Features</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#faq">FAQ</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

Sigil Code is a coding agent that works in your terminal. You describe the change, and it reads your code, edits files, runs your tests and shell commands, and keeps going until the job is done. It talks to any model with an OpenAI-compatible endpoint: OpenAI, OpenRouter, DeepSeek, Groq, Together and Anthropic's compat endpoint, or a local model served by Ollama, vLLM, LM Studio or llama.cpp.

> **Disclaimer:** Sigil Code is an independent open-source project. It's inspired by Claude Code but isn't affiliated with or endorsed by Anthropic or Claude Code, and it contains no proprietary code.

## Why Sigil Code

- Small and local models keep working. They often stall: an empty reply, a raw `<use_tool>` XML dump, or "I'll now run..." with no tool call after it. Auto-nudge spots these and gets the model moving again, and a message sanitizer fixes the payloads that make strict providers answer with HTTP 400.
- Windows is a first-class platform, with no WSL needed. Git-Bash path translation and `msvcrt` keyboard input are built in, and Linux and macOS work too.
- You can read all of it. Sigil is about 56k lines of plain Python across 111 modules, with two core dependencies (`prompt_toolkit` and `rich`). If you want it to behave differently, the code is right there.
- The full agent toolkit comes in the box: parallel tool calls, context compaction, sub-agents and background agents, checkpoints and `/rewind`, hooks, skills, custom commands and MCP servers.
- 1,728 tests across 123 files run offline against a fake model server, and CI runs them on Linux and Windows with Python 3.10 and 3.12 on every pull request and push to `main`.

## Quick start

```bash
git clone https://github.com/X-AlphaCoder-X/sigil-code.git
cd sigil-code
pip install -e .            # or: pip install -e ".[all]"
cp .env.example .env        # set OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
```

Then start a chat:

```bash
sigil                                              # interactive chat
sigil agent-chat --allow-write --allow-shell       # chat with edits and shell allowed up front
sigil agent "fix the failing test" --allow-write --allow-shell   # one-shot task
cat error.log | sigil -p "explain this" --output-format json     # print mode for scripts
```

<details>
<summary>More ways to run it</summary>

```bash
# resume the latest session
sigil -r

# resume a specific session
sigil --resume-session-id <session-id>

# list saved sessions and pick one interactively
sigil sessions

# continue the latest session (also works with -p)
sigil -c

# named session in its own git worktree
sigil -w feature-x -n feature-x

# MCP servers and background agents
sigil mcp add files -- npx -y @modelcontextprotocol/server-filesystem .
sigil agent-restart <bg_id>

# pick up an old session (non-interactive)
sigil agent-resume <session-id>

# without installing
python -m src.main agent-chat ...
```

</details>

Both `sigil` and `sigilcode` work as the command name. `.env` files load automatically from the repo root (git ignores it), the `--cwd` folder and `~/.sigil/.env`. They only fill variables that aren't set yet, and the names of values a project folder filled in are printed on stderr at start-up (`SIGIL_QUIET_PROJECT_ENV=1` hides that line).

Optional extras: `[pdf]` (pypdf, for reading PDFs), `[yaml]` (PyYAML), `[ast]` (tree-sitter with Python and C++ grammars), `[tokenizers]` (tiktoken), `[dev]` (pytest), or `[all]`.

> **On Windows?** Run it in Windows Terminal or [ConEmu](https://conemu.github.io/), not the legacy PowerShell/cmd window. The legacy console mangles Unicode and drops keys like Shift+Tab. The full walkthrough is in [docs/WINDOWS.md](docs/WINDOWS.md).

## Use any model

### Local models

vLLM, matching the defaults in `.env.example`:

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

```env
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=local-token
OPENAI_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct
```

For Ollama, set `OPENAI_BASE_URL=http://127.0.0.1:11434/v1` and `OPENAI_MODEL` to the model you've pulled. For LM Studio, use `OPENAI_BASE_URL=http://127.0.0.1:1234/v1` and the loaded model's name. llama.cpp's server works the same way.

### Hosted models

Any OpenAI-compatible provider works the same way: set the base URL, key and model name. Anthropic has an OpenAI-compatible endpoint, so you can point Sigil Code straight at it:

```env
OPENAI_BASE_URL=https://api.anthropic.com/v1/
OPENAI_API_KEY=<your Anthropic API key>
OPENAI_MODEL=claude-sonnet-4-5
```

`temperature` is only sent when you pass `--temperature`, so Claude 4.7+ works out of the box. With an anthropic.com base URL Sigil uses the native Messages API (prompt caching, streaming, thinking), because Anthropic's OpenAI-compatible endpoint does not cache prompts. `SIGIL_ANTHROPIC_NATIVE=0` keeps the compatible endpoint. If a provider rejects other sampling params, strip them with `SIGIL_EXCLUDE_PARAMS=top_p`.

### Provider compatibility

- `SIGIL_EXCLUDE_PARAMS` removes payload keys a provider won't accept (for example `temperature,top_p`).
- A message sanitizer fixes things strict providers reject with HTTP 400: orphan tool results, empty assistant turns and duplicate ids.
- Tool schemas get normalized so `parameters.type` is always `object`.
- Tool calls a model writes as text (Hermes, Qwen3-Coder or JSON style) still run, and `<think>` blocks are kept out of the reply.
- An opt-in RPM/TPM limiter (`SIGIL_RPM_LIMIT`, `SIGIL_TPM_LIMIT`, off unless set) keeps you under a provider's rate limits.
- Tokens are counted with tiktoken or HF tokenizers if you have them installed. Otherwise it uses a heuristic.

### Prompt caching

Prompt caching is on by default. Before each model call Sigil looks at the base URL and the model
and sends what that provider documents for reusing a repeated prompt prefix. This covers the main
loop, subagents, background agents, compaction and other summaries, the `web_fetch` prompt, the
architect editor, fallback models and side questions (`/btw`).

| Provider (host) | What Sigil sends |
|---|---|
| Anthropic (anthropic.com, native API) | `cache_control` on the tool list, the system prompt, the end of the previous request and the end of this one (at most 4) |
| OpenRouter, Claude and Gemini models | `cache_control` on the system prompt and the last messages, plus the `x-session-id` header |
| OpenRouter, other models | `x-session-id` header (the upstream provider caches on its own) |
| Alibaba DashScope (aliyuncs.com) | `cache_control` markers (5-minute cache, no TTL choice) |
| NanoGPT, Claude models | `cache_control` markers |
| OpenAI, Mistral, Together | `prompt_cache_key` (a hash of the model and system prompt, never a path or name) |
| Azure OpenAI | nothing for models before gpt-5.6 (they reject the fields) |
| xAI | `x-grok-conv-id` header |
| Fireworks | `x-session-affinity` header |
| Kimi (moonshot.ai) | `prompt_cache_options` for kimi-k3 models when the TTL is 1h |
| Ollama | `keep_alive` set to the TTL unless it is 5m (Ollama's own default) |
| Gemini API, Vertex AI | implicit caching, plus an explicit `cachedContents` cache for the system prompt and tools when the price rule below says it pays |
| Amazon Bedrock, Claude and Nova models | the Converse API with `cachePoint` blocks (Claude: tools, system and messages, 1h when the TTL is 1h. Nova: system and messages, 5 minutes) |
| Cerebras, DeepInfra, Kimi | `prompt_cache_key` (Kimi also `prompt_cache_options` for kimi-k3 at 1h) |
| DeepSeek, Groq, Z.ai, SambaNova, vLLM, llama.cpp | nothing, because these cache on their own |
| Chutes, LM Studio, Bedrock and any other server | `prompt_cache_key` (no documented scheme, so the OpenAI one) |

Sigil finds the provider from the host name, and for local servers from the port or name.
For a server it does not recognise by host (a local server on another port, a proxy or a custom
domain), set `SIGIL_PROMPT_CACHE_PROVIDER` to the provider whose rules apply: `ollama`, `vllm`,
`llamacpp`, `lmstudio`, `openrouter`, `openai`, `anthropic`, `deepseek` and the other names in
the table. A known host keeps its own rules, and an unknown name is reported once and ignored.
It changes what is sent for caching and the cache prices, not how Sigil talks to the server.

Settings, later ones win: the `promptCache` setting (`{"mode": "auto", "ttl": "5m"}` in
`settings.json`), then `SIGIL_PROMPT_CACHE` and `SIGIL_PROMPT_CACHE_TTL`, then `/cache` for the
session. Modes: `auto` (the table), `on` (also put markers on servers that are not known to take
them) and `off` (no cache field, marker or header at all). TTL: `1h` (default), `5m`, or `24h`
(OpenAI extended retention, treated as 1h elsewhere). With `1h` or `24h`, OpenAI's
`prompt_cache_retention` is sent only to the models OpenAI lists for it (gpt-5.5, gpt-5.4, gpt-5.2,
gpt-5.1, gpt-5 and their codex/pro variants on that list, gpt-4.1). The default applies to the main agent,
subagents and background agents. On Anthropic a 1-hour cache write costs 2x the input price and a
5-minute one 1.25x (reads cost the same), so `SIGIL_PROMPT_CACHE_TTL=5m` is cheaper when you rarely
pause between messages. On Ollama the TTL is sent as `keep_alive`, so the model stays loaded that long.

If a server answers HTTP 400 or 422 and the error names a cache field Sigil sent (or says an
unknown field was sent without naming another one), Sigil sends the same request once more
without the cache fields. When that works it stops sending them to that endpoint for the rest of
the session. Any other 400 or 422 (a context that is too long, a bad message) is shown at once.
Both decisions are written to the log. A streamed reply is only retried before its first token.

`/cache` shows the mode, TTL, what is sent to the current provider, the hit rate and the cache read
and write tokens. `/cache on|off|auto`, `/cache ttl 5m|1h|24h` and `/cache reset` change it for the
session, and background agents started afterwards get the same choice. `/cost`, `/usage`, `/stats`,
the status line JSON (`cache`) and the OpenTelemetry records show the same numbers. Input totals in
`/cost`, `/stats` and print mode include cached tokens, and `/cost` also shows the uncached part.

Cache reads and writes are priced from the input price with the provider's published ratio. For
DeepSeek and Z.ai that covers the models on their price pages, and their other models use the input
price. Gemini and Vertex AI reads cost 10% of the input price, Amazon Bedrock Nova reads 25% (Nova
cache writes are free) and SambaNova reads 10%. Set `--cache-read-cost-per-million`,
`--cache-write-cost-per-million` or `--cache-write-1h-cost-per-million` to use your own.

Input and output prices come from `--input-cost-per-million` and `--output-cost-per-million`,
else from the model's known prices: the server's model list where it publishes them (OpenRouter, and
NanoGPT when asked with `?detailed=true`), with your own table on top: the `modelPrices` setting or
`SIGIL_MODEL_PRICES` (a JSON object or the path of a JSON file), for example
`{"claude-sonnet-4-5": {"input": 3, "output": 15}, "glm-5@https://nano-gpt.com/api/v1": {"input": 0.6,
"output": 2.2, "cache_read": 0.11}}` in USD per million tokens (keys `input`, `output`,
`cache_read`, `cache_write`, `cache_write_1h`). The server list is only asked while
`SIGIL_DETECT_MODEL` is on (the default). All prices are derived again when the model or endpoint
changes (`/model`, `/profile`, `--profile`, `--agent`, a fallback model, a subagent or background
agent on another model, `agent-resume --model` or `--base-url`). A price set with a flag is kept.
Each model call is priced when it is made, by the model that answered, so a later switch does not
change the cost of earlier turns and a fallback turn uses the fallback's prices. `agent-resume`
keeps the session's stored prices unless you pass price flags. `background_agent_status` shows a
finished background agent's tokens, cache reads and writes, and cost.

Model calls outside the main loop count in the same totals: `/btw`, `/compact` and automatic
compaction, rewind summaries, the `web_fetch` prompt, `prompt` and `agent` hooks, subagents, the
architect's editor and the fast model. Each is priced by the model that served it. `/cost` and
`/stats` show them on a "Side calls" line (already included in the totals above it), `/cache`
and the status line JSON (`cost.side_cost_usd`) include them, and they count against
`--max-budget-usd` and the token budgets. The session file keeps them apart from the main totals
and writes them whole each time, so a resume never counts them twice. `/compact` keeps the
session's real token, cache and cost totals. Only the context estimate gets smaller.

`/btw`, compaction and rewind summaries reuse the conversation's prompt cache: they send the main
loop's own tools, system prompt and messages, unchanged and in the same order, from the same
model, and add one message with the request. `tool_choice` is left as it is, because Anthropic
documents that changing it invalidates the cached messages. The model is told in text not to
call tools, a tool call in that reply is never run, and a reply with no text is asked once more
(the method Anthropic's compaction docs give). Only a request that is too long for the context,
or a second reply without text, falls back to a plain-text transcript without tools. With a fast
model set (`SIGIL_FAST_MODEL` or `fastModel`) summaries go to it and cannot read the main
model's cache.

Explicit caches on Gemini and Vertex AI. Both also cache on their own (implicit caching), and an
implicit hit costs the same as a read from an explicit cache. An explicit cache (a `cachedContents`
resource holding the system prompt and the tool list) costs storage per hour on top, so Sigil only
creates one where the published prices say it pays: when requests per hour x the share of the prefix
that implicit caching missed x (input price minus cached price) is more than the storage price per
hour. It measures the first two from the last replies (at least 3 within 15 minutes). For Gemini 2.5
Flash ($0.30 input, $0.03 cached, $1.00 storage per million tokens per hour) that is about 3.7
requests an hour with no implicit hits, and for Gemini 3.1 Pro it is 2.5. Google dates some price
changes (Gemini 3.6 and 3.8 Flash cost twice as much from 2027-01-01), and the price for the day of the
request is used. The prefix must reach the
model's minimum (2,048 tokens for Gemini 2.5, 4,096 for newer models). Gemini 4 Argon has implicit
caching only. A request that uses the cache leaves out the system messages and tools (they are in
the cache) and names it in `extra_body.google.cached_content`. The cache lives for the TTL above and
is renewed while in use. If creating it fails, the request goes out without it (logged). If a
request that names it gets an error about the cache (403 or 404, or a 400 that names the cached
content), it is sent once more without the cache and the cache is deleted. A rate limit (429), an
auth error or another 400 keeps the cache and is handled like any other error. Two failed creates
turn explicit caches off for that endpoint for the run.

Amazon Bedrock. Bedrock caches prompts only on its own Converse API (its docs mark Chat
Completions as not supported for current Claude and Nova models), so with a
`bedrock-runtime.<region>.amazonaws.com` (or `bedrock-runtime-fips.<region>.amazonaws.com`) base URL
Sigil talks Converse to Claude and Nova models
(`SIGIL_BEDROCK_CONVERSE=0` keeps the OpenAI-compatible endpoint, and other Bedrock models such as
gpt-oss stay on it). It puts `cachePoint` blocks in the same four places as on Anthropic's API,
streams over Bedrock's event stream, sends thinking blocks back unchanged in tool loops and reads
`cacheReadInputTokens`, `cacheWriteInputTokens` and the 1-hour split in `cacheDetails`. A 400 that
names `cachePoint` is retried once without cache points. Amazon Nova caches at most 20,000 tokens of
prompt (AWS does not say what happens past that), so on Nova no cache point is placed after more
than that: it moves back to the last block that fits, or is left out (logged once per model). The
effort level reaches Nova 2 models as
`reasoningConfig` (`low`, `medium`, `high`, where `xhigh` and `max` send `high`, which leaves out the
temperature as Bedrock requires). A JSON schema (`--response-schema-file`) goes as `outputConfig`
structured output, with `additionalProperties: false` added to objects that do not set it. If
Bedrock answers 400 about it, the schema goes as an instruction instead, for that model for the rest
of the run. Auth: a Bedrock API key in
`AWS_BEARER_TOKEN_BEDROCK` (or as the usual key setting), else AWS Signature Version 4 from
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` or a profile in
`~/.aws/credentials` (`AWS_PROFILE`). Keys are never logged or saved.

Fireworks reports cached prompt tokens in the `fireworks-cached-prompt-tokens` response header
(non-streamed replies) and in `perf_metrics` of the last streamed chunk, and Sigil counts them as cache
reads when the usage object itself has none.

Google does not document the cached-token count on its OpenAI-compatible endpoints, so a miss only
counts once the endpoint has reported a cached count above 0. Until then `auto` stays with implicit
caching. Vertex prices come from Google's model-garden price page (Gemini 2.5 Pro, 2.5 Flash, 2.5
Flash-Lite and 3.1 Pro Preview). Other Vertex models get no explicit cache in `auto`
(`SIGIL_EXPLICIT_CACHE=on` creates one anyway). Vertex's minimums are 2,048 tokens for
Gemini 2, 6,144 for 3.0 Flash Preview, 3.1 Pro Preview, 3.7 and 3.8 Flash, and 4,096 otherwise.
Caches are named `sigil-<install id>-<process>-<random>` and deleted at exit and at the end of a
session. Caches a crashed run of this install left behind are deleted in the background at start-up,
and again at the first use of an endpoint, for every endpoint with a usable key (the main key for its
own server, `SIGIL_API_KEY_<HOST>` for another, `SIGIL_GOOGLE_TOKEN_COMMAND` for Vertex). Each
delete waits at most `SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT` seconds, the whole clean-up stops after
ten times that, and failures are logged. Only caches whose name carries this install's id are touched. At most `SIGIL_EXPLICIT_CACHE_MAX` (4) live at once. `/cache` lists them with their size,
expiry and storage cost so far, `/cache delete <name|all>` deletes them, and `/cache explicit
on|off|auto` and `SIGIL_EXPLICIT_CACHE` choose the mode (`on` skips the price rule, `off` never
creates one). `/cost`, `/stats`, the status line (`cost.cache_storage_usd`) and OpenTelemetry
(`sigil.explicit_cache` records) include the storage cost, and so do `--max-budget-usd`, the cost in
print mode and a background agent's cost. The storage cost is saved in the session file and
restored when the session is resumed (`-c`, `-r`, `/resume`, `agent-resume`), never counted twice.
`/clear` starts a new count, and a subagent's caches count in its parent's session. On Vertex the bearer token expires after
an hour: set `SIGIL_GOOGLE_TOKEN_COMMAND` (for example `gcloud auth print-access-token`) and Sigil
runs it (without a shell) for a new token every `SIGIL_GOOGLE_TOKEN_REFRESH_SECONDS` (2700). The
token is never logged or saved.

Things that start a new cache entry: changing the model, the effort or thinking level, the tool
list (adding an MCP server, or `/mcp disable` / `/mcp enable`, once each), and the request after
`/compact` or a `/rewind` summary (the conversation itself changed). A `/loop` run does not: it is a
new message at the end, and the tool list stays the same. A plain `/rewind` keeps the earlier part of the conversation cached. Older tool results moved to files by microcompaction
are moved in groups of `SIGIL_MICROCOMPACT_BATCH` (default 3) while the cache is warm, so the
cached part is rewritten once per group instead of every turn.

## Features

### In the chat

- A live display shows turn panels, tool-call boxes and result previews as they happen. There's no mouse capture, so your terminal scrollback still works.
- You can type while it works. Your text waits in a queue with a `✎` prompt and reaches the agent at the next turn boundary (`↳ delivered to agent`). Whatever's left over becomes your next prompt.
- Esc on an empty line stops the run and keeps the partial work, and Esc while typing clears the line. Esc Esc on an empty prompt opens `/rewind`.
- `/rewind [n] [both|conversation|code]` rolls back the conversation, the files, or both. Edits made with the write, edit and notebook tools are checkpointed per prompt. Bash changes are rolled back too for files the edit tools already touched. Files that only bash touches aren't tracked.
- Shift+Tab cycles the permission mode through `default`, `acceptEdits` and `plan`. `/mode <name>` also accepts `bypass` (alias `yolo`). Plan mode only allows reading and exploring: write tools and shell commands that aren't read-only are blocked.
- Background agents are listed under the input. Press Down to open one and watch its live log.
- Tab completes slash commands (built-in and custom) and `@` paths, and `@path/to/file` in a prompt attaches that file.
- `!command` runs a shell command right away, with the same permission checks, and its output goes with your next prompt. `/resume` switches sessions inside a chat.
- `read_file` on a screenshot shows it to vision models, `--image file.png` attaches one to the first prompt, and Alt+V pastes from the clipboard.
- `/loop <prompt>` repeats a prompt. With no interval the agent picks each next run (1 minute to 1 hour) and prints when and why (`↻ Loop loop_1: next run in 5m at 14:32: build still running`). It ends the loop itself when the task is done. `/loop 10m <prompt>` or `/loop every 10 minutes <prompt>` runs on a fixed interval, `/loop` lists what's scheduled and `/loop stop` ends it. Loops are saved with the session and come back on resume. Any that expired (7 days) or came due while Sigil was closed are named instead.
- Ctrl+J, Alt+Enter, or `\` then Enter gives you multiline input. Prompt history is saved across sessions (Up, Ctrl+R). Ctrl+G edits the prompt in `$EDITOR`, and `editorMode: "vim"` in settings gives you vi keys.

<details>
<summary>More chat features</summary>

- Opt-in helpers: `--plugin-dir` loads plugin folders, `/mcp login <server>` signs in to OAuth MCP servers and `--architect --editor-model <model>` splits planning and editing. There are also `SIGIL_REPO_MAP=1`, `SIGIL_AUTO_MEMORY=1`, `SIGIL_SESSION_MEMORY=1`, `SIGIL_SANDBOX=1` (Linux/macOS) and `/rewind <n> summarize-from|summarize-to`.
- Sessions and CLI: `sigil purge`, `sigil sessions --search <text>`, `/resume <text>`, `/export --redact`, `--plain`, `--session-id`, `--fork-session`, `--no-session-persistence`, `sigil completion bash`, `sigil doctor`. See [CHANGELOG.md](CHANGELOG.md).
- Checks: `/test` and `/lint` run your project's commands and hand failures to the agent, and `autoTest: true` runs tests after edits. `/profile <name>` switches provider.
- Session commands: `/add-dir`, `/btw`, `/memory add|edit`, `/permissions allow|ask|deny|remove`, `/model` (lists the server's models), `/agents`, `/mode dontAsk`.
- Output styles: `/output-style concise|explanatory|learning` or your own in `.sigil/output-styles/`.
- Status: `/statusline <command>` shows your own status text, and Ctrl+T prints the todo list as it changes.

</details>

### Agent core

- The tool loop runs tool calls in parallel. You can set budgets for turns, tokens, cost and tool calls. Chat has no turn cap by default (`--max-turns 0`), and one-shot `agent` stops at 12.
- Context management keeps long sessions going:
  - Auto-compaction writes an LLM summary. It runs preflight, retries when a prompt is too long, and has a fallback.
  - Microcompaction moves old, bulky tool results into scratchpad files, and the model can read them back with `read_file`.
  - Files you edited recently get restored after compaction.
  - AST "shredding" cuts files that haven't been touched for a few turns down to their signatures (tree-sitter is optional).
  - There's also snipping, and `/context` shows where your tokens are going.
- Auto-nudge catches stalled turns: an empty reply, raw `<use_tool>` XML, `finish_reason=tool_calls` with no calls, or a reply that ends with "I'll now run..." and no call. It only checks how the reply ends, so a finished report won't get flagged. Nudges are capped per session. Pick the strategy (`off`, `structural`, `heuristic`) with `--nudge-strategy` or `SIGIL_NUDGE_STRATEGY`, or set `SIGIL_AUTO_NUDGE=0` to turn it off. Left unset, it's on with the configured default.
- Retries use exponential backoff and can be tuned with the `SIGIL_RETRY_*` and `SIGIL_OVERLOADED_RETRY_*` env vars.
- Sessions are saved to disk under `<--cwd>/.port_sessions/`. You can resume, export, rename, tag and copy them.
- `ask_user_question` asks you in the chat and waits for your answer. Typing ahead pauses while it asks.
- Pass a JSON schema with `--response-schema-file` for structured output.

### Tools (77 built in)

| Group | Tools |
|---|---|
| Files | `read_file` (ranges, `allow_large`), `write_file`, `edit_file`, `notebook_edit`, `list_dir`, `glob_search`, `grep_search` |
| Shell | `bash` and, on Windows, `powershell` (`run_in_background` for servers and watchers, then `bash_output` / `bash_kill`, and `/bashes` lists them) with a security layer that blocks command substitution, heredocs, hidden/locale-quoting tricks and destructive patterns. Also does Windows path translation and read-only classification. Timeout is 2 min by default, 10 min max |
| Web | `web_fetch` (HTML to Markdown, optional `prompt` answered by a side model call, 15-min cache, http to https upgrade, cross-host redirects reported instead of followed), `web_search` (Tavily or Brave, `domains` / `blocked_domains`, automatic API-key failover) |
| Sub-agents | `delegate_agent` (nested agents, subtasks run in dependency order, `agent: Explore` / `Plan` are read-only, `general-purpose` or any `.sigil/agents/*.md`), `background_agent_start` / `status` / `kill` (separate processes) |
| Planning | `update_plan`, `plan_get`, `plan_clear`, `todo_write`, `task_*` dependency graph (create, start, block, complete, cancel, next, list, get, update) |
| Teams | `team_create` / `list` / `get` / `delete`, `send_message`, `team_messages` |
| Code intel | `LSP`: definitions, references, hover, document and workspace symbols, call hierarchy for `.py`, `.js`, `.ts` and more |
| MCP | `mcp_list_tools`, `mcp_call_tool`, `mcp_list_resources`, `mcp_read_resource` (real stdio transport, schemas load lazily) |
| Skills & agents | `skill_search` / `get` / `list`, `agent_search` / `get` / `list` |
| Meta | `create_new_tool` (the agent writes its own Python tool at runtime), `tool_search`, `rollback_workspace`, `ask_user_question`, `sleep`, `enter_plan_mode` / `exit_plan_mode`, `wait_for_mcp_servers`, `worktree_enter` / `exit` / `status`, `workflow_list` / `get` / `run`, `remote_trigger`, `config_get` / `set` / `list`, `account_*`, `remote_*`, `search_*` |
| Opt-in | `cron_create` / `cron_list` / `cron_delete` and `schedule_wakeup` (scheduled prompts and `/loop` in a chat), `memory_save` (`SIGIL_AUTO_MEMORY=1`), `session_note` (`SIGIL_SESSION_MEMORY=1`), `editor_edit` (`--architect`) |

### Slash commands (92)

<details>
<summary>Show all 92</summary>

```
account add-dir agent-kill agent-view agents architect ask bashes branch btw bug cache cd
cleanup_tools clear compact config context context-raw copy cost deep-link delete_tool diff
direct-connect disconnect doctor effort exit export files focus goal help hooks keybindings
lint login logout loop lsp map mcp memory messages mode model nudge output-style permissions
plan plugin plugins profile prompt read-only recap release-notes reload-plugins reload-skills
remote remotes rename resource resources resume rewind search ssh stats status statusline
subtask tag task task-next tasks team teams teleport terminal-setup test theme thinking
token-budget tools trigger triggers trust workflow workflows worktree
```

</details>

Aliases include `/quit`, `/tokens`, `/todos` and `/settings`. `/init`, `/review` and `/security-review` are built-in prompt commands, and your custom commands and skills run as `/name` too. Type `/help` inside a session for descriptions.

### Extensibility

- Hooks live in `settings.json` (`.sigil/settings.json` and `~/.sigil/settings.json`). There are 31 events, including `PreToolUse`, `PostToolUse`, `UserPromptSubmit` and `Stop`, and a hook can be a command, an HTTP call, a prompt, an agent or an MCP tool. Tool names are mapped, so matchers like `"Bash|Edit"` work. A hook can block a tool call or add notes to it. Edits to the hooks in these files apply on the next hook call, without a restart, and a file that is half saved or not valid JSON keeps the last good hooks (`SIGIL_HOOKS_RELOAD=0` reads them only at start).
- Memory files `SIGIL.md`, `SIGIL.local.md` and `AGENTS.md` are picked up from the current directory and every parent above it. Turn them off with `--disable-sigil-md`.
- Skills are `SKILL.md` folders in `.sigil/skills/`, and each one also runs as `/skill-name`.
- Custom commands are Markdown files in `.sigil/commands/` or `~/.sigil/commands/` that become `/name`, with `$ARGUMENTS`, `$0`, `$1`, `` !`cmd` `` output, `@file`, and frontmatter `description`, `argument-hint`, `model`, `allowed-tools`.
- Agents in `.sigil/agents/*.md` and `~/.sigil/agents/*.md` (frontmatter `name`, `description`, `tools`, `disallowedTools`, `model`) are subagent types for `delegate_agent`. Model aliases such as `haiku` map through `SIGIL_MODEL_HAIKU=<your model>`, and skills-repo folders are set with `SIGIL_SKILL_REPOS`.
- MCP servers can be stdio (`command`), HTTP (`"type": "http", "url"`) or SSE, listed in `.mcp.json` or `.sigil-mcp.json`. Each server runs once per session and its tools appear as `mcp__<server>__<tool>`. A stdio server doesn't get your SSH agent socket unless you set `SIGIL_MCP_SSH_AGENT=1` (or set `SSH_AUTH_SOCK` in that server's own `env` block). None come preconfigured, but there's an example in `examples/`.
- Web search providers are configured in `.sigil-search.json` (example in `examples/`), with keys in `TAVILY_API_KEY` / `BRAVE_SEARCH_API_KEY`.
- Custom tools are Python files in `src/custom_tools/`, or the agent can write one with `create_new_tool`. Your own `~/.sigil/custom_tools/` load every session. A repository's `src/custom_tools/` are imported only after you trust the folder (see [Folder trust](#folder-trust)). A custom tool can't take a built-in tool's name or shadow one of your own.
- Other manifests: `.sigil-config.json`, `.sigil-hooks.json`, `.sigil-lsp.json`, `.sigil-policy.json`, `.sigil-workflows.json`, `.sigil-triggers.json`, `.sigil-remote.json`, `.sigil-account.json`, `.sigil-teams.json`, `.sigil-ask-user.json` (pre-set answers for `ask_user_question`).

## Safety and permissions

Sigil Code starts read-only. It can read your code but can't change anything until you allow it.

| Flag | Effect |
|---|---|
| `--allow-write` | Lets it edit files |
| `--allow-shell` | Lets it run shell commands |
| `--unsafe` | Turns off the destructive-command guard. Only use this in a sandbox |
| `--permission-prompts allow\|deny` | With no terminal to ask in, allow or deny calls that need approval (default deny) |

### Permission rules

`permissions.allow`, `ask` and `deny` in `~/.sigil/settings.json`, `.sigil/settings.json` and `.sigil/settings.local.json` use this syntax: `Bash(git diff *)`, `Read(./.env)`, `Edit(src/**)`, `WebFetch(domain:example.com)`, `mcp__server__tool`. Deny wins, then ask, then allow. In chat, a call that needs approval asks: `y` allow once, `a` always allow (saved to `.sigil/settings.local.json`), `n` deny.

### Folder trust

A cloned repository can carry things that run code or reach your keys. The first time a folder has any of them, the chat lists them (commands and URLs, and env by name) and asks once: trust or skip. They are:

- hooks, a `statusLine` command and `env` in `.sigil/settings.json` / `.sigil/settings.local.json`, MCP servers in `.mcp.json`, and a `.env`
- settings that grant more: allow rules, `additionalDirectories`, a start mode, `sandbox`, `profiles`, other model hosts, `extraHeaders`, test/lint commands
- commands and skills with `allowed-tools`, agents with hooks, custom tools in `src/custom_tools/`, and plugin manifests
- search providers that would get your key on another host, and `@path` imports that reach outside the folder
- pre-set answers in `.sigil-ask-user.json` / `.sigil/ask-user.json`. A pre-set answer that is used reaches the model labelled as coming from that file, not as something you typed.

`deny` and `ask` rules always apply, and a project file never starts bypass mode. The answer is kept per folder in `~/.sigil/trusted_folders.json`, not in the repo, and Sigil asks again when those items change. Headless runs (`-p`, background agents) don't ask: untrusted items are skipped and named on stderr and in the log. `/trust` and `sigil trust [status|grant|skip|revoke]` show or change it. Servers you add with `sigil mcp add` and a `/statusline` you set run without asking, and `~/.sigil` is yours and isn't checked. `SIGIL_TRUST_ALL=1` (or `"trustAllProjects": true` in `~/.sigil/settings.json`) turns the question off.

### Security notes

Credentials are dropped when a request is redirected to another origin, the main API key only goes to its own server, MCP OAuth endpoints must be HTTPS, and plan mode checks every command on a shell line. Each limit has an override, listed under "Security overrides" in `.env.example`. To report a vulnerability, see [SECURITY.md](SECURITY.md).

### Other folders

Set `SIGIL_OTHER_DIRECTORIES=yes` (env or `.env`) to let the agent read files anywhere and save them anywhere writes are allowed, or `SIGIL_ALLOWED_DIRECTORIES=D:/data;E:/out` for only those folders. `~/.ssh`, `~/.aws`, `~/.gnupg`, OS folders and Sigil's key, trust and settings files in `~/.sigil` stay write-protected. `permissions.protectedPaths` adds paths, `SIGIL_PROTECTED_PATHS` replaces the list (empty = none).

## Configuration

Every setting can go in your shell environment or a `.env` file. [.env.example](.env.example) lists all of them with their defaults.

<details>
<summary>Environment variables</summary>

| Variable | Default | What it does |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | Chat completions endpoint |
| `OPENAI_API_KEY` | `local-token` | API key. Order used everywhere (chat, `-p`, resume, background agents): `--api-key`, then `SIGIL_API_KEY_<HOST>` (for example `SIGIL_API_KEY_OPENROUTER_AI`), `SIGIL_API_KEY`, `OPENAI_API_KEY` |
| `SIGIL_API_KEY` | | API key read before `OPENAI_API_KEY`. `SIGIL_API_KEY_<HOST>` is the key for one server only |
| `OPENAI_MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Model name |
| `TAVILY_API_KEY` | | Key for Tavily web search |
| `BRAVE_SEARCH_API_KEY` | | Key for Brave web search |
| `SIGIL_AUTO_NUDGE` | on | `0` turns auto-nudge off |
| `SIGIL_NUDGE_STRATEGY` | | `off`, `structural` or `heuristic` |
| `SIGIL_EFFORT_LEVEL` | | Effort level (`off` or `none`, `low`, `medium`, `high`, `xhigh`, `max`, `auto`). Sent as `reasoning_effort`, `thinking` or `chat_template_kwargs.enable_thinking`, whichever the provider reads. `/effort` sets it per session |
| `SIGIL_EFFORT_FIELD` | | Force the effort field: `reasoning_effort`, `chat_template_kwargs`, `thinking` or `none` |
| `SIGIL_CONTEXT_WINDOW` | | Context window size |
| `SIGIL_DISABLE_MICROCOMPACT` | | Turns microcompaction off |
| `SIGIL_MICROCOMPACT_KEEP_RECENT` | `5` | How many recent tool results microcompaction leaves alone |
| `SIGIL_PROMPT_CACHE` | `auto` | Prompt caching: `auto`, `on` or `off` (see Prompt caching). `/cache` changes it per session |
| `SIGIL_PROMPT_CACHE_TTL` | `1h` | Prompt cache lifetime: `1h`, `5m` or `24h`, sent in each provider's own form, for the main agent and all subagents |
| `SIGIL_PROMPT_CACHE_PROVIDER` | | Provider whose cache rules and prices apply to a server not recognised by its host (for example `ollama`, `vllm`, `openrouter`) |
| `SIGIL_MODEL_PRICES` | | Your model prices in USD per million tokens, a JSON object or the path of a JSON file (see Prompt caching), or the `modelPrices` setting |
| `SIGIL_ANTHROPIC_NATIVE` | on for anthropic.com | `0` uses Anthropic's OpenAI-compatible endpoint (no prompt caching) |
| `SIGIL_ANTHROPIC_STREAM` | `1` | `0` makes the native Anthropic client wait for the whole reply instead of streaming |
| `SIGIL_THINKING_BINDING` | `drop` | Claude 5.1+ (native API): what the API does with a replayed thinking block whose earlier context changed. `drop` leaves it out, `error` fails the request (Sigil then strips it and sends again), `off` sends no `block_binding` |
| `SIGIL_REASONING_REPLAY` | auto | `reasoning_content` sent back on assistant messages: auto for DeepSeek and Kimi K3, K3.1 Pro and k2.7-code (their APIs need it), `on` for every server, `off` for none |
| `SIGIL_MICROCOMPACT_BATCH` | `3` | While the prompt cache is warm, move older tool results to files this many at a time |
| `SIGIL_BEDROCK_CONVERSE` | on for Claude and Nova | `0` sends Bedrock Claude and Nova models to the OpenAI-compatible endpoint (no prompt caching) instead of the Converse API |
| `SIGIL_EXPLICIT_CACHE` | `auto` | Explicit Gemini / Vertex caches: `auto` (when the price rule says it pays), `on`, `off`. `/cache explicit` per session |
| `SIGIL_EXPLICIT_CACHE_MAX` | `4` | Most explicit caches one Sigil process keeps at once (1 to 64) |
| `SIGIL_EXPLICIT_CACHE_DELETE_TIMEOUT` | `5` | Seconds each explicit cache delete may take at exit, session end or the start-up clean-up (which stops after ten times this). An undeleted cache still expires |
| `SIGIL_GOOGLE_TOKEN_COMMAND` | | Command that prints a Google access token for Vertex, for example `gcloud auth print-access-token`. It runs without a shell and is never logged |
| `SIGIL_GOOGLE_TOKEN_REFRESH_SECONDS` | `2700` | How old the Vertex token may get before the command runs again (60 to 3300) |
| `SIGIL_HOOKS_RELOAD` | `1` | Hooks in the settings files are re-read on the next call after an edit, and `0` reads them only at start |
| `SIGIL_EXCLUDE_PARAMS` | | Comma-separated payload keys to strip |
| `SIGIL_RPM_LIMIT` | `0` (off) | Requests per minute. Calls wait to stay under it |
| `SIGIL_TPM_LIMIT` | `0` (off) | Tokens per minute. Calls wait to stay under it |
| `SIGIL_MAX_SUBAGENTS_PER_SESSION` | `200` | Sub-agent cap per session |
| `SIGIL_MAX_WEB_SEARCHES_PER_SESSION` | `200` | Web search cap per session |
| `SIGIL_DISABLE_WEB_FETCH` | | Turns `web_fetch` off |
| `SIGIL_RETRY_*`, `SIGIL_OVERLOADED_RETRY_*` | | Retry and backoff tuning |
| `SIGIL_OTHER_DIRECTORIES` | | `yes` lets the file tools use folders outside the workspace |
| `SIGIL_ALLOWED_DIRECTORIES` | | Only these extra folders (`;` on Windows, `:` elsewhere) |
| `SIGIL_PROTECTED_PATHS` | `~/.ssh`, `~/.aws`, `~/.gnupg`, OS folders, Sigil's key/trust/settings files in `~/.sigil` | Write-protected paths outside the workspace. It replaces the list (empty = none), and `permissions.protectedPaths` only adds |
| `SIGIL_MCP_PERSISTENT` | `1` | Keep one MCP server process per session (`0`: one per request) |
| `SIGIL_MCP_MAX_CONNECTIONS` | `16` | MCP server processes kept open at once |
| `SIGIL_STRIP_THINK` | `1` | Take `<think>` blocks and `reasoning_content` out of replies |
| `SIGIL_TEXT_TOOL_CALLS` | `1` | Run tool calls the model wrote as text (Hermes, Qwen3-Coder, JSON) |
| `SIGIL_VISION` | `1` | `0` turns image input off |
| `SIGIL_SESSION_STORE` | `project` | `home` keeps sessions in `~/.sigil/projects/` instead of `<cwd>/.port_sessions` |
| `SIGIL_API_KEY_<HOST>` | | Key for a `model@url` on another server (fallback, `fastModel`, editor model), e.g. `SIGIL_API_KEY_OPENROUTER_AI`. The main key is never sent to another host |
| `SIGIL_TRUST_ALL` | | `1` trusts every folder's project items without asking (see Folder trust) |
| `SIGIL_CONFIG_SET_SENSITIVE` | | `allow`: the model's `config_set` may change permissions, hooks, env and similar keys without asking |
| `SIGIL_LOOP_RESTORE` | `1` | `0` doesn't restore saved loops and scheduled prompts on resume |
| `SIGIL_DISABLE_CRON` | | `1` turns `/loop`, `cron_*` and `schedule_wakeup` off |
| `SIGIL_CRON_MIN_INTERVAL` | `60` | Shortest interval (seconds) for a prompt the assistant schedules with `cron_create` |
| `SIGIL_RAW_TERMINAL_OUTPUT` | | `1` prints model and tool text with its escape sequences (they are removed by default) |
| `SIGIL_SUBPROCESS_ENV_SCRUB` | | `1` removes every credential-looking variable from the environment of stdio MCP servers (Sigil's own keys are always removed) |
| `SIGIL_GIT_TIMEOUT` | `60` | Seconds a worktree git call may take |
| `SIGIL_MARKETPLACE_MAX_BYTES` | `5000000` | Size cap for a downloaded `marketplace.json` |
| `SIGIL_READ_MAX_BYTES` | `52428800` | Most bytes `read_file` reads from one file (0 = no limit). On a bigger file a line range still works, bounded by the same cap |
| `SIGIL_GREP_MAX_FILE_BYTES` | `10485760` | `grep_search` skips files bigger than this when it walks a folder (0 = no limit) |
| `SIGIL_GREP_MAX_FILES` | `20000` | `grep_search` stops after this many files (0 = no limit) |
| `SIGIL_MCP_SSE_MAX_CHARS` | `10000000` | Largest server-sent event an MCP SSE connection keeps |
| `SIGIL_LSP_CACHE_MAX` | `5000` | Files the code-intelligence index keeps in memory |
| `SIGIL_GIT_BASH_PATH` | | Windows: the `bash.exe` the bash tool runs |
| `SIGIL_QUIET_PROJECT_ENV` | | `1` hides the start-up line naming values a project `.env` or settings filled in |
| `SIGIL_MAX_BACKGROUND_AGENTS` | `4` | Background agents running at once (`0` = no limit) |
| `SIGIL_BACKGROUND_MAX_TURNS` | `40` | Turn cap for background agents |
| `SIGIL_BACKGROUND_STALL_MINUTES` | `15` | Mark a quiet background agent stalled (`SIGIL_BACKGROUND_STALL_KILL=1` stops it) |
| `SIGIL_NOTIFY` | `bell` | `toast` also shows a desktop notification, `off` turns it off |
| `SIGIL_CLIPBOARD` | | `off` makes `/copy` write only a file |
| `SIGIL_SHELL_GUARD` | `on` | `warn` runs blocked shell commands with a warning, `off` skips the shell guard (bypass mode uses `warn`) |
| `SIGIL_BACKGROUND_ASK` | `deny` | `allow` approves ask rules and shell-guard questions when there is no terminal |
| `SIGIL_DISABLE_USER_SETTINGS` | | Skip `~/.sigil/settings.json` |
| `SIGIL_MODEL_SONNET`, `_OPUS`, `_HAIKU`, `_FABLE` | | Model used when a command or agent asks for that model alias |
| `SIGIL_SKILL_REPOS` | (none) | Skills-repo folder names scanned in the workspace |
| `SIGIL_MAX_BACKGROUND_SHELLS` | `8` | Background shells running at once |
| `SIGIL_LOG_DIR` | `~/.sigil/logs` | Log folder (errors always, bodies with `--debug`), empty = no log file |
| `SIGIL_LOG_RETENTION_DAYS` | `2` | Logs older than this many days are deleted at start and hourly while running |
| `SIGIL_DEBUG` | | `1` logs raw requests and responses |
| `SIGIL_HISTORY_FILE` | `~/.sigil/prompt_history` | Prompt history file, empty = memory only |
| `SIGIL_BACKGROUND_SHELL_TIMEOUT` | off | Seconds before a background shell is stopped (checked by a watchdog, also when nobody reads the output) |
| `SIGIL_BACKGROUND_SHELL_MAX_BYTES` | `1073741824` (1 GiB) | A background shell is stopped when its output file passes this size (`0` = no limit) |
| `SIGIL_BACKGROUND_SHELL_CHECK_SECONDS` | `1` | How often the background-shell watchdog checks time and size |
| `SIGIL_BASH_STREAM_MAX_CHARS` | 10 x tool output limit (120000), at most 240000 | Output one bash command keeps and streams. Past it the middle is dropped, the start and end are kept (`0` = no limit) |
| `SIGIL_SKILL_PATHS_SCAN_LIMIT` | `20000` | Files and folders looked at when checking a skill's `paths:` (`.git`, `node_modules` and virtual envs skipped, `0` = no limit) |
| `SIGIL_SSH_AGENT` | on | `off` removes `SSH_AUTH_SOCK` / `SSH_AGENT_PID` from every child process (bash, background shells, hooks, status line, MCP servers, subagents, background agents) |
| `SIGIL_MCP_SSH_AGENT` | | `1` passes the SSH agent socket to stdio MCP servers too (they don't get it by default) |
| `SIGIL_HTTP_KEEP_AUTH_ON_REDIRECT` | | `1` keeps `Authorization` and other credentials on a redirect to another origin (dropped by default) |
| `SIGIL_MCP_ALLOW_INSECURE_OAUTH` | | `1` allows MCP OAuth endpoints that aren't HTTPS (default: HTTPS, or http on localhost) |
| `SIGIL_MARKETPLACE_ALLOW_INSECURE_SOURCES` | | `1` allows local or plain-http plugin sources in a marketplace fetched from a remote |
| `SIGIL_BACKGROUND_RESTART_UNLIMITED` | | `1` lets the model's `background_agent_restart` keep every option of the saved record |
| `SIGIL_RESUME_STORED_PERMISSIONS` | | `1` makes `agent-resume` take folders and permissions from the stored session file |

</details>

<details>
<summary>Other CLI flags</summary>

`--agent <name>`, `--agents '<json>'`, `--settings <json|file>`, `--setting-sources`, `--mcp-config <file>`, `--strict-mcp-config`, `--safe-mode`, `-p --input-format stream-json`, `--permission-mode default|acceptEdits|plan|bypass`, `--allowed-tools "Bash(git *) Read"`, `--disallowed-tools Bash`, `--tools Read,Grep,Glob`, `--debug`, `--debug-file <path>`, `--model`, `--base-url`, `--api-key`, `--cwd`, `--add-dir`, `--stream`, `--max-total-tokens`, `--max-budget-usd`, `--max-tool-calls`, `--max-model-calls`, `--max-session-turns`, `--response-schema-file`, `--nudge-strategy`, `--system-prompt`, `--append-system-prompt`, `--override-system-prompt`, `--disable-sigil-md`.

</details>

## FAQ

**Can I run it fully offline with a local model?**
Yes. Point `OPENAI_BASE_URL` at a local server such as Ollama, vLLM, LM Studio or llama.cpp. Only the features that reach out to other services need a connection, such as web search, web fetch, remote MCP servers and plugin marketplaces.

**Does it need WSL on Windows?**
No. It runs natively, and the `bash` tool uses the `bash.exe` that comes with Git for Windows. Use Windows Terminal or ConEmu for the best terminal experience. [docs/WINDOWS.md](docs/WINDOWS.md) has the details.

**Is it safe to open a repository I just cloned?**
Sigil starts read-only, and anything in a project folder that could run code or reach your keys waits for your answer in the folder trust prompt. See [Folder trust](#folder-trust).

**Which model works best?**
Any model with good tool calling. Sigil's defaults match Qwen3-Coder served by vLLM, and auto-nudge is there to help smaller models finish the job.

**Can I use it in scripts or CI?**
Yes. `sigil -p "..."` prints only the answer and returns an exit code, `--output-format json` or `stream-json` gives you structured output, and piped stdin goes in front of the prompt.

## Project layout

```
sigil-code/
├── src/                     # 111 modules
│   ├── main.py              # CLI entry point
│   ├── agent_runtime.py     # agent loop
│   ├── agent_tools.py       # built-in tools
│   ├── agent_prompting.py   # system prompt
│   ├── openai_compat.py     # model client
│   ├── compact.py           # auto-compaction
│   ├── microcompact.py      # tool-result offloading
│   ├── bash_security.py     # shell guard
│   ├── checkpoints.py       # per-prompt file checkpoints
│   ├── rewind.py            # /rewind
│   ├── permission_modes.py  # default / acceptEdits / plan / bypass
│   ├── message_queue.py     # type-while-it-works queue
│   ├── command_hooks.py     # settings.json hooks
│   ├── mcp_runtime.py       # MCP client
│   ├── skills_runtime.py    # skills and agent personas
│   ├── sigil_display.py     # terminal rendering
│   └── ...
├── tests/                   # 123 test files, run offline
├── benchmarks/              # optional eval harness
├── examples/                # .sigil-mcp.json, .sigil-search.json, settings.json
├── .github/workflows/tests.yml
├── .env.example
├── CHANGELOG.md
├── MISSING_FEATURES.md      # roadmap
├── pyproject.toml
└── LICENSE
```

The `benchmarks/` folder is optional. It has runners for HumanEval, Aider, BFCL, AIME, Codeforces, BigBench and terminal-bench. The terminal-bench runner needs Apptainer, which runs each task's Docker image. To run Sigil under [Harbor](https://github.com/harbor-framework/harbor) instead, use `harbor run --agent-import-path harbor_adapter:SigilCodeInstalledAgent` with `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` set (the options are listed at the top of `harbor_adapter.py`).

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

The suite doesn't need a network connection or a GPU, because it runs against a fake local model server. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, the layout, how to add a tool or slash command, and PR guidelines. Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Roadmap

[MISSING_FEATURES.md](MISSING_FEATURES.md) lists what's still open and what isn't planned. The main open items:

- A bash sandbox on native Windows (Linux and macOS have one, use WSL2 on Windows)
- The stateless HTTP transport from the 2026-07-28 MCP revision, and OAuth Client ID Metadata Documents
- A PyPI package (the `src` package name has to change first)
- Asking whether to keep a `-w` worktree when the session ends

## License

Sigil Code is released under the [GNU AGPL-3.0](LICENSE). You can use, modify and share it, but if you distribute a modified version or run one as a network service, you have to release its source under the AGPL-3.0 too.

## Acknowledgements

Sigil Code is inspired by the community Python port of Claude Code by [instructkr](https://github.com/instructkr), and by the design of Anthropic's Claude Code. It has been rewritten and re-architected into its own project. It isn't affiliated with or endorsed by Anthropic or Claude Code.
