# Example configs

Copy any of these into your project root (or `~/.sigil/` for settings) and edit.
None of them contain keys: secrets are read from environment variables / `.env`.

| File | Copy to | Purpose |
|------|---------|---------|
| `.sigil-mcp.json` | `<project>/.sigil-mcp.json` | MCP servers (stdio). Add or remove whatever you like. |
| `.sigil-search.json` | `<project>/.sigil-search.json` | Web search providers with automatic key failover. |
| `settings.json` | `<project>/.sigil/settings.json` | Command hooks (PreToolUse / PostToolUse / Stop ...). |
