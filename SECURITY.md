# Security Policy

## Supported versions

Only the latest commit on the main branch is supported.

## Reporting a vulnerability

If you find a security vulnerability, please open a private security advisory
on GitHub (Security tab -> Report a vulnerability) instead of a public issue.

Include what you found, how to reproduce it, and its impact. You should hear
back within a few days. Please avoid public disclosure until a fix is released.

## Scope

- The agent harness itself (tool loop, session storage, permission modes).
- Anything the repo ships (CLI entry points, scripts, docs).

Out of scope: the models you connect sigil-code to, third-party API providers,
and anything reachable only by already having full shell access on your own
machine (the tool intentionally runs shell commands you approve).
