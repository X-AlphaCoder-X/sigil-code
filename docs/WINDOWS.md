# Running Sigil Code on Windows

Sigil Code runs on Windows 10, Windows 11 and Windows Server 2016 or newer. The Python side needs nothing special. What matters is the terminal you run it in.

**Short version:** if you're on Windows Terminal, you're set. If you're stuck with the legacy console (the classic blue Windows PowerShell 5.1 window, or plain `cmd.exe`), install [ConEmu](https://conemu.github.io/) and run Sigil inside it. It's a much better experience.

## What you need

| Thing | Why | Get it |
|---|---|---|
| Python 3.10+ | Runs the agent | [python.org](https://www.python.org/downloads/windows/) or Miniconda. Tick "Add python.exe to PATH". |
| Git for Windows | Provides `git` and the `bash.exe` that the `bash` tool uses | [git-scm.com](https://git-scm.com/download/win) |
| A modern terminal | Colours, Unicode boxes, Shift+Tab and Esc handling | ConEmu (recommended with the legacy console) or Windows Terminal |
| Node.js (optional) | Only if you want `npx`-based MCP servers | [nodejs.org](https://nodejs.org/) |

## Why ConEmu instead of the legacy console

The classic console host (`conhost.exe`) is what you get when you open Windows PowerShell 5.1 or `cmd.exe` from the Start menu on anything older than Windows 11. It works, but it was never built for apps like this:

- **Unicode is patchy.** Box-drawing characters and symbols such as `✎ ⏵⏵ ↳` often show up as `?` or empty squares. Microsoft's own write-up says the console ["has limited support for Unicode"](https://devblogs.microsoft.com/commandline/windows-command-line-inside-the-windows-console).
- **ANSI colours are hit and miss.** Virtual terminal sequences have to be switched on per process, and PowerShell in a conhost window doesn't always do it, so you can get raw escape codes instead of colour ([PowerShell#20898](https://github.com/PowerShell/PowerShell/issues/20898)).
- **Keys get lost.** Shift+Tab and a quick double Esc don't always reach the app.
- **Resizing breaks the live display**, and the scrollback is short.

[ConEmu](https://conemu.github.io/) is a free, open-source terminal that fixes all of that. You get proper Unicode, xterm 256 colours, tabs, a big scrollback and reliable keys. It hosts your existing shell (cmd, Windows PowerShell, PowerShell 7 or Git Bash), so nothing else changes. It also works on Windows Server 2016/2019, where you can't get Windows Terminal (that needs Windows 10 1903+ or Server 2022), and you can use the portable zip without admin rights.

| Your setup | What to do |
|---|---|
| Windows 11, or Windows 10 with Windows Terminal | Use Windows Terminal. See [Other terminals](#other-windows-terminals). |
| Windows 10 without Windows Terminal, Server 2016/2019, or "I just open PowerShell" | Install ConEmu. Follow the steps below. |
| VS Code | The built-in terminal is fine. |

## Setup with ConEmu, step by step

### 1. Install Python and Git

Install both, then open a new console and check:

```bat
python --version
git --version
where bash
```

`where bash` should list `C:\Program Files\Git\bin\bash.exe` (or `usr\bin\bash.exe`). If it doesn't, add `C:\Program Files\Git\bin` to your PATH.

### 2. Install ConEmu

Download ConEmu from [conemu.github.io](https://conemu.github.io/) (the installer or the portable 7z). On first launch:

1. In the Fast Configuration dialog, pick any palette. "Monokai" or "<Solarized>" both read well.
2. Choose `{Shells::cmd}` (or `{Shells::PowerShell}` if you prefer PowerShell) as the startup task for now.

Then open **Settings** (Win+Alt+P) and change these:

| Setting | Where | Value |
|---|---|---|
| Font | General > Fonts | `Consolas`, `Cascadia Mono` or any Nerd Font, size 14+ |
| Unicode | General > Fonts | Tick "Use Unicode font" settings, keep "Alternative font" on so missing glyphs fall back |
| ANSI | Features > Colors | "ANSI X3.64 / xterm 256 colors" turned on (on by default) |
| Inject ConEmuHk | Features | Ticked. Needed for proper ANSI in child processes |
| Keys | Keys & Macro | Untick anything bound to plain `Esc` or `Shift+Tab` if you've customised keys before |
| Scrollback | Features > Text cursor / Size & Pos | Buffer height 9999 so long sessions keep their history |

Click **Save settings**.

### 3. Get Sigil Code

```bat
cd %USERPROFILE%
git clone https://github.com/X-AlphaCoder-X/sigil-code.git
cd sigil-code
python -m pip install -e .
copy .env.example .env
notepad .env
```

Fill in `OPENAI_BASE_URL`, `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` and save.

### 4. Make a ConEmu task for it

This gives you a one-click "Sigil" tab with UTF-8 already set.

Settings > Startup > Tasks > click **+**, name it `Sigil`, and paste this into the commands box (change the path if you cloned somewhere else):

```bat
cmd.exe /k "chcp 65001 >nul & set PYTHONUTF8=1 & set PYTHONIOENCODING=utf-8 & cd /d %USERPROFILE%\sigil-code & sigil agent-chat --allow-write --allow-shell"
```

Prefer PowerShell? Use this instead:

```powershell
powershell.exe -NoExit -Command "chcp 65001 > $null; $env:PYTHONUTF8='1'; cd $env:USERPROFILE\sigil-code; sigil agent-chat --allow-write --allow-shell"
```

Optional: tick "Default task for new console" so every new tab opens Sigil.

To open the agent in a different project, change `cd /d ...` to that project's folder and add `--cwd .`, or just start a plain cmd tab, `cd` to your project and run `sigil agent-chat --allow-write --allow-shell` there.

### 5. Check it works

In the Sigil tab you should see the prompt with a toolbar underneath. Try:

- type `/help` and press Enter: the command list should render with colours
- press **Shift+Tab**: the toolbar should switch to `accept edits`, then `plan mode`
- ask it something that runs a tool, and while it works, type a message: you should see `✎` and your text
- press **Esc** on an empty line during a run: it should stop

If boxes or arrows show as `?`, the font is missing glyphs. Switch to Cascadia Mono or a Nerd Font.

## Other Windows terminals

**Windows Terminal** (Win10 1903+, Win11, Server 2022): works out of the box. Add a profile with this command line:

```
cmd.exe /k "chcp 65001 >nul & cd /d %USERPROFILE%\sigil-code & sigil agent-chat --allow-write --allow-shell"
```

**Legacy PowerShell / cmd window**: usable in a pinch. Run `chcp 65001` first and pick a TrueType font in the window properties, but expect the issues listed above. ConEmu is the better option.

**Git Bash (mintty)**: start with `winpty sigil agent-chat ...` or Esc and arrow keys may not work. Prefer ConEmu or Windows Terminal.

**VS Code terminal**: works. Note that VS Code grabs some shortcuts, so check Shift+Tab isn't bound there.

## Windows notes

- **Paths in shell commands.** The `bash` tool runs Git Bash. Sigil translates `C:\Users\you\...` and `/c/Users/you/...` for you, and the model is told to use forward slashes.
- **Encoding.** If you see `UnicodeEncodeError`, set `PYTHONUTF8=1` (the ConEmu task above already does).
- **Long tasks.** Bash commands time out after 2 minutes by default (max 10). For longer jobs, ask the agent to use a background agent.
- **Antivirus.** Real-time scanning of `.port_sessions/` and `__pycache__` can slow things down. Excluding the repo folder helps on Server boxes with Defender on.
- **Running over RDP / on servers.** Everything works over Remote Desktop. ConEmu's "Quake style" slide-down mode is handy for keeping an agent one hotkey away.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sigil` is not recognized | Python's `Scripts` folder isn't on PATH. Use `python -m src.main agent-chat ...` from the repo, or add `%APPDATA%\Python\Python3xx\Scripts` / your env's `Scripts` folder to PATH. |
| `bash` tool fails with "not found" | Git for Windows isn't installed or `bash.exe` isn't on PATH (see step 1). Or point `SIGIL_GIT_BASH_PATH` at it, for example `C:\Program Files\Git\bin\bash.exe`. `bash` in `System32` (the WSL launcher) is skipped. |
| `?` instead of boxes or arrows | Change the font (step 2) and make sure `chcp 65001` ran. |
| Shift+Tab does nothing | A ConEmu hotkey is taking it. Check Keys & Macro. Use `/mode plan` as a fallback. |
| Typed text doesn't show while the agent runs | Make sure you're on the latest version. The live echo needs a real console (ConEmu, Windows Terminal, conhost), not a pipe. |
| Connection refused to `127.0.0.1:8000` | Your model server isn't running, or `OPENAI_BASE_URL` in `.env` points to the wrong place. |
