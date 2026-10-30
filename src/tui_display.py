from __future__ import annotations

import sys
import time
from typing import Any, TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.theme import Theme

# Custom theme for the Sigil TUI
SIGIL_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "thought": "italic dim white",
    "tool": "bold magenta",
    "metrics": "bold green",
    "replay": "italic dim grey70",
    "divider": "bold gold3",
})

class SigilTUIDisplay:
    """
    A Rich-based Terminal User Interface for Sigil Code.
    Provides a real-time dashboard for agent activity.
    """

    def __init__(self, out: TextIO | None = None):
        self.console = Console(file=out or sys.stdout, theme=SIGIL_THEME)
        self.layout = self._build_layout()
        self.live = Live(self.layout, console=self.console, refresh_per_second=4, screen=False)
        
        self.chat_log: list[Text] = []
        self.metrics = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "turns": 0,
        }
        self.status = {
            "current_tool": "Idle",
            "active_agent": "Sigil (Root)",
            "turn_index": 0,
            "start_time": time.time(),
            "last_action": "Waiting for input...",
        }
        
    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="main", ratio=2),
            Layout(name="side", ratio=1)
        )
        layout["side"].split_column(
            Layout(name="metrics", size=10),
            Layout(name="status")
        )
        return layout

    def start(self):
        self.live.start()

    def stop(self):
        self.live.stop()

    def _update_main(self):
        # Render the last 20 lines of chat log
        if len(self.chat_log) > 500:  # only the last 25 are shown: keep the list bounded
            del self.chat_log[:-500]
        log_group = Group(*self.chat_log[-25:])
        self.layout["main"].update(
            Panel(log_group, title="[bold cyan]Agent Activity[/]", border_style="cyan")
        )

    def _update_metrics(self):
        table = Table.grid(expand=True)
        table.add_column(style="bold white")
        table.add_column(justify="right", style="metrics")
        
        table.add_row("Input Tokens:", f"{self.metrics['input_tokens']:,}")
        table.add_row("Output Tokens:", f"{self.metrics['output_tokens']:,}")
        table.add_row("Total Context:", f"{self.metrics['total_tokens']:,}")
        table.add_row("Cost Estimate:", f"${self.metrics['cost_usd']:.4f}")
        table.add_row("Session Turns:", str(self.metrics['turns']))
        
        self.layout["metrics"].update(
            Panel(table, title="[bold green]Metrics[/]", border_style="green")
        )

    def _update_status(self):
        elapsed = time.time() - self.status['start_time']
        
        status_text = Text()
        status_text.append(f"Agent: ", style="bold")
        status_text.append(f"{self.status['active_agent']}\n", style="info")
        
        status_text.append(f"Tool : ", style="bold")
        status_text.append(f"{self.status['current_tool']}\n", style="tool")
        
        status_text.append(f"Turn : ", style="bold")
        status_text.append(f"{self.status['turn_index']}\n", style="info")
        
        status_text.append(f"Time : ", style="bold")
        status_text.append(f"{elapsed:.1f}s\n\n", style="info")
        
        status_text.append(f"Last Action:\n", style="bold underline")
        status_text.append(f"{self.status['last_action'][:100]}...", style="italic dim")

        self.layout["status"].update(
            Panel(status_text, title="[bold yellow]System Status[/]", border_style="yellow")
        )

    def on_event(self, event: dict[str, Any], replay: bool = False) -> None:
        etype = event.get("type")
        style_prefix = "replay" if replay else ""
        
        if etype == "turn_start":
            self.status["turn_index"] = event.get("turn_index", 0) + 1
            self.metrics["turns"] = self.status["turn_index"]
            text = Text(f"\n╭─ Turn {self.status['turn_index']} ─╮", style="bold cyan")
            if replay:
                text.stylize("italic dim")
            self.chat_log.append(text)
            
        elif etype == "model_query":
            if not replay:
                from .prompt_constants import random_spinner_verb

                self.status["last_action"] = f"{random_spinner_verb()}..."

        elif etype == "assistant_response":
            content = event.get("content", "")
            if content:
                style = style_prefix if replay else "white"
                self.chat_log.append(Text(content, style=style))
                if not replay:
                    self.status["last_action"] = "Communicating..."
                
        elif etype == "tool_start":
            tool_name = event.get("tool_name", "unknown")
            if not replay:
                self.status["current_tool"] = tool_name
                self.status["last_action"] = f"Executing {tool_name}"
            
            style = "italic dim" if replay else "tool italic"
            self.chat_log.append(Text(f"  🔧 Executing {tool_name}...", style=style))
            
        elif etype == "tool_result":
            if not replay:
                self.status["current_tool"] = "Idle"
            ok = event.get("ok", True)
            status = "✓" if ok else "✗"
            color = "dim" if replay else ("success" if ok else "error")
            self.chat_log.append(Text(f"  {status} {event.get('tool_name')} completed.", style=color))

        elif etype == "resumption_marker":
            self.chat_log.append(Text("\n[ --- RESUMING LIVE EXECUTION --- ]\n", style="divider justify=center"))

        elif etype == "usage":
            usage = event.get("usage", {})
            self.metrics["input_tokens"] = usage.get("input_tokens", 0)
            self.metrics["output_tokens"] = usage.get("output_tokens", 0)
            self.metrics["total_tokens"] = usage.get("total_tokens", 0)
            self.metrics["cost_usd"] = event.get("total_cost_usd", 0.0)

        elif etype == "auto_nudge":
            self.chat_log.append(Text(f"  ⚠️ Action Nudge: {event.get('message')}", style="warning"))
            self.status["last_action"] = "Nudging Agent..."

        # Update all panels
        self._update_main()
        self._update_metrics()
        self._update_status()
