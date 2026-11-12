"""
Terminal UI layer for the Compass Typing interface.

Owns all design tokens, the NeonState model, and every Rich rendering
function. All public render_* functions are pure — they return Rich
renderables and never call console.print() directly.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from synneon.input.compass import Compass

# ── Design tokens ──────────────────────────────────────────────────────────────

COLORS: dict[str, str] = {
    "up": "#3498db",
    "right": "#e67e22",
    "down": "#9b59b6",
    "left": "#2ecc71",
}
ARROWS: dict[str, str] = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
LABELS: dict[str, str] = {"up": "UP", "right": "RIGHT", "down": "DOWN", "left": "LEFT"}

PANEL_WIDTH: int = 26
TOTAL_WIDTH: int = PANEL_WIDTH * 3
EVENT_LOG_MAX: int = 8

# Event kind → (Rich colour, icon)
_LOG_STYLE: dict[str, tuple[str, str]] = {
    "connect": ("#2ecc71", "⬆"),
    "disconnect": ("#e74c3c", "⬇"),
    "wearing": ("#3498db", "👁"),
    "removed": ("#888888", "○"),
    "gesture": ("#e67e22", "⟹"),
    "blink": ("#9b59b6", "◉"),
    "double_blink": ("#f1c40f", "◉◉"),
    "error": ("#e74c3c", "✗"),
    "info": ("#666666", "·"),
}

# ── Neon device state model ────────────────────────────────────────────────────


@dataclass
class NeonState:
    """
    Thread-safe container for Neon device status and event log.

    Written by the background Neon thread; read (via snapshot()) by the
    main UI thread.
    """

    connected: bool = False
    wearing: bool = False
    device_label: str = "—"
    last_gesture: str | None = None
    last_gesture_ts: float = 0.0
    last_blink_ts: float = 0.0
    blink_count: int = 0
    revision: int = 0
    visual_context: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: deque[tuple[str, str, str]] = field(default_factory=lambda: deque(maxlen=EVENT_LOG_MAX))

    def log(self, kind: str, detail: str) -> None:
        """Append a timestamped entry to the circular event log (thread-safe)."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with self.lock:
            self.events.append((ts, kind, detail))
            self.revision += 1

    def snapshot(self) -> dict:
        """Return a point-in-time copy of all mutable fields (thread-safe)."""
        with self.lock:
            return {
                "connected": self.connected,
                "wearing": self.wearing,
                "device_label": self.device_label,
                "last_gesture": self.last_gesture,
                "last_gesture_ts": self.last_gesture_ts,
                "last_blink_ts": self.last_blink_ts,
                "blink_count": self.blink_count,
                "revision": self.revision,
                "visual_context": self.visual_context,
                "events": list(self.events),
            }


# ── Rich components ────────────────────────────────────────────────────────────

_GAP = Padding("", (0, 0))


def _zone_panel(direction: str, compass: Compass, active: str | None) -> Panel:
    letters = compass.zones[direction]
    color = COLORS[direction]
    is_active = direction == active

    if not letters:
        content = Text("  —", style="dim")
    elif len(letters) == 1:
        content = Text()
        content.append(f"\n  {letters[0]}", style="bold white underline")
    else:
        half = (len(letters) + 1) // 2
        content = Text()
        content.append("  " + " ".join(letters[:half]) + "\n", style=f"bold {color}")
        content.append("  " + " ".join(letters[half:]), style=color)

    return Panel(
        Align(content, vertical="middle"),
        title=f"[bold {color}]{ARROWS[direction]}[/bold {color}]",
        subtitle=(
            f"[bold {color}]▸ {LABELS[direction]}[/bold {color}]"
            if is_active
            else f"[#555555]{LABELS[direction]}[/#555555]"
        ),
        border_style=color if is_active else "#444444",
        style="on #1e1e1e" if is_active else "on #111111",
        box=box.HEAVY if is_active else box.ROUNDED,
        width=PANEL_WIDTH,
        height=8,
        padding=(0, 1),
    )


def _center_panel(depth: int) -> Panel:
    return Panel(
        Align.center(Text("✛", style="#444444"), vertical="middle"),
        subtitle=(f"[#444444]depth {depth}[/#444444]" if depth else "[#333333]root[/#333333]"),
        box=box.ROUNDED,
        border_style="#2a2a2a",
        style="on #0d0d0d",
        width=PANEL_WIDTH,
        height=8,
    )


def _compass_grid(compass: Compass, active: str | None) -> Table:
    blank = Text(" " * PANEL_WIDTH)
    g = Table.grid(padding=0)
    g.add_column(width=PANEL_WIDTH)
    g.add_column(width=PANEL_WIDTH)
    g.add_column(width=PANEL_WIDTH)
    g.add_row(blank, _zone_panel("up", compass, active), blank)
    g.add_row(
        _zone_panel("left", compass, active),
        _center_panel(len(compass.history_stack)),
        _zone_panel("right", compass, active),
    )
    g.add_row(blank, _zone_panel("down", compass, active), blank)
    return g


def _picker_panel(candidates: list[dict], cursor: int) -> Panel:
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(width=8, no_wrap=True)
    tbl.add_column(width=TOTAL_WIDTH - 16, no_wrap=False)

    for i, c in enumerate(candidates):
        selected = i == cursor
        if selected:
            tbl.add_row(
                Text(f"▶ [{c['rank']}]", style="bold #3498db"),
                Text(c["text"], style="bold white on #1e3a4a"),
            )
        else:
            tbl.add_row(
                Text(f"  [{c['rank']}]", style="#555555"),
                Text(c["text"], style="#aaaaaa"),
            )

    return Panel(
        tbl,
        title="[bold #3498db]Expansion Candidates[/bold #3498db]",
        subtitle="[#555555]↑↓ move   ↵ select   ⌫ cancel[/#555555]",
        border_style="#3498db",
        style="on #0a1520",
        box=box.HEAVY,
        width=TOTAL_WIDTH,
        padding=(0, 2),
    )


def _output_panel(text: str, flash: bool) -> Panel:
    t = Text()
    t.append(text, style="bold white")
    t.append("▌", style="bold white blink")
    return Panel(
        t,
        title="[#555555]Input[/#555555]",
        border_style="#22a6b3" if flash else "#3a3a3a",
        style="on #0a1a1a" if flash else "on #111111",
        box=box.HEAVY,
        width=TOTAL_WIDTH,
        height=3,
        padding=(0, 2),
    )


def _title_panel() -> Panel:
    title = Text()
    palette = ["#3498db", "#3498db", "#2ecc71", "#2ecc71", "#e67e22", "#e67e22", "#9b59b6"]
    for ch, col in zip("COMPASS", palette, strict=True):
        title.append(ch, style=f"bold {col}")
    title.append("  TYPING", style="bold #cccccc")
    return Panel(
        Align.center(title),
        border_style="#222222",
        style="on #0a0a0a",
        box=box.ROUNDED,
        width=TOTAL_WIDTH,
        height=3,
        padding=(0, 0),
    )


def _footer_panel(status: str, pick_mode: bool, neon_mode: bool) -> Panel:
    keys = Table.grid(padding=(0, 3))
    keys.add_column(style="bold #3498db", width=12, no_wrap=True)
    keys.add_column(style="#666666", no_wrap=True)

    if pick_mode:
        keys.add_row("↑ ↓", "move selection")
        keys.add_row("↵", "confirm")
        keys.add_row("⌫", "cancel")
        keys.add_row("q", "quit")
    else:
        keys.add_row("👁 gaze" if neon_mode else "↑ ↓ ← →", "navigate zones")
        keys.add_row("⌫", "go up / delete")
        keys.add_row("↵", "expand via API")
        keys.add_row("q", "quit")

    if status.startswith("✓"):
        st_style = "bold #2ecc71"
    elif status.startswith("✗"):
        st_style = "bold #e74c3c"
    elif status.startswith("Zoom"):
        st_style = "#e67e22"
    else:
        st_style = "#888888"

    row = Table.grid(padding=0)
    row.add_column(width=40)
    row.add_column(width=36)
    row.add_row(keys, Align.center(Text(status, style=st_style), vertical="middle"))

    return Panel(row, border_style="#222222", style="on #0a0a0a", box=box.ROUNDED, width=TOTAL_WIDTH, padding=(0, 1))


def _invisible(height: int = 1) -> Panel:
    return Panel(
        Text(""),
        border_style="#111111",
        style="on #0a0a0a",
        box=box.MINIMAL,
        width=TOTAL_WIDTH,
        height=height,
        padding=(0, 0),
    )


def _device_panel(state: dict) -> Panel:
    row = Table.grid(padding=(0, 3))
    row.add_column(width=22, no_wrap=True)
    row.add_column(width=22, no_wrap=True)
    row.add_column(width=22, no_wrap=True)
    row.add_column(width=TOTAL_WIDTH - 66 - 9, no_wrap=True)

    conn = (
        Text("⬆ CONNECTED", style="bold #2ecc71")
        if state["connected"]
        else Text("⬇ DISCONNECTED", style="bold #e74c3c")
    )
    if state["connected"]:
        wear = Text("👁 WORN", style="#3498db") if state["wearing"] else Text("○ NOT WORN", style="#666666")
    else:
        wear = Text("")

    now = time.monotonic()
    if state["last_gesture"] and (now - state["last_gesture_ts"]) < 2.0:
        d = state["last_gesture"]
        gesture_txt = Text(f"{ARROWS.get(d.lower(), '⟹')} {d}", style=f"bold {COLORS.get(d.lower(), '#e67e22')}")
    else:
        gesture_txt = Text("")

    row.add_row(conn, wear, Text(state["device_label"], style="#555555"), gesture_txt)
    return Panel(
        row,
        title="[#444444]Neon Device[/#444444]",
        border_style="#222222",
        style="on #0a0a0a",
        box=box.ROUNDED,
        width=TOTAL_WIDTH,
        height=3,
        padding=(0, 1),
    )


def _log_panel(state: dict) -> Panel:
    events = state["events"]
    tbl = Table.grid(padding=(0, 2))
    tbl.add_column(width=12, no_wrap=True, style="#444444")
    tbl.add_column(width=3, no_wrap=True)
    tbl.add_column(width=TOTAL_WIDTH - 25, no_wrap=True)

    if not events:
        tbl.add_row("", Text("·", style="#333333"), Text("Waiting for Neon events…", style="#333333"))
    else:
        now = time.monotonic()
        for ts, kind, detail in reversed(events):
            color, icon = _LOG_STYLE.get(kind, ("#666666", "·"))
            detail_style = f"bold {color}" if kind == "gesture" and (now - state["last_gesture_ts"]) < 1.0 else color
            tbl.add_row(Text(ts, style="#383838"), Text(icon, style=f"bold {color}"), Text(detail, style=detail_style))

    return Panel(
        tbl,
        title="[#444444]Event Log[/#444444]",
        subtitle=f"[#333333]last {min(len(events), EVENT_LOG_MAX)} events[/#333333]",
        border_style="#1e1e1e",
        style="on #080808",
        box=box.ROUNDED,
        width=TOTAL_WIDTH,
        padding=(0, 1),
    )


# ── Layout composers ───────────────────────────────────────────────────────────


def render_typing(
    compass: Compass,
    active: str | None,
    flash: bool,
    status: str,
    neon_state: dict | None,
    neon_mode: bool,
) -> Group:
    parts: list = [
        _GAP,
        _title_panel(),
        _GAP,
        _output_panel(compass.current_text, flash),
        _GAP,
        _compass_grid(compass, active),
        _GAP,
        _footer_panel(status, pick_mode=False, neon_mode=neon_mode),
    ]
    if neon_mode and neon_state is not None:
        parts += [_GAP, _device_panel(neon_state), _log_panel(neon_state)]
    return Group(*parts)


def render_picking(
    compass: Compass,
    candidates: list[dict],
    cursor: int,
    status: str,
    neon_state: dict | None,
    neon_mode: bool,
) -> Group:
    parts: list = [
        _GAP,
        _title_panel(),
        _GAP,
        _output_panel(compass.current_text, False),
        _GAP,
        _picker_panel(candidates, cursor),
        _GAP,
        _footer_panel(status, pick_mode=True, neon_mode=neon_mode),
    ]
    if neon_mode and neon_state is not None:
        parts += [_GAP, _device_panel(neon_state), _log_panel(neon_state)]
    return Group(*parts)
