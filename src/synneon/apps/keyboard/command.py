"""
Compass Typing — command behavior layer.

Contains all runtime behavior: key reading, HTTP API calls, the Neon
gesture pipeline, and the Typer command with its main event loop.

The UI layer (_ui.py) is the only intra-package import. Everything
here either talks to external services or drives the terminal session.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import httpx
import typer
from prompt_toolkit.input import create_input
from prompt_toolkit.keys import Keys
from pynput.keyboard import Key
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from synneon.apps.keyboard._ui import NeonState, render_picking, render_typing
from synneon.apps.shared.blink import BlinkManager
from synneon.apps.shared.config import load_blink_config, load_gestures_config, load_neon_config
from synneon.input.compass import Compass

console = Console()

_KEY_MAP: dict[str, Key] = {
    "UP": Key.up,
    "DOWN": Key.down,
    "LEFT": Key.left,
    "RIGHT": Key.right,
}


# ── Key reading ────────────────────────────────────────────────────────────────


def _read_key(inp, timeout: float = 0.1) -> str | None:
    """
    Read one semantic action from ``inp`` within ``timeout`` seconds.

    Returns one of: "up" | "down" | "left" | "right" | "backspace" |
    "enter" | "quit" — or ``None`` if no recognised key was pressed.
    """
    _MAP = {
        Keys.Up: "up",
        Keys.Down: "down",
        Keys.Left: "left",
        Keys.Right: "right",
        Keys.Backspace: "backspace",
        Keys.ControlH: "backspace",
        Keys.ControlJ: "enter",
        Keys.ControlM: "enter",
        Keys.ControlC: "quit",
    }
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for kp in inp.read_keys():
            k = kp.key
            if k in _MAP:
                return _MAP[k]
            if isinstance(k, str) and k.lower() == "q":
                return "quit"
        time.sleep(0.01)
    return None


# ── HTTP helpers ───────────────────────────────────────────────────────────────


async def _expand(tokens: str, endpoint: str, context: dict | None = None, timeout_s: float = 15.0) -> list[dict]:
    """POST tokens to the abbreviation-expansion endpoint; return candidates list."""
    payload = {
        "tokens": tokens,
        "profile": "default",
        "context": context
        or {
            "emotion": {"label": "neutral", "intensity": 0.0},
            "parsed_visual": {"focused_object": None, "detected_objects": []},
            "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": None},
        },
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()
    return resp.json().get("candidates", [])


async def _speak(text: str, base_url: str, timeout_s: float = 30.0) -> None:
    """
    POST text to the TTS endpoint, receive MP3 bytes, and play them in a
    background daemon thread (non-blocking for the caller).
    """
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(f"{base_url}/audio/speak", json={"text": text})
        resp.raise_for_status()
        audio_bytes = resp.content

    def _play(data: bytes) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            if platform.system() == "Darwin":
                subprocess.run(["afplay", tmp_path], check=True)
            else:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", tmp_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        finally:
            os.remove(tmp_path)

    threading.Thread(target=_play, args=(audio_bytes,), daemon=True).start()


# ── Neon gesture pipeline ──────────────────────────────────────────────────────


async def _neon_gesture_loop(
    config_path: Path | None,
    stop_event: asyncio.Event,
    state: NeonState,
    display: bool = False,
    debug: bool = False,
) -> None:
    """
    Connect to the Neon device, stream gaze + eye events, emit OS arrow-key
    presses on gesture detection, and write status to ``state`` for the UI log.
    """
    from pupil_labs.realtime_api.streaming import BlinkEventData
    from pynput.keyboard import Controller

    from synneon.apps.keyboard._ui import ARROWS
    from synneon.apps.shared.blink import close_viewer, ensure_viewer
    from synneon.input.detector import DetectorState, detect_gestures
    from synneon.neon.session import connect, stream_eye_events, stream_gaze, stream_video
    from synneon.schemas.neons import GazeSample

    neon_config = load_neon_config(config_path)
    gestures_config = load_gestures_config(config_path)
    blink_config = load_blink_config(config_path)
    compass_config = gestures_config.compass

    output_dir = Path("captures")
    output_dir.mkdir(parents=True, exist_ok=True)

    device_label = getattr(neon_config, "address", str(config_path or "neon"))
    state.log("info", f"Connecting to {device_label}…")

    try:
        device = await connect(neon_config)
    except Exception as exc:
        state.log("error", f"Connection failed: {exc}")
        return

    with state.lock:
        state.connected = True
        state.device_label = device_label
    state.log("connect", f"Connected  {device_label}")

    def on_context_update(ctx: dict) -> None:
        with state.lock:
            state.visual_context = ctx
        state.log("info", "Visual context updated")

    detector_state = DetectorState()
    keyboard = Controller()
    blink_manager = BlinkManager(blink_config, keyboard, on_context=on_context_update, debug=debug)

    sample_time_s = max(float(getattr(compass_config, "sample_time_length", 0.5)), 0.01)
    window: deque[GazeSample] = deque()
    wearing_prev = False

    async def _video() -> None:
        async for frame in stream_video(device):
            blink_manager.update_data(frame=frame, gaze=None)

    async def _eyes() -> None:
        async for event in stream_eye_events(device):
            if isinstance(event, BlinkEventData):
                double = await blink_manager.handle_blink(event.start_time_ns / 1e9, output_dir)
                with state.lock:
                    state.blink_count += 1
                    state.last_blink_ts = time.monotonic()
                if double:
                    state.log("double_blink", f"Double blink #{state.blink_count}")
                else:
                    state.log("blink", f"Blink #{state.blink_count}")

    async def _gaze() -> None:
        nonlocal wearing_prev
        async for sample in stream_gaze(device):
            worn = bool(sample.is_worn)
            blink_manager.update_data(frame=None, gaze=sample if worn else None)

            if worn != wearing_prev:
                wearing_prev = worn
                with state.lock:
                    state.wearing = worn
                state.log("wearing" if worn else "removed", "Device put on" if worn else "Device removed")

            window.append(sample)
            while window and (sample.timestamp - window[0].timestamp) > sample_time_s:
                window.popleft()

            event = detect_gestures(list(window), detector_state, compass_config)
            if event and event.type.value in _KEY_MAP:
                direction = event.type.value
                keyboard.press(_KEY_MAP[direction])
                keyboard.release(_KEY_MAP[direction])
                with state.lock:
                    state.last_gesture = direction
                    state.last_gesture_ts = time.monotonic()
                state.log("gesture", f"{ARROWS.get(direction.lower(), '⟹')} {direction}")

    tasks = [asyncio.create_task(t) for t in [_video(), _eyes(), _gaze(), stop_event.wait()]]
    if display:
        ensure_viewer(debug=debug)

    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except Exception as e:
                state.log("error", f"Task failed: {e}")
    finally:
        close_viewer()

        with state.lock:
            state.connected = False
        state.log("disconnect", f"Disconnected  {device_label}")
        await device.close()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _start_neon_thread(
    config_path: Path | None,
    stop_event: asyncio.Event,
    state: NeonState,
    display: bool = False,
    debug: bool = False,
) -> None:
    """Launch the Neon pipeline in a daemon thread with its own event loop."""

    def _run() -> None:
        import sys

        if not debug:
            devnull = open(os.devnull, "w")
            old_stderr, sys.stderr = sys.stderr, devnull

        try:
            asyncio.run(_neon_gesture_loop(config_path, stop_event, state, display=display, debug=debug))
        finally:
            if not debug:
                sys.stderr = old_stderr
                devnull.close()  # type: ignore

    threading.Thread(target=_run, daemon=True).start()


def _run_async_with_ui_updates(coro, redraw, interval: float = 0.12):
    """
    Run *coro* in a daemon thread while calling *redraw()* every *interval*
    seconds on the main thread so the event log stays live during HTTP I/O.

    Returns the coroutine result or re-raises its exception.
    """
    result_q: queue.Queue = queue.Queue()

    def _runner() -> None:
        try:
            result_q.put(("ok", asyncio.run(coro)))
        except Exception as exc:
            result_q.put(("err", exc))

    threading.Thread(target=_runner, daemon=True).start()
    while True:
        try:
            tag, value = result_q.get(timeout=interval)
        except queue.Empty:
            redraw()
            continue
        if tag == "ok":
            return value
        raise value


# ── Typer command ──────────────────────────────────────────────────────────────


def type_text(
    endpoint: str = typer.Option(
        "http://127.0.0.1:8000/inference/expand",
        "--endpoint",
        "-e",
        help="Abbreviation-expansion API endpoint.",
    ),
    speak_output: bool = typer.Option(
        False,
        "--speak",
        help="Speak the selected expansion via the server TTS endpoint.",
    ),
    neon: bool = typer.Option(
        False,
        "--neon",
        help="Drive the compass with a Neon eye-tracker (capture-and-gesture mode).",
    ),
    config_path: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to Neon TOML config file (used with --neon).",
    ),
    display: bool = typer.Option(
        False,
        "--display",
        help="Show live video feed with gaze overlay (used with --neon).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable verbose logging and disable fullscreen mode.",
    ),
) -> None:
    """
    Launch the Compass typing interface.

    Arrow keys navigate the 4-zone alphabet grid. Drill down until a single
    letter remains — it types automatically and resets to the full alphabet.
    Backspace goes up one level, or deletes the last typed character at root.
    Enter fetches expansion candidates; pick one with ↑↓ and confirm with ↵.

    Pass --neon to drive the compass with eye-gaze gestures from a Neon device.
    Pass --speak to play the selected phrase as audio via the server TTS endpoint.
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG, force=True)
    else:
        # Silence logging before the Rich UI takes over the terminal.
        logging.root.handlers.clear()
        logging.root.addHandler(logging.NullHandler())
        logging.root.setLevel(logging.CRITICAL)

    base_url = f"{urlparse(endpoint).scheme}://{urlparse(endpoint).netloc}"
    compass = Compass(abbreviation_expansion_endpoint=endpoint)
    neon_state = NeonState()

    status: str = "Ready — gaze to navigate" if neon else "Ready — use arrow keys to type"
    candidates: list[dict] = []
    cursor: int = 0

    neon_stop = asyncio.Event()
    if neon:
        _start_neon_thread(config_path, neon_stop, neon_state, display=display, debug=debug)

    def snap() -> dict | None:
        return neon_state.snapshot() if neon else None

    inp = create_input()
    with Live(
        render_typing(compass, None, False, status, snap(), neon),
        console=console,
        screen=not debug,
        auto_refresh=False,
    ) as live:

        def redraw_typing(active: str | None = None, flash: bool = False, msg: str = "") -> None:
            live.update(render_typing(compass, active, flash, msg, snap(), neon), refresh=True)

        def redraw_picking(msg: str = "") -> None:
            live.update(render_picking(compass, candidates, cursor, msg, snap(), neon), refresh=True)

        with inp.raw_mode():
            last_revision: int = -1

            while True:
                key = _read_key(inp, timeout=0.15)

                if key is None:
                    if neon:
                        s = snap()
                        if s["revision"] != last_revision:
                            last_revision = s["revision"]
                            redraw_picking(msg=status) if candidates else redraw_typing(msg=status)
                    continue

                if key == "quit":
                    break

                if candidates:
                    candidates, cursor, status = _on_pick(
                        key,
                        candidates,
                        cursor,
                        compass,
                        base_url,
                        speak_output,
                        status,
                        redraw_typing,
                        redraw_picking,
                    )
                else:
                    s = snap()
                    candidates, cursor, status = _on_type(
                        key,
                        compass,
                        endpoint,
                        status,
                        redraw_typing,
                        redraw_picking,
                        context=s.get("visual_context") if s else None,
                    )
                if candidates:
                    redraw_picking(msg=status)

    if neon:
        neon_stop.set()

    console.print(
        Panel(
            f"[bold]Final text:[/bold]  [cyan]{compass.current_text or '(empty)'}[/cyan]",
            border_style="#2ecc71",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


# ── Mode handlers ──────────────────────────────────────────────────────────────


def _on_pick(
    key: str,
    candidates: list[dict],
    cursor: int,
    compass: Compass,
    base_url: str,
    speak_output: bool,
    status: str,
    redraw_typing,
    redraw_picking,
) -> tuple[list[dict], int, str]:
    if key == "up":
        cursor = (cursor - 1) % len(candidates)
        return candidates, cursor, status

    if key == "down":
        cursor = (cursor + 1) % len(candidates)
        return candidates, cursor, status

    if key == "enter":
        chosen = candidates[cursor]["text"]
        compass.clear()
        if speak_output:
            status = f"✓ Speaking: {chosen}"
            redraw_typing(msg=status)
            try:
                _run_async_with_ui_updates(
                    _speak(chosen, base_url),
                    lambda: redraw_typing(msg=status),
                )
            except Exception as exc:
                status = f"✗ TTS failed: {exc}"
                redraw_typing(msg=status)
        else:
            status = f"✓ Selected: {chosen}"
            redraw_typing(msg=status)
        return [], 0, status

    if key == "backspace":
        status = "Cancelled — keep typing"
        redraw_typing(msg=status)
        return [], 0, status

    return candidates, cursor, status  # left/right ignored in pick mode


def _on_type(
    key: str,
    compass: Compass,
    endpoint: str,
    status: str,
    redraw_typing,
    redraw_picking,
    context: dict | None = None,
) -> tuple[list[dict], int, str]:
    if key in ("up", "down", "left", "right"):
        prev_len = len(compass.current_text)
        compass.process_input(key)
        typed_char = len(compass.current_text) > prev_len
        if typed_char:
            status = f"Typed  '{compass.current_text[-1]}'"
        else:
            letters = compass.zones.get(key, [])
            status = f"Zoom → [{' '.join(letters[:6])}{'…' if len(letters) > 6 else ''}]"
        redraw_typing(active=key, flash=typed_char, msg=status)
        return [], 0, status

    if key == "backspace":
        compass.back()
        status = "Back"
        redraw_typing(msg=status)
        return [], 0, status

    if key == "enter":
        buf = compass.current_text
        if not buf:
            status = "Nothing to expand yet"
            redraw_typing(msg=status)
            return [], 0, status
        status = f"Expanding '{buf}'…"
        redraw_typing(msg=status)
        try:
            candidates = _run_async_with_ui_updates(
                _expand(buf, endpoint, context=context),
                lambda: redraw_typing(msg=status),
            )
            if candidates:
                status = f"Pick an expansion for '{buf}'"
                return candidates, 0, status
            status = "✗ No candidates returned"
            redraw_typing(msg=status)
        except Exception as exc:
            status = f"✗ {exc}"
            redraw_typing(msg=status)

    return [], 0, status
