from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import typer
from pynput.keyboard import Controller, Key, Listener

from synneon.apps.shared.blink import key_map
from synneon.apps.shared.config import load_gestures_config, load_neon_config
from synneon.input.compass import Compass
from synneon.input.detector import DetectorState, detect_gestures
from synneon.neon.session import connect, stream_gaze
from synneon.neon.synthetic import synthetic_gaze_stream
from synneon.schemas.gestures import CompassSettings
from synneon.schemas.neons import GazeSample

logger = logging.getLogger("synneon")


async def process_compass_gestures(
    gaze_stream,
    compass_config: CompassSettings,
    typing: bool = False,
    stop_event: asyncio.Event | None = None,
):
    """
    Core logic for gesture detection and compass navigation.
    """
    detector_state = DetectorState()
    compass_obj: Compass | None = Compass() if typing else None
    stop_keyboard_listener = asyncio.Event()
    loop = asyncio.get_running_loop()
    keyboard = Controller()

    sample_time_length_s = float(getattr(compass_config, "sample_time_length", 0.5))
    if sample_time_length_s <= 0:
        sample_time_length_s = 0.5

    window: deque[GazeSample] = deque()

    async def _send_and_reset_on_enter() -> None:
        if not compass_obj:
            return
        payload_text = compass_obj.current_text
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if not payload_text:
            compass_obj.reset_compass()
            typer.secho(f"[{timestamp}] ↺ COMPASS RESET (Empty)", fg=typer.colors.YELLOW, dim=True)
            return
        try:
            await compass_obj.expand_abbreviation()
            typer.secho(f"[{timestamp}] 🚀 COMPASS SENT: '{payload_text}'", fg=typer.colors.GREEN, bold=True)
        except Exception as e:
            typer.secho(f"[{timestamp}] ❌ COMPASS ERROR: {e}", fg=typer.colors.RED)
        finally:
            compass_obj.reset_compass()
            typer.secho(f"[{timestamp}] ↺ COMPASS RESET", fg=typer.colors.YELLOW, dim=True)

    def _on_press(key):
        if stop_keyboard_listener.is_set():
            return
        if key == Key.enter and typing:
            loop.call_soon_threadsafe(asyncio.create_task, _send_and_reset_on_enter())
        if stop_event and hasattr(key, "char") and key.char == "q":
            loop.call_soon_threadsafe(stop_event.set)

    listener = Listener(on_press=_on_press)
    listener.start()

    try:
        async for sample in gaze_stream:
            if stop_event and stop_event.is_set():
                break

            window.append(sample)
            while window and (sample.timestamp - window[0].timestamp) > sample_time_length_s:
                window.popleft()

            event = detect_gestures(list(window), detector_state, compass_config)
            if event:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                color = typer.colors.GREEN
                if event.type.value == "UP":
                    color = typer.colors.CYAN
                elif event.type.value == "DOWN":
                    color = typer.colors.MAGENTA
                elif event.type.value == "LEFT":
                    color = typer.colors.YELLOW
                elif event.type.value == "RIGHT":
                    color = typer.colors.BLUE

                typer.secho(
                    f"[{timestamp}] GESTURE: {event.type.value:5} | Mag: {event.magnitude:6.2f} | Dur: {event.duration:4.2f}s",
                    fg=color,
                    bold=True,
                )

                if typing and compass_obj is not None:
                    gesture_to_direction = {"UP": "up", "RIGHT": "right", "DOWN": "down", "LEFT": "left"}
                    direction = gesture_to_direction.get(event.type.value)
                    if direction:
                        compass_obj.process_input(direction)
                        status = f"  └─ COMPASS: [ {compass_obj.current_text or '...'} ]"
                        typer.secho(status, fg=typer.colors.WHITE, dim=True)

                if event.type.value in key_map:
                    key = key_map[event.type.value]
                    keyboard.press(key)
                    keyboard.release(key)
    except asyncio.CancelledError:
        pass
    finally:
        stop_keyboard_listener.set()
        listener.stop()


async def run_compass_loop(
    neon_config,
    compass_config: CompassSettings,
    dry_run: bool,
    typing: bool = False,
):
    """
    Main event loop for the compass command.
    """
    if dry_run:
        device_name = "Synthetic Device"
        gaze_stream = synthetic_gaze_stream()
        device = None
    else:
        try:
            device = await connect(neon_config)
            gaze_stream = stream_gaze(device)
            device_name = "Neon Device"
        except Exception as e:
            logger.error(f"Failed to connect to Neon: {e}")
            raise typer.Exit(code=1) from e

    typer.echo("\n" + "=" * 60)
    typer.secho(f" COMPASS GESTURE DETECTION: {device_name} ", fg=typer.colors.BLACK, bg=typer.colors.WHITE, bold=True)
    typer.echo("=" * 60)
    typer.echo(f"Source: {device_name}")
    typer.echo(f"Typing Mode: {'ENABLED' if typing else 'DISABLED'}")
    if typing:
        typer.secho("Instructions: Perform gestures (UP, DOWN, LEFT, RIGHT) to navigate.", fg=typer.colors.CYAN)
        typer.secho("              Press ENTER on your keyboard to expand/send.", fg=typer.colors.CYAN)
    typer.echo("-" * 60)
    typer.secho("Press Ctrl+C to stop.", fg=typer.colors.YELLOW)
    typer.echo("-" * 60 + "\n")

    try:
        await process_compass_gestures(
            gaze_stream=gaze_stream,
            compass_config=compass_config,
            typing=typing,
        )
    finally:
        if device:
            await device.close()
        logger.info("Shutting down...")


def compass(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run with synthetic data"),
    typing: bool = typer.Option(
        False,
        "--typing",
        help="Enable typing mode (gesture-driven Compass). Press Enter to reset the Compass state.",
    ),
):
    """
    Start the gesture detection loop.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    neon_config = load_neon_config(config_path)
    gestures_config = load_gestures_config(config_path)

    try:
        asyncio.run(run_compass_loop(neon_config, gestures_config.compass, dry_run, typing))
    except KeyboardInterrupt:
        pass
