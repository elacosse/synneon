from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path

import typer
from pupil_labs.realtime_api.streaming import BlinkEventData
from pynput.keyboard import Controller, Listener

from synneon.apps.shared.blink import BlinkManager, key_map
from synneon.apps.shared.config import load_blink_config, load_gestures_config, load_neon_config
from synneon.input.detector import DetectorState, detect_gestures
from synneon.neon.session import connect, stream_eye_events, stream_gaze, stream_video
from synneon.schemas.gestures import CompassSettings
from synneon.schemas.neons import BlinkConfig, GazeSample

logger = logging.getLogger("synneon")


async def run_combined_processor(
    device,
    compass_config: CompassSettings,
    blink_config: BlinkConfig,
    output_dir: Path,
    typing: bool,
    save: bool,
    stop_event: asyncio.Event,
):
    loop = asyncio.get_running_loop()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_state = DetectorState()
    keyboard = Controller()
    blink_manager = BlinkManager(blink_config, keyboard, save=save)

    eye_stream = stream_eye_events(device)
    video_stream = stream_video(device)
    gaze_stream = stream_gaze(device)

    sample_time_length_s = float(getattr(compass_config, "sample_time_length", 0.5))
    if sample_time_length_s <= 0:
        sample_time_length_s = 0.5
    window: deque[GazeSample] = deque()

    def _on_press(key):
        if hasattr(key, "char") and key.char == "q":
            loop.call_soon_threadsafe(stop_event.set)

    listener = Listener(on_press=_on_press)
    listener.start()

    async def update_video():
        async for frame in video_stream:
            blink_manager.update_data(frame=frame, gaze=None)

    async def update_eye_events():
        async for event in eye_stream:
            if isinstance(event, BlinkEventData):
                ts = event.start_time_ns / 1e9
                await blink_manager.handle_blink(ts, output_dir)

    async def update_gaze_and_gestures():
        async for sample in gaze_stream:
            blink_manager.update_data(frame=None, gaze=sample if sample.is_worn else None)
            window.append(sample)
            while window and (sample.timestamp - window[0].timestamp) > sample_time_length_s:
                window.popleft()
            event = detect_gestures(list(window), detector_state, compass_config)
            if event:
                timestamp_str = __import__("datetime").datetime.now().strftime("%H:%M:%S.%f")[:-3]
                color = typer.colors.GREEN
                if event.type.value == "UP":
                    color = typer.colors.CYAN
                elif event.type.value == "DOWN":
                    color = typer.colors.MAGENTA
                elif event.type.value == "LEFT":
                    color = typer.colors.YELLOW
                elif event.type.value == "RIGHT":
                    color = typer.colors.BLUE
                typer.secho(f"[{timestamp_str}] GESTURE: {event.type.value:5}", fg=color, bold=True)
                if typing and event.type.value in key_map:
                    key = key_map[event.type.value]
                    typer.secho(f"  └─ KEYBOARD: Pressing {key}", fg=typer.colors.WHITE, dim=True)
                    keyboard.press(key)
                    keyboard.release(key)

    tasks = [
        asyncio.create_task(t)
        for t in [update_video(), update_eye_events(), update_gaze_and_gestures(), stop_event.wait()]
    ]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        listener.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def capture_and_gesture(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    output_dir: Path = typer.Option(Path("captures"), "--output", "-o", help="Directory to save captures"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    typing: bool = typer.Option(
        False,
        "--typing",
        help="Enable typing mode (emit keyboard events for gestures).",
    ),
    save: bool = typer.Option(
        False,
        "--save",
        help="Save captured images to disk (default: use a temp file, discard after display).",
    ),
):
    """
    Combined mode: Detect double blinks for snapshots AND directional gestures for keyboard input.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    neon_config = load_neon_config(config_path)
    gestures_config = load_gestures_config(config_path)
    blink_config = load_blink_config(config_path)

    async def run_combined() -> None:
        device = await connect(neon_config)
        stop_event = asyncio.Event()

        typer.secho("\n" + "=" * 60)
        typer.secho(" CAPTURE & GESTURE MODE ", fg=typer.colors.BLACK, bg=typer.colors.WHITE, bold=True)
        typer.echo("=" * 60)
        typer.echo(f"Typing (Keyboard): {'ENABLED' if typing else 'DISABLED'}")
        typer.echo(f"Save captures: {'ENABLED' if save else 'DISABLED'}")
        typer.echo(f"Output: {output_dir}")
        typer.echo("-" * 60)
        typer.secho("Press 'q' or Ctrl+C to stop.", fg=typer.colors.YELLOW)
        typer.echo("-" * 60 + "\n")

        try:
            await run_combined_processor(
                device, gestures_config.compass, blink_config, output_dir, typing, save, stop_event
            )
        finally:
            await device.close()
            logger.info("Shutting down...")

    try:
        asyncio.run(run_combined())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
