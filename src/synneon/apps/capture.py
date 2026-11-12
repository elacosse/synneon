from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from pupil_labs.realtime_api.streaming import BlinkEventData
from pynput.keyboard import Controller, Listener

from synneon.apps.shared.blink import BlinkManager
from synneon.apps.shared.config import load_blink_config, load_neon_config
from synneon.neon.session import connect, stream_eye_events, stream_gaze, stream_video

logger = logging.getLogger("synneon")


async def process_capture(
    device,
    blink_config,
    output_dir: Path,
    stop_event: asyncio.Event,
):
    """
    Core logic for blink detection and snapshot capture.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    keyboard = Controller()
    blink_manager = BlinkManager(blink_config, keyboard)

    eye_stream = stream_eye_events(device)
    video_stream = stream_video(device)
    gaze_stream = stream_gaze(device)

    async def update_gaze():
        async for gaze in gaze_stream:
            blink_manager.update_data(frame=None, gaze=gaze if gaze.is_worn else None)

    async def update_video():
        async for frame in video_stream:
            blink_manager.update_data(frame=frame, gaze=None)

    async def process_eyes():
        async for event in eye_stream:
            if isinstance(event, BlinkEventData):
                ts = event.start_time_ns / 1e9
                await blink_manager.handle_blink(ts, output_dir)

    def _on_press(key):
        if hasattr(key, "char") and key.char == "q":
            loop.call_soon_threadsafe(stop_event.set)

    listener = Listener(on_press=_on_press)
    listener.start()

    gaze_task = asyncio.create_task(update_gaze())
    video_task = asyncio.create_task(update_video())
    eyes_task = asyncio.create_task(process_eyes())
    stop_task = asyncio.create_task(stop_event.wait())

    try:
        await asyncio.wait([gaze_task, video_task, eyes_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
    finally:
        listener.stop()
        for t in [gaze_task, video_task, eyes_task, stop_task]:
            t.cancel()
        await asyncio.gather(gaze_task, video_task, eyes_task, stop_task, return_exceptions=True)


def capture(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    output_dir: Path = typer.Option(Path("captures"), "--output", "-o", help="Directory to save captures"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """
    Detect double blinks and capture a snapshot with gaze overlay.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    neon_config = load_neon_config(config_path)
    blink_config = load_blink_config(config_path)

    async def run_capture() -> None:
        device = await connect(neon_config)
        stop_event = asyncio.Event()

        typer.secho("\n" + "=" * 60)
        typer.secho(" CAPTURE MODE: Blink Actions ", fg=typer.colors.BLACK, bg=typer.colors.WHITE, bold=True)
        typer.echo("=" * 60)
        typer.echo(f"Output Directory: {output_dir}")
        typer.echo("-" * 60)
        typer.secho("Press 'q' or Ctrl+C to stop.", fg=typer.colors.YELLOW)
        typer.echo("-" * 60 + "\n")

        try:
            await process_capture(
                device=device,
                blink_config=blink_config,
                output_dir=output_dir,
                stop_event=stop_event,
            )
        finally:
            await device.close()
            logger.info("Shutting down...")

    try:
        asyncio.run(run_capture())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
