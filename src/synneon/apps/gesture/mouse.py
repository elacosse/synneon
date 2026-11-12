from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import typer
from pynput.keyboard import Controller

from synneon.apps.shared.blink import key_map
from synneon.apps.shared.config import load_gestures_config
from synneon.input.detector import DetectorState, detect_gestures
from synneon.input.mouse import MouseStreamConfig, stream_mouse_gaze
from synneon.schemas.neons import GazeSample

logger = logging.getLogger("synneon")


def mouse(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    poll_hz: float = typer.Option(240.0, "--poll-hz", help="How often to poll the OS mouse pointer position"),
    emit_hz: float = typer.Option(60.0, "--emit-hz", help="How often to emit samples into the gesture detector"),
):
    """
    Track mouse pointer position as gaze-like input for gesture detection.

    This does not connect to a Neon device.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    gestures_config = load_gestures_config(config_path)

    async def run_mouse_loop() -> None:
        detector_state = DetectorState()
        compass_config = gestures_config.compass

        sample_time_length_s = float(getattr(compass_config, "sample_time_length", 0.5))
        if sample_time_length_s <= 0:
            logger.warning("gestures.compass.sample_time_length must be > 0; falling back to 0.5s")
            sample_time_length_s = 0.5

        trajectories: deque[list[GazeSample]] = deque()
        window: deque[GazeSample] = deque()

        keyboard = Controller()

        stream = stream_mouse_gaze(config=MouseStreamConfig(poll_hz=poll_hz, emit_hz=emit_hz))
        logger.info("Listening for gestures from Mouse Pointer...")

        async for sample in stream:
            window.append(sample)

            newest_ts = sample.timestamp
            while window and (newest_ts - window[0].timestamp) > sample_time_length_s:
                window.popleft()

            trajectories.append(list(window))

            while trajectories:
                trajectory = trajectories.popleft()
                event = detect_gestures(trajectory, detector_state, compass_config)

                if event:
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    typer.echo(
                        f"[{timestamp}] GESTURE DETECTED: {event.type.value} (Mag: {event.magnitude:.2f}, Dur: {event.duration:.2f}s)"
                    )

                    if event.type.value in key_map:
                        key = key_map[event.type.value]
                        logger.info(f"Pressing Key: {key}")
                        keyboard.press(key)
                        keyboard.release(key)

    try:
        asyncio.run(run_mouse_loop())
    except KeyboardInterrupt:
        pass
