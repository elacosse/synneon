from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from synneon.apps.shared.config import load_neon_config
from synneon.neon.session import _get_rtsp_url, connect
from synneon.services.physiology.pupilometry import run_realtime_pupilometry_plot

logger = logging.getLogger("synneon")


def pupil(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    window_s: float = typer.Option(30.0, "--window-s", help="Sliding window duration in seconds"),
    refresh_hz: float = typer.Option(20.0, "--refresh-hz", help="Plot refresh rate (redraw cadence) in Hz"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """
    Plot real-time pupillometry (left/right pupil diameter in mm) from a Neon device.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    neon_config = load_neon_config(config_path)

    async def run_pupil() -> None:
        device = await connect(neon_config)
        try:
            rtsp_url = await _get_rtsp_url(device, "gaze")
            await run_realtime_pupilometry_plot(
                rtsp_url=rtsp_url,
                window_s=window_s,
                refresh_hz=refresh_hz,
            )
        finally:
            await device.close()

    try:
        asyncio.run(run_pupil())
    except KeyboardInterrupt:
        pass
