from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from synneon.apps.shared.config import load_neon_config
from synneon.neon.session import connect
from synneon.visualization.visualizer import Visualizer

logger = logging.getLogger("synneon")

display_app = typer.Typer(help="Display mode")


@display_app.command()
def gaze(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """
    Show live video feed with gaze overlay.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    neon_config = load_neon_config(config_path)

    async def run_display():
        device = await connect(neon_config)
        try:
            viz = Visualizer(device)
            await viz.run()
        except Exception as e:
            logger.error(f"Display failed: {e}")
            raise typer.Exit(code=1) from e
        finally:
            await device.close()

    try:
        asyncio.run(run_display())
    except KeyboardInterrupt:
        pass
