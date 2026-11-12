"""`synneon app`: the dashboard. Serves the REST API, the `/ws` WebSocket hub, and the built React UI.

It does not touch hardware: run `synneon client` (real Neon or `--mock`) to feed it live data.
"""

import logging
from pathlib import Path

import typer
import uvicorn
from fastapi.staticfiles import StaticFiles

from synneon.server import app as api_app

logger = logging.getLogger(__name__)

DIST_DIR = Path(__file__).parent.parent / "ux" / "dist"


def run_app(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host to bind to. Use 0.0.0.0 only on a trusted network: there is no auth."
    ),
    port: int = typer.Option(8765, help="Port to run the dashboard on"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """
    Run the SynNeon Dashboard: REST API, /ws, and the React frontend on /.
    Start `synneon client` separately for Neon (or mock) data.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Mounted last so the API and WebSocket routes take precedence.
    if DIST_DIR.exists():
        api_app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
    else:
        logger.warning(f"Static directory {DIST_DIR} not found. Did you run 'npm run build'?")

    logger.info(f"Starting SynNeon Dashboard on {host}:{port}")
    uvicorn.run(api_app, host=host, port=port, log_level="debug" if debug else "info")
