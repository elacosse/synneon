from __future__ import annotations

import logging

import typer
from dotenv import load_dotenv

from synneon.apps.app import run_app
from synneon.apps.capture import capture
from synneon.apps.combined import capture_and_gesture
from synneon.apps.display import display_app
from synneon.apps.gesture import gesture_app
from synneon.apps.keyboard import keyboard_app
from synneon.apps.neon_client import run_client
from synneon.apps.pupil import pupil
from synneon.apps.run_example import app as example_app
from synneon.apps.video.app import video_app
from synneon.neon.calibrate import main as calibrate_main

load_dotenv()

app = typer.Typer(help="SynNeon - CCI CLI")

app.add_typer(example_app, name="example")
app.add_typer(gesture_app, name="gesture")
app.add_typer(display_app, name="display")
app.add_typer(keyboard_app, name="keyboard")
app.add_typer(video_app, name="video")

app.command()(pupil)
app.command()(capture)
app.command(name="capture-and-gesture")(capture_and_gesture)
app.command(name="calibrate")(calibrate_main)
app.command(name="app")(run_app)
app.command(name="client")(run_client)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    app()


if __name__ == "__main__":
    typer.run(main)
