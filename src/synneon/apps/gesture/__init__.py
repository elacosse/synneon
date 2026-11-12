import typer

gesture_app = typer.Typer(help="Gesture recognition mode")

from synneon.apps.gesture import compass, mouse  # noqa: E402, F401

gesture_app.command()(compass.compass)
gesture_app.command()(mouse.mouse)
