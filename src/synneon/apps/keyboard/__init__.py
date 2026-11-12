"""
Keyboard sub-app: Compass Typing terminal interface.

Exposes ``keyboard_app`` — a Typer sub-app with a single command,
``type_text``, that launches the Rich keyboard UI.
"""

import typer

from synneon.apps.keyboard.command import type_text

keyboard_app = typer.Typer(
    name="keyboard",
    help="Compass Typing",
    add_completion=False,
)

keyboard_app.command()(type_text)
