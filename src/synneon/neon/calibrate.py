import asyncio
import select
import sys
import termios
import threading
import tomllib
import tty
from pathlib import Path

import typer

from synneon.neon.session import connect, stream_gaze
from synneon.schemas.neons import NeonConfig

app = typer.Typer()


def load_neons_config(config_path: Path) -> NeonConfig:
    """
    Load a neons TOML config file into a NeonConfig.

    Handles two layouts:
      - Neons-specific:  [device] address = "..." / port = ...
      - Main config:     [neons] client_addr = "..."
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Neons-specific file with [device] section
    if "device" in data:
        device = data["device"]
        addr = device.get("address", "neon.local")
        port = device.get("port", 8080)
        return NeonConfig(client_addr=f"{addr}:{port}", client_url=f"{addr}:{port}")

    # Main config layout with [neons] section
    if "neons" in data:
        return NeonConfig(**data["neons"])

    return NeonConfig()


def load_calibration(config_path: Path) -> dict[str, float] | None:
    """
    Read existing calibration bounds from the [calibration] section in the config TOML.

    Returns a dict with keys min_x, max_x, min_y, max_y, or None if missing.
    """
    if not config_path.exists():
        return None

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    cal = data.get("calibration")
    if not cal:
        return None

    required = {"min_x", "max_x", "min_y", "max_y"}
    if not required.issubset(cal):
        return None

    return {k: float(cal[k]) for k in required}


def save_calibration(config_path: Path, data: dict[str, float]):
    """
    Write calibration bounds into the [calibration] section of the config TOML.

    Preserves all other sections by doing a targeted text replacement.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cal_block = (
        "[calibration]\n"
        f"min_x = {data['min_x']}\n"
        f"max_x = {data['max_x']}\n"
        f"min_y = {data['min_y']}\n"
        f"max_y = {data['max_y']}\n"
    )

    if not config_path.exists():
        config_path.write_text(cal_block)
        return

    lines = config_path.read_text().splitlines()
    new_lines: list[str] = []
    in_cal_section = False
    cal_section_replaced = False

    for line in lines:
        stripped = line.strip()

        if stripped == "[calibration]":
            in_cal_section = True
            cal_section_replaced = True
            new_lines.extend(cal_block.splitlines())
            continue

        if in_cal_section:
            if stripped.startswith("[") and stripped.endswith("]"):
                in_cal_section = False
                new_lines.append(line)
            continue

        new_lines.append(line)

    if not cal_section_replaced:
        if new_lines and new_lines[-1] != "":
            new_lines.append("")
        new_lines.extend(cal_block.splitlines())

    config_path.write_text("\n".join(new_lines) + "\n")


def _is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def _input_thread(stop_event: threading.Event):
    """Listen for 'q' key press in raw mode to signal stopping."""
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not stop_event.is_set():
            if _is_data():
                c = sys.stdin.read(1)
                if c.lower() == "q":
                    stop_event.set()
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


async def run_calibration(neon_config: NeonConfig, config_path: Path):
    """Connect to Neon and track min/max gaze coordinates."""
    typer.secho(f"Connecting to Neon at {neon_config.client_addr}...", fg=typer.colors.CYAN)
    try:
        device = await connect(neon_config)
    except Exception as e:
        typer.secho(f"Failed to connect: {e}", fg=typer.colors.RED)
        return

    typer.secho("Connected!", fg=typer.colors.GREEN, bold=True)
    typer.echo("\n" + "=" * 60)
    typer.secho(" CALIBRATION STARTED ", fg=typer.colors.BLACK, bg=typer.colors.WHITE, bold=True)
    typer.echo("=" * 60)
    typer.echo("1. Move your eyes to the extreme edges of your field of view.")
    typer.echo("2. Look as far UP, DOWN, LEFT, and RIGHT as possible.")
    typer.echo("3. The bounds below will update as you reach new extremes.")
    typer.echo("-" * 60)
    typer.secho("Press 'q' to SAVE bounds and QUIT.", fg=typer.colors.YELLOW, bold=True)
    typer.echo("-" * 60 + "\n")

    min_x = float("inf")
    max_x = -float("inf")
    min_y = float("inf")
    max_y = -float("inf")

    stop_event = threading.Event()
    t = threading.Thread(target=_input_thread, args=(stop_event,), daemon=True)
    t.start()

    try:
        async for gaze in stream_gaze(device):
            if stop_event.is_set():
                break
            if not gaze.is_worn:
                continue

            updated = False
            if gaze.x < min_x:
                min_x = gaze.x
                updated = True
            if gaze.x > max_x:
                max_x = gaze.x
                updated = True
            if gaze.y < min_y:
                min_y = gaze.y
                updated = True
            if gaze.y > max_y:
                max_y = gaze.y
                updated = True

            if updated:
                # Use a cleaner format for live updates
                msg = f"\r[LIVE] X: [{min_x:8.2f} ↔ {max_x:8.2f}]  Y: [{min_y:8.2f} ↔ {max_y:8.2f}]"
                typer.secho(msg, nl=False, fg=typer.colors.MAGENTA)
    except Exception as e:
        typer.secho(f"\nError during streaming: {e}", fg=typer.colors.RED)
    finally:
        stop_event.set()
        t.join(timeout=1.0)

    typer.echo("\n\n" + "-" * 60)
    typer.secho("Stopping calibration and saving data...", fg=typer.colors.CYAN)

    if min_x == float("inf"):
        typer.secho("No gaze data collected. Skipping save.", fg=typer.colors.YELLOW)
        return

    calibration_data = {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
    }

    try:
        save_calibration(config_path, calibration_data)
        typer.secho(f"✓ Calibration successfully saved to: {config_path.resolve()}", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        typer.secho(f"Error saving calibration: {e}", fg=typer.colors.RED)


def get_default_config_path() -> Path:
    """Resolve the default config path: local first, then package-relative."""
    local_path = Path("configs/neons/default.toml")
    if local_path.exists():
        return local_path
    return Path(__file__).resolve().parents[3] / "configs" / "neons" / "default.toml"


@app.command()
def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to neons config TOML file",
    ),
):
    """Connect to Neon glasses and calibrate max eye positions."""
    if config is None:
        config = get_default_config_path()

    try:
        neon_config = load_neons_config(config)
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading config: {e}")
        raise typer.Exit(code=1) from e

    existing = load_calibration(config)
    if existing:
        typer.secho("Found existing calibration bounds:", fg=typer.colors.CYAN)
        typer.echo(f"  X: {existing['min_x']:.2f} to {existing['max_x']:.2f}")
        typer.echo(f"  Y: {existing['min_y']:.2f} to {existing['max_y']:.2f}")
        typer.echo("")

    try:
        asyncio.run(run_calibration(neon_config, config))
    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user.")


if __name__ == "__main__":
    app()
