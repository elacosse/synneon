"""Audio capture CLI: record from a chosen microphone to a WAV file."""

import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import typer
from pynput import keyboard
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

app = typer.Typer(help="Record audio from a microphone to a WAV file.")
console = Console()

SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = "int16"
WAVEFORM_WIDTH = 60  # number of columns in the waveform bar
WAVEFORM_HEIGHT = 8  # number of rows tall
CHUNK = 1024

# Block elements from thin → full, used for vertical amplitude mapping
_BLOCKS = " ▁▂▃▄▅▆▇█"


def _list_input_devices() -> list[tuple[int, str]]:
    devices = sd.query_devices()
    return [
        (i, d["name"])
        for i, d in enumerate(devices)  # type: ignore[arg-type]
        if d["max_input_channels"] > 0  # type: ignore[index]
    ]


def _choose_device(devices: list[tuple[int, str]]) -> int:
    console.print("\n[bold cyan]Available microphones:[/bold cyan]")
    for idx, (dev_id, name) in enumerate(devices):
        console.print(f"  [yellow]{idx + 1}[/yellow]. {name}  [dim](id={dev_id})[/dim]")

    while True:
        raw = typer.prompt("\nSelect microphone number")
        try:
            choice = int(raw)
            if 1 <= choice <= len(devices):
                return devices[choice - 1][0]
        except ValueError:
            pass
        console.print("[red]Invalid choice, try again.[/red]")


def _render_waveform(samples: np.ndarray, elapsed: float) -> Panel:
    """Build a Rich Panel containing a scrolling waveform and elapsed time."""
    # Normalise to [-1, 1]
    if samples.size == 0:
        amplitudes = [0.0] * WAVEFORM_WIDTH
    else:
        norm = samples.astype(np.float32) / 32768.0
        # Downsample/upsample to exactly WAVEFORM_WIDTH columns
        indices = np.linspace(0, len(norm) - 1, WAVEFORM_WIDTH).astype(int)
        amplitudes = np.abs(norm[indices]).tolist()

    # Build a multi-row waveform: each column is a vertical bar
    rows: list[Text] = []
    for row in range(WAVEFORM_HEIGHT, 0, -1):
        threshold = row / WAVEFORM_HEIGHT  # fraction this row represents
        line = Text()
        for amp in amplitudes:
            # How much of this cell is filled?
            fill = (amp - (threshold - 1 / WAVEFORM_HEIGHT)) * WAVEFORM_HEIGHT
            fill = max(0.0, min(1.0, fill))
            block_idx = int(fill * (len(_BLOCKS) - 1))
            char = _BLOCKS[block_idx]
            # Colour: green → yellow → red as amplitude rises
            if amp < 0.4:
                colour = "bright_green"
            elif amp < 0.75:
                colour = "yellow"
            else:
                colour = "bright_red"
            line.append(char, style=colour)
        rows.append(line)

    # Elapsed time line
    elapsed_text = Text(justify="center")
    elapsed_text.append("  ● REC  ", style="bold red")
    elapsed_text.append(f"{elapsed:.1f}s", style="bold white")
    elapsed_text.append("  —  press ", style="dim")
    elapsed_text.append("SPACE", style="bold cyan")
    elapsed_text.append(" to stop", style="dim")

    combined = Text("\n").join(rows) + Text("\n\n") + elapsed_text

    return Panel(
        combined,
        title="[bold red]Recording[/bold red]",
        border_style="red",
        expand=False,
        width=WAVEFORM_WIDTH + 4,
    )


@app.command()
def record(
    output: Path = typer.Argument(..., help="Output WAV file path (e.g. recording.wav)"),
):
    """Interactively choose a microphone, then press SPACE to start/stop recording."""

    devices = _list_input_devices()
    if not devices:
        console.print("[red]No input devices found.[/red]")
        raise typer.Exit(1)

    device_id = _choose_device(devices)
    device_name = next(name for did, name in devices if did == device_id)
    console.print(f"\n[green]Using:[/green] {device_name}")
    console.print("\n[bold]Press [cyan]SPACE[/cyan] to start recording.[/bold]")

    # --- shared state ---
    recording = threading.Event()
    frames: list[bytes] = []
    start_time: list[float] = []
    latest_chunk: list[np.ndarray] = [np.zeros(CHUNK, dtype=np.int16)]
    lock = threading.Lock()

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.space:
            if not recording.is_set():
                recording.set()
                start_time.append(time.monotonic())
            else:
                recording.clear()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Wait for first spacebar press
    recording.wait()

    audio_done = threading.Event()

    def _capture() -> None:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device_id,
            blocksize=CHUNK,
        ) as stream:
            while recording.is_set():
                data, _ = stream.read(CHUNK)
                chunk = data[:, 0] if data.ndim > 1 else data.flatten()
                with lock:
                    latest_chunk[0] = chunk.copy()
                frames.append(data.tobytes())
        audio_done.set()

    capture_thread = threading.Thread(target=_capture, daemon=True)
    capture_thread.start()

    with Live(console=console, refresh_per_second=20) as live:
        while recording.is_set():
            with lock:
                chunk = latest_chunk[0].copy()
            elapsed = time.monotonic() - start_time[0]
            live.update(_render_waveform(chunk, elapsed))
            time.sleep(0.05)

    audio_done.wait()
    listener.stop()

    # Save to WAV
    output = output.with_suffix(".wav")
    with wave.open(str(output), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    console.print(f"\n[bold green]✓ Saved →[/bold green] [cyan]{output.resolve()}[/cyan]")


if __name__ == "__main__":
    app()
