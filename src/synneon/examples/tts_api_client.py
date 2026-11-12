import tomllib
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

from synneon.schemas.tts import TTSConfig

app = typer.Typer(help="Synneon TTS/STT REST Client")
console = Console()

DEFAULT_CONFIG_PATH = Path("configs/tts/default.toml")
DEFAULT_BASE_URL = "http://localhost:8000"


def load_tts_config(config_path: Path, profile: str) -> TTSConfig:
    """Load a TTSConfig from a TOML file for the given profile."""
    if not config_path.exists():
        alt_path = Path(__file__).parent.parent.parent.parent / config_path
        if not alt_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        config_path = alt_path

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    audio_data = data.get("tts", {}).get(profile)
    if not audio_data:
        raise ValueError(f"Profile '{profile}' not found in {config_path}")

    return TTSConfig(**audio_data)


@app.command()
def speak(
    text: str = typer.Argument(..., help="Text to synthesise into speech"),
    output: Path = typer.Option(Path("speech.mp3"), "--output", "-o", help="Path to save the MP3 file"),
    profile: str = typer.Option("default", "--profile", "-p", help="TTS configuration profile"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to TTS TOML config"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--url", "-u", help="Base URL of the Synneon server"),
    voice: str = typer.Option(None, "--voice", "-v", help="Override the voice from config"),
):
    """
    Synthesise text to speech via the REST API and save the result as an MP3.

    Voice and model are loaded from the TTS config profile; pass --voice to override.
    """
    try:
        settings = load_tts_config(config_path, profile)
    except Exception as e:
        console.print(f"[bold red]Config error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    resolved_voice = voice or settings.voice or "helpful woman"

    console.print(
        Panel.fit(
            f"[bold blue]Endpoint:[/bold blue] {base_url}/audio/speak\n"
            f"[bold blue]Profile:[/bold blue] {profile}\n"
            f"[bold blue]Model:[/bold blue] {settings.model}\n"
            f"[bold blue]Voice:[/bold blue] {resolved_voice}\n"
            f"[bold blue]Text:[/bold blue] {text}",
            title="TTS Speak Request",
        )
    )

    params = {"text": text, "profile": profile}
    if voice:
        params["voice"] = voice

    try:
        with httpx.Client(timeout=60.0) as client:
            console.print(f"\n[yellow]Sending request to {base_url}/audio/speak...[/yellow]")
            response = client.post(f"{base_url}/audio/speak", json=params)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]Server error ({e.response.status_code}):[/bold red] {e.response.text}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Connection error:[/bold red] {e}")
        console.print("[dim]Make sure the server is running with 'uv run python -m synneon.server'[/dim]")
        raise typer.Exit(code=1) from e

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)
    console.print(f"\n[bold green]Audio saved to:[/bold green] {output}")
    console.print(f"[dim]{len(response.content):,} bytes[/dim]")


@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Path to the audio file to transcribe"),
    profile: str = typer.Option("default", "--profile", "-p", help="TTS configuration profile"),
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to TTS TOML config"),
    base_url: str = typer.Option(DEFAULT_BASE_URL, "--url", "-u", help="Base URL of the Synneon server"),
):
    """
    Transcribe an audio file via the REST API using the configured STT model.

    Model and language are loaded from the TTS config profile.
    """
    if not audio_file.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file {audio_file} does not exist.")
        raise typer.Exit(code=1)

    try:
        settings = load_tts_config(config_path, profile)
    except Exception as e:
        console.print(f"[bold red]Config error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    console.print(
        Panel.fit(
            f"[bold blue]Endpoint:[/bold blue] {base_url}/audio/transcribe\n"
            f"[bold blue]Profile:[/bold blue] {profile}\n"
            f"[bold blue]Model:[/bold blue] {settings.model}\n"
            f"[bold blue]Language:[/bold blue] {settings.language}\n"
            f"[bold blue]File:[/bold blue] {audio_file}",
            title="STT Transcribe Request",
        )
    )

    try:
        with httpx.Client(timeout=120.0) as client:
            console.print(f"\n[yellow]Uploading {audio_file.name} to {base_url}/audio/transcribe...[/yellow]")
            with open(audio_file, "rb") as f:
                response = client.post(
                    f"{base_url}/audio/transcribe",
                    files={"file": (audio_file.name, f, "audio/mpeg")},
                    params={"profile": profile},
                )
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]Server error ({e.response.status_code}):[/bold red] {e.response.text}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[bold red]Connection error:[/bold red] {e}")
        console.print("[dim]Make sure the server is running with 'uv run python -m synneon.server'[/dim]")
        raise typer.Exit(code=1) from e

    transcript = result.get("text", "")
    console.print(
        Panel(
            transcript or "[dim](empty)[/dim]",
            title="[bold green]Transcription[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
