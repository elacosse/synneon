import asyncio
import base64
import tempfile
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(help="Test client for all SynNeon REST API endpoints.")
console = Console()

# A minimal valid 1x1 PNG image for testing endpoints without needing a real file.
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f"
    b"\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def _run_tests(
    base_url: str,
    test_health: bool = True,
    test_expand: bool = True,
    test_image: bool = True,
    test_image_b64: bool = True,
    test_tts: bool = True,
    test_stt: bool = True,
    test_stt_b64: bool = True,
    image_path: Path | None = None,
    audio_path: Path | None = None,
):
    console.print(Panel.fit(f"Testing SynNeon API at [bold cyan]{base_url}[/bold cyan]"))

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        # 1. Health Check
        if test_health:
            console.print("\n[bold yellow]1. Testing GET /health[/bold yellow]")
        try:
            resp = await client.get("/health")
            resp.raise_for_status()
            console.print(f"[green]Success![/green] {resp.json()}")
        except Exception as e:
            console.print(f"[red]Failed:[/red] {e}")

        # 2. Abbreviation Expansion
        if test_expand:
            console.print("\n[bold yellow]2. Testing POST /inference/expand[/bold yellow]")
            try:
                payload = {
                    "context": {},  # Assuming empty context is acceptable or has defaults
                    "tokens": "h r u",
                    "profile": "default",
                }
                resp = await client.post("/inference/expand", json=payload)
                resp.raise_for_status()
                console.print(f"[green]Success![/green] {resp.json()}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
                if isinstance(e, httpx.HTTPStatusError):
                    console.print(e.response.text)

        import mimetypes

        if image_path and image_path.exists():
            image_data = image_path.read_bytes()
            image_filename = image_path.name
            image_mimetype = mimetypes.guess_type(image_filename)[0] or "image/png"
        else:
            image_data = MINIMAL_PNG
            image_filename = "test.png"
            image_mimetype = "image/png"

        # 3. Image to Context (Multipart)
        if test_image:
            console.print("\n[bold yellow]3. Testing POST /inference/image2context (Multipart)[/bold yellow]")
            try:
                files = {
                    "file": (image_filename, image_data, image_mimetype),
                }
                params = {
                    "profile": "image2context",
                }
                resp = await client.post("/inference/image2context", files=files, params=params)
                resp.raise_for_status()
                console.print(f"[green]Success![/green] {resp.json()}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
                if isinstance(e, httpx.HTTPStatusError):
                    console.print(e.response.text)

        # 4. Image to Context (Base64)
        if test_image_b64:
            console.print("\n[bold yellow]4. Testing POST /inference/image2context/base64[/bold yellow]")
            try:
                b64_image = base64.b64encode(image_data).decode("utf-8")
                payload = {
                    "image_base64": b64_image,
                    "media_type": image_mimetype,
                    "profile": "image2context",
                }
                resp = await client.post("/inference/image2context/base64", json=payload)
                resp.raise_for_status()
                console.print(f"[green]Success![/green] {resp.json()}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
                if isinstance(e, httpx.HTTPStatusError):
                    console.print(e.response.text)

        # 5. Text to Speech (TTS)
        audio_bytes = None
        tmp_path = None
        if test_tts:
            console.print("\n[bold yellow]5. Testing POST /audio/speak[/bold yellow]")
            try:
                params = {
                    "text": "Hello, this is a test of the text to speech API.",
                    "profile": "default",
                }
                resp = await client.post("/audio/speak", params=params)
                resp.raise_for_status()
                audio_bytes = resp.content
                console.print(f"[green]Success![/green] Received {len(audio_bytes)} bytes of audio.")

                # Save for STT testing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = Path(tmp.name)
                console.print(f"Saved TTS output to temporary file: {tmp_path}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
                if isinstance(e, httpx.HTTPStatusError):
                    console.print(e.response.text)

        # Determine audio source for STT
        stt_audio_path = None
        stt_audio_bytes = None
        stt_mimetype = "audio/wav"
        stt_filename = "test_audio.wav"

        if audio_path and audio_path.exists():
            stt_audio_path = audio_path
            stt_audio_bytes = audio_path.read_bytes()
            stt_filename = audio_path.name
            import mimetypes

            stt_mimetype = mimetypes.guess_type(stt_filename)[0] or "audio/wav"
            console.print(f"\n[dim]Using provided audio file for STT tests: {stt_audio_path}[/dim]")
        else:
            if (test_stt or test_stt_b64) and not audio_bytes:
                if not test_tts:
                    console.print("\n[dim]TTS test skipped. Using a dummy audio payload for STT tests.[/dim]")
                else:
                    console.print(
                        "\n[yellow]TTS failed or returned empty. Using a dummy audio payload for STT tests.[/yellow]"
                    )
                # Minimal dummy WAV header + silence
                audio_bytes = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_stt:
                    tmp_stt.write(audio_bytes)
                    tmp_path = Path(tmp_stt.name)

            stt_audio_bytes = audio_bytes
            stt_audio_path = tmp_path

        # 6. Speech to Text (Multipart)
        if test_stt and stt_audio_path:
            console.print("\n[bold yellow]6. Testing POST /audio/transcribe (Multipart)[/bold yellow]")
            try:
                with open(stt_audio_path, "rb") as f:
                    files = {
                        "file": (stt_filename, f, stt_mimetype),
                    }
                    params = {
                        "profile": "default",
                    }
                    resp = await client.post("/audio/transcribe", files=files, params=params)
                    resp.raise_for_status()
                    console.print(f"[green]Success![/green] Transcription: {resp.json()}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
                if isinstance(e, httpx.HTTPStatusError):
                    console.print(e.response.text)

        # 7. Speech to Text (Base64)
        if test_stt_b64 and stt_audio_bytes:
            console.print("\n[bold yellow]7. Testing POST /audio/transcribe/base64[/bold yellow]")
            try:
                b64_audio = base64.b64encode(stt_audio_bytes).decode("utf-8")
                payload = {
                    "audio_base64": b64_audio,
                    "media_type": stt_mimetype,
                    "profile": "default",
                }
                resp = await client.post("/audio/transcribe/base64", json=payload)
                resp.raise_for_status()
                console.print(f"[green]Success![/green] Transcription: {resp.json()}")
            except Exception as e:
                console.print(f"[red]Failed:[/red] {e}")
                if isinstance(e, httpx.HTTPStatusError):
                    console.print(e.response.text)

        # Cleanup
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
            console.print(f"\n[dim]Cleaned up temporary file: {tmp_path}[/dim]")

    console.print("\n[bold green]All tests completed![/bold green]")


@app.command(name="run")
def run_tests(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Base URL of the SynNeon API server"),
    health: bool = typer.Option(False, "--health", help="Run health check test"),
    expand: bool = typer.Option(False, "--expand", help="Run expand abbreviation test"),
    image: bool = typer.Option(False, "--image", help="Run image2context test (multipart)"),
    image_b64: bool = typer.Option(False, "--image-b64", help="Run image2context test (base64)"),
    tts: bool = typer.Option(False, "--tts", help="Run text-to-speech test"),
    stt: bool = typer.Option(False, "--stt", help="Run speech-to-text test (multipart)"),
    stt_b64: bool = typer.Option(False, "--stt-b64", help="Run speech-to-text test (base64)"),
    all_tests: bool = typer.Option(False, "--all", help="Run all tests explicitly"),
    image_path: Path = typer.Option(None, "--image-path", help="Path to an image file for image2context tests"),
    audio_path: Path = typer.Option(None, "--audio-path", help="Path to an audio file for transcribe tests"),
):
    """
    Run specific or all integration tests against the SynNeon API.
    If no specific tests are toggled, it will run all tests by default.
    """
    # If no specific tests are set, default to running all
    run_any = health or expand or image or image_b64 or tts or stt or stt_b64
    if all_tests or not run_any:
        health = expand = image = image_b64 = tts = stt = stt_b64 = True

    asyncio.run(
        _run_tests(
            base_url=url,
            test_health=health,
            test_expand=expand,
            test_image=image,
            test_image_b64=image_b64,
            test_tts=tts,
            test_stt=stt,
            test_stt_b64=stt_b64,
            image_path=image_path,
            audio_path=audio_path,
        )
    )


# Alias for backward compatibility if someone explicitly calls run-all
@app.command(name="run-all", hidden=True)
def run_all(url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Base URL of the SynNeon API server")):
    asyncio.run(_run_tests(url))


if __name__ == "__main__":
    app()
