"""
Redis seeder — simulates a full new-user session against a live server and populates
Redis with realistic data so you can inspect key shapes in Redis Insight.

Usage:
    # Synthetic mode — no files needed
    uv run python -m synneon.examples.seed_redis

    # Sample mode — loads real capture data
    uv run python -m synneon.examples.seed_redis --data examples/capture_sample_v1

    # Custom user ID
    uv run python -m synneon.examples.seed_redis -id test_abc
"""

from __future__ import annotations

import asyncio
import base64
import csv
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Populate Redis with a realistic simulated user session.")
console = Console()


# ---------------------------------------------------------------------------
# Data payload
# ---------------------------------------------------------------------------


@dataclass
class DataPayload:
    scene_image_b64: str  # middle frame of video, JPEG base64
    audio_b64: str  # WAV base64
    video_frames: list[tuple[float, str]]  # [(timestamp_s, jpeg_b64), ...]
    gaze_points: list[tuple[float, float, float]]  # [(ts, x, y), ...]
    pupil_data: list[tuple[float, float, float]]  # [(ts, left_mm, right_mm), ...]
    conversation: list[tuple[str, str]]  # [(speaker_label, text), ...] — empty → use STT


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_synthetic_jpeg() -> bytes:
    """Generate a minimal 10×10 red JPEG using numpy + OpenCV."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # BGR red
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_synthetic_wav() -> bytes:
    """Build a minimal valid WAV: 0.1 s silence, 8 kHz mono 16-bit."""
    sample_rate = 8000
    n_samples = sample_rate // 10
    pcm = b"\x00\x00" * n_samples
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def _make_synthetic_data() -> DataPayload:
    jpeg = _make_synthetic_jpeg()
    jpeg_b64 = base64.b64encode(jpeg).decode()
    wav_b64 = base64.b64encode(_make_synthetic_wav()).decode()

    # 5 frames at 0.5 s intervals; middle frame (index 2) is the scene image
    video_frames = [(i * 0.5, jpeg_b64) for i in range(5)]
    scene_image_b64 = video_frames[len(video_frames) // 2][1]

    gaze_points: list[tuple[float, float, float]] = [
        (0.0, 640.0, 400.0),
        (0.5, 700.0, 420.0),
        (1.0, 680.0, 390.0),
        (1.5, 720.0, 410.0),
        (2.0, 660.0, 395.0),
    ]
    pupil_data: list[tuple[float, float, float]] = [
        (0.0, 3.0, 3.1),
        (0.5, 3.1, 3.2),
        (1.0, 2.9, 3.0),
        (1.5, 3.0, 3.1),
        (2.0, 3.1, 3.0),
    ]
    conversation: list[tuple[str, str]] = [
        ("SPEAKER_1", "Good morning Alice, how was the commute today?"),
        ("SPEAKER_2", "Not bad, trains were actually on time for once. How about you?"),
        ("SPEAKER_1", "Drove in, traffic was surprisingly light. Ready for the sprint review?"),
        ("SPEAKER_2", "Almost. I just need to finish the slide for the new memory endpoint."),
        ("SPEAKER_1", "Did you see the Redis Insight dashboard? The key shapes look really clean."),
        ("SPEAKER_2", "Yeah, the sorted sets for conversation history worked out nicely."),
        ("SPEAKER_1", "The score-based ordering makes windowed queries super fast."),
        ("SPEAKER_2", "One thing I noticed — the speaker back-fill logic took a bit to click."),
        ("SPEAKER_1", "Right, diarization labels come in as SPEAKER_1 and SPEAKER_2 generically."),
        ("SPEAKER_2", "Then we map them to real people after the fact. Clever but easy to miss."),
        ("SPEAKER_1", "Have you tested the neon-context endpoint with live capture data yet?"),
        ("SPEAKER_2", "Not yet. Waiting for the Pupil Labs device to arrive."),
        ("SPEAKER_1", "Should be here by Thursday. I ordered the updated model with the new sensor."),
        ("SPEAKER_2", "Perfect timing. The fixation algorithm needs real gaze data to tune properly."),
        ("SPEAKER_1", "For now the synthetic data is good enough to verify the pipeline end to end."),
        ("SPEAKER_2", "True. The saliency map output looked reasonable even on the test frames."),
        ("SPEAKER_1", "Gaussian sigma at sixty works well for the resolution we are targeting."),
        ("SPEAKER_2", "I bumped the min peak threshold to avoid noise spikes at the image borders."),
        ("SPEAKER_1", "Good call. Those were throwing off the focused object detection last week."),
        ("SPEAKER_2", "Should we add a config option to clamp gaze points to the image bounds?"),
        ("SPEAKER_1", "Good idea. Let me add it to the backlog after the review."),
        ("SPEAKER_2", "Also, the expand endpoint — should it surface memory summaries to the user?"),
        ("SPEAKER_1", "I think so. Showing context snippets would help with longer sessions."),
        ("SPEAKER_2", "We could render them as tooltips in the AAC overlay."),
        ("SPEAKER_1", "Or as a collapsible panel. Let's mock it up and see which feels less cluttered."),
        ("SPEAKER_2", "Sounds good. I'll sketch it out this afternoon."),
        ("SPEAKER_1", "One more thing — the Redis TTL for conversation entries, should we keep it at seven days?"),
        ("SPEAKER_2", "Maybe extend it to thirty for the user profile and speaker registry."),
        ("SPEAKER_1", "Makes sense. Speakers don't change often so longer retention is fine."),
        ("SPEAKER_2", "Alright, let's get set up for the review. I'll share my screen."),
    ]

    return DataPayload(
        scene_image_b64=scene_image_b64,
        audio_b64=wav_b64,
        video_frames=video_frames,
        gaze_points=gaze_points,
        pupil_data=pupil_data,
        conversation=conversation,
    )


# ---------------------------------------------------------------------------
# Sample data loader
# ---------------------------------------------------------------------------


def _load_sample_data(data_dir: Path) -> DataPayload:
    video_path = data_dir / "video.mp4"
    audio_path = data_dir / "audio.wav"
    csv_path = data_dir / "data.csv"

    for p in (video_path, audio_path, csv_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    # ── Video frames ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        raise ValueError(f"Could not read frames from {video_path}")

    n_samples = 5
    sample_indices = {int(i * (total_frames - 1) / (n_samples - 1)) for i in range(n_samples)}
    middle_idx = total_frames // 2
    all_indices = sorted(sample_indices | {middle_idx})

    video_frames: list[tuple[float, str]] = []
    scene_image_b64 = ""

    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf.tobytes()).decode()
        if idx == middle_idx:
            scene_image_b64 = b64
        if idx in sample_indices:
            video_frames.append((ts, b64))

    cap.release()

    # Fallback if middle frame read failed
    if not scene_image_b64 and video_frames:
        scene_image_b64 = video_frames[len(video_frames) // 2][1]

    # ── Audio ─────────────────────────────────────────────────────────────
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

    # ── CSV → gaze + pupil ────────────────────────────────────────────────
    gaze_points: list[tuple[float, float, float]] = []
    pupil_data: list[tuple[float, float, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ts = float(row["timestamp"])
            gaze_points.append((ts, float(row["gaze_x"]), float(row["gaze_y"])))
            pupil_data.append((ts, float(row["pupil_left"]), float(row["pupil_right"])))

    return DataPayload(
        scene_image_b64=scene_image_b64,
        audio_b64=audio_b64,
        video_frames=video_frames,
        gaze_points=gaze_points,
        pupil_data=pupil_data,
        conversation=[],  # populated via STT transcription in sample mode
    )


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


async def _step(
    client: httpx.AsyncClient,
    label: str,
    method: str,
    path: str,
    **kwargs,
) -> dict | list | None:
    try:
        resp = await client.request(method, path, **kwargs)
        if resp.status_code in (200, 201, 204):
            console.print(f"  [green]✓[/green] {label} [{resp.status_code}]")
            if resp.status_code == 204 or not resp.content:
                return {}
            return resp.json()
        console.print(f"  [red]✗[/red] {label} [{resp.status_code}] {resp.text[:200]}")
        return None
    except Exception as e:
        console.print(f"  [red]✗[/red] {label} — {e}")
        return None


# ---------------------------------------------------------------------------
# Fixed session inputs
# ---------------------------------------------------------------------------

_NEON_CONFIGS = {
    "blink_config": {
        "double_blink_action": None,
        "double_blink_delta": 0.3,
        "circle_annotation_radius": 75,
    },
    "fixation_config": {
        "algorithm": "I-DT",
        "min_duration_ms": 100.0,
        "max_dispersion_pixels": 30.0,
    },
    "saliency_config": {
        "max_peaks": 5,
        "min_peak_threshold_pct": 0.2,
        "marker_shape": "X",
        "marker_radius": 15,
        "gaussian_sigma": 60.0,
        "colormap": 2,
        "alpha": 0.6,
    },
    "recording_config": {"max_duration_s": -1.0},
}

_STEERING_CONTEXT = {
    "emotion": {"label": "neutral", "intensity": 0.0},
    "parsed_visual": {"focused_object": "screen", "detected_objects": ["keyboard", "desk"]},
    "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": "office"},
}


# ---------------------------------------------------------------------------
# Main session
# ---------------------------------------------------------------------------


async def _run_session(base_url: str, payload: DataPayload, mode: str, user_id: str) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]Synneon Redis Seeder[/bold cyan]\n"
            f"Server: [yellow]{base_url}[/yellow]   Mode: [magenta]{mode}[/magenta]   User: [green]{user_id}[/green]"
        )
    )

    uid = {"user_id": user_id}  # query param threaded through every store-backed call

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        # ── 1. Health ─────────────────────────────────────────────────────
        console.print("\n[bold]1. Health check[/bold]")
        health = await _step(client, "GET /health", "GET", "/health", params=uid)
        if health is None:
            console.print("[red]Server unreachable — aborting.[/red]")
            raise typer.Exit(1)
        console.print(f"   redis: {health.get('redis', '?')}")

        # ── 0 (pre-flight). Bootstrap user namespace ───────────────────────
        console.print(f"\n[bold]0. Bootstrap user '{user_id}'[/bold]")
        boot = await _step(client, "POST /context/users", "POST", "/context/users", params=uid)
        if boot:
            console.print(f"   session_id: {boot.get('session_id')}")

        # ── 2. Register conversation partner ──────────────────────────────
        console.print("\n[bold]2. Register speaker — Alice (colleague)[/bold]")
        alice_body = {
            "id": "",
            "user_id": user_id,
            "name": "Alice",
            "relationship": "colleague",
            "is_self": False,
        }
        alice = await _step(client, "POST /context/speakers", "POST", "/context/speakers", json=alice_body, params=uid)
        alice_id: str | None = alice.get("id") if alice else None
        if alice_id:
            console.print(f"   id: {alice_id}")

        # ── 3. Neon multimodal inference ──────────────────────────────────
        console.print("\n[bold]3. Neon multimodal inference (save_to_memory=true)[/bold]")
        neon_body = {
            "scene_image_base64": payload.scene_image_b64,
            "audio_base64": payload.audio_b64,
            "video_base64": payload.video_frames,
            "gaze_points": payload.gaze_points,
            "pupil_data": payload.pupil_data,
            "profile": "default",
            "save_to_memory": True,
            **_NEON_CONFIGS,
        }
        neon_result = await _step(
            client,
            "POST /inference/neon-context",
            "POST",
            "/inference/neon-context",
            json=neon_body,
            params=uid,
        )
        if neon_result:
            focused = (neon_result.get("parsed_visual") or {}).get("focused_object")
            console.print(f"   focused_object: {focused}")

        # ── 4. Image-to-context from scene frame ──────────────────────────
        console.print("\n[bold]4. Image-to-context — scene frame (save_to_memory=true)[/bold]")
        image_body = {
            "image_base64": payload.scene_image_b64,
            "media_type": "image/jpeg",
            "profile": "image2context",
            "save_to_memory": True,
        }
        img_result = await _step(
            client,
            "POST /inference/image2context/base64",
            "POST",
            "/inference/image2context/base64",
            json=image_body,
            params=uid,
        )
        if img_result:
            focused = (img_result.get("parsed_visual") or {}).get("focused_object")
            console.print(f"   focused_object: {focused}")

        # ── 5. Seed conversation ──────────────────────────────────────────
        if payload.conversation:
            console.print(f"\n[bold]5. Seed conversation — {len(payload.conversation)} lines (synthetic)[/bold]")
            entries_body = [
                {"user_id": user_id, "session_id": "", "speaker_label": label, "text": text, "timestamp": 0.0}
                for label, text in payload.conversation
            ]
            seeded = await _step(
                client,
                "POST /context/conversation",
                "POST",
                "/context/conversation",
                json=entries_body,
                params=uid,
            )
            if isinstance(seeded, list):
                console.print(f"   saved {len(seeded)} entries")
        else:
            console.print("\n[bold]5. Transcribe audio (save_to_memory=true)[/bold]")
            audio_body = {
                "audio_base64": payload.audio_b64,
                "media_type": "audio/wav",
                "profile": "default",
                "save_to_memory": True,
            }
            transcription = await _step(
                client,
                "POST /audio/transcribe/base64",
                "POST",
                "/audio/transcribe/base64",
                json=audio_body,
                params=uid,
            )
            if transcription:
                console.print(f"   text: {str(transcription.get('text', ''))[:80]}")

        # ── 6. Map diarization label → Alice ──────────────────────────────
        console.print("\n[bold]6. Map SPEAKER_2 → Alice[/bold]")
        conv_peek = await _step(client, "GET /context/conversation", "GET", "/context/conversation", params=uid)
        session_id: str | None = None
        if isinstance(conv_peek, list) and conv_peek:
            session_id = conv_peek[0].get("session_id")

        if session_id and alice_id:
            await _step(
                client,
                f"POST /context/sessions/{session_id}/speaker-map",
                "POST",
                f"/context/sessions/{session_id}/speaker-map",
                params={"label": "SPEAKER_2", "speaker_id": alice_id, **uid},
            )
        else:
            console.print("   [yellow]Skipped — session_id or alice_id unavailable[/yellow]")

        # ── 7. Abbreviation expansion (enriched from Redis) ───────────────
        console.print("\n[bold]7. Abbreviation expansion — enriched from Redis context[/bold]")
        expand_body = {"context": _STEERING_CONTEXT, "tokens": "h-w-a-y", "profile": "default"}
        expansion = await _step(
            client,
            "POST /inference/expand",
            "POST",
            "/inference/expand",
            json=expand_body,
            params=uid,
        )
        top_candidate = ""
        if expansion:
            candidates = expansion.get("candidates", [])
            if candidates:
                top_candidate = candidates[0].get("text", "")
                console.print(f"   top candidate: {top_candidate}")

        # ── 8. Speak the result ───────────────────────────────────────────
        console.print("\n[bold]8. Text-to-speech[/bold]")
        speak_text = top_candidate or "How are you today?"
        tts = await _step(
            client,
            "POST /audio/speak",
            "POST",
            "/audio/speak",
            params={"text": speak_text, "profile": "default"},
        )
        if tts is not None:
            console.print("   WAV response received")

        # ── 9–11. Read-back ───────────────────────────────────────────────
        console.print("\n[bold]Redis read-back[/bold]")
        conversation = await _step(client, "GET /context/conversation", "GET", "/context/conversation", params=uid)
        memories = await _step(client, "GET /context/memories", "GET", "/context/memories", params=uid)
        speakers = await _step(client, "GET /context/speakers", "GET", "/context/speakers", params=uid)

        # ── Summary ───────────────────────────────────────────────────────
        console.print()

        if isinstance(conversation, list) and conversation:
            t = Table(show_header=True, header_style="bold")
            t.add_column("Speaker Label")
            t.add_column("Speaker ID")
            t.add_column("Text")
            for e in conversation:
                t.add_row(
                    e.get("speaker_label", ""),
                    e.get("speaker_id") or "—",
                    str(e.get("text", ""))[:60],
                )
            console.print(Panel(t, title="[cyan]Conversation[/cyan]"))

        if isinstance(memories, list) and memories:
            t = Table(show_header=True, header_style="bold")
            t.add_column("Type")
            t.add_column("Summary")
            for m in memories:
                t.add_row(m.get("type", ""), str(m.get("summary", ""))[:80])
            console.print(Panel(t, title="[cyan]Memories[/cyan]"))

        if isinstance(speakers, list) and speakers:
            t = Table(show_header=True, header_style="bold")
            t.add_column("Name")
            t.add_column("Relationship")
            t.add_column("Is Self")
            for s in speakers:
                t.add_row(s.get("name", ""), s.get("relationship") or "—", str(s.get("is_self", False)))
            console.print(Panel(t, title="[cyan]Speakers[/cyan]"))

        console.print(
            "\n[bold green]Done.[/bold green] Open [link=http://localhost:5540]http://localhost:5540[/link] to inspect Redis keys."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@app.command(name="run")
def run(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Base URL of the SynNeon API server."),
    data: Path | None = typer.Option(
        None,
        "--data",
        "-d",
        help=(
            "Path to a capture directory containing video.mp4, audio.wav, and data.csv. "
            "Omit to use synthetic data generated in-memory."
        ),
    ),
    user_id: str = typer.Option(
        "default",
        "--user-id",
        "-id",
        help="User ID namespace for this session. Must match the server's memory config user_id.",
    ),
):
    """
    Simulate a full new-user session: register a speaker, run Neon and image inference,
    transcribe audio, map diarization labels, expand an abbreviation, and speak the result.
    All memory-enabled endpoints use save_to_memory=True.
    """
    if data is not None:
        if not data.is_dir():
            console.print(f"[red]--data is not a directory: {data}[/red]")
            raise typer.Exit(1)
        console.print(f"[dim]Loading sample data from {data}[/dim]\n")
        payload = _load_sample_data(data)
        mode = f"sample ({data.name})"
    else:
        console.print("[dim]No --data path — using synthetic data[/dim]\n")
        payload = _make_synthetic_data()
        mode = "synthetic"

    asyncio.run(_run_session(url, payload, mode, user_id))


if __name__ == "__main__":
    typer.run(run)
