"""CLI utility for running end-to-end inference pipelines using example capture data."""

from __future__ import annotations

import asyncio
import csv
import re
import time
import tomllib
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Run end-to-end example inference pipelines.")
console = Console()

ALL_STEPS = ["seed", "stt", "visualize", "neon", "expand", "tts"]


# ---------------------------------------------------------------------------
# File parsing helpers
# ---------------------------------------------------------------------------


def parse_conversation_file(path: Path, user_id: str, session_id: str) -> list:
    """Parse conversation.txt line-by-line into ConversationEntry objects.

    Each non-empty line must be in the format 'other: text' or 'user: text'.
    Lines with unrecognised prefixes default to role='other'.
    """
    from synneon.schemas.memory import ConversationEntry

    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(other|user):\s+(.+)$", line)
        if m:
            role, text = m.group(1), m.group(2)
        else:
            role, text = "other", line
        entries.append(
            ConversationEntry(user_id=user_id, session_id=session_id, role=role, text=text, source="loaded_and_parsed")
        )
    return entries


def parse_csv_file(
    path: Path, max_rows: int | None = 200
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Parse data.csv into gaze_points and pupil_data tuples.

    Args:
        max_rows: Downsample to at most this many rows. Pass None to return all rows.

    Returns:
        gaze_points: list of (timestamp, gaze_x, gaze_y)
        pupil_data:  list of (timestamp, pupil_left, pupil_right)
    Raises:
        ValueError if required columns are missing.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if rows:
        required = {"timestamp", "gaze_x", "gaze_y", "pupil_left", "pupil_right"}
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"data.csv missing required columns: {missing}")

    if max_rows is None:
        sampled = rows
    else:
        step = max(1, len(rows) // max_rows)
        sampled = rows[::step]

    gaze_points = [(float(r["timestamp"]), float(r["gaze_x"]), float(r["gaze_y"])) for r in sampled]
    pupil_data = [(float(r["timestamp"]), float(r["pupil_left"]), float(r["pupil_right"])) for r in sampled]
    return gaze_points, pupil_data


def extract_all_frames_as_ndarray(path: Path) -> list:
    """Extract all frames from a video file as numpy arrays (BGR).

    Returns list of np.ndarray frames in display order.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_reference_frame_png(path: Path, mode: str = "middle") -> bytes | None:
    """Return the frame `mode` selects (first|middle|last) as PNG bytes, or None if the video has no frames.

    Picks the same frame as `process_video_frames_for_saliency`, so the raw scene image
    and the saliency-peaks image show the same moment.
    """
    import cv2

    frames = extract_all_frames_as_ndarray(path)
    if not frames:
        return None
    if mode == "first":
        frame = frames[0]
    elif mode == "last":
        frame = frames[-1]
    else:
        frame = frames[len(frames) // 2]
    ok, buf = cv2.imencode(".png", frame)
    return buf.tobytes() if ok else None


def extract_video_frames(path: Path, max_frames: int = 4) -> list[tuple[float, bytes]]:
    """Extract up to max_frames evenly-spaced JPEG frames from a video file.

    Returns list of (timestamp_seconds, jpeg_bytes).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total <= max_frames:
        indices = list(range(total))
    else:
        indices = [int(i * total / max_frames) for i in range(max_frames)]

    frames: list[tuple[float, bytes]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = idx / fps
        _, buf = cv2.imencode(".jpg", frame)
        frames.append((timestamp, buf.tobytes()))

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _resolve_path(p: str | Path) -> Path:
    """Resolve a config path — absolute, relative to cwd, or relative to project root."""
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    # Try from the project root (4 levels up: apps → synneon → src → package root)
    pkg_root = Path(__file__).parent.parent.parent.parent
    alt = pkg_root / p
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Config file not found: {p}")


def _load_toml(path: str | Path) -> dict:
    resolved = _resolve_path(path)
    with open(resolved, "rb") as f:
        return tomllib.load(f)


def load_agent_settings(toml_path: str | Path, profile: str):
    from synneon.schemas.agents import AgentSettings

    data = _load_toml(toml_path)
    agent_data = data.get("agent", {}).get(profile)
    if not agent_data:
        raise ValueError(f"Agent profile '{profile}' not found in {toml_path}")
    return AgentSettings(**agent_data)


def load_memory_config(toml_path: str | Path, profile: str = "default"):
    from synneon.schemas.memory import MemoryConfig

    data = _load_toml(toml_path)
    memory_data = data.get("memory", {}).get(profile)
    if not memory_data:
        return MemoryConfig().with_env_overrides()
    return MemoryConfig(**memory_data).with_env_overrides()


def load_neons_configs(toml_path: str | Path):
    from synneon.schemas.neons import BlinkConfig, FixationConfig, RecordingConfig, SaliencyConfig

    data = _load_toml(toml_path)
    return (
        BlinkConfig(**data.get("blink", {})),
        FixationConfig(**data.get("fixation", {})),
        SaliencyConfig(**data.get("saliency", {})),
        RecordingConfig(**data.get("recording", {})),
    )


def load_stt_config(toml_path: str | Path, profile: str = "default"):
    from synneon.schemas.tts import STTConfig

    data = _load_toml(toml_path)
    stt_data = data.get("stt", {}).get(profile)
    if not stt_data:
        raise ValueError(f"STT profile '{profile}' not found in {toml_path}")
    return STTConfig(**stt_data)


def load_tts_config(toml_path: str | Path, profile: str = "default"):
    from synneon.schemas.tts import TTSConfig

    data = _load_toml(toml_path)
    tts_data = data.get("tts", {}).get(profile)
    if not tts_data:
        raise ValueError(f"TTS profile '{profile}' not found in {toml_path}")
    return TTSConfig(**tts_data)


# ---------------------------------------------------------------------------
# Rich output helpers
# ---------------------------------------------------------------------------


def _print_step(name: str) -> None:
    console.print(Panel(f"[bold cyan]Step: {name.upper()}[/bold cyan]", expand=False))


def _print_table(title: str, rows: dict, verbose: bool = False) -> None:
    t = Table(title=title, show_header=True, header_style="bold magenta")
    t.add_column("Field", style="cyan", no_wrap=True)
    t.add_column("Value")
    for k, v in rows.items():
        s = str(v)
        if not verbose and len(s) > 200:
            s = s[:200] + "…"
        t.add_row(str(k), s)
    console.print(t)


def _ok(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def _warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] {msg}")


def _err(msg: str) -> None:
    console.print(f"[bold red]✗[/bold red] {msg}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def _run_pipeline(
    example: Path,
    user: str,
    interpreter: str,
    memory: str,
    neons: str,
    stt: str,
    tts: str,
    save: bool,
    active_steps: list[str],
    profile: str,
    neon_profile: str,
    frame_mode: str,
    neon_scene: str,
    verbose: bool,
) -> None:
    session_id = str(uuid.uuid4())
    store = None

    if save:
        from redis.asyncio import Redis

        from synneon.services.memory.store import ContextStore

        mem_cfg = load_memory_config(memory)
        mem_cfg = mem_cfg.model_copy(update={"user_id": user})
        redis_client = Redis.from_url(mem_cfg.redis_url, decode_responses=True)
        store = ContextStore(redis=redis_client, config=mem_cfg)
        await store.bootstrap()
        await store.start_session(session_id)
        console.print(f"[dim]Redis: {mem_cfg.redis_url} | user: {user} | session: {session_id}[/dim]")
    else:
        console.print(f"[dim]Dry-run (--no-save) | user: {user} | session: {session_id}[/dim]")

    # In-memory conversation list used by expand when store is not available
    conv_entries: list = []
    expansion = None

    # ---- SEED ---------------------------------------------------------------
    if "seed" in active_steps:
        _print_step("seed")
        conv_path = example / "conversation.txt"
        if not conv_path.exists():
            _warn("conversation.txt not found — skipping seed")
        else:
            entries = parse_conversation_file(conv_path, user_id=user, session_id=session_id)
            conv_entries = entries
            if save and store:
                now = time.time()
                for i, entry in enumerate(entries):
                    entry = entry.model_copy(update={"timestamp": now + i * 0.1})
                    await store.save_utterance(entry)
            _ok(f"Seeded {len(entries)} conversation entries")
            _print_table(
                "Conversation",
                {f"[{e.role}] #{i + 1}": e.text for i, e in enumerate(entries)},
                verbose,
            )

    # ---- STT ----------------------------------------------------------------
    if "stt" in active_steps:
        _print_step("stt")
        audio_path = example / "audio.wav"
        if not audio_path.exists():
            _warn("audio.wav not found — skipping stt")
        else:
            try:
                from synneon.schemas.memory import ConversationEntry
                from synneon.services.audio.stt import transcribe

                stt_cfg = load_stt_config(stt)
                result = await transcribe(audio_path.read_bytes(), "audio/wav", stt_cfg)
                transcription = result.text
                utterance_texts = [seg.text for seg in result.segments]

                # Always build entries and append to conv_entries so expand always sees them
                now = time.time()
                new_entries = [
                    ConversationEntry(
                        user_id=user,
                        session_id=session_id,
                        role="other",
                        text=text,
                        timestamp=now + i * 0.001,
                    )
                    for i, text in enumerate(utterance_texts)
                    if text
                ]
                conv_entries.extend(new_entries)

                if save and store:
                    for entry in new_entries:
                        await store.save_utterance(entry)

                _ok("Audio transcribed")
                _print_table("Transcription", {"text": transcription}, verbose)

            except Exception as e:
                _warn(f"STT step failed: {e}")
                if verbose:
                    console.print_exception()

    # ---- VISUALIZE ----------------------------------------------------------
    if "visualize" in active_steps:
        _print_step("visualize")
        video_path = example / "video.mp4"
        csv_path = example / "data.csv"
        if not video_path.exists() or not csv_path.exists():
            _warn("video.mp4 or data.csv not found — skipping visualize")
        else:
            try:
                import cv2

                from synneon.services.perception.vision import process_video_frames_for_saliency

                _, fixation_cfg, saliency_cfg, _ = load_neons_configs(neons)

                # Use full undownsampled gaze for saliency accuracy
                full_gaze, _ = parse_csv_file(csv_path, max_rows=None)
                raw_frames = extract_all_frames_as_ndarray(video_path)
                _ok(f"Loaded {len(full_gaze)} gaze samples and {len(raw_frames)} frames for saliency")

                result = process_video_frames_for_saliency(
                    raw_frames,
                    full_gaze,
                    mode=frame_mode,
                    gaussian_sigma=saliency_cfg.gaussian_sigma,
                    colormap=getattr(cv2, saliency_cfg.colormap)
                    if isinstance(saliency_cfg.colormap, str)
                    else saliency_cfg.colormap,
                    alpha=saliency_cfg.alpha,
                    max_peaks=saliency_cfg.max_peaks,
                    min_peak_threshold_pct=saliency_cfg.min_peak_threshold_pct,
                    marker_shape=saliency_cfg.marker_shape,
                    marker_radius=saliency_cfg.marker_radius,
                    min_duration_ms=fixation_cfg.min_duration_ms,
                    max_dispersion_pixels=fixation_cfg.max_dispersion_pixels,
                )

                if result is not None:
                    saliency_gray, saliency_overlay, peaks_overlay, trajectory_overlay, fixations_overlay = result

                    outputs: list[tuple[str, object]] = [
                        ("saliency_map.png", saliency_gray),
                        ("saliency_overlay.png", saliency_overlay),
                    ]
                    if peaks_overlay is not None:
                        outputs.append(("saliency_peaks.png", peaks_overlay))
                    if trajectory_overlay is not None:
                        outputs.append(("gaze_trajectory.png", trajectory_overlay))
                    if fixations_overlay is not None:
                        outputs.append(("gaze_fixations.png", fixations_overlay))

                    saved_files = []
                    for fname, img in outputs:
                        out_path = example / fname
                        cv2.imwrite(str(out_path), img)
                        saved_files.append(fname)

                    _ok(f"Visualization outputs: {', '.join(saved_files)}")
                    _print_table(
                        "Visualize",
                        {
                            "frame_mode": frame_mode,
                            "gaze_samples": len(full_gaze),
                            "video_frames": len(raw_frames),
                            "outputs": ", ".join(saved_files),
                        },
                        verbose,
                    )
                else:
                    _warn("process_video_frames_for_saliency returned None (empty frames list?)")

            except Exception as e:
                _warn(f"Visualize step failed: {e}")
                if verbose:
                    console.print_exception()

    # ---- NEON ---------------------------------------------------------------
    neon_result = None  # stays None when the neon step is skipped or fails
    if "neon" in active_steps:
        _print_step("neon")
        video_path = example / "video.mp4"
        audio_path = example / "audio.wav"
        csv_path = example / "data.csv"
        try:
            from synneon.agents.base import VLMNeon2ContextAgent
            from synneon.schemas.memory import InferenceRecord, InferenceType, MemoryEntry, MemoryType
            from synneon.schemas.neons import NeonCapturePayload
            from synneon.schemas.results import Image2ContextResult
            from synneon.services.memory.store import derive_summary

            neon_settings = load_agent_settings(interpreter, neon_profile)
            blink_cfg, fixation_cfg, saliency_cfg, recording_cfg = load_neons_configs(neons)

            gaze_points: list = []
            pupil_data: list = []
            if csv_path.exists():
                gaze_points, pupil_data = parse_csv_file(csv_path)
                _ok(f"Loaded {len(gaze_points)} gaze samples from data.csv")

            video_frames: list = []
            if video_path.exists():
                video_frames = extract_video_frames(video_path)
                _ok(f"Extracted {len(video_frames)} video frames")

            audio_bytes = audio_path.read_bytes() if audio_path.exists() else None

            scene_image: bytes | None = None
            if neon_scene == "peaks":
                peaks_path = example / "saliency_peaks.png"
                if peaks_path.exists():
                    scene_image = peaks_path.read_bytes()
                    _ok("Using saliency_peaks.png as scene image for neon VLM")
                else:
                    _warn("--neon-scene peaks: saliency_peaks.png not found (run visualize step first); using raw")

            if scene_image is None and video_path.exists():
                scene_image = extract_reference_frame_png(video_path, frame_mode)
                if scene_image is not None:
                    _ok(f"Using the raw {frame_mode} video frame as scene image for neon VLM")

            payload = NeonCapturePayload(
                video=video_frames,
                gaze_points=gaze_points,
                pupil_data=pupil_data,
                audio=audio_bytes,
                scene_image=scene_image,
                blink_config=blink_cfg,
                fixation_config=fixation_cfg,
                saliency_config=saliency_cfg,
                recording_config=recording_cfg,
            )

            audio_transcript = None
            if audio_bytes and not neon_settings.accepts_audio:
                from synneon.services.audio.stt import transcribe

                audio_transcript = (await transcribe(audio_bytes, "audio/wav", load_stt_config(stt))).text or None

            neon_agent = VLMNeon2ContextAgent(settings=neon_settings, output_type=Image2ContextResult)
            run_result = await neon_agent.run(payload, audio_transcript=audio_transcript)
            neon_result = run_result.output

            if save and store:
                mem_entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    user_id=user,
                    session_id=session_id,
                    type=MemoryType.VLM_NEON_CONTEXT,
                    summary=derive_summary(neon_result),
                    data=neon_result.model_dump(),
                )
                await store.save_memory(mem_entry)

                record = InferenceRecord(
                    id=str(uuid.uuid4()),
                    user_id=user,
                    session_id=session_id,
                    inference_type=InferenceType.IMAGE2TEXT,
                    endpoint="neon-context",
                    profile=neon_profile,
                    agent_config=neon_agent.settings.model_dump(),
                    context_snapshot={
                        "has_scene_image": scene_image is not None,
                        "neon_scene": neon_scene,
                        "has_audio": audio_bytes is not None,
                        "video_frames_count": len(video_frames),
                        "gaze_points_count": len(gaze_points),
                        "has_pupil_data": len(pupil_data) > 0,
                    },
                    result_snapshot=neon_result.model_dump(),
                )
                await store.save_inference_record(record)

            _ok("Neon context inference complete")
            _print_table("Neon Context Result", neon_result.model_dump(), verbose)

        except Exception as e:
            _warn(f"Neon step failed: {e}")
            if verbose:
                console.print_exception()

    # ---- EXPAND -------------------------------------------------------------
    if "expand" in active_steps:
        _print_step("expand")
        abbrev_path = example / "abbreviation_for_inference.txt"
        if not abbrev_path.exists():
            _warn("abbreviation_for_inference.txt not found — skipping expand")
        else:
            tokens = abbrev_path.read_text().strip()
            if not tokens:
                _warn("abbreviation_for_inference.txt is empty — skipping expand")
            else:
                try:
                    from synneon.agents.base import CognitiveSteeringAgent
                    from synneon.agents.expansion import parse_expansion
                    from synneon.schemas.context import SteeringContext
                    from synneon.schemas.memory import InferenceRecord, InferenceType
                    from synneon.services.memory.store import format_conversation, format_memories

                    agent_settings = load_agent_settings(interpreter, profile)
                    agent = CognitiveSteeringAgent(settings=agent_settings, output_type=str)
                    context = SteeringContext()
                    if neon_result is not None:
                        # Ground the expansion in what the neon step just inferred, with or without Redis.
                        context = SteeringContext(
                            parsed_visual=neon_result.parsed_visual, pragmatic=neon_result.pragmatic
                        )
                        _ok("Using the neon step's result as steering context")

                    if save and store:
                        history = await store.get_recent_conversation(session_id)
                        memories = await store.get_recent_memories()
                        conversation_str = format_conversation(history)
                        memories_str = format_memories(memories)
                    else:
                        conversation_str = format_conversation(conv_entries)
                        memories_str = ""

                    prompt_kwargs = dict(
                        tokens=tokens,
                        conversation_history=conversation_str,
                        recent_memories=memories_str,
                    )
                    system_prompt, user_prompt = agent.builder.build_prompts(context, **prompt_kwargs)

                    run_result = await agent.run(context, **prompt_kwargs)
                    raw_text = str(run_result.output)
                    expansion = parse_expansion(raw_text)

                    if save and store:
                        await store.save_last_expand_context(
                            session_id,
                            {
                                "tokens": tokens,
                                "steering_context": context.model_dump(),
                                "conversation_history": conversation_str,
                                "recent_memories": memories_str,
                                "model": agent_settings.model,
                                "profile": profile,
                                "candidates": expansion.model_dump().get("candidates", []),
                                "system_prompt": system_prompt,
                                "user_prompt": user_prompt,
                            },
                        )

                        record = InferenceRecord(
                            id=str(uuid.uuid4()),
                            user_id=user,
                            session_id=session_id,
                            inference_type=InferenceType.TEXT2TEXT,
                            endpoint="expand",
                            profile=profile,
                            agent_config=agent.settings.model_dump(),
                            context_snapshot={
                                "tokens": tokens,
                                "steering_context": context.model_dump(),
                                "conversation_history": conversation_str,
                                "recent_memories": memories_str,
                            },
                            context={"system_prompt": system_prompt, "user_prompt": user_prompt},
                            result_snapshot=expansion.model_dump(),
                        )
                        await store.save_inference_record(record)

                    _ok(f"Expanded '{tokens}'")
                    _print_table(
                        "Expansion Candidates",
                        {f"#{c.rank}": c.text for c in expansion.candidates},
                        verbose,
                    )

                except Exception as e:
                    _err(f"Expand step failed: {e}")
                    if verbose:
                        console.print_exception()

    # ---- TTS ----------------------------------------------------------------
    if "tts" in active_steps:
        _print_step("tts")
        if expansion is None or not expansion.candidates:
            _warn("No expansion result available — skipping TTS (run expand step first)")
        else:
            try:
                from synneon.services.audio.tts import synthesize

                tts_cfg = load_tts_config(tts)
                top_text = expansion.candidates[0].text
                audio_out = await synthesize(top_text, tts_cfg)
                output_path = example / "output_tts.wav"
                output_path.write_bytes(audio_out)
                _ok(f"TTS audio written to {output_path}")
                _print_table("TTS", {"text": top_text, "output": str(output_path)}, verbose)

            except Exception as e:
                _warn(f"TTS step failed: {e}")
                if verbose:
                    console.print_exception()

    # Cleanup
    if save and store:
        await store.end_session()
        await store._redis.aclose()

    console.print(Panel("[bold green]Pipeline complete[/bold green]", expand=False))


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


@app.command(name="run")
def run(
    example: Annotated[
        Path,
        typer.Option(
            help="Path to example folder containing audio.wav, data.csv, video.mp4, "
            "conversation.txt, abbreviation_for_inference.txt"
        ),
    ] = Path("examples/capture_sample_v1"),
    user: Annotated[str, typer.Option(help="User ID for the session")] = "default",
    interpreter: Annotated[
        str, typer.Option(help="Path to interpreter agent TOML config")
    ] = "configs/interpreter/agent.toml",
    latents: Annotated[str | None, typer.Option(help="Path to latents config TOML (reserved for future use)")] = None,
    memory: Annotated[str, typer.Option(help="Path to memory/Redis config TOML")] = "configs/memory/default.toml",
    neons: Annotated[str, typer.Option(help="Path to Neon device config TOML")] = "configs/neons/default.toml",
    stt: Annotated[str, typer.Option(help="Path to STT config TOML")] = "configs/stt/default.toml",
    tts: Annotated[str, typer.Option(help="Path to TTS config TOML")] = "configs/tts/default.toml",
    save: Annotated[bool, typer.Option("--save/--no-save", help="Persist results to Redis")] = False,
    steps: Annotated[
        str,
        typer.Option(help="Comma-separated steps to run: seed,stt,visualize,neon,expand,tts  (or 'all')"),
    ] = "all",
    profile: Annotated[str, typer.Option(help="Interpreter agent profile for the expand step")] = "default",
    neon_profile: Annotated[str, typer.Option(help="Agent profile for the neon-context step")] = "image2context",
    frame_mode: Annotated[
        str, typer.Option(help="Reference frame for saliency computation: first|middle|last")
    ] = "middle",
    neon_scene: Annotated[
        str, typer.Option(help="Scene image passed to neon VLM: raw (video frame) or peaks (saliency_peaks.png)")
    ] = "raw",
    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose", help="Show full config and context dumps")] = False,
) -> None:
    """Run an end-to-end inference pipeline using example capture data."""
    active_steps = ALL_STEPS if steps.strip().lower() == "all" else [s.strip() for s in steps.split(",")]
    invalid = set(active_steps) - set(ALL_STEPS)
    if invalid:
        _err(f"Unknown steps: {invalid}. Valid steps: {ALL_STEPS}")
        raise typer.Exit(1)

    if frame_mode not in ("first", "middle", "last"):
        _err(f"Invalid --frame-mode '{frame_mode}'. Must be: first, middle, last")
        raise typer.Exit(1)

    if neon_scene not in ("raw", "peaks"):
        _err(f"Invalid --neon-scene '{neon_scene}'. Must be: raw, peaks")
        raise typer.Exit(1)

    asyncio.run(
        _run_pipeline(
            example=example,
            user=user,
            interpreter=interpreter,
            memory=memory,
            neons=neons,
            stt=stt,
            tts=tts,
            save=save,
            active_steps=active_steps,
            profile=profile,
            neon_profile=neon_profile,
            frame_mode=frame_mode,
            neon_scene=neon_scene,
            verbose=verbose,
        )
    )
