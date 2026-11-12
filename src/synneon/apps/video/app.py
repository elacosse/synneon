from __future__ import annotations

import asyncio
import base64
import csv
import logging
import queue
import threading
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import httpx
import numpy as np
import typer
from pupil_labs.realtime_api import Device, receive_gaze_data
from pupil_labs.realtime_api.streaming import BlinkEventData
from pynput.keyboard import Key, Listener
from rich.console import Console
from rich.panel import Panel

from synneon.apps.shared.config import (
    load_blink_config,
    load_fixation_config,
    load_neon_config,
    load_recording_config,
    load_saliency_config,
)
from synneon.neon.session import (
    _get_rtsp_url,
    connect,
    stream_audio,
    stream_eye_events,
    stream_video,
)
from synneon.schemas.neons import NeonCapturePayload
from synneon.services.perception.vision import process_video_frames_for_saliency

logger = logging.getLogger("synneon")
console = Console()

video_app = typer.Typer(help="Video acquisition app")


@video_app.command(name="capture-all")
def capture_all(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    output_dir: Path = typer.Option(Path("captures"), "--output", "-o", help="Directory to save captures"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    max_peaks: int = typer.Option(5, "--max-peaks", help="Number of local gaze maxima to identify on the saliency map"),
    analyze: bool = typer.Option(
        True, "--analyze/--no-analyze", help="Send data to API for analysis. If --no-analyze, saves to disk instead."
    ),
    api_url: str = typer.Option("http://localhost:8000", help="Base URL of the SynNeon API server"),
    profile: str = typer.Option("image2context", help="The VLM agent profile to use for analysis"),
):
    """
    Acquire audio, video, gaze, and pupillometry on double blink or spacebar.
    Stops on second double blink or spacebar.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("synneon").setLevel(logging.DEBUG)

    neon_config = load_neon_config(config_path)
    blink_config = load_blink_config(config_path)
    fixation_config = load_fixation_config(config_path)
    saliency_config = load_saliency_config(config_path)
    recording_config = load_recording_config(config_path)

    async def run():
        device = await connect(neon_config)
        stop_event = asyncio.Event()
        typer.secho("\n" + "=" * 60)
        typer.secho(" ACQUISITION MODE ", fg=typer.colors.BLACK, bg=typer.colors.WHITE, bold=True)
        typer.echo("=" * 60)
        typer.secho("Awaiting Double Blink or SPACEBAR to trigger acquisition.", fg=typer.colors.CYAN)
        typer.secho("Press 'q' or Ctrl+C to stop the program.", fg=typer.colors.YELLOW)
        typer.echo("-" * 60 + "\n")
        try:
            await run_acquisition(
                device,
                stop_event,
                blink_config.double_blink_delta,
                output_dir,
                max_peaks,
                blink_config,
                fixation_config,
                saliency_config,
                recording_config,
                not analyze,  # save_to_disk is inverse of analyze
                analyze,
                api_url,
                profile,
            )
        finally:
            await device.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")


async def run_acquisition(
    device: Device,
    stop_event: asyncio.Event,
    double_blink_delta: float,
    output_dir: Path,
    max_peaks: int,
    blink_config,
    fixation_config,
    saliency_config,
    recording_config,
    save_to_disk: bool,
    analyze: bool,
    api_url: str,
    profile: str,
):
    gaze_url = await _get_rtsp_url(device, "gaze")

    class State:
        recording: bool = False
        latest_frame: np.ndarray | None = None
        latest_gaze: tuple[float, float, bool] | None = None
        latest_pupil: tuple[float, float] | None = None
        latest_audio_rms: float = 0.0
        blinks: deque[float] = deque(maxlen=2)

        # Disk saving
        session_dir: Path | None = None
        video_writer: cv2.VideoWriter | None = None
        audio_frames: list[np.ndarray] = []
        video_frames: list[tuple[float, np.ndarray]] = []
        gaze_points: list[tuple[float, float, float]] = []
        pupil_data_points: list[tuple[float, float, float]] = []
        audio_sample_rate: int = 48000
        csv_file = None
        csv_writer = None

        save_queue: queue.Queue = queue.Queue()
        recording_timeout_task: asyncio.Task | None = None

    state = State()
    loop = asyncio.get_running_loop()

    def save_worker():
        while True:
            item = state.save_queue.get()
            if item is None:
                break
            try:
                action = item[0]
                if action == "video":
                    writer, frame = item[1], item[2]
                    writer.write(frame)
                elif action == "csv":
                    writer, row = item[1], item[2]
                    writer.writerow(row)
                elif action == "close_video":
                    writer = item[1]
                    writer.release()
                elif action == "close_csv":
                    file_obj = item[1]
                    file_obj.close()
                elif action == "audio_end":
                    audio_frames, sample_rate, audio_path = item[1], item[2], item[3]
                    if not audio_frames:
                        continue
                    audio_data = np.concatenate(audio_frames, axis=1)
                    audio_data_int16 = (audio_data * 32767).astype(np.int16)
                    channels = audio_data.shape[0]

                    with wave.open(str(audio_path), "wb") as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        if channels > 1:
                            audio_data_int16 = audio_data_int16.T.flatten()
                        else:
                            audio_data_int16 = audio_data_int16.flatten()
                        wf.writeframes(audio_data_int16.tobytes())
                elif action == "saliency":
                    (
                        video_frames_list,
                        gaze_points_list,
                        overlay_path,
                        gray_path,
                        peaks_path,
                        trajectory_path,
                        fixations_path,
                    ) = (
                        item[1],
                        item[2],
                        item[3],
                        item[4],
                        item[5],
                        item[6],
                        item[7],
                    )
                    if video_frames_list:
                        actual_max_peaks = max_peaks if max_peaks != 5 else saliency_config.max_peaks
                        raw_frames = [f for _, f in video_frames_list]
                        result = process_video_frames_for_saliency(
                            raw_frames,
                            gaze_points=gaze_points_list,
                            mode="middle",
                            gaussian_sigma=saliency_config.gaussian_sigma,
                            colormap=saliency_config.colormap,
                            alpha=saliency_config.alpha,
                            max_peaks=actual_max_peaks,
                            min_peak_threshold_pct=saliency_config.min_peak_threshold_pct,
                            marker_shape=saliency_config.marker_shape,
                            marker_radius=saliency_config.marker_radius,
                            min_duration_ms=fixation_config.min_duration_ms,
                            max_dispersion_pixels=fixation_config.max_dispersion_pixels,
                        )
                        if result is not None:
                            saliency_gray, saliency_overlay, peaks_overlay, trajectory_overlay, fixations_overlay = (
                                result
                            )
                            cv2.imwrite(str(gray_path), saliency_gray)
                            cv2.imwrite(str(overlay_path), saliency_overlay)
                            if peaks_overlay is not None:
                                cv2.imwrite(str(peaks_path), peaks_overlay)
                            if trajectory_overlay is not None:
                                cv2.imwrite(str(trajectory_path), trajectory_overlay)
                            if fixations_overlay is not None:
                                cv2.imwrite(str(fixations_path), fixations_overlay)
            except Exception as e:
                logger.error(f"Save worker error: {e}")
            finally:
                state.save_queue.task_done()

    save_thread = threading.Thread(target=save_worker, daemon=True)
    save_thread.start()

    def toggle_recording():
        state.recording = not state.recording
        if state.recording:
            logger.info("Acquisition STARTED")
            state.audio_frames = []
            state.video_frames = []
            state.gaze_points = []
            state.pupil_data_points = []

            if save_to_disk:
                state.session_dir = output_dir / f"acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                state.session_dir.mkdir(parents=True, exist_ok=True)

                # CSV Setup
                csv_path = state.session_dir / "data.csv"
                state.csv_file = open(csv_path, "w", newline="")
                state.csv_writer = csv.writer(state.csv_file)
                state.csv_writer.writerow(
                    ["timestamp", "gaze_x", "gaze_y", "gaze_worn", "pupil_left", "pupil_right", "audio_rms"]
                )

            if recording_config.max_duration_s > 0:

                async def timeout_worker():
                    try:
                        await asyncio.sleep(recording_config.max_duration_s)
                        if state.recording:
                            logger.info(f"Auto-stopping acquisition after {recording_config.max_duration_s}s")
                            toggle_recording()
                    except asyncio.CancelledError:
                        pass

                state.recording_timeout_task = loop.create_task(timeout_worker())
        else:
            logger.info("Acquisition STOPPED")
            if state.recording_timeout_task is not None:
                state.recording_timeout_task.cancel()
                state.recording_timeout_task = None

            if state.video_writer:
                state.save_queue.put(("close_video", state.video_writer))
                state.video_writer = None

            if state.csv_file:
                state.save_queue.put(("close_csv", state.csv_file))
                state.csv_file = None

            if save_to_disk and state.audio_frames and state.session_dir:
                audio_path = state.session_dir / "audio.wav"
                state.save_queue.put(("audio_end", state.audio_frames, state.audio_sample_rate, audio_path))

            # Build the NeonCapturePayload asynchronously
            if state.video_frames:

                def build_payload(
                    v_frames: list[tuple[float, np.ndarray]],
                    g_points: list[tuple[float, float, float]],
                    p_points: list[tuple[float, float, float]],
                    a_frames: list[np.ndarray],
                    sample_rate: int,
                    cfg_blink,
                    cfg_fixation,
                    cfg_saliency,
                    cfg_recording,
                ) -> NeonCapturePayload:
                    import io
                    import wave

                    # 1. Extract middle frame as scene_image
                    scene_bytes = None
                    if v_frames:
                        middle_frame = v_frames[len(v_frames) // 2][1]
                        success, encoded = cv2.imencode(".png", middle_frame)
                        if success:
                            scene_bytes = encoded.tobytes()

                    # 2. Extract video pairs (ts, png_bytes)
                    compressed_video = []
                    for ts, frm in v_frames:
                        success, encoded = cv2.imencode(".png", frm)
                        if success:
                            compressed_video.append((ts, encoded.tobytes()))

                    # 3. Create audio bytes
                    audio_bytes = None
                    if a_frames:
                        try:
                            audio_data = np.concatenate(a_frames, axis=1)
                            audio_data_int16 = (audio_data * 32767).astype(np.int16)
                            channels = audio_data.shape[0]

                            wav_io = io.BytesIO()
                            with wave.open(wav_io, "wb") as wf:
                                wf.setnchannels(channels)
                                wf.setsampwidth(2)
                                wf.setframerate(sample_rate)
                                if channels > 1:
                                    audio_data_int16 = audio_data_int16.T.flatten()
                                else:
                                    audio_data_int16 = audio_data_int16.flatten()
                                wf.writeframes(audio_data_int16.tobytes())
                            audio_bytes = wav_io.getvalue()
                        except Exception as e:
                            logger.error(f"Failed to encode audio for payload: {e}")

                    return NeonCapturePayload(
                        scene_image=scene_bytes,
                        video=compressed_video,
                        gaze_points=g_points,
                        pupil_data=p_points,
                        audio=audio_bytes,
                        blink_config=cfg_blink,
                        fixation_config=cfg_fixation,
                        saliency_config=cfg_saliency,
                        recording_config=cfg_recording,
                    )

                v_frames_copy = list(state.video_frames)
                g_points_copy = list(state.gaze_points)
                p_points_copy = list(state.pupil_data_points)
                a_frames_copy = list(state.audio_frames)

                async def generate_payload():
                    try:
                        payload = await asyncio.to_thread(
                            build_payload,
                            v_frames_copy,
                            g_points_copy,
                            p_points_copy,
                            a_frames_copy,
                            state.audio_sample_rate,
                            blink_config,
                            fixation_config,
                            saliency_config,
                            recording_config,
                        )
                        logger.info(f"Successfully generated NeonCapturePayload with {len(payload.video)} frames.")
                        if analyze:
                            await analyze_payload(payload, api_url, profile)
                    except Exception as e:
                        logger.error(f"Failed to build or analyze payload: {e}")

                loop.create_task(generate_payload())

            if save_to_disk and state.video_frames and state.session_dir:
                saliency_path = state.session_dir / "saliency_overlay.png"
                saliency_gray_path = state.session_dir / "saliency_map.png"
                peaks_path = state.session_dir / "saliency_peaks.png"
                trajectory_path = state.session_dir / "gaze_trajectory.png"
                fixations_path = state.session_dir / "gaze_fixations.png"
                state.save_queue.put(
                    (
                        "saliency",
                        state.video_frames,
                        list(state.gaze_points),
                        saliency_path,
                        saliency_gray_path,
                        peaks_path,
                        trajectory_path,
                        fixations_path,
                    )
                )

    def on_press(key):
        try:
            if key == Key.space:
                loop.call_soon_threadsafe(toggle_recording)
            elif hasattr(key, "char") and key.char == "q":
                loop.call_soon_threadsafe(stop_event.set)
        except Exception:
            pass

    listener = Listener(on_press=on_press)
    listener.start()

    async def consume_video():
        while not stop_event.is_set():
            try:
                async for frame in stream_video(device):
                    if stop_event.is_set():
                        break
                    bgr_frame = frame.av_frame.to_ndarray(format="bgr24")
                    state.latest_frame = bgr_frame

                    if state.recording:
                        ts = getattr(frame, "timestamp_unix_seconds", 0.0)
                        state.video_frames.append((ts, bgr_frame))

                        gx, gy, worn = state.latest_gaze if state.latest_gaze else (0.0, 0.0, False)
                        if worn:
                            state.gaze_points.append((ts, gx, gy))

                        if save_to_disk:
                            if state.video_writer is None and state.session_dir is not None:
                                h, w = bgr_frame.shape[:2]
                                video_path = state.session_dir / "video.mp4"
                                state.video_writer = cv2.VideoWriter(
                                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
                                )
                            if state.video_writer is not None:
                                state.save_queue.put(("video", state.video_writer, bgr_frame))

                            if state.csv_writer is not None:
                                pl, pr = state.latest_pupil if state.latest_pupil else (0.0, 0.0)
                                rms = state.latest_audio_rms
                                # The save worker is the only writer; writing here too duplicated every row.
                                state.save_queue.put(("csv", state.csv_writer, [ts, gx, gy, worn, pl, pr, rms]))
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Video stream error: {e}. Reconnecting...")
                    await asyncio.sleep(1)

    async def consume_gaze_pupil():
        while not stop_event.is_set():
            try:
                async for gaze in receive_gaze_data(gaze_url):
                    if stop_event.is_set():
                        break
                    state.latest_gaze = (gaze.x, gaze.y, getattr(gaze, "worn", True))
                    if hasattr(gaze, "pupil_diameter_left") and hasattr(gaze, "pupil_diameter_right"):
                        state.latest_pupil = (gaze.pupil_diameter_left, gaze.pupil_diameter_right)

                        if state.recording:
                            state.pupil_data_points.append(
                                (gaze.timestamp_unix_seconds, gaze.pupil_diameter_left, gaze.pupil_diameter_right)
                            )
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Gaze stream error: {e}. Reconnecting...")
                    await asyncio.sleep(1)

    async def consume_audio():
        while not stop_event.is_set():
            try:
                async for frame in stream_audio(device):
                    if stop_event.is_set():
                        break
                    state.audio_sample_rate = frame.av_frame.sample_rate
                    raw_audio = frame.av_frame.to_ndarray()

                    # Normalize to float32 in range [-1.0, 1.0]
                    if raw_audio.dtype == np.int16:
                        audio_float = raw_audio.astype(np.float32) / 32768.0
                    elif raw_audio.dtype == np.int32:
                        audio_float = raw_audio.astype(np.float32) / 2147483648.0
                    else:
                        audio_float = raw_audio.astype(np.float32)

                    if state.recording:
                        state.audio_frames.append(audio_float)

                    if audio_float.size > 0:
                        rms = np.sqrt(np.mean(np.square(audio_float)))
                        state.latest_audio_rms = float(rms)
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Audio stream error: {e}. Reconnecting...")
                    await asyncio.sleep(1)

    async def consume_eye_events():
        while not stop_event.is_set():
            try:
                async for event in stream_eye_events(device):
                    if stop_event.is_set():
                        break
                    if isinstance(event, BlinkEventData):
                        ts = event.start_time_ns / 1e9
                        state.blinks.append(ts)
                        if len(state.blinks) == 2:
                            delta = state.blinks[1] - state.blinks[0]
                            if delta <= double_blink_delta:
                                loop.call_soon_threadsafe(toggle_recording)
                                state.blinks.clear()
            except Exception as e:
                if not stop_event.is_set():
                    logger.error(f"Eye event stream error: {e}. Reconnecting...")
                    await asyncio.sleep(1)

    tasks = [
        asyncio.create_task(consume_video()),
        asyncio.create_task(consume_gaze_pupil()),
        asyncio.create_task(consume_audio()),
        asyncio.create_task(consume_eye_events()),
    ]

    window_open = False
    window_name = "Neon Acquisition Data"

    try:
        while not stop_event.is_set():
            if state.recording and state.latest_frame is not None:
                if not window_open:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    window_open = True

                display_img = state.latest_frame.copy()
                h, w = display_img.shape[:2]

                # Draw recording indicator
                cv2.rectangle(display_img, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
                cv2.putText(display_img, "ACQUIRING DATA", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Pupil data
                if state.latest_pupil:
                    pl, pr = state.latest_pupil
                    pupil_text = f"Pupil L: {pl:.2f}mm  R: {pr:.2f}mm"
                    cv2.putText(display_img, pupil_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Audio data
                audio_text = f"Audio RMS: {state.latest_audio_rms:.2f}"
                cv2.putText(display_img, audio_text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

                # Gaze data
                if state.latest_gaze:
                    gx, gy, worn = state.latest_gaze
                    gaze_text = f"Gaze X: {gx:.1f}  Y: {gy:.1f}  Worn: {worn}"
                    cv2.putText(display_img, gaze_text, (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

                    if worn:
                        px, py = int(gx), int(gy)
                        px = max(0, min(px, w - 1))
                        py = max(0, min(py, h - 1))
                        cv2.circle(display_img, (px, py), 15, (0, 0, 255), 3)
                        cv2.circle(display_img, (px, py), 5, (0, 0, 255), -1)

                cv2.imshow(window_name, display_img)
                cv2.waitKey(1)

            elif not state.recording:
                if window_open:
                    cv2.destroyWindow(window_name)
                    cv2.waitKey(1)
                    window_open = False

            await asyncio.sleep(0.01)

    finally:
        if state.recording:
            toggle_recording()  # Ensure files are saved properly

        stop_event.set()
        listener.stop()
        for t in tasks:
            t.cancel()
        if window_open:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        await asyncio.gather(*tasks, return_exceptions=True)

        state.save_queue.put(None)
        save_thread.join(timeout=5.0)


async def analyze_payload(payload: NeonCapturePayload, api_url: str, profile: str):
    """
    Transmit the generated payload to the REST API for VLM analysis.
    """
    console.print("\n[bold yellow]Transmitting data to AI for analysis...[/bold yellow]")

    # Base64 encode the attachments if they exist
    scene_b64 = base64.b64encode(payload.scene_image).decode("utf-8") if payload.scene_image else None
    audio_b64 = base64.b64encode(payload.audio).decode("utf-8") if payload.audio else None

    request_data = {
        "scene_image_base64": scene_b64,
        "audio_base64": audio_b64,
        "video_base64": [],  # Omit full video array to prevent massive HTTP payloads
        "gaze_points": payload.gaze_points,
        "pupil_data": payload.pupil_data,
        "blink_config": payload.blink_config.model_dump(),
        "fixation_config": payload.fixation_config.model_dump(),
        "saliency_config": payload.saliency_config.model_dump(),
        "recording_config": payload.recording_config.model_dump(),
        "profile": profile,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{api_url}/inference/neon-context", json=request_data)
            response.raise_for_status()

            result = response.json()

            console.print("\n[bold green]VLM Analysis Result:[/bold green]")
            import json

            if "parsed_visual" in result:
                console.print(
                    Panel(json.dumps(result["parsed_visual"], indent=2), border_style="green", title="Parsed Visual")
                )

            if "pragmatic" in result:
                console.print(
                    Panel(json.dumps(result["pragmatic"], indent=2), border_style="blue", title="Pragmatic Context")
                )

    except httpx.HTTPStatusError as e:
        console.print(f"[bold red]API Error ({e.response.status_code}):[/bold red] {e.response.text}")
    except Exception as e:
        console.print(f"[bold red]Connection Error:[/bold red] {e}")
        console.print(f"[dim]Ensure the SynNeon server is running at {api_url}[/dim]")
