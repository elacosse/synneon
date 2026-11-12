"""`synneon client`: connects a Neon device (or mock data) to a running SynNeon server.

Telemetry goes to the server's `/ws/neon` socket, which relays it to dashboard browsers. A double blink
starts and stops a capture; the client renders the gaze visualizations and posts the capture to
`POST /inference/neon-context`. The server needs no hardware, and this client needs no Redis.
"""

import asyncio
import base64
import json
import logging
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
import typer
from dotenv import load_dotenv
from pupil_labs.realtime_api import Device, receive_gaze_data
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from synneon.apps.shared.blink import BlinkManager
from synneon.apps.shared.config import (
    load_blink_config,
    load_fixation_config,
    load_neon_config,
    load_saliency_config,
)
from synneon.neon.session import (
    _get_rtsp_url,
    connect,
    stream_eye_events,
    stream_gaze,
    stream_video,
)
from synneon.schemas.neons import BlinkConfig, FixationConfig, RecordingConfig, SaliencyConfig
from synneon.schemas.requests import NeonCaptureInferenceRequest
from synneon.services.physiology.pupilometry import _nan, _to_mm_or_nan

load_dotenv()

logger = logging.getLogger(__name__)

# The capture endpoint runs the VLM synchronously, so allow it time.
CAPTURE_TIMEOUT_S = 180.0
MAX_RECONNECT_DELAY_S = 10.0

app_state: dict[str, Any] = {
    "hardware_connected": False,
    "device_addr": "No device",
    "recent_frames": deque(maxlen=8),  # (ts: float, jpeg: bytes)
    "recent_gaze": deque(maxlen=12000),  # (ts: float, x: float, y: float) — 60s @ 200 Hz
    "recording": False,  # True while between start/end double-blink
    "worn": None,  # True when device is on face (gaze stream confirms); None until known
    "app_config": {"pupil": False, "gesture": False, "mock": False, "profile": "default"},
    "neon_profile": "image2context",
}


class ServerLink:
    """The client's connection to the server: events over `/ws/neon`, captures over HTTP."""

    def __init__(self) -> None:
        self.api_url = "http://127.0.0.1:8765"
        self.user_id = "default"
        self.ws: ClientConnection | None = None

    async def send(self, message: dict[str, Any]) -> None:
        """Send an event to the server; dropped while disconnected (state is resent on reconnect)."""
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps(message))
        except WebSocketException:
            pass

    async def post_capture(self, request: NeonCaptureInferenceRequest) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=CAPTURE_TIMEOUT_S) as http:
            response = await http.post(
                f"{self.api_url}/inference/neon-context",
                params={"user_id": self.user_id},
                json=request.model_dump(mode="json"),
            )
        if response.is_error:
            raise RuntimeError(f"Capture inference failed ({response.status_code}): {response.text}")
        return response.json()

    async def _send_state(self) -> None:
        await self.send({"event": "app_config", **app_state["app_config"]})
        await self.send(
            {
                "event": "neon_status",
                "connected": app_state["hardware_connected"],
                "device_addr": app_state["device_addr"],
            }
        )
        if app_state["worn"] is not None:
            await self.send({"event": "worn_status", "worn": app_state["worn"]})

    async def run_forever(self, neon_config_path: Path | None) -> None:
        """Keep a `/ws/neon` connection open, reconnecting with backoff, and handle server commands."""
        ws_url = self.api_url.replace("http", "ws", 1).rstrip("/") + "/ws/neon"
        delay = 1.0
        while True:
            try:
                async with ws_connect(ws_url) as ws:
                    self.ws = ws
                    delay = 1.0
                    logger.info(f"Connected to {ws_url}")
                    await self._send_state()
                    async for raw in ws:
                        command = json.loads(raw).get("command")
                        if command == "fake_blink":
                            logger.info("FAKE double blink from server")
                            await toggle_recording(neon_config_path)
            except (OSError, WebSocketException, ValueError) as e:
                logger.warning(f"Server connection to {ws_url} lost or refused: {e}")
            finally:
                self.ws = None
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY_S)


server = ServerLink()


def _start_recording():
    """Mark recording active and flush stale buffer data so captures only span the recording window."""
    app_state["recent_frames"].clear()
    app_state["recent_gaze"].clear()
    app_state["recording"] = True


async def toggle_recording(neon_config_path: Path | None) -> None:
    """A double blink starts a capture, and the next one ends it and sends it for inference."""
    if not app_state["recording"]:
        _start_recording()
        logger.info("Double blink — starting capture")
        await server.send({"event": "neon_recording_started"})
    else:
        app_state["recording"] = False
        logger.info("Double blink — ending capture")
        await server.send({"event": "neon_recording_stopped"})
        asyncio.create_task(_run_neon_capture(neon_config_path))


def _encode_bgr_png(img) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    import cv2

    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


async def _run_neon_capture(neon_config_path: Path | None) -> None:
    """Run the two-phase capture pipeline: CV here first, then VLM inference on the server."""
    import cv2
    import numpy as np

    from synneon.services.perception.vision import (
        compute_gaze_saliency_map,
        detect_fixations_idt,
        draw_fixations,
        draw_gaze_trajectory,
    )

    frames = list(app_state["recent_frames"])
    gaze = list(app_state["recent_gaze"])

    if not frames:
        await server.send({"event": "neon_capture_error", "detail": "No frames buffered"})
        return

    try:
        blink_cfg = load_blink_config(neon_config_path)
    except Exception:
        blink_cfg = BlinkConfig()

    try:
        fixation_cfg = load_fixation_config(neon_config_path)
        saliency_cfg = load_saliency_config(neon_config_path)
    except Exception:
        fixation_cfg = FixationConfig()
        saliency_cfg = SaliencyConfig()

    # ── Phase 1: local CV (fast) ──────────────────────────────────────────────
    fixations: list[dict] = []
    viz_b64: dict[str, str] = {}
    frame_b64 = base64.b64encode(frames[-1][1]).decode()

    # Capture duration from gaze timestamps (most accurate clock source)
    duration_s: float = 0.0
    if len(gaze) >= 2:
        duration_s = round(gaze[-1][0] - gaze[0][0], 2)
    elif len(frames) >= 2:
        duration_s = round(frames[-1][0] - frames[0][0], 2)

    try:
        buf = np.frombuffer(frames[-1][1], dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        fixations = detect_fixations_idt(
            gaze,
            min_duration_ms=fixation_cfg.min_duration_ms,
            max_dispersion_pixels=fixation_cfg.max_dispersion_pixels,
        )

        saliency_map_gray, saliency_overlay, peaks_overlay = compute_gaze_saliency_map(
            frame_bgr,
            gaze,
            gaussian_sigma=saliency_cfg.gaussian_sigma,
            colormap=saliency_cfg.colormap,
            alpha=saliency_cfg.alpha,
            max_peaks=saliency_cfg.max_peaks,
            min_peak_threshold_pct=saliency_cfg.min_peak_threshold_pct,
            marker_shape=saliency_cfg.marker_shape,
            marker_radius=saliency_cfg.marker_radius,
        )

        trajectory_img = draw_gaze_trajectory(frame_bgr, gaze)
        fixation_img = draw_fixations(frame_bgr, fixations)

        viz_b64["saliency_map_b64"] = _encode_bgr_png(cv2.cvtColor(saliency_map_gray, cv2.COLOR_GRAY2BGR))
        viz_b64["saliency_overlay_b64"] = _encode_bgr_png(saliency_overlay)
        if peaks_overlay is not None:
            viz_b64["saliency_peaks_b64"] = _encode_bgr_png(peaks_overlay)
        viz_b64["gaze_trajectory_b64"] = _encode_bgr_png(trajectory_img)
        viz_b64["gaze_fixations_b64"] = _encode_bgr_png(fixation_img)
    except Exception as e:
        logger.warning(f"Fixation pipeline failed: {e}")

    # The server keeps the images for /viz/live-frame and tells browsers to fetch them
    # before the VLM finishes.
    await server.send(
        {
            "event": "neon_capture_processed",
            "fixations": fixations,
            "duration_s": duration_s,
            "images": {"frame_b64": frame_b64, **viz_b64},
        }
    )

    # ── Phase 2: VLM inference on the server (slow) ───────────────────────────
    request = NeonCaptureInferenceRequest(
        scene_image_base64=frame_b64,
        video_base64=[(ts, base64.b64encode(jpeg).decode()) for ts, jpeg in frames],
        gaze_points=[(ts, x, y) for ts, x, y in gaze],
        pupil_data=[],
        blink_config=blink_cfg,
        fixation_config=fixation_cfg,
        saliency_config=saliency_cfg,
        recording_config=RecordingConfig(),
        profile=app_state["neon_profile"],
        save_to_memory=True,
    )

    try:
        result = await server.post_capture(request)
    except Exception as e:
        logger.error(f"Neon capture inference failed: {e}")
        await server.send({"event": "neon_capture_error", "detail": str(e)})
        return

    await server.send({"event": "neon_capture_complete", "result": result})


async def mock_hardware_loop(use_pupil: bool):
    """Background task that sends mock hardware data to the server."""
    app_state["hardware_connected"] = True
    app_state["device_addr"] = "Mock Device (localhost)"
    await server.send({"event": "neon_status", "connected": True, "device_addr": app_state["device_addr"]})

    logger.info("Started MOCK hardware loop")

    start_time = time.monotonic()

    while True:
        try:
            current_time = time.monotonic()
            elapsed = current_time - start_time

            # Send mock pupil data at ~20Hz
            if use_pupil:
                # Generate realistic-looking pupil fluctuations (hippus + noise)
                base_size = 4.5
                hippus_l = math.sin(elapsed) * 0.2
                hippus_r = math.sin(elapsed + 0.3) * 0.2
                noise_l = (random.random() - 0.5) * 0.1
                noise_r = (random.random() - 0.5) * 0.1

                left_mm = base_size + hippus_l + noise_l
                right_mm = base_size + hippus_r + noise_r

                await server.send(
                    {"event": "pupil_data", "left_mm": left_mm, "right_mm": right_mm, "timestamp": time.time()}
                )

            await asyncio.sleep(0.05)  # 20Hz

        except Exception as e:
            logger.error(f"Mock hardware loop error: {e}")
            await asyncio.sleep(1)


async def hardware_loop(neon_config_path: Path | None, use_pupil: bool, use_gesture: bool = False):
    """Background task to manage Neon connection and event streaming"""
    while True:
        device = None
        tasks: list[asyncio.Task[None]] = []
        try:
            neon_config = load_neon_config(neon_config_path)
            blink_config = load_blink_config(neon_config_path)
            # Suppress image capture
            blink_config.double_blink_action = None

            logger.info(f"Connecting to Neon using config: {neon_config_path}")
            device = await connect(neon_config)

            app_state["hardware_connected"] = True
            app_state["device_addr"] = neon_config.client_addr
            await server.send({"event": "neon_status", "connected": True, "device_addr": neon_config.client_addr})

            blink_manager = BlinkManager(blink_config, keyboard=None)

            async def event_task(device=device, blink_manager=blink_manager):
                from pupil_labs.realtime_api.streaming import BlinkEventData

                dummy_dir = Path("/tmp")
                logger.info("Eye events stream starting…")
                event_count = 0
                async for event in stream_eye_events(device):
                    event_count += 1
                    logger.debug(f"Eye event #{event_count}: {type(event).__name__} — {event}")
                    if isinstance(event, BlinkEventData):
                        if not app_state["worn"]:
                            logger.debug("Blink event ignored — device not worn")
                            continue
                        logger.info(f"Blink event received (rtp_ts={getattr(event, 'rtp_ts_unix_seconds', None)})")
                        try:
                            ts = getattr(event, "rtp_ts_unix_seconds", None) or time.time()
                            await server.send({"event": "blink_detected", "ts": float(ts)})
                            is_double = await blink_manager.handle_blink(float(ts), dummy_dir)
                            logger.info(
                                f"handle_blink → is_double={is_double}, buffer={list(blink_manager.blink_timestamps)}, delta_cfg={blink_manager.config.double_blink_delta}"
                            )
                            if is_double:
                                await toggle_recording(neon_config_path)
                        except Exception as e:
                            logger.warning(f"Blink event handling error: {e}")
                logger.info(f"Eye events stream ended after {event_count} events")

            tasks.append(asyncio.create_task(event_task()))

            async def video_buffer_task(device=device):
                import cv2

                try:
                    async for frame in stream_video(device):
                        try:
                            ts = float(getattr(frame, "timestamp_unix_seconds", time.time()))
                            img = frame.av_frame.to_ndarray(format="bgr24")
                            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            app_state["recent_frames"].append((ts, buf.tobytes()))
                        except Exception as e:
                            logger.debug(f"Video buffer frame error: {e}")
                except Exception as e:
                    logger.warning(f"Video buffer stream error (blink detection continues): {e}")

            tasks.append(asyncio.create_task(video_buffer_task()))

            async def gaze_buffer_task(device=device):
                """Buffer gaze data for capture pipeline; send worn-status changes."""
                last_worn: bool | None = None
                try:
                    async for gaze in stream_gaze(device):
                        app_state["recent_gaze"].append((gaze.timestamp, gaze.x, gaze.y))
                        if gaze.is_worn != last_worn:
                            last_worn = gaze.is_worn
                            app_state["worn"] = gaze.is_worn
                            await server.send({"event": "worn_status", "worn": gaze.is_worn})
                except Exception as e:
                    logger.warning(f"Gaze buffer stream error: {e}")

            tasks.append(asyncio.create_task(gaze_buffer_task()))

            if use_pupil:

                async def pupil_task(device=device):
                    gaze_rtsp_url = await _get_rtsp_url(device, "gaze")
                    last_send_time = 0

                    async for gaze in receive_gaze_data(gaze_rtsp_url):
                        # Mimic stream_pupillometry_from_gaze_rtsp using receive_gaze_data
                        has_pupil = hasattr(gaze, "pupil_diameter_left") and hasattr(gaze, "pupil_diameter_right")
                        if not has_pupil:
                            continue

                        worn = bool(getattr(gaze, "worn", True))
                        ts = float(gaze.timestamp_unix_seconds)

                        if not worn:
                            left = _nan()
                            right = _nan()
                        else:
                            left = _to_mm_or_nan(getattr(gaze, "pupil_diameter_left", _nan()))
                            right = _to_mm_or_nan(getattr(gaze, "pupil_diameter_right", _nan()))

                        # Throttle to ~20 Hz (50ms)
                        current_time = time.monotonic()
                        if current_time - last_send_time >= 0.05:
                            last_send_time = current_time
                            await server.send(
                                {"event": "pupil_data", "left_mm": left, "right_mm": right, "timestamp": ts}
                            )

                tasks.append(asyncio.create_task(pupil_task()))

            if use_gesture:
                from collections import deque as _deque

                from synneon.apps.shared.config import load_gestures_config
                from synneon.input.detector import DetectorState, detect_gestures
                from synneon.schemas.neons import GazeSample

                gestures_config = load_gestures_config(neon_config_path)
                compass_config = gestures_config.compass
                sample_time_length_s = float(getattr(compass_config, "sample_time_length", 0.5))

                async def gesture_task(
                    device=device, compass_config=compass_config, sample_time_length_s=sample_time_length_s
                ):
                    detector_state = DetectorState()
                    window: _deque[GazeSample] = _deque()
                    gesture_to_direction = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right"}
                    async for gaze in stream_gaze(device):
                        window.append(gaze)
                        while window and (gaze.timestamp - window[0].timestamp) > sample_time_length_s:
                            window.popleft()
                        event = detect_gestures(list(window), detector_state, compass_config)
                        if event:
                            direction = gesture_to_direction.get(event.type.value)
                            if direction:
                                await server.send({"event": "compass_gesture", "direction": direction})

                tasks.append(asyncio.create_task(gesture_task()))

            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Hardware loop error: {e}")
            app_state["hardware_connected"] = False
            await server.send({"event": "neon_status", "connected": False, "device_addr": app_state["device_addr"]})
        finally:
            # Runs on errors and on Ctrl-C: stop the remaining streams, then close the device's HTTP session.
            await _shutdown_device(device, tasks)
        await asyncio.sleep(5)  # Wait before reconnecting


async def _shutdown_device(device: Device | None, tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if device is not None:
        try:
            await device.close()
        except Exception as e:
            logger.debug(f"Closing Neon device failed: {e}")


async def _run_client(config_path: Path | None, pupil: bool, gesture: bool, mock: bool) -> None:
    loop = mock_hardware_loop(pupil) if mock else hardware_loop(config_path, pupil, gesture)
    hardware = asyncio.create_task(loop)
    try:
        await server.run_forever(config_path)
    finally:
        hardware.cancel()


def run_client(
    api_url: str = typer.Option(
        "http://127.0.0.1:8765", "--api-url", help="SynNeon server to connect to (`synneon app` or `synneon.server`)"
    ),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to configuration file"),
    pupil: bool = typer.Option(False, "--pupil", help="Enable pupil diameter streaming"),
    gesture: bool = typer.Option(False, "--gesture", help="Enable gaze gesture detection for compass navigation"),
    mock: bool = typer.Option(False, "--mock", help="Send mock hardware data instead of connecting to a Neon"),
    profile: str = typer.Option("default", "--profile", help="Agent profile shown on the dashboard"),
    neon_profile: str = typer.Option("image2context", "--neon-profile", help="Agent profile for captures"),
    user_id: str = typer.Option("default", "--user-id", help="User ID that captures are saved under"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """
    Connect a Neon device (or --mock data) to a running SynNeon server and its dashboard.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    server.api_url = api_url.rstrip("/")
    server.user_id = user_id
    app_state["app_config"] = {"pupil": pupil, "gesture": gesture, "mock": mock, "profile": profile}
    app_state["neon_profile"] = neon_profile

    logger.info(f"Starting SynNeon Neon client against {server.api_url}")
    asyncio.run(_run_client(config_path, pupil, gesture, mock))
