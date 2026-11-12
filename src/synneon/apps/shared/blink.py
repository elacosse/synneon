from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
import typer
from pupil_labs.realtime_api import VideoFrame
from pynput.keyboard import Controller, Key

from synneon.schemas.neons import BlinkConfig, GazeSample

# Single persistent viewer subprocess shared across all BlinkManager instances.
_viewer_proc: subprocess.Popen | None = None
_viewer_lock = threading.Lock()


def ensure_viewer(debug: bool = False) -> subprocess.Popen:
    global _viewer_proc
    with _viewer_lock:
        if _viewer_proc is not None:
            status = _viewer_proc.poll()
            if status is not None:
                if debug:
                    typer.secho(
                        f"[viewer] Process {_viewer_proc.pid} died with {status}. Restarting.",
                        fg=typer.colors.YELLOW,
                    )
                _viewer_proc = None

        if _viewer_proc is None:
            if debug:
                typer.secho("[viewer] Starting new viewer process...", fg=typer.colors.YELLOW)
            _viewer_proc = subprocess.Popen(
                [sys.executable, "-m", "synneon.apps.shared.viewer"],
                stdin=subprocess.PIPE,
                stdout=None if debug else subprocess.DEVNULL,
                stderr=None if debug else subprocess.DEVNULL,
            )
        return _viewer_proc


def close_viewer() -> None:
    global _viewer_proc
    with _viewer_lock:
        if _viewer_proc is not None:
            _viewer_proc.terminate()
            try:
                _viewer_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _viewer_proc.kill()
            _viewer_proc = None


# Maps gesture direction strings to pynput keys
key_map = {
    "UP": Key.up,
    "DOWN": Key.down,
    "LEFT": Key.left,
    "RIGHT": Key.right,
}


def _clamp_int(v: float, lo: int, hi: int) -> int:
    """Clamp and convert to int, guarding against NaN/inf."""
    try:
        if v != v:  # NaN
            return lo
        if v == float("inf"):
            return hi
        if v == float("-inf"):
            return lo
    except Exception:
        return lo
    return max(lo, min(hi, int(v)))


def _describe_and_display(
    image_path: Path,
    url: str = "http://localhost:8000/inference/image2context",
    profile: str = "image2context",
    on_context: Callable[[dict], None] | None = None,
    debug: bool = False,
):
    """Call image2context API and push result to the persistent viewer window."""
    typer.secho(f"Describing capture: {image_path.name}...", fg=typer.colors.YELLOW)
    ext = image_path.suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    try:
        with httpx.Client(timeout=60.0) as client:
            with open(image_path, "rb") as f:
                response = client.post(
                    url, files={"file": (image_path.name, f, media_type)}, params={"profile": profile}
                )
            response.raise_for_status()
            result = response.json()
            if on_context:
                on_context(result)
    except Exception as e:
        typer.secho(f"[describe] Error: {e}", fg=typer.colors.RED)
        return

    msg = json.dumps({"image_path": str(image_path), "result": result}) + "\n"
    try:
        viewer = ensure_viewer(debug=debug)
        viewer.stdin.write(msg.encode())
        viewer.stdin.flush()
    except Exception as e:
        typer.secho(f"[viewer] send error: {e}", fg=typer.colors.RED)


class BlinkManager:
    def __init__(
        self,
        config: BlinkConfig,
        keyboard: Controller,
        save: bool = False,
        on_context: Callable[[dict], None] | None = None,
        debug: bool = False,
    ):
        self.config = config
        self.keyboard = keyboard
        self.save = save
        self.on_context = on_context
        self.debug = debug
        self.blink_timestamps: deque[float] = deque(maxlen=2)
        self.last_frame: VideoFrame | None = None
        self.last_valid_gaze: GazeSample | None = None

    def update_data(self, frame: VideoFrame | None, gaze: GazeSample | None):
        if frame:
            self.last_frame = frame
        if gaze:
            self.last_valid_gaze = gaze

    async def handle_blink(self, timestamp: float, output_dir: Path) -> bool:
        """Handle a blink event. Returns True if a double blink was detected."""
        self.blink_timestamps.append(timestamp)

        # Check for double blink
        if len(self.blink_timestamps) == 2:
            delta = self.blink_timestamps[1] - self.blink_timestamps[0]
            if delta <= self.config.double_blink_delta:
                await self._execute_action(
                    self.config.double_blink_action, output_dir, self.last_frame, self.last_valid_gaze
                )
                self.blink_timestamps.clear()
                return True
        return False

    async def _execute_action(self, action: str | None, output_dir: Path, frame=None, gaze=None):
        if action is None:
            return

        if action == "capture_world_camera_image_with_gaze_annotation":
            if frame and gaze:
                import tempfile

                import cv2

                img = frame.av_frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                x, y = _clamp_int(gaze.x, 0, w - 1), _clamp_int(gaze.y, 0, h - 1)
                cv2.circle(img, (x, y), self.config.circle_annotation_radius, (0, 0, 255), 3)

                if self.save:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filename = output_dir / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                    cv2.imwrite(str(filename), img)
                    typer.secho(f"Captured: {filename}", fg=typer.colors.BLUE)
                else:
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    filename = Path(tmp.name)
                    cv2.imwrite(str(filename), img)

                threading.Thread(
                    target=_describe_and_display,
                    args=(filename,),
                    kwargs={"on_context": self.on_context, "debug": self.debug},
                    daemon=True,
                ).start()
