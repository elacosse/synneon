from __future__ import annotations

import asyncio
import platform
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from synneon.schemas.neons import GazeSample

# Import pynput only when needed to avoid overhead if unused,
# but we'll load it once in the generator.
try:
    from pynput import mouse  # type: ignore[import-not-found]
except ImportError:
    mouse = None  # type: ignore


@dataclass(frozen=True)
class MouseStreamConfig:
    """
    Configuration for mouse position sampling.

    The stream yields `GazeSample` instances whose `x`/`y` are normalized to [0, 1]
    in screen coordinates (origin at top-left).

    Notes:
    - `poll_hz`: Ignored in this efficient implementation. The mouse position
      is queried synchronously immediately before emission to ensure minimal latency.
    - `emit_hz`: Controls how frequently we yield samples.
    """

    poll_hz: float = 240.0
    emit_hz: float = 60.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _get_screen_size_px() -> tuple[float, float]:
    """
    Return (width_px, height_px) for the primary display.
    """
    if platform.system() == "Darwin":
        # Use Quartz directly via PyObjC (already pulled in by pynput on macOS).
        try:
            from Quartz import (  # type: ignore[import-not-found]
                CGDisplayBounds,
                CGMainDisplayID,
            )

            bounds = CGDisplayBounds(CGMainDisplayID())
            return float(bounds.size.width), float(bounds.size.height)
        except ImportError:
            pass  # Fallback to tkinter if Quartz is missing (unlikely on macOS with pynput)

    # Cross-platform fallback.
    try:
        import tkinter as tk  # noqa: PLC0415

        root = tk.Tk()
        root.withdraw()
        width = float(root.winfo_screenwidth())
        height = float(root.winfo_screenheight())
        root.destroy()
        return width, height
    except Exception:
        # Last resort fallback to avoid crashing if headless
        return 1920.0, 1080.0


async def stream_mouse_gaze(
    *,
    is_worn: bool = True,
    config: MouseStreamConfig | None = None,
) -> AsyncIterator[GazeSample]:
    """
    Async stream of mouse positions as gaze-like samples.

    Yields:
        `GazeSample(x, y, timestamp, is_worn)` where:
        - `x`, `y` are normalized floats in [0, 1] (top-left origin)
        - `timestamp` is `time.monotonic()` seconds
        - `is_worn` is configurable (defaults True to behave like "device worn")
    """
    cfg = config or MouseStreamConfig()

    if cfg.emit_hz <= 0:
        raise ValueError("MouseStreamConfig.emit_hz must be > 0")

    period = 1.0 / float(cfg.emit_hz)

    # Initialize controller and screen size once
    if mouse is None:
        raise RuntimeError("pynput is not installed.")

    controller = mouse.Controller()

    # Cache screen size.
    # NOTE: If screen resolution changes during the session, this will be wrong.
    # But for efficiency we don't want to query this every frame if it requires heavy calls.
    # Quartz call is fast, but let's be safe.
    width, height = _get_screen_size_px()
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    inv_width = 1.0 / width
    inv_height = 1.0 / height

    # Timing loop state
    t_target = time.monotonic()

    # Main Loop
    while True:
        now = time.monotonic()

        # Sleep until the next target time
        sleep_duration = t_target - now
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
            # Re-read time after sleep
            now = time.monotonic()

        # --- Critical Section: Poll & Yield ---
        # We poll synchronously. Benchmark shows this takes ~60us on macOS,
        # which is negligible compared to the ~16ms frame time (60Hz).
        # This avoids the overhead of thread switching and executor queues.
        try:
            x_px, y_px = controller.position
            # Normalize
            x = _clamp01(float(x_px) * inv_width)
            y = _clamp01(float(y_px) * inv_height)

            yield GazeSample(x=x, y=y, timestamp=now, is_worn=is_worn)
        except Exception:
            # If polling fails (e.g. screen lock, permissions), just yield center or last?
            # Better to yield nothing and wait? Or just continue.
            # We'll continue to avoid breaking the loop.
            pass

        # --- Timing Update ---
        t_target += period

        # Lag recovery:
        # If we are more than 1 period behind, reset the schedule to avoid bursting.
        # This effectively drops frames that we missed.
        if t_target < now - period:
            t_target = now + period
