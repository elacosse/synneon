from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PupilSample:
    """
    A single pupillometry sample.

    Notes:
    - `left_mm`/`right_mm` may be NaN to intentionally create a visual discontinuity ("gap").
    - `worn` indicates whether the device is being worn at the time of the sample.
    """

    timestamp_unix_seconds: float
    left_mm: float
    right_mm: float
    worn: bool


@dataclass(frozen=True)
class PupilometryPlotConfig:
    """
    Configuration for plotting and buffering.

    Attributes:
        window_s: Sliding window duration in seconds.
        refresh_hz: Plot refresh rate (redraw cadence) in Hz.
        title: Plot title.
    """

    window_s: float = 30.0
    refresh_hz: float = 20.0
    title: str = "Neon Pupillometry (mm) - Left / Right"


def _is_finite_number(v: Any) -> bool:
    try:
        fv = float(v)
    except Exception:
        return False
    return math.isfinite(fv)


def _nan() -> float:
    return float("nan")


def _to_mm_or_nan(v: Any) -> float:
    """
    Convert an input value to float if finite; otherwise return NaN.

    This is used to ensure Matplotlib line plots break on invalid values.
    """
    if not _is_finite_number(v):
        return _nan()
    return float(v)


def prune_sliding_window_in_place(samples: deque[PupilSample], *, window_s: float) -> None:
    """
    Prune samples older than newest_timestamp - window_s, in-place.

    Args:
        samples: Deque of time-ordered samples.
        window_s: Window size in seconds. Must be > 0.
    """
    if window_s <= 0:
        raise ValueError("window_s must be > 0")

    if not samples:
        return

    newest_ts = samples[-1].timestamp_unix_seconds
    cutoff = newest_ts - float(window_s)
    while samples and samples[0].timestamp_unix_seconds < cutoff:
        samples.popleft()


async def stream_pupillometry_from_gaze_rtsp(
    rtsp_url: str,
    *,
    max_no_pupil_s: float = 5.0,
    gap_on_not_worn: bool = True,
) -> AsyncIterator[PupilSample]:
    """
    Stream pupillometry from the Neon RTSP gaze stream.

    This reads from `pupil_labs.realtime_api.streaming.receive_gaze_data` and yields
    `PupilSample`s when the incoming gaze packet is an eyestate variant that includes
    `pupil_diameter_left`/`pupil_diameter_right`.

    Behavior:
    - If `gap_on_not_worn=True` and `worn == False`, yields a sample with NaN for both eyes
      (Option A: show a gap for both traces).
    - If no pupillometry-capable packets are seen for `max_no_pupil_s`, raises RuntimeError.

    Args:
        rtsp_url: RTSP URL for the Neon gaze stream.
        max_no_pupil_s: Max seconds to wait before erroring if no eyestate packets appear.
        gap_on_not_worn: When not worn, yield NaN/NaN to force gaps (both eyes).

    Yields:
        PupilSample instances in arrival order.
    """
    # Local import to keep module import-time light and to avoid hard dependency
    # during tests that do not exercise streaming.
    from pupil_labs.realtime_api.streaming import receive_gaze_data  # type: ignore

    start_monotonic = time.monotonic()
    saw_pupil_packet = False

    async for gaze in receive_gaze_data(rtsp_url):
        # Many packet types exist; pupillometry is present only on eyestate variants.
        has_pupil = hasattr(gaze, "pupil_diameter_left") and hasattr(gaze, "pupil_diameter_right")
        if not has_pupil:
            if not saw_pupil_packet and (time.monotonic() - start_monotonic) >= max_no_pupil_s:
                raise RuntimeError(
                    "No pupillometry fields detected in gaze stream within timeout. "
                    "Expected eyestate gaze packets with pupil_diameter_left/right."
                )
            continue

        if not saw_pupil_packet:
            saw_pupil_packet = True
            logger.info("Detected pupillometry-capable gaze packet type: %s", type(gaze).__name__)

        # Required fields across gaze packet variants in this library:
        # - worn (bool)
        # - timestamp_unix_seconds (float)
        worn = bool(getattr(gaze, "worn", True))
        ts = float(gaze.timestamp_unix_seconds)

        if not worn and gap_on_not_worn:
            yield PupilSample(timestamp_unix_seconds=ts, left_mm=_nan(), right_mm=_nan(), worn=False)
            continue

        left = _to_mm_or_nan(getattr(gaze, "pupil_diameter_left", _nan()))
        right = _to_mm_or_nan(getattr(gaze, "pupil_diameter_right", _nan()))
        yield PupilSample(timestamp_unix_seconds=ts, left_mm=left, right_mm=right, worn=worn)


def _snapshot_xy_for_plot(
    samples: Iterable[PupilSample],
    *,
    window_s: float,
) -> tuple[list[float], list[float], list[float]]:
    """
    Convert samples into plot-ready x/y lists.

    X is relative time in seconds ending at 0 (newest sample at x=0),
    spanning approximately [-window_s, 0].

    Returns:
        xs, lefts, rights
    """
    s_list = list(samples)
    if not s_list:
        return [], [], []

    newest_ts = s_list[-1].timestamp_unix_seconds
    xs: list[float] = []
    lefts: list[float] = []
    rights: list[float] = []
    for s in s_list:
        xs.append(s.timestamp_unix_seconds - newest_ts)
        lefts.append(s.left_mm)
        rights.append(s.right_mm)

    # Ensure the x-axis window is stable for empty leading windows; caller sets xlim.
    _ = window_s
    return xs, lefts, rights


async def run_realtime_pupilometry_plot(
    *,
    rtsp_url: str,
    window_s: float = 30.0,
    refresh_hz: float = 20.0,
    max_no_pupil_s: float = 5.0,
    title: str = "Neon Pupillometry (mm) - Left / Right",
) -> None:
    """
    Run a real-time Matplotlib interactive plot of pupillometry with a sliding window.

    This function:
    - spawns an asyncio task to consume the RTSP gaze stream and append to a rolling buffer
    - runs a Matplotlib GUI window loop that refreshes at `refresh_hz`
    - displays left/right pupil diameter over the last `window_s` seconds
    - inserts NaN gaps for both traces when `worn == False` (Option A)

    Args:
        rtsp_url: RTSP URL for Neon gaze stream.
        window_s: Sliding window duration in seconds (default 30.0).
        refresh_hz: Plot refresh rate in Hz (default 20.0).
        max_no_pupil_s: Time to wait for pupillometry-capable packets before failing.
        title: Window title.
    """
    if window_s <= 0:
        raise ValueError("window_s must be > 0")
    if refresh_hz <= 0:
        raise ValueError("refresh_hz must be > 0")

    # Import matplotlib lazily so importing synneon doesn't require a GUI backend.
    import matplotlib.pyplot as plt  # type: ignore

    buffer: deque[PupilSample] = deque()
    buffer_lock = asyncio.Lock()
    stop_event = asyncio.Event()

    async def _consume_stream() -> None:
        try:
            async for sample in stream_pupillometry_from_gaze_rtsp(
                rtsp_url,
                max_no_pupil_s=max_no_pupil_s,
                gap_on_not_worn=True,
            ):
                if stop_event.is_set():
                    break
                async with buffer_lock:
                    buffer.append(sample)
                    prune_sliding_window_in_place(buffer, window_s=window_s)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Pupillometry stream task failed.")
            raise
        finally:
            stop_event.set()

    stream_task = asyncio.create_task(_consume_stream())

    # --- Matplotlib interactive window setup ---
    plt.ion()
    fig, ax = plt.subplots()
    try:
        # Window title (best-effort; backend-dependent)
        try:
            fig.canvas.manager.set_window_title("synneon pupil")  # type: ignore[attr-defined]
        except Exception:
            pass

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pupil diameter (mm)")

        (line_left,) = ax.plot([], [], label="Left", linewidth=1.5)
        (line_right,) = ax.plot([], [], label="Right", linewidth=1.5)

        ax.set_xlim(-float(window_s), 0.0)
        ax.legend(loc="upper right")

        period_s = 1.0 / float(refresh_hz)

        while not stop_event.is_set() and plt.fignum_exists(fig.number):
            async with buffer_lock:
                xs, lefts, rights = _snapshot_xy_for_plot(buffer, window_s=window_s)

            # Update line data
            line_left.set_data(xs, lefts)
            line_right.set_data(xs, rights)

            # Keep x-axis fixed to the sliding window
            ax.set_xlim(-float(window_s), 0.0)

            # Autoscale y with a small margin; if all values are NaN, skip autoscale.
            any_finite = any(math.isfinite(v) for v in lefts) or any(math.isfinite(v) for v in rights)
            if any_finite:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            await asyncio.sleep(period_s)

    finally:
        stop_event.set()
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # Stream task exceptions already logged.
            pass
        try:
            plt.close(fig)
        except Exception:
            pass
