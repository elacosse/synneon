"""Tests for the standalone Neon client (`synneon client`)."""

from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest

from synneon.apps import neon_client


def _jpeg() -> bytes:
    _, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    return jpeg.tobytes()


async def test_neon_capture_uses_fixation_and_saliency_from_toml(tmp_path):
    """The capture reads [fixation] and [saliency] from the Neon TOML instead of schema defaults."""
    config = tmp_path / "neon.toml"
    config.write_text(
        '[fixation]\nmin_duration_ms = 250.0\n\n[saliency]\nmarker_shape = "circle"\nmarker_radius = 33\n'
        "min_peak_threshold_pct = 0.7\n"
    )
    state = {**neon_client.app_state, "recent_frames": [(0.0, _jpeg())], "recent_gaze": [(0.0, 10.0, 10.0)]}
    with (
        patch.dict(neon_client.app_state, state),
        patch.object(neon_client.server, "send", AsyncMock()),
        patch.object(neon_client.server, "post_capture", AsyncMock(return_value={})) as post,
    ):
        await neon_client._run_neon_capture(config)

    request = post.call_args.args[0]
    assert request.fixation_config.min_duration_ms == 250.0
    assert request.saliency_config.marker_shape == "circle"
    assert request.saliency_config.marker_radius == 33
    assert request.saliency_config.min_peak_threshold_pct == 0.7


async def test_neon_capture_sends_images_then_result():
    """Phase 1 ships the images to the server; phase 2 posts the capture and relays the result."""
    frames = [(1.0, _jpeg()), (2.0, _jpeg())]
    gaze = [(1.0, 10.0, 10.0), (2.5, 12.0, 11.0)]
    state = {**neon_client.app_state, "recent_frames": frames, "recent_gaze": gaze, "neon_profile": "custom"}
    send = AsyncMock()
    result = {"description": "a desk"}
    with (
        patch.dict(neon_client.app_state, state),
        patch.object(neon_client.server, "send", send),
        patch.object(neon_client.server, "post_capture", AsyncMock(return_value=result)) as post,
    ):
        await neon_client._run_neon_capture(None)

    processed, complete = (call.args[0] for call in send.call_args_list)
    assert processed["event"] == "neon_capture_processed"
    assert processed["duration_s"] == 1.5
    assert {"frame_b64", "gaze_trajectory_b64", "gaze_fixations_b64"} <= processed["images"].keys()
    assert complete == {"event": "neon_capture_complete", "result": result}

    request = post.call_args.args[0]
    assert request.profile == "custom"
    assert request.save_to_memory is True
    assert len(request.video_base64) == 2
    assert request.gaze_points == gaze


async def test_neon_capture_reports_server_failure():
    state = {**neon_client.app_state, "recent_frames": [(0.0, _jpeg())], "recent_gaze": []}
    send = AsyncMock()
    with (
        patch.dict(neon_client.app_state, state),
        patch.object(neon_client.server, "send", send),
        patch.object(neon_client.server, "post_capture", AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        await neon_client._run_neon_capture(None)

    assert send.call_args.args[0] == {"event": "neon_capture_error", "detail": "boom"}


async def test_neon_capture_without_frames_reports_error():
    send = AsyncMock()
    with (
        patch.dict(neon_client.app_state, {"recent_frames": deque(), "recent_gaze": deque()}),
        patch.object(neon_client.server, "send", send),
    ):
        await neon_client._run_neon_capture(None)

    send.assert_awaited_once_with({"event": "neon_capture_error", "detail": "No frames buffered"})


async def test_toggle_recording_starts_then_stops_and_captures():
    send = AsyncMock()
    state = {"recording": False, "recent_frames": deque([(0.0, b"old")]), "recent_gaze": deque()}
    with (
        patch.dict(neon_client.app_state, state),
        patch.object(neon_client.server, "send", send),
        patch.object(neon_client, "_run_neon_capture", AsyncMock()) as capture,
    ):
        await neon_client.toggle_recording(None)
        assert neon_client.app_state["recording"] is True
        assert not neon_client.app_state["recent_frames"]

        await neon_client.toggle_recording(None)
        assert neon_client.app_state["recording"] is False
        await asyncio.sleep(0)  # let the capture task run

    events = [call.args[0]["event"] for call in send.call_args_list]
    assert events == ["neon_recording_started", "neon_recording_stopped"]
    capture.assert_awaited_once_with(None)


async def test_send_is_dropped_while_disconnected():
    link = neon_client.ServerLink()
    await link.send({"event": "pupil_data"})  # no socket: must not raise


def _device() -> MagicMock:
    device = MagicMock()
    device.close = AsyncMock()
    return device


async def _forever(*_args):
    await asyncio.Event().wait()
    yield  # pragma: no cover


async def _until(condition) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


async def test_hardware_loop_closes_the_device_when_cancelled():
    """Ctrl-C cancels the loop; the device's HTTP session must be closed, not left for the GC to warn about."""
    device = _device()
    with (
        patch.dict(neon_client.app_state, {"hardware_connected": False}),
        patch.object(neon_client.server, "send", AsyncMock()),
        patch.object(neon_client, "connect", AsyncMock(return_value=device)),
        patch.object(neon_client, "stream_eye_events", _forever),
        patch.object(neon_client, "stream_video", _forever),
        patch.object(neon_client, "stream_gaze", _forever),
    ):
        loop = asyncio.create_task(neon_client.hardware_loop(None, use_pupil=False))
        await _until(lambda: neon_client.app_state["hardware_connected"])
        loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop

    device.close.assert_awaited_once()


async def test_hardware_loop_closes_the_device_before_reconnecting():
    """A stream failure closes the old device (and its other streams) before the next connection attempt."""
    device = _device()

    async def failing_events(*_args):
        raise RuntimeError("stream dropped")
        yield  # pragma: no cover

    with (
        patch.dict(neon_client.app_state, {"hardware_connected": False}),
        patch.object(neon_client.server, "send", AsyncMock()),
        patch.object(neon_client, "connect", AsyncMock(return_value=device)),
        patch.object(neon_client, "stream_eye_events", failing_events),
        patch.object(neon_client, "stream_video", _forever),
        patch.object(neon_client, "stream_gaze", _forever),
    ):
        loop = asyncio.create_task(neon_client.hardware_loop(None, use_pupil=False))
        await _until(lambda: device.close.await_count == 1)
        assert neon_client.app_state["hardware_connected"] is False
        loop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loop
