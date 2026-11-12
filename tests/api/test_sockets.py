"""Tests for the WebSocket hub that relays the Neon client to dashboard browsers.

Starlette's TestClient runs each WebSocket on its own event loop, so relaying between sockets
deadlocks there. These tests run the router on a real in-process uvicorn server instead.
"""

from __future__ import annotations

import asyncio
import base64
import json
import socket
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from websockets.asyncio.client import ClientConnection, connect

from synneon.api import sockets

NO_CLIENT = [
    {"event": "app_config", "pupil": False, "gesture": False, "mock": False},
    {"event": "neon_status", "connected": False, "device_addr": "No device"},
    {"event": "worn_status", "worn": None},
]


@pytest.fixture
async def base_url(monkeypatch) -> AsyncIterator[str]:
    monkeypatch.setattr(sockets, "manager", sockets.ConnectionManager())
    app = FastAPI()
    app.include_router(sockets.router)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    yield f"127.0.0.1:{port}"
    server.should_exit = True
    await task


async def _receive(ws: ClientConnection, n: int) -> list[dict]:
    return [json.loads(await asyncio.wait_for(ws.recv(), timeout=5)) for _ in range(n)]


async def _until(condition) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


async def test_browser_gets_no_client_state_on_connect(base_url):
    async with connect(f"ws://{base_url}/ws") as browser:
        assert await _receive(browser, 3) == NO_CLIENT


async def test_client_events_reach_browsers_and_reset_on_disconnect(base_url):
    status = {"event": "neon_status", "connected": True, "device_addr": "Mock Device (localhost)"}
    pupil = {"event": "pupil_data", "left_mm": 4.5, "right_mm": 4.4, "timestamp": 1.0}
    async with connect(f"ws://{base_url}/ws") as browser:
        await _receive(browser, 3)
        async with connect(f"ws://{base_url}/ws/neon") as neon:
            await neon.send(json.dumps(status))
            await neon.send(json.dumps(pupil))
            assert await _receive(browser, 2) == [status, pupil]

            # A tab opened now sees the device as connected.
            async with connect(f"ws://{base_url}/ws") as late_browser:
                assert (await _receive(late_browser, 3))[1] == status

        assert await _receive(browser, 3) == NO_CLIENT


async def test_capture_images_are_kept_for_live_frame_not_broadcast(base_url):
    jpeg = b"\xff\xd8fake-jpeg"
    processed = {
        "event": "neon_capture_processed",
        "fixations": [],
        "duration_s": 1.5,
        "images": {"frame_b64": base64.b64encode(jpeg).decode()},
    }
    async with httpx.AsyncClient(base_url=f"http://{base_url}") as http:
        assert (await http.get("/viz/live-frame")).status_code == 404
        async with connect(f"ws://{base_url}/ws") as browser, connect(f"ws://{base_url}/ws/neon") as neon:
            await _receive(browser, 3)
            await neon.send(json.dumps(processed))
            assert await _receive(browser, 1) == [
                {"event": "neon_capture_processed", "fixations": [], "duration_s": 1.5}
            ]

        response = await http.get("/viz/live-frame", params={"type": "scene"})
        assert response.status_code == 200
        assert response.content == jpeg
        assert response.headers["content-type"] == "image/jpeg"
        assert (await http.get("/viz/live-frame", params={"type": "saliency_map"})).status_code == 404


async def test_fake_blink_is_forwarded_to_the_client(base_url):
    async with httpx.AsyncClient(base_url=f"http://{base_url}") as http:
        assert (await http.post("/debug/fake-blink")).status_code == 409
        async with connect(f"ws://{base_url}/ws/neon") as neon:
            await _until(lambda: sockets.manager.client_ws is not None)
            assert (await http.post("/debug/fake-blink")).json() == {"status": "sent"}
            assert await _receive(neon, 1) == [{"command": "fake_blink"}]


async def test_new_client_replaces_the_old_one(base_url):
    async with connect(f"ws://{base_url}/ws/neon") as first:
        await _until(lambda: sockets.manager.client_ws is not None)
        async with connect(f"ws://{base_url}/ws/neon"):
            await asyncio.wait_for(first.wait_closed(), timeout=5)
            assert sockets.manager.client_ws is not None


async def test_broadcast_drops_dead_sockets():
    manager = sockets.ConnectionManager()
    alive, dead = AsyncMock(), AsyncMock()
    dead.send_json.side_effect = RuntimeError("closed")
    manager.active_connections = [alive, dead]

    await manager.broadcast({"event": "x"})

    alive.send_json.assert_awaited_once_with({"event": "x"})
    assert manager.active_connections == [alive]
