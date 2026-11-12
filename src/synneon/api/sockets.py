"""WebSocket hub: dashboard browsers on `/ws`, the Neon client (`synneon client`) on `/ws/neon`.

The server never talks to hardware. It relays the client's events to every browser, remembers the
latest device state so a newly opened tab is current, and keeps the latest capture images for
`GET /viz/live-frame`.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# State events the client sends; the latest of each is replayed to every new browser.
# These values are also what browsers are told when no client is attached.
_NO_CLIENT_STATE: dict[str, dict[str, Any]] = {
    "app_config": {"event": "app_config", "pupil": False, "gesture": False, "mock": False},
    "neon_status": {"event": "neon_status", "connected": False, "device_addr": "No device"},
    "worn_status": {"event": "worn_status", "worn": None},
}

LIVE_FRAME_TYPES: dict[str, tuple[str, str]] = {
    "scene": ("frame_b64", "image/jpeg"),
    "saliency_map": ("saliency_map_b64", "image/png"),
    "saliency_overlay": ("saliency_overlay_b64", "image/png"),
    "saliency_peaks": ("saliency_peaks_b64", "image/png"),
    "gaze_fixations": ("gaze_fixations_b64", "image/png"),
    "gaze_trajectory": ("gaze_trajectory_b64", "image/png"),
}


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self.client_ws: WebSocket | None = None
        self.state: dict[str, dict[str, Any]] = copy.deepcopy(_NO_CLIENT_STATE)
        self.last_capture: dict[str, str] | None = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        for message in self.state.values():
            await websocket.send_json(message)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

    async def send_to_client(self, message: dict[str, Any]) -> bool:
        if self.client_ws is None:
            return False
        await self.client_ws.send_json(message)
        return True

    async def relay_from_client(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        if event == "neon_capture_processed":
            # Images are served by /viz/live-frame; browsers only get the notification.
            images = message.pop("images", None)
            if isinstance(images, dict):
                self.last_capture = images
        if event in self.state:
            self.state[event] = message
        await self.broadcast(message)

    async def client_gone(self) -> None:
        self.client_ws = None
        self.state = copy.deepcopy(_NO_CLIENT_STATE)
        for message in self.state.values():
            await self.broadcast(message)


manager = ConnectionManager()
router = APIRouter()


@router.websocket("/ws")
async def dashboard_socket(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            logger.debug(f"Received from dashboard: {await websocket.receive_text()}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/neon")
async def neon_client_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    previous, manager.client_ws = manager.client_ws, websocket
    if previous is not None:
        logger.info("A new Neon client connected; closing the previous one")
        try:
            await previous.close()
        except Exception:
            pass
    try:
        while True:
            try:
                message = json.loads(await websocket.receive_text())
            except ValueError:
                continue
            if isinstance(message, dict):
                await manager.relay_from_client(message)
    except WebSocketDisconnect:
        pass
    finally:
        if manager.client_ws is websocket:
            logger.info("Neon client disconnected")
            await manager.client_gone()


@router.get("/viz/live-frame")
async def get_live_frame(type: str = "scene") -> Response:
    """Serve a visualization image from the most recent live capture."""
    capture = manager.last_capture
    if not capture:
        raise HTTPException(status_code=404, detail="No capture yet")
    b64_key, media_type = LIVE_FRAME_TYPES.get(type, LIVE_FRAME_TYPES["scene"])
    b64_data = capture.get(b64_key)
    if not b64_data:
        raise HTTPException(status_code=404, detail=f"No '{type}' visualization available yet")
    return Response(content=base64.b64decode(b64_data), media_type=media_type)


@router.post("/debug/fake-blink")
async def fake_blink() -> dict[str, str]:
    """Ask the connected Neon client to act as if it saw a double blink (starts or stops a capture)."""
    if not await manager.send_to_client({"command": "fake_blink"}):
        raise HTTPException(status_code=409, detail="No Neon client connected")
    return {"status": "sent"}
