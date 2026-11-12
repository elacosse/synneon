"""Shared broadcast helper — lets routes.py emit WebSocket events without importing app.py."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

_broadcast: Callable[[dict], Awaitable[None]] | None = None


def set_broadcast(fn: Callable[[dict], Awaitable[None]]) -> None:
    global _broadcast
    _broadcast = fn


async def broadcast(message: dict) -> None:
    if _broadcast is not None:
        await _broadcast(message)
