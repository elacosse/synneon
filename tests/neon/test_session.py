"""Gaze stream parsing: guards the firmware packet sizes the Neon library must handle."""

import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pupil_labs.realtime_api.streaming.base import RTSPData, RTSPRawStreamer
from pupil_labs.realtime_api.streaming.gaze import RTSPGazeStreamer

from synneon.neon import session
from synneon.schemas.neons import GazeSample, NeonConfig

TIMESTAMP = 1_700_000_000.5


def _packet(n_floats_after_worn: int, *, worn: bool = True) -> RTSPData:
    """Build a gaze packet: x, y, worn byte, then n floats numbered 1.0, 2.0, ..."""
    extra = [float(i) for i in range(1, n_floats_after_worn + 1)]
    raw = struct.pack(f"!ffB{n_floats_after_worn}f", 100.0, 200.0, 255 if worn else 0, *extra)
    return RTSPData(raw=raw, timestamp_unix_seconds=TIMESTAMP)


async def _parse(*packets: RTSPData) -> list:
    """Run packets through the library's gaze parser without a network connection."""

    async def raw_receive(self):
        for packet in packets:
            yield packet

    with patch.object(RTSPRawStreamer, "receive", raw_receive):
        streamer = RTSPGazeStreamer.__new__(RTSPGazeStreamer)
        return [item async for item in streamer.receive()]


class TestGazePacketSizes:
    async def test_105_byte_packet_is_parsed(self):
        """Newer Neon firmware sends 105 bytes; older library versions crashed on it."""
        packet = _packet(24)
        assert len(packet.raw) == 105

        (gaze,) = await _parse(packet)

        assert (gaze.x, gaze.y, gaze.worn) == (100.0, 200.0, True)
        assert gaze.timestamp_unix_seconds == TIMESTAMP
        assert gaze.pupil_diameter_left == 1.0
        assert gaze.pupil_diameter_right == 8.0
        assert (gaze.mono_left_x, gaze.mono_right_y) == (21.0, 24.0)

    async def test_89_byte_packet_is_still_parsed(self):
        packet = _packet(20, worn=False)
        assert len(packet.raw) == 89

        (gaze,) = await _parse(packet)

        assert gaze.worn is False
        assert gaze.eyelid_aperture_right == 20.0

    async def test_unknown_size_raises(self):
        """An unknown packet size must fail loudly, not look like an empty gaze stream."""
        with pytest.raises(KeyError):
            await _parse(RTSPData(raw=b"\x00" * 13, timestamp_unix_seconds=TIMESTAMP))


class TestStreamGaze:
    async def test_yields_gaze_samples_from_library_stream(self):
        (parsed,) = await _parse(_packet(24))

        async def fake_receive(url):
            assert url == "rtsp://neon/gaze"
            yield parsed

        async def fake_url(device, sensor):
            return "rtsp://neon/gaze"

        with (
            patch.object(session, "receive_gaze_data", fake_receive),
            patch.object(session, "_get_rtsp_url", fake_url),
        ):
            samples = [s async for s in session.stream_gaze(MagicMock())]

        assert samples == [GazeSample(x=100.0, y=200.0, timestamp=TIMESTAMP, is_worn=True)]


async def test_connect_closes_the_device_when_the_phone_is_unreachable():
    """Device() opens an aiohttp session; a failed status check must close it so retries do not leak."""
    device = MagicMock()
    device.get_status = AsyncMock(side_effect=OSError("no route to host"))
    device.close = AsyncMock()
    with patch.object(session, "Device", return_value=device), pytest.raises(OSError):
        await session.connect(NeonConfig(client_addr="192.0.2.1:8080"))

    device.close.assert_awaited_once()
