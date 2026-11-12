import logging
from collections.abc import AsyncGenerator

from pupil_labs.realtime_api import (
    AudioFrame,
    Device,
    VideoFrame,
    receive_audio_frames,
    receive_eye_events_data,
    receive_gaze_data,
    receive_video_frames,
)
from pupil_labs.realtime_api.streaming import BlinkEventData, FixationEventData

from synneon.schemas.neons import GazeSample, NeonConfig

logger = logging.getLogger(__name__)


async def connect(config: NeonConfig) -> Device:
    """
    Establish connection to the Neon device.

    Returns a Device instance.
    """
    parts = config.client_addr.split(":")
    ip = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 8080

    logger.info(f"Connecting to Neon at {ip}:{port}...")

    device = Device(address=ip, port=port)

    # Verify connectivity by fetching status
    try:
        status = await device.get_status()
        logger.info(f"Connected to {status.phone.device_name} (Battery: {status.phone.battery_level}%)")
    except Exception as e:
        logger.error(f"Could not fetch status from Neon device at {ip}:{port}: {e}")
        # Device() opens an aiohttp session on construction; close it so retries do not leak sessions.
        await device.close()
        raise

    return device


async def _get_rtsp_url(device: Device, sensor_name: str) -> str:
    """
    Resolve the RTSP URL for a specific sensor.
    """
    # 1. Query device status to find the specific sensor's RTSP info
    try:
        status = await device.get_status()
        # Look for the sensor with DIRECT connection (RTSP)
        target_sensor = next(
            (s for s in status.sensors if s.sensor == sensor_name and s.conn_type == "DIRECT"),
            None,
        )
        if target_sensor:
            url = f"rtsp://{target_sensor.ip}:{target_sensor.port}?{target_sensor.params}"
            logger.info(f"Discovered RTSP URL for {sensor_name}: {url}")
            return url

        logger.warning(f"Sensor '{sensor_name}' not found in device status or no DIRECT connection available.")
        available = [f"{s.sensor} ({s.conn_type})" for s in status.sensors]
        logger.info(f"Available sensors: {available}")

    except Exception as e:
        logger.warning(f"Failed to fetch status for RTSP discovery of {sensor_name}: {e}")

    # 2. Check if device has a global rtsp_out_url (some versions/devices)
    rtsp_url = getattr(device, "rtsp_out_url", None)
    if rtsp_url:
        logger.warning(f"Using global rtsp_out_url for {sensor_name}: {rtsp_url}")
        return rtsp_url

    # 3. Fallback
    fallback = f"rtsp://{device.address}:8554/"
    logger.warning(f"Could not discover RTSP URL for {sensor_name}, using fallback: {fallback}")
    return fallback


async def stream_gaze(device: Device) -> AsyncGenerator[GazeSample, None]:
    """
    Generator yielding standardized GazeSample tuples.

    Args:
        device: Connected Neon Device instance.

    Yields:
        GazeSample: Tuple containing (x, y, timestamp, is_worn).
    """
    logger.info("Starting gaze stream...")

    rtsp_url = await _get_rtsp_url(device, "gaze")

    try:
        async for gaze in receive_gaze_data(rtsp_url):
            yield GazeSample(
                x=gaze.x,
                y=gaze.y,
                timestamp=gaze.timestamp_unix_seconds,
                is_worn=gaze.worn,
            )
    except Exception as e:
        logger.error(f"Gaze stream interrupted: {e}")
        raise
    finally:
        logger.info("Gaze stream stopped.")


async def stream_video(device: Device) -> AsyncGenerator[VideoFrame, None]:
    """
    Generator yielding video frames.

    Args:
        device: Connected Neon Device instance.

    Yields:
        VideoFrame: Frame data containing image and timestamp.
    """
    logger.info("Starting video stream...")

    # The scene camera is typically named 'world' in sensors
    rtsp_url = await _get_rtsp_url(device, "world")

    try:
        async for frame in receive_video_frames(rtsp_url):
            yield frame
    except Exception as e:
        logger.error(f"Video stream interrupted: {e}")
        raise
    finally:
        logger.info("Video stream stopped.")


async def stream_eye_events(
    device: Device,
) -> AsyncGenerator[BlinkEventData | FixationEventData, None]:
    """
    Generator yielding eye events (blinks, fixations).

    Args:
        device: Connected Neon Device instance.

    Yields:
        BlinkEventData | FixationEventData: Eye event data.
    """
    logger.info("Starting eye events stream...")

    rtsp_url = await _get_rtsp_url(device, "eye_events")

    try:
        async for event in receive_eye_events_data(rtsp_url):
            yield event
    except Exception as e:
        logger.error(f"Eye events stream interrupted: {e}")
        raise
    finally:
        logger.info("Eye events stream stopped.")


async def stream_audio(device: Device) -> AsyncGenerator[AudioFrame, None]:
    """
    Generator yielding audio frames.

    Args:
        device: Connected Neon Device instance.

    Yields:
        AudioFrame: Frame data containing audio and timestamp.
    """
    logger.info("Starting audio stream...")

    rtsp_url = await _get_rtsp_url(device, "audio")

    try:
        async for frame in receive_audio_frames(rtsp_url):
            yield frame
    except Exception as e:
        logger.error(f"Audio stream interrupted: {e}")
        raise
    finally:
        logger.info("Audio stream stopped.")
