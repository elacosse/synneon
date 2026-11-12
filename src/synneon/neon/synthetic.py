import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator

from synneon.schemas.neons import GazeSample

logger = logging.getLogger(__name__)


async def synthetic_gaze_stream() -> AsyncGenerator[GazeSample, None]:
    """
    Generates synthetic gaze data for dry-run testing.
    Simulates random noise and occasional swipe gestures.
    """
    logger.info("Starting synthetic gaze stream...")

    # Center of a normalized field (0,0 is top-left usually)
    x, y = 0.5, 0.5
    start_time = time.time()

    while True:
        current_time = time.time()

        # Decide if we are doing a gesture
        # Every 4 seconds cycle
        cycle = (current_time - start_time) % 4.0

        if 1.0 < cycle < 1.15:
            # Quick swipe Right (+x)
            # Velocity ~ 0.05 / 0.01 = 5.0 units/s (Threshold default is 2.0)
            x += 0.05
        elif 2.5 < cycle < 2.65:
            # Quick swipe Left (-x)
            x -= 0.05
        else:
            # Return to center slowly / noise
            target_x, _target_y = 0.5, 0.5
            # drift back
            x += (target_x - x) * 0.05
            # noise
            x += random.uniform(-0.001, 0.001)
            y += random.uniform(-0.001, 0.001)

        # Clamp
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        yield GazeSample(x=x, y=y, timestamp=current_time, is_worn=True)

        # approximate 100Hz
        await asyncio.sleep(0.01)
