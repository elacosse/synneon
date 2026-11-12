import asyncio
import logging

import cv2
from pupil_labs.realtime_api import Device

from synneon.neon.session import stream_gaze, stream_video
from synneon.schemas.neons import GazeSample

logger = logging.getLogger(__name__)


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


class Visualizer:
    def __init__(self, device: Device):
        self.device = device
        self.latest_gaze: GazeSample | None = None
        self._running = False

    async def _update_gaze(self):
        """Task to continuously update the latest gaze sample."""
        try:
            async for gaze in stream_gaze(self.device):
                self.latest_gaze = gaze
                if not self._running:
                    break
        except Exception as e:
            logger.error(f"Gaze update task failed: {e}")

    async def run(self):
        """
        Starts the video stream and displays it with gaze overlay.
        """
        self._running = True

        # Start gaze update task in background
        gaze_task = asyncio.create_task(self._update_gaze())

        logger.info("Starting visualizer window...")
        window_name = "Neon Live Feed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            async for frame in stream_video(self.device):
                if not self._running:
                    break

                # Convert av.VideoFrame to numpy array (BGR for OpenCV)
                img = frame.av_frame.to_ndarray(format="bgr24")

                # Get latest gaze
                gaze = self.latest_gaze

                # Lightweight on-frame debugging to confirm overlay is happening
                h, w = img.shape[:2]
                cv2.putText(
                    img,
                    f"frame: {w}x{h}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if gaze is None:
                    cv2.putText(
                        img,
                        "gaze: None",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    # Note: gaze.x/y are expected to be normalized [0,1], but in practice may drift slightly outside.
                    gx, gy = gaze.x, gaze.y
                    worn = gaze.is_worn

                    cv2.putText(
                        img,
                        f"gaze: x={gx:.3f} y={gy:.3f} worn={worn}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if worn:
                        # Map normalized gaze to pixel coordinates.
                        px = gx
                        py = gy

                        x = _clamp_int(px, 0, w - 1)
                        y = _clamp_int(py, 0, h - 1)

                        # Draw a high-visibility marker (ring + crosshair + small filled center)
                        # This avoids "invisible" cases where a filled circle blends into similarly-colored content.
                        cv2.circle(img, (x, y), 24, (0, 0, 255), 3)
                        cv2.line(img, (x - 30, y), (x + 30, y), (0, 0, 255), 2)
                        cv2.line(img, (x, y - 30), (x, y + 30), (0, 0, 255), 2)
                        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

                        # Also display the computed pixel coords to catch mapping issues quickly
                        cv2.putText(
                            img,
                            f"px={x} py={y}",
                            (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.imshow(window_name, img)

                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit signal received.")
                    self._running = False
                    break

        except Exception as e:
            logger.error(f"Visualizer run loop failed: {e}")
        finally:
            self._running = False
            gaze_task.cancel()
            try:
                await gaze_task
            except asyncio.CancelledError:
                pass
            cv2.destroyAllWindows()
            logger.info("Visualizer stopped.")
