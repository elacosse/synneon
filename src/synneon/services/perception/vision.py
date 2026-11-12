import cv2
import numpy as np


def compute_gaze_saliency_map(
    image: np.ndarray,
    gaze_points: list[tuple[float, ...]],
    gaussian_sigma: float = 60.0,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.6,
    max_peaks: int = 5,
    min_peak_threshold_pct: float = 0.2,
    marker_shape: str = "X",
    marker_radius: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Computes a gaze-based saliency map for a given BGR image using eye-gaze trajectories
    and returns the raw saliency map (grayscale), an overlaid heatmap, and an optional peaks overlay.

    Args:
        image: The input BGR image frame (numpy array).
        gaze_points: List of (x, y) pixel coordinates representing gaze locations.
        gaussian_sigma: Standard deviation for the Gaussian blur applied to gaze fixations.
                        Larger values create broader foveal "hotspots".
        colormap: OpenCV colormap to use for the heatmap visualization.
        alpha: Maximum opacity for the heatmap overlay.
        max_peaks: Number of local maxima to identify and draw on the overlay.
        min_peak_threshold_pct: Ignore peaks that are less than this percentage of the global maximum intensity.

    Returns:
        A tuple of (saliency_map_gray, saliency_overlay_bgr, peaks_overlay_bgr):
            - saliency_map_gray: The grayscale 8-bit saliency map [0-255].
            - saliency_overlay_bgr: The original image with the gaze heatmap overlaid.
            - peaks_overlay_bgr: The original image with red X marks at the local maxima.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image.")

    h, w = image.shape[:2]

    # 1. Create a sparse map of gaze fixations
    sparse_map = np.zeros((h, w), dtype=np.float32)

    if gaze_points:
        for pt in gaze_points:
            x, y = pt[-2], pt[-1]
            px, py = int(round(x)), int(round(y))
            # Only accumulate points that fall within the image bounds
            if 0 <= px < w and 0 <= py < h:
                sparse_map[py, px] += 1.0

        # 2. Apply Gaussian Blur to create a continuous heatmap spreading out from the fixations
        # A kernel size of (0, 0) allows OpenCV to automatically size the kernel based on sigma
        saliency_map = cv2.GaussianBlur(sparse_map, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)

        # 3. Normalize the continuous map to 0-255
        max_val = np.max(saliency_map)
        if max_val > 0:
            saliency_map = cv2.normalize(saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    else:
        # If there are no gaze points, the saliency map remains completely blank
        saliency_map = sparse_map

    saliency_map_gray = np.uint8(saliency_map)

    # 4. Apply colormap to the normalized grayscale map
    heatmap = cv2.applyColorMap(saliency_map_gray, colormap)

    # 5. Overlay the heatmap onto the original image
    # We create a dynamic alpha mask based on the saliency intensity.
    # This ensures that areas with zero gaze points remain perfectly visible as the un-tinted original image.
    heatmap_weight = (saliency_map_gray.astype(np.float32) / 255.0) * alpha
    heatmap_weight = np.expand_dims(heatmap_weight, axis=-1)

    # Blend: output = heatmap * weight + original * (1 - weight)
    saliency_overlay_bgr = (heatmap * heatmap_weight + image * (1.0 - heatmap_weight)).astype(np.uint8)

    peaks_overlay_bgr = None

    if max_peaks > 0 and gaze_points:
        peaks_overlay_bgr = image.copy()
        # Find local maxima via dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated = cv2.dilate(saliency_map_gray, kernel)
        local_max = (saliency_map_gray == dilated) & (saliency_map_gray > 20)
        local_max_uint8 = np.uint8(local_max) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(local_max_uint8, connectivity=8)

        peaks = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            px, py = int(round(cx)), int(round(cy))
            intensity = int(saliency_map_gray[py, px])
            peaks.append((px, py, intensity))

        peaks.sort(key=lambda p: p[2], reverse=True)

        if peaks:
            max_intensity = peaks[0][2]
            threshold_val = max_intensity * min_peak_threshold_pct
            peaks = [p for p in peaks if p[2] >= threshold_val]

        peaks = peaks[:max_peaks]

        # Draw markers
        for px, py, _ in peaks:
            thickness = 3
            color_bg = (0, 0, 0)
            color_fg = (0, 0, 255)  # Red

            if marker_shape.lower() == "circle":
                cv2.circle(peaks_overlay_bgr, (px, py), marker_radius, color_bg, thickness + 2, cv2.LINE_AA)
                cv2.circle(peaks_overlay_bgr, (px, py), marker_radius, color_fg, thickness, cv2.LINE_AA)
            else:
                cv2.line(
                    peaks_overlay_bgr,
                    (px - marker_radius, py - marker_radius),
                    (px + marker_radius, py + marker_radius),
                    color_bg,
                    thickness + 2,
                )
                cv2.line(
                    peaks_overlay_bgr,
                    (px + marker_radius, py - marker_radius),
                    (px - marker_radius, py + marker_radius),
                    color_bg,
                    thickness + 2,
                )

                cv2.line(
                    peaks_overlay_bgr,
                    (px - marker_radius, py - marker_radius),
                    (px + marker_radius, py + marker_radius),
                    color_fg,
                    thickness,
                )
                cv2.line(
                    peaks_overlay_bgr,
                    (px + marker_radius, py - marker_radius),
                    (px - marker_radius, py + marker_radius),
                    color_fg,
                    thickness,
                )

    return saliency_map_gray, saliency_overlay_bgr, peaks_overlay_bgr


def draw_gaze_trajectory(image: np.ndarray, gaze_points: list[tuple[float, ...]]) -> np.ndarray:
    """
    Draws a continuous gaze trajectory on the image with a time-based color gradient.
    """
    overlay = image.copy()
    n_points = len(gaze_points)
    if n_points < 2:
        return overlay

    segments = []
    for i in range(n_points - 1):
        p1 = (int(round(gaze_points[i][-2])), int(round(gaze_points[i][-1])))
        p2 = (int(round(gaze_points[i + 1][-2])), int(round(gaze_points[i + 1][-1])))
        segments.append((p1, p2))

    # Draw thick black outlines first so they don't overlap the inner colored lines at joints
    for p1, p2 in segments:
        cv2.line(overlay, p1, p2, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)

    # Draw colored inner lines
    for i, (p1, p2) in enumerate(segments):
        # Map time fraction to a color using JET colormap (0=Start/Blue, 255=End/Red)
        t = i / (n_points - 1)
        val_gray = np.array([[[int(t * 255)]]], dtype=np.uint8)
        color_bgr = cv2.applyColorMap(val_gray, cv2.COLORMAP_JET)[0, 0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        cv2.line(overlay, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    return overlay


def detect_fixations_idt(
    gaze_points: list[tuple[float, ...]], min_duration_ms: float = 100.0, max_dispersion_pixels: float = 30.0
) -> list[dict]:
    """
    Identify fixations from gaze points using the I-DT algorithm.
    Expects gaze_points as (timestamp_s, x, y) or (x, y).
    """
    if not gaze_points:
        return []

    points = []
    for i, pt in enumerate(gaze_points):
        if len(pt) >= 3:
            ts_ms = pt[0] * 1000.0
            x, y = pt[-2], pt[-1]
        else:
            ts_ms = i * (1000.0 / 200.0)  # Assume ~200Hz if no timestamp
            x, y = pt[-2], pt[-1]
        points.append((ts_ms, x, y))

    fixations = []
    i = 0
    n = len(points)

    while i < n:
        window = [points[i]]
        j = i + 1

        while j < n:
            window.append(points[j])
            xs = [p[1] for p in window]
            ys = [p[2] for p in window]
            dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))

            if dispersion > max_dispersion_pixels:
                window.pop()
                break
            j += 1

        if len(window) > 1:
            duration = window[-1][0] - window[0][0]
            if duration >= min_duration_ms:
                cx = sum(p[1] for p in window) / len(window)
                cy = sum(p[2] for p in window) / len(window)
                fixations.append(
                    {
                        "centroid_x": cx,
                        "centroid_y": cy,
                        "start_ts": window[0][0],
                        "end_ts": window[-1][0],
                        "duration_ms": duration,
                    }
                )
                i = j
                continue
        i += 1

    return fixations


def draw_fixations(image: np.ndarray, fixations: list[dict]) -> np.ndarray:
    """
    Draws fixations on the image with lines for saccades and circles for fixations (radius = duration).
    """
    overlay = image.copy()
    if not fixations:
        return overlay

    n_fixations = len(fixations)
    for i in range(n_fixations - 1):
        p1 = (int(round(fixations[i]["centroid_x"])), int(round(fixations[i]["centroid_y"])))
        p2 = (int(round(fixations[i + 1]["centroid_x"])), int(round(fixations[i + 1]["centroid_y"])))

        t = i / max(1, n_fixations - 1)
        val_gray = np.array([[[int(t * 255)]]], dtype=np.uint8)
        color_bgr = cv2.applyColorMap(val_gray, cv2.COLORMAP_JET)[0, 0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        # Draw black border for saccade
        cv2.line(overlay, p1, p2, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
        # Draw colored inner line
        cv2.line(overlay, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    for i, fix in enumerate(fixations):
        cx = int(round(fix["centroid_x"]))
        cy = int(round(fix["centroid_y"]))
        radius = int(max(5, min(50, fix["duration_ms"] / 10.0)))

        t = i / max(1, n_fixations - 1)
        val_gray = np.array([[[int(t * 255)]]], dtype=np.uint8)
        color_bgr = cv2.applyColorMap(val_gray, cv2.COLORMAP_JET)[0, 0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

        # Draw solid inner circle
        cv2.circle(overlay, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        # Draw black border
        cv2.circle(overlay, (cx, cy), radius, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return overlay


def process_video_frames_for_saliency(
    frames: list[np.ndarray],
    gaze_points: list[tuple[float, ...]],
    mode: str = "middle",
    gaussian_sigma: float = 60.0,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.6,
    max_peaks: int = 5,
    min_peak_threshold_pct: float = 0.2,
    marker_shape: str = "X",
    marker_radius: int = 15,
    min_duration_ms: float = 100.0,
    max_dispersion_pixels: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
    """
    Selects a specific frame from a list of video frames and computes a gaze saliency map, trajectory, and fixations on it.

    Args:
        frames: List of BGR image frames (numpy arrays).
        gaze_points: List of (x, y) coordinates representing the gaze trajectory during the sequence.
        mode: Which frame to select ("first", "middle", or "last").
        gaussian_sigma: Parameterization for the gaze foveal region spreading (smoothing).
        colormap: Parameterization for the heatmap overlay.
        alpha: Blending weight.
        max_peaks: Number of local maxima to identify and draw.

    Returns:
        (saliency_map_gray, saliency_overlay_bgr, peaks_overlay_bgr, trajectory_overlay_bgr) or None if frames list is empty.
    """
    if not frames:
        return None

    if mode == "first":
        selected_frame = frames[0]
    elif mode == "last":
        selected_frame = frames[-1]
    elif mode == "middle":
        selected_frame = frames[len(frames) // 2]
    else:
        # default to middle if unknown mode
        selected_frame = frames[len(frames) // 2]

    saliency_map_gray, saliency_overlay_bgr, peaks_overlay_bgr = compute_gaze_saliency_map(
        selected_frame,
        gaze_points=gaze_points,
        gaussian_sigma=gaussian_sigma,
        colormap=colormap,
        alpha=alpha,
        max_peaks=max_peaks,
        min_peak_threshold_pct=min_peak_threshold_pct,
        marker_shape=marker_shape,
        marker_radius=marker_radius,
    )

    trajectory_overlay_bgr = draw_gaze_trajectory(selected_frame, gaze_points)

    fixations = detect_fixations_idt(gaze_points, min_duration_ms, max_dispersion_pixels)
    fixations_overlay_bgr = draw_fixations(selected_frame, fixations)

    return saliency_map_gray, saliency_overlay_bgr, peaks_overlay_bgr, trajectory_overlay_bgr, fixations_overlay_bgr
