import math

import pytest

from synneon.input.detector import DetectorState, _moving_average_smooth, detect_gestures
from synneon.schemas.gestures import CompassSettings, GestureType
from synneon.schemas.neons import GazeSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    min_x: float = 0.0,
    max_x: float = 1.0,
    min_y: float = 0.0,
    max_y: float = 1.0,
    threshold: float = 0.8,
    dwell_time: float = 0.5,
    cooldown: float = 0.0,
    smoothing_factor: float = 0.0,
) -> CompassSettings:
    return CompassSettings(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        threshold=threshold,
        dwell_time=dwell_time,
        cooldown=cooldown,
        smoothing_factor=smoothing_factor,
    )


def _worn(x: float, y: float, t: float) -> GazeSample:
    return GazeSample(x=x, y=y, timestamp=t, is_worn=True)


def _not_worn(x: float, y: float, t: float) -> GazeSample:
    return GazeSample(x=x, y=y, timestamp=t, is_worn=False)


def _right_dwell(start_t: float = 100.0, n: int = 6) -> list[GazeSample]:
    """Samples dwelling firmly in the RIGHT zone (x=0.95, [0,1] range, T=0.8 → right zone x>=0.8)."""
    return [_worn(0.95, 0.5, start_t + i * 0.1) for i in range(n)]


def _left_dwell(start_t: float = 100.0, n: int = 6) -> list[GazeSample]:
    return [_worn(0.05, 0.5, start_t + i * 0.1) for i in range(n)]


def _up_dwell(start_t: float = 100.0, n: int = 6) -> list[GazeSample]:
    return [_worn(0.5, 0.05, start_t + i * 0.1) for i in range(n)]


def _down_dwell(start_t: float = 100.0, n: int = 6) -> list[GazeSample]:
    return [_worn(0.5, 0.95, start_t + i * 0.1) for i in range(n)]


def _center_samples(start_t: float = 100.0, n: int = 6) -> list[GazeSample]:
    return [_worn(0.5, 0.5, start_t + i * 0.1) for i in range(n)]


# ---------------------------------------------------------------------------
# _moving_average_smooth
# ---------------------------------------------------------------------------


class TestMovingAverageSmooth:
    def test_length_preserved(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [0.0] * 5
        ts = list(range(5))
        result = _moving_average_smooth(xs, ys, ts, smoothing_factor=0.5)
        assert len(result) == 5

    def test_sf_zero_no_smoothing(self):
        """k=1 → each point is unchanged."""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [5.0, 6.0, 7.0, 8.0]
        ts = [0.0, 1.0, 2.0, 3.0]
        result = _moving_average_smooth(xs, ys, ts, smoothing_factor=0.0)
        for i, (rx, ry, rt) in enumerate(result):
            assert rx == pytest.approx(xs[i])
            assert ry == pytest.approx(ys[i])
            assert rt == ts[i]

    def test_sf_025_k3_correct_averages(self):
        """k=3 → boundary-clamped 3-point average."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [0.0] * 5
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = _moving_average_smooth(xs, ys, ts, smoothing_factor=0.25)
        # i=0: window [0,1], avg = (1+2)/2 = 1.5
        assert result[0][0] == pytest.approx(1.5)
        # i=2: window [1,3], avg = (2+3+4)/3 = 3.0
        assert result[2][0] == pytest.approx(3.0)
        # i=4: window [3,4], avg = (4+5)/2 = 4.5
        assert result[4][0] == pytest.approx(4.5)

    def test_sf_negative_clamped_to_k1(self):
        xs = [1.0, 2.0, 3.0]
        ys = [0.0] * 3
        ts = [0.0, 1.0, 2.0]
        result = _moving_average_smooth(xs, ys, ts, smoothing_factor=-1.0)
        for i, (rx, _, _) in enumerate(result):
            assert rx == pytest.approx(xs[i])

    def test_sf_above_one_clamped_to_k9(self):
        """Should not raise; just clamps to k=9."""
        xs = list(range(20))
        ys = [0.0] * 20
        ts = list(range(20))
        result = _moving_average_smooth(xs, ys, ts, smoothing_factor=2.0)
        assert len(result) == 20

    def test_timestamps_preserved(self):
        xs = [1.0, 2.0, 3.0]
        ys = [0.0] * 3
        ts = [10.0, 20.0, 30.0]
        result = _moving_average_smooth(xs, ys, ts, smoothing_factor=0.0)
        assert [r[2] for r in result] == ts

    def test_single_sample(self):
        result = _moving_average_smooth([5.0], [3.0], [99.0], smoothing_factor=0.5)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(5.0)
        assert result[0][1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# detect_gestures
# ---------------------------------------------------------------------------


class TestDetectGestures:
    def test_empty_trajectory_returns_none(self):
        assert detect_gestures([], DetectorState(), _config()) is None

    def test_fewer_than_four_worn_samples_returns_none(self):
        samples = [_worn(0.9, 0.5, t) for t in [100.0, 100.1, 100.2]]
        assert detect_gestures(samples, DetectorState(), _config()) is None

    def test_not_worn_samples_excluded(self):
        """Only not-worn samples → filtered to 0, returns None."""
        samples = [_not_worn(0.95, 0.5, t) for t in [100.0, 100.1, 100.2, 100.3, 100.4, 100.5]]
        assert detect_gestures(samples, DetectorState(), _config()) is None

    def test_mixed_worn_filters_correctly(self):
        """With 3 worn and 3 not-worn, worn count < 4 → None."""
        samples = [
            _worn(0.95, 0.5, 100.0),
            _not_worn(0.95, 0.5, 100.1),
            _worn(0.95, 0.5, 100.2),
            _not_worn(0.95, 0.5, 100.3),
            _worn(0.95, 0.5, 100.4),
            _not_worn(0.95, 0.5, 100.5),
        ]
        assert detect_gestures(samples, DetectorState(), _config()) is None

    def test_detects_right(self):
        event = detect_gestures(_right_dwell(), DetectorState(), _config())
        assert event is not None
        assert event.type == GestureType.RIGHT

    def test_detects_left(self):
        event = detect_gestures(_left_dwell(), DetectorState(), _config())
        assert event is not None
        assert event.type == GestureType.LEFT

    def test_detects_up(self):
        event = detect_gestures(_up_dwell(), DetectorState(), _config())
        assert event is not None
        assert event.type == GestureType.UP

    def test_detects_down(self):
        event = detect_gestures(_down_dwell(), DetectorState(), _config())
        assert event is not None
        assert event.type == GestureType.DOWN

    def test_center_gaze_no_gesture(self):
        assert detect_gestures(_center_samples(), DetectorState(), _config()) is None

    def test_event_timestamp_matches_trajectory_end(self):
        samples = _right_dwell(start_t=200.0)
        event = detect_gestures(samples, DetectorState(), _config())
        assert event is not None
        assert event.timestamp == pytest.approx(200.0 + 5 * 0.1)  # last sample

    def test_event_duration_positive(self):
        event = detect_gestures(_right_dwell(), DetectorState(), _config())
        assert event is not None
        assert event.duration >= _config().dwell_time

    def test_event_magnitude_positive(self):
        """RIGHT gesture at x=0.95 vs center=0.5 → magnitude > 0."""
        event = detect_gestures(_right_dwell(), DetectorState(), _config())
        assert event is not None
        assert event.magnitude > 0.0

    def test_magnitude_formula(self):
        """Magnitude is Euclidean distance of endpoint from range center."""
        # endpoint at (0.95, 0.5), center=(0.5, 0.5)
        samples = _right_dwell()
        event = detect_gestures(samples, DetectorState(), _config())
        assert event is not None
        expected = math.sqrt((0.95 - 0.5) ** 2 + (0.5 - 0.5) ** 2)
        assert event.magnitude == pytest.approx(expected, abs=0.01)

    def test_state_updated_after_detection(self):
        state = DetectorState()
        samples = _right_dwell(start_t=100.0)
        detect_gestures(samples, state, _config())
        assert state.last_gesture_time > 0.0

    def test_cooldown_blocks_repeat_detection(self):
        state = DetectorState()
        cfg = _config(cooldown=1.0)
        samples = _right_dwell(start_t=100.0)
        first = detect_gestures(samples, state, cfg)
        assert first is not None
        # Same trajectory — end_t - last_gesture_time = 0 < cooldown
        second = detect_gestures(samples, state, cfg)
        assert second is None

    def test_cooldown_allows_detection_after_interval(self):
        state = DetectorState()
        cfg = _config(cooldown=1.0)
        first_samples = _right_dwell(start_t=100.0)
        detect_gestures(first_samples, state, cfg)
        # New samples well after cooldown
        second_samples = _right_dwell(start_t=102.0)
        event = detect_gestures(second_samples, state, cfg)
        assert event is not None

    def test_dwell_time_not_met_returns_none(self):
        """dwell_time=2.0 but trajectory only spans 0.5s → None."""
        cfg = _config(dwell_time=2.0)
        samples = _right_dwell()  # spans 0.5s
        assert detect_gestures(samples, DetectorState(), cfg) is None

    def test_none_calibration_bounds_fallback_to_unit_range(self):
        cfg = CompassSettings(
            min_x=None,
            max_x=None,
            min_y=None,
            max_y=None,
            threshold=0.8,
            dwell_time=0.5,
            cooldown=0.0,
            smoothing_factor=0.0,
        )
        # x=0.95 is in right zone (>= 0.8) of [0,1] fallback range
        event = detect_gestures(_right_dwell(), DetectorState(), cfg)
        assert event is not None
        assert event.type == GestureType.RIGHT

    def test_tuple_trajectory_accepted(self):
        event = detect_gestures(tuple(_right_dwell()), DetectorState(), _config())
        assert event is not None
