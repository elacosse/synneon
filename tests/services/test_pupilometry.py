import math
from collections import deque

import pytest

from synneon.services.physiology.pupilometry import (
    PupilometryPlotConfig,
    PupilSample,
    _is_finite_number,
    _snapshot_xy_for_plot,
    _to_mm_or_nan,
    prune_sliding_window_in_place,
)


def _sample(ts: float, left: float, right: float, worn: bool = True) -> PupilSample:
    return PupilSample(timestamp_unix_seconds=ts, left_mm=left, right_mm=right, worn=worn)


# ---------------------------------------------------------------------------
# _is_finite_number
# ---------------------------------------------------------------------------


class TestIsFiniteNumber:
    @pytest.mark.parametrize("v", [0.0, 1.5, -3.14, 100, "2.5"])
    def test_finite_values(self, v):
        assert _is_finite_number(v) is True

    @pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf"), "nan", "inf", None, "abc"])
    def test_non_finite_values(self, v):
        assert _is_finite_number(v) is False


# ---------------------------------------------------------------------------
# _to_mm_or_nan
# ---------------------------------------------------------------------------


class TestToMmOrNan:
    def test_finite_float_returned(self):
        assert _to_mm_or_nan(3.5) == pytest.approx(3.5)

    def test_integer_converted_to_float(self):
        result = _to_mm_or_nan(4)
        assert result == pytest.approx(4.0)

    def test_nan_input_returns_nan(self):
        assert math.isnan(_to_mm_or_nan(float("nan")))

    def test_inf_input_returns_nan(self):
        assert math.isnan(_to_mm_or_nan(float("inf")))

    def test_none_returns_nan(self):
        assert math.isnan(_to_mm_or_nan(None))

    def test_string_number_converted(self):
        assert _to_mm_or_nan("2.5") == pytest.approx(2.5)

    def test_invalid_string_returns_nan(self):
        assert math.isnan(_to_mm_or_nan("not_a_number"))


# ---------------------------------------------------------------------------
# PupilSample
# ---------------------------------------------------------------------------


class TestPupilSample:
    def test_construction(self):
        s = _sample(1000.0, 3.1, 3.2)
        assert s.timestamp_unix_seconds == 1000.0
        assert s.left_mm == pytest.approx(3.1)
        assert s.right_mm == pytest.approx(3.2)
        assert s.worn is True

    def test_not_worn(self):
        s = _sample(1.0, float("nan"), float("nan"), worn=False)
        assert s.worn is False
        assert math.isnan(s.left_mm)

    def test_frozen(self):
        s = _sample(1.0, 3.0, 3.0)
        with pytest.raises((AttributeError, TypeError)):
            s.left_mm = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PupilometryPlotConfig
# ---------------------------------------------------------------------------


class TestPupilometryPlotConfig:
    def test_defaults(self):
        cfg = PupilometryPlotConfig()
        assert cfg.window_s == 30.0
        assert cfg.refresh_hz == 20.0
        assert "Pupillometry" in cfg.title

    def test_custom(self):
        cfg = PupilometryPlotConfig(window_s=10.0, refresh_hz=5.0, title="My Plot")
        assert cfg.window_s == 10.0


# ---------------------------------------------------------------------------
# prune_sliding_window_in_place
# ---------------------------------------------------------------------------


class TestPruneSlidingWindow:
    def test_empty_deque_is_noop(self):
        d: deque[PupilSample] = deque()
        prune_sliding_window_in_place(d, window_s=10.0)
        assert len(d) == 0

    def test_window_zero_raises(self):
        d: deque[PupilSample] = deque([_sample(1.0, 3.0, 3.0)])
        with pytest.raises(ValueError, match="window_s must be > 0"):
            prune_sliding_window_in_place(d, window_s=0)

    def test_window_negative_raises(self):
        d: deque[PupilSample] = deque([_sample(1.0, 3.0, 3.0)])
        with pytest.raises(ValueError):
            prune_sliding_window_in_place(d, window_s=-5.0)

    def test_all_samples_within_window_unchanged(self):
        d = deque([_sample(90.0, 3.0, 3.0), _sample(95.0, 3.0, 3.0), _sample(100.0, 3.0, 3.0)])
        prune_sliding_window_in_place(d, window_s=30.0)
        assert len(d) == 3

    def test_removes_samples_older_than_cutoff(self):
        # newest=100.0, window=10.0, cutoff=90.0
        # sample at 89.9 < 90.0 → removed
        d = deque(
            [
                _sample(89.9, 3.0, 3.0),
                _sample(95.0, 3.0, 3.0),
                _sample(100.0, 3.0, 3.0),
            ]
        )
        prune_sliding_window_in_place(d, window_s=10.0)
        assert len(d) == 2
        assert d[0].timestamp_unix_seconds == pytest.approx(95.0)

    def test_sample_exactly_at_cutoff_is_kept(self):
        # cutoff = 100.0 - 10.0 = 90.0
        # sample at 90.0: 90.0 < 90.0 is False → NOT pruned
        d = deque(
            [
                _sample(90.0, 3.0, 3.0),
                _sample(100.0, 3.0, 3.0),
            ]
        )
        prune_sliding_window_in_place(d, window_s=10.0)
        assert len(d) == 2

    def test_removes_multiple_old_samples(self):
        d = deque([_sample(float(t), 3.0, 3.0) for t in range(100, 110)] + [_sample(200.0, 3.0, 3.0)])
        prune_sliding_window_in_place(d, window_s=10.0)
        # cutoff = 200.0 - 10.0 = 190.0; all timestamps 100-109 are < 190.0
        assert all(s.timestamp_unix_seconds >= 190.0 for s in d)

    def test_all_samples_older_than_window_all_removed(self):
        d = deque([_sample(1.0, 3.0, 3.0), _sample(2.0, 3.0, 3.0), _sample(100.0, 3.0, 3.0)])
        # newest=100.0, window=1.0, cutoff=99.0 → 1.0 and 2.0 pruned, 100.0 kept
        prune_sliding_window_in_place(d, window_s=1.0)
        assert len(d) == 1
        assert d[0].timestamp_unix_seconds == pytest.approx(100.0)

    def test_single_sample_unchanged(self):
        d = deque([_sample(50.0, 3.0, 3.0)])
        prune_sliding_window_in_place(d, window_s=10.0)
        assert len(d) == 1


# ---------------------------------------------------------------------------
# _snapshot_xy_for_plot
# ---------------------------------------------------------------------------


class TestSnapshotXYForPlot:
    def test_empty_returns_empty_lists(self):
        xs, lefts, rights = _snapshot_xy_for_plot([], window_s=30.0)
        assert xs == []
        assert lefts == []
        assert rights == []

    def test_single_sample_x_is_zero(self):
        samples = [_sample(100.0, 3.1, 3.2)]
        xs, lefts, rights = _snapshot_xy_for_plot(samples, window_s=30.0)
        assert len(xs) == 1
        assert xs[0] == pytest.approx(0.0)
        assert lefts[0] == pytest.approx(3.1)
        assert rights[0] == pytest.approx(3.2)

    def test_two_samples_relative_times(self):
        samples = [_sample(100.0, 3.0, 3.0), _sample(101.0, 4.0, 4.0)]
        xs, lefts, rights = _snapshot_xy_for_plot(samples, window_s=30.0)
        assert xs[0] == pytest.approx(-1.0)
        assert xs[1] == pytest.approx(0.0)

    def test_newest_sample_always_at_zero(self):
        samples = [_sample(float(t), 3.0, 3.0) for t in [50.0, 75.0, 100.0]]
        xs, _, _ = _snapshot_xy_for_plot(samples, window_s=30.0)
        assert xs[-1] == pytest.approx(0.0)

    def test_nan_values_preserved(self):
        nan = float("nan")
        samples = [_sample(100.0, nan, nan, worn=False)]
        _, lefts, rights = _snapshot_xy_for_plot(samples, window_s=30.0)
        assert math.isnan(lefts[0])
        assert math.isnan(rights[0])

    def test_list_lengths_match(self):
        samples = [_sample(float(t), 3.0, 3.0) for t in range(10)]
        xs, lefts, rights = _snapshot_xy_for_plot(samples, window_s=30.0)
        assert len(xs) == len(lefts) == len(rights) == 10

    def test_window_s_parameter_accepted(self):
        """window_s is accepted but currently unused (x-axis set by caller)."""
        samples = [_sample(100.0, 3.0, 3.0)]
        xs, _, _ = _snapshot_xy_for_plot(samples, window_s=5.0)
        assert xs == [pytest.approx(0.0)]

    def test_deque_input_accepted(self):
        d = deque([_sample(100.0, 3.0, 3.2), _sample(101.0, 3.1, 3.3)])
        xs, lefts, rights = _snapshot_xy_for_plot(d, window_s=30.0)
        assert len(xs) == 2
        assert lefts[1] == pytest.approx(3.1)
