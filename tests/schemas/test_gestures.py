import pytest
from pydantic import ValidationError

from synneon.schemas.gestures import (
    CompassSettings,
    GestureEvent,
    GesturesSettings,
    GestureType,
)


class TestCompassSettings:
    def test_defaults(self):
        s = CompassSettings()
        assert s.threshold == 0.8
        assert s.dwell_time == 0.5
        assert s.min_x == pytest.approx(-45.1537)
        assert s.max_x == pytest.approx(1490.5511)
        assert s.min_y == pytest.approx(-45.1537)
        assert s.max_y == pytest.approx(1129.8431)
        assert s.velocity_threshold == 2.0
        assert s.min_displacement == 0.15
        assert s.cooldown == 1.0
        assert s.smoothing_factor == 0.3
        assert s.sample_time_length == 1.0

    def test_calibration_bounds_can_be_none(self):
        s = CompassSettings(min_x=None, max_x=None, min_y=None, max_y=None)
        assert s.min_x is None
        assert s.max_x is None

    def test_custom_values(self):
        s = CompassSettings(threshold=0.5, dwell_time=0.3, cooldown=0.5)
        assert s.threshold == 0.5
        assert s.dwell_time == 0.3
        assert s.cooldown == 0.5

    def test_round_trip(self):
        s = CompassSettings(min_x=10.0, max_x=500.0, min_y=20.0, max_y=400.0)
        assert CompassSettings.model_validate(s.model_dump()) == s


class TestGesturesSettings:
    def test_default_contains_compass(self):
        gs = GesturesSettings()
        assert isinstance(gs.compass, CompassSettings)

    def test_nested_compass_customizable(self):
        gs = GesturesSettings(compass=CompassSettings(threshold=0.6))
        assert gs.compass.threshold == 0.6


class TestGestureType:
    def test_all_values(self):
        assert GestureType.UP == "UP"
        assert GestureType.DOWN == "DOWN"
        assert GestureType.LEFT == "LEFT"
        assert GestureType.RIGHT == "RIGHT"

    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            GestureEvent(type="DIAGONAL", timestamp=0.0)


class TestGestureEvent:
    def test_required_fields(self):
        event = GestureEvent(type=GestureType.UP, timestamp=1234567890.0)
        assert event.type == GestureType.UP
        assert event.timestamp == 1234567890.0

    def test_optional_defaults(self):
        event = GestureEvent(type=GestureType.DOWN, timestamp=0.0)
        assert event.magnitude == 0.0
        assert event.duration == 0.0

    def test_all_fields(self):
        event = GestureEvent(
            type=GestureType.LEFT,
            timestamp=100.0,
            magnitude=0.85,
            duration=0.6,
        )
        assert event.magnitude == 0.85
        assert event.duration == 0.6

    def test_type_accepts_string_value(self):
        event = GestureEvent(type="RIGHT", timestamp=0.0)
        assert event.type == GestureType.RIGHT

    @pytest.mark.parametrize("direction", ["UP", "DOWN", "LEFT", "RIGHT"])
    def test_all_directions_valid(self, direction):
        event = GestureEvent(type=direction, timestamp=0.0)
        assert event.type.value == direction

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            GestureEvent(timestamp=0.0)  # type: ignore[call-arg]

    def test_missing_timestamp_raises(self):
        with pytest.raises(ValidationError):
            GestureEvent(type=GestureType.UP)  # type: ignore[call-arg]
