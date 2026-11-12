import pytest
from pydantic import ValidationError

from synneon.schemas.neons import BlinkConfig, GazeSample, NeonConfig


class TestNeonConfig:
    def test_defaults(self):
        cfg = NeonConfig()
        assert cfg.client_addr == "192.168.1.79:8080"
        assert cfg.client_url == "neon.local:8080"

    def test_custom_values(self):
        cfg = NeonConfig(client_addr="10.0.0.1:8080", client_url="mydevice.local:8080")
        assert cfg.client_addr == "10.0.0.1:8080"
        assert cfg.client_url == "mydevice.local:8080"

    def test_round_trip(self):
        cfg = NeonConfig(client_addr="1.2.3.4:9090", client_url="dev.local:9090")
        assert NeonConfig.model_validate(cfg.model_dump()) == cfg


class TestBlinkConfig:
    def test_defaults(self):
        cfg = BlinkConfig()
        assert cfg.double_blink_action == "capture_world_camera_image_with_gaze_annotation"
        assert cfg.double_blink_delta == 0.3
        assert cfg.circle_annotation_radius == 75

    def test_action_can_be_none(self):
        cfg = BlinkConfig(double_blink_action=None)
        assert cfg.double_blink_action is None

    def test_custom_values(self):
        cfg = BlinkConfig(double_blink_delta=0.5, circle_annotation_radius=50)
        assert cfg.double_blink_delta == 0.5
        assert cfg.circle_annotation_radius == 50

    def test_round_trip(self):
        cfg = BlinkConfig(double_blink_delta=0.2, circle_annotation_radius=100)
        assert BlinkConfig.model_validate(cfg.model_dump()) == cfg


class TestGazeSample:
    def test_required_fields(self):
        sample = GazeSample(x=0.5, y=0.3, timestamp=1000.0, is_worn=True)
        assert sample.x == 0.5
        assert sample.y == 0.3
        assert sample.timestamp == 1000.0
        assert sample.is_worn is True

    def test_not_worn(self):
        sample = GazeSample(x=0.0, y=0.0, timestamp=0.0, is_worn=False)
        assert sample.is_worn is False

    def test_negative_coordinates_valid(self):
        sample = GazeSample(x=-10.0, y=-5.0, timestamp=1.0, is_worn=True)
        assert sample.x == -10.0

    @pytest.mark.parametrize("missing", ["x", "y", "timestamp", "is_worn"])
    def test_missing_required_field_raises(self, missing):
        data = {"x": 0.5, "y": 0.3, "timestamp": 1.0, "is_worn": True}
        del data[missing]
        with pytest.raises(ValidationError):
            GazeSample(**data)

    def test_round_trip(self):
        sample = GazeSample(x=100.5, y=200.3, timestamp=9999.9, is_worn=True)
        assert GazeSample.model_validate(sample.model_dump()) == sample
