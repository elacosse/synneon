import pytest

from synneon.apps.shared.config import load_blink_config, load_gestures_config, load_neon_config


def _write_toml(tmp_path, content: str):
    p = tmp_path / "config.toml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# load_neon_config
# ---------------------------------------------------------------------------


class TestLoadNeonConfig:
    def test_loads_neons_section(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[neons]
client_addr = "10.0.0.5:8080"
client_url = "device.local:8080"
""",
        )
        cfg = load_neon_config(p)
        assert cfg.client_addr == "10.0.0.5:8080"
        assert cfg.client_url == "device.local:8080"

    def test_loads_device_section(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[device]
address = "10.0.0.2"
port = 9090
""",
        )
        cfg = load_neon_config(p)
        assert cfg.client_addr == "10.0.0.2:9090"
        assert cfg.client_url == "10.0.0.2:9090"

    def test_device_section_default_port(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[device]
address = "neon.local"
""",
        )
        cfg = load_neon_config(p)
        assert "neon.local" in cfg.client_addr
        assert "8080" in cfg.client_addr

    def test_no_matching_section_uses_defaults(self, tmp_path):
        p = _write_toml(tmp_path, "[other]\nkey = 1\n")
        cfg = load_neon_config(p)
        assert cfg.client_addr == "192.168.1.79:8080"

    def test_file_not_found_raises(self, tmp_path):
        from pathlib import Path

        with pytest.raises(FileNotFoundError):
            load_neon_config(Path(tmp_path / "nonexistent.toml"))


# ---------------------------------------------------------------------------
# load_blink_config
# ---------------------------------------------------------------------------


class TestLoadBlinkConfig:
    def test_loads_blink_section(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[blink]
double_blink_action = "my_action"
double_blink_delta = 0.4
circle_annotation_radius = 60
""",
        )
        cfg = load_blink_config(p)
        assert cfg.double_blink_action == "my_action"
        assert cfg.double_blink_delta == pytest.approx(0.4)
        assert cfg.circle_annotation_radius == 60

    def test_loads_capture_section_fallback(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[capture]
double_blink_action = "capture_action"
double_blink_delta = 0.2
""",
        )
        cfg = load_blink_config(p)
        assert cfg.double_blink_action == "capture_action"
        assert cfg.double_blink_delta == pytest.approx(0.2)

    def test_no_matching_section_uses_defaults(self, tmp_path):
        p = _write_toml(tmp_path, "[other]\nkey = 1\n")
        cfg = load_blink_config(p)
        assert cfg.double_blink_delta == pytest.approx(0.3)
        assert cfg.circle_annotation_radius == 75

    def test_blink_section_preferred_over_capture(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[blink]
double_blink_delta = 0.1

[capture]
double_blink_delta = 0.9
""",
        )
        cfg = load_blink_config(p)
        assert cfg.double_blink_delta == pytest.approx(0.1)

    def test_null_action_allowed(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[blink]
double_blink_action = ""
double_blink_delta = 0.3
""",
        )
        cfg = load_blink_config(p)
        # Empty string is valid for the model
        assert cfg.double_blink_action == ""


# ---------------------------------------------------------------------------
# load_gestures_config
# ---------------------------------------------------------------------------


class TestLoadGesturesConfig:
    def test_loads_gestures_compass_section(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[gestures.compass]
threshold = 0.7
dwell_time = 0.3
cooldown = 0.5
""",
        )
        cfg = load_gestures_config(p)
        assert cfg.compass.threshold == pytest.approx(0.7)
        assert cfg.compass.dwell_time == pytest.approx(0.3)
        assert cfg.compass.cooldown == pytest.approx(0.5)

    def test_no_gestures_section_uses_defaults(self, tmp_path):
        p = _write_toml(tmp_path, "[other]\nkey = 1\n")
        cfg = load_gestures_config(p)
        assert cfg.compass.threshold == pytest.approx(0.8)

    def test_calibration_merged_into_compass(self, tmp_path):
        p = _write_toml(
            tmp_path,
            """
[gestures.compass]
threshold = 0.8

[calibration]
min_x = 10.0
max_x = 500.0
min_y = 20.0
max_y = 400.0
""",
        )
        cfg = load_gestures_config(p)
        assert cfg.compass.min_x == pytest.approx(10.0)
        assert cfg.compass.max_x == pytest.approx(500.0)
        assert cfg.compass.min_y == pytest.approx(20.0)
        assert cfg.compass.max_y == pytest.approx(400.0)

    def test_calibration_partial_merge(self, tmp_path):
        """Only present calibration keys are merged; missing keys keep compass defaults."""
        p = _write_toml(
            tmp_path,
            """
[gestures.compass]
min_x = 0.0
max_x = 1000.0
min_y = 0.0
max_y = 800.0

[calibration]
min_x = 5.0
""",
        )
        cfg = load_gestures_config(p)
        assert cfg.compass.min_x == pytest.approx(5.0)
        assert cfg.compass.max_x == pytest.approx(1000.0)  # unchanged

    def test_calibration_without_gestures_section(self, tmp_path):
        """Calibration merges into default CompassSettings if no [gestures] section."""
        p = _write_toml(
            tmp_path,
            """
[calibration]
min_x = 100.0
max_x = 900.0
min_y = 50.0
max_y = 700.0
""",
        )
        cfg = load_gestures_config(p)
        assert cfg.compass.min_x == pytest.approx(100.0)
        assert cfg.compass.max_x == pytest.approx(900.0)
