from __future__ import annotations

import tomllib
from pathlib import Path

from synneon.schemas.gestures import GesturesSettings
from synneon.schemas.neons import BlinkConfig, FixationConfig, NeonConfig, RecordingConfig, SaliencyConfig


def _load_neon_toml(path: Path | None) -> dict:
    """Load raw TOML data from a config file, or return empty dict."""
    if path is None:
        # Try local configs directory first, then fall back to package-relative path
        path = Path("configs/neons/default.toml")
        if not path.exists():
            path = Path(__file__).resolve().parents[4] / "configs" / "neons" / "default.toml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_neon_config(path: Path | None) -> NeonConfig:
    data = _load_neon_toml(path)

    # Handle [neons] section
    if "neons" in data:
        return NeonConfig(**data["neons"])

    # Handle [device] section (common in calibrate.py layout)
    if "device" in data:
        device = data["device"]
        addr = device.get("address", "neon.local")
        port = device.get("port", 8080)
        return NeonConfig(client_addr=f"{addr}:{port}", client_url=f"{addr}:{port}")

    return NeonConfig()


def load_gestures_config(path: Path | None) -> GesturesSettings:
    data = _load_neon_toml(path)
    settings = GesturesSettings(**data["gestures"]) if "gestures" in data else GesturesSettings()

    # Merge calibration values if they exist (populated by the calibrate command)
    if "calibration" in data:
        cal = data["calibration"]
        for key in ["min_x", "max_x", "min_y", "max_y"]:
            if key in cal and cal[key] is not None:
                setattr(settings.compass, key, float(cal[key]))

    return settings


def load_blink_config(path: Path | None) -> BlinkConfig:
    data = _load_neon_toml(path)
    if "blink" in data:
        return BlinkConfig(**data["blink"])
    if "capture" in data:
        return BlinkConfig(**data["capture"])
    return BlinkConfig()


def load_fixation_config(path: Path | None) -> FixationConfig:
    data = _load_neon_toml(path)
    if "fixation" in data:
        return FixationConfig(**data["fixation"])
    return FixationConfig()


def load_saliency_config(path: Path | None) -> SaliencyConfig:
    data = _load_neon_toml(path)
    if "saliency" in data:
        return SaliencyConfig(**data["saliency"])
    return SaliencyConfig()


def load_recording_config(path: Path | None) -> RecordingConfig:
    data = _load_neon_toml(path)
    if "recording" in data:
        return RecordingConfig(**data["recording"])
    return RecordingConfig()
