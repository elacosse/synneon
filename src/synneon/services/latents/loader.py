import tomllib
from functools import lru_cache
from pathlib import Path

from synneon.schemas.latents import EmotionLatentConfig, LatentConfig, LatentThreshold

_LATENTS_CONFIG_DIR = Path("configs/latents")


@lru_cache(maxsize=8)
def load_latent_config(name: str = "default") -> LatentConfig:
    """Load and cache a latent config by name from configs/latents/{name}.toml."""
    rel_path = _LATENTS_CONFIG_DIR / f"{name}.toml"

    if rel_path.exists():
        path = rel_path
    else:
        # Fallback: resolve from package root
        # src/synneon/services/latents/loader.py -> 5 parents -> project root
        pkg_root = Path(__file__).parent.parent.parent.parent.parent
        abs_path = pkg_root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Latent config not found: {rel_path}")
        path = abs_path

    with open(path, "rb") as f:
        data = tomllib.load(f)

    emotion_data = data.get("emotion", {})
    raw_thresholds = emotion_data.pop("thresholds", [])
    thresholds = [LatentThreshold(**t) for t in raw_thresholds]
    emotion_cfg = EmotionLatentConfig(**emotion_data, thresholds=thresholds)
    return LatentConfig(emotion=emotion_cfg)
