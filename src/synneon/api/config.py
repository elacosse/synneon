import tomllib
from pathlib import Path

from synneon.schemas.agents import AgentSettings
from synneon.schemas.memory import MemoryConfig
from synneon.schemas.tts import STTConfig, TTSConfig

# Configuration paths (relative to project root)
AGENT_CONFIG_PATH = Path("configs/interpreter/agent.toml")
AUDIO_CONFIG_PATH = Path("configs/tts/default.toml")
STT_CONFIG_PATH = Path("configs/stt/default.toml")
MEMORY_CONFIG_PATH = Path("configs/memory/default.toml")


def get_agent_settings(profile: str) -> AgentSettings:
    """Load agent settings from the TOML configuration."""
    if not AGENT_CONFIG_PATH.exists():
        # Fallback to absolute check if relative fails in some environments
        # Moving up to find project root: src/synneon/api/config.py -> src/synneon/api -> src/synneon -> src -> root
        alt_path = Path(__file__).parent.parent.parent.parent / AGENT_CONFIG_PATH
        if not alt_path.exists():
            raise FileNotFoundError(f"Config file not found at {AGENT_CONFIG_PATH}")
        path = alt_path
    else:
        path = AGENT_CONFIG_PATH

    with open(path, "rb") as f:
        data = tomllib.load(f)

    agent_data = data.get("agent", {}).get(profile)
    if not agent_data:
        raise ValueError(f"Profile '{profile}' not found in {path}")

    return AgentSettings(**agent_data)


def get_tts_settings(profile: str) -> TTSConfig:
    """Load audio (STT/TTS) settings from the TOML configuration."""
    if not AUDIO_CONFIG_PATH.exists():
        # Fallback to absolute check if relative fails in some environments
        alt_path = Path(__file__).parent.parent.parent.parent / AUDIO_CONFIG_PATH
        if not alt_path.exists():
            raise FileNotFoundError(f"Config file not found at {AUDIO_CONFIG_PATH}")
        path = alt_path
    else:
        path = AUDIO_CONFIG_PATH

    with open(path, "rb") as f:
        data = tomllib.load(f)

    audio_data = data.get("tts", {}).get(profile)
    if not audio_data:
        raise ValueError(f"Profile '{profile}' not found in {path}")

    return TTSConfig(**audio_data)


def get_stt_settings(profile: str) -> STTConfig:
    """Load audio (STT) settings from the TOML configuration."""
    if not STT_CONFIG_PATH.exists():
        # Fallback to absolute check if relative fails in some environments
        alt_path = Path(__file__).parent.parent.parent.parent / STT_CONFIG_PATH
        if not alt_path.exists():
            raise FileNotFoundError(f"Config file not found at {STT_CONFIG_PATH}")
        path = alt_path
    else:
        path = STT_CONFIG_PATH

    with open(path, "rb") as f:
        data = tomllib.load(f)

    stt_data = data.get("stt", {}).get(profile)
    if not stt_data:
        raise ValueError(f"Profile '{profile}' not found in {path}")

    return STTConfig(**stt_data)


def get_memory_settings(profile: str = "default") -> MemoryConfig:
    """Load memory/Redis settings from the TOML configuration (SYNNEON_REDIS_URL overrides `redis_url`)."""
    if not MEMORY_CONFIG_PATH.exists():
        alt_path = Path(__file__).parent.parent.parent.parent / MEMORY_CONFIG_PATH
        if not alt_path.exists():
            return MemoryConfig().with_env_overrides()  # fall back to defaults
        path = alt_path
    else:
        path = MEMORY_CONFIG_PATH

    with open(path, "rb") as f:
        data = tomllib.load(f)

    memory_data = data.get("memory", {}).get(profile)
    if not memory_data:
        return MemoryConfig().with_env_overrides()

    return MemoryConfig(**memory_data).with_env_overrides()
