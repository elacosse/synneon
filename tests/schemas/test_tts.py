import pytest
from pydantic import ValidationError

from synneon.schemas.tts import STTConfig, TTSConfig


class TestTTSConfig:
    def test_defaults_to_openai_realtime(self):
        cfg = TTSConfig()
        assert cfg.description is None
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-realtime-2.1-mini"
        assert cfg.voice == "marin"

    def test_gemini_provider(self):
        cfg = TTSConfig(provider="google", model="gemini-3.8-flash-tts", voice="Kore")
        assert cfg.provider == "google"
        assert cfg.voice == "Kore"

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValidationError):
            TTSConfig(provider="together")

    def test_round_trip(self):
        cfg = TTSConfig(provider="google", model="gemini-3.8-flash-tts", voice="Puck", description="test")
        assert TTSConfig.model_validate(cfg.model_dump()) == cfg


class TestSTTConfig:
    def test_defaults_to_openai(self):
        cfg = STTConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-transcribe"
        assert cfg.languages == ["en"]
        assert cfg.prompt is None
        assert cfg.thinking is None

    def test_gemini_with_thinking(self):
        cfg = STTConfig(provider="google", model="gemini-3.8-flash", thinking="low")
        assert cfg.thinking == "low"

    def test_invalid_thinking_rejected(self):
        with pytest.raises(ValidationError):
            STTConfig(thinking="very-high")
