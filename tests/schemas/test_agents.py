import pytest
from pydantic import ValidationError

from synneon.schemas.agents import AgentSettings


class TestAgentSettings:
    def test_required_fields(self):
        settings = AgentSettings(
            model="google:gemini-3.8-flash",
            system_prompt="You are a helpful assistant. Context: {context}",
            prompt_template="Expand: {tokens}",
        )
        assert settings.model == "google:gemini-3.8-flash"
        assert settings.system_prompt == "You are a helpful assistant. Context: {context}"
        assert settings.prompt_template == "Expand: {tokens}"

    def test_optional_fields_default_to_none(self):
        settings = AgentSettings(
            model="m",
            system_prompt="s",
            prompt_template="p",
        )
        assert settings.description is None
        assert settings.provider is None
        assert settings.thinking is None

    def test_optional_fields_can_be_set(self):
        settings = AgentSettings(
            model="m",
            system_prompt="s",
            prompt_template="p",
            description="A test agent",
            provider="google",
            thinking="high",
        )
        assert settings.description == "A test agent"
        assert settings.provider == "google"
        assert settings.thinking == "high"

    @pytest.mark.parametrize("missing", ["model", "system_prompt", "prompt_template"])
    def test_missing_required_field_raises(self, missing):
        data = {"model": "m", "system_prompt": "s", "prompt_template": "p"}
        del data[missing]
        with pytest.raises(ValidationError):
            AgentSettings(**data)

    def test_round_trip(self):
        settings = AgentSettings(
            model="google:gemini-3.8-flash",
            system_prompt="sys",
            prompt_template="usr",
            description="desc",
            provider="google",
            thinking="high",
        )
        assert AgentSettings.model_validate(settings.model_dump()) == settings

    def test_invalid_thinking_rejected(self):
        with pytest.raises(ValidationError):
            AgentSettings(model="m", system_prompt="s", prompt_template="p", thinking="google-high")

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("google:gemini-3.8-flash", True),
            ("google-cloud:gemini-3.8-flash", True),
            ("openai:gpt-6.1-sol", False),
            ("alignenv:aac-interpreter", False),
        ],
    )
    def test_accepts_audio(self, model, expected):
        assert AgentSettings(model=model, system_prompt="s", prompt_template="p").accepts_audio is expected
