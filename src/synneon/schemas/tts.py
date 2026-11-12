from typing import Literal

from pydantic import BaseModel, Field

from synneon.schemas.agents import ThinkingEffort

Provider = Literal["openai", "google"]


class STTConfig(BaseModel):
    """Speech-to-text profile."""

    provider: Provider = "openai"
    model: str = "gpt-transcribe"
    prompt: str | None = Field(
        None, description="Context for the transcriber: OpenAI `prompt`, Gemini system instruction."
    )
    languages: list[str] = Field(default_factory=lambda: ["en"], description="BCP-47 language hints.")
    thinking: ThinkingEffort | None = Field(None, description="Gemini only: thinking_level for transcription.")


class TTSConfig(BaseModel):
    """Text-to-speech profile."""

    description: str | None = None
    provider: Provider = "openai"
    model: str = "gpt-realtime-2.1-mini"
    voice: str = "marin"
