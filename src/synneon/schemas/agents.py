from typing import Literal

from pydantic import BaseModel, Field

ThinkingEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]


class AgentSettings(BaseModel):
    """
    Configuration settings for a CognitiveSteeringAgent.

    Defines the model to be used, as well as the system and user prompt templates
    which are used for context injection.
    """

    model: str = Field(
        ...,
        description="pydantic-ai model string with provider prefix (e.g. 'openai:gpt-6.1-sol', 'google:gemini-3.8-flash').",
    )
    system_prompt: str = Field(..., description="Template for the system instructions, supporting {context} injection.")
    prompt_template: str = Field(
        ..., description="Template for the user message, supporting {context} and {tokens} injection."
    )

    # Metadata and provider-specific configurations
    description: str | None = Field(None, description="Human-readable description of this agent profile.")
    provider: str | None = Field(
        None, description="'openai' or 'google' (informational), or 'alignenv' to delegate to the external API."
    )
    endpoint_url: str | None = Field(
        None, description="Base URL of the external API. Required when provider is 'alignenv'."
    )
    thinking: ThinkingEffort | None = Field(
        None,
        description="Provider-agnostic reasoning effort; pydantic-ai maps it to the nearest level the model supports.",
    )

    @property
    def accepts_audio(self) -> bool:
        """Whether the model can take raw audio input (Gemini can; OpenAI's Responses API cannot)."""
        return self.model.startswith(("google:", "google-cloud:"))
