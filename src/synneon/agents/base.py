from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.capabilities import Thinking

from synneon.schemas.agents import AgentSettings
from synneon.schemas.context import SteeringContext
from synneon.schemas.neons import NeonCapturePayload

if TYPE_CHECKING:
    from synneon.schemas.latents import LatentConfig


def _capabilities(settings: AgentSettings) -> list[Any]:
    """Translate provider-agnostic settings into pydantic-ai capabilities."""
    return [Thinking(effort=settings.thinking)] if settings.thinking else []


class CognitiveSteeringContextBuilder:
    """Builder for system and user prompts using SteeringContext."""

    settings: AgentSettings

    def __init__(self, settings: AgentSettings, latent_config: LatentConfig | None = None):
        self.settings = settings
        self._latent_config = latent_config

    def build_prompts(self, context: SteeringContext, **kwargs: Any) -> tuple[str, str]:
        """Build the system and user prompts by injecting context."""
        if self._latent_config is not None:
            adj = context.emotion.intensity_qualifier_from_config(self._latent_config.emotion)
            emotion_str = f"{adj} {context.emotion.label.value}"
            context_str = f"Emotion: {emotion_str}; Visual: {context.parsed_visual}; Pragmatic: {context.pragmatic}"
        else:
            emotion_str = str(context.emotion)
            context_str = str(context)
        # Combine context with other formatting arguments (e.g., tokens)
        format_args = {"context": context_str, "emotion": emotion_str, **kwargs}

        # We use a custom dict to ignore missing keys during partial formatting
        # (e.g. when building system prompt without having user tokens yet)
        class PartialFormatter(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        formatter = PartialFormatter(**format_args)
        system_prompt = self.settings.system_prompt.format_map(formatter)
        user_prompt = self.settings.prompt_template.format_map(formatter)

        return system_prompt, user_prompt


class CognitiveSteeringAgent[OutputT]:
    """Generic wrapper around pydantic_ai.Agent.

    Utilizes AgentSettings for configuration and CognitiveSteeringContextBuilder
    for dynamic prompt construction based on SteeringContext.
    """

    settings: AgentSettings
    builder: CognitiveSteeringContextBuilder
    agent: Agent[SteeringContext, OutputT]

    def __init__(
        self,
        settings: AgentSettings,
        output_type: type[OutputT],
        latent_config: LatentConfig | None = None,
        **kwargs: Any,
    ):
        self.settings = settings
        self.builder = CognitiveSteeringContextBuilder(settings, latent_config=latent_config)

        # Initialize the agent with configuration from settings.
        model_name = settings.model

        self.agent = Agent(
            model=model_name,
            output_type=output_type,
            deps_type=SteeringContext,
            capabilities=_capabilities(settings),
            **kwargs,
        )

    async def run(self, context: SteeringContext, **kwargs: Any):
        """Builds prompts from context and executes the agent."""
        # Both prompts are built here, with the same arguments, so every placeholder is filled
        # and the prompts a caller logs via `builder.build_prompts` are the ones the model receives.
        system_prompt, user_prompt = self.builder.build_prompts(context, **kwargs)
        return await self.agent.run(user_prompt, deps=context, instructions=system_prompt)

    def run_sync(self, context: SteeringContext, **kwargs: Any):
        """Synchronous version of run."""
        system_prompt, user_prompt = self.builder.build_prompts(context, **kwargs)
        return self.agent.run_sync(user_prompt, deps=context, instructions=system_prompt)


class VLMImage2ContextAgent[OutputT]:
    """Agent for Vision-Language Model tasks, converting images to context/descriptions.

    Utilizes AgentSettings for configuration and supports passing images (bytes or base64)
    directly to the model.
    """

    settings: AgentSettings
    agent: Agent[None, str]
    output_type: type[OutputT]

    def __init__(
        self,
        settings: AgentSettings,
        output_type: type[OutputT],
        **kwargs: Any,
    ):
        self.settings = settings
        self.output_type = output_type
        self.agent = Agent(
            model=settings.model,
            system_prompt=settings.system_prompt,
            capabilities=_capabilities(settings),
            **kwargs,
        )

    def _prepare_data(self, image_data: bytes | str) -> bytes:
        if isinstance(image_data, str):
            if "," in image_data:
                image_data = image_data.split(",")[1]
            return base64.b64decode(image_data)
        return image_data

    def _format_prompt(self, **kwargs: Any) -> str:
        class PartialFormatter(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        formatter = PartialFormatter(**kwargs)
        return self.settings.prompt_template.format_map(formatter)

    def _parse_output(self, text: str) -> OutputT:
        import re

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            text = text.strip()
        return self.output_type.model_validate_json(text)

    async def run(self, image_data: bytes | str, media_type: str = "image/jpeg", **kwargs: Any):
        """Run the VLM agent on an image. image_data can be raw bytes or a base64 string."""
        import dataclasses

        data = self._prepare_data(image_data)
        user_prompt = self._format_prompt(**kwargs)

        result = await self.agent.run(
            [
                user_prompt,
                BinaryContent(data=data, media_type=media_type),
            ]
        )
        parsed_output = self._parse_output(result.output)
        return dataclasses.replace(result, output=parsed_output)

    def run_sync(self, image_data: bytes | str, media_type: str = "image/jpeg", **kwargs: Any):
        """Synchronous version of run."""
        import dataclasses

        data = self._prepare_data(image_data)
        user_prompt = self._format_prompt(**kwargs)

        result = self.agent.run_sync(
            [
                user_prompt,
                BinaryContent(data=data, media_type=media_type),
            ]
        )
        parsed_output = self._parse_output(result.output)
        return dataclasses.replace(result, output=parsed_output)


class VLMNeon2ContextAgent[OutputT]:
    """Agent for extracting context from comprehensive Neon capture data.

    Ingests the NeonCapturePayload, formatting telemetry and configuration as text
    while providing the scene image and audio as native multimodal attachments.
    """

    settings: AgentSettings
    agent: Agent[None, str]
    output_type: type[OutputT]

    def __init__(
        self,
        settings: AgentSettings,
        output_type: type[OutputT],
        **kwargs: Any,
    ):
        self.settings = settings
        self.output_type = output_type
        self.agent = Agent(
            model=settings.model,
            system_prompt=settings.system_prompt,
            capabilities=_capabilities(settings),
            **kwargs,
        )

    def _format_prompt(self, payload: NeonCapturePayload, audio_transcript: str | None = None, **kwargs: Any) -> str:
        # We format a markdown string of the payload data to inject
        data_summary = [
            "### Neon Capture Telemetry & Configuration",
            "**Configurations:**",
            f"- Blink: {payload.blink_config.model_dump_json()}",
            f"- Fixation: {payload.fixation_config.model_dump_json()}",
            f"- Saliency: {payload.saliency_config.model_dump_json()}",
            f"- Recording: {payload.recording_config.model_dump_json()}",
            "",
            "**Telemetry Summary:**",
            f"- Video Frames Captured: {len(payload.video)}",
            f"- Gaze Points Captured: {len(payload.gaze_points)}",
            f"- Pupil Data Samples Captured: {len(payload.pupil_data)}",
            "",
        ]

        # Computed features first: fixations say where attention rested, which raw samples do not.
        if payload.gaze_points:
            from synneon.services.perception.vision import detect_fixations_idt

            fixations = detect_fixations_idt(
                payload.gaze_points,
                min_duration_ms=payload.fixation_config.min_duration_ms,
                max_dispersion_pixels=payload.fixation_config.max_dispersion_pixels,
            )
            if fixations:
                longest = sorted(fixations, key=lambda f: f["duration_ms"], reverse=True)[:5]
                data_summary.append(f"**Gaze Fixations ({len(fixations)} detected; longest first, scene pixels):**")
                for f in longest:
                    data_summary.append(
                        f"  - x={f['centroid_x']:.0f}, y={f['centroid_y']:.0f}, duration={f['duration_ms']:.0f} ms"
                    )

        # Truncate raw telemetry if there are too many to fit in context
        # Provide a small sample of the data to give the model an idea
        if payload.gaze_points:
            data_summary.append("**Gaze Points (First 10 samples):**")
            for pt in payload.gaze_points[:10]:
                data_summary.append(f"  - {pt}")

        if payload.pupil_data:
            data_summary.append("**Pupil Data (First 10 samples):**")
            for pt in payload.pupil_data[:10]:
                data_summary.append(f"  - {pt}")

        if audio_transcript:
            data_summary.append("**Ambient Audio Transcript:**")
            data_summary.append(audio_transcript)

        payload_context = "\n".join(data_summary)

        class PartialFormatter(dict):
            def __missing__(self, key: str) -> str:
                if key == "neon_context":
                    return payload_context
                return "{" + key + "}"

        # Combine provided kwargs with our generated neon context
        format_args = {"neon_context": payload_context, **kwargs}
        formatter = PartialFormatter(**format_args)

        formatted_prompt = self.settings.prompt_template.format_map(formatter)
        if "{neon_context}" not in self.settings.prompt_template:
            formatted_prompt += f"\n\n{payload_context}"

        return formatted_prompt

    def _build_messages(self, payload: NeonCapturePayload, user_prompt: str) -> list[Any]:
        messages: list[Any] = [user_prompt]

        if payload.scene_image:
            # The dashboard sends JPEG frames; the offline runner sends PNG.
            is_jpeg = payload.scene_image[:3] == b"\xff\xd8\xff"
            messages.append(
                BinaryContent(data=payload.scene_image, media_type="image/jpeg" if is_jpeg else "image/png")
            )

        # Models without audio input get a transcript in the prompt instead (see `audio_transcript`).
        if payload.audio and self.settings.accepts_audio:
            messages.append(BinaryContent(data=payload.audio, media_type="audio/wav"))

        return messages

    def _parse_output(self, text: str) -> OutputT:
        import re

        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            text = text.strip()
        return self.output_type.model_validate_json(text)

    async def run(self, payload: NeonCapturePayload, **kwargs: Any):
        """Run the agent on a comprehensive Neon capture payload.

        Pass `audio_transcript=` when the model cannot take audio (see `AgentSettings.accepts_audio`).
        """
        import dataclasses

        user_prompt = self._format_prompt(payload, **kwargs)
        messages = self._build_messages(payload, user_prompt)
        result = await self.agent.run(messages)
        parsed_output = self._parse_output(result.output)
        return dataclasses.replace(result, output=parsed_output)

    def run_sync(self, payload: NeonCapturePayload, **kwargs: Any):
        """Synchronous version of run."""
        import dataclasses

        user_prompt = self._format_prompt(payload, **kwargs)
        messages = self._build_messages(payload, user_prompt)
        result = self.agent.run_sync(messages)
        parsed_output = self._parse_output(result.output)
        return dataclasses.replace(result, output=parsed_output)
