from pydantic import BaseModel, Field

from synneon.schemas.context import SteeringContext
from synneon.schemas.neons import BlinkConfig, FixationConfig, RecordingConfig, SaliencyConfig


class CognitiveSteeringInferenceRequest(BaseModel):
    """
    Request schema for performing inference with a CognitiveSteeringAgent.
    """

    context: SteeringContext = Field(
        ..., description="The simplified 'State of the World' snapshot used to steer the agent."
    )
    tokens: str = Field(..., description="The specific abbreviation or input tokens to be processed by the agent.")
    profile: str = Field(
        "default", description="The configuration profile name to use (e.g., 'default' from agent.toml)."
    )
    more_candidates: bool = Field(
        False, description="If True, requests alternative candidates using sample & rerank at a higher temperature."
    )


class Image2ContextRequest(BaseModel):
    """
    Request schema for performing image-to-context inference using base64 encoded images.
    """

    image_base64: str = Field(..., description="Base64 encoded image data.")
    media_type: str = Field("image/jpeg", description="MIME type of the image (e.g., 'image/png').")
    profile: str = Field(
        "image2context", description="The configuration profile name to use (e.g., 'image2context' from agent.toml)."
    )
    save_to_memory: bool = Field(False, description="If True, persist the inference result as a MemoryEntry.")


class Audio2TextRequest(BaseModel):
    """
    Request schema for performing audio-to-text inference (transcription/diarization) using base64 encoded audio.
    """

    audio_base64: str = Field(..., description="Base64 encoded audio data.")
    media_type: str = Field("audio/wav", description="MIME type of the audio (e.g., 'audio/wav', 'audio/mp3').")
    profile: str = Field(
        "default", description="The STT configuration profile name to use (e.g., 'default' from stt/default.toml)."
    )
    save_to_memory: bool = Field(
        False, description="If True, persist diarized utterances as ConversationEntry records."
    )


class UtteranceRequest(BaseModel):
    """Request schema for saving a single user utterance to the conversation store."""

    text: str = Field(..., description="The utterance text to persist.")
    role: str = Field("user", description="Speaker role: 'user' (glasses wearer) or 'other' (ambient speaker).")
    source: str = Field("expansion", description="Origin of the utterance (e.g. 'expansion', 'stt').")


class SpeakRequest(BaseModel):
    """Request body for text-to-speech. Keeps the spoken text out of the URL and access logs."""

    text: str = Field(..., min_length=1, description="Text to synthesise into speech.")
    profile: str = Field("default", description="The TTS configuration profile to use.")
    voice: str | None = Field(None, description="Voice name. Overrides the profile config.")
    model: str | None = Field(None, description="TTS model. Overrides the profile config.")
    emotion_label: str | None = Field(None, description="Emotion label (e.g. 'happy', 'sad').")
    emotion_intensity: float = Field(0.0, ge=0.0, le=1.0, description="Emotion intensity 0–1.")


class NeonCaptureInferenceRequest(BaseModel):
    """
    Request schema for performing inference over a complex Neon multimodal capture using JSON.
    Binary data (images, audio) must be base64 encoded strings.
    """

    # Optional Base64 Media
    scene_image_base64: str | None = Field(None, description="Base64 encoded middle scene image.")
    audio_base64: str | None = Field(None, description="Base64 encoded wav audio data.")
    video_base64: list[tuple[float, str]] = Field(
        default_factory=list, description="Chronological list of (timestamp, base64_image_string) frames."
    )

    # Telemetry
    gaze_points: list[tuple[float, float, float]] = Field(
        default_factory=list, description="List of (timestamp, x, y)."
    )
    pupil_data: list[tuple[float, float, float]] = Field(
        default_factory=list, description="List of (timestamp, left_mm, right_mm)."
    )

    # Required Configurations
    blink_config: BlinkConfig
    fixation_config: FixationConfig
    saliency_config: SaliencyConfig
    recording_config: RecordingConfig

    # Inference Profile
    profile: str = Field("default", description="The VLM configuration profile name to use from agent.toml.")
    stt_profile: str = Field(
        "default", description="STT profile used to transcribe the audio when the agent's model cannot take audio."
    )
    save_to_memory: bool = Field(False, description="If True, persist the inference result as a MemoryEntry.")
