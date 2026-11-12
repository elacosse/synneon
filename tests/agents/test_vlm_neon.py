import pytest
from pydantic import BaseModel

# Use TestModel so we do not hit real LLM APIs during unit tests
from pydantic_ai.models.test import TestModel

from synneon.agents.base import VLMNeon2ContextAgent
from synneon.schemas.agents import AgentSettings
from synneon.schemas.neons import (
    BlinkConfig,
    FixationConfig,
    NeonCapturePayload,
    RecordingConfig,
    SaliencyConfig,
)
from synneon.schemas.results import Image2ContextResult


class MockResult(BaseModel):
    analysis: str
    gaze_summary: str


@pytest.fixture
def mock_agent_settings():
    return AgentSettings(
        model="test",
        system_prompt="You are a VLM analyzing Neon capture data.",
        prompt_template="Here is the context: {neon_context}\n\nAnalyze this data.",
    )


@pytest.fixture
def mock_payload():
    return NeonCapturePayload(
        scene_image=b"fake_image_bytes",
        audio=b"fake_audio_bytes",
        video=[(1.0, b"frame1"), (2.0, b"frame2")],
        gaze_points=[(1.0, 100.0, 200.0), (1.1, 101.0, 201.0)],
        pupil_data=[(1.0, 4.2, 4.3), (1.1, 4.3, 4.4)],
        blink_config=BlinkConfig(),
        fixation_config=FixationConfig(),
        saliency_config=SaliencyConfig(),
        recording_config=RecordingConfig(),
    )


def test_vlm_neon_agent_sync(mock_agent_settings, mock_payload):
    """Test the synchronous execution of the VLMNeon2ContextAgent with TestModel."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=Image2ContextResult)

    # Override the model with TestModel for offline testing
    mock_json = '{"parsed_visual": {"focused_object": "test", "detected_objects": []}, "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": "test"}}'
    with agent.agent.override(model=TestModel(custom_output_text=mock_json)):
        result = agent.run_sync(mock_payload)

    # Since we are using TestModel, it generates dummy data conforming to the Image2ContextResult schema
    assert isinstance(result.output, Image2ContextResult)
    assert hasattr(result.output, "parsed_visual")
    assert hasattr(result.output, "pragmatic")


@pytest.mark.asyncio
async def test_vlm_neon_agent_async(mock_agent_settings, mock_payload):
    """Test the asynchronous execution of the VLMNeon2ContextAgent with TestModel."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=Image2ContextResult)

    mock_json = '{"parsed_visual": {"focused_object": "test", "detected_objects": []}, "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": "test"}}'
    with agent.agent.override(model=TestModel(custom_output_text=mock_json)):
        result = await agent.run(mock_payload)

    assert isinstance(result.output, Image2ContextResult)
    assert hasattr(result.output, "parsed_visual")
    assert hasattr(result.output, "pragmatic")


def test_vlm_neon_agent_prompt_formatting(mock_agent_settings, mock_payload):
    """Test that the neon_context correctly injects configurations and telemetry."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=str)

    prompt = agent._format_prompt(mock_payload)

    # Verify formatting template injected the dynamic neon_context
    assert "Here is the context:" in prompt
    assert "### Neon Capture Telemetry & Configuration" in prompt

    # Verify telemetry counts are accurate
    assert "Video Frames Captured: 2" in prompt
    assert "Gaze Points Captured: 2" in prompt
    assert "Pupil Data Samples Captured: 2" in prompt

    # Verify sample data was truncated and injected
    assert "(1.0, 100.0, 200.0)" in prompt
    assert "(1.0, 4.2, 4.3)" in prompt

    # Verify configuration json dumps are present
    assert "max_peaks" in prompt  # from SaliencyConfig
    assert "max_dispersion_pixels" in prompt  # from FixationConfig


def test_vlm_neon_agent_message_builder(mock_agent_settings, mock_payload):
    """Test that the BinaryContent attachments are correctly constructed for an audio-capable model."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=str)
    agent.settings = mock_agent_settings.model_copy(update={"model": "google:gemini-3.8-flash"})

    messages = agent._build_messages(mock_payload, "Test User Prompt")

    # Message list should contain: [User Prompt String, Image Attachment, Audio Attachment]
    assert len(messages) == 3
    assert messages[0] == "Test User Prompt"

    # Verify the image attachment (pydantic_ai.BinaryContent)
    assert messages[1].media_type == "image/png"
    assert messages[1].data == b"fake_image_bytes"

    # Verify the audio attachment
    assert messages[2].media_type == "audio/wav"
    assert messages[2].data == b"fake_audio_bytes"


def test_vlm_neon_agent_missing_attachments(mock_agent_settings, mock_payload):
    """Test that the message builder handles empty multimodal attachments gracefully."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=str)

    # Remove attachments
    mock_payload.scene_image = None
    mock_payload.audio = None

    messages = agent._build_messages(mock_payload, "Test User Prompt")

    # Message list should only contain the text prompt
    assert len(messages) == 1
    assert messages[0] == "Test User Prompt"


def test_vlm_neon_agent_skips_audio_for_models_without_audio_input(mock_agent_settings, mock_payload):
    """OpenAI models cannot take audio: the clip is dropped and a transcript goes in the prompt instead."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=str)
    agent.settings = mock_agent_settings.model_copy(update={"model": "openai:gpt-6.1-terra"})

    messages = agent._build_messages(mock_payload, "Test User Prompt")
    assert [getattr(m, "media_type", None) for m in messages] == [None, "image/png"]

    prompt = agent._format_prompt(mock_payload, audio_transcript="SPEAKER_1: pass the water")
    assert "**Ambient Audio Transcript:**" in prompt
    assert "SPEAKER_1: pass the water" in prompt


def test_vlm_neon_agent_prompt_includes_fixations(mock_agent_settings, mock_payload):
    """Computed fixations are sent, not just raw gaze samples."""
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=Image2ContextResult)
    # 300 ms of steady gaze at (500, 400): one fixation.
    payload = mock_payload.model_copy(update={"gaze_points": [(i * 0.01, 500.0, 400.0) for i in range(31)]})

    prompt = agent._format_prompt(payload)

    assert "Gaze Fixations (1 detected" in prompt
    assert "x=500, y=400, duration=300 ms" in prompt


@pytest.mark.parametrize(
    ("image", "media_type"), [(b"\xff\xd8\xff\xe0jpeg", "image/jpeg"), (b"\x89PNG\r\n", "image/png")]
)
def test_vlm_neon_agent_labels_scene_image_by_content(mock_agent_settings, mock_payload, image, media_type):
    agent = VLMNeon2ContextAgent(settings=mock_agent_settings, output_type=Image2ContextResult)
    payload = mock_payload.model_copy(update={"scene_image": image})

    messages = agent._build_messages(payload, "prompt")

    assert messages[1].media_type == media_type
