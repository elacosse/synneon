import pytest
from pydantic import ValidationError

from synneon.schemas.context import (
    CommunicativeMotive,
    EmotionContext,
    EmotionLabel,
    ParsedVisualContext,
    PragmaticContext,
    SteeringContext,
)
from synneon.schemas.requests import CognitiveSteeringInferenceRequest, Image2ContextRequest


def _make_context(**kwargs) -> SteeringContext:
    return SteeringContext(**kwargs)


class TestCognitiveSteeringInferenceRequest:
    def test_required_fields(self):
        req = CognitiveSteeringInferenceRequest(
            context=_make_context(),
            tokens="h-w-a-y",
        )
        assert isinstance(req.context, SteeringContext)
        assert req.tokens == "h-w-a-y"

    def test_profile_defaults_to_default(self):
        req = CognitiveSteeringInferenceRequest(
            context=_make_context(),
            tokens="abc",
        )
        assert req.profile == "default"

    def test_profile_can_be_overridden(self):
        req = CognitiveSteeringInferenceRequest(
            context=_make_context(),
            tokens="abc",
            profile="custom",
        )
        assert req.profile == "custom"

    def test_missing_context_raises(self):
        with pytest.raises(ValidationError):
            CognitiveSteeringInferenceRequest(tokens="abc")  # type: ignore[call-arg]

    def test_missing_tokens_raises(self):
        with pytest.raises(ValidationError):
            CognitiveSteeringInferenceRequest(context=_make_context())  # type: ignore[call-arg]

    def test_context_with_full_state(self):
        ctx = SteeringContext(
            emotion=EmotionContext(label=EmotionLabel.URGENCY, intensity=0.9),
            parsed_visual=ParsedVisualContext(focused_object="exit sign"),
            pragmatic=PragmaticContext(
                intent=CommunicativeMotive.IMPERATIVE,
                physical_setting="hospital",
            ),
        )
        req = CognitiveSteeringInferenceRequest(context=ctx, tokens="g-m-o-o-n")
        assert req.context.emotion.label == EmotionLabel.URGENCY
        assert req.context.parsed_visual.focused_object == "exit sign"

    def test_round_trip(self):
        req = CognitiveSteeringInferenceRequest(
            context=_make_context(),
            tokens="t-t-t",
            profile="default",
        )
        assert CognitiveSteeringInferenceRequest.model_validate(req.model_dump()) == req


class TestImage2ContextRequest:
    def test_required_fields(self):
        req = Image2ContextRequest(image_base64="abc123==")
        assert req.image_base64 == "abc123=="

    def test_media_type_defaults_to_jpeg(self):
        req = Image2ContextRequest(image_base64="data")
        assert req.media_type == "image/jpeg"

    def test_profile_defaults_to_image2context(self):
        req = Image2ContextRequest(image_base64="data")
        assert req.profile == "image2context"

    def test_custom_media_type(self):
        req = Image2ContextRequest(image_base64="data", media_type="image/png")
        assert req.media_type == "image/png"

    def test_custom_profile(self):
        req = Image2ContextRequest(image_base64="data", profile="custom")
        assert req.profile == "custom"

    def test_missing_image_base64_raises(self):
        with pytest.raises(ValidationError):
            Image2ContextRequest()  # type: ignore[call-arg]

    def test_round_trip(self):
        req = Image2ContextRequest(image_base64="abc==", media_type="image/png")
        assert Image2ContextRequest.model_validate(req.model_dump()) == req
