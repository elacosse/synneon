import pytest
from pydantic import ValidationError

from synneon.schemas.context import (
    CommunicativeMotive,
    EmotionContext,
    EmotionLabel,
    ParsedVisualContext,
    PragmaticContext,
    SocialAudience,
    SteeringContext,
)
from synneon.schemas.latents import EmotionLatentConfig, LatentThreshold


class TestEmotionLabel:
    def test_all_values(self):
        assert EmotionLabel.NEUTRAL == "neutral"
        assert EmotionLabel.JOY == "joy"
        assert EmotionLabel.FRUSTRATION == "frustration"
        assert EmotionLabel.URGENCY == "urgency"
        assert EmotionLabel.FATIGUE == "fatigue"
        # Plutchik primaries
        assert EmotionLabel.HAPPY == "happy"
        assert EmotionLabel.TRUST == "trust"
        assert EmotionLabel.FEAR == "fear"
        assert EmotionLabel.SURPRISE == "surprise"
        assert EmotionLabel.SAD == "sad"
        assert EmotionLabel.DISGUST == "disgust"
        assert EmotionLabel.ANGER == "anger"
        assert EmotionLabel.ANTICIPATION == "anticipation"
        # Sub-emotions
        assert EmotionLabel.JOYFUL == "joyful"
        assert EmotionLabel.FRUSTRATED == "frustrated"
        assert EmotionLabel.EAGER == "eager"


class TestEmotionContext:
    def test_defaults(self):
        ctx = EmotionContext()
        assert ctx.label == EmotionLabel.NEUTRAL
        assert ctx.intensity == 0.0

    def test_intensity_bounds_valid(self):
        EmotionContext(intensity=0.0)
        EmotionContext(intensity=1.0)
        EmotionContext(intensity=0.5)

    def test_intensity_below_zero_raises(self):
        with pytest.raises(ValidationError):
            EmotionContext(intensity=-0.01)

    def test_intensity_above_one_raises(self):
        with pytest.raises(ValidationError):
            EmotionContext(intensity=1.01)

    @pytest.mark.parametrize(
        "intensity, expected",
        [
            (0.0, "no"),
            (0.2, "no"),  # boundary: not > 0.2
            (0.21, "slight"),
            (0.4, "slight"),  # boundary: not > 0.4
            (0.41, "mild"),
            (0.6, "mild"),  # boundary: not > 0.6
            (0.61, "moderate"),
            (0.8, "moderate"),  # boundary: not > 0.8
            (0.81, "extreme"),
            (1.0, "extreme"),
        ],
    )
    def test_intensity_qualifier(self, intensity, expected):
        assert EmotionContext(intensity=intensity).intensity_qualifier == expected

    def test_str_includes_qualifier_and_label(self):
        ctx = EmotionContext(label=EmotionLabel.JOY, intensity=0.9)
        assert str(ctx) == "extreme joy"

    def test_str_neutral_no_intensity(self):
        ctx = EmotionContext()
        assert str(ctx) == "no neutral"


class TestParsedVisualContext:
    def test_defaults(self):
        ctx = ParsedVisualContext()
        assert ctx.focused_object is None
        assert ctx.detected_objects == []

    def test_with_focused_object(self):
        ctx = ParsedVisualContext(focused_object="water bottle")
        assert ctx.focused_object == "water bottle"

    def test_with_detected_objects(self):
        ctx = ParsedVisualContext(detected_objects=["chair", "table"])
        assert ctx.detected_objects == ["chair", "table"]

    def test_str_empty(self):
        assert str(ParsedVisualContext()) == "nothing specific in view"

    def test_str_focused_only(self):
        ctx = ParsedVisualContext(focused_object="cup")
        assert str(ctx) == "focusing on cup"

    def test_str_detected_only(self):
        ctx = ParsedVisualContext(detected_objects=["book", "pen"])
        assert str(ctx) == "nearby objects: book, pen"

    def test_str_both(self):
        ctx = ParsedVisualContext(focused_object="cup", detected_objects=["book"])
        assert str(ctx) == "focusing on cup; nearby objects: book"


class TestCommunicativeMotive:
    def test_all_values(self):
        assert CommunicativeMotive.DECLARATIVE == "declarative"
        assert CommunicativeMotive.IMPERATIVE == "imperative"
        assert CommunicativeMotive.INTERROGATIVE == "interrogative"
        assert CommunicativeMotive.SOCIAL == "social"


class TestSocialAudience:
    def test_all_values(self):
        assert SocialAudience.FAMILY == "family"
        assert SocialAudience.MEDICAL == "medical"
        assert SocialAudience.STRANGER == "stranger"
        assert SocialAudience.UNKNOWN == "unknown"


class TestPragmaticContext:
    def test_defaults(self):
        ctx = PragmaticContext()
        assert ctx.intent == CommunicativeMotive.DECLARATIVE
        assert ctx.audience == SocialAudience.UNKNOWN
        assert ctx.physical_setting is None

    def test_str_without_setting(self):
        ctx = PragmaticContext()
        assert str(ctx) == "intent: declarative, audience: unknown"

    def test_str_with_setting(self):
        ctx = PragmaticContext(
            intent=CommunicativeMotive.IMPERATIVE,
            audience=SocialAudience.MEDICAL,
            physical_setting="hospital",
        )
        assert str(ctx) == "intent: imperative, audience: medical, setting: hospital"

    def test_custom_values(self):
        ctx = PragmaticContext(
            intent=CommunicativeMotive.INTERROGATIVE,
            audience=SocialAudience.FAMILY,
        )
        assert ctx.intent == CommunicativeMotive.INTERROGATIVE
        assert ctx.audience == SocialAudience.FAMILY


class TestSteeringContext:
    def test_defaults(self):
        ctx = SteeringContext()
        assert isinstance(ctx.emotion, EmotionContext)
        assert isinstance(ctx.parsed_visual, ParsedVisualContext)
        assert isinstance(ctx.pragmatic, PragmaticContext)

    def test_default_submodels_are_independent_instances(self):
        a = SteeringContext()
        b = SteeringContext()
        assert a.emotion is not b.emotion
        assert a.parsed_visual is not b.parsed_visual

    def test_str_format(self):
        ctx = SteeringContext()
        s = str(ctx)
        assert s.startswith("Emotion:")
        assert "Visual:" in s
        assert "Pragmatic:" in s

    def test_round_trip(self):
        ctx = SteeringContext(
            emotion=EmotionContext(label=EmotionLabel.URGENCY, intensity=0.75),
            parsed_visual=ParsedVisualContext(focused_object="door"),
            pragmatic=PragmaticContext(
                intent=CommunicativeMotive.IMPERATIVE,
                audience=SocialAudience.STRANGER,
                physical_setting="lobby",
            ),
        )
        assert SteeringContext.model_validate(ctx.model_dump()) == ctx


def _make_emotion_latent_config() -> EmotionLatentConfig:
    return EmotionLatentConfig(
        thresholds=[
            LatentThreshold(min=0.05, max=0.2, adjective="slightly"),
            LatentThreshold(min=0.2, max=0.4, adjective="somewhat"),
            LatentThreshold(min=0.4, max=0.6, adjective="moderately"),
            LatentThreshold(min=0.6, max=0.8, adjective="very"),
            LatentThreshold(min=0.8, max=1.0, adjective="extremely"),
        ]
    )


class TestEmotionContextWithConfig:
    @pytest.mark.parametrize(
        "intensity, expected",
        [
            (0.0, "no"),  # below all thresholds → fallback to hardcoded
            (0.04, "no"),  # below first threshold → fallback
            (0.05, "slightly"),
            (0.1, "slightly"),
            (0.2, "slightly"),
            (0.21, "somewhat"),
            (0.4, "somewhat"),
            (0.41, "moderately"),
            (0.6, "moderately"),
            (0.61, "very"),
            (0.8, "very"),
            (0.81, "extremely"),
            (1.0, "extremely"),
        ],
    )
    def test_intensity_qualifier_from_config(self, intensity, expected):
        cfg = _make_emotion_latent_config()
        ctx = EmotionContext(intensity=intensity)
        assert ctx.intensity_qualifier_from_config(cfg) == expected

    def test_tts_style_renders_label(self):
        cfg = _make_emotion_latent_config()
        ctx = EmotionContext(label=EmotionLabel.HAPPY, intensity=0.7)
        assert ctx.tts_style(cfg) == "Speak in a very happy tone."

    def test_tts_style_with_sub_emotion(self):
        cfg = _make_emotion_latent_config()
        ctx = EmotionContext(label=EmotionLabel.JOYFUL, intensity=0.5)
        assert ctx.tts_style(cfg) == "Speak in a moderately joyful tone."

    def test_tts_style_below_threshold_returns_empty(self):
        cfg = _make_emotion_latent_config()
        ctx = EmotionContext(label=EmotionLabel.HAPPY, intensity=0.0)
        assert ctx.tts_style(cfg) == ""

    def test_tts_style_just_below_min_threshold_returns_empty(self):
        cfg = _make_emotion_latent_config()
        ctx = EmotionContext(label=EmotionLabel.HAPPY, intensity=0.04)
        assert ctx.tts_style(cfg) == ""
