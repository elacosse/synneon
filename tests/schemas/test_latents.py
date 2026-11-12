import pytest

from synneon.schemas.latents import EmotionLatentConfig, LatentConfig, LatentThreshold


def _default_thresholds() -> list[LatentThreshold]:
    return [
        LatentThreshold(min=0.05, max=0.2, adjective="slightly"),
        LatentThreshold(min=0.2, max=0.4, adjective="somewhat"),
        LatentThreshold(min=0.4, max=0.6, adjective="moderately"),
        LatentThreshold(min=0.6, max=0.8, adjective="very"),
        LatentThreshold(min=0.8, max=1.0, adjective="extremely"),
    ]


class TestLatentThreshold:
    def test_construction(self):
        t = LatentThreshold(min=0.0, max=0.5, adjective="mildly")
        assert t.min == 0.0
        assert t.max == 0.5
        assert t.adjective == "mildly"


class TestEmotionLatentConfig:
    @pytest.mark.parametrize(
        "intensity, expected",
        [
            (0.0, ""),  # below all thresholds
            (0.04, ""),
            (0.05, "slightly"),
            (0.2, "slightly"),  # inclusive boundary
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
    def test_resolve_adjective(self, intensity, expected):
        cfg = EmotionLatentConfig(thresholds=_default_thresholds())
        assert cfg.resolve_adjective(intensity) == expected

    def test_resolve_adjective_empty_thresholds(self):
        cfg = EmotionLatentConfig()
        assert cfg.resolve_adjective(0.5) == ""

    def test_render_tts_style(self):
        cfg = EmotionLatentConfig(thresholds=_default_thresholds())
        assert cfg.render_tts_style("joy", 0.7) == "Speak in a very joy tone."
        assert cfg.render_tts_style("happy", 0.1) == "Speak in a slightly happy tone."

    def test_render_tts_style_below_thresholds_is_empty(self):
        cfg = EmotionLatentConfig(thresholds=_default_thresholds())
        assert cfg.render_tts_style("happy", 0.0) == ""

    def test_render_tts_style_custom_template(self):
        cfg = EmotionLatentConfig(thresholds=_default_thresholds(), tts_style="{adjective}/{emotion}")
        assert cfg.render_tts_style("sad", 0.9) == "extremely/sad"


class TestLatentConfig:
    def test_defaults(self):
        cfg = LatentConfig()
        assert isinstance(cfg.emotion, EmotionLatentConfig)
        assert cfg.emotion.thresholds == []

    def test_with_emotion(self):
        emotion_cfg = EmotionLatentConfig(thresholds=_default_thresholds())
        cfg = LatentConfig(emotion=emotion_cfg)
        assert len(cfg.emotion.thresholds) == 5
