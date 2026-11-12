from synneon.schemas.latents import LatentConfig
from synneon.services.latents.loader import load_latent_config


class TestLoadLatentConfig:
    def test_loads_default(self):
        cfg = load_latent_config("default")
        assert isinstance(cfg, LatentConfig)

    def test_has_five_thresholds(self):
        cfg = load_latent_config("default")
        assert len(cfg.emotion.thresholds) == 5

    def test_threshold_adjectives(self):
        cfg = load_latent_config("default")
        adjectives = [t.adjective for t in cfg.emotion.thresholds]
        assert adjectives == ["slightly", "somewhat", "moderately", "very", "extremely"]

    def test_tts_style_template(self):
        cfg = load_latent_config("default")
        assert "{emotion}" in cfg.emotion.tts_style
        assert "{adjective}" in cfg.emotion.tts_style

    def test_render_tts_style(self):
        cfg = load_latent_config("default")
        result = cfg.emotion.render_tts_style("happy", 0.5)
        assert result == "Speak in a moderately happy tone."

    def test_lru_cache_returns_same_object(self):
        a = load_latent_config("default")
        b = load_latent_config("default")
        assert a is b
