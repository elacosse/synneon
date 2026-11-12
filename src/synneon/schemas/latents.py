from pydantic import BaseModel


class LatentThreshold(BaseModel):
    min: float
    max: float
    adjective: str


class EmotionLatentConfig(BaseModel):
    min: float = 0.0
    max: float = 1.0
    thresholds: list[LatentThreshold] = []
    tts_style: str = "Speak in a {adjective} {emotion} tone."

    def resolve_adjective(self, intensity: float) -> str:
        """Return the first matching threshold adjective, or '' if below all thresholds."""
        for t in self.thresholds:
            if t.min <= intensity <= t.max:
                return t.adjective
        return ""

    def render_tts_style(self, emotion_label: str, intensity: float) -> str:
        """Render the tts_style delivery hint, or '' if the intensity is below all thresholds."""
        adjective = self.resolve_adjective(intensity)
        if not adjective:
            return ""
        return self.tts_style.format(emotion=emotion_label, adjective=adjective)


class LatentConfig(BaseModel):
    emotion: EmotionLatentConfig = EmotionLatentConfig()
