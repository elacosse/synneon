from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from synneon.schemas.latents import EmotionLatentConfig


# ==========================================
# 1. Emotional Context (Abstract Latents)
# ==========================================
# Drives: Activation Steering (Vector Injection)
# Why: Changes the "vibe" or "feeling" of the generation without changing the topic.
class EmotionLabel(StrEnum):
    # Baseline
    NEUTRAL = "neutral"
    # Legacy values (kept for backward compat)
    JOY = "joy"
    FRUSTRATION = "frustration"
    URGENCY = "urgency"
    FATIGUE = "fatigue"
    # 8 Plutchik primaries (match React EMOTION_WHEEL id values)
    HAPPY = "happy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SAD = "sad"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    # Sub-emotions (match React EMOTION_WHEEL subs, lowercased)
    JOYFUL = "joyful"
    CONFIDENT = "confident"
    PEACEFUL = "peaceful"
    HOPEFUL = "hopeful"
    CAUTIOUS = "cautious"
    ANXIOUS = "anxious"
    AMAZED = "amazed"
    CONFUSED = "confused"
    UNHAPPY = "unhappy"
    HURT = "hurt"
    DISAPPROVAL = "disapproval"
    DISLIKE = "dislike"
    CRITICAL = "critical"
    FRUSTRATED = "frustrated"
    INTERESTED = "interested"
    EAGER = "eager"


class EmotionContext(BaseModel):
    label: EmotionLabel = EmotionLabel.NEUTRAL
    # The "Volume Knob" for the vector injection (0.0 to 1.0)
    intensity: float = Field(0.0, ge=0.0, le=1.0)

    @property
    def intensity_qualifier(self) -> str:
        """
        Converts numeric intensity into a qualitative adverb for UI/prompting.
        """
        if self.intensity > 0.8:
            return "extreme"
        if self.intensity > 0.6:
            return "moderate"
        if self.intensity > 0.4:
            return "mild"
        if self.intensity > 0.2:
            return "slight"
        return "no"

    def intensity_qualifier_from_config(self, config: EmotionLatentConfig) -> str:
        """Config-driven adjective lookup. Falls back to hardcoded property if no band matches."""
        adjective = config.resolve_adjective(self.intensity)
        return adjective if adjective else self.intensity_qualifier

    def tts_style(self, config: EmotionLatentConfig) -> str:
        """Render a plain-language TTS delivery hint for this emotion using the latent config template.

        Returns an empty string if the intensity falls below all defined thresholds.
        """
        return config.render_tts_style(self.label.value, self.intensity)

    def __str__(self) -> str:
        return f"{self.intensity_qualifier} {self.label.value}"


# ==========================================
# 2. ParsedVisual Context (Semantic Grounding)
# ==========================================
# Drives: Prompt Engineering (Text Injection)
# Why: Tells the model *what* physical objects exist to be talked about.
class ParsedVisualContext(BaseModel):
    # The specific object the user is fixating on right now (Gaze Dwell)
    focused_object: str | None = None

    # Other objects detected in the camera frame (Peripheral Awareness)
    detected_objects: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        parts = []
        if self.focused_object:
            parts.append(f"focusing on {self.focused_object}")
        if self.detected_objects:
            parts.append(f"nearby objects: {', '.join(self.detected_objects)}")
        return "; ".join(parts) if parts else "nothing specific in view"


# ==========================================
# 3. Pragmatic Context (Communication Motives)
# ==========================================
# Drives: System Instruction & Style Constraints
# Why: Distinguishes "I want water" (Imperative) from "The water is cold" (Declarative).
class CommunicativeMotive(StrEnum):
    DECLARATIVE = "declarative"  # Stating a fact / Storytelling
    IMPERATIVE = "imperative"  # Requesting / Commanding
    INTERROGATIVE = "interrogative"  # Asking a question
    SOCIAL = "social"  # Greetings / Phatic communication


class SocialAudience(StrEnum):
    FAMILY = "family"  # Casual, intimate
    MEDICAL = "medical"  # Formal, precise
    STRANGER = "stranger"  # Polite, guarded
    UNKNOWN = "unknown"


class PragmaticContext(BaseModel):
    intent: CommunicativeMotive = CommunicativeMotive.DECLARATIVE
    audience: SocialAudience = SocialAudience.UNKNOWN

    # E.g., "bedroom", "noisy_restaurant"
    physical_setting: str | None = None

    def __str__(self) -> str:
        parts = [f"intent: {self.intent.value}", f"audience: {self.audience.value}"]
        if self.physical_setting:
            parts.append(f"setting: {self.physical_setting}")
        return ", ".join(parts)


# ==========================================
# The Root Context Object
# ==========================================
class SteeringContext(BaseModel):
    """
    The simplified 'State of the World' snapshot.
    Passed to the Inference Engine to align the LLM.
    """

    emotion: EmotionContext = Field(default_factory=EmotionContext)
    parsed_visual: ParsedVisualContext = Field(default_factory=ParsedVisualContext)
    pragmatic: PragmaticContext = Field(default_factory=PragmaticContext)

    def __str__(self) -> str:
        return f"Emotion: {self.emotion}; Visual: {self.parsed_visual}; Pragmatic: {self.pragmatic}"
