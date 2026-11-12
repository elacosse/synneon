from __future__ import annotations

import os
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

REDIS_URL_ENV = "SYNNEON_REDIS_URL"


class UserProfile(BaseModel):
    id: str
    name: str
    created_at: float = Field(default_factory=time.time)


class ConversationEntry(BaseModel):
    user_id: str
    session_id: str
    role: str = "other"  # "other" (ambient/nearby speaker) or "user" (wearer)
    text: str
    timestamp: float = Field(default_factory=time.time)
    source: str = "stt"


class MemoryType(StrEnum):
    VLM_IMAGE_CONTEXT = "vlm_image_context"  # image2context VLM result
    VLM_NEON_CONTEXT = "vlm_neon_context"  # neon-context VLM result (video + gaze + audio)
    TRANSCRIPTION = "transcription"
    NOTE = "note"


class InferenceType(StrEnum):
    TEXT2TEXT = "text2text"  # abbreviation expansion (text in, text out)
    IMAGE2TEXT = "image2text"  # VLM inference (image/video/audio in, text out)


class MemoryEntry(BaseModel):
    id: str
    user_id: str
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    type: MemoryType
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    media_refs: list[str] = Field(default_factory=list)


class InferenceRecord(BaseModel):
    id: str
    user_id: str
    session_id: str
    timestamp: float = Field(default_factory=time.time)
    inference_type: InferenceType  # drives Redis key namespace
    endpoint: str  # "expand" | "image2context" | "neon-context"
    profile: str
    agent_config: dict[str, Any] = Field(default_factory=dict)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)  # full prompt sent to LLM
    result_snapshot: dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    redis_url: str = "redis://localhost:6379/0"
    conversation_window: int = 20
    memory_window: int = 5
    inference_window: int = 50
    conversation_ttl_days: int = 7
    user_id: str = "default"

    def with_env_overrides(self) -> MemoryConfig:
        """Let SYNNEON_REDIS_URL replace `redis_url`, so a container can reach Redis by service name."""
        url = os.environ.get(REDIS_URL_ENV)
        return self.model_copy(update={"redis_url": url}) if url else self
