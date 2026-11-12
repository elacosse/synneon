from __future__ import annotations

import json
import time

from redis.asyncio import Redis

from synneon.schemas.memory import (
    ConversationEntry,
    InferenceRecord,
    InferenceType,
    MemoryConfig,
    MemoryEntry,
    UserProfile,
)

_DEFAULT_USER_ID = "default"


def _user_key(user_id: str, *parts: str) -> str:
    return ":".join(["user", user_id, *parts])


class ContextStore:
    def __init__(self, redis: Redis, config: MemoryConfig):
        self._redis = redis
        self._config = config
        self._user_id = config.user_id

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def bootstrap(self) -> None:
        """Ensure the default UserProfile and self SpeakerProfile exist."""
        await self.get_or_create_user(self._user_id)

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

    async def get_or_create_user(self, user_id: str = _DEFAULT_USER_ID) -> UserProfile:
        key = _user_key(user_id, "profile")
        raw = await self._redis.get(key)
        if raw:
            return UserProfile.model_validate_json(raw)

        profile = UserProfile(id=user_id, name="User")
        await self._redis.set(key, profile.model_dump_json())
        await self._redis.zadd("users", {user_id: profile.created_at})
        return profile

    async def update_user(self, profile: UserProfile) -> None:
        key = _user_key(profile.id, "profile")
        await self._redis.set(key, profile.model_dump_json())

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def start_session(self, session_id: str, device_address: str = "") -> None:
        now = time.time()
        session_meta = {
            "session_id": session_id,
            "started_at": now,
            "device_address": device_address,
        }
        current_key = _user_key(self._user_id, "current_session")
        meta_key = _user_key(self._user_id, "session", session_id, "meta")

        await self._redis.set(current_key, json.dumps(session_meta))
        await self._redis.hset(meta_key, mapping={"started_at": now, "device_address": device_address})

    async def end_session(self) -> None:
        current_key = _user_key(self._user_id, "current_session")
        raw = await self._redis.get(current_key)
        if not raw:
            return

        data = json.loads(raw)
        session_id = data.get("session_id")
        if session_id:
            meta_key = _user_key(self._user_id, "session", session_id, "meta")
            await self._redis.hset(meta_key, "ended_at", time.time())

    async def get_current_session_id(self) -> str | None:
        current_key = _user_key(self._user_id, "current_session")
        raw = await self._redis.get(current_key)
        if not raw:
            return None
        return json.loads(raw).get("session_id")

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    async def save_utterance(self, entry: ConversationEntry) -> None:
        conv_key = _user_key(self._user_id, "session", entry.session_id, "conversation")
        ttl_secs = self._config.conversation_ttl_days * 86400
        payload = entry.model_dump_json()
        await self._redis.zadd(conv_key, {payload: entry.timestamp})
        await self._redis.expire(conv_key, ttl_secs)

    async def get_recent_conversation(self, session_id: str, limit: int | None = None) -> list[ConversationEntry]:
        if limit is None:
            limit = self._config.conversation_window
        conv_key = _user_key(self._user_id, "session", session_id, "conversation")
        # ZRANGE with REV returns highest score (latest) first; we want oldest→newest for context
        raw_items = await self._redis.zrange(conv_key, 0, limit - 1, desc=True)
        entries = [ConversationEntry.model_validate_json(item) for item in raw_items]
        return list(reversed(entries))  # restore chronological order

    # ------------------------------------------------------------------
    # Memories (cross-session)
    # ------------------------------------------------------------------

    async def save_memory(self, entry: MemoryEntry) -> None:
        memory_key = _user_key(self._user_id, "memory", entry.id)
        memories_key = _user_key(self._user_id, "memories")
        await self._redis.set(memory_key, entry.model_dump_json())
        await self._redis.zadd(memories_key, {entry.id: entry.timestamp})

    async def get_recent_memories(self, limit: int | None = None) -> list[MemoryEntry]:
        if limit is None:
            limit = self._config.memory_window
        memories_key = _user_key(self._user_id, "memories")
        memory_ids = await self._redis.zrange(memories_key, 0, limit - 1, desc=True)
        entries = []
        for mid in memory_ids:
            memory_key = _user_key(self._user_id, "memory", mid)
            raw = await self._redis.get(memory_key)
            if raw:
                try:
                    entries.append(MemoryEntry.model_validate_json(raw))
                except Exception:
                    pass  # skip entries saved with old/unknown schema
        return list(reversed(entries))  # chronological order

    # ------------------------------------------------------------------
    # Inference records
    # ------------------------------------------------------------------

    async def save_inference_record(self, record: InferenceRecord) -> None:
        inference_key = _user_key(self._user_id, "inference", record.inference_type, record.id)
        inferences_key = _user_key(self._user_id, "inferences", record.inference_type)
        await self._redis.set(inference_key, record.model_dump_json())
        await self._redis.zadd(inferences_key, {record.id: record.timestamp})

    async def _get_inferences_for_type(self, inference_type: InferenceType, limit: int) -> list[InferenceRecord]:
        inferences_key = _user_key(self._user_id, "inferences", inference_type)
        record_ids = await self._redis.zrange(inferences_key, 0, limit - 1, desc=True)
        entries = []
        for rid in record_ids:
            inference_key = _user_key(self._user_id, "inference", inference_type, rid)
            raw = await self._redis.get(inference_key)
            if raw:
                entries.append(InferenceRecord.model_validate_json(raw))
        return list(reversed(entries))  # chronological order

    async def get_recent_inferences(
        self, limit: int | None = None, inference_type: InferenceType | None = None
    ) -> list[InferenceRecord]:
        if limit is None:
            limit = self._config.inference_window
        if inference_type is not None:
            return await self._get_inferences_for_type(inference_type, limit)
        # Merge both types, sort by timestamp, return most recent `limit` entries
        text2text = await self._get_inferences_for_type(InferenceType.TEXT2TEXT, limit)
        image2text = await self._get_inferences_for_type(InferenceType.IMAGE2TEXT, limit)
        merged = sorted(text2text + image2text, key=lambda r: r.timestamp)
        return merged[-limit:] if len(merged) > limit else merged

    async def get_inference_record(self, record_id: str, inference_type: InferenceType) -> InferenceRecord | None:
        inference_key = _user_key(self._user_id, "inference", inference_type, record_id)
        raw = await self._redis.get(inference_key)
        return InferenceRecord.model_validate_json(raw) if raw else None

    # ------------------------------------------------------------------
    # Last expand context (dedicated key for fast widget retrieval)
    # ------------------------------------------------------------------

    async def save_last_expand_context(self, session_id: str, data: dict) -> None:
        key = _user_key(self._user_id, f"session:{session_id}:last_expand_context")
        await self._redis.set(key, json.dumps(data))

    async def get_last_expand_context(self, user_id: str, session_id: str) -> dict | None:
        key = _user_key(user_id, f"session:{session_id}:last_expand_context")
        raw = await self._redis.get(key)
        return json.loads(raw) if raw else None


# ------------------------------------------------------------------
# Formatting helpers for LLM injection
# ------------------------------------------------------------------


def format_conversation(entries: list[ConversationEntry]) -> str:
    if not entries:
        return ""
    lines = []
    for entry in entries:
        label = "User" if entry.role == "user" else "Other"
        lines.append(f"{label}: {entry.text}")
    return "\n".join(lines)


def format_memories(entries: list[MemoryEntry]) -> str:
    lines = [f"• {entry.summary}" for entry in entries if entry.summary.strip()]
    return "\n".join(lines)


def derive_summary(result: object) -> str:
    """Extract a human-readable summary from any inference result object."""
    # Transcription / plain-text results
    text = getattr(result, "text", None)
    if text and isinstance(text, str):
        return text[:200]

    # VLM results: Image2ContextResult / ParsedVisualContext / PragmaticContext
    parts = []
    parsed_visual = getattr(result, "parsed_visual", None)
    pragmatic = getattr(result, "pragmatic", None)

    if parsed_visual:
        focused = getattr(parsed_visual, "focused_object", None)
        if focused:
            parts.append(f"Focus: {focused}")
        detected = getattr(parsed_visual, "detected_objects", [])
        if detected:
            parts.append(f"Objects: {', '.join(detected[:5])}")

    if pragmatic:
        setting = getattr(pragmatic, "physical_setting", None)
        intent = getattr(pragmatic, "intent", None)
        audience = getattr(pragmatic, "audience", None)
        if setting:
            parts.append(f"Setting: {setting}")
        if intent:
            parts.append(f"Intent: {intent}")
        if audience:
            parts.append(f"Audience: {audience}")

    if parts:
        return "; ".join(parts)

    # Generic fallback
    return str(result)[:200]
