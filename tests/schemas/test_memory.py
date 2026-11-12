"""Tests for schemas/memory.py."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from synneon.schemas.memory import (
    ConversationEntry,
    InferenceRecord,
    InferenceType,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
    UserProfile,
)

# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


class TestUserProfile:
    def test_minimal_construction(self):
        profile = UserProfile(id="u1", name="Alice")
        assert profile.id == "u1"
        assert profile.name == "Alice"

    def test_created_at_defaults_to_now(self):
        before = time.time()
        profile = UserProfile(id="u1", name="Alice")
        after = time.time()
        assert before <= profile.created_at <= after

    def test_explicit_created_at(self):
        ts = 1_000_000.0
        profile = UserProfile(id="u1", name="Alice", created_at=ts)
        assert profile.created_at == ts

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            UserProfile(name="Alice")

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            UserProfile(id="u1")

    def test_round_trip_json(self):
        profile = UserProfile(id="u1", name="Alice")
        assert UserProfile.model_validate_json(profile.model_dump_json()) == profile


# ---------------------------------------------------------------------------
# ConversationEntry
# ---------------------------------------------------------------------------


class TestConversationEntry:
    def test_minimal_construction(self):
        entry = ConversationEntry(user_id="u1", session_id="sess1", text="Hello")
        assert entry.role == "other"
        assert entry.source == "stt"

    def test_role_defaults_to_other(self):
        entry = ConversationEntry(user_id="u1", session_id="sess1", text="Hi")
        assert entry.role == "other"

    def test_role_can_be_user(self):
        entry = ConversationEntry(user_id="u1", session_id="sess1", role="user", text="Hi")
        assert entry.role == "user"

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        entry = ConversationEntry(user_id="u1", session_id="sess1", text="Hi")
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_custom_source(self):
        entry = ConversationEntry(user_id="u1", session_id="sess1", text="Hi", source="manual")
        assert entry.source == "manual"

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            ConversationEntry(user_id="u1", session_id="sess1")

    def test_round_trip_json(self):
        entry = ConversationEntry(
            user_id="u1",
            session_id="sess1",
            role="other",
            text="Hello there",
        )
        assert ConversationEntry.model_validate_json(entry.model_dump_json()) == entry


# ---------------------------------------------------------------------------
# MemoryType
# ---------------------------------------------------------------------------


class TestMemoryType:
    def test_enum_values(self):
        assert MemoryType.VLM_IMAGE_CONTEXT == "vlm_image_context"
        assert MemoryType.VLM_NEON_CONTEXT == "vlm_neon_context"
        assert MemoryType.TRANSCRIPTION == "transcription"
        assert MemoryType.NOTE == "note"

    def test_all_members_present(self):
        names = {m.value for m in MemoryType}
        assert names == {"vlm_image_context", "vlm_neon_context", "transcription", "note"}


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------


class TestMemoryEntry:
    def test_minimal_construction(self):
        entry = MemoryEntry(
            id="m1",
            user_id="u1",
            session_id="sess1",
            type=MemoryType.NOTE,
            summary="A note",
        )
        assert entry.data == {}
        assert entry.media_refs == []

    def test_data_and_media_refs(self):
        entry = MemoryEntry(
            id="m1",
            user_id="u1",
            session_id="sess1",
            type=MemoryType.VLM_NEON_CONTEXT,
            summary="Video context",
            data={"key": "value"},
            media_refs=["ref1", "ref2"],
        )
        assert entry.data == {"key": "value"}
        assert entry.media_refs == ["ref1", "ref2"]

    def test_type_accepts_string_enum(self):
        entry = MemoryEntry(
            id="m1",
            user_id="u1",
            session_id="sess1",
            type="transcription",
            summary="...",
        )
        assert entry.type == MemoryType.TRANSCRIPTION

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            MemoryEntry(
                id="m1",
                user_id="u1",
                session_id="sess1",
                type="bogus_type",
                summary="...",
            )

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            MemoryEntry(id="m1", user_id="u1")

    def test_round_trip_json(self):
        entry = MemoryEntry(
            id="m1",
            user_id="u1",
            session_id="sess1",
            type=MemoryType.VLM_IMAGE_CONTEXT,
            summary="Focus: screen; Objects: keyboard",
            data={"score": 0.9},
        )
        assert MemoryEntry.model_validate_json(entry.model_dump_json()) == entry


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------


class TestInferenceRecord:
    def test_minimal_construction(self):
        record = InferenceRecord(
            id="r1",
            user_id="u1",
            session_id="sess1",
            inference_type=InferenceType.TEXT2TEXT,
            endpoint="expand",
            profile="default",
        )
        assert record.agent_config == {}
        assert record.context_snapshot == {}
        assert record.result_snapshot == {}

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        record = InferenceRecord(
            id="r1",
            user_id="u1",
            session_id="s1",
            inference_type=InferenceType.TEXT2TEXT,
            endpoint="expand",
            profile="default",
        )
        after = time.time()
        assert before <= record.timestamp <= after

    def test_stores_agent_config(self):
        record = InferenceRecord(
            id="r1",
            user_id="u1",
            session_id="s1",
            inference_type=InferenceType.TEXT2TEXT,
            endpoint="expand",
            profile="default",
            agent_config={"model": "google-gla:gemini-flash", "system_prompt": "You are helpful"},
        )
        assert record.agent_config["model"] == "google-gla:gemini-flash"
        assert record.agent_config["system_prompt"] == "You are helpful"

    def test_stores_context_and_result_snapshots(self):
        record = InferenceRecord(
            id="r1",
            user_id="u1",
            session_id="s1",
            inference_type=InferenceType.IMAGE2TEXT,
            endpoint="image2context",
            profile="image2context",
            context_snapshot={"media_type": "image/jpeg"},
            result_snapshot={"parsed_visual": {"focused_object": "laptop"}},
        )
        assert record.context_snapshot["media_type"] == "image/jpeg"
        assert record.result_snapshot["parsed_visual"]["focused_object"] == "laptop"

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            InferenceRecord(user_id="u1", session_id="s1")

    def test_round_trip_json(self):
        record = InferenceRecord(
            id="r1",
            user_id="u1",
            session_id="sess1",
            inference_type=InferenceType.TEXT2TEXT,
            endpoint="expand",
            profile="default",
            agent_config={"model": "gpt-4", "provider": "openai"},
            context_snapshot={"tokens": "h-w", "conversation_history": "Other: Hello"},
            result_snapshot={"candidates": ["hello world"]},
        )
        assert InferenceRecord.model_validate_json(record.model_dump_json()) == record


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.redis_url == "redis://localhost:6379/0"
        assert cfg.conversation_window == 20
        assert cfg.memory_window == 5
        assert cfg.inference_window == 50
        assert cfg.conversation_ttl_days == 7
        assert cfg.user_id == "default"

    def test_custom_values(self):
        cfg = MemoryConfig(
            redis_url="redis://myhost:6380/1",
            conversation_window=50,
            memory_window=10,
            conversation_ttl_days=14,
            user_id="alice",
        )
        assert cfg.redis_url == "redis://myhost:6380/1"
        assert cfg.conversation_window == 50
        assert cfg.memory_window == 10
        assert cfg.conversation_ttl_days == 14
        assert cfg.user_id == "alice"

    def test_round_trip(self):
        cfg = MemoryConfig(user_id="bob", conversation_window=30)
        assert MemoryConfig.model_validate(cfg.model_dump()) == cfg


class TestMemoryConfigEnvOverrides:
    def test_env_var_replaces_redis_url(self, monkeypatch):
        from synneon.schemas.memory import MemoryConfig

        monkeypatch.setenv("SYNNEON_REDIS_URL", "redis://redis:6379/0")
        cfg = MemoryConfig(redis_url="redis://localhost:6379/0", user_id="alice").with_env_overrides()
        assert cfg.redis_url == "redis://redis:6379/0"
        assert cfg.user_id == "alice"

    def test_unset_env_var_keeps_configured_url(self, monkeypatch):
        from synneon.schemas.memory import MemoryConfig

        monkeypatch.delenv("SYNNEON_REDIS_URL", raising=False)
        cfg = MemoryConfig(redis_url="redis://myhost:6380/1").with_env_overrides()
        assert cfg.redis_url == "redis://myhost:6380/1"

    def test_loaders_apply_the_override(self, monkeypatch, tmp_path):
        from synneon.api import config
        from synneon.apps.run_example import load_memory_config

        toml = tmp_path / "memory.toml"
        toml.write_text('[memory.default]\nredis_url = "redis://localhost:6379/0"\n')
        monkeypatch.setenv("SYNNEON_REDIS_URL", "redis://redis:6379/0")
        monkeypatch.setattr(config, "MEMORY_CONFIG_PATH", toml)

        assert config.get_memory_settings().redis_url == "redis://redis:6379/0"
        assert load_memory_config(toml).redis_url == "redis://redis:6379/0"
