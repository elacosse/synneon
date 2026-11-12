"""Tests for services/memory/store.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from synneon.schemas.memory import (
    ConversationEntry,
    InferenceRecord,
    InferenceType,
    MemoryConfig,
    MemoryEntry,
    MemoryType,
    UserProfile,
)
from synneon.services.memory.store import (
    ContextStore,
    _user_key,
    derive_summary,
    format_conversation,
    format_memories,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_redis(*, get=None, zrange=None, hget=None):
    """Build an AsyncMock Redis with sensible defaults."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=get)
    redis.set = AsyncMock()
    redis.zadd = AsyncMock()
    redis.expire = AsyncMock()
    redis.zrange = AsyncMock(return_value=zrange or [])
    redis.hset = AsyncMock()
    redis.hget = AsyncMock(return_value=hget)
    redis.ping = AsyncMock(return_value=True)

    pipe = AsyncMock()
    pipe.zrem = MagicMock()
    pipe.zadd = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


def make_config(**kwargs) -> MemoryConfig:
    defaults = dict(
        redis_url="redis://localhost:6379/0",
        conversation_window=20,
        memory_window=5,
        inference_window=50,
        conversation_ttl_days=7,
        user_id="default",
    )
    defaults.update(kwargs)
    return MemoryConfig(**defaults)


def make_store(redis=None, **config_kwargs) -> ContextStore:
    if redis is None:
        redis = make_redis()
    return ContextStore(redis=redis, config=make_config(**config_kwargs))


# ---------------------------------------------------------------------------
# _user_key helper
# ---------------------------------------------------------------------------


class TestUserKey:
    def test_single_part(self):
        assert _user_key("alice", "profile") == "user:alice:profile"

    def test_multiple_parts(self):
        assert _user_key("alice", "session", "abc", "meta") == "user:alice:session:abc:meta"

    def test_no_extra_parts(self):
        assert _user_key("bob") == "user:bob"


# ---------------------------------------------------------------------------
# UserProfile operations
# ---------------------------------------------------------------------------


class TestGetOrCreateUser:
    async def test_returns_existing_user_from_redis(self):
        profile = UserProfile(id="default", name="User")
        redis = make_redis(get=profile.model_dump_json())
        store = make_store(redis=redis)

        result = await store.get_or_create_user("default")

        assert result.id == "default"
        assert result.name == "User"
        redis.set.assert_not_called()

    async def test_creates_new_user_when_not_in_redis(self):
        redis = make_redis(get=None)
        store = make_store(redis=redis)

        result = await store.get_or_create_user("default")

        assert result.id == "default"
        assert result.name == "User"
        redis.set.assert_called_once()

    async def test_new_user_is_added_to_users_sorted_set(self):
        redis = make_redis(get=None)
        store = make_store(redis=redis)

        await store.get_or_create_user("default")

        zadd_calls = [str(c) for c in redis.zadd.call_args_list]
        assert any("users" in c for c in zadd_calls)

    async def test_bootstrap_calls_get_or_create_user(self):
        redis = make_redis(get=None)
        store = make_store(redis=redis)

        await store.bootstrap()

        redis.set.assert_called()


class TestUpdateUser:
    async def test_update_user_writes_to_redis(self):
        redis = make_redis()
        store = make_store(redis=redis)
        profile = UserProfile(id="default", name="Updated Name")

        await store.update_user(profile)

        redis.set.assert_called_once()
        key_arg = redis.set.call_args[0][0]
        assert "default" in key_arg
        assert "profile" in key_arg


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    async def test_start_session_writes_current_session(self):
        redis = make_redis()
        store = make_store(redis=redis)

        await store.start_session("sess-123", device_address="192.168.1.1")

        set_keys = [c[0][0] for c in redis.set.call_args_list]
        assert any("current_session" in k for k in set_keys)

    async def test_start_session_writes_meta_hash(self):
        redis = make_redis()
        store = make_store(redis=redis)

        await store.start_session("sess-123")

        redis.hset.assert_called()
        hset_key = redis.hset.call_args[0][0]
        assert "sess-123" in hset_key

    async def test_get_current_session_id_returns_id(self):
        session_meta = json.dumps({"session_id": "sess-abc"})
        redis = make_redis(get=session_meta)
        store = make_store(redis=redis)

        result = await store.get_current_session_id()

        assert result == "sess-abc"

    async def test_get_current_session_id_returns_none_when_missing(self):
        redis = make_redis(get=None)
        store = make_store(redis=redis)

        result = await store.get_current_session_id()

        assert result is None

    async def test_end_session_does_nothing_when_no_current_session(self):
        redis = make_redis(get=None)
        store = make_store(redis=redis)

        await store.end_session()  # should not raise

        redis.hset.assert_not_called()

    async def test_end_session_marks_ended_at(self):
        session_meta = json.dumps({"session_id": "sess-abc"})
        redis = make_redis(get=session_meta)
        store = make_store(redis=redis)

        await store.end_session()

        redis.hset.assert_called_once()
        hset_args = redis.hset.call_args[0]
        assert "sess-abc" in hset_args[0]
        assert hset_args[1] == "ended_at"


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


class TestConversation:
    async def test_save_utterance_adds_to_sorted_set(self):
        redis = make_redis()
        store = make_store(redis=redis)
        entry = ConversationEntry(user_id="default", session_id="sess1", text="Hello")

        await store.save_utterance(entry)

        redis.zadd.assert_called_once()
        key = redis.zadd.call_args[0][0]
        assert "sess1" in key
        assert "conversation" in key

    async def test_save_utterance_sets_ttl(self):
        redis = make_redis()
        store = make_store(redis=redis, conversation_ttl_days=7)
        entry = ConversationEntry(user_id="default", session_id="sess1", text="Hello")

        await store.save_utterance(entry)

        redis.expire.assert_called_once()
        ttl_arg = redis.expire.call_args[0][1]
        assert ttl_arg == 7 * 86400

    async def test_get_recent_conversation_returns_entries_chronologically(self):
        entry1 = ConversationEntry(user_id="default", session_id="sess1", text="First", timestamp=1000.0)
        entry2 = ConversationEntry(user_id="default", session_id="sess1", role="user", text="Second", timestamp=2000.0)
        # Redis returns latest first (desc=True), so entry2, entry1
        redis = make_redis(zrange=[entry2.model_dump_json(), entry1.model_dump_json()])
        store = make_store(redis=redis)

        results = await store.get_recent_conversation("sess1")

        assert len(results) == 2
        assert results[0].text == "First"
        assert results[1].text == "Second"

    async def test_get_recent_conversation_uses_config_window(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis, conversation_window=15)

        await store.get_recent_conversation("sess1")

        zrange_kwargs = redis.zrange.call_args
        assert 14 in zrange_kwargs[0] or 14 in zrange_kwargs[1].values()

    async def test_get_recent_conversation_respects_limit_override(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis, conversation_window=20)

        await store.get_recent_conversation("sess1", limit=5)

        zrange_kwargs = redis.zrange.call_args
        assert 4 in zrange_kwargs[0] or 4 in zrange_kwargs[1].values()

    async def test_get_recent_conversation_returns_empty_when_no_entries(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis)

        results = await store.get_recent_conversation("sess1")

        assert results == []


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


class TestMemories:
    async def test_save_memory_stores_entry(self):
        redis = make_redis()
        store = make_store(redis=redis)
        entry = MemoryEntry(
            id="m1",
            user_id="default",
            session_id="sess1",
            type=MemoryType.NOTE,
            summary="A note",
        )

        await store.save_memory(entry)

        redis.set.assert_called_once()
        key_arg = redis.set.call_args[0][0]
        assert "m1" in key_arg

    async def test_save_memory_adds_to_sorted_set(self):
        redis = make_redis()
        store = make_store(redis=redis)
        entry = MemoryEntry(
            id="m1",
            user_id="default",
            session_id="sess1",
            type=MemoryType.NOTE,
            summary="A note",
        )

        await store.save_memory(entry)

        redis.zadd.assert_called_once()
        key_arg = redis.zadd.call_args[0][0]
        assert "memories" in key_arg

    async def test_get_recent_memories_returns_chronologically(self):
        e1 = MemoryEntry(
            id="m1",
            user_id="default",
            session_id="sess1",
            type=MemoryType.NOTE,
            summary="First",
            timestamp=1000.0,
        )
        e2 = MemoryEntry(
            id="m2",
            user_id="default",
            session_id="sess1",
            type=MemoryType.NOTE,
            summary="Second",
            timestamp=2000.0,
        )

        async def fake_get(key):
            if "m1" in key:
                return e1.model_dump_json()
            if "m2" in key:
                return e2.model_dump_json()
            return None

        # Redis returns latest first (desc=True): m2, m1
        redis = make_redis(zrange=["m2", "m1"])
        redis.get = fake_get
        store = make_store(redis=redis, memory_window=5)

        results = await store.get_recent_memories()

        assert len(results) == 2
        assert results[0].summary == "First"
        assert results[1].summary == "Second"

    async def test_get_recent_memories_uses_config_window(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis, memory_window=3)

        await store.get_recent_memories()

        zrange_kwargs = redis.zrange.call_args
        assert 2 in zrange_kwargs[0] or 2 in zrange_kwargs[1].values()

    async def test_get_recent_memories_respects_limit_override(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis, memory_window=5)

        await store.get_recent_memories(limit=2)

        zrange_kwargs = redis.zrange.call_args
        assert 1 in zrange_kwargs[0] or 1 in zrange_kwargs[1].values()


# ---------------------------------------------------------------------------
# Inference records
# ---------------------------------------------------------------------------


def _make_inference_record(record_id: str = "r1", endpoint: str = "expand") -> InferenceRecord:
    return InferenceRecord(
        id=record_id,
        user_id="default",
        session_id="sess1",
        inference_type=InferenceType.TEXT2TEXT,
        endpoint=endpoint,
        profile="default",
        agent_config={"model": "test-model"},
        context_snapshot={"tokens": "h-w"},
        result_snapshot={"candidates": ["hello world"]},
    )


class TestInferenceRecords:
    async def test_save_inference_record_stores_entry(self):
        redis = make_redis()
        store = make_store(redis=redis)
        record = _make_inference_record()

        await store.save_inference_record(record)

        redis.set.assert_called_once()
        key_arg = redis.set.call_args[0][0]
        assert "r1" in key_arg
        assert "inference" in key_arg

    async def test_save_inference_record_adds_to_sorted_set(self):
        redis = make_redis()
        store = make_store(redis=redis)
        record = _make_inference_record()

        await store.save_inference_record(record)

        redis.zadd.assert_called_once()
        key_arg = redis.zadd.call_args[0][0]
        assert "inferences" in key_arg

    async def test_get_recent_inferences_returns_chronologically(self):
        r1 = _make_inference_record("r1")
        r1 = r1.model_copy(update={"timestamp": 1000.0})
        r2 = _make_inference_record("r2")
        r2 = r2.model_copy(update={"timestamp": 2000.0})

        async def fake_get(key):
            if "r1" in key:
                return r1.model_dump_json()
            if "r2" in key:
                return r2.model_dump_json()
            return None

        redis = make_redis(zrange=["r2", "r1"])
        redis.get = fake_get
        store = make_store(redis=redis, inference_window=50)

        results = await store.get_recent_inferences(inference_type=InferenceType.TEXT2TEXT)

        assert len(results) == 2
        assert results[0].id == "r1"
        assert results[1].id == "r2"

    async def test_get_recent_inferences_uses_config_window(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis, inference_window=10)

        await store.get_recent_inferences()

        zrange_kwargs = redis.zrange.call_args
        assert 9 in zrange_kwargs[0] or 9 in zrange_kwargs[1].values()

    async def test_get_recent_inferences_respects_limit_override(self):
        redis = make_redis(zrange=[])
        store = make_store(redis=redis, inference_window=50)

        await store.get_recent_inferences(limit=3)

        zrange_kwargs = redis.zrange.call_args
        assert 2 in zrange_kwargs[0] or 2 in zrange_kwargs[1].values()

    async def test_get_inference_record_returns_record(self):
        record = _make_inference_record("r1")
        redis = make_redis(get=record.model_dump_json())
        store = make_store(redis=redis)

        result = await store.get_inference_record("r1", InferenceType.TEXT2TEXT)

        assert result is not None
        assert result.id == "r1"
        assert result.endpoint == "expand"

    async def test_get_inference_record_returns_none_when_missing(self):
        redis = make_redis(get=None)
        store = make_store(redis=redis)

        result = await store.get_inference_record("nonexistent", InferenceType.TEXT2TEXT)

        assert result is None


# ---------------------------------------------------------------------------
# format_conversation helper
# ---------------------------------------------------------------------------


class TestFormatConversation:
    def test_empty_entries_returns_empty_string(self):
        assert format_conversation([]) == ""

    def test_other_role_labeled_as_other(self):
        entries = [
            ConversationEntry(user_id="u1", session_id="s1", role="other", text="Hi"),
        ]
        assert format_conversation(entries) == "Other: Hi"

    def test_user_role_labeled_as_user(self):
        entries = [
            ConversationEntry(user_id="u1", session_id="s1", role="user", text="Hello"),
        ]
        assert format_conversation(entries) == "User: Hello"

    def test_mixed_roles(self):
        entries = [
            ConversationEntry(user_id="u1", session_id="s1", role="other", text="Hi"),
            ConversationEntry(user_id="u1", session_id="s1", role="user", text="Hello"),
        ]
        result = format_conversation(entries)
        assert result == "Other: Hi\nUser: Hello"

    def test_default_role_renders_as_other(self):
        entry = ConversationEntry(user_id="u1", session_id="s1", text="ambient speech")
        assert format_conversation([entry]) == "Other: ambient speech"


# ---------------------------------------------------------------------------
# format_memories helper
# ---------------------------------------------------------------------------


class TestFormatMemories:
    def test_empty_entries_returns_empty_string(self):
        assert format_memories([]) == ""

    def test_formats_as_bullet_list(self):
        entries = [
            MemoryEntry(id="m1", user_id="u1", session_id="s1", type=MemoryType.NOTE, summary="First memory"),
            MemoryEntry(id="m2", user_id="u1", session_id="s1", type=MemoryType.NOTE, summary="Second memory"),
        ]
        result = format_memories(entries)
        assert result == "• First memory\n• Second memory"

    def test_single_entry(self):
        entries = [MemoryEntry(id="m1", user_id="u1", session_id="s1", type=MemoryType.NOTE, summary="Only one")]
        assert format_memories(entries) == "• Only one"


# ---------------------------------------------------------------------------
# derive_summary helper
# ---------------------------------------------------------------------------


class TestDeriveSummary:
    def test_extracts_focused_object(self):
        result = MagicMock()
        result.parsed_visual = MagicMock(focused_object="laptop", detected_objects=["keyboard"])
        result.pragmatic = None

        summary = derive_summary(result)
        assert "Focus: laptop" in summary

    def test_extracts_detected_objects(self):
        result = MagicMock()
        result.parsed_visual = MagicMock(focused_object=None, detected_objects=["desk", "chair", "monitor"])
        result.pragmatic = None

        summary = derive_summary(result)
        assert "desk" in summary

    def test_extracts_pragmatic_fields(self):
        result = MagicMock()
        result.parsed_visual = None
        result.pragmatic = MagicMock(
            physical_setting="office",
            intent="work",
            audience="self",
        )

        summary = derive_summary(result)
        assert "Setting: office" in summary
        assert "Intent: work" in summary
        assert "Audience: self" in summary

    def test_fallback_to_model_dump(self):
        result = MagicMock(spec=[])  # no parsed_visual or pragmatic attrs
        result.model_dump = MagicMock(return_value={"key": "value"})

        summary = derive_summary(result)
        assert len(summary) <= 200

    def test_fallback_to_str_when_no_model_dump(self):
        class Plain:
            def __str__(self):
                return "plain string"

        summary = derive_summary(Plain())
        assert "plain string" in summary

    def test_limits_generic_summary_to_200_chars(self):
        result = MagicMock(spec=[])
        long_data = {"x": "a" * 500}
        result.model_dump = MagicMock(return_value=long_data)

        summary = derive_summary(result)
        assert len(summary) <= 200
