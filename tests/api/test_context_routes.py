"""Tests for context/memory API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import synneon.api.routes as routes_module
from synneon.api.dependencies import get_context_store
from synneon.schemas.context import ParsedVisualContext, PragmaticContext
from synneon.schemas.memory import (
    ConversationEntry,
    InferenceRecord,
    InferenceType,
    MemoryEntry,
    MemoryType,
)
from synneon.schemas.results import Image2ContextResult

# ---------------------------------------------------------------------------
# App setup with store dependency override
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(routes_module.router)


def _make_mock_store(
    *,
    session_id: str | None = "test-session",
    conversation: list[ConversationEntry] | None = None,
    memories: list[MemoryEntry] | None = None,
) -> MagicMock:
    store = MagicMock()
    store._user_id = "default"
    store._redis = AsyncMock()
    store._redis.ping = AsyncMock(return_value=True)
    store.get_current_session_id = AsyncMock(return_value=session_id)
    store.get_recent_conversation = AsyncMock(return_value=conversation or [])
    store.get_recent_memories = AsyncMock(return_value=memories or [])
    store.save_memory = AsyncMock()
    store.save_utterance = AsyncMock()
    store.save_inference_record = AsyncMock()
    store.save_last_expand_context = AsyncMock()
    store.get_recent_inferences = AsyncMock(return_value=[])
    store.get_inference_record = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_store() -> MagicMock:
    return _make_mock_store()


@pytest.fixture
def client(mock_store) -> TestClient:
    _app.dependency_overrides[get_context_store] = lambda: mock_store
    with TestClient(_app) as c:
        yield c
    _app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clear_caches():
    routes_module._agents.clear()
    routes_module._vlm_agents.clear()
    routes_module._neon_agents.clear()
    yield
    routes_module._agents.clear()
    routes_module._vlm_agents.clear()
    routes_module._neon_agents.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv_entry(text: str = "Hello", role: str = "other") -> ConversationEntry:
    return ConversationEntry(
        user_id="default",
        session_id="test-session",
        role=role,
        text=text,
        timestamp=1000.0,
    )


def _make_memory_entry(summary: str = "A note") -> MemoryEntry:
    return MemoryEntry(
        id="m1",
        user_id="default",
        session_id="test-session",
        type=MemoryType.NOTE,
        summary=summary,
    )


def _make_inference_record(record_id: str = "r1") -> InferenceRecord:
    return InferenceRecord(
        id=record_id,
        user_id="default",
        session_id="test-session",
        inference_type=InferenceType.TEXT2TEXT,
        endpoint="expand",
        profile="default",
        agent_config={"model": "test-model"},
        context_snapshot={"tokens": "h-w"},
        result_snapshot={"candidates": ["hello world"]},
    )


def _mock_vlm_result() -> Image2ContextResult:
    return Image2ContextResult(
        parsed_visual=ParsedVisualContext(focused_object="screen", detected_objects=["keyboard"]),
        pragmatic=PragmaticContext(physical_setting="office"),
    )


# ---------------------------------------------------------------------------
# GET /context/conversation
# ---------------------------------------------------------------------------


class TestGetConversation:
    def test_returns_200(self, client):
        response = client.get("/context/conversation")
        assert response.status_code == 200

    def test_returns_empty_list_when_no_entries(self, client, mock_store):
        mock_store.get_recent_conversation = AsyncMock(return_value=[])
        response = client.get("/context/conversation")
        assert response.json() == []

    def test_returns_entries_for_current_session(self, client, mock_store):
        entry = _make_conv_entry("Hello world")
        mock_store.get_recent_conversation = AsyncMock(return_value=[entry])
        response = client.get("/context/conversation")
        data = response.json()
        assert len(data) == 1
        assert data[0]["text"] == "Hello world"

    def test_uses_current_session_id_when_not_specified(self, client, mock_store):
        mock_store.get_current_session_id = AsyncMock(return_value="current-sess")
        mock_store.get_recent_conversation = AsyncMock(return_value=[])
        client.get("/context/conversation")
        mock_store.get_recent_conversation.assert_called_once_with("current-sess", limit=20)

    def test_uses_explicit_session_id_when_provided(self, client, mock_store):
        mock_store.get_recent_conversation = AsyncMock(return_value=[])
        client.get("/context/conversation?session_id=explicit-sess")
        mock_store.get_recent_conversation.assert_called_once_with("explicit-sess", limit=20)

    def test_respects_limit_param(self, client, mock_store):
        mock_store.get_recent_conversation = AsyncMock(return_value=[])
        client.get("/context/conversation?limit=5")
        mock_store.get_recent_conversation.assert_called_once_with("test-session", limit=5)

    def test_returns_empty_when_no_session_active(self, client, mock_store):
        mock_store.get_current_session_id = AsyncMock(return_value=None)
        response = client.get("/context/conversation")
        assert response.json() == []
        assert response.status_code == 200

    def test_limit_below_1_returns_422(self, client):
        response = client.get("/context/conversation?limit=0")
        assert response.status_code == 422

    def test_limit_above_200_returns_422(self, client):
        response = client.get("/context/conversation?limit=201")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /context/memories
# ---------------------------------------------------------------------------


class TestGetMemories:
    def test_returns_200(self, client):
        response = client.get("/context/memories")
        assert response.status_code == 200

    def test_returns_empty_list_when_no_memories(self, client, mock_store):
        mock_store.get_recent_memories = AsyncMock(return_value=[])
        response = client.get("/context/memories")
        assert response.json() == []

    def test_returns_memory_entries(self, client, mock_store):
        entry = _make_memory_entry("Focus: laptop")
        mock_store.get_recent_memories = AsyncMock(return_value=[entry])
        response = client.get("/context/memories")
        data = response.json()
        assert len(data) == 1
        assert data[0]["summary"] == "Focus: laptop"
        assert data[0]["type"] == "note"

    def test_respects_limit_param(self, client, mock_store):
        mock_store.get_recent_memories = AsyncMock(return_value=[])
        client.get("/context/memories?limit=3")
        mock_store.get_recent_memories.assert_called_once_with(limit=3)

    def test_limit_below_1_returns_422(self, client):
        response = client.get("/context/memories?limit=0")
        assert response.status_code == 422

    def test_limit_above_100_returns_422(self, client):
        response = client.get("/context/memories?limit=101")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /health with store dependency
# ---------------------------------------------------------------------------


class TestHealthWithStore:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_redis_connected_when_ping_succeeds(self, client, mock_store):
        mock_store._redis.ping = AsyncMock(return_value=True)
        response = client.get("/health")
        assert response.json()["redis"] == "connected"

    def test_health_redis_unavailable_when_ping_fails(self, client, mock_store):
        mock_store._redis.ping = AsyncMock(side_effect=Exception("Connection refused"))
        response = client.get("/health")
        assert response.json()["redis"] == "unavailable"


# ---------------------------------------------------------------------------
# /inference/expand - memory enrichment
# ---------------------------------------------------------------------------


_BASE_CONTEXT = {
    "emotion": {"label": "neutral", "intensity": 0.0},
    "parsed_visual": {"focused_object": None, "detected_objects": []},
    "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": None},
}
_EXPAND_BODY = {"context": _BASE_CONTEXT, "tokens": "h-w-a-y", "profile": "default"}


class TestExpandWithMemoryEnrichment:
    def _make_agent(self, raw: str = "1. Hello world") -> MagicMock:
        agent = MagicMock()
        agent.builder.build_prompts.return_value = ("sys prompt", "user prompt")
        run_result = MagicMock()
        run_result.output = raw
        agent.run = AsyncMock(return_value=run_result)
        return agent

    def test_expand_fetches_conversation_history(self, client, mock_store):
        routes_module._agents["default"] = self._make_agent()
        entry = _make_conv_entry("Prior utterance")
        mock_store.get_recent_conversation = AsyncMock(return_value=[entry])

        client.post("/inference/expand", json=_EXPAND_BODY)

        mock_store.get_recent_conversation.assert_called_once()

    def test_expand_fetches_memories(self, client, mock_store):
        routes_module._agents["default"] = self._make_agent()
        client.post("/inference/expand", json=_EXPAND_BODY)
        mock_store.get_recent_memories.assert_called_once()

    def test_expand_passes_conversation_history_to_agent(self, client, mock_store):
        agent = self._make_agent()
        routes_module._agents["default"] = agent
        entry = _make_conv_entry("Prior utterance")
        mock_store.get_recent_conversation = AsyncMock(return_value=[entry])

        client.post("/inference/expand", json=_EXPAND_BODY)

        _, kwargs = agent.run.call_args
        assert "conversation_history" in kwargs
        assert "Prior utterance" in kwargs["conversation_history"]

    def test_expand_passes_memories_to_agent(self, client, mock_store):
        agent = self._make_agent()
        routes_module._agents["default"] = agent
        mem = _make_memory_entry("Focus: screen")
        mock_store.get_recent_memories = AsyncMock(return_value=[mem])

        client.post("/inference/expand", json=_EXPAND_BODY)

        _, kwargs = agent.run.call_args
        assert "recent_memories" in kwargs
        assert "Focus: screen" in kwargs["recent_memories"]

    def test_expand_passes_empty_strings_when_no_history_or_memories(self, client, mock_store):
        agent = self._make_agent()
        routes_module._agents["default"] = agent
        mock_store.get_recent_conversation = AsyncMock(return_value=[])
        mock_store.get_recent_memories = AsyncMock(return_value=[])

        client.post("/inference/expand", json=_EXPAND_BODY)

        _, kwargs = agent.run.call_args
        assert kwargs["conversation_history"] == ""
        assert kwargs["recent_memories"] == ""


# ---------------------------------------------------------------------------
# /inference/image2context/base64 - save_to_memory flag
# ---------------------------------------------------------------------------


class TestImage2ContextSaveToMemory:
    def _make_vlm_agent(self) -> MagicMock:
        result = _mock_vlm_result()
        agent = MagicMock()
        run_result = MagicMock()
        run_result.output = result
        agent.run = AsyncMock(return_value=run_result)
        return agent

    def test_save_to_memory_false_does_not_persist(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": False},
        )
        mock_store.save_memory.assert_not_called()

    def test_save_to_memory_true_persists_entry(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        mock_store.get_current_session_id = AsyncMock(return_value="sess-123")
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": True},
        )
        mock_store.save_memory.assert_called_once()

    def test_save_to_memory_entry_has_correct_type(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        mock_store.get_current_session_id = AsyncMock(return_value="sess-123")
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": True},
        )
        saved_entry = mock_store.save_memory.call_args[0][0]
        assert saved_entry.type == MemoryType.VLM_IMAGE_CONTEXT

    def test_save_to_memory_entry_uses_current_session(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        mock_store.get_current_session_id = AsyncMock(return_value="my-session")
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": True},
        )
        saved_entry = mock_store.save_memory.call_args[0][0]
        assert saved_entry.session_id == "my-session"

    def test_save_to_memory_starts_a_session_when_the_user_has_none(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        mock_store.get_current_session_id = AsyncMock(return_value=None)
        mock_store.start_session = AsyncMock()
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": True},
        )
        started = mock_store.start_session.call_args.args[0]
        saved_entry = mock_store.save_memory.call_args[0][0]
        assert saved_entry.session_id == started != "unknown"


# ---------------------------------------------------------------------------
# /audio/transcribe - save_to_memory flag
# ---------------------------------------------------------------------------


class TestTranscribeSaveToMemory:
    def _post(self, client, segments, save: bool):
        from unittest.mock import patch

        from synneon.schemas.results import TranscriptionResult, TranscriptSegment
        from synneon.schemas.tts import STTConfig

        result = TranscriptionResult(
            text="\n".join(f"{spk}: {txt}" for spk, txt in segments),
            segments=[TranscriptSegment(speaker=spk, text=txt) for spk, txt in segments],
        )
        with (
            patch("synneon.api.routes.get_stt_settings", return_value=STTConfig()),
            patch("synneon.api.routes.transcribe", new=AsyncMock(return_value=result)),
        ):
            return client.post(
                "/audio/transcribe",
                files={"file": ("a.wav", b"audio", "audio/wav")},
                params={"save_to_memory": str(save).lower()},
            )

    def test_save_to_memory_false_does_not_persist(self, client, mock_store):
        self._post(client, [("SPEAKER_1", "Hello")], save=False)
        mock_store.save_utterance.assert_not_called()

    def test_save_to_memory_true_persists_each_segment(self, client, mock_store):
        self._post(client, [("SPEAKER_1", "Hello"), ("SPEAKER_2", "World")], save=True)
        assert mock_store.save_utterance.call_count == 2
        texts = [c.args[0].text for c in mock_store.save_utterance.call_args_list]
        assert texts == ["Hello", "World"]

    def test_saved_utterances_have_role_other(self, client, mock_store):
        self._post(client, [("SPEAKER_1", "Hi")], save=True)
        assert mock_store.save_utterance.call_args[0][0].role == "other"

    def test_empty_transcript_saves_nothing(self, client, mock_store):
        self._post(client, [], save=True)
        mock_store.save_utterance.assert_not_called()

    def test_base64_route_persists_segments(self, client, mock_store):
        import base64
        from unittest.mock import patch

        from synneon.schemas.results import TranscriptionResult, TranscriptSegment
        from synneon.schemas.tts import STTConfig

        result = TranscriptionResult(
            text="SPEAKER_1: Hey", segments=[TranscriptSegment(speaker="SPEAKER_1", text="Hey")]
        )
        with (
            patch("synneon.api.routes.get_stt_settings", return_value=STTConfig()),
            patch("synneon.api.routes.transcribe", new=AsyncMock(return_value=result)) as tr,
        ):
            response = client.post(
                "/audio/transcribe/base64",
                json={
                    "audio_base64": base64.b64encode(b"audio").decode(),
                    "media_type": "audio/mp4",
                    "save_to_memory": True,
                },
            )
        assert response.status_code == 200
        assert tr.call_args.args[:2] == (b"audio", "audio/mp4")
        mock_store.save_utterance.assert_called_once()


# ---------------------------------------------------------------------------
# GET /context/inferences
# ---------------------------------------------------------------------------


class TestGetInferences:
    def test_returns_200(self, client):
        response = client.get("/context/inferences")
        assert response.status_code == 200

    def test_returns_empty_list_when_no_records(self, client, mock_store):
        mock_store.get_recent_inferences = AsyncMock(return_value=[])
        response = client.get("/context/inferences")
        assert response.json() == []

    def test_returns_inference_records(self, client, mock_store):
        record = _make_inference_record("r1")
        mock_store.get_recent_inferences = AsyncMock(return_value=[record])
        response = client.get("/context/inferences")
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "r1"
        assert data[0]["endpoint"] == "expand"
        assert data[0]["agent_config"]["model"] == "test-model"

    def test_respects_limit_param(self, client, mock_store):
        mock_store.get_recent_inferences = AsyncMock(return_value=[])
        client.get("/context/inferences?limit=5")
        mock_store.get_recent_inferences.assert_called_once_with(limit=5, inference_type=None)

    def test_limit_below_1_returns_422(self, client):
        response = client.get("/context/inferences?limit=0")
        assert response.status_code == 422

    def test_limit_above_100_returns_422(self, client):
        response = client.get("/context/inferences?limit=101")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /context/inferences/{record_id}
# ---------------------------------------------------------------------------


class TestGetInferenceById:
    def test_returns_200_when_found(self, client, mock_store):
        record = _make_inference_record("r1")
        mock_store.get_inference_record = AsyncMock(return_value=record)
        response = client.get("/context/inferences/r1?inference_type=text2text")
        assert response.status_code == 200

    def test_returns_record_fields(self, client, mock_store):
        record = _make_inference_record("r1")
        mock_store.get_inference_record = AsyncMock(return_value=record)
        response = client.get("/context/inferences/r1?inference_type=text2text")
        data = response.json()
        assert data["id"] == "r1"
        assert data["profile"] == "default"
        assert data["context_snapshot"]["tokens"] == "h-w"

    def test_returns_404_when_not_found(self, client, mock_store):
        mock_store.get_inference_record = AsyncMock(return_value=None)
        response = client.get("/context/inferences/missing?inference_type=text2text")
        assert response.status_code == 404

    def test_calls_store_with_correct_id(self, client, mock_store):
        record = _make_inference_record("r-abc")
        mock_store.get_inference_record = AsyncMock(return_value=record)
        client.get("/context/inferences/r-abc?inference_type=text2text")
        mock_store.get_inference_record.assert_called_once_with("r-abc", InferenceType.TEXT2TEXT)


# ---------------------------------------------------------------------------
# Expand auto-saves InferenceRecord
# ---------------------------------------------------------------------------


class TestExpandSavesInferenceRecord:
    def _make_agent(self, raw: str = "1. Hello world") -> MagicMock:
        agent = MagicMock()
        agent.settings = MagicMock()
        agent.settings.model_dump = MagicMock(return_value={"model": "test-model"})
        agent.builder.build_prompts.return_value = ("sys prompt", "user prompt")
        run_result = MagicMock()
        run_result.output = raw
        agent.run = AsyncMock(return_value=run_result)
        return agent

    def test_expand_saves_inference_record(self, client, mock_store):
        routes_module._agents["default"] = self._make_agent()
        _EXPAND_BODY = {
            "context": {
                "emotion": {"label": "neutral", "intensity": 0.0},
                "parsed_visual": {"focused_object": None, "detected_objects": []},
                "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": None},
            },
            "tokens": "h-w",
            "profile": "default",
        }
        client.post("/inference/expand", json=_EXPAND_BODY)
        mock_store.save_inference_record.assert_called_once()

    def test_expand_record_has_correct_endpoint(self, client, mock_store):
        routes_module._agents["default"] = self._make_agent()
        _EXPAND_BODY = {
            "context": {
                "emotion": {"label": "neutral", "intensity": 0.0},
                "parsed_visual": {"focused_object": None, "detected_objects": []},
                "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": None},
            },
            "tokens": "h-w",
            "profile": "default",
        }
        client.post("/inference/expand", json=_EXPAND_BODY)
        record = mock_store.save_inference_record.call_args[0][0]
        assert record.endpoint == "expand"
        assert record.profile == "default"
        assert "tokens" in record.context_snapshot
        assert record.agent_config == {"model": "test-model"}


# ---------------------------------------------------------------------------
# Image2Context auto-saves InferenceRecord
# ---------------------------------------------------------------------------


class TestImage2ContextSavesInferenceRecord:
    def _make_vlm_agent(self) -> MagicMock:
        result = _mock_vlm_result()
        agent = MagicMock()
        agent.settings = MagicMock()
        agent.settings.model_dump = MagicMock(return_value={"model": "vlm-model"})
        run_result = MagicMock()
        run_result.output = result
        agent.run = AsyncMock(return_value=run_result)
        return agent

    def test_image2context_saves_inference_record(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": False},
        )
        mock_store.save_inference_record.assert_called_once()

    def test_image2context_record_has_correct_endpoint(self, client, mock_store):
        routes_module._vlm_agents["image2context"] = self._make_vlm_agent()
        client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "save_to_memory": False},
        )
        record = mock_store.save_inference_record.call_args[0][0]
        assert record.endpoint == "image2context"
        assert "media_type" in record.context_snapshot
