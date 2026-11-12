from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import synneon.api.routes as routes_module
from synneon.api.dependencies import get_context_store
from synneon.schemas.context import ParsedVisualContext, PragmaticContext
from synneon.schemas.results import Image2ContextResult
from synneon.schemas.tts import TTSConfig

# Build a minimal test app using only the router (avoids server.py side effects)
_app = FastAPI()
_app.include_router(routes_module.router)


def _make_mock_store() -> MagicMock:
    """Minimal ContextStore mock sufficient for the existing endpoint tests."""
    store = MagicMock()
    store._user_id = "default"
    store._redis = AsyncMock()
    store._redis.ping = AsyncMock(return_value=True)
    store.get_current_session_id = AsyncMock(return_value="test-session")
    store.get_recent_conversation = AsyncMock(return_value=[])
    store.get_recent_memories = AsyncMock(return_value=[])
    store.list_speakers = AsyncMock(return_value=[])
    store.save_memory = AsyncMock()
    store.save_utterance = AsyncMock()
    store.save_inference_record = AsyncMock()
    store.save_last_expand_context = AsyncMock()
    store.resolve_speaker = AsyncMock(return_value=None)
    return store


@pytest.fixture
def client():
    mock_store = _make_mock_store()
    _app.dependency_overrides[get_context_store] = lambda: mock_store
    with TestClient(_app) as c:
        yield c
    _app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clear_caches():
    routes_module._agents.clear()
    routes_module._vlm_agents.clear()
    yield
    routes_module._agents.clear()
    routes_module._vlm_agents.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_expansion_agent(raw_output: str = "1. Hello world\n2. How are you") -> MagicMock:
    agent = MagicMock()
    agent.builder.build_prompts.return_value = ("sys prompt", "user prompt")
    run_result = MagicMock()
    run_result.output = raw_output
    agent.run = AsyncMock(return_value=run_result)
    return agent


def _mock_vlm_agent(result: Image2ContextResult | None = None) -> MagicMock:
    if result is None:
        result = Image2ContextResult(
            parsed_visual=ParsedVisualContext(focused_object="mug", detected_objects=["table"]),
            pragmatic=PragmaticContext(physical_setting="kitchen"),
        )
    agent = MagicMock()
    run_result = MagicMock()
    run_result.output = result
    agent.run = AsyncMock(return_value=run_result)
    return agent


_BASE_CONTEXT = {
    "emotion": {"label": "neutral", "intensity": 0.0},
    "parsed_visual": {"focused_object": None, "detected_objects": []},
    "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": None},
}

_EXPAND_BODY = {"context": _BASE_CONTEXT, "tokens": "h-w-a-y", "profile": "default"}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_profiles_loaded_empty_initially(self, client):
        response = client.get("/health")
        assert response.json()["profiles_loaded"] == []

    def test_health_profiles_loaded_reflects_cache(self, client):
        routes_module._agents["default"] = _mock_expansion_agent()
        response = client.get("/health")
        assert "default" in response.json()["profiles_loaded"]


# ---------------------------------------------------------------------------
# /inference/expand
# ---------------------------------------------------------------------------


class TestExpandEndpoint:
    def test_expand_returns_200(self, client):
        routes_module._agents["default"] = _mock_expansion_agent()
        response = client.post("/inference/expand", json=_EXPAND_BODY)
        assert response.status_code == 200

    def test_expand_response_has_candidates(self, client):
        routes_module._agents["default"] = _mock_expansion_agent("1. Hello world\n2. How are you")
        response = client.post("/inference/expand", json=_EXPAND_BODY)
        data = response.json()
        assert "candidates" in data
        assert len(data["candidates"]) == 2

    def test_expand_candidate_text_and_rank(self, client):
        routes_module._agents["default"] = _mock_expansion_agent("1. Hello world")
        response = client.post("/inference/expand", json=_EXPAND_BODY)
        candidates = response.json()["candidates"]
        assert candidates[0]["text"] == "Hello world"
        assert candidates[0]["rank"] == 1

    def test_expand_calls_agent_run(self, client):
        mock_agent = _mock_expansion_agent()
        routes_module._agents["default"] = mock_agent
        client.post("/inference/expand", json=_EXPAND_BODY)
        mock_agent.run.assert_called_once()

    def test_expand_passes_tokens_to_agent(self, client):
        mock_agent = _mock_expansion_agent()
        routes_module._agents["default"] = mock_agent
        client.post("/inference/expand", json=_EXPAND_BODY)
        _, kwargs = mock_agent.run.call_args
        assert kwargs.get("tokens") == "h-w-a-y"

    def test_expand_agent_error_returns_500(self, client):
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("LLM failed"))
        routes_module._agents["default"] = agent
        response = client.post("/inference/expand", json=_EXPAND_BODY)
        assert response.status_code == 500

    def test_expand_uses_profile_from_body(self, client):
        routes_module._agents["custom"] = _mock_expansion_agent("1. custom result")
        body = {**_EXPAND_BODY, "profile": "custom"}
        response = client.post("/inference/expand", json=body)
        assert response.status_code == 200

    def test_expand_missing_tokens_returns_422(self, client):
        body = {"context": _BASE_CONTEXT}
        response = client.post("/inference/expand", json=body)
        assert response.status_code == 422

    def test_expand_missing_context_returns_422(self, client):
        body = {"tokens": "abc"}
        response = client.post("/inference/expand", json=body)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /inference/image2context/base64
# ---------------------------------------------------------------------------


class TestImage2ContextBase64Endpoint:
    def test_returns_200(self, client):
        routes_module._vlm_agents["image2context"] = _mock_vlm_agent()
        response = client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "media_type": "image/jpeg"},
        )
        assert response.status_code == 200

    def test_response_structure(self, client):
        routes_module._vlm_agents["image2context"] = _mock_vlm_agent()
        response = client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "media_type": "image/jpeg"},
        )
        data = response.json()
        assert "parsed_visual" in data
        assert "pragmatic" in data

    def test_parsed_visual_content_returned(self, client):
        routes_module._vlm_agents["image2context"] = _mock_vlm_agent()
        response = client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8=", "media_type": "image/jpeg"},
        )
        assert response.json()["parsed_visual"]["focused_object"] == "mug"

    def test_agent_error_returns_500(self, client):
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("VLM failed"))
        routes_module._vlm_agents["image2context"] = agent
        response = client.post(
            "/inference/image2context/base64",
            json={"image_base64": "SGVsbG8="},
        )
        assert response.status_code == 500

    def test_missing_image_base64_returns_422(self, client):
        response = client.post("/inference/image2context/base64", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# /inference/image2context (file upload)
# ---------------------------------------------------------------------------


class TestImage2ContextFileEndpoint:
    def test_returns_200(self, client):
        routes_module._vlm_agents["image2context"] = _mock_vlm_agent()
        response = client.post(
            "/inference/image2context",
            files={"file": ("test.jpg", b"\xff\xd8\xff", "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_has_expected_keys(self, client):
        routes_module._vlm_agents["image2context"] = _mock_vlm_agent()
        response = client.post(
            "/inference/image2context",
            files={"file": ("test.jpg", b"\xff\xd8\xff", "image/jpeg")},
        )
        data = response.json()
        assert "parsed_visual" in data
        assert "pragmatic" in data


# ---------------------------------------------------------------------------
# /audio/speak
# ---------------------------------------------------------------------------


class TestSpeakEndpoint:
    def _speak(self, client, params, audio=b"RIFF\x00\x00\x00\x00WAVEfmt ", **synth_kwargs):
        with (
            patch("synneon.api.routes.get_tts_settings", return_value=TTSConfig()),
            patch("synneon.api.routes.synthesize", new=AsyncMock(return_value=audio, **synth_kwargs)) as synth,
        ):
            response = client.post("/audio/speak", params=params)
        return response, synth

    def test_returns_wav_bytes(self, client):
        response, _ = self._speak(client, {"text": "hello world", "profile": "default"}, audio=b"wav-bytes")
        assert response.status_code == 200
        assert "audio/wav" in response.headers["content-type"]
        assert response.content == b"wav-bytes"

    def test_passes_profile_settings_and_text(self, client):
        _, synth = self._speak(client, {"text": "hello"})
        text, cfg = synth.call_args.args
        assert text == "hello"
        assert cfg.provider == "openai"
        assert synth.call_args.kwargs["style"] is None

    def test_voice_and_model_overrides(self, client):
        _, synth = self._speak(client, {"text": "hi", "voice": "cedar", "model": "custom-model"})
        cfg = synth.call_args.args[1]
        assert (cfg.voice, cfg.model) == ("cedar", "custom-model")

    def test_emotion_becomes_style_hint(self, client):
        _, synth = self._speak(client, {"text": "hi", "emotion_label": "happy", "emotion_intensity": 0.7})
        assert synth.call_args.kwargs["style"] == "Speak in a very happy tone."

    def test_unknown_emotion_speaks_without_style(self, client):
        _, synth = self._speak(client, {"text": "hi", "emotion_label": "not-an-emotion", "emotion_intensity": 0.7})
        assert synth.call_args.kwargs["style"] is None

    def test_missing_text_param_returns_422(self, client):
        response = client.post("/audio/speak")
        assert response.status_code == 422

    def test_tts_error_returns_500(self, client):
        response, _ = self._speak(client, {"text": "hello"}, side_effect=RuntimeError("TTS down"))
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /audio/transcribe
# ---------------------------------------------------------------------------


class TestTranscribeEndpoint:
    def test_returns_transcription(self, client):
        from synneon.schemas.results import TranscriptionResult, TranscriptSegment
        from synneon.schemas.tts import STTConfig

        result = TranscriptionResult(
            text="SPEAKER_1: this is a test", segments=[TranscriptSegment(speaker="SPEAKER_1", text="this is a test")]
        )
        with (
            patch("synneon.api.routes.get_stt_settings", return_value=STTConfig()),
            patch("synneon.api.routes.transcribe", new=AsyncMock(return_value=result)) as tr,
        ):
            response = client.post(
                "/audio/transcribe",
                files={"file": ("audio.webm", b"fake_audio", "audio/webm;codecs=opus")},
            )
        assert response.status_code == 200
        assert response.json()["segments"][0]["text"] == "this is a test"
        audio, media_type, _cfg = tr.call_args.args
        assert (audio, media_type) == (b"fake_audio", "audio/webm;codecs=opus")

    def test_missing_file_returns_422(self, client):
        response = client.post("/audio/transcribe")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Hardening: JSON speech body, viz path containment, error mapping
# ---------------------------------------------------------------------------


class TestSpeakJsonBody:
    def test_text_in_json_body(self, client):
        with (
            patch("synneon.api.routes.get_tts_settings", return_value=TTSConfig()),
            patch("synneon.api.routes.synthesize", new=AsyncMock(return_value=b"wav")) as synth,
        ):
            response = client.post("/audio/speak", json={"text": "hello from the body", "voice": "ash"})
        assert response.status_code == 200
        text, cfg = synth.call_args.args
        assert text == "hello from the body"
        assert cfg.voice == "ash"

    def test_missing_text_is_422(self, client):
        assert client.post("/audio/speak").status_code == 422


class TestVizImagePathContainment:
    def test_serves_example_inside_project(self, client):
        response = client.get("/viz/image", params={"example": "examples/capture_sample_v1", "file": "video.mp4"})
        assert response.status_code == 200

    def test_rejects_absolute_path_outside_project(self, client, tmp_path):
        (tmp_path / "video.mp4").write_bytes(b"not for serving")
        response = client.get("/viz/image", params={"example": str(tmp_path), "file": "video.mp4"})
        assert response.status_code == 404

    def test_rejects_parent_traversal(self, client, tmp_path):
        response = client.get("/viz/image", params={"example": "examples/../../..", "file": "video.mp4"})
        assert response.status_code == 404


class TestErrorMapping:
    def test_invalid_base64_audio_is_400(self, client):
        with patch("synneon.api.routes.get_stt_settings", return_value=MagicMock()):
            response = client.post("/audio/transcribe/base64", json={"audio_base64": "a"})
        assert response.status_code == 400
        assert "base64" in response.json()["detail"]
