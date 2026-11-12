from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import synneon.api.routes as routes_module
from synneon.api.dependencies import get_context_store
from synneon.schemas.agents import AgentSettings

# Build a minimal test app using only the router
_app = FastAPI()
_app.include_router(routes_module.router)

_BASE_CONTEXT = {
    "emotion": {"label": "neutral", "intensity": 0.0},
    "parsed_visual": {"focused_object": None, "detected_objects": []},
    "pragmatic": {"intent": "declarative", "audience": "unknown", "physical_setting": None},
}


def _make_mock_store() -> MagicMock:
    store = MagicMock()
    store._user_id = "default"
    store._redis = AsyncMock()
    store.get_current_session_id = AsyncMock(return_value="test-session")
    store.get_recent_conversation = AsyncMock(return_value=[])
    store.get_recent_memories = AsyncMock(return_value=[])
    store.save_inference_record = AsyncMock()
    store.save_last_expand_context = AsyncMock()
    return store


@pytest.fixture
def client():
    mock_store = _make_mock_store()

    # Mock history returns to test the 'A' / 'B' speaker mapping
    msg1 = MagicMock()
    msg1.role = "other"
    msg1.text = "How are you?"

    msg2 = MagicMock()
    msg2.role = "user"
    msg2.text = "I am fine."

    mock_store.get_recent_conversation = AsyncMock(return_value=[msg1, msg2])

    _app.dependency_overrides[get_context_store] = lambda: mock_store
    with TestClient(_app) as c:
        yield c
    _app.dependency_overrides.clear()


@pytest.fixture
def mock_alignenv_agent():
    settings = AgentSettings(
        model="openai:gpt-3.5",
        system_prompt="sys",
        prompt_template="usr",
        provider="alignenv",
        endpoint_url="https://alignenv.example.test/",
    )
    agent = MagicMock()
    agent.settings = settings
    routes_module._agents["alignenv"] = agent
    yield
    routes_module._agents.pop("alignenv", None)


class TestAlignEnvInterceptor:
    @patch("synneon.api.routes.httpx.AsyncClient.post")
    def test_smart_token_formatting_handles_mixed_inputs(self, mock_post, client, mock_alignenv_agent):
        """Test that cyo[can] [you]o[the]l correctly parses to spaced Qwen tokens."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"candidates": ["Can you open the laptop?"]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Raw string matching what the UX might send if the user types poorly
        body = {"context": _BASE_CONTEXT, "tokens": "cyo[can] [you]o[the]l", "profile": "alignenv"}

        response = client.post("/inference/expand", json=body)
        assert response.status_code == 200

        # Verify the payload structure
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        # Check tokenization
        assert payload["abbreviation"] == "c y o [can] [you] o [the] l"

        # Check History mapping (A=user, B=other)
        assert len(payload["history"]) == 2
        assert payload["history"][0] == {"speaker": "B", "text": "How are you?"}
        assert payload["history"][1] == {"speaker": "A", "text": "I am fine."}

    @patch("synneon.api.routes.httpx.AsyncClient.post")
    def test_visual_referents_formatting(self, mock_post, client, mock_alignenv_agent):
        """Test that visual contexts are flattened into the referents list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"candidates": ["Result"]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        context_with_visuals = _BASE_CONTEXT.copy()
        context_with_visuals["parsed_visual"] = {"focused_object": "coffee mug", "detected_objects": ["laptop", "desk"]}

        body = {"context": context_with_visuals, "tokens": "c y o", "profile": "alignenv"}

        response = client.post("/inference/expand", json=body)
        assert response.status_code == 200

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]

        # Notice capitalization is applied from the routes mapping
        assert payload["referents"] == ["Coffee mug", "Laptop", "Desk"]

    @patch("synneon.api.routes.httpx.AsyncClient.post")
    def test_more_candidates_routes_to_expand_more(self, mock_post, client, mock_alignenv_agent):
        """Test that setting more_candidates=True routes to the /expand/more endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"candidates": ["More candidates result"]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        body = {"context": _BASE_CONTEXT, "tokens": "c y o", "profile": "alignenv", "more_candidates": True}

        response = client.post("/inference/expand", json=body)
        assert response.status_code == 200

        args, _ = mock_post.call_args
        assert args[0] == "https://alignenv.example.test/expand/more"
