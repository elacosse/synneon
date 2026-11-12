import base64

from synneon.agents.base import CognitiveSteeringContextBuilder, VLMImage2ContextAgent
from synneon.schemas.agents import AgentSettings
from synneon.schemas.context import (
    CommunicativeMotive,
    EmotionContext,
    EmotionLabel,
    ParsedVisualContext,
    PragmaticContext,
    SteeringContext,
)


def _settings(
    *,
    system_prompt: str = "System: context={context}",
    prompt_template: str = "User: tokens={tokens}",
    model: str = "test-model",
) -> AgentSettings:
    return AgentSettings(model=model, system_prompt=system_prompt, prompt_template=prompt_template)


def _context(**kwargs) -> SteeringContext:
    return SteeringContext(**kwargs)


# ---------------------------------------------------------------------------
# CognitiveSteeringContextBuilder
# ---------------------------------------------------------------------------


class TestCognitiveSteeringContextBuilder:
    def test_substitutes_context_in_system_prompt(self):
        settings = _settings(system_prompt="You have context: {context}")
        builder = CognitiveSteeringContextBuilder(settings)
        ctx = _context()
        sys_p, _ = builder.build_prompts(ctx)
        assert "{context}" not in sys_p
        assert str(ctx) in sys_p

    def test_substitutes_tokens_in_user_prompt(self):
        settings = _settings(prompt_template="Expand {tokens} please")
        builder = CognitiveSteeringContextBuilder(settings)
        sys_p, user_p = builder.build_prompts(_context(), tokens="h-w-a-y")
        assert "h-w-a-y" in user_p
        assert "{tokens}" not in user_p

    def test_unknown_keys_left_as_placeholders(self):
        """PartialFormatter should leave {unknown} untouched."""
        settings = _settings(
            system_prompt="System {context} {unknown_key}",
            prompt_template="User {tokens}",
        )
        builder = CognitiveSteeringContextBuilder(settings)
        sys_p, _ = builder.build_prompts(_context(), tokens="x")
        assert "{unknown_key}" in sys_p

    def test_extra_kwargs_available_in_user_prompt(self):
        settings = _settings(prompt_template="Tokens: {tokens}; Setting: {setting}")
        builder = CognitiveSteeringContextBuilder(settings)
        _, user_p = builder.build_prompts(_context(), tokens="abc", setting="hospital")
        assert "hospital" in user_p

    def test_context_str_used_not_repr(self):
        settings = _settings(system_prompt="{context}")
        builder = CognitiveSteeringContextBuilder(settings)
        ctx = SteeringContext(
            emotion=EmotionContext(label=EmotionLabel.JOY, intensity=0.9),
            parsed_visual=ParsedVisualContext(focused_object="apple"),
            pragmatic=PragmaticContext(
                intent=CommunicativeMotive.IMPERATIVE,
                physical_setting="kitchen",
            ),
        )
        sys_p, _ = builder.build_prompts(ctx)
        assert sys_p == str(ctx)

    def test_returns_tuple_of_two_strings(self):
        builder = CognitiveSteeringContextBuilder(_settings())
        result = builder.build_prompts(_context())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_system_prompt_with_no_placeholders(self):
        settings = _settings(system_prompt="Static system prompt")
        builder = CognitiveSteeringContextBuilder(settings)
        sys_p, _ = builder.build_prompts(_context())
        assert sys_p == "Static system prompt"


# ---------------------------------------------------------------------------
# VLMImage2ContextAgent._prepare_data
# ---------------------------------------------------------------------------


class TestVLMPrepareData:
    """Test the _prepare_data helper in isolation without hitting any LLM."""

    def _make_agent(self) -> VLMImage2ContextAgent:
        # Minimal settings; the agent object is never run in these tests
        settings = AgentSettings(
            model="test-model",
            system_prompt="sys",
            prompt_template="usr",
        )
        # Avoid actual Agent construction which would require a real model name
        agent = object.__new__(VLMImage2ContextAgent)
        agent.settings = settings
        return agent

    def test_bytes_returned_as_is(self):
        agent = self._make_agent()
        data = b"\x89PNG\r\n\x1a\n"
        assert agent._prepare_data(data) == data

    def test_base64_string_decoded(self):
        agent = self._make_agent()
        original = b"Hello, world!"
        encoded = base64.b64encode(original).decode()
        assert agent._prepare_data(encoded) == original

    def test_data_uri_prefix_stripped_before_decode(self):
        agent = self._make_agent()
        original = b"image bytes"
        encoded = base64.b64encode(original).decode()
        data_uri = f"data:image/jpeg;base64,{encoded}"
        assert agent._prepare_data(data_uri) == original

    def test_empty_bytes_returned_as_is(self):
        agent = self._make_agent()
        assert agent._prepare_data(b"") == b""

    def test_base64_string_without_prefix(self):
        agent = self._make_agent()
        original = b"test"
        encoded = base64.b64encode(original).decode()
        result = agent._prepare_data(encoded)
        assert result == original


# ---------------------------------------------------------------------------
# CognitiveSteeringAgent — what the model actually receives
# ---------------------------------------------------------------------------


class TestCognitiveSteeringAgentPromptDelivery:
    """The prompts the model receives must match what `builder.build_prompts` returns for logging."""

    async def test_system_prompt_placeholders_are_filled_for_the_model(self):
        from pydantic_ai.messages import ModelResponse, TextPart
        from pydantic_ai.models.function import FunctionModel

        from synneon.agents.base import CognitiveSteeringAgent

        seen: dict = {}

        def capture(messages, info):
            seen["system"] = info.instructions
            seen["user"] = "\n".join(
                str(part.content) for m in messages for part in getattr(m, "parts", []) if hasattr(part, "content")
            )
            return ModelResponse(parts=[TextPart("1. ok")])

        settings = _settings(
            model="test",
            system_prompt="Context {context}. Memories: {recent_memories}. History: {conversation_history}",
            prompt_template="Letters: {tokens}",
        )
        agent = CognitiveSteeringAgent(settings=settings, output_type=str)
        kwargs = dict(tokens="G S Y T", recent_memories="MEMORY_MARK", conversation_history="HISTORY_MARK")

        with agent.agent.override(model=FunctionModel(capture)):
            await agent.run(SteeringContext(), **kwargs)

        logged_system, logged_user = agent.builder.build_prompts(SteeringContext(), **kwargs)
        assert seen["system"] == logged_system
        assert "MEMORY_MARK" in seen["system"]
        assert "HISTORY_MARK" in seen["system"]
        assert "{" not in seen["system"]
        assert logged_user in seen["user"]


class TestDefaultProfilePrompts:
    """The shipped expansion profiles keep untrusted text out of the system prompt."""

    def test_history_and_memories_live_in_the_user_message(self):
        import tomllib
        from pathlib import Path

        profiles = tomllib.loads(Path("configs/interpreter/agent.toml").read_text())["agent"]
        for name in ("default", "default-gemini"):
            assert "{recent_memories}" not in profiles[name]["system_prompt"]
            assert "{conversation_history}" not in profiles[name]["system_prompt"]
            assert "{recent_memories}" in profiles[name]["prompt_template"]
            assert "{conversation_history}" in profiles[name]["prompt_template"]
