from synneon.agents.expansion import format_prompt, parse_expansion
from synneon.schemas.agents import AgentSettings


def _settings(prompt_template: str = "Expand: {tokens}") -> AgentSettings:
    return AgentSettings(
        model="test-model",
        system_prompt="You are a helpful assistant.",
        prompt_template=prompt_template,
    )


class TestParseExpansion:
    def test_numbered_dot_list(self):
        text = "1. Hello world\n2. How are you doing\n3. Have you eaten yet"
        result = parse_expansion(text)
        assert len(result.candidates) == 3
        assert result.candidates[0].text == "Hello world"
        assert result.candidates[1].text == "How are you doing"
        assert result.candidates[2].text == "Have you eaten yet"

    def test_numbered_paren_list(self):
        text = "1) First option\n2) Second option"
        result = parse_expansion(text)
        assert len(result.candidates) == 2
        assert result.candidates[0].text == "First option"
        assert result.candidates[1].text == "Second option"

    def test_dash_list(self):
        text = "- item one\n- item two\n- item three"
        result = parse_expansion(text)
        assert len(result.candidates) == 3
        assert result.candidates[0].text == "item one"

    def test_ranks_assigned_sequentially(self):
        text = "1. first\n2. second\n3. third"
        result = parse_expansion(text)
        assert [c.rank for c in result.candidates] == [1, 2, 3]

    def test_truncates_to_five(self):
        lines = "\n".join(f"{i + 1}. option {i + 1}" for i in range(6))
        result = parse_expansion(lines)
        assert len(result.candidates) == 5
        assert result.candidates[4].rank == 5

    def test_exactly_five_candidates_allowed(self):
        lines = "\n".join(f"{i + 1}. option {i + 1}" for i in range(5))
        result = parse_expansion(lines)
        assert len(result.candidates) == 5

    def test_empty_string_returns_empty(self):
        result = parse_expansion("")
        assert result.candidates == []

    def test_whitespace_only_returns_empty(self):
        result = parse_expansion("   \n\n   ")
        assert result.candidates == []

    def test_fallback_plain_lines(self):
        """No bullets/numbers → falls back to splitting on newlines."""
        text = "he wanted to go outside\nhow was the game on a pitch"
        result = parse_expansion(text)
        assert len(result.candidates) == 2
        assert result.candidates[0].text == "he wanted to go outside"
        assert result.candidates[1].text == "how was the game on a pitch"

    def test_fallback_skips_blank_lines(self):
        text = "line one\n\nline two\n\n"
        result = parse_expansion(text)
        assert len(result.candidates) == 2

    def test_trailing_whitespace_stripped(self):
        text = "1. some text   \n2. other text  "
        result = parse_expansion(text)
        assert result.candidates[0].text == "some text"
        assert result.candidates[1].text == "other text"

    def test_leading_whitespace_in_list_item(self):
        text = "  1. indented item"
        result = parse_expansion(text)
        assert len(result.candidates) == 1
        assert result.candidates[0].text == "indented item"

    def test_single_candidate(self):
        result = parse_expansion("1. just one")
        assert len(result.candidates) == 1
        assert result.candidates[0].rank == 1


class TestFormatPrompt:
    def test_formats_tokens(self):
        settings = _settings(prompt_template="Expand the abbreviation: {tokens}")
        result = format_prompt(settings, "h-w-a-y")
        assert result == "Expand the abbreviation: h-w-a-y"

    def test_empty_tokens(self):
        settings = _settings(prompt_template="Input: {tokens}")
        result = format_prompt(settings, "")
        assert result == "Input: "

    def test_extra_kwargs_passed_through(self):
        settings = _settings(prompt_template="{tokens} in context {ctx}")
        result = format_prompt(settings, "abc", ctx="hospital")
        assert result == "abc in context hospital"
