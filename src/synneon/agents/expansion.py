import re
from typing import Any

from synneon.agents.base import CognitiveSteeringAgent
from synneon.schemas.agents import AgentSettings
from synneon.schemas.results import ExpansionCandidate, ExpansionResult

# Matches lines like "1. some text", "2) some text", "- some text"
_LINE_PATTERN = re.compile(r"^\s*(?:\d+[.)]\s*|-\s*)(.*\S)", re.MULTILINE)


def create_expansion_agent(settings: AgentSettings) -> CognitiveSteeringAgent[str]:
    """Create an L0 abbreviation expansion agent that returns plain text."""
    return CognitiveSteeringAgent(
        settings=settings,
        output_type=str,
    )


def format_prompt(settings: AgentSettings, text: str, **kwargs: Any) -> str:
    """Format the abbreviation text into a user prompt using the configured template."""
    return settings.prompt_template.format(tokens=text, **kwargs)


def parse_expansion(text: str) -> ExpansionResult:
    """Parse plain-text model output into an ExpansionResult.

    Handles numbered lists ("1. ...", "2) ..."), dash-prefixed lists ("- ..."),
    and falls back to treating each non-empty line as a candidate.
    Truncates to 5 candidates max.
    """
    matches = _LINE_PATTERN.findall(text)

    if not matches:
        # Fallback: treat each non-empty line as a candidate
        matches = [line.strip() for line in text.splitlines() if line.strip()]

    candidates = [ExpansionCandidate(text=match, rank=i + 1) for i, match in enumerate(matches[:5])]
    return ExpansionResult(candidates=candidates)
