from pydantic import BaseModel, Field

from synneon.schemas.context import ParsedVisualContext, PragmaticContext


class ExpansionCandidate(BaseModel):
    text: str
    rank: int = Field(ge=1, le=5)


class ExpansionResult(BaseModel):
    candidates: list[ExpansionCandidate] = Field(max_length=5)
    system_prompt: str | None = None
    user_prompt: str | None = None


class TranscriptSegment(BaseModel):
    speaker: str = Field(description="Speaker label, e.g. 'SPEAKER_1'.")
    text: str = Field(description="What that speaker said, verbatim.")


class TranscriptionResult(BaseModel):
    text: str
    segments: list[TranscriptSegment] = []


class Image2ContextResult(BaseModel):
    parsed_visual: ParsedVisualContext
    pragmatic: PragmaticContext
