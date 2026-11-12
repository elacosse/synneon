import pytest
from pydantic import ValidationError

from synneon.schemas.context import ParsedVisualContext, PragmaticContext
from synneon.schemas.results import (
    ExpansionCandidate,
    ExpansionResult,
    Image2ContextResult,
    TranscriptionResult,
)


class TestExpansionCandidate:
    def test_valid(self):
        c = ExpansionCandidate(text="hello world", rank=1)
        assert c.text == "hello world"
        assert c.rank == 1

    @pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
    def test_rank_valid_range(self, rank):
        c = ExpansionCandidate(text="x", rank=rank)
        assert c.rank == rank

    def test_rank_zero_raises(self):
        with pytest.raises(ValidationError):
            ExpansionCandidate(text="x", rank=0)

    def test_rank_six_raises(self):
        with pytest.raises(ValidationError):
            ExpansionCandidate(text="x", rank=6)

    def test_rank_negative_raises(self):
        with pytest.raises(ValidationError):
            ExpansionCandidate(text="x", rank=-1)

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            ExpansionCandidate(rank=1)  # type: ignore[call-arg]

    def test_missing_rank_raises(self):
        with pytest.raises(ValidationError):
            ExpansionCandidate(text="x")  # type: ignore[call-arg]


class TestExpansionResult:
    def test_empty_candidates(self):
        result = ExpansionResult(candidates=[])
        assert result.candidates == []

    def test_single_candidate(self):
        result = ExpansionResult(candidates=[ExpansionCandidate(text="hi", rank=1)])
        assert len(result.candidates) == 1

    def test_max_five_candidates(self):
        candidates = [ExpansionCandidate(text=f"option {i}", rank=i) for i in range(1, 6)]
        result = ExpansionResult(candidates=candidates)
        assert len(result.candidates) == 5

    def test_six_candidates_raises(self):
        candidates = [ExpansionCandidate(text=f"opt {i}", rank=min(i, 5)) for i in range(1, 7)]
        with pytest.raises(ValidationError):
            ExpansionResult(candidates=candidates)

    def test_round_trip(self):
        result = ExpansionResult(
            candidates=[
                ExpansionCandidate(text="how was the game on a pitch", rank=1),
                ExpansionCandidate(text="he wanted to go out and play", rank=2),
            ]
        )
        assert ExpansionResult.model_validate(result.model_dump()) == result


class TestTranscriptionResult:
    def test_valid(self):
        result = TranscriptionResult(text="hello world")
        assert result.text == "hello world"

    def test_empty_string_valid(self):
        result = TranscriptionResult(text="")
        assert result.text == ""

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            TranscriptionResult()  # type: ignore[call-arg]

    def test_round_trip(self):
        result = TranscriptionResult(text="transcribed speech")
        assert TranscriptionResult.model_validate(result.model_dump()) == result


class TestImage2ContextResult:
    def test_valid(self):
        result = Image2ContextResult(
            parsed_visual=ParsedVisualContext(focused_object="mug"),
            pragmatic=PragmaticContext(),
        )
        assert result.parsed_visual.focused_object == "mug"
        assert result.pragmatic is not None

    def test_missing_parsed_visual_raises(self):
        with pytest.raises(ValidationError):
            Image2ContextResult(pragmatic=PragmaticContext())  # type: ignore[call-arg]

    def test_missing_pragmatic_raises(self):
        with pytest.raises(ValidationError):
            Image2ContextResult(parsed_visual=ParsedVisualContext())  # type: ignore[call-arg]

    def test_round_trip(self):
        result = Image2ContextResult(
            parsed_visual=ParsedVisualContext(
                focused_object="laptop",
                detected_objects=["desk", "lamp"],
            ),
            pragmatic=PragmaticContext(physical_setting="office"),
        )
        assert Image2ContextResult.model_validate(result.model_dump()) == result
