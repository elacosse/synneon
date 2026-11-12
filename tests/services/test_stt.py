import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synneon.schemas.tts import STTConfig
from synneon.services.audio import stt


def _openai_client(text: str) -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.audio.transcriptions.create = AsyncMock(return_value=SimpleNamespace(text=text))
    return client


def _gemini_client(output_text: str) -> MagicMock:
    client = MagicMock()
    aio = client.aio
    aio.__aenter__ = AsyncMock(return_value=aio)
    aio.__aexit__ = AsyncMock(return_value=None)
    aio.interactions.create = AsyncMock(return_value=SimpleNamespace(output_text=output_text))
    aio.files.upload = AsyncMock(return_value=SimpleNamespace(uri="files/abc", mime_type="audio/wav"))
    return client


class TestOpenAI:
    async def test_single_segment_with_named_file(self):
        client = _openai_client("  hello there  ")
        cfg = STTConfig(prompt="AAC context", languages=["en", "fr"])
        with patch("openai.AsyncOpenAI", return_value=client):
            result = await stt.transcribe(b"abc", "audio/webm;codecs=opus", cfg)

        assert [(s.speaker, s.text) for s in result.segments] == [("SPEAKER_1", "hello there")]
        assert result.text == "SPEAKER_1: hello there"
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-transcribe"
        assert kwargs["file"] == ("audio.webm", b"abc", "audio/webm")
        assert kwargs["prompt"] == "AAC context"
        assert kwargs["languages"] == ["en", "fr"]

    async def test_empty_transcript_has_no_segments(self):
        with patch("openai.AsyncOpenAI", return_value=_openai_client("   ")):
            result = await stt.transcribe(b"abc", "audio/wav", STTConfig())
        assert result.segments == []
        assert result.text == ""


class TestGemini:
    CFG = STTConfig(provider="google", model="gemini-3.8-flash", prompt="Label speakers", thinking="low")

    async def test_structured_segments_inline_audio(self):
        payload = {"segments": [{"speaker": "SPEAKER_1", "text": "Hi"}, {"speaker": "SPEAKER_2", "text": "Hey"}]}
        client = _gemini_client(json.dumps(payload))
        with patch("google.genai.Client", return_value=client):
            result = await stt.transcribe(b"wavdata", "audio/wav", self.CFG)

        assert [(s.speaker, s.text) for s in result.segments] == [("SPEAKER_1", "Hi"), ("SPEAKER_2", "Hey")]
        req = client.aio.interactions.create.call_args.kwargs
        assert req["model"] == "gemini-3.8-flash"
        assert req["system_instruction"] == "Label speakers"
        assert req["generation_config"] == {"thinking_level": "low"}
        assert req["response_format"]["mime_type"] == "application/json"
        assert "segments" in req["response_format"]["schema"]["properties"]
        audio_part = req["input"][1]
        assert audio_part == {"type": "audio", "data": base64.b64encode(b"wavdata").decode(), "mime_type": "audio/wav"}
        client.aio.files.upload.assert_not_called()
        client.aio.__aexit__.assert_awaited_once()

    async def test_large_audio_uses_files_api(self, monkeypatch):
        monkeypatch.setattr(stt, "GEMINI_INLINE_LIMIT", 4)
        client = _gemini_client('{"segments": []}')
        with patch("google.genai.Client", return_value=client):
            await stt.transcribe(b"0123456789", "audio/wav", self.CFG)

        client.aio.files.upload.assert_awaited_once()
        audio_part = client.aio.interactions.create.call_args.kwargs["input"][1]
        assert audio_part == {"type": "audio", "uri": "files/abc", "mime_type": "audio/wav"}


async def test_unknown_provider_raises():
    cfg = STTConfig.model_construct(provider="together", model="x", prompt=None, languages=[], thinking=None)
    with pytest.raises(ValueError, match="Unsupported STT provider"):
        await stt.transcribe(b"", "audio/wav", cfg)
