import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synneon.schemas.tts import TTSConfig
from synneon.services.audio import tts


class _FakeConnection:
    def __init__(self, events):
        self._events = events
        self.session = SimpleNamespace(update=AsyncMock())
        self.response = SimpleNamespace(create=AsyncMock())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _openai_client(conn: _FakeConnection) -> MagicMock:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.realtime.connect = MagicMock(return_value=conn)
    return client


def _delta(pcm: bytes):
    return SimpleNamespace(type="response.output_audio.delta", delta=base64.b64encode(pcm).decode())


def _done(status="completed"):
    return SimpleNamespace(type="response.done", response=SimpleNamespace(status=status, status_details=None))


class TestOpenAIRealtime:
    async def test_collects_pcm_into_wav(self):
        conn = _FakeConnection([_delta(b"\x01\x00" * 3), _delta(b"\x02\x00" * 2), _done()])
        client = _openai_client(conn)
        with patch("openai.AsyncOpenAI", return_value=client):
            wav = await tts.synthesize("Hello there", TTSConfig(voice="cedar"), style="Speak in a very happy tone.")

        with wave.open(io.BytesIO(wav)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
            assert w.readframes(w.getnframes()) == b"\x01\x00" * 3 + b"\x02\x00" * 2

        client.realtime.connect.assert_called_once_with(
            model="gpt-realtime-2.1-mini", websocket_connection_options={"happy_eyeballs_delay": 0.25}
        )
        session = conn.session.update.call_args.kwargs["session"]
        assert session["output_modalities"] == ["audio"]
        assert session["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "cedar"}
        assert session["instructions"].endswith("Speak in a very happy tone.")
        response = conn.response.create.call_args.kwargs["response"]
        assert response["conversation"] == "none"
        assert response["input"][0]["content"] == [{"type": "input_text", "text": "Hello there"}]

    async def test_error_event_raises(self):
        err = SimpleNamespace(type="error", error=SimpleNamespace(message="bad voice"))
        with patch("openai.AsyncOpenAI", return_value=_openai_client(_FakeConnection([err]))):
            with pytest.raises(RuntimeError, match="bad voice"):
                await tts.synthesize("hi", TTSConfig())

    async def test_incomplete_response_raises(self):
        conn = _FakeConnection([_delta(b"\x00\x00"), _done(status="cancelled")])
        with patch("openai.AsyncOpenAI", return_value=_openai_client(conn)):
            with pytest.raises(RuntimeError, match="cancelled"):
                await tts.synthesize("hi", TTSConfig())


class TestGemini:
    CFG = TTSConfig(provider="google", model="gemini-3.8-flash-tts", voice="Kore")

    @staticmethod
    def _client(output_audio) -> MagicMock:
        """genai.Client() whose `.aio` is an async context manager yielding the async client."""
        client = MagicMock()
        aio = client.aio
        aio.__aenter__ = AsyncMock(return_value=aio)
        aio.__aexit__ = AsyncMock(return_value=None)
        aio.interactions.create = AsyncMock(return_value=SimpleNamespace(output_audio=output_audio))
        return client

    async def test_returns_decoded_wav_with_style(self):
        client = self._client(SimpleNamespace(data=base64.b64encode(b"RIFFwav").decode()))
        with patch("google.genai.Client", return_value=client):
            audio = await tts.synthesize("Hi", self.CFG, style="Speak in a somewhat sad tone.")

        assert audio == b"RIFFwav"
        req = client.aio.interactions.create.call_args.kwargs
        assert req["model"] == "gemini-3.8-flash-tts"
        assert req["response_format"] == {"type": "audio", "mime_type": "audio/wav"}
        assert req["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
        content = req["input"][0]["content"][0]
        assert content["text"] == "Hi"
        assert content["annotations"] == [{"type": "speech_metadata", "style": "Speak in a somewhat sad tone."}]

    async def test_no_style_means_no_annotations(self):
        client = self._client(SimpleNamespace(data=b"RIFFraw"))
        with patch("google.genai.Client", return_value=client):
            assert await tts.synthesize("Hi", self.CFG) == b"RIFFraw"
        client.aio.__aexit__.assert_awaited_once()
        assert "annotations" not in client.aio.interactions.create.call_args.kwargs["input"][0]["content"][0]

    async def test_missing_audio_raises(self):
        client = self._client(None)
        with patch("google.genai.Client", return_value=client):
            with pytest.raises(RuntimeError, match="no audio"):
                await tts.synthesize("Hi", self.CFG)
