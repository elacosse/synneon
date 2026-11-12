"""Text-to-speech across providers. Both return complete WAV bytes.

OpenAI: `gpt-realtime-2.1-mini` over the Realtime API (the /v1/audio/speech models
shut down on 2027-01-06). Gemini: `gemini-3.8-flash-tts` via the Interactions API.
"""

from __future__ import annotations

import base64
import io
import wave

from synneon.schemas.tts import TTSConfig

REALTIME_SAMPLE_RATE = 24000  # Realtime `audio/pcm` output: 24 kHz mono s16le, no header.

# Race IPv4 against IPv6 when opening the Realtime socket. Without this the websocket client tries
# addresses one at a time, so a network with a dead IPv6 route times out before reaching IPv4.
_WEBSOCKET_OPTIONS = {"happy_eyeballs_delay": 0.25}

_READ_VERBATIM = (
    "You are a text-to-speech voice for an AAC user. Speak the user's message aloud exactly as written, "
    "word for word, in the language it is written in. Do not answer it, comment on it, or add anything."
)


async def synthesize(text: str, cfg: TTSConfig, style: str | None = None) -> bytes:
    """Speak `text`; `style` is a plain-language delivery hint such as 'Speak in a very happy tone.'"""
    if cfg.provider == "openai":
        return await _synthesize_openai_realtime(text, cfg, style)
    if cfg.provider == "google":
        return await _synthesize_gemini(text, cfg, style)
    raise ValueError(f"Unsupported TTS provider: {cfg.provider}")


def pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _synthesize_openai_realtime(text: str, cfg: TTSConfig, style: str | None) -> bytes:
    from openai import AsyncOpenAI

    instructions = f"{_READ_VERBATIM} {style}" if style else _READ_VERBATIM
    pcm = bytearray()
    async with (
        AsyncOpenAI() as client,
        client.realtime.connect(model=cfg.model, websocket_connection_options=_WEBSOCKET_OPTIONS) as conn,
    ):
        await conn.session.update(
            session={
                "type": "realtime",
                "output_modalities": ["audio"],
                "instructions": instructions,
                "audio": {
                    "output": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                        "voice": cfg.voice,
                    }
                },
            }
        )
        # Out-of-band response: nothing is kept in conversation state between requests.
        await conn.response.create(
            response={
                "conversation": "none",
                "output_modalities": ["audio"],
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}],
            }
        )
        async for event in conn:
            if event.type == "response.output_audio.delta":
                pcm += base64.b64decode(event.delta)
            elif event.type == "error":
                raise RuntimeError(f"Realtime TTS error: {event.error.message}")
            elif event.type == "response.done":
                if event.response.status != "completed":
                    raise RuntimeError(
                        f"Realtime TTS response {event.response.status}: {event.response.status_details}"
                    )
                break
    return pcm16_to_wav(bytes(pcm), REALTIME_SAMPLE_RATE)


async def _synthesize_gemini(text: str, cfg: TTSConfig, style: str | None) -> bytes:
    from google import genai

    content: dict = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]

    async with genai.Client().aio as client:
        interaction = await client.interactions.create(
            model=cfg.model,
            input=[{"type": "user_input", "content": [content]}],
            response_format={"type": "audio", "mime_type": "audio/wav"},
            generation_config={"speech_config": [{"voice": cfg.voice}]},
        )
    if interaction.output_audio is None or not interaction.output_audio.data:
        raise RuntimeError("Gemini TTS returned no audio")
    data = interaction.output_audio.data
    return base64.b64decode(data) if isinstance(data, str) else bytes(data)
