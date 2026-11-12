"""Speech-to-text across providers.

OpenAI: `gpt-transcribe` via /v1/audio/transcriptions (no speaker labels).
Gemini: Interactions API with a JSON schema, so speaker turns come back structured.
"""

from __future__ import annotations

import base64

from pydantic import BaseModel

from synneon.schemas.results import TranscriptionResult, TranscriptSegment
from synneon.schemas.tts import STTConfig

# Gemini inline requests are capped at 20 MB in total; leave headroom for the prompt.
GEMINI_INLINE_LIMIT = 18 * 1024 * 1024
DEFAULT_SPEAKER = "SPEAKER_1"

_EXTENSIONS = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "audio/mp4": "mp4",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}


class _Transcript(BaseModel):
    """Structured-output schema for Gemini."""

    segments: list[TranscriptSegment]


def _base_mime(media_type: str) -> str:
    """Strip parameters such as `;codecs=opus` that browsers add to MediaRecorder types."""
    return media_type.split(";", 1)[0].strip().lower() or "audio/wav"


async def transcribe(audio: bytes, media_type: str, cfg: STTConfig) -> TranscriptionResult:
    mime = _base_mime(media_type)
    if cfg.provider == "openai":
        segments = await _transcribe_openai(audio, mime, cfg)
    elif cfg.provider == "google":
        segments = await _transcribe_gemini(audio, mime, cfg)
    else:
        raise ValueError(f"Unsupported STT provider: {cfg.provider}")
    text = "\n".join(f"{s.speaker}: {s.text}" for s in segments)
    return TranscriptionResult(text=text, segments=segments)


async def _transcribe_openai(audio: bytes, mime: str, cfg: STTConfig) -> list[TranscriptSegment]:
    from openai import AsyncOpenAI, omit

    filename = f"audio.{_EXTENSIONS.get(mime, 'wav')}"
    async with AsyncOpenAI() as client:
        result = await client.audio.transcriptions.create(
            model=cfg.model,
            file=(filename, audio, mime),
            prompt=cfg.prompt or omit,
            languages=cfg.languages or omit,
        )
    text = result.text.strip()
    return [TranscriptSegment(speaker=DEFAULT_SPEAKER, text=text)] if text else []


async def _transcribe_gemini(audio: bytes, mime: str, cfg: STTConfig) -> list[TranscriptSegment]:
    import io

    from google import genai

    hints = f" Expected languages: {', '.join(cfg.languages)}." if cfg.languages else ""
    request = {
        "model": cfg.model,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": _Transcript.model_json_schema(),
        },
    }
    if cfg.prompt:
        request["system_instruction"] = cfg.prompt
    if cfg.thinking:
        request["generation_config"] = {"thinking_level": cfg.thinking}

    async with genai.Client().aio as client:
        if len(audio) > GEMINI_INLINE_LIMIT:
            uploaded = await client.files.upload(file=io.BytesIO(audio), config={"mime_type": mime})
            audio_part = {"type": "audio", "uri": uploaded.uri, "mime_type": uploaded.mime_type or mime}
        else:
            audio_part = {"type": "audio", "data": base64.b64encode(audio).decode(), "mime_type": mime}
        request["input"] = [
            {"type": "text", "text": f"Transcribe this audio verbatim, one segment per speaker turn.{hints}"},
            audio_part,
        ]
        interaction = await client.interactions.create(**request)
    return _Transcript.model_validate_json(interaction.output_text or '{"segments": []}').segments
