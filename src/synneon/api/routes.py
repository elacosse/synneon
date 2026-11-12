import base64
import binascii
import re
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from rich.console import Console

from synneon.agents.base import CognitiveSteeringAgent, VLMImage2ContextAgent, VLMNeon2ContextAgent
from synneon.agents.expansion import parse_expansion
from synneon.api.broadcast import broadcast as ws_broadcast
from synneon.api.config import get_agent_settings, get_stt_settings, get_tts_settings
from synneon.api.dependencies import ContextStoreDep
from synneon.schemas.context import EmotionContext, EmotionLabel
from synneon.schemas.memory import ConversationEntry, InferenceRecord, InferenceType, MemoryEntry, MemoryType
from synneon.schemas.neons import NeonCapturePayload
from synneon.schemas.requests import (
    Audio2TextRequest,
    CognitiveSteeringInferenceRequest,
    Image2ContextRequest,
    NeonCaptureInferenceRequest,
    SpeakRequest,
    UtteranceRequest,
)
from synneon.schemas.results import (
    ExpansionResult,
    Image2ContextResult,
    TranscriptionResult,
)
from synneon.services.audio.stt import transcribe
from synneon.services.audio.tts import synthesize
from synneon.services.latents.loader import load_latent_config
from synneon.services.memory.store import (
    derive_summary,
    format_conversation,
    format_memories,
)

# Setup Router
router = APIRouter()
console = Console()

# Cache for initialized agents to avoid redundant setup
_agents: dict[str, CognitiveSteeringAgent[str]] = {}
_vlm_agents: dict[str, VLMImage2ContextAgent[Image2ContextResult]] = {}
_neon_agents: dict[str, VLMNeon2ContextAgent[Image2ContextResult]] = {}


def get_or_create_agent(profile: str) -> CognitiveSteeringAgent[str]:
    """Retrieve or initialize a CognitiveSteeringAgent for the given profile."""
    if profile not in _agents:
        try:
            settings = get_agent_settings(profile)
            latent_cfg = load_latent_config("default")
            # We use str output type for the baseline abbreviation expansion
            _agents[profile] = CognitiveSteeringAgent(settings=settings, output_type=str, latent_config=latent_cfg)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize agent for profile {profile}:[/bold red] {e}")
            raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}") from e
    return _agents[profile]


def get_or_create_vlm_agent(profile: str) -> VLMImage2ContextAgent[Image2ContextResult]:
    """Retrieve or initialize a VLMImage2ContextAgent for the given profile."""
    if profile not in _vlm_agents:
        try:
            settings = get_agent_settings(profile)
            # We use Image2ContextResult output type for structured VLM extraction
            _vlm_agents[profile] = VLMImage2ContextAgent(settings=settings, output_type=Image2ContextResult)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize VLM agent for profile {profile}:[/bold red] {e}")
            raise HTTPException(status_code=500, detail=f"VLM Agent initialization failed: {str(e)}") from e
    return _vlm_agents[profile]


def get_or_create_neon_agent(profile: str) -> VLMNeon2ContextAgent[Image2ContextResult]:
    """Retrieve or initialize a VLMNeon2ContextAgent for the given profile."""
    if profile not in _neon_agents:
        try:
            settings = get_agent_settings(profile)
            _neon_agents[profile] = VLMNeon2ContextAgent(settings=settings, output_type=Image2ContextResult)
        except Exception as e:
            console.print(f"[bold red]Failed to initialize VLM Neon agent for profile {profile}:[/bold red] {e}")
            raise HTTPException(status_code=500, detail=f"VLM Neon Agent initialization failed: {str(e)}") from e
    return _neon_agents[profile]


async def _session_id(store) -> str:
    """Current session for the store's user; starts one if that user has none yet."""
    session_id = await store.get_current_session_id()
    if not session_id:
        session_id = str(uuid.uuid4())
        await store.start_session(session_id)
    return session_id


def _warn_record_not_saved(endpoint: str, error: Exception) -> None:
    """A failed InferenceRecord save must not fail the request, but it must not be silent either."""
    console.print(f"[bold yellow]Could not save the inference record for {endpoint}:[/bold yellow] {error}")


def _decode_base64(data: str, what: str) -> bytes:
    try:
        return base64.b64decode(data)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 in {what}: {e}") from e


# ============================================================
# Inference Endpoints
# ============================================================


@router.post("/inference/expand", response_model=ExpansionResult)
async def expand_abbreviation(request: CognitiveSteeringInferenceRequest, store: ContextStoreDep):
    """
    Perform abbreviation expansion using cognitive steering.

    Injects context into the system and user prompts and returns parsed expansion candidates.
    Enriches the prompt with recent conversation history and cross-session memories from the store.
    """
    cached = _agents.get(request.profile)
    settings = cached.settings if cached else get_agent_settings(request.profile)
    agent = None if settings.provider == "alignenv" else get_or_create_agent(request.profile)

    try:
        # Fetch stored context for LLM enrichment
        session_id = await _session_id(store)
        history = await store.get_recent_conversation(session_id)
        memories = await store.get_recent_memories()

        conversation_str = format_conversation(history)
        memories_str = format_memories(memories)

        if settings.provider == "alignenv":
            from synneon.schemas.results import ExpansionCandidate, ExpansionResult

            # 1. Regex Tokenizer
            # Matches any bracketed word OR any single non-whitespace character
            tokens = re.findall(r"\[.*?\]|[^\[\]\s]", request.tokens)
            modal_abbreviation = " ".join(tokens)

            # 2. Format History for AlignEnv API (A = User, B = Interlocutor)
            modal_history = []
            for entry in history:
                speaker = "A" if entry.role == "user" else "B"
                modal_history.append({"speaker": speaker, "text": entry.text})

            # 3. Format Visual Context into Flat Referents
            referents = []
            if request.context.parsed_visual:
                if request.context.parsed_visual.focused_object:
                    referents.append(request.context.parsed_visual.focused_object.capitalize())
                for obj in request.context.parsed_visual.detected_objects:
                    referents.append(obj.capitalize())

            # 4. Construct Payload & Execute Call
            payload = {"abbreviation": modal_abbreviation, "history": modal_history, "referents": referents}

            await ws_broadcast({"event": "expand_inference_started"})
            system_prompt = "Delegated to AlignEnv Modal API"
            user_prompt = f"Payload: {payload}"

            if not settings.endpoint_url:
                raise HTTPException(
                    status_code=500, detail=f"Profile '{request.profile}' has no endpoint_url configured."
                )
            endpoint_path = "/expand/more" if request.more_candidates else "/expand"
            try:
                async with httpx.AsyncClient() as client:
                    res = await client.post(
                        f"{settings.endpoint_url.rstrip('/')}{endpoint_path}",
                        json=payload,
                        timeout=30.0,
                    )
                    res.raise_for_status()
                    data = res.json()

                candidates = [
                    ExpansionCandidate(text=c, rank=i + 1) for i, c in enumerate(data.get("candidates", [])[:5])
                ]
                expansion = ExpansionResult(candidates=candidates)

            except Exception as e:
                console.print(f"[bold red]Modal API inference error:[/bold red] {e}")
                raise HTTPException(status_code=502, detail=f"Modal API failed: {str(e)}") from e

            await ws_broadcast({"event": "expand_inference_complete"})

        else:
            prompt_kwargs = dict(
                tokens=request.tokens,
                conversation_history=conversation_str,
                recent_memories=memories_str,
            )
            system_prompt, user_prompt = agent.builder.build_prompts(request.context, **prompt_kwargs)

            await ws_broadcast({"event": "expand_inference_started"})
            run_result = await agent.run(request.context, **prompt_kwargs)
            raw_text = str(run_result.output)
            expansion = parse_expansion(raw_text)
            await ws_broadcast({"event": "expand_inference_complete"})

        await store.save_last_expand_context(
            session_id,
            {
                "tokens": request.tokens,
                "steering_context": request.context.model_dump(),
                "conversation_history": conversation_str,
                "recent_memories": memories_str,
                "model": settings.model,
                "profile": request.profile,
                "candidates": expansion.model_dump().get("candidates", []),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
        )

        try:
            record = InferenceRecord(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                inference_type=InferenceType.TEXT2TEXT,
                endpoint="expand",
                profile=request.profile,
                agent_config=settings.model_dump(),
                context_snapshot={
                    "tokens": request.tokens,
                    "steering_context": request.context.model_dump(),
                    "conversation_history": conversation_str,
                    "recent_memories": memories_str,
                },
                context={"system_prompt": system_prompt, "user_prompt": user_prompt},
                result_snapshot=expansion.model_dump(),
            )
            await store.save_inference_record(record)
        except Exception as e:
            _warn_record_not_saved("expand", e)

        expansion.system_prompt = system_prompt
        expansion.user_prompt = user_prompt
        return expansion

    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[bold red]Inference error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}") from e


@router.post("/inference/image2context", response_model=Image2ContextResult)
async def image_to_context(
    store: ContextStoreDep,
    file: UploadFile = File(...),
    profile: str = Query("image2context", description="The VLM agent profile to use."),
    save_to_memory: bool = Query(False, description="If True, persist the result as a MemoryEntry."),
):
    """
    Perform image-to-context inference using a VLM agent.
    """
    agent = get_or_create_vlm_agent(profile)

    try:
        content = await file.read()
        media_type = file.content_type or "image/jpeg"

        run_result = await agent.run(content, media_type=media_type)
        result = run_result.output
        session_id = await _session_id(store)

        if save_to_memory:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                timestamp=time.time(),
                type=MemoryType.VLM_IMAGE_CONTEXT,
                summary=derive_summary(result),
                data=result.model_dump(),
            )
            await store.save_memory(entry)

        record_endpoint = "image2context"
        try:
            record = InferenceRecord(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                inference_type=InferenceType.IMAGE2TEXT,
                endpoint="image2context",
                profile=profile,
                agent_config=agent.settings.model_dump(),
                context_snapshot={"media_type": media_type},
                context={"system_prompt": agent.settings.system_prompt, "user_prompt": agent._format_prompt()},
                result_snapshot=result.model_dump(),
            )
            await store.save_inference_record(record)
        except Exception as e:
            _warn_record_not_saved(record_endpoint, e)

        return result

    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[bold red]VLM Inference error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"VLM Inference failed: {str(e)}") from e


@router.post("/inference/image2context/base64", response_model=Image2ContextResult)
async def image_to_context_base64(request: Image2ContextRequest, store: ContextStoreDep):
    """
    Perform image-to-context inference using base64 encoded image data.
    """
    agent = get_or_create_vlm_agent(request.profile)

    try:
        run_result = await agent.run(request.image_base64, media_type=request.media_type)
        result = run_result.output
        session_id = await _session_id(store)

        if request.save_to_memory:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                timestamp=time.time(),
                type=MemoryType.VLM_IMAGE_CONTEXT,
                summary=derive_summary(result),
                data=result.model_dump(),
            )
            await store.save_memory(entry)

        record_endpoint = "image2context"
        try:
            record = InferenceRecord(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                inference_type=InferenceType.IMAGE2TEXT,
                endpoint="image2context",
                profile=request.profile,
                agent_config=agent.settings.model_dump(),
                context_snapshot={"media_type": request.media_type},
                context={"system_prompt": agent.settings.system_prompt, "user_prompt": agent._format_prompt()},
                result_snapshot=result.model_dump(),
            )
            await store.save_inference_record(record)
        except Exception as e:
            _warn_record_not_saved(record_endpoint, e)

        return result

    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[bold red]VLM Base64 Inference error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"VLM Base64 Inference failed: {str(e)}") from e


@router.post("/inference/neon-context", response_model=Image2ContextResult)
async def neon_capture_to_context(request: NeonCaptureInferenceRequest, store: ContextStoreDep):
    """
    Perform multimodal inference on a comprehensive Neon capture payload.
    """
    agent = get_or_create_neon_agent(request.profile)

    try:
        # Decode optional base64 attachments back into raw bytes
        scene_bytes = (
            _decode_base64(request.scene_image_base64, "scene_image_base64") if request.scene_image_base64 else None
        )
        audio_bytes = _decode_base64(request.audio_base64, "audio_base64") if request.audio_base64 else None

        video_bytes_list = []
        for ts, b64_frame in request.video_base64:
            video_bytes_list.append((ts, _decode_base64(b64_frame, "video_base64")))

        # Construct the raw payload schema expected by the agent
        payload = NeonCapturePayload(
            scene_image=scene_bytes,
            video=video_bytes_list,
            gaze_points=request.gaze_points,
            pupil_data=request.pupil_data,
            audio=audio_bytes,
            blink_config=request.blink_config,
            fixation_config=request.fixation_config,
            saliency_config=request.saliency_config,
            recording_config=request.recording_config,
        )

        # Models without audio input (e.g. OpenAI) get a transcript of the clip instead.
        audio_transcript = None
        if payload.audio and not agent.settings.accepts_audio:
            transcript = await transcribe(payload.audio, "audio/wav", get_stt_settings(request.stt_profile))
            audio_transcript = transcript.text or None

        # Run the agent
        neon_user_prompt = agent._format_prompt(payload, audio_transcript=audio_transcript)
        run_result = await agent.run(payload, audio_transcript=audio_transcript)
        result = run_result.output
        session_id = await _session_id(store)

        if request.save_to_memory:
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                timestamp=time.time(),
                type=MemoryType.VLM_NEON_CONTEXT,
                summary=derive_summary(result),
                data=result.model_dump(),
            )
            await store.save_memory(entry)

        record_endpoint = "neon-context"
        try:
            record = InferenceRecord(
                id=str(uuid.uuid4()),
                user_id=store._user_id,
                session_id=session_id,
                inference_type=InferenceType.IMAGE2TEXT,
                endpoint="neon-context",
                profile=request.profile,
                agent_config=agent.settings.model_dump(),
                context_snapshot={
                    "has_scene_image": request.scene_image_base64 is not None,
                    "has_audio": request.audio_base64 is not None,
                    "video_frames_count": len(request.video_base64),
                    "gaze_points_count": len(request.gaze_points) if request.gaze_points else 0,
                    "has_pupil_data": request.pupil_data is not None,
                },
                context={"system_prompt": agent.settings.system_prompt, "user_prompt": neon_user_prompt},
                result_snapshot=result.model_dump(),
            )
            await store.save_inference_record(record)
        except Exception as e:
            _warn_record_not_saved(record_endpoint, e)

        return result

    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[bold red]Neon Context Inference error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Neon Context Inference failed: {str(e)}") from e


# ============================================================
# Audio Endpoints
# ============================================================


async def _save_transcript(store, result: TranscriptionResult) -> None:
    """Persist each transcript segment as an ambient ('other') conversation entry."""
    session_id = await _session_id(store)
    now = time.time()
    for i, segment in enumerate(result.segments):
        entry = ConversationEntry(
            user_id=store._user_id,
            session_id=session_id,
            role="other",
            text=segment.text,
            timestamp=now + i * 0.001,
        )
        await store.save_utterance(entry)


@router.post("/audio/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(
    store: ContextStoreDep,
    file: UploadFile = File(...),
    profile: str = Query("default", description="The STT configuration profile to use."),
    save_to_memory: bool = Query(
        False, description="If True, persist transcript segments as ConversationEntry records."
    ),
):
    """Transcribe an uploaded audio file with the profile's provider (OpenAI or Gemini)."""
    try:
        settings = get_stt_settings(profile)
        content = await file.read()
        result = await transcribe(content, file.content_type or "audio/wav", settings)
        if save_to_memory:
            await _save_transcript(store, result)
        return result
    except Exception as e:
        console.print(f"[bold red]Transcription error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}") from e


@router.post("/audio/transcribe/base64", response_model=TranscriptionResult)
async def transcribe_audio_base64(request: Audio2TextRequest, store: ContextStoreDep):
    """Transcribe a base64 encoded audio snippet with the profile's provider (OpenAI or Gemini)."""
    try:
        settings = get_stt_settings(request.profile)
        audio_bytes = _decode_base64(request.audio_base64, "audio_base64")
        result = await transcribe(audio_bytes, request.media_type, settings)
        if request.save_to_memory:
            await _save_transcript(store, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[bold red]Base64 Transcription error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"Base64 Transcription failed: {str(e)}") from e


@router.post("/audio/speak")
async def speak_text(
    body: SpeakRequest | None = Body(None, description="Preferred: send the text in a JSON body."),
    text: str | None = Query(
        None, description="Deprecated: text in the URL ends up in access logs. Send a JSON body instead."
    ),
    profile: str = Query("default", description="The TTS configuration profile to use."),
    voice: str | None = Query(None, description="Voice name. Overrides the profile config."),
    model: str | None = Query(None, description="TTS model. Overrides the profile config."),
    emotion_label: str | None = Query(
        None, description="Emotion label (e.g. 'happy', 'sad'). Adds a delivery hint to the TTS request."
    ),
    emotion_intensity: float = Query(0.0, ge=0.0, le=1.0, description="Emotion intensity 0–1."),
):
    """
    Synthesise text to speech with the profile's provider and return WAV audio bytes.
    Optionally steers delivery with an emotion style hint when emotion_label is provided.

    Send the request as a JSON body (`SpeakRequest`). Query parameters still work for existing clients.
    """
    if body is not None:
        text, profile = body.text, body.profile
        voice, model = body.voice, body.model
        emotion_label, emotion_intensity = body.emotion_label, body.emotion_intensity
    if not text:
        raise HTTPException(status_code=422, detail="`text` is required (JSON body or query parameter).")

    try:
        settings = get_tts_settings(profile)
        overrides = {k: v for k, v in {"voice": voice, "model": model}.items() if v}
        if overrides:
            settings = settings.model_copy(update=overrides)

        style = None
        if emotion_label is not None:
            try:
                latent_cfg = load_latent_config("default")
                emotion = EmotionContext(label=EmotionLabel(emotion_label), intensity=emotion_intensity)
                style = emotion.tts_style(latent_cfg.emotion) or None
            except ValueError:
                pass  # Unknown emotion label — speak without a style hint

        audio_bytes = await synthesize(text, settings, style=style)
        return Response(content=audio_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        console.print(f"[bold red]TTS error:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}") from e


# ============================================================
# Context / Memory Query Endpoints
# ============================================================


@router.post("/context/conversation", response_model=list[ConversationEntry], status_code=201)
async def seed_conversation(entries: list[ConversationEntry], store: ContextStoreDep):
    """Directly save a list of ConversationEntry records. Intended for development/seeding."""
    session_id = await _session_id(store)
    now = time.time()
    saved: list[ConversationEntry] = []
    for i, entry in enumerate(entries):
        entry = entry.model_copy(
            update={
                "user_id": store._user_id,
                "session_id": session_id,
                "timestamp": entry.timestamp if entry.timestamp else now + i * 0.1,
            }
        )
        await store.save_utterance(entry)
        saved.append(entry)
    return saved


@router.post("/context/utterance", response_model=ConversationEntry, status_code=201)
async def save_utterance(request: UtteranceRequest, store: ContextStoreDep):
    """Save a single utterance to the conversation store for the current session."""
    session_id = await _session_id(store)
    entry = ConversationEntry(
        user_id=store._user_id,
        session_id=session_id,
        role=request.role,
        text=request.text,
        source=request.source,
    )
    await store.save_utterance(entry)
    return entry


@router.get("/context/conversation", response_model=list[ConversationEntry])
async def get_conversation(
    store: ContextStoreDep,
    session_id: str | None = Query(None, description="Session ID. Omit to use the current session."),
    limit: int = Query(20, ge=1, le=200, description="Maximum number of utterances to return."),
):
    """Return recent conversation entries for a session, oldest first."""
    resolved_session = session_id or await store.get_current_session_id()
    if not resolved_session:
        return []
    return await store.get_recent_conversation(resolved_session, limit=limit)


@router.get("/context/memories", response_model=list[MemoryEntry])
async def get_memories(
    store: ContextStoreDep,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of memory entries to return."),
):
    """Return recent cross-session memory entries, oldest first."""
    return await store.get_recent_memories(limit=limit)


@router.get("/context/inferences", response_model=list[InferenceRecord])
async def get_inferences(
    store: ContextStoreDep,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of inference records to return."),
    inference_type: InferenceType | None = Query(
        None, description="Filter by inference type: text2text or image2text."
    ),
):
    """Return recent inference records, oldest first. Optionally filter by inference_type."""
    return await store.get_recent_inferences(limit=limit, inference_type=inference_type)


@router.get("/context/inferences/{record_id}", response_model=InferenceRecord)
async def get_inference(
    record_id: str,
    store: ContextStoreDep,
    inference_type: InferenceType = Query(..., description="Inference type required to resolve the record key."),
):
    """Return a single inference record by ID."""
    record = await store.get_inference_record(record_id, inference_type)
    if not record:
        raise HTTPException(status_code=404, detail=f"Inference record '{record_id}' not found")
    return record


@router.get("/context/expand-context")
async def get_expand_context(
    store: ContextStoreDep,
    user_id: str = Query(..., description="User ID to scope the lookup."),
    session_id: str = Query(..., description="Session ID to scope the lookup."),
):
    """Return the last expand inference context stored in Redis for the given user and session."""
    data = await store.get_last_expand_context(user_id, session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="No expand context found")
    return data


# ============================================================
# User Bootstrap
# ============================================================


@router.post("/context/users", status_code=201)
async def bootstrap_user(store: ContextStoreDep, user_id: str | None = None):
    """Bootstrap a user profile and start a fresh session. Idempotent — safe to call multiple times."""
    if user_id:
        store._user_id = user_id
    await store.bootstrap()
    session_id = str(uuid.uuid4())
    await store.start_session(session_id)
    return {"user_id": store._user_id, "session_id": session_id}


# ============================================================
# Health Check
# ============================================================


@router.get("/viz/image")
async def get_viz_image(example: str = Query(...), file: str = Query(...)):
    """Serve a visualization image or video from an example capture directory.

    Allowed files: saliency_map.png, saliency_overlay.png, saliency_peaks.png,
    gaze_trajectory.png, gaze_fixations.png, video.mp4
    """
    _ALLOWED_FILES = {
        "saliency_map.png",
        "saliency_overlay.png",
        "saliency_peaks.png",
        "gaze_trajectory.png",
        "gaze_fixations.png",
        "video.mp4",
    }
    if file not in _ALLOWED_FILES:
        raise HTTPException(status_code=400, detail=f"File '{file}' is not a permitted visualization file.")

    # Resolve relative to the working directory first, then the project root
    # (routes.py lives at src/synneon/api/routes.py → 4 levels up is the package root).
    # The resolved file must stay inside one of those two roots.
    pkg_root = Path(__file__).parent.parent.parent.parent.resolve()
    roots = [Path.cwd().resolve(), pkg_root]
    candidate = None
    for root in roots:
        resolved = (root / example / file).resolve()
        if resolved.is_relative_to(root) and resolved.exists():
            candidate = resolved
            break

    if candidate is None:
        raise HTTPException(status_code=404, detail=f"File not found: {example}/{file}")

    media_type = "video/mp4" if file.endswith(".mp4") else "image/png"
    return FileResponse(str(candidate), media_type=media_type)


_VIZ_PNG_FILES = {
    "saliency_map.png",
    "saliency_overlay.png",
    "saliency_peaks.png",
    "gaze_trajectory.png",
    "gaze_fixations.png",
}


@router.get("/viz/examples")
async def list_viz_examples():
    """List example capture directories that have pre-generated visualization PNGs."""
    pkg_root = Path(__file__).parent.parent.parent.parent
    examples_dir = pkg_root / "examples"
    if not examples_dir.is_dir():
        return []

    results = []
    for subdir in sorted(examples_dir.iterdir()):
        if not subdir.is_dir():
            continue
        present = [f for f in _VIZ_PNG_FILES if (subdir / f).exists()]
        if present:
            results.append(
                {
                    "name": subdir.name,
                    "path": f"examples/{subdir.name}",
                    "files": sorted(present),
                }
            )
    return results


@router.get("/health")
async def health_check(store: ContextStoreDep):
    """Service health check."""
    try:
        await store._redis.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "unavailable"

    return {
        "status": "ok",
        "profiles_loaded": list(_agents.keys()),
        "redis": redis_status,
    }
