# synneon Agent Instructions

## Project Overview

`synneon` is a Cognitive Computer Interaction (CCI) toolkit designed for the [Pupil Labs Neon](https://pupil-labs.com/products/neon) eye-tracker. It enables:

- **Gaze Gestures**: Detecting directional dwell/swipe events (UP, DOWN, LEFT, RIGHT) from real-time gaze streams.
- **AAC Interfaces**: A compass-style terminal keyboard driven by eye gestures or arrow keys.
- **Cognitive Steering**: AI-assisted abbreviation expansion and context-aware inference via a local REST API.
- **Multimodal API**: REST endpoints for VLM image analysis, text-to-speech (TTS), and speech-to-text (STT).
- **Pupilometry**: Real-time pupil diameter streaming and interactive plots.

## Environment & Dependency Management

- **uv**: This project strictly uses [uv](https://github.com/astral-sh/uv) for Python management.
    - Always use `uv run`, `uv add`, and `uv sync`.
    - Do not use `pip` directly.
- **Python version**: 3.12–3.14 (`.python-version` pins 3.13 for local development).
- **uv version**: `uv.toml` requires uv ≥ 0.13.7 (the lockfile format). For direct uv commands, set
  `UV_CONFIG_FILE=uv.toml` so user-level settings are not merged; the Makefile does this already.
- **Makefile**: Shortcuts for common tasks:
    - `make setup`: Installs dependencies and the pre-commit hooks.
    - `make check` (or just `make`): Read-only lint plus the pytest suite. CI runs the same checks.
    - `make lint`: Read-only codespell, ruff check, ruff format --check, and yamllint.
    - `make format`: Applies codespell, ruff fixes, and formatting.
    - `make typecheck`: ty via `uv check`. Not yet part of `make check`, since existing code has open diagnostics.
    - `make build`: Builds wheel and sdist with the locked `build` dependency group.
- **Pre-commit**: `make setup` installs the hooks; they run the tool versions pinned in `uv.lock`.
- **CI**: `.github/workflows/ci.yml` runs lint, tests on Python 3.12–3.14, and a non-blocking `uv audit`.
  It is manual-only (Actions tab or `gh workflow run ci.yml`); add push/pull_request triggers to automate it.

## Repository Structure

```
src/synneon/
├── synneon.py          # CLI root — Typer app, sub-commands registered here
├── server.py           # FastAPI app + uvicorn CLI entry point (REST only, port 8000)
├── agents/             # pydantic-ai LLM agent wrappers
│   ├── base.py         # CognitiveSteeringContextBuilder, CognitiveSteeringAgent, VLMImage2ContextAgent, VLMNeon2ContextAgent
│   └── expansion.py    # Abbreviation expansion agent factory + output parser
├── api/
│   ├── routes.py       # All REST endpoints (/inference/*, /audio/*, /context/*, /viz/*, /health)
│   ├── config.py       # TOML profile loaders: agent, STT, TTS, memory
│   ├── dependencies.py # ContextStoreDep — store scoped per `user_id` query param
│   ├── broadcast.py    # Lets routes emit WebSocket events without importing sockets.py
│   └── sockets.py      # WebSocket hub: /ws (browsers), /ws/neon (Neon client), /viz/live-frame, /debug/fake-blink
├── apps/               # CLI command implementations (Typer)
│   ├── app.py          # `synneon app` — dashboard: REST + /ws + built React UI (port 8765), no hardware
│   ├── neon_client.py  # `synneon client` — Neon device or --mock → /ws/neon; posts captures over HTTP
│   ├── run_example.py  # `synneon example run` — offline pipeline over a recorded capture
│   ├── capture.py      # `synneon capture` — double-blink world-camera snapshot
│   ├── combined.py     # `synneon capture-and-gesture` — blink capture + gesture
│   ├── display.py      # `synneon display gaze` — live OpenCV gaze overlay
│   ├── pupil.py        # `synneon pupil` — real-time pupillometry plot
│   ├── audio/
│   │   └── capture.py  # Local audio capture
│   ├── video/
│   │   └── app.py      # `synneon video capture-all` — synchronized recording + saliency
│   ├── gesture/
│   │   ├── compass.py  # `synneon gesture compass`
│   │   └── mouse.py    # `synneon gesture mouse`
│   ├── keyboard/
│   │   ├── command.py  # `synneon keyboard type-text` — main event loop
│   │   └── _ui.py      # Pure Rich rendering layer (NeonState, all panel renderers)
│   └── shared/
│       ├── config.py   # TOML config loaders (NeonConfig, GesturesSettings, BlinkConfig)
│       ├── blink.py    # BlinkManager — double-blink detection + VLM dispatch
│       └── viewer.py   # Tkinter capture viewer subprocess (reads JSON from stdin)
├── examples/           # Runnable scripts (not CLI commands)
│   ├── agent.py                # Run CognitiveSteeringAgent directly from TOML+YAML
│   ├── agent_api_client.py     # HTTP client for /inference/* endpoints
│   ├── call_service.py         # Minimal service call example
│   ├── image2context_agent.py  # Run VLMImage2ContextAgent directly
│   ├── seed_redis.py           # Seed Redis with a simulated session
│   ├── test_all_routes.py      # Exercise every route against a running server
│   └── tts_api_client.py       # HTTP client for /audio/* endpoints
├── input/              # Core logic
│   ├── detector.py     # detect_gestures() — primary gesture detection (used by apps/)
│   ├── gestures.py     # Older gesture variant (present but not imported by apps/)
│   ├── compass.py      # Compass — alphabet navigation + abbreviation expansion state
│   ├── mouse.py        # Mouse-pointer synthetic gaze stream
│   └── tts.py          # Local TTS via macOS `say`
├── neon/               # Pupil Labs Neon hardware abstraction
│   ├── session.py      # connect(), stream_gaze(), stream_video(), stream_eye_events(), stream_audio()
│   ├── calibrate.py    # `synneon calibrate` — interactive gaze bounds calibration
│   └── synthetic.py    # synthetic_gaze_stream() for --dry-run mode
├── schemas/            # Pydantic models
│   ├── agents.py       # AgentSettings (incl. `accepts_audio`)
│   ├── context.py      # SteeringContext, EmotionContext, ParsedVisualContext, PragmaticContext
│   ├── gestures.py     # CompassSettings, GesturesSettings, GestureType, GestureEvent
│   ├── latents.py      # LatentConfig, EmotionLatentConfig (emotion → adjective / TTS style)
│   ├── memory.py       # UserProfile, ConversationEntry, MemoryEntry, InferenceRecord, MemoryConfig
│   ├── neons.py        # NeonConfig, BlinkConfig, FixationConfig, SaliencyConfig, GazeSample, NeonCapturePayload
│   ├── requests.py     # CognitiveSteeringInferenceRequest, Image2ContextRequest, NeonCaptureInferenceRequest
│   ├── results.py      # ExpansionResult, Image2ContextResult, TranscriptionResult
│   └── tts.py          # STTConfig, TTSConfig
├── services/           # Compute-heavy layers
│   ├── audio/          # stt.py / tts.py — provider-agnostic STT and TTS (OpenAI or Gemini)
│   ├── latents/        # loader.py — loads configs/latents/<name>.toml
│   ├── memory/         # store.py — ContextStore (Redis) + prompt formatting helpers
│   ├── perception/     # vision.py — gaze saliency, peaks, trajectory, I-DT fixations
│   ├── physiology/     # pupilometry.py — streaming + real-time plot; emotion.py — stub
│   └── inference/      # engine.py, hooks.py, loader.py — stubs
├── ux/                 # React dashboard (Vite): app.jsx, main.jsx, index.html; built into ux/dist
└── visualization/
    └── visualizer.py   # Visualizer — OpenCV live video + gaze crosshair overlay

configs/
├── neons/default.toml          # Device address, blink, calibration, gestures, fixation, saliency, recording
├── interpreter/agent.toml      # LLM agent profiles: default, default-gemini, image2context, image2context-gemini, alignenv
├── latents/default.toml        # Emotion intensity bands + `tts_style` template
├── memory/default.toml         # Redis URL, context windows, TTL, default user
├── stt/default.toml            # STT profiles: [stt.default] (OpenAI), [stt.gemini]
└── tts/default.toml            # TTS profiles: [tts.default] (OpenAI Realtime), [tts.gemini]

docker/
├── Dockerfile                  # Two-stage image: Node builds ux/, uv installs the locked Python deps
└── docker-compose.yml          # synneon (backend + built UI, mock hardware), redis, redisinsight

examples/capture_sample_v1/     # Recorded capture used by `synneon example run` and the README walkthrough

data/
├── examples/                   # example_1.yaml + example_1.png (test inputs)
└── steering_vectors/           # Reserved for .pt steering vector files

tests/                          # Test suite matching src/ (agents, api, apps, input, neon, schemas, services)
agents/                         # Project-level design docs and research proposals (no Python code)
```

## Key Developer Workflows

- **CLI Entrypoint**: `uv run synneon --help`
- **API Server**: `uv run python -m synneon.server`
- **Simulating Hardware**: Use `--dry-run` (synthetic gaze) or `synneon gesture mouse` (mouse-as-gaze) to test without a physical Neon device.
- **Dashboard**: `uv run synneon app` (REST + `/ws` + React UI on port 8765), plus `uv run synneon client --mock --pupil` in another terminal for live Neon data. The two are separate processes: the app never touches hardware, and the client talks to it only through `/ws/neon` and `POST /inference/neon-context`. Build the UI first with `(cd src/synneon/ux && npm ci && npm run build)`; use `npm ci`, not `npm install`, so `package-lock.json` is not rewritten.
- **Full stack in Docker**: `docker compose -f docker/docker-compose.yml up -d --build` runs Redis, Redis Insight, and the `synneon` service (the dashboard app, no hardware). Add `--profile mock` to also run the `neon-mock` client; for a real device, run `synneon client` on the host. Code and `configs/` are copied into the image, so rebuild after changing them. Run CLI commands in the container with `docker compose -f docker/docker-compose.yml exec synneon synneon ...`.
- **Redis only**: `docker compose -f docker/docker-compose.yml up -d redis redisinsight`, then run the app on the host. Do not run the host app and the `synneon` container together; both bind port 8765.
- **Offline pipeline**: `uv run synneon example run --example examples/capture_sample_v1` (add `--save` to persist to Redis).
- **Walkthrough**: the step-by-step no-hardware walkthrough is in `README.md` ("Walkthrough"); architectural breakdown and dataflow diagrams are in `ARCHITECTURE.md`.
- **Configuration**: All runtime parameters come from TOML files in `configs/`. The default Neon config is `configs/neons/default.toml`.
- **Environment Variables**: API keys (`OPENAI_API_KEY`, `GOOGLE_API_KEY`; `GEMINI_API_KEY` is accepted as a fallback) go in `.env`; `load_dotenv()` is called at startup. `SYNNEON_REDIS_URL` overrides `redis_url` from `configs/memory/default.toml`; Compose sets it to `redis://redis:6379/0`. Compose passes `.env` to the container at run time — never copy it into the image.

## AI / LLM Stack

- **pydantic-ai**: Used as the LLM agent abstraction layer (`src/synneon/agents/`).
- **Providers**: OpenAI is the default everywhere; Gemini is a first-class alternative selected by profile. Never call a vendor SDK from a route — go through `services/audio/{stt,tts}.py` or a pydantic-ai agent.
- **Abbreviation expansion**: `CognitiveSteeringAgent[str]` — `default` profile uses `openai:gpt-6.1-terra`; `default-gemini` uses `google:gemini-3.8-flash`.
- **Image-to-context**: `VLMImage2ContextAgent[Image2ContextResult]` — `image2context` profile; accepts raw bytes or base64 image.
- **STT**: `POST /audio/transcribe*` → `services/audio/stt.py`. OpenAI `gpt-transcribe` (one segment, no speaker labels) or Gemini via the Interactions API with a JSON schema (speaker turns).
- **TTS**: `POST /audio/speak` (JSON body; never put spoken text in the URL) → `services/audio/tts.py`, returns WAV. OpenAI uses the Realtime API (`gpt-realtime-2.1-mini`; the `/v1/audio/speech` models shut down 2027-01-06); Gemini uses `gemini-3.8-flash-tts`. Emotion is sent as a plain-language style hint (`tts_style` in `configs/latents/default.toml`), never as inline markup.
- **Neon capture audio**: Gemini models receive the WAV directly; OpenAI models (no audio input on the Responses API) get an STT transcript in the prompt (`AgentSettings.accepts_audio`).
- **Agent profiles** are defined in `configs/interpreter/agent.toml` under `[agent.<profile>]`. Each profile specifies `model`, `system_prompt`, `prompt_template`, and optionally `provider` and `thinking` (provider-agnostic effort: minimal | low | medium | high | xhigh | max, passed to pydantic-ai's `Thinking` capability).

## Implementation Guidelines

- **Type Safety**: Use Pydantic models for all data structures and API payloads. Add new models to `schemas/`.
- **Modularity**: CLI command logic lives in `src/synneon/apps/`; core business logic lives in `src/synneon/input/` or `src/synneon/services/`. Keep `_ui.py` files as pure rendering layers with no side effects.
- **Prompts**: Untrusted text (transcripts, VLM output, memories) goes in the user message inside tags, never in the system prompt. `CognitiveSteeringAgent.run` builds both prompts from the same arguments, so log prompts via `builder.build_prompts` with those arguments.
- **Network**: Servers bind `127.0.0.1` by default and there is no authentication; Compose publishes ports on loopback only. Containers pass `--host 0.0.0.0` explicitly.
- **Async**: Use `asyncio` for hardware streaming (`neon/session.py`) and API handling. Blocking UI work (Tkinter, Matplotlib) runs in subprocesses or daemon threads with their own event loops.
- **Configuration**: Load device/gesture/blink settings via `apps/shared/config.py`. Load agent/STT/TTS/memory settings via `api/config.py`. Never hardcode addresses or model names.
- **Containers**: The image has no display, audio device, or git metadata. `docker/Dockerfile` sets `PYNPUT_BACKEND=dummy` and `UV_DYNAMIC_VERSIONING_BYPASS`, and installs the system libraries OpenCV, sounddevice, and `evdev` need; keep those in step when adding dependencies that import at CLI startup.
- **Testing**: Add unit tests in `tests/` for any new logic, especially gesture detection and data parsing. Use `synthetic_gaze_stream()` to avoid hardware dependencies in tests.
