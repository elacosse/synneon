# Development

## Setting Up uv

This project is set up to use [uv](https://docs.astral.sh/uv/) to manage Python and
dependencies. First, be sure you
[have uv installed](https://docs.astral.sh/uv/getting-started/installation/).

Then [fork the synneon/synneon
repo](https://github.com/synneon/synneon/fork) (having your own
fork will make it easier to contribute) and
[clone it](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

## Basic Developer Workflows

The `Makefile` offers shortcuts to `uv` commands for developer convenience. CI is
manual-only (start it from the GitHub Actions tab or with `gh workflow run ci.yml`) and runs the
same `make lint` and `pytest` checks.

```shell
# First, install all dependencies and the pre-commit hooks.
make setup

# Run all read-only checks and tests:
make check

# Read-only linting, formatting, spelling, and YAML checks:
make lint

# Apply formatting and auto-fixes:
make format

# Type check with ty (not yet part of `make check`):
make typecheck

# Run tests:
make test

# Build a wheel and source distribution with locked build dependencies:
make build

# Delete all the build artifacts:
make clean

# Upgrade dependencies to compatible versions:
make upgrade

# To run tests by hand:
uv run pytest   # all tests
uv run pytest -s tests/module/test_some_file.py  # one test file, showing outputs

# Build and install current dev executables, to let you use your dev copies
# as local tools:
uv tool install --editable .

# Dependency management directly with uv:
# Add a new dependency:
uv add package_name
# Add a development dependency:
uv add --dev package_name
# Update to latest compatible versions (including dependencies on git repos):
uv sync --upgrade
# Update a specific package:
uv lock --upgrade-package package_name
# Update dependencies on a package:
uv add package_name@latest

# Run a shell within the Python environment:
uv venv
source .venv/bin/activate
```

See [uv docs](https://docs.astral.sh/uv/) for details.

## Run the CLI

```shell
# Show all top-level commands
uv run synneon --help

# Gesture detection (Neon device)
uv run synneon gesture compass --config configs/neons/default.toml

# Gesture detection (synthetic data, no device required)
uv run synneon gesture compass --dry-run

# Mouse-pointer gesture detection (no device required)
uv run synneon gesture mouse

# Compass keyboard — arrow-key driven
uv run synneon keyboard type-text

# Compass keyboard — Neon eye-tracker driven
uv run synneon keyboard type-text --neon --config configs/neons/default.toml

# Compass keyboard — with TTS speech output on confirmation
uv run synneon keyboard type-text --speak

# Live gaze overlay
uv run synneon display gaze --config configs/neons/default.toml

# Double-blink snapshot capture
uv run synneon capture --config configs/neons/default.toml

# Combined blink capture + gesture detection
uv run synneon capture-and-gesture --config configs/neons/default.toml

# Video app: Record full payload and send to REST API for VLM analysis
uv run synneon video capture-all

# Video app: Disable REST API and save .mp4, .wav, and .csv locally
uv run synneon video capture-all --no-analyze

# Real-time pupillometry plot
uv run synneon pupil --config configs/neons/default.toml

# Gaze calibration
uv run synneon calibrate --config configs/neons/default.toml

# REST API server
uv run python -m synneon.server

# Dashboard (REST API + WebSocket + React UI on port 8765); build the UI first
(cd src/synneon/ux && npm ci && npm run build)
uv run synneon app

# Neon client feeding the dashboard (drop --mock for a real device)
uv run synneon client --mock --pupil

# Offline pipeline over the recorded example capture (add --save to persist to Redis)
uv run synneon example run --example examples/capture_sample_v1

# Seed Redis with a synthetic session (no files needed)
uv run python -m synneon.examples.seed_redis

# Seed with a custom user ID namespace
uv run python -m synneon.examples.seed_redis -id test_abc

# Seed using real capture data
uv run python -m synneon.examples.seed_redis --data examples/capture_sample_v1
```

## Running with Docker

`docker/docker-compose.yml` runs the whole stack. The `synneon` service is the dashboard app (backend
plus the built React UI) with no hardware; `redis` and `redisinsight` are the data services. The
`neon-mock` service (mock Neon client) only starts with `--profile mock`.

```shell
# Build and start everything (http://127.0.0.1:8765, Redis Insight on http://localhost:5540)
docker compose -f docker/docker-compose.yml up -d --build

# Same, plus the mock Neon client
docker compose -f docker/docker-compose.yml --profile mock up -d --build

# Follow the app logs
docker compose -f docker/docker-compose.yml logs -f synneon

# Run a CLI command inside the container
docker compose -f docker/docker-compose.yml exec synneon synneon example run --save

# Start only Redis and Redis Insight, and run the app on the host with uv
docker compose -f docker/docker-compose.yml up -d redis redisinsight

# Stop (add -v to also delete the Redis volume)
docker compose -f docker/docker-compose.yml down
```

Things to know when changing the image:

- The build context is the repository root, filtered by `.dockerignore`. `.env` is excluded; Compose
  passes it to the container at run time through `env_file`.
- Source and `configs/` are copied in at build time, so rebuild after editing them.
- `SYNNEON_REDIS_URL` (set by Compose to `redis://redis:6379/0`) overrides `redis_url` from
  `configs/memory/default.toml`.
- The container has no display or git metadata: `PYNPUT_BACKEND=dummy` lets the CLI import `pynput`,
  and `UV_DYNAMIC_VERSIONING_BYPASS` supplies the version that normally comes from git tags.
- `evdev` (a Linux dependency of `pynput`) has no wheel and is compiled during the build, which is why
  the image installs `gcc` and kernel headers.
- The host app and the `synneon` container both bind port 8765; run one or the other.

## Pre-commit Hooks

This project uses `pre-commit` to enforce code style and quality. To use it, you need to install the hooks once:

```shell
uv run pre-commit install
```

Now, the hooks will run automatically every time you make a commit. You can also run them manually:

```shell
uv run pre-commit run --all-files
```


## Directory Structure

```
synneon/
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock, uv.toml            # Locked dependencies and uv settings
├── README.md                   # User guide, including the walkthrough
├── ARCHITECTURE.md             # System architecture and dataflow diagrams
├── AGENTS.md                   # AI agent coding instructions
├── CHANGELOG.md                # Release notes and history
├── Makefile                    # Developer workflow shortcuts
├── .env                        # Secrets (API keys); copy from .env.example, never commit
├── .dockerignore               # Keeps .env, .git, and build artifacts out of the image
│
├── configs/                    # TOML configuration files
│   ├── interpreter/agent.toml  # LLM agent profiles (OpenAI and Gemini variants, alignenv)
│   ├── latents/default.toml    # Emotion intensity bands and TTS style template
│   ├── memory/default.toml     # Redis URL, context windows, TTL
│   ├── neons/default.toml      # Device, blink, calibration, gestures, fixation, saliency, recording
│   ├── stt/default.toml        # Speech-to-text profiles
│   └── tts/default.toml        # Text-to-speech profiles
│
├── docker/
│   ├── Dockerfile              # Two-stage image: Node builds the UI, uv installs Python deps
│   └── docker-compose.yml      # synneon + redis + redisinsight
│
├── examples/
│   └── capture_sample_v1/      # Recorded capture for `synneon example run`
│
├── data/                       # Artifacts (not code)
│   ├── examples/               # Example input data (images and yaml context)
│   └── steering_vectors/       # Reserved for steering vector .pt files
│
├── agents/                     # Agent design docs (no code)
│   ├── RePPIT/
│   └── research/
│
├── tests/                      # Test suite, mirroring src/synneon/
│   ├── agents/                 # Agent wrappers and prompt building
│   ├── api/                    # Routes, context endpoints, alignenv delegation
│   ├── apps/                   # Config loaders and the example runner
│   ├── input/                  # Compass and gesture detector
│   ├── neon/                   # Synthetic gaze stream
│   ├── schemas/                # Pydantic models
│   └── services/               # Store, STT, TTS, latents, pupillometry
│
└── src/
    └── synneon/                # Main package
        ├── __init__.py
        ├── synneon.py          # CLI entrypoint — assembles Typer app from apps/
        ├── server.py           # FastAPI app setup (REST only, port 8000)
        │
        ├── agents/             # pydantic-ai agent wrappers
        │   ├── base.py         # CognitiveSteeringAgent, VLMImage2ContextAgent, VLMNeon2ContextAgent
        │   └── expansion.py    # create_expansion_agent, parse_expansion
        │
        ├── apps/               # CLI command modules
        │   ├── app.py          # `synneon app` — dashboard: REST + /ws + built React UI (no hardware)
        │   ├── neon_client.py  # `synneon client` — Neon device or mock → /ws/neon + capture POSTs
        │   ├── run_example.py  # `synneon example run` — offline pipeline runner
        │   ├── pupil.py        # `synneon pupil` — real-time pupillometry plot
        │   ├── display.py      # `synneon display gaze` — live video + gaze overlay
        │   ├── capture.py      # `synneon capture` — double-blink snapshot
        │   ├── combined.py     # `synneon capture-and-gesture` — blink + gesture
        │   ├── audio/
        │   │   └── capture.py  # Local audio capture logic
        │   ├── gesture/
        │   │   ├── __init__.py # gesture_app Typer sub-app
        │   │   ├── compass.py  # `synneon gesture compass` — eye-gesture navigation
        │   │   └── mouse.py    # `synneon gesture mouse` — mouse-as-gaze input
        │   ├── video/
        │   │   └── app.py      # `synneon video` — capture-all / saliency
        │   ├── keyboard/
        │   │   ├── __init__.py # keyboard_app Typer sub-app
        │   │   ├── _ui.py      # Design tokens, NeonState, all Rich rendering
        │   │   └── command.py  # Key reading, HTTP, Neon pipeline, main loop
        │   └── shared/
        │       ├── config.py   # Config loaders (neon, gestures, blink)
        │       ├── blink.py    # BlinkManager, double-blink capture + VLM dispatch
        │       └── viewer.py   # Tkinter capture viewer subprocess
        │
        ├── examples/           # Runnable example scripts
        │   ├── agent.py        # Run CognitiveSteeringAgent directly
        │   ├── agent_api_client.py
        │   ├── call_service.py
        │   ├── image2context_agent.py
        │   ├── seed_redis.py   # Seed Redis with a full simulated session (supports -id, --data)
        │   ├── test_all_routes.py # Run full integration test suite against API
        │   └── tts_api_client.py
        │
        ├── input/              # Input processing
        │   ├── compass.py      # Compass abbreviation expander
        │   ├── detector.py     # Gesture detection logic
        │   ├── gestures.py     # Gesture primitives (legacy)
        │   ├── mouse.py        # Mouse-pointer gaze stream
        │   └── tts.py          # Local TTS (macOS say)
        │
        ├── neon/               # Pupil Labs Neon device integration
        │   ├── calibrate.py    # `synneon calibrate` command
        │   ├── session.py      # Device connection + stream helpers
        │   └── synthetic.py    # Synthetic gaze stream for dry-run mode
        │
        ├── api/                # HTTP / WebSocket layer
        │   ├── routes.py       # REST endpoints
        │   ├── dependencies.py # ContextStoreDep — scoped per user_id query param
        │   ├── config.py       # Agent/STT/TTS/memory settings loaders
        │   ├── broadcast.py    # WebSocket broadcast hook used by routes
        │   └── sockets.py      # WebSocket hub (/ws for UI, /ws/neon for client)
        │
        ├── schemas/            # Pydantic data models
        │   ├── agents.py
        │   ├── context.py
        │   ├── gestures.py
        │   ├── latents.py      # LatentConfig, EmotionLatentConfig
        │   ├── memory.py       # UserProfile, ConversationEntry, MemoryEntry, InferenceRecord, MemoryConfig
        │   ├── neons.py        # NeonConfig, GazeSample, BlinkConfig, FixationConfig, SaliencyConfig, NeonCapturePayload
        │   ├── requests.py
        │   ├── results.py
        │   └── tts.py          # STTConfig, TTSConfig
        │
        ├── services/           # Heavy compute layers
        │   ├── audio/
        │   │   ├── stt.py      # Speech-to-text (OpenAI or Gemini)
        │   │   └── tts.py      # Text-to-speech (OpenAI Realtime or Gemini)
        │   ├── latents/
        │   │   └── loader.py   # Loads configs/latents/<name>.toml
        │   ├── memory/
        │   │   └── store.py    # ContextStore — Redis sorted-set schema, TTL, prompt formatters
        │   ├── inference/      # LLM & steering logic (stubs)
        │   │   ├── engine.py
        │   │   ├── hooks.py
        │   │   └── loader.py
        │   ├── perception/
        │   │   └── vision.py   # Gaze saliency, peaks, trajectory, I-DT fixations
        │   └── physiology/
        │       ├── emotion.py  # (stub)
        │       └── pupilometry.py
        │
        ├── ux/                 # React dashboard (Vite)
        │   ├── app.jsx         # All dashboard panels
        │   ├── main.jsx, index.html, vite.config.js
        │   └── dist/           # Build output served by `synneon app` (not committed)
        │
        └── visualization/
            └── visualizer.py   # Live video + gaze overlay renderer
```

### Video Acquisition & Saliency

The `synneon video capture-all` command in `src/synneon/apps/video/app.py` streams audio, video, gaze, and eye events. It writes synchronously to disk using a background threading queue.

The `src/synneon/services/perception/vision.py` module processes lists of frames and gaze points to generate:
- Gaze Saliency Heatmaps
- Gaze Trajectories
- I-DT Fixation Detection (circles and saccades)
- Local Maxima Peaks (red X or circles, dynamically filtered by relative peak intensity threshold)

These parameters are configured via `[saliency]`, `[fixation]`, and `[recording]` in `configs/neons/default.toml`.
