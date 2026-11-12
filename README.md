# SynNeon

**SynNeon** is a Cognitive Computer Interaction (CCI) toolkit for the [Pupil Labs Neon](https://pupil-labs.com/products/neon) eye-tracker. It turns raw gaze data into intent:

1. **Gaze gesture detection:** recognizes directional swipes and dwells (UP, DOWN, LEFT, RIGHT).
2. **Compass keyboard:** an AAC-style typing interface driven by eyes or arrow keys, with LLM-powered abbreviation expansion.
3. **Context server:** a local REST API for abbreviation expansion, image-to-context analysis, speech-to-text, text-to-speech, and Redis-backed session memory.
4. **Dashboard:** a React UI for live pupillometry, gaze gestures, audio capture, transcripts, and blink-triggered scene analysis.

## How It Works

An AAC user types only the first letter of each word (`G S Y T`). SynNeon picks the likely sentences using context the user never types:

1. **Capture.** The glasses stream gaze, scene video, and audio. A double-blink triggers a scene capture.
2. **Reduce.** Gaze becomes fixations and saliency peaks; a vision model turns the annotated scene into the object in focus, the setting, and the kind of utterance; overheard speech becomes conversation turns.
3. **Store.** Those results are saved per user in Redis as memories and conversation history.
4. **Expand.** `POST /inference/expand` builds one prompt from the letters, the current context, the last 20 turns, and the last 5 memories, and returns up to five ranked sentences.
5. **Speak and log.** The chosen sentence is spoken with the user's emotion as a delivery hint. Every model call is saved as an `InferenceRecord` with the config, context, exact prompts, and result.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the components and data flow.

## Demo and Paper

- **Video:** [CCI Tech Demo](https://youtu.be/hLwuvXr1QGw) shows the system running with the Neon glasses.
- **Paper:** [Multi-System Cognitive Steering for Generative Communication](https://arxiv.org/abs/2610.19177) (Lacosse, Tano, and McNamee, 2026) introduces Cognitive Steering, the idea behind this toolkit: grounding abbreviation expansion in multimodal context such as the scene, emotion, and communication motive. The paper describes a related proof of concept and its experiments, not this code, so its results are not reproduced here.

---

## Installation

SynNeon requires **Python 3.12–3.14** and [`uv`](https://docs.astral.sh/uv/) ≥ 0.13.7 (see [installation.md](installation.md)). The server and dashboard also need Redis, which the included Docker Compose file provides.

```shell
git clone https://github.com/elacosse/synneon.git
cd synneon
make setup            # uv sync + pre-commit hooks (or just: uv sync)
cp .env.example .env  # then fill in your API key(s)
```

Models default to OpenAI (`OPENAI_API_KEY`). Every agent, STT, and TTS setting also has a Gemini profile, which uses `GOOGLE_API_KEY` (`GEMINI_API_KEY` is accepted too).

## Quick Start

Everything in Docker (Redis, the dashboard, and a mock Neon client), then open http://127.0.0.1:8765:

```shell
docker compose -f docker/docker-compose.yml --profile mock up -d --build
```

Or run the app on the host, with only Redis in Docker:

```shell
# 1. Redis (and Redis Insight)
docker compose -f docker/docker-compose.yml up -d redis redisinsight

# 2a. REST API only (http://127.0.0.1:8000, docs at /docs)
uv run python -m synneon.server

# 2b. Or the full dashboard (REST API + React UI + WebSocket, http://127.0.0.1:8765)
(cd src/synneon/ux && npm ci && npm run build)
uv run synneon app

# 3. In another terminal, feed it Neon data (drop --mock when a Neon device is connected)
uv run synneon client --mock --pupil
```

Don't run the host app and the `synneon` container at the same time: both use port 8765.

Run `uv run synneon --help` to see every command.

> **Security:** the API has no authentication and Redis has no password. Servers bind `127.0.0.1` and Compose publishes ports on loopback only. Expose them only on a trusted network.

---

## Walkthrough (no Neon device needed)

This walkthrough runs the whole pipeline on a laptop with mock hardware and the pre-recorded capture in `examples/capture_sample_v1`. You need Docker, `jq`, and an `OPENAI_API_KEY` in `.env`; no local Python or Node is required. Run every command from the repository root.

### 1. Start the stack

```bash
docker compose -f docker/docker-compose.yml --profile mock up -d --build
```

The first build takes a few minutes (it compiles the frontend and installs the Python dependencies); later builds are cached. Check that all four services are up and that `synneon` reports `(healthy)`, which it does once it reaches Redis:

```bash
docker compose -f docker/docker-compose.yml ps
curl -s http://127.0.0.1:8765/health
```

Expect `{"status":"ok","profiles_loaded":[],"redis":"connected"}`. To follow model calls and errors, keep `docker compose -f docker/docker-compose.yml logs -f synneon` open in a second terminal.

To start from an empty store (this deletes every stored conversation, memory, and inference record):

```bash
docker compose -f docker/docker-compose.yml exec redis redis-cli FLUSHDB
```

### 2. Run the offline steps

Seed the conversation and render the saliency and fixation images. This makes no model calls.

```bash
docker compose -f docker/docker-compose.yml exec synneon synneon example run --steps seed,visualize
```

View the result at http://127.0.0.1:8765/viz/image?example=examples/capture_sample_v1&file=saliency_peaks.png (or `gaze_fixations.png`).

### 3. Run the full pipeline

This takes about a minute. It transcribes the audio, sends the gaze-annotated scene to the vision model, stores the result as a memory, expands `G S Y T` using that scene and the seeded conversation, and synthesizes speech.

```bash
docker compose -f docker/docker-compose.yml exec synneon synneon example run --user default --save --neon-scene peaks
```

Every model call was recorded with its model, context snapshot, and result:

```bash
curl -s 'http://127.0.0.1:8765/context/inferences?user_id=default&limit=5' | jq '.[] | {endpoint, profile, model: .agent_config.model, context_snapshot, result_snapshot}'
curl -s 'http://127.0.0.1:8765/context/memories?user_id=default' | jq '.[] | {type, summary}'
```

### 4. Expand an abbreviation in the dashboard

1. Open http://127.0.0.1:8765. Each page load starts a new session. The Connection panel should show `User ID: default`, `Backend Status: Connected`, and `Mock Device (localhost)`.
2. Give the new session some conversation. Run this *after* the page has loaded, because it writes to the current session:

    ```bash
    curl -s -X POST 'http://127.0.0.1:8765/context/conversation?user_id=default' -H 'Content-Type: application/json' -d '[{"user_id":"default","session_id":"-","role":"other","text":"Do you want to schedule a quick call later to discuss the next steps?"},{"user_id":"default","session_id":"-","role":"user","text":"Sure, does 2 PM work for you?"},{"user_id":"default","session_id":"-","role":"other","text":"Yes."}]' | jq length
    ```

3. Optionally, click the **Emotion** wheel to set a label and intensity.
4. In **Letter Grid**, click `g`, `s`, `y`, `t`, then **Send**. After about 10 seconds the candidates appear; the top one should be close to "Great, see you then."
5. Click **Inspect** to see the exact system and user prompts. The user prompt holds the stored scene memories and the conversation inside `<memories>` and `<conversation>` tags.
6. Click a candidate to speak it. The chosen sentence is saved as a `user` utterance for the next turn.

The same expansion from the API side:

```bash
curl -s 'http://127.0.0.1:8765/context/inferences?user_id=default&inference_type=text2text&limit=1' | jq '.[0] | {profile, model: .agent_config.model, prompts: .context, candidates: [.result_snapshot.candidates[].text]}'
```

Redis Insight at http://localhost:5540 (add a database with host `redis`, port `6379`) shows the raw `user:default:*` keys.

### 5. Shut down

```bash
docker compose -f docker/docker-compose.yml down       # keeps data
docker compose -f docker/docker-compose.yml down -v    # also deletes the Redis volume
```

### Troubleshooting

| Symptom | Cause and fix |
| :--- | :--- |
| `env file .../.env not found` | Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. |
| `synneon` stays `(health: starting)` or turns `(unhealthy)` | Read `docker compose -f docker/docker-compose.yml logs synneon`. The container is only healthy when it can reach Redis. |
| `port is already allocated` for 8765, 6379, or 5540 | Another process is using that port (for example a `uv run synneon app` left running). Stop it and start the stack again. |
| The Connection panel shows `No device` | No Neon client is connected. Start the stack with `--profile mock`, and check `docker compose -f docker/docker-compose.yml logs neon-mock`. |
| A code, config, or `.env` change has no effect | Run `up -d --build` again. Code and configs are copied into the image at build time; `.env` is read when the container is created. |
| Expansion or vision step returns HTTP 500 | Usually a missing or invalid API key. The reason is printed in the `synneon` logs. |
| The top candidate is unrelated to the conversation | The conversation is missing from the current session. Reload the dashboard, post the conversation again, and resend. Model output also varies between runs. |
| The dashboard cannot be reached from another device | Expected: ports are published on `127.0.0.1` only. See [Docker](#docker). |
| Neon Capture panel says `No frames buffered` | Expected with mock hardware: there is no camera. Scene context comes from the example pipeline instead. |
| No sound when picking a candidate | Check that the browser tab is not muted. `curl -s -X POST http://127.0.0.1:8765/audio/speak -H 'Content-Type: application/json' -d '{"text":"hello"}' -o hello.wav` tests speech on its own. |
| Old, unrelated memories appear in the prompt | Redis still holds earlier runs. Flush it and rerun steps 3 and 4. |

To run the example pipeline on the host instead of in the container, use `uv run synneon example run --example examples/capture_sample_v1 ...` with Redis running.

---

## Compass Keyboard

The Compass Keyboard (`synneon keyboard`) maps the alphabet to four directional zones. You drill down into zones until a letter is selected, or type abbreviations that an AI agent expands.

```shell
# Arrow keys
uv run synneon keyboard type-text

# Neon gaze control, speech output, and live video
uv run synneon keyboard type-text --neon --config configs/neons/default.toml --speak --display
```

| Flag | Description |
| :--- | :--- |
| `--neon` | Drive the interface with real-time eye gestures from a connected Neon device (use with `--config`). |
| `--speak` | Speak the selected expansion via the server's `/audio/speak` endpoint. |
| `--display` | With `--neon`, show the scene camera with a gaze overlay. |
| `--endpoint <URL>` | Expansion endpoint (default `http://127.0.0.1:8000/inference/expand`). |
| `--debug` | Verbose logging; disables fullscreen. |

> Abbreviation expansion (`Enter`) and speech (`--speak`) need the [server](#server--api) running.

| Input | Typing mode | Picking mode (expansion) |
| :--- | :--- | :--- |
| **`↑` `↓` `←` `→`** | Navigate / drill into zone | Move selection cursor |
| **`Enter`** | Expand abbreviation | Confirm selection |
| **`Backspace`** | Go back / delete character | Cancel / return to typing |
| **`q`** | Quit | - |

---

## Gaze Gesture Detection

The `gesture` command group runs gesture detection without the keyboard UI, for testing calibration or integrating gestures into other tools.

### Compass Gestures

Detects `UP`, `DOWN`, `LEFT`, `RIGHT` swipes and dwells.

```shell
# Real Neon device
uv run synneon gesture compass --config configs/neons/default.toml

# Synthetic gaze data
uv run synneon gesture compass --dry-run

# Drive the Compass state with gestures (Enter resets it)
uv run synneon gesture compass --config configs/neons/default.toml --typing
```

Tune thresholds in `configs/neons/default.toml` under `[gestures.compass]`.

### Mouse Gestures

Uses the mouse pointer as a gaze stream, to test the detector without wearing the glasses.

```shell
uv run synneon gesture mouse --config configs/neons/default.toml
```

`--poll-hz` (default 240) and `--emit-hz` (default 60) control pointer sampling and detector input rates.

---

## Utilities

### Calibration (`calibrate`)

Records the bounds of your gaze so gestures map accurately. Look at the extreme edges of your field of view (up, down, left, right) while it runs, then press `q` to save the bounds to the config file.

```shell
uv run synneon calibrate --config configs/neons/default.toml
```

### Live Gaze Display (`display`)

Shows the Neon scene camera with a real-time gaze overlay.

```shell
uv run synneon display gaze --config configs/neons/default.toml
```

### Snapshot Capture (`capture`)

Captures a scene-camera snapshot, annotated with your gaze position, on each double-blink.

```shell
uv run synneon capture --config configs/neons/default.toml --output captures/
```

### Capture + Gestures (`capture-and-gesture`)

Runs double-blink snapshots and directional gestures together. `--typing` emits arrow-key events for each gesture (this is what `keyboard type-text --neon` uses); `--save` keeps captured images on disk.

```shell
uv run synneon capture-and-gesture --config configs/neons/default.toml --typing
```

### Real-time Pupillometry (`pupil`)

Plots left and right pupil diameter in real time. `--window-s` (default 30) sets the plot window and `--refresh-hz` (default 20) the redraw rate.

```shell
uv run synneon pupil --config configs/neons/default.toml
```

---

## Video Acquisition

`synneon video capture-all` records synchronized video, audio, gaze, and pupillometry from a Neon device.

- **Triggering:** start and stop recording with the Spacebar or a double-blink. `max_duration_s` under `[recording]` in the Neon config stops it automatically.
- **Outputs**, in `captures/acquisition_YYYYMMDD_HHMMSS/`:
  - `video.mp4`: full-resolution scene-camera video.
  - `audio.wav`: audio from the device.
  - `data.csv`: one row per scene-camera frame with timestamp, gaze X/Y, blink and worn status, left/right pupil diameter (mm), and audio RMS.
- **Gaze visualizations:**
  - `saliency_map.png`: grayscale attention heatmap.
  - `saliency_overlay.png`: color heatmap blended onto the median frame.
  - `saliency_peaks.png`: markers ("X" or circles) on the strongest attention peaks; weak peaks are dropped by a relative threshold.
  - `gaze_trajectory.png`: the gaze path over time, colored blue to red.
  - `gaze_fixations.png`: I-DT fixations as circles sized by duration, joined by saccade lines.

```bash
# Analyze mode (default): send the capture to the API for vision-model context extraction
uv run synneon video capture-all

# Save mode: no network calls; keep all files on disk
uv run synneon video capture-all --no-analyze
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--config <PATH>` | — | Neon config TOML |
| `--analyze / --no-analyze` | `--analyze` | Send the capture to the API, or only save it |
| `--api-url <URL>` | `http://localhost:8000` | Analysis server |
| `--profile <NAME>` | `image2context` | Agent profile for the capture |
| `--max-peaks <INT>` | `5` | Attention peaks drawn on `saliency_peaks.png` |
| `--output <DIR>` | `captures/` | Output root directory |
| `--debug` | off | Verbose stream logs |

---

## Server & API

Abbreviation expansion, image-to-context, STT, and TTS are served by the local server, which needs Redis (see [Docker](#docker)).

```shell
uv run python -m synneon.server
```

Options: `--host` (default `127.0.0.1`), `--port` (default `8000`), and `--agent-config`, `--audio-config`, `--stt-config`, `--memory-config` to point at alternative TOML files. Interactive API docs are at `/docs`. The dashboard (`synneon app`) mounts the same routes on port `8765`.

**Inference**
* `POST /inference/expand`: context-aware abbreviation expansion. Body: `tokens`, `profile`, `context` (emotion, visual, pragmatic). The prompt also includes recent conversation history and cross-session memories from Redis, placed in the user message inside `<conversation>` and `<memories>` tags. The response carries up to five ranked candidates and the exact prompts that were sent.
* `POST /inference/image2context`: vision-model analysis of a scene image (multipart upload).
* `POST /inference/image2context/base64`: the same, from base64 image data.
* `POST /inference/neon-context`: multimodal analysis of a complete Neon capture (JSON with base64 scene frames, audio, and gaze telemetry).

**Audio**
* `POST /audio/speak`: text-to-speech; returns WAV. Send a JSON body (`text`, optional `profile`, `voice`, `model`, `emotion_label`, `emotion_intensity`) so the text stays out of URLs and access logs; query parameters still work. An emotion becomes a plain-language style hint.
* `POST /audio/transcribe`: speech-to-text (multipart upload).
* `POST /audio/transcribe/base64`: speech-to-text from base64 audio.

**Context & Memory** *(requires Redis)*
* `POST /context/users`: create a user profile and start a fresh session. Idempotent.
* `GET /context/conversation`: recent utterances for a session, oldest first.
* `POST /context/conversation`: save a list of `ConversationEntry` records (for seeding).
* `POST /context/utterance`: save one utterance to the current session.
* `GET /context/memories`: recent cross-session memory entries.
* `GET /context/inferences`: recent inference records (full context and config snapshots), oldest first. Query: `limit` (1–100, default 20), optional `inference_type` (`text2text` or `image2text`).
* `GET /context/inferences/{record_id}?inference_type=<type>`: one inference record (404 if not found).
* `GET /context/expand-context?user_id=<id>&session_id=<id>`: the context used by the last expansion.

**Visualization**
* `GET /viz/examples`: example folders that have generated saliency and gaze images.
* `GET /viz/image?example=<path>&file=<name>`: one of those images (or `video.mp4`). `example` must resolve inside the working directory or the project root.

**Health**
* `GET /health`: service and Redis connectivity check.

**Errors:** invalid base64 input returns `400`, a failed call to an external agent endpoint (the `alignenv` profile) returns `502`, and other inference failures return `500` with the reason in `detail`.

> **Opt-in persistence:** inference and transcription endpoints accept a `save_to_memory` flag (default `false`). When set, results are stored in Redis and included in later prompts.

> **Users:** every store-backed endpoint accepts `?user_id=<id>`, which scopes all Redis keys to `user:<id>:...` for that request. A user with no session gets one on first use.

### Configuration

Runtime settings live in `configs/`:

* `configs/interpreter/agent.toml`: agent profiles (model, prompts, `thinking` effort), with OpenAI and Gemini variants.
* `configs/stt/default.toml` and `configs/tts/default.toml`: speech-to-text and text-to-speech profiles.
* `configs/latents/default.toml`: the intensity bands that turn an emotion into an adjective for the expansion prompt, and the `tts_style` template that turns it into a speaking-style hint.
* `configs/memory/default.toml`: Redis URL and memory windows (below).
* `configs/neons/default.toml`: device address, blink timing, calibration bounds, gesture thresholds, I-DT fixation parameters, saliency styling, and the recording timeout.
* `.env` (copy `.env.example`): `OPENAI_API_KEY`, `GOOGLE_API_KEY`.

| Memory key | Default | Description |
| :--- | :--- | :--- |
| `redis_url` | `redis://localhost:6379/0` | Redis connection string. `SYNNEON_REDIS_URL` overrides it. |
| `conversation_window` | `20` | Max utterances included in the prompt. |
| `memory_window` | `5` | Max cross-session memories included in the prompt. |
| `inference_window` | `50` | Default number of inference records returned when listing them in code (not in `memory/default.toml`). |
| `conversation_ttl_days` | `7` | Days before conversation data expires. |
| `user_id` | `default` | Default namespace for stored data. |

---

## Example Pipeline Runner

`synneon example run` runs the inference pipeline end to end on a recorded capture. It needs no Neon device or running server; model steps need API keys and `--save` needs Redis.

### Example folder layout

```
examples/capture_sample_v1/
├── audio.wav                      # Ambient audio recording
├── data.csv                       # Gaze, pupil, and audio RMS telemetry
├── video.mp4                      # Scene camera footage
├── conversation.txt               # Seed conversation (one 'role: text' entry per line)
└── abbreviation_for_inference.txt # Abbreviation tokens to expand (e.g. "G S Y T")
```

`conversation.txt` has one entry per line, with the role `other` (ambient speaker) or `user` (glasses wearer):

```
other: Hello, can you help me today?
user: Sure, what do you need?
other: Great, let me know.
```

`abbreviation_for_inference.txt` holds one line with the tokens to expand, such as `G S Y T`.

### Usage

```bash
# Print results only; no Redis needed
uv run synneon example run --example examples/capture_sample_v1

# Persist everything to Redis
uv run synneon example run --example examples/capture_sample_v1 --save

# Only the seed and expand steps, for a custom user, with full dumps
uv run synneon example run --example examples/capture_sample_v1 --steps seed,expand --user alice --verbose
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--example` | `examples/capture_sample_v1` | Path to the example folder |
| `--user` | `default` | User ID for the Redis session |
| `--interpreter` | `configs/interpreter/agent.toml` | Agent TOML for expand/neon steps |
| `--latents` | — | Latents TOML (reserved) |
| `--memory` | `configs/memory/default.toml` | Redis connection and window settings |
| `--neons` | `configs/neons/default.toml` | Blink, fixation, saliency, and recording config |
| `--stt` | `configs/stt/default.toml` | Speech-to-text config |
| `--tts` | `configs/tts/default.toml` | Text-to-speech config |
| `--save / --no-save` | `--no-save` | Persist results to Redis |
| `--steps` | `all` | Comma-separated subset: `seed,stt,visualize,neon,expand,tts` |
| `--profile` | `default` | Agent profile for the `expand` step |
| `--neon-profile` | `image2context` | Agent profile for the `neon` step |
| `--frame-mode` | `middle` | Reference frame for saliency: `first`, `middle`, or `last` |
| `--neon-scene` | `raw` | Scene image sent to the vision model: `raw` (video frame) or `peaks` (`saliency_peaks.png`) |
| `--verbose / --no-verbose` | `--no-verbose` | Show full config and context dumps |

| Step | What it does |
| :--- | :--- |
| `seed` | Parses `conversation.txt` into `ConversationEntry` records; saves them with `--save`. |
| `stt` | Transcribes `audio.wav` with the configured STT provider; saves utterances with `--save`. |
| `visualize` | Computes saliency maps and gaze overlays and writes the five PNGs to the example folder. |
| `neon` | Builds a `NeonCapturePayload` from video frames and the gaze/pupil CSV and runs the vision agent; saves an `InferenceRecord` and `MemoryEntry` with `--save`. The result is passed to `expand` as context. |
| `expand` | Expands `abbreviation_for_inference.txt` using the conversation, the `neon` result (when it ran), and stored memories (with `--save`); saves an `InferenceRecord` with `--save`. |
| `tts` | Speaks the top candidate and writes `output_tts.wav` to the example folder. |

Errors in `stt`, `visualize`, `neon`, and `tts` are reported as warnings and do not stop the pipeline.

---

## Dashboard (`synneon app`)

The dashboard and the Neon device run as two separate processes:

- `synneon app` serves the REST API, a `/ws` WebSocket for browsers, and the React UI from `src/synneon/ux/` (audio recording, transcript, pupillometry, gaze gestures, captures, and a manual keyboard). It never touches hardware and needs Redis.
- `synneon client` connects to a Neon device (or sends `--mock` data) and streams its events to the app's `/ws/neon` socket, which relays them to every open dashboard. A double-blink records a capture; the client renders the gaze visualizations and posts the capture to `POST /inference/neon-context` (`image2context` agent, saved to memory). It reconnects if the app restarts; the dashboard shows `No device` while no client is connected.

```shell
# Build the UI once (output goes to src/synneon/ux/dist, which the app serves)
(cd src/synneon/ux && npm ci && npm run build)

uv run synneon app

# In another terminal: a Neon device, or --mock for synthetic data
uv run synneon client --config configs/neons/default.toml --pupil --gesture
uv run synneon client --mock --pupil
```

Then open http://127.0.0.1:8765.

- `synneon app` options: `--host` (default `127.0.0.1`), `--port` (default 8765), `--debug`.
- `synneon client` options: `--api-url` (default `http://127.0.0.1:8765`), `--config`, `--pupil`, `--gesture`, `--mock`, `--profile` (shown on the dashboard, default `default`), `--neon-profile` (capture agent, default `image2context`), `--user-id` (default `default`), `--debug`.
- `curl -X POST http://127.0.0.1:8765/debug/fake-blink` acts as a double-blink on the connected client (HTTP 409 when none is connected).

For UI development with hot reload, keep `synneon app` running and start Vite, which proxies `/ws`, `/context`, `/audio`, and `/viz` to port 8765:

```shell
cd src/synneon/ux
npm run dev   # http://localhost:5173
```

---

## Docker

The Compose file runs Redis (with append-only persistence), Redis Insight, and the `synneon` service: the backend and the built React frontend in one container. The image is built from `docker/Dockerfile` (a Node stage builds the UI; a `uv` stage installs the locked Python dependencies).

```bash
docker compose -f docker/docker-compose.yml up -d --build                  # stack without a Neon client
docker compose -f docker/docker-compose.yml --profile mock up -d --build   # plus a mock Neon client
docker compose -f docker/docker-compose.yml up -d redis redisinsight       # data services only
```

| Service | URL | Description |
| :--- | :--- | :--- |
| **synneon** | http://127.0.0.1:8765 | Dashboard, REST API, `/ws`, and `/ws/neon`; no hardware |
| **neon-mock** | — | Mock Neon client; only with `--profile mock` |
| **redis** | `localhost:6379` | Redis with append-only persistence |
| **redisinsight** | http://localhost:5540 | Redis GUI; connect to host `redis`, port `6379` |

- API keys are read from `.env` in the repository root at run time; they are not copied into the image.
- `SYNNEON_REDIS_URL` points the app at the `redis` service inside the container.
- Code and `configs/` are copied in at build time, so rebuild (`up -d --build`) after changing them.
- Ports are published on `127.0.0.1` only. The container listens on `0.0.0.0` internally; the loopback port mapping is what limits access. Widen the mappings only on a trusted network.
- For a real Neon device, run the client on the host, where the device is reachable: `uv run synneon client --config configs/neons/default.toml --pupil`.
- Run CLI commands in the container with `docker compose -f docker/docker-compose.yml exec synneon synneon ...`.

---

## Development

See [development.md](development.md) for contributing, tests, and project structure, and [AGENTS.md](AGENTS.md) for conventions.

```shell
make check       # lint + tests (read-only, same as CI)
make format      # apply formatting and auto-fixes
make typecheck   # ty (not yet part of make check)
```

CI (`.github/workflows/ci.yml`) is manual-only: start it from the GitHub Actions tab or with `gh workflow run ci.yml`.

## Citation

If you use SynNeon or build on Cognitive Steering, please cite the paper:

```bibtex
@misc{lacosse2026cognitivesteering,
  title         = {Multi-System Cognitive Steering for Generative Communication},
  author        = {Lacosse, Eric and Tano, Pablo and McNamee, Daniel C.},
  year          = {2026},
  eprint        = {2610.19177},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.19177}
}
```

## License

[MIT](LICENSE)
