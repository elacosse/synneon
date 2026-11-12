# syntax=docker/dockerfile:1
#
# SynNeon dashboard image: REST API + WebSocket + built React UI in one process.
# Build context is the repository root (see docker/docker-compose.yml).
# API keys are NOT baked in; Compose passes them from .env at run time.

# ---- Stage 1: build the React dashboard ----
FROM node:22-bookworm-slim AS ui

WORKDIR /ui
COPY src/synneon/ux/package.json src/synneon/ux/package-lock.json ./
RUN npm ci
COPY src/synneon/ux/ ./
RUN npm run build


# ---- Stage 2: Python backend that also serves the built UI ----
FROM ghcr.io/astral-sh/uv:0.13.7-python3.13-trixie-slim AS app

ENV DEBIAN_FRONTEND=noninteractive

# libgl1/libglib2.0-0t64: OpenCV. libportaudio2: sounddevice (imported by the CLI).
# gcc/libc6-dev/linux-libc-dev: pynput's Linux dependency `evdev` has no wheel and compiles from source.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0t64 libportaudio2 gcc libc6-dev linux-libc-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# UV_DYNAMIC_VERSIONING_BYPASS: the version normally comes from git, which is not in the image.
# PYNPUT_BACKEND=dummy: pynput is imported by the CLI but there is no display in a container.
ENV UV_CONFIG_FILE=/app/uv.toml \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_DYNAMIC_VERSIONING_BYPASS=0.0.0 \
    PYNPUT_BACKEND=dummy \
    PYDANTIC_AI_NO_BANNER=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock uv.toml README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
COPY examples ./examples
COPY --from=ui /ui/dist ./src/synneon/ux/dist
RUN uv sync --locked --no-dev

EXPOSE 8765

# The dashboard only; Neon data comes from `synneon client` (on the host, or the `mock` Compose profile).
# Binds 0.0.0.0 inside the container; Compose publishes the port on the host's loopback only.
CMD ["synneon", "app", "--host", "0.0.0.0"]
