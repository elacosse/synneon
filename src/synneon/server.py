import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import typer
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from redis.asyncio import Redis

from synneon.api import config, routes, sockets
from synneon.api.broadcast import set_broadcast
from synneon.api.config import get_memory_settings
from synneon.api.dependencies import set_context_store
from synneon.services.memory.store import ContextStore

load_dotenv()

DEFAULT_AGENT_CONFIG = Path("configs/interpreter/agent.toml")
DEFAULT_AUDIO_CONFIG = Path("configs/tts/default.toml")
DEFAULT_STT_CONFIG = Path("configs/stt/default.toml")
DEFAULT_MEMORY_CONFIG = Path("configs/memory/default.toml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    memory_cfg = get_memory_settings()
    redis = Redis.from_url(memory_cfg.redis_url, decode_responses=True)
    store = ContextStore(redis=redis, config=memory_cfg)

    set_context_store(store)
    set_broadcast(sockets.manager.broadcast)
    await store.bootstrap()

    session_id = str(uuid.uuid4())
    await store.start_session(session_id)

    yield

    await store.end_session()
    await redis.aclose()


app = FastAPI(
    title="Synneon Cognitive Steering Service",
    description="REST API for context-aware inference and cognitive steering.",
    version="0.1.0",
    lifespan=lifespan,
)

# Include the cognitive steering routes and the WebSocket hub (/ws, /ws/neon)
app.include_router(routes.router)
app.include_router(sockets.router)

cli = typer.Typer(help="Synneon API Server CLI")


@cli.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", "--host", "-h", help="Host to bind to. Use 0.0.0.0 only on a trusted network: there is no auth."
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to"),
    agent_config: Path = typer.Option(
        DEFAULT_AGENT_CONFIG,
        "--agent-config",
        "-a",
        help="Path to the agent configuration TOML",
    ),
    audio_config: Path = typer.Option(
        DEFAULT_AUDIO_CONFIG,
        "--audio-config",
        "-s",
        help="Path to the audio/TTS configuration TOML",
    ),
    stt_config: Path = typer.Option(
        DEFAULT_STT_CONFIG,
        "--stt-config",
        "-t",
        help="Path to the STT configuration TOML",
    ),
    memory_config: Path = typer.Option(
        DEFAULT_MEMORY_CONFIG,
        "--memory-config",
        "-m",
        help="Path to the memory/Redis configuration TOML",
    ),
):
    """
    Start the Synneon Cognitive Steering REST API server.
    """
    # Override the default paths in the config module
    config.AGENT_CONFIG_PATH = agent_config
    config.AUDIO_CONFIG_PATH = audio_config
    config.STT_CONFIG_PATH = stt_config
    config.MEMORY_CONFIG_PATH = memory_config

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
