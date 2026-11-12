#!/bin/bash


# Start the modal app in the background
exec uv run modal serve src/syn_serve/modal/serve_dam.py &

# Start the Uvicorn app in the foreground
echo "Starting Uvicorn app..."
exec uv run uvicorn src.syn_serve.api.syn_server:app --host 0.0.0.0 --port 8000
