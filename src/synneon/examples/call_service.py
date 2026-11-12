#!/usr/bin/env python3
"""Example client for the AAC expansion Modal endpoint.

Usage:
    python scripts/call_service.py [--base-url URL]

The base URL is printed by ``modal deploy``.  It looks like:
    https://<workspace>--alignenv-aac-interpreter-api.modal.run

Interactive API docs are available at:
    <base-url>/docs
"""

from __future__ import annotations

import argparse
import json

import requests

# Replace with your actual Modal URL after deploying.
DEFAULT_BASE_URL = "https://pablotanoretamales--alignenv-aac-interpreter-api.modal.run"


def main():
    parser = argparse.ArgumentParser(description="Call the AAC expansion endpoint")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the deployed API")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    payload = {
        "abbreviation": "[can] [you] o [the] l [for] [me] , p ?",
        "history": [
            {"speaker": "B", "text": "Hey, how was your weekend?"},
            {"speaker": "A", "text": "It was great, I went to the beach."},
            {"speaker": "B", "text": "Nice! Let's get started"},
        ],
        "referents": ["Whiteboard", "Laptop", "Coffee mug", "Table", "Man"],
    }

    # --- Primary expansion (dual-pass beam search) ---
    print("=" * 60)
    print("POST /expand  (dual-pass beam search)")
    print("=" * 60)
    print(f"Request: {json.dumps(payload, indent=2)}\n")

    resp = requests.post(f"{base}/expand", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print("Candidates:")
    for i, c in enumerate(data["candidates"], 1):
        print(f"  {i}. {c}")

    # --- More candidates (sample & rerank, temp=1.2) ---
    print()
    print("=" * 60)
    print("POST /expand/more  (sample & rerank, temp=1.2)")
    print("=" * 60)
    print(f"Request: {json.dumps(payload, indent=2)}\n")

    resp = requests.post(f"{base}/expand/more", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print("More candidates:")
    for i, c in enumerate(data["candidates"], 1):
        print(f"  {i}. {c}")

    print(f"\nInteractive docs: {base}/docs")


if __name__ == "__main__":
    main()
