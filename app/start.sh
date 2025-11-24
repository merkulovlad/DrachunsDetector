#!/usr/bin/env bash

# Unified start script for local development and container runs.
# Creates an optional virtual environment (skip with SKIP_VENV=1),
# installs dependencies once, and then launches the FastAPI app.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INFO_PREFIX="[start]"

echo "${INFO_PREFIX} Violence Detection System"

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo "${INFO_PREFIX} Creating virtual environment (.venv)"
        python3 -m venv .venv
    fi

    # shellcheck disable=SC1091
    source .venv/bin/activate

    if [[ ! -f ".venv/.deps_installed" ]]; then
        echo "${INFO_PREFIX} Installing Python dependencies"
        pip install --no-cache-dir -r requirements.txt
        touch .venv/.deps_installed
        echo "${INFO_PREFIX} Dependencies installed"
    else
        echo "${INFO_PREFIX} Dependencies already installed"
    fi
else
    echo "${INFO_PREFIX} SKIP_VENV=1 -> using system Python"
fi

if [[ -f ".env" ]]; then
    echo "${INFO_PREFIX} Using .env configuration"
else
    echo "${INFO_PREFIX} .env not found (copy .env.example and set checkpoints)"
fi

HOST_VALUE=${HOST:-0.0.0.0}
PORT_VALUE=${PORT:-8000}

echo "${INFO_PREFIX} Starting FastAPI server on ${HOST_VALUE}:${PORT_VALUE}"
exec python main.py
