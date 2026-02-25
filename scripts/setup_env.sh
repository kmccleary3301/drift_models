#!/usr/bin/env bash
set -euo pipefail

uv venv .venv
uv sync --extra dev --extra imagenet --extra sdvae
