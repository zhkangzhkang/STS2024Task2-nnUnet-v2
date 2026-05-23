#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entrypoint. The implementation now targets nnU-Net v2.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/scripts/predict_v2.sh" "$@"
