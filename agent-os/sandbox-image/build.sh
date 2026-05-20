#!/usr/bin/env bash
# Build the AgentOS sandbox image used by the Docker sandbox executor.
set -euo pipefail

IMAGE_NAME="${SANDBOX_IMAGE:-agent-sandbox:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building sandbox image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo "Done. Image: $IMAGE_NAME"
