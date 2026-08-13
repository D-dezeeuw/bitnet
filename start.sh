#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-bitnet-2b-api}"
CONTAINER="${CONTAINER:-bitnet-2b}"
NETWORK="${NETWORK:-nginx-proxy-manager_default}"
STATIC_IP="${STATIC_IP:-172.22.0.25}"
MODEL_FILE="ggml-model-i2_s.gguf"
# Pinned for the same reason the upstream source is: the HF repo is a moving
# target (last updated 2025-12-17, well after its April 2025 release), so an
# unpinned download can silently change the model under a rebuild.
MODEL_REPO="microsoft/bitnet-b1.58-2B-4T-gguf"
MODEL_REVISION="a1f2f1c765812aa8af3f6eda4a313707064bba15"
MODEL_SHA256="4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"

# Publish a host port for local testing: HOST_PORT=8010 ./start.sh
# Left empty by default -- the container is reached through the reverse proxy.
HOST_PORT="${HOST_PORT:-}"

if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: $MODEL_FILE not found in current directory." >&2
  echo "Download it first:" >&2
  echo "  huggingface-cli download $MODEL_REPO $MODEL_FILE \\" >&2
  echo "    --revision $MODEL_REVISION --local-dir ." >&2
  exit 1
fi

# Warn rather than fail: a deliberate swap (a fine-tune, a different quant) is
# legitimate, but an unnoticed one is not.
if command -v sha256sum > /dev/null 2>&1; then
  echo "Verifying model checksum..."
  ACTUAL_SHA="$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)"
  if [ "$ACTUAL_SHA" != "$MODEL_SHA256" ]; then
    echo "WARNING: $MODEL_FILE does not match the pinned revision." >&2
    echo "  expected $MODEL_SHA256" >&2
    echo "  actual   $ACTUAL_SHA" >&2
    echo "  pinned revision: $MODEL_REVISION" >&2
    echo "Continuing. If this was not intentional, re-download at that revision." >&2
  else
    echo "Model matches pinned revision $MODEL_REVISION"
  fi
fi

echo "Building BitNet 2B Docker image..."
docker build -t "$IMAGE" .

docker rm -f "$CONTAINER" 2>/dev/null || true

RUN_ARGS=(
  --name "$CONTAINER"
  --network "$NETWORK"
  --ip "$STATIC_IP"
  --restart unless-stopped
)

# Authentication is off unless a key is supplied. Without one every /v1 route
# is open to anything that can reach the container.
if [ -n "${BITNET_API_KEY:-}" ]; then
  RUN_ARGS+=(-e "BITNET_API_KEY=${BITNET_API_KEY}")
  echo "API key authentication: ENABLED"
else
  echo "API key authentication: DISABLED (set BITNET_API_KEY to require one)"
fi

# /download is served from a mount, so download.zip is optional and never a
# build dependency. The route returns 404 when nothing is mounted.
if [ -f "download.zip" ]; then
  RUN_ARGS+=(-v "$(pwd)/download.zip:/app/downloads/download.zip:ro")
  echo "Download file: mounted"
fi

if [ -n "$HOST_PORT" ]; then
  RUN_ARGS+=(-p "${HOST_PORT}:8010")
fi

echo "Starting BitNet 2B container..."
docker run -d "${RUN_ARGS[@]}" "$IMAGE"

cat <<EOF

BitNet 2B API running on network $NETWORK
Internal: http://$CONTAINER:8010

EOF

if [ -n "$HOST_PORT" ]; then
  cat <<EOF
Published on the host at port $HOST_PORT:

  curl http://localhost:${HOST_PORT}/health
  curl http://localhost:${HOST_PORT}/v1/models
  curl -s http://localhost:${HOST_PORT}/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages":[{"role":"user","content":"Hello"}]}'
EOF
else
  cat <<EOF
No host port published, so localhost will not reach it. Test from inside the
container, or re-run with HOST_PORT=8010 ./start.sh

  docker exec $CONTAINER curl -s http://127.0.0.1:8010/health
  docker exec $CONTAINER curl -s http://127.0.0.1:8010/v1/models
EOF
fi
