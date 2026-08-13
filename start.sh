#!/bin/bash
set -euo pipefail

IMAGE="${IMAGE:-bitnet-2b-api}"
CONTAINER="${CONTAINER:-bitnet-2b}"
NETWORK="${NETWORK:-nginx-proxy-manager_default}"
STATIC_IP="${STATIC_IP:-172.22.0.25}"
MODEL_FILE="ggml-model-i2_s.gguf"

# Publish a host port for local testing: HOST_PORT=8010 ./start.sh
# Left empty by default -- the container is reached through the reverse proxy.
HOST_PORT="${HOST_PORT:-}"

if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: $MODEL_FILE not found in current directory." >&2
  echo "Download it first:" >&2
  echo "  huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf $MODEL_FILE --local-dir ." >&2
  exit 1
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
