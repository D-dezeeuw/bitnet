#!/bin/bash
set -e

IMAGE="bitnet-2b-api"
CONTAINER="bitnet-2b"
NETWORK="nginx-proxy-manager_default"
STATIC_IP="172.22.0.25"

# Check model file exists
if [ ! -f "ggml-model-i2_s.gguf" ]; then
  echo "ERROR: ggml-model-i2_s.gguf not found in current directory."
  echo "Download it first:"
  echo "  huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf --local-dir ."
  exit 1
fi

echo "Building BitNet 2B Docker image..."
docker build -t "$IMAGE" .

# Remove existing container if present
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "Starting BitNet 2B container..."
docker run -d \
  --name "$CONTAINER" \
  --network "$NETWORK" \
  --ip "$STATIC_IP" \
  --restart unless-stopped \
  "$IMAGE"

echo ""
echo "BitNet 2B API running on network $NETWORK"
echo "Internal: http://$CONTAINER:8010  (no host port — access via reverse proxy only)"
echo ""
echo "Health:  curl http://localhost:8010/health"
echo "Models:  curl http://localhost:8010/v1/models"
echo "Chat:    curl -s http://localhost:8010/v1/chat/completions \\"
echo '           -H "Content-Type: application/json" \'
echo '           -d '\''{"messages":[{"role":"user","content":"Hello"}]}'\'''
