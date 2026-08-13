#!/bin/bash
set -e

MODEL="${MODEL_PATH:-/app/models/ggml-model-i2_s.gguf}"
PORT="${LLAMA_SERVER_PORT:-8080}"
THREADS="${BITNET_THREADS:-4}"
CTX="${BITNET_CTX_SIZE:-2048}"

echo "Starting llama-server on internal port $PORT..."
llama-server \
    -m "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    -t "$THREADS" \
    -c "$CTX" &

LLAMA_PID=$!

echo "Waiting for llama-server to become ready..."
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "llama-server ready (took ${i}s)."
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-server process exited unexpectedly."
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: llama-server failed to start within 120s."
        exit 1
    fi
    sleep 1
done

echo "Starting FastAPI on port 8010..."
exec uvicorn app:app --host 0.0.0.0 --port 8010
