#!/bin/bash
# Starts llama-server, waits for it, then starts the API. Both are supervised:
# if either dies the container exits so the restart policy can recycle it.
# Previously a dead backend went unnoticed and the API returned 503 forever.
set -uo pipefail

MODEL="${MODEL_PATH:-/app/models/model.gguf}"
PORT="${LLAMA_SERVER_PORT:-8080}"
THREADS="${BITNET_THREADS:-4}"
CTX="${BITNET_CTX_SIZE:-4096}"
API_PORT="${BITNET_API_PORT:-8010}"
STARTUP_TIMEOUT="${BITNET_STARTUP_TIMEOUT:-120}"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    exit 1
fi

echo "Starting llama-server on internal port $PORT (threads=$THREADS ctx=$CTX)..."
# --parallel 1 is explicit: -c is the TOTAL KV cache split across slots, so if
# the default slot count were ever above 1 each slot would silently get a
# fraction of CTX and the API would advertise a context it does not have.
#
# --special makes the server render any special tokens the model emits as
# text, where the API's string stops can match them. On the PINNED backend it
# is belt-and-braces rather than load-bearing: this llama.cpp revision
# force-adds "<|eot_id|>" and "<|end_of_text|>" to its end-of-generation set
# by token text (llama-vocab.cpp) and stops at token level, despite the GGUF
# metadata declaring only <|end_of_text|> as eos. An earlier comment here
# claimed --special was the fix for the model looping; live probing disproved
# that -- when output loops to n_predict, the model emitted no end token at
# all, which no stop configuration can fix. The API's loop guard handles that
# case.
llama-server \
    -m "$MODEL" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --parallel 1 \
    --special \
    -t "$THREADS" \
    -c "$CTX" &
LLAMA_PID=$!

cleanup() {
    kill "$LLAMA_PID" "${API_PID:-}" 2>/dev/null
}
trap cleanup TERM INT EXIT

echo "Waiting for llama-server to become ready..."
ready=0
for i in $(seq 1 "$STARTUP_TIMEOUT"); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "llama-server ready (took ${i}s)."
        ready=1
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited during startup." >&2
        exit 1
    fi
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "ERROR: llama-server failed to start within ${STARTUP_TIMEOUT}s." >&2
    exit 1
fi

echo "Starting FastAPI on port $API_PORT..."
# --proxy-headers so logs record the real client address rather than the
# reverse proxy's. forwarded-allow-ips is restricted to the proxy: trusting
# X-Forwarded-For from anywhere would let callers spoof their own address.
uvicorn app:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --proxy-headers \
    --forwarded-allow-ips "${BITNET_TRUSTED_PROXIES:-127.0.0.1}" &
API_PID=$!

# Exit as soon as either process does, so a half-dead container is replaced
# rather than lingering and serving errors.
wait -n "$LLAMA_PID" "$API_PID"
STATUS=$?
if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama-server exited (status $STATUS); shutting down." >&2
else
    echo "API exited (status $STATUS); shutting down." >&2
fi
exit "$STATUS"
