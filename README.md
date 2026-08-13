# BitNet Docker API

Dockerized Microsoft BitNet b1.58-2B-4T with an OpenAI-compatible API and web inference UI.

## Quick Start

```bash
# 1. Place ggml-model-i2_s.gguf next to the Dockerfile

# 2. Build and run
./start.sh

# 3. Test
curl http://localhost:8010/health
```

## API Endpoints

All endpoints are OpenAI-compatible.

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST /v1/summarize` | Summarize conversation context |
| `GET /inference` | Web inference UI |
| `GET /download` | Download static file |

### Example: Chat Completion

```bash
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

### Example: Streaming

```bash
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Web Inference UI

Open `http://localhost:8010/inference` in a browser.

Features:

- Streaming token output
- Markdown rendering
- Configurable temperature, top_p, max tokens, and stop sequences
- Context usage indicator (green/yellow/orange/red) — click to clear
- Continue button when output is truncated
- Stop button to halt generation
- Conversation history with hybrid summarization for context compaction

## Architecture

```text
┌─────────────────────────────────┐
│         Docker Container        │
│                                 │
│  llama-server (127.0.0.1:8080)  │
│         ↑                       │
│         │ proxy                  │
│         ↓                       │
│  FastAPI + Uvicorn (:8010)      │
│                                 │
└─────────────────────────────────┘
         ↑
    Nginx Proxy Manager
         ↑
    https://bitnet.blockworlds.nl
```

The Docker image is a multi-stage build:

1. **Builder** (`python:3.11-slim`) — clones BitNet repo, compiles TL2 lookup kernels via `setup_env.py`, builds `llama-server`
2. **Runtime** (`python:3.13-slim`) — runs `llama-server` + FastAPI proxy

## Configuration

Environment variables (set via Dockerfile ARGs at build time):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/model.gguf` | Path to GGUF model |
| `MODEL_ID` | `bitnet-b1.58-2B-4T` | Model ID returned by API |
| `LLAMA_SERVER_PORT` | `8080` | Internal llama-server port |
| `BITNET_THREADS` | `4` | Inference threads |
| `BITNET_CTX_SIZE` | `4096` | Context window size |

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build |
| `start.sh` | Build + run script |
| `app.py` | FastAPI proxy + web UI |
| `entrypoint.sh` | Container startup (llama-server then FastAPI) |
| `openclaw.json` | OpenClaw agent framework config |

## Network

Container joins `nginx-proxy-manager_default` with static IP `172.22.0.25`. Reverse proxy to port 8010 over HTTP (not HTTPS).
