# BitNet Docker API

Dockerized Microsoft BitNet b1.58-2B-4T with an OpenAI-compatible API and web inference UI.

## Quick Start

```bash
# 1. Place ggml-model-i2_s.gguf next to the Dockerfile
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-gguf ggml-model-i2_s.gguf \
  --revision a1f2f1c765812aa8af3f6eda4a313707064bba15 --local-dir .

# 2. Build and run, publishing a host port so you can reach it locally
HOST_PORT=8010 ./start.sh

# 3. Test
curl http://localhost:8010/health
```

Without `HOST_PORT` the container publishes no host port and is reachable only
over its Docker network (see [Network](#network)); `start.sh` prints the
commands that apply to whichever mode you used.

## API Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | no | Health check, including backend status |
| `GET /v1/status` | no | Busy flag, context size, model id, whether auth is on |
| `GET /v1/models` | yes | List available models |
| `POST /v1/chat/completions` | yes | Chat completions (streaming + non-streaming) |
| `POST /v1/summarize` | yes | Summarize conversation context |
| `POST /v1/auth` | — | Exchange an API key for a UI session cookie |
| `GET /inference` | no | Web inference UI |
| `GET /download` | no | Optional file download (404 when nothing is mounted) |

"Auth" applies only when `BITNET_API_KEY` is set. When it is not set, those
routes are open to anything that can reach the container.

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

### Example: With authentication

```bash
curl -s http://localhost:8010/v1/chat/completions \
  -H "Authorization: Bearer $BITNET_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Authentication

Set `BITNET_API_KEY` to require a key on every `/v1` route:

```bash
BITNET_API_KEY=$(openssl rand -hex 32) ./start.sh
```

API clients send it as `Authorization: Bearer <key>` or `X-API-Key: <key>`.

The browser UI cannot send a header, so it exchanges the key once via
`POST /v1/auth` for an HMAC-signed, HttpOnly, SameSite=Strict session cookie.
Open `/inference`, enter the key, and the UI works normally from then on.

## Concurrency

The bitnet.cpp backend serves one request at a time, so the API holds a single
inference slot. A caller waits up to `BITNET_QUEUE_TIMEOUT` seconds for it and
then receives `503`. `GET /v1/status` reports whether the slot is busy.

## Web Inference UI

Open `http://localhost:8010/inference` in a browser.

Features:

- Streaming token output
- Markdown rendering, sanitized before it reaches the DOM
- Configurable temperature, top_p, max tokens, and stop sequences
- Context usage indicator (green/yellow/orange/red) — click to compact
- Continue button when output is truncated, resuming the reply rather than restarting it
- Stop button to halt generation
- Conversation history with hybrid summarization for context compaction

All assets are served from the container. The UI makes no third-party requests
and works with outbound network access blocked.

## Model

`bitnet-b1.58-2B-4T` is the current and only official **generative** BitNet
model. Microsoft publishes six BitNet repos, and the newer ones do not replace
it:

| Repo | Released | Use here |
|------|----------|----------|
| `bitnet-b1.58-2B-4T-gguf` | Apr 2025 | **This one** — I2_S, ready for bitnet.cpp |
| `bitnet-b1.58-2B-4T` | Apr 2025 | Master weights, for transformers/fine-tuning |
| `bitnet-b1.58-2B-4T-bf16` | Apr 2025 | BF16 master weights, not for CPU inference |
| `bitnet-embedding-0.6b` | Jul 2026 | Embeddings — produces vectors, not text |
| `bitnet-embedding-270m` | Jul 2026 | Embeddings |
| `VibeVoice-ASR-BitNet` | Jul 2026 | Speech recognition |

The 2026 releases are embedding and ASR models; neither can serve a chat API.
`BitNet a4.8` (4-bit activations) is a paper, not a released model. So there is
no newer or better option for this deployment.

The GGUF is **pinned by revision and checksum** in `start.sh`. The HF repo was
last updated 2025-12-17, months after its April 2025 release, so an unpinned
`huggingface-cli download` is a moving target that can change the model under a
rebuild. `start.sh` verifies the checksum and warns on a mismatch rather than
failing, since swapping in a fine-tune is a legitimate thing to do deliberately.
To move to a different revision, update `MODEL_REVISION` and `MODEL_SHA256`
together.

## Prompt format

Prompts are built in `app.py` to match the chat template in
`tokenizer_config.json` of `microsoft/bitnet-b1.58-2B-4T`:

```text
User: hi<|eot_id|>Assistant: hello<|eot_id|>Assistant:
```

Roles are capitalized, turns are separated by `<|eot_id|>` with no newline, and
the generation prompt is `Assistant: ` with a trailing space.

This is deliberately **not** delegated to llama-server's own
`/v1/chat/completions`. The chat template embedded in `ggml-model-i2_s.gguf` is
mangled — it emits `Human:` and `BITNETAssistant:`, places `eos_token` after the
generation prompt, and drops `system` messages — so the backend would apply a
worse format than the one built here.

Note also that the GGUF declares `eos_token` as `<|end_of_text|>` while the
trained turn terminator is `<|eot_id|>`. llama-server's automatic EOS stop
therefore never fires at end of turn, which is why `<|eot_id|>` is passed as an
explicit stop sequence.

## Architecture

```text
┌─────────────────────────────────┐
│         Docker Container        │
│                                 │
│  llama-server (127.0.0.1:8080)  │
│         ↑                       │
│         │ proxy                 │
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

1. **Builder** (`python:3.11-slim-bookworm`) — clones bitnet.cpp at a pinned
   commit, generates the LUT kernel header, and compiles `llama-server` and
   `llama-cli` via cmake.
2. **Runtime** (`python:3.13-slim-bookworm`) — runs `llama-server` + the FastAPI
   proxy.

Both stages are pinned to the same Debian suite because the runtime copies
compiled shared libraries out of the builder.

The build deliberately bypasses `setup_env.py`. On this path that script only
runs kernel codegen and cmake — its `prepare_model()` step skips conversion
whenever the GGUF already exists, and its `compile()` step runs *before*
`prepare_model()` and touches no model file. Calling cmake directly means the
builder needs neither the 1.2 GB model nor BitNet's `requirements.txt` (~2 GB of
torch, whose only consumer is the HF→GGUF converter that never runs here), and
it lets the build restrict targets and use `-j`.

Inference runs on **I2_S** kernels. The LUT codegen step is still required
because `src/CMakeLists.txt` compiles `ggml-bitnet-lut.cpp` unconditionally and
that file includes the generated header, but TL2 itself is disabled via
`-DBITNET_X86_TL2=OFF`, matching what `setup_env.py` passes on x86_64.

### Why the binary is called `llama-server`

This runs **bitnet.cpp**, not stock llama.cpp — the binary name is inherited,
not a sign of the wrong engine. There is no separate `bitnet-server`.

bitnet.cpp's own top-level `CMakeLists.txt` builds both its ternary kernels and
a vendored copy of llama.cpp:

```cmake
add_subdirectory(src)                  # ggml-bitnet-lut.cpp, ggml-bitnet-mad.cpp
add_subdirectory(3rdparty/llama.cpp)   # vendored dependency, not an alternative
set(LLAMA_BUILD_SERVER ON CACHE BOOL "Build llama.cpp server" FORCE)
```

llama.cpp is a component *inside* bitnet.cpp, and the server is force-enabled by
bitnet.cpp itself, so it is a first-class output of this build rather than
something bolted on here. The simplest proof that the ternary kernels are live:
stock llama.cpp cannot load an I2_S model at all, and this one serves one.

Note that the upstream changelog's embedding-model guides (0.6B/270M) and the
TL1/TL2 tiling work in `docs/codegen.md` do not apply to this deployment — the
first covers embedding models rather than generative ones, and the second covers
lookup-table kernels that are disabled here.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BITNET_API_KEY` | _(unset)_ | Require this key on `/v1` routes. Unset means open. |
| `MODEL_PATH` | `/app/models/model.gguf` | Path to GGUF model |
| `MODEL_ID` | `bitnet-b1.58-2B-4T` | Model ID returned by the API |
| `LLAMA_SERVER_PORT` | `8080` | Internal llama-server port |
| `BITNET_THREADS` | `4` | Inference threads (see note below) |
| `BITNET_CTX_SIZE` | `4096` | Context window size, served to the UI via `/v1/status` |
| `BITNET_QUEUE_TIMEOUT` | `3` | Seconds to wait for the inference slot before `503` |
| `BITNET_READ_TIMEOUT` | `900` | Backend read timeout for non-streaming requests |
| `BITNET_MAX_MESSAGES` | `200` | Maximum messages per request |
| `BITNET_SESSION_TTL` | `86400` | UI session cookie lifetime in seconds |
| `BITNET_ROLE_STOP_FALLBACK` | `0` | Re-enable legacy `"User:"` string stops |
| `BITNET_TRUSTED_PROXIES` | `127.0.0.1` | Hosts allowed to set `X-Forwarded-For` |
| `BITNET_DOWNLOAD_PATH` | `/app/downloads/download.zip` | File served by `/download` |

`BITNET_THREADS` defaults to 4, which is a conservative guess rather than a
measured value; upstream benchmarks x86 at 8 threads. Measure on the deployment
host before changing it — more threads than physical cores usually costs
throughput.

ggml defaults `GGML_NATIVE=ON`, so the binary is compiled `-march=native` for
whichever machine ran `docker build`. That is the right setting here because
`start.sh` builds on the host that serves. It only matters if you ever build the
image somewhere else and copy it — a different CPU would fault on instructions
the build host had.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The test suite runs against a stub of llama-server, so it needs neither the
model file nor a compiled backend. CI runs lint, tests, and hadolint; the image
build is not run in CI because it compiles bitnet.cpp from source and requires
the model in the build context.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build |
| `start.sh` | Build + run script |
| `app.py` | FastAPI proxy |
| `static/` | Web UI (HTML, CSS, JS, vendored libraries) |
| `entrypoint.sh` | Container startup and process supervision |
| `requirements.txt` | Runtime dependencies |
| `tests/` | Test suite, run against a stubbed backend |

## Network

The container joins `nginx-proxy-manager_default` with static IP `172.22.0.25`.
Reverse proxy to port 8010 over HTTP (not HTTPS).

Because no host port is published by default, `localhost:8010` will not reach
it. Use `HOST_PORT=8010 ./start.sh` for local testing, or exec into the
container.
