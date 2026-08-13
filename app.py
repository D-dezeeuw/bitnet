"""OpenAI-compatible HTTP facade in front of a single-slot bitnet.cpp llama-server.

The backend is bitnet.cpp's fork of llama-server, which serves one request at a
time on CPU. Everything here exists to make that safe to expose: a single
inference slot with deterministic admission, bounded inputs, and prompt
construction that matches what the model was actually trained on.
"""

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, ValidationError

from mcp_server import MCPDispatch, build_mcp_server

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bitnet-api")

BASE_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Single source of truth for configuration.

    Context size in particular used to be hardcoded in three places that
    disagreed (the UI, the Dockerfile, and entrypoint.sh); it is read once here
    and served to the UI from /v1/status so it can never drift again.
    """

    def __init__(self) -> None:
        self.llama_port = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
        self.llama_url = f"http://127.0.0.1:{self.llama_port}"
        self.model_id = os.environ.get("MODEL_ID", "bitnet-b1.58-2B-4T")
        self.ctx_size = int(os.environ.get("BITNET_CTX_SIZE", "4096"))
        self.api_key = os.environ.get("BITNET_API_KEY") or None

        # Admission control. queue_timeout is how long a caller waits for the
        # single inference slot before being told to come back.
        self.queue_timeout = float(os.environ.get("BITNET_QUEUE_TIMEOUT", "3"))

        # A 2B model on a handful of CPU threads is slow. The old blanket 120s
        # read timeout turned long completions into spurious 503s while the
        # backend was still working.
        self.read_timeout = float(os.environ.get("BITNET_READ_TIMEOUT", "900"))
        self.connect_timeout = float(os.environ.get("BITNET_CONNECT_TIMEOUT", "5"))

        self.max_messages = int(os.environ.get("BITNET_MAX_MESSAGES", "200"))
        self.session_ttl = int(os.environ.get("BITNET_SESSION_TTL", "86400"))

        # llama-server defaults repeat_penalty to 1.0, which is no penalty at
        # all. A 2B model left unpenalised loops: observed output restated the
        # same sentence until it hit n_predict instead of ending its turn.
        # 1.1 is llama.cpp's own long-standing default and is the smallest value
        # that reliably breaks that cycle; much above ~1.2 starts degrading
        # fluency, since legitimate repetition (names, list items) is punished
        # too. repeat_last_n is the window it applies over.
        self.repeat_penalty = float(os.environ.get("BITNET_REPEAT_PENALTY", "1.1"))
        self.repeat_last_n = int(os.environ.get("BITNET_REPEAT_LAST_N", "64"))

        # Pre-MDL-1 the prompt carried no turn separators, so generation had to
        # be stopped by string-matching "User:". That truncated any reply which
        # legitimately contained the string. The correct template makes
        # <|eot_id|> load-bearing instead; set this to re-enable the old
        # fallback if a model revision ever stops emitting it.
        self.role_stop_fallback = _env_flag("BITNET_ROLE_STOP_FALLBACK", False)

        self.static_dir = Path(os.environ.get("BITNET_STATIC_DIR", BASE_DIR / "static"))
        self.download_path = Path(
            os.environ.get("BITNET_DOWNLOAD_PATH", BASE_DIR / "downloads" / "download.zip")
        )

    @property
    def max_tokens_cap(self) -> int:
        return self.ctx_size


settings = Settings()

# Rough chars-per-token for this tokenizer. Only used to keep the prompt from
# crowding out the reply; the backend enforces the real limit.
CHARS_PER_TOKEN = 3.5

# Turn terminator the model was trained to emit. Note the GGUF declares
# eos_token as <|end_of_text|>, NOT this, so llama-server's automatic EOS stop
# never fires at end of turn -- listing it explicitly is required, not a hack.
EOT = "<|eot_id|>"
GENERATION_PROMPT = "Assistant: "
DEFAULT_STOPS = [EOT, "<|end_of_text|>"]
ROLE_STOP_FALLBACK = ["User:", "user:"]

SESSION_COOKIE = "bitnet_session"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to llama-server at %s", settings.llama_url)
    if settings.api_key:
        logger.info("MCP endpoint enabled at /mcp")
    else:
        logger.warning("MCP endpoint disabled: it requires BITNET_API_KEY")
    logger.info(
        "ctx=%d max_tokens_cap=%d auth=%s",
        settings.ctx_size,
        settings.max_tokens_cap,
        "on" if settings.api_key else "OFF",
    )
    if not settings.api_key:
        logger.warning(
            "BITNET_API_KEY is not set - all /v1 endpoints are unauthenticated."
        )
    # Created inside the running loop rather than at import: an asyncio.Lock
    # binds to the first loop that touches it, so a module-level one is a
    # latent footgun for anything that runs more than one loop.
    app.state.inference_lock = asyncio.Lock()
    app.state.client = httpx.AsyncClient(
        base_url=settings.llama_url,
        timeout=httpx.Timeout(
            connect=settings.connect_timeout,
            read=settings.read_timeout,
            write=30.0,
            pool=30.0,
        ),
    )
    # Built per-lifespan, not once at import. A StreamableHTTPSessionManager
    # refuses a second .run(), so a module-level instance works in production
    # (one lifespan) but breaks the moment anything starts the app twice in a
    # process -- which the test suite does for every test.
    #
    # A mounted sub-application's lifespan never runs, so .run() has to be
    # entered here or the first /mcp request fails on a missing task group.
    mcp_server = build_mcp_server(_mcp_generate, _mcp_status)
    app.state.mcp_asgi = mcp_server.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    # The session manager is an anyio task group, which must be entered and
    # exited from the same task. Holding it open across `yield` here would not
    # be: ASGI startup and shutdown can run in different tasks, which surfaces
    # as "Attempted to exit cancel scope in a different task". Giving it a
    # dedicated task that owns the whole `async with` keeps both ends together,
    # and cancelling that task is what closes it.
    started = asyncio.Event()

    async def _run_session_manager() -> None:
        async with mcp_server.session_manager.run():
            started.set()
            await asyncio.Event().wait()  # until cancelled at shutdown

    manager_task = asyncio.create_task(_run_session_manager())
    await started.wait()

    try:
        yield
    finally:
        manager_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await manager_task
        app.state.mcp_asgi = None
        await app.state.client.aclose()


app = FastAPI(title="BitNet API", lifespan=lifespan, docs_url=None, redoc_url=None)

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = CSP
    return response


# --------------------------------------------------------------------------
# Authentication
#
# The API key gates every /v1 route. The browser UI cannot send a bearer
# header, so it exchanges the key once for an HMAC-signed session cookie --
# without this, enabling BITNET_API_KEY silently broke /inference and the only
# two working states were "open to the internet" or "UI broken".
# --------------------------------------------------------------------------


def _sign(payload: str) -> str:
    assert settings.api_key is not None
    return hmac.new(
        settings.api_key.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()


def issue_session_token(now: int | None = None) -> str:
    expires = int(now if now is not None else time.time()) + settings.session_ttl
    return f"{expires}.{_sign(str(expires))}"


def session_token_valid(token: str) -> bool:
    if not settings.api_key or not token:
        return False
    expires_raw, _, signature = token.partition(".")
    if not signature:
        return False
    try:
        expires = int(expires_raw)
    except ValueError:
        return False
    if expires < time.time():
        return False
    return hmac.compare_digest(signature, _sign(expires_raw))


def _presented_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def check_api_key(request: Request) -> None:
    """Reject the request unless it carries a valid key or session cookie."""
    if not settings.api_key:
        return
    presented = _presented_key(request)
    if presented and hmac.compare_digest(presented, settings.api_key):
        return
    if session_token_valid(request.cookies.get(SESSION_COOKIE, "")):
        return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------


class Message(BaseModel):
    # Constrained to the three trained roles: an arbitrary string here would be
    # interpolated straight into the prompt and could forge a turn boundary.
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=settings.max_messages)
    max_tokens: int = Field(default=256, ge=1, le=settings.max_tokens_cap)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    n: int | None = Field(default=None, ge=1)
    stop: list[str] | None = Field(default=None, max_length=4)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    # Distinct from frequency_penalty: that one is OpenAI's additive logit
    # penalty, this is llama.cpp's multiplicative one, and the backend applies
    # them independently. Left unset the server default applies; 1.0 disables
    # it, which is what produced the looping this exists to stop.
    repeat_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    repeat_last_n: int | None = Field(default=None, ge=0, le=2048)
    # Resume the trailing assistant turn instead of opening a new one.
    continuation: bool = False


class SummarizeRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=settings.max_messages)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def build_prompt(messages: list[Message], *, continuation: bool = False) -> str:
    """Render messages using the model's real chat template.

    Mirrors tokenizer_config.json from microsoft/bitnet-b1.58-2B-4T:

        {role|capitalize}: {content|trim}<|eot_id|> ... then "Assistant: "

    Deliberately NOT delegated to llama-server's /v1/chat/completions: the
    template embedded in ggml-model-i2_s.gguf is mangled (it emits "Human:" and
    "BITNETAssistant:", places eos_token after the generation prompt, and drops
    system messages), so the backend would apply a worse format than this.

    BOS is not prepended -- llama-server adds it from add_bos_token metadata.
    """
    if continuation and messages and messages[-1].role == "assistant":
        head = "".join(
            f"{m.role.capitalize()}: {m.content.strip()}{EOT}" for m in messages[:-1]
        )
        # lstrip only: a trailing space in the partial is a real token boundary.
        return head + GENERATION_PROMPT + messages[-1].content.lstrip()
    body = "".join(f"{m.role.capitalize()}: {m.content.strip()}{EOT}" for m in messages)
    return body + GENERATION_PROMPT


def prompt_budget_chars(max_tokens: int) -> int:
    """Chars of prompt that still leave room for the requested reply."""
    return int(max(1, settings.ctx_size - max_tokens) * CHARS_PER_TOKEN)


def validate_budget(messages: list[Message], max_tokens: int) -> None:
    total = sum(len(m.content) for m in messages)
    budget = prompt_budget_chars(max_tokens)
    if total > budget:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt too long ({total} chars). With max_tokens={max_tokens} "
                f"and a {settings.ctx_size}-token context the limit is {budget} chars."
            ),
        )


def resolve_stops(extra: list[str] | None) -> list[str]:
    stops = list(DEFAULT_STOPS)
    if settings.role_stop_fallback:
        stops += ROLE_STOP_FALLBACK
    if extra:
        stops = [s for s in extra if s] + stops
    return stops


def make_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# --------------------------------------------------------------------------
# Admission control
# --------------------------------------------------------------------------


class Slot:
    """A held inference slot. Release is idempotent so the streaming path can
    release from both the generator's finally and the response background task
    without double-releasing."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self._lock.release()


async def acquire_slot(lock: asyncio.Lock) -> Slot:
    """Take the single inference slot or fail with 503.

    Acquiring here rather than checking `lock.locked()` and acquiring later
    closes the race where two callers both passed the check and the second
    silently blocked on an already-200 streaming response.
    """
    try:
        await asyncio.wait_for(lock.acquire(), timeout=settings.queue_timeout)
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing another request. Please retry.",
        ) from None
    return Slot(lock)


def backend_payload(req: ChatRequest) -> dict:
    payload = {
        "prompt": build_prompt(req.messages, continuation=req.continuation),
        "n_predict": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
        "stop": resolve_stops(req.stop),
        # Always sent, unlike the optional sampler fields below: omitting it
        # means llama-server applies its own 1.0 default and the model loops.
        # The per-request value wins so a caller can still opt out with 1.0.
        "repeat_penalty": (
            req.repeat_penalty
            if req.repeat_penalty is not None
            else settings.repeat_penalty
        ),
        "repeat_last_n": (
            req.repeat_last_n
            if req.repeat_last_n is not None
            else settings.repeat_last_n
        ),
    }
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.presence_penalty is not None:
        payload["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        payload["frequency_penalty"] = req.frequency_penalty
    return payload


def _backend_error(exc: Exception) -> HTTPException:
    if isinstance(exc, httpx.HTTPStatusError):
        logger.error(
            "llama-server returned %s: %s", exc.response.status_code, exc.response.text
        )
        return HTTPException(status_code=502, detail="Inference backend error")
    logger.error("llama-server unavailable: %s", exc)
    return HTTPException(status_code=503, detail="Inference backend unavailable")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, request: Request):
    check_api_key(request)
    if req.model and req.model != settings.model_id:
        raise HTTPException(
            status_code=404, detail=f"Model '{req.model}' not found on this server"
        )
    if req.n is not None and req.n != 1:
        raise HTTPException(
            status_code=400, detail="Only n=1 is supported by this server"
        )
    validate_budget(req.messages, req.max_tokens)

    payload = backend_payload(req)
    logger.info(
        "Chat request: %d message(s), max_tokens=%d, stream=%s, continuation=%s",
        len(req.messages),
        req.max_tokens,
        req.stream,
        req.continuation,
    )

    slot = await acquire_slot(request.app.state.inference_lock)

    if req.stream:
        return StreamingResponse(
            stream_completion(request, payload, slot),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        start = time.time()
        try:
            resp = await request.app.state.client.post("/completion", json=payload)
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise _backend_error(exc) from exc
        data = resp.json()
    finally:
        slot.release()

    content = data.get("content", "")
    tokens_predicted = data.get("tokens_predicted", 0)
    tokens_evaluated = data.get("tokens_evaluated", 0)
    # Report what actually happened rather than always claiming "stop".
    finish_reason = "length" if data.get("stopped_limit", False) else "stop"
    logger.info(
        "Chat response: %d chars in %dms (%s)",
        len(content),
        int((time.time() - start) * 1000),
        finish_reason,
    )
    return {
        "id": make_chat_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": tokens_evaluated,
            "completion_tokens": tokens_predicted,
            "total_tokens": tokens_evaluated + tokens_predicted,
        },
    }


async def stream_completion(
    request: Request, payload: dict, slot: Slot
) -> AsyncIterator[str]:
    start = time.time()
    total = 0
    chat_id = make_chat_id()
    created = int(time.time())
    # No read timeout while streaming: token gaps on CPU inference are long and
    # legitimate, and the slot is held for the whole duration anyway.
    stream_timeout = httpx.Timeout(
        connect=settings.connect_timeout, read=None, write=30.0, pool=30.0
    )
    try:
        async with request.app.state.client.stream(
            "POST", "/completion", json=payload, timeout=stream_timeout
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                chunk = json.loads(raw)
                token = chunk.get("content", "")
                stop = chunk.get("stop", False)
                total += len(token)
                finish_reason = None
                if stop:
                    finish_reason = (
                        "length" if chunk.get("stopped_limit", False) else "stop"
                    )
                sse = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": settings.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {} if stop else {"content": token},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse)}\n\n"
                if stop:
                    break
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as SSE
        logger.error("Stream error: %s", exc)
        yield f"data: {json.dumps({'error': 'Inference backend error'})}\n\n"
    finally:
        slot.release()
    logger.info(
        "Stream response: %d chars in %dms", total, int((time.time() - start) * 1000)
    )
    yield "data: [DONE]\n\n"


SUMMARY_MAX_TOKENS = 200


@app.post("/v1/summarize")
async def summarize_context(req: SummarizeRequest, request: Request):
    """Compact a conversation into a short context summary.

    This is an inference endpoint like any other: it authenticates, takes the
    single slot, and honours the same prompt budget.
    """
    check_api_key(request)
    validate_budget(req.messages, SUMMARY_MAX_TOKENS)

    conversation = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    prompt = build_prompt(
        [
            Message(
                role="user",
                content=(
                    "Below is a conversation. Summarize the key facts, decisions, "
                    "user preferences, and important context in a concise paragraph. "
                    "Preserve any specific names, numbers, or technical details. "
                    "Do not add commentary.\n\n" + conversation
                ),
            )
        ]
    )
    payload = {
        "prompt": prompt,
        "n_predict": SUMMARY_MAX_TOKENS,
        "temperature": 0.3,
        "stop": resolve_stops(None),
    }
    logger.info("Summarize request: %d messages", len(req.messages))

    slot = await acquire_slot(request.app.state.inference_lock)
    try:
        try:
            resp = await request.app.state.client.post("/completion", json=payload)
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise _backend_error(exc) from exc
        data = resp.json()
    finally:
        slot.release()

    summary = data.get("content", "").strip()
    logger.info("Summary generated: %d chars", len(summary))
    return {"summary": summary}


@app.post("/v1/auth")
async def create_session(request: Request, response: Response):
    """Exchange an API key for a session cookie so the browser UI can call /v1."""
    if not settings.api_key:
        return {"authenticated": True, "required": False}
    body = {}
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - empty or malformed body is just a failure
        pass
    presented = (body.get("api_key") or "").strip() or _presented_key(request)
    if not presented or not hmac.compare_digest(presented, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    forwarded = request.headers.get("x-forwarded-proto", "")
    secure = request.url.scheme == "https" or forwarded.split(",")[0].strip() == "https"
    response.set_cookie(
        SESSION_COOKIE,
        issue_session_token(),
        max_age=settings.session_ttl,
        httponly=True,
        samesite="strict",
        secure=secure,
        path="/",
    )
    return {"authenticated": True, "required": True}


@app.get("/download")
async def download():
    if not settings.download_path.is_file():
        raise HTTPException(status_code=404, detail="No download is available")
    return FileResponse(
        settings.download_path,
        filename=settings.download_path.name,
        media_type="application/zip",
    )


@app.get("/v1/models")
def list_models(request: Request):
    check_api_key(request)
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "microsoft",
            }
        ],
    }


@app.get("/v1/status")
async def status(request: Request):
    """Runtime facts the UI needs. Context size is served from here so it is
    never hardcoded client-side."""
    return {
        "busy": request.app.state.inference_lock.locked(),
        "model": settings.model_id,
        "context_size": settings.ctx_size,
        "max_tokens_cap": settings.max_tokens_cap,
        "auth_required": bool(settings.api_key),
    }


@app.get("/health")
async def health(request: Request):
    try:
        resp = await request.app.state.client.get("/health", timeout=5.0)
        return {"status": "ok", "backend": resp.json()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check failed: %s", exc)
        raise HTTPException(
            status_code=503, detail={"status": "degraded", "backend": "unavailable"}
        ) from exc


@app.get("/")
async def root():
    """Send the bare domain to the UI.

    There has never been a route here -- the original app.py served the UI at
    /inference only -- so hitting the domain returned a bare JSON 404 that reads
    like the deployment is broken when it is fine. Redirect rather than moving
    the UI, so existing links to /inference keep working.
    """
    return RedirectResponse("/inference")


@app.get("/inference")
async def inference_ui():
    index = settings.static_dir / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI assets are not installed")
    return FileResponse(index, media_type="text/html")


if settings.static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
else:  # pragma: no cover - only hit in a misconfigured deployment
    logger.warning("Static directory %s missing; /inference will 404", settings.static_dir)


# --------------------------------------------------------------------------
# MCP endpoint
#
# MCP clients (Claude Desktop's connectors, among others) cannot consume
# /v1/chat/completions: a connector is an MCP server providing tools, not an
# OpenAI-compatible chat endpoint. /mcp wraps the same inference path in the
# protocol those clients speak.
# --------------------------------------------------------------------------


async def _mcp_generate(
    prompt: str, system: str | None, max_tokens: int, temperature: float
) -> dict:
    """Run one non-streaming completion for an MCP tool call.

    Goes through ChatRequest and backend_payload rather than talking to the
    backend directly, so the prompt template, stop tokens and validation stay
    identical to /v1/chat/completions. A second construction path here is
    exactly how the prompt format drifted before.
    """
    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=prompt))

    try:
        req = ChatRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
    except ValidationError as exc:
        # Surface the constraint rather than a pydantic dump: the caller is a
        # model deciding what to do next, and "max_tokens must be 1..4096" is
        # actionable where a traceback is not.
        raise ValueError(f"Invalid arguments: {exc.errors()[0]['msg']}") from exc

    validate_budget(req.messages, req.max_tokens)
    payload = backend_payload(req)

    slot = await acquire_slot(app.state.inference_lock)
    try:
        try:
            resp = await app.state.client.post("/completion", json=payload)
            resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise ValueError(f"BitNet backend unavailable: {exc}") from exc
        data = resp.json()
    finally:
        slot.release()

    return {
        "content": data.get("content", ""),
        "finish_reason": "length" if data.get("stopped_limit", False) else "stop",
    }


async def _mcp_status() -> dict:
    try:
        resp = await app.state.client.get("/health", timeout=5.0)
        backend = "ok" if resp.status_code == 200 else "degraded"
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP status check failed: %s", exc)
        backend = "unavailable"
    return {
        "backend": backend,
        "model": settings.model_id,
        "context_size": settings.ctx_size,
        "max_tokens_cap": settings.max_tokens_cap,
        "busy": app.state.inference_lock.locked(),
    }


# Both the target app and the key are resolved per request: lifespan rebuilds
# the former, and the latter is read from settings so tests (and anything that
# reconfigures at runtime) are not stuck with an import-time snapshot.
#
# The MCP app itself is built with stateless_http (no per-client state worth
# keeping, and a stateless endpoint survives a restart mid-conversation) and
# json_response (every tool returns a single result with no interim progress,
# and plain JSON passes through reverse proxies that buffer SSE).
app.add_middleware(
    MCPDispatch,
    get_target=lambda: getattr(app.state, "mcp_asgi", None),
    get_api_key=lambda: settings.api_key,
)
