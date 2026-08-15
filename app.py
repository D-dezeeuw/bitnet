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
from starlette.background import BackgroundTask

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

        # Off by default, after briefly being 1.1. It penalises every token in
        # the window, including the function words and subject nouns a sentence
        # structurally needs, so on a 2B model it does not so much suppress
        # repetition as force lower-probability substitutes: output degraded
        # into "strings aren't be taken" and "then is won't show". DRY below
        # handles the phrase-level repetition this was reaching for, without
        # touching individual words. Raise it only if DRY proves insufficient.
        self.repeat_penalty = float(os.environ.get("BITNET_REPEAT_PENALTY", "1.0"))
        self.repeat_last_n = int(os.environ.get("BITNET_REPEAT_LAST_N", "64"))

        # DRY sampling. repeat_penalty acts on single tokens in a window, which
        # does little against this model's actual failure: a phrase repeated
        # with small substitutions ("let's make it simple/easy/clear for you").
        # DRY penalises repeated n-grams and is the sampler built for that.
        # llama-server defaults dry_multiplier to 0.0, i.e. off, so it has to be
        # sent explicitly. 0.8 is llama.cpp's suggested starting strength;
        # allowed_length 2 lets ordinary bigrams recur while catching longer
        # cycles. Set the multiplier to 0.0 to disable.
        self.dry_multiplier = float(os.environ.get("BITNET_DRY_MULTIPLIER", "0.8"))
        self.dry_base = float(os.environ.get("BITNET_DRY_BASE", "1.75"))
        self.dry_allowed_length = int(os.environ.get("BITNET_DRY_ALLOWED_LENGTH", "2"))

        # Sampling is deliberately conservative. 1.58-bit quantisation has
        # already blurred the model's probability distribution, so a temperature
        # that reads as merely lively on a large model is destructive here --
        # every coherent reply observed came from temperature 0, every rambling
        # one from 0.7. min_p truncates the tail by relative probability, which
        # holds up better than top_p when the distribution is this flat.
        self.temperature = float(os.environ.get("BITNET_TEMPERATURE", "0.3"))
        self.min_p = float(os.environ.get("BITNET_MIN_P", "0.1"))

        # Prepended when a caller sends no system message. Without any framing
        # the model drifts into free association rather than answering; small
        # instruct models lean on a system turn to stay anchored. Set empty to
        # send nothing.
        self.system_prompt = os.environ.get(
            "BITNET_SYSTEM_PROMPT",
            "You are a helpful, factual assistant. Answer the user's question "
            "directly and concisely, in a few short sentences unless more "
            "detail is clearly needed. Never repeat a sentence you have "
            "already written. When the question is answered, stop. If you do "
            "not know something, say you do not know.",
        ).strip()

        # Server-side guardrail against degenerate repetition. This model can
        # fail to emit any end-of-turn token and then restate its answer until
        # n_predict; the guard watches the generated text, and when a phrase
        # repeats this many times consecutively it aborts the backend request
        # (the pinned llama-server polls for disconnect and frees its slot),
        # trims the repeats, and finishes the response cleanly. 0 disables.
        # 4 rather than 3: a degenerate loop restates dozens of times, so
        # the extra repeat costs little, while three identical consecutive
        # lines (code, list items) are plausible legitimate output that a
        # threshold of 3 would corrupt.
        self.loop_guard_repeats = (
            int(os.environ.get("BITNET_LOOP_GUARD_REPEATS", "4"))
            if _env_flag("BITNET_LOOP_GUARD", True)
            else 0
        )

        # Which chat template to render. "hf" is tokenizer_config.json's
        # (User:/Assistant: with <|eot_id|> terminators) -- the documented
        # format for the instruct weights. "bitnet" is the template Microsoft's
        # own GGUF conversion script embeds in the file (Human:/BITNETAssistant:
        # with blank lines and <|end_of_text|> terminators) and what their demo
        # effectively uses. The two disagree; which one this quantised
        # checkpoint actually answers better under is an empirical question,
        # so it is a deploy-time switch rather than a hardcode.
        fmt = os.environ.get("BITNET_PROMPT_FORMAT", "hf").strip().lower()
        if fmt not in ("hf", "bitnet"):
            logger.warning("Unknown BITNET_PROMPT_FORMAT %r; using 'hf'", fmt)
            fmt = "hf"
        self.prompt_format = fmt

        # Stop when the model writes the next speaker's label. Once a
        # historical fallback (BITNET_ROLE_STOP_FALLBACK, default off) on the
        # theory that <|eot_id|> was load-bearing; probing the deployment
        # disproved that -- the model routinely ends a turn by writing
        # "Human:" or "User:" and no end token at all, so this is often the
        # ONLY stop that can fire. The cost is a reply that legitimately
        # contains "User:" mid-sentence being cut there, which is a far
        # smaller harm than generating to the token cap.
        self.role_stops = _env_flag("BITNET_ROLE_STOPS", True)

        # Kill switch for the MCP endpoint. It shares a process with the API, so
        # a fault there takes the whole container down with it -- entrypoint.sh
        # exits when uvicorn does, and the restart policy then loops. Turning
        # this off keeps /v1 and the UI serving while /mcp is investigated,
        # rather than leaving a crash-looping container as the only option.
        self.mcp_enabled = _env_flag("BITNET_MCP_ENABLED", True)

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

# End-of-turn handling, established against the pinned backend rather than
# assumed (the llama.cpp fork vendored by the BitNet commit in the Dockerfile):
#
# - The GGUF's metadata is incomplete: it declares eos as <|end_of_text|>
#   (128001) and carries no eot key, while the HF repo's generation_config
#   stops on BOTH 128001 and <|eot_id|> (128009).
# - The pinned llama.cpp neutralises that itself: llama-vocab.cpp force-adds
#   "<|eot_id|>" and "<|end_of_text|>" to its end-of-generation set BY TOKEN
#   TEXT, and the server stops at token level on either. This needs neither
#   --special nor these string stops.
# - The strings below are therefore belt-and-braces for other backends, not
#   the mechanism. When output still runs to n_predict, the model failed to
#   EMIT an end token at all -- no stop configuration can fix that, which is
#   what LoopGuard below is for.
EOT = "<|eot_id|>"
END_OF_TEXT = "<|end_of_text|>"

# No trailing space, deliberately. "Assistant: " encodes the bare space as a
# standalone token, but in training text that space is merged into the reply's
# first token (" Sure"). Conditioning a 1.58-bit model on that out-of-
# distribution boundary token is a credible degenerate-start trigger. The model
# emits the leading space itself now, and the API strips it from the first
# token of a fresh (non-continuation) reply.
GENERATION_PROMPT = "Assistant:"
DEFAULT_STOPS = [EOT, END_OF_TEXT]

# The next turn's role label, per format. Load-bearing rather than a fallback:
# probing the deployment showed the model frequently ends its turn by writing
# the NEXT SPEAKER'S LABEL instead of an end-of-turn token -- an observed reply
# ran "...It is why things fall down.Human: Explain string theory..." and then
# answered itself. That label is unambiguously a turn boundary, so cutting
# there is correct, and it is the only stop that fires when the model emits no
# end token at all. This is what llama.cpp's own -r/reverse-prompt does.
ROLE_STOPS = {
    "hf": ["User:", "System:"],
    "bitnet": ["Human:"],
}

SESSION_COOKIE = "bitnet_session"

# How long startup waits for the MCP session manager before giving up on it and
# serving without /mcp. Generous, because it only bounds a failure path.
MCP_STARTUP_TIMEOUT = 30.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to llama-server at %s", settings.llama_url)
    if not settings.mcp_enabled:
        logger.warning("MCP endpoint disabled by BITNET_MCP_ENABLED")
    elif not settings.api_key:
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
    app.state.mcp_asgi = None
    manager_task: asyncio.Task | None = None

    if settings.mcp_enabled:
        # Built per-lifespan, not once at import. A StreamableHTTPSessionManager
        # refuses a second .run(), so a module-level instance works in
        # production (one lifespan) but breaks any process that starts the app
        # twice -- which the test suite does for every test.
        #
        # A mounted sub-application's lifespan never runs, so .run() has to be
        # entered here or the first /mcp request fails on a missing task group.
        mcp_server = build_mcp_server(_mcp_generate, _mcp_status)
        mcp_asgi = mcp_server.streamable_http_app(
            streamable_http_path="/",
            json_response=True,
            stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            ),
        )
        # The session manager is an anyio task group, which must be entered and
        # exited from the same task. Holding it open across `yield` would not
        # be: ASGI startup and shutdown can run in different tasks, which
        # surfaces as "Attempted to exit cancel scope in a different task".
        # A dedicated task owning the whole `async with` keeps both ends
        # together, and cancelling it is what closes the group.
        started = asyncio.Event()

        async def _run_session_manager() -> None:
            async with mcp_server.session_manager.run():
                started.set()
                await asyncio.Event().wait()  # until cancelled at shutdown

        manager_task = asyncio.create_task(_run_session_manager())

        # Bounded, and watching the task as well as the event. A bare
        # `await started.wait()` blocks forever if the manager raises before
        # setting it: uvicorn never finishes starting, the healthcheck never
        # passes, and the container restart-loops with no error to show for it.
        # Whatever goes wrong here, the API and UI must still come up.
        waiter = asyncio.create_task(started.wait())
        try:
            await asyncio.wait(
                {waiter, manager_task},
                timeout=MCP_STARTUP_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            waiter.cancel()

        if started.is_set():
            app.state.mcp_asgi = mcp_asgi
            logger.info("MCP endpoint ready at /mcp")
        else:
            exc = manager_task.done() and manager_task.exception()
            logger.error(
                "MCP session manager failed to start (%s); /mcp is disabled but "
                "the API and UI are unaffected.",
                exc or f"no response in {MCP_STARTUP_TIMEOUT}s",
            )
            manager_task.cancel()
            manager_task = None
    else:
        logger.warning("MCP endpoint disabled by BITNET_MCP_ENABLED")

    try:
        yield
    finally:
        if manager_task is not None:
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
    # The UI is versioned with the image, not the URL, so a browser holding a
    # cached app.js silently runs an older client against a newer API -- which
    # presented as a deployed fix appearing not to work at all. no-cache means
    # "revalidate before use", and StaticFiles serves ETags, so an unchanged
    # asset still costs only a 304.
    if request.url.path.startswith("/static") or request.url.path == "/inference":
        response.headers["Cache-Control"] = "no-cache"
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


def request_is_authorized(request: Request) -> bool:
    """True when the request carries a valid key or session cookie."""
    if not settings.api_key:
        return True
    presented = _presented_key(request)
    if presented and hmac.compare_digest(presented, settings.api_key):
        return True
    return session_token_valid(request.cookies.get(SESSION_COOKIE, ""))


def check_api_key(request: Request) -> None:
    """Reject the request unless it carries a valid key or session cookie."""
    if not request_is_authorized(request):
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
    # Defaults to the configured value rather than a literal, so the server-wide
    # setting is what a caller who says nothing actually gets.
    temperature: float = Field(default=settings.temperature, ge=0.0, le=2.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
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
    # 0.0 disables DRY; the upper bound is well past anything useful, but
    # leaves room to experiment on a model that loops this readily.
    dry_multiplier: float | None = Field(default=None, ge=0.0, le=5.0)
    # Resume the trailing assistant turn instead of opening a new one.
    continuation: bool = False


class SummarizeRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=settings.max_messages)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def build_prompt(messages: list[Message], *, continuation: bool = False) -> str:
    """Render messages in the configured chat template.

    "hf" mirrors tokenizer_config.json from microsoft/bitnet-b1.58-2B-4T:

        {role|capitalize}: {content|trim}<|eot_id|> ... then "Assistant:"

    "bitnet" mirrors the template embedded in the GGUF itself (written by
    Microsoft's conversion script and used by their own demo):

        Human: {content}\n\nBITNETAssistant: {reply}<|end_of_text|> ...

    with two deliberate repairs to that template: the eos_token it appends
    directly after the generation prompt is dropped (generating after an eos
    is nonsensical -- a conversion-script bug), and a system message, which it
    has no branch for, is folded into the first user turn.

    Neither format ends with a trailing space: the model emits the reply's
    leading space itself (see GENERATION_PROMPT). BOS is not prepended --
    llama-server adds it from add_bos_token metadata.
    """
    if settings.prompt_format == "bitnet":
        return _build_prompt_bitnet(messages, continuation=continuation)

    if continuation and messages and messages[-1].role == "assistant":
        head = "".join(
            f"{m.role.capitalize()}: {m.content.strip()}{EOT}" for m in messages[:-1]
        )
        # The space the fresh path omits is re-added here: tokenizing
        # "Assistant: partial" in one string BPE-merges " partial" exactly as
        # training did, so the continuation path stays in-distribution.
        return head + GENERATION_PROMPT + " " + messages[-1].content.lstrip()
    body = "".join(f"{m.role.capitalize()}: {m.content.strip()}{EOT}" for m in messages)
    return body + GENERATION_PROMPT


BITNET_GENERATION_PROMPT = "BITNETAssistant:"


def _build_prompt_bitnet(
    messages: list[Message], *, continuation: bool = False
) -> str:
    """Render the GGUF-embedded template. See build_prompt for the repairs.

    The template has no system branch, so EVERY system message -- wherever it
    sits, including the compaction summaries the UI injects mid-conversation
    -- is folded into the next user turn rather than dropped; dropping one
    silently would discard exactly the summarized history compaction was
    preserving. A message list that ends without a user turn still gets the
    generation prompt appended, so generation always starts inside an
    assistant turn rather than dangling after an end-of-text token.
    """
    msgs = list(messages)

    partial: str | None = None
    if continuation and msgs and msgs[-1].role == "assistant":
        partial = msgs[-1].content.lstrip()
        msgs = msgs[:-1]

    parts: list[str] = []
    pending_system: list[str] = []
    for m in msgs:
        if m.role == "system":
            pending_system.append(m.content.strip())
        elif m.role == "user":
            content = m.content.strip()
            if pending_system:
                content = "\n\n".join([*pending_system, content])
                pending_system = []
            parts.append(f"Human: {content}\n\n{BITNET_GENERATION_PROMPT}")
        elif m.role == "assistant":
            parts.append(f" {m.content.strip()}{END_OF_TEXT}")
    if pending_system:
        # System content with no user turn after it still reaches the model.
        parts.append(
            "Human: " + "\n\n".join(pending_system)
            + f"\n\n{BITNET_GENERATION_PROMPT}"
        )

    prompt = "".join(parts)
    if partial is not None:
        if not prompt.endswith(BITNET_GENERATION_PROMPT):
            prompt += BITNET_GENERATION_PROMPT
        return prompt + " " + partial
    if not prompt.endswith(BITNET_GENERATION_PROMPT):
        prompt += BITNET_GENERATION_PROMPT
    return prompt


def with_default_system(messages: list[Message], max_tokens: int) -> list[Message]:
    """Prepend the configured system prompt when the caller supplied none.

    Applied before budget validation, not inside build_prompt, so the added
    characters are counted against the context like any other message rather
    than slipping in after the check meant to bound them.

    Skipped when it would not fit. The prompt is an improvement, not a
    requirement, and silently injecting it into a request that was already at
    the context limit would turn a working call into a 400 -- at
    max_tokens == ctx_size the budget is a few characters, so any system prompt
    overflows it. The caller's own content always takes priority.

    A caller's own system message also wins: this fills a gap, it does not
    override intent.
    """
    if not settings.system_prompt:
        return messages
    if any(m.role == "system" for m in messages):
        return messages
    candidate = [Message(role="system", content=settings.system_prompt), *messages]
    if sum(len(m.content) for m in candidate) > prompt_budget_chars(max_tokens):
        logger.debug("Default system prompt skipped: no room in the budget")
        return messages
    return candidate


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
    if settings.role_stops:
        stops += ROLE_STOPS.get(settings.prompt_format, [])
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
        # DRY, sent for the same reason: the backend default of 0.0 disables it.
        # repeat_penalty alone does not fix this model's failure mode, which is
        # a repeated *phrase* rather than a repeated token -- observed output
        # cycled "let's make it simple/easy/clear for you" indefinitely. A token
        # penalty barely touches that, because each variant differs by a word
        # and the shared tokens are spread across a long window. DRY penalises
        # repeated n-grams, which is precisely the pattern.
        "dry_multiplier": (
            req.dry_multiplier
            if req.dry_multiplier is not None
            else settings.dry_multiplier
        ),
        "dry_base": settings.dry_base,
        "dry_allowed_length": settings.dry_allowed_length,
        # -1 means "consider the whole context" -- a loop that starts early must
        # still be penalised once generation is well past it.
        "dry_penalty_last_n": -1,
        # Sent always for the same reason as the penalties: the useful value is
        # not the backend's default. min_p keeps tokens by relative probability,
        # which degrades more gracefully than top_p on a flat distribution.
        "min_p": req.min_p if req.min_p is not None else settings.min_p,
        # Also always sent, because OMITTING top_p does not disable it -- the
        # backend then applies its own 0.95 default, silently overlaying the
        # min_p-first design with a second truncation. 1.0 is the value that
        # actually turns it off.
        "top_p": req.top_p if req.top_p is not None else 1.0,
    }
    if req.presence_penalty is not None:
        payload["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        payload["frequency_penalty"] = req.frequency_penalty
    return payload


def finish_reason_from(data: dict) -> Literal["length", "stop"]:
    """Map a llama-server result onto an OpenAI finish_reason.

    Reads both spellings on purpose. Older llama.cpp reports the boolean
    `stopped_limit`; newer versions replaced it with `stop_type`, one of
    "limit" / "eos" / "word". Which one bitnet.cpp's fork emits depends on the
    llama.cpp revision it vendors, and reading only the old name meant a reply
    truncated at max_tokens was reported as a clean "stop" -- observed in
    production, where a 120-token cap returned exactly 120 tokens, cut
    mid-sentence, labelled "stop".

    That mattered beyond cosmetics: the UI's Continue button and the MCP tool's
    truncation notice both key off this, so neither fired when they should.
    """
    if data.get("stop_type") == "limit":
        return "length"
    if data.get("stopped_limit", False):
        return "length"
    return "stop"


def _backend_error(exc: Exception) -> HTTPException:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.text
        except httpx.ResponseNotRead:  # streaming response never read
            body = "<unread streaming body>"
        logger.error(
            "llama-server returned %s: %s", exc.response.status_code, body
        )
        return HTTPException(status_code=502, detail="Inference backend error")
    logger.error("llama-server unavailable: %s", exc)
    return HTTPException(status_code=503, detail="Inference backend unavailable")


class LoopGuard:
    """Detects and trims degenerate repetition in generated text.

    The failure this exists for: the model emits no end-of-turn token and
    restates its answer -- verbatim or lightly mutated -- until n_predict.
    Sampler penalties cannot end a turn (they only demote tokens already
    seen; they cannot promote an end token the model is not producing), so
    the proxy has to be able to cut the generation itself.

    Detection is deliberately conservative: it trips only when the text ENDS
    with the same block of at least MIN_PHRASE characters repeated `repeats`
    times consecutively. Legitimate prose almost never does that; degenerate
    loops always eventually do. Mutated loops (word-substituted restatements)
    are left to DRY, which is the sampler built for them.
    """

    MIN_PHRASE = 12
    # 400 rather than 200: the headline failure is the model restating its
    # ENTIRE answer verbatim, and a two-sentence answer is easily 200+ chars.
    MAX_PHRASE = 400
    # Blocks made of one or two distinct characters -- dashes, table rules
    # like "---|---|", pure whitespace -- repeat legitimately in banners,
    # markdown and formatting. They only count as a loop at this much larger
    # size, so a 40-dash divider survives while a 120+ char character flood
    # (a real degenerate mode, including endless newlines) is still caught.
    LOW_DIVERSITY_MIN = 40

    def __init__(self, repeats: int) -> None:
        # 0 disables; anything else is clamped to >=2, because repeats=1
        # would make the all() below vacuously true and abort every
        # generation at its 12th character.
        self.repeats = max(repeats, 2) if repeats else 0
        self.text = ""
        self.tripped = False
        self._phrase = ""

    def feed(self, token_text: str) -> bool:
        """Append newly generated text; True when the loop threshold is hit."""
        self.text += token_text
        if not self.repeats or self.tripped or len(token_text) == 0:
            return self.tripped
        n = self.repeats
        limit = min(self.MAX_PHRASE, len(self.text) // n)
        for size in range(self.MIN_PHRASE, limit + 1):
            block = self.text[-size:]
            if len(set(block)) <= 2 and size < self.LOW_DIVERSITY_MIN:
                continue
            if all(
                self.text[-(i + 1) * size : len(self.text) - i * size] == block
                for i in range(1, n)
            ):
                self.tripped = True
                self._phrase = block
                logger.warning(
                    "Loop guard tripped: %r repeated %dx; aborting generation",
                    block[:60],
                    n,
                )
                break
        return self.tripped

    def trimmed(self) -> str:
        """The text with trailing repeats collapsed to a single occurrence.

        The detected block can be a rotation of the true phrase when the
        tripping chunk straddles a repeat boundary, which leaves a partial
        copy dangling after the collapse; the final pass cuts that fragment
        so the reply does not end mid-word with a visible loop artifact.
        """
        if not self.tripped:
            return self.text
        out = self.text
        while out.endswith(self._phrase * 2):
            out = out[: -len(self._phrase)]
        for k in range(len(self._phrase) - 1, 0, -1):
            if out.endswith(self._phrase + self._phrase[:k]):
                out = out[:-k]
                break
        return out


async def run_completion(
    state, payload: dict, *, strip_leading_space: bool
) -> dict:
    """Run one completion against llama-server, streaming backend-side.

    Every non-streaming caller goes through here rather than client.post, for
    one reason: the loop guard can only save compute if it sees text as it is
    generated. On a guard trip the stream context is exited early, httpx
    closes the connection, and the pinned llama-server polls for disconnect
    and frees its single slot -- so a runaway generation costs seconds, not
    the full n_predict budget.

    Returns content, finish_reason, token counts (from the backend's final
    chunk when generation completed; estimated from chunk count when the
    guard aborted first), and whether the guard tripped.
    """
    payload = {**payload, "stream": True}
    guard = LoopGuard(settings.loop_guard_repeats)
    chunk_count = 0
    tokens_predicted = 0
    tokens_evaluated = 0
    saw_final_chunk = False
    finish_reason: str = "stop"
    stream_timeout = httpx.Timeout(
        connect=settings.connect_timeout,
        read=settings.read_timeout,
        write=30.0,
        pool=30.0,
    )
    try:
        async with state.client.stream(
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
                if chunk.get("stop", False):
                    # Final chunk: authoritative counts and stop reason. Its
                    # content is empty on this backend; do not feed the guard.
                    saw_final_chunk = True
                    tokens_predicted = chunk.get("tokens_predicted", chunk_count)
                    tokens_evaluated = chunk.get("tokens_evaluated", 0)
                    finish_reason = finish_reason_from(chunk)
                    break
                chunk_count += 1
                if guard.feed(chunk.get("content", "")):
                    # Exiting the context closes the connection; the backend
                    # cancels the generation and frees its slot. The final
                    # counts chunk never arrives, so both counts are
                    # estimates: chunks are ~1 token each, and the prompt is
                    # estimated by the same chars-per-token ratio the budget
                    # check uses. Approximate beats reporting a 0-token
                    # prompt to clients doing context accounting.
                    saw_final_chunk = True
                    finish_reason = "stop"
                    tokens_predicted = chunk_count
                    tokens_evaluated = int(
                        len(payload.get("prompt", "")) / CHARS_PER_TOKEN
                    )
                    break
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        raise _backend_error(exc) from exc

    if not saw_final_chunk:
        # The stream ended with neither the backend's final chunk nor a guard
        # trip -- the generation did not complete. Labelling the partial text
        # "stop" would present a truncated reply as finished (and silence the
        # UI's Continue button and the MCP truncation notice), so report it
        # as cut short.
        logger.warning("Backend stream ended without a final chunk")
        finish_reason = "length"
        tokens_predicted = chunk_count
        tokens_evaluated = int(len(payload.get("prompt", "")) / CHARS_PER_TOKEN)

    content = guard.trimmed()
    if strip_leading_space and content.startswith(" "):
        # The generation prompt no longer carries the boundary space, so the
        # model's first token supplies it; it is formatting, not content.
        content = content[1:]
    return {
        "content": content,
        "finish_reason": finish_reason,
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": tokens_evaluated,
        "guard_tripped": guard.tripped,
    }


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
    req.messages = with_default_system(req.messages, req.max_tokens)
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

    # Whether the prompt actually resumed a partial assistant turn -- not
    # merely whether the caller set the flag. build_prompt ignores
    # continuation when the last message is not an assistant turn (the UI hits
    # this by stopping a reply before the first token), and the leading-space
    # strip must follow what the prompt did, or the boundary space leaks into
    # fresh replies.
    resumed = bool(
        req.continuation and req.messages and req.messages[-1].role == "assistant"
    )

    if req.stream:
        return StreamingResponse(
            stream_completion(request, payload, slot, resumed),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            # Belt-and-braces slot release: if the generator is never iterated
            # (setup failure before the first chunk), its finally never runs.
            # Release is idempotent, so double-running is harmless.
            background=BackgroundTask(slot.release),
        )

    try:
        start = time.time()
        result = await run_completion(
            request.app.state, payload, strip_leading_space=not resumed
        )
    finally:
        slot.release()

    content = result["content"]
    tokens_predicted = result["tokens_predicted"]
    tokens_evaluated = result["tokens_evaluated"]
    finish_reason = result["finish_reason"]
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
    request: Request, payload: dict, slot: Slot, continuation: bool = False
) -> AsyncIterator[str]:
    start = time.time()
    total = 0
    chat_id = make_chat_id()
    created = int(time.time())
    guard = LoopGuard(settings.loop_guard_repeats)
    first_token = True
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
                if first_token and token:
                    if not continuation and token.startswith(" "):
                        token = token[1:]
                    first_token = False
                total += len(token)
                finish_reason = None
                if stop:
                    finish_reason = finish_reason_from(chunk)
                elif guard.feed(token):
                    # Repetition already streamed cannot be unsent, but the
                    # generation is cut here: dropping out of the stream
                    # context closes the connection and the backend frees its
                    # slot. The client sees a clean final chunk.
                    stop = True
                    finish_reason = "stop"
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
        result = await run_completion(
            request.app.state, payload, strip_leading_space=True
        )
    finally:
        slot.release()

    summary = result["content"].strip()
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
    payload = {
        "busy": request.app.state.inference_lock.locked(),
        "model": settings.model_id,
        "context_size": settings.ctx_size,
        "max_tokens_cap": settings.max_tokens_cap,
        "auth_required": bool(settings.api_key),
    }
    # The UI needs this to keep the framing prompt alive through context
    # compaction (the summary is a system message, which would otherwise
    # suppress the default). Served only to authorized callers: /v1/status
    # itself must stay reachable pre-auth, but an operator's custom
    # BITNET_SYSTEM_PROMPT is configuration, not something every anonymous
    # visitor should read while the key gates the rest of /v1.
    if request_is_authorized(request):
        payload["default_system_prompt"] = settings.system_prompt
    return payload


@app.get("/v1/config")
async def config(request: Request):
    """The full effective configuration, for the UI's info panel.

    Every knob that shapes a reply, resolved to the value actually in force --
    so diagnosing "why did it answer like that" is reading one screen rather
    than correlating .env against defaults against what the container was
    started with. Authenticated: it exposes the system prompt and the
    deployment's tuning.
    """
    check_api_key(request)
    return {
        "model": {
            "id": settings.model_id,
            "context_size": settings.ctx_size,
            "max_tokens_cap": settings.max_tokens_cap,
        },
        "sampling": {
            "temperature": settings.temperature,
            "min_p": settings.min_p,
            "top_p": "not sent unless the caller sets it (backend default 0.95 is overridden with 1.0 = off)",
            "repeat_penalty": settings.repeat_penalty,
            "repeat_last_n": settings.repeat_last_n,
            "dry_multiplier": settings.dry_multiplier,
            "dry_base": settings.dry_base,
            "dry_allowed_length": settings.dry_allowed_length,
            "dry_penalty_last_n": -1,
        },
        "prompt": {
            "format": settings.prompt_format,
            "system_prompt": settings.system_prompt,
            "generation_prompt": (
                BITNET_GENERATION_PROMPT
                if settings.prompt_format == "bitnet"
                else GENERATION_PROMPT
            ),
            "example": build_prompt(
                [
                    Message(role="system", content="<system prompt>"),
                    Message(role="user", content="<your message>"),
                ]
            ),
        },
        "stops": {
            "default": list(DEFAULT_STOPS),
            "role_labels": (
                ROLE_STOPS.get(settings.prompt_format, []) if settings.role_stops else []
            ),
            "role_stops_enabled": settings.role_stops,
        },
        "guardrails": {
            "loop_guard_repeats": settings.loop_guard_repeats,
            "loop_guard_enabled": bool(settings.loop_guard_repeats),
            "queue_timeout": settings.queue_timeout,
            "read_timeout": settings.read_timeout,
            "max_messages": settings.max_messages,
        },
        "endpoints": {
            "mcp_enabled": settings.mcp_enabled,
            "auth_required": bool(settings.api_key),
        },
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
    # Same framing the REST path gets; a system argument from the caller wins.
    messages = with_default_system(messages, max_tokens)

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
            result = await run_completion(
                app.state, payload, strip_leading_space=True
            )
        except HTTPException as exc:
            raise ValueError(f"BitNet backend unavailable: {exc.detail}") from exc
    finally:
        slot.release()

    return {
        "content": result["content"],
        "finish_reason": result["finish_reason"],
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
