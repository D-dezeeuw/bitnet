"""MCP server exposing this deployment as tools for MCP clients.

Claude Desktop and similar clients speak MCP, not the OpenAI chat API, so they
cannot consume /v1/chat/completions directly -- a "connector" is an MCP server
that gives the client *tools*, not an endpoint that serves a model. This module
wraps the same inference path behind an MCP endpoint so those clients can call
this model as a tool.

Auth is a query parameter as well as a header. That is unusual and deliberate:
the client's connector dialog accepts a single URL and offers no way to attach
an Authorization header, so a URL like

    https://host/mcp?key=<key>

is the only shape that fits. The tradeoff is real -- see QueryKeyAuth below.
"""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

logger = logging.getLogger("bitnet.mcp")

# Advertised to clients on connect. Distinct from the API's model_id, which
# names the model rather than this server.
SERVER_NAME = "bitnet"

GenerateFn = Callable[[str, str | None, int, float], Awaitable[dict[str, Any]]]
StatusFn = Callable[[], Awaitable[dict[str, Any]]]


def build_mcp_server(generate: GenerateFn, status: StatusFn) -> MCPServer:
    """Build the MCP server.

    The two callables are injected rather than imported so this module stays
    independent of app.py and can be tested without starting the API.
    """
    server = MCPServer(
        name=SERVER_NAME,
        title="BitNet",
        instructions=(
            "A locally hosted BitNet b1.58 2B model. It runs on CPU and handles "
            "one request at a time, so replies take seconds to minutes and "
            "concurrent calls are rejected rather than queued indefinitely. "
            "Best for short, self-contained prompts. Being a 2B model, it is "
            "considerably less capable than a frontier model -- prefer it when "
            "the point is to use this specific local model, not when you simply "
            "need the best answer."
        ),
    )

    @server.tool(
        name="bitnet_chat",
        title="Ask BitNet",
        description=(
            "Send a prompt to the locally hosted BitNet 2B model and return its "
            "reply. Runs on CPU with a single inference slot: a short prompt "
            "typically takes seconds, a long one minutes. If another request "
            "holds the slot this fails immediately with a busy error rather "
            "than waiting -- retry shortly."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    )
    async def bitnet_chat(
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Generate a reply from the local BitNet model.

        Args:
            prompt: The user message to send.
            system: Optional system instruction placed before the prompt.
            max_tokens: Maximum tokens to generate. Capped by the server's
                context size; asking for more than fits is an error, not a
                silent truncation.
            temperature: Sampling temperature, 0.0 to 2.0. Lower is more
                deterministic.
        """
        result = await generate(prompt, system, max_tokens, temperature)
        content = result.get("content", "")
        if result.get("finish_reason") == "length":
            content += (
                f"\n\n[truncated at the {max_tokens}-token limit; "
                "call again with a higher max_tokens for the rest]"
            )
        return content

    @server.tool(
        name="bitnet_status",
        title="BitNet status",
        description=(
            "Report whether the BitNet backend is reachable, which model is "
            "loaded, and the context window. Use this to check availability "
            "before a long generation, or to diagnose a failing bitnet_chat."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    )
    async def bitnet_status() -> dict[str, Any]:
        """Return backend health, model id, and context size."""
        return await status()

    return server


class MCPDispatch:
    """ASGI middleware serving the MCP app at /mcp, gated by an API key.

    Middleware rather than app.mount() for two reasons. Starlette's Mount
    compiles to "/mcp/{path:path}", so a request to bare /mcp does not match it
    and the router answers 307 to /mcp/ instead -- a trailing slash the client's
    connector URL would then have to carry, and a redirect on POST that not
    every client follows. Middleware also runs ahead of routing, so the key is
    checked before anything else touches the request.

    On the query parameter: a key in a URL is materially weaker than one in a
    header. It lands in the reverse proxy's access log, in shell history, and in
    any error report that echoes the request line, and nothing that redacts
    Authorization will redact it. It is supported because the connector UI
    accepts only a URL, so the alternative is not "use a header" but "cannot
    connect at all". Treat the resulting URL as the secret it contains: rotate
    the key if it leaks, and prefer the header where the client allows one.
    """

    #: Query parameter carrying the key.
    PARAM = "key"

    #: Paths served. Both spellings, so neither client is wrong.
    PATHS = frozenset({"/mcp", "/mcp/"})

    def __init__(
        self,
        app: Any,
        get_target: Callable[[], Any],
        get_api_key: Callable[[], str | None],
    ) -> None:
        self.app = app
        # Getters, not values. The target is rebuilt on every lifespan, and the
        # key is read from settings at request time so that changing it does not
        # require re-importing the module -- binding either at construction
        # pinned a stale value.
        self.get_target = get_target
        self.get_api_key = get_api_key

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("path") not in self.PATHS:
            await self.app(scope, receive, send)
            return

        api_key = self.get_api_key()

        # An unauthenticated MCP endpoint on a public host would let anyone
        # spend the single inference slot, so this refuses to serve at all
        # rather than serving openly. /v1 routes stay open without a key for
        # backward compatibility; a new endpoint has no such history.
        if not api_key:
            await self._reject(
                send,
                503,
                "The MCP endpoint requires BITNET_API_KEY to be set on the "
                "server. It is disabled while no key is configured.",
            )
            return

        if not self._authorized(scope, api_key):
            await self._reject(
                send,
                401,
                "Missing or invalid API key. Supply it as ?key=<key> in the "
                "URL, or as an Authorization: Bearer <key> header.",
            )
            return

        target = self.get_target()
        if target is None:  # pragma: no cover - only outside a running lifespan
            await self._reject(
                send, 503, "The MCP endpoint is not running yet; retry shortly."
            )
            return

        # The MCP app is built with streamable_http_path="/", so it expects the
        # request at its own root regardless of which spelling arrived here.
        await target({**scope, "path": "/", "raw_path": b"/"}, receive, send)

    def _authorized(self, scope: dict, api_key: str) -> bool:
        for presented in self._presented(scope):
            # compare_digest rather than == so a wrong key cannot be recovered
            # by timing the comparison.
            if presented and hmac.compare_digest(presented, api_key):
                return True
        return False

    def _presented(self, scope: dict) -> list[str]:
        found: list[str] = []

        raw_qs = scope.get("query_string", b"").decode("latin-1")
        for pair in raw_qs.split("&"):
            name, sep, value = pair.partition("=")
            if sep and name == self.PARAM:
                from urllib.parse import unquote_plus

                found.append(unquote_plus(value))

        for raw_name, raw_value in scope.get("headers", []):
            name = raw_name.decode("latin-1").lower()
            value = raw_value.decode("latin-1")
            if name == "authorization" and value.startswith("Bearer "):
                found.append(value[7:].strip())
            elif name == "x-api-key":
                found.append(value.strip())

        return found

    async def _reject(self, send: Any, status: int, message: str) -> None:
        # A JSON-RPC error shape, since the caller is an MCP client. id is null
        # because the request body is not parsed before rejecting it.
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32001, "message": message},
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
