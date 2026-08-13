import asyncio
import json
import os
import sys
import time
import uuid
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# Configure logging to stdout so Docker/Portainer can capture it
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bitnet-api")

LLAMA_SERVER_URL = f"http://127.0.0.1:{os.environ.get('LLAMA_SERVER_PORT', '8080')}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Connecting to llama-server at {LLAMA_SERVER_URL}")
    app.state.client = httpx.AsyncClient(
        base_url=LLAMA_SERVER_URL,
        timeout=120.0,
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="BitNet API", lifespan=lifespan, docs_url=None, redoc_url=None)

# Server-wide inference lock — only one request at a time
inference_lock = asyncio.Lock()

# Optional API key — set BITNET_API_KEY env var to enable
API_KEY = os.environ.get("BITNET_API_KEY")
MAX_PROMPT_CHARS = 16000  # ~4k tokens worth of input
MAX_TOKENS_CAP = 4096


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


def check_api_key(request: Request):
    """Check API key for /v1/ endpoints. Skips if BITNET_API_KEY is not set."""
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
    else:
        token = request.headers.get("x-api-key", "").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "bitnet-b1.58-2B-4T"
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[list] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


MODEL_ID = os.environ.get("MODEL_ID", "bitnet-b1.58-2B-4T")


def make_chat_id():
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def build_prompt(messages: List[Message]) -> str:
    prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
    prompt += "\nassistant:"
    return prompt


@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest, request: Request):
    check_api_key(request)
    # Input validation
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail=f"Prompt too long ({total_chars} chars, max {MAX_PROMPT_CHARS})")
    if len(req.messages) > 200:
        raise HTTPException(status_code=400, detail="Too many messages (max 200)")
    req.max_tokens = min(req.max_tokens, MAX_TOKENS_CAP)
    if inference_lock.locked():
        raise HTTPException(status_code=503, detail="Server is busy processing another request. Please wait.")

    prompt = build_prompt(req.messages)
    stop_seqs = ["<|eot_id|>", "</s>", "user:", "User:"]
    if req.stop:
        stop_seqs = req.stop + stop_seqs
    payload = {
        "prompt": prompt,
        "n_predict": req.max_tokens,
        "temperature": req.temperature,
        "stream": req.stream,
        "stop": stop_seqs,
    }
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    logger.info(f"Chat request: {len(req.messages)} message(s), max_tokens={req.max_tokens}, stream={req.stream}")

    if req.stream:
        # Lock is held inside the generator so it spans the full streaming duration
        return StreamingResponse(
            locked_stream(request, payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with inference_lock:
        start = time.time()
        try:
            resp = await request.app.state.client.post("/completion", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"llama-server returned {e.response.status_code}: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            logger.error(f"llama-server unavailable: {e}")
            raise HTTPException(status_code=503, detail=f"llama-server unavailable: {e}")
        data = resp.json()
        duration_ms = int((time.time() - start) * 1000)
        content = data.get("content", "")
        tokens_predicted = data.get("tokens_predicted", 0)
        tokens_evaluated = data.get("tokens_evaluated", 0)
        logger.info(f"Chat response: {len(content)} chars in {duration_ms}ms")
        return {
            "id": make_chat_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": tokens_evaluated,
                "completion_tokens": tokens_predicted,
                "total_tokens": tokens_evaluated + tokens_predicted,
            },
        }


async def locked_stream(request: Request, payload: dict):
    """Wraps stream_response and holds the inference lock for the full duration."""
    async with inference_lock:
        async for chunk in stream_response(request, payload):
            yield chunk


async def stream_response(request: Request, payload: dict):
    start = time.time()
    total_content = ""
    chat_id = make_chat_id()
    created = int(time.time())
    try:
        async with request.app.state.client.stream("POST", "/completion", json=payload) as resp:
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
                total_content += token
                finish_reason = None
                if stop:
                    if chunk.get("stopped_limit", False):
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                sse = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token} if not stop else {},
                        "finish_reason": finish_reason,
                    }],
                }
                yield f"data: {json.dumps(sse)}\n\n"
                if stop:
                    break
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    duration_ms = int((time.time() - start) * 1000)
    logger.info(f"Stream response: {len(total_content)} chars in {duration_ms}ms")
    yield "data: [DONE]\n\n"


@app.post("/v1/summarize")
async def summarize_context(req: ChatRequest, request: Request):
    """Summarize a list of messages into a concise context summary."""
    conversation = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    prompt = (
        "Below is a conversation. Summarize the key facts, decisions, user preferences, "
        "and important context in a concise paragraph. Preserve any specific names, numbers, "
        "or technical details. Do not add commentary.\n\n"
        f"{conversation}\n\n"
        "Summary:"
    )
    payload = {
        "prompt": prompt,
        "n_predict": 200,
        "temperature": 0.3,
        "stop": ["<|eot_id|>", "</s>", "\n\n"],
    }
    logger.info(f"Summarize request: {len(req.messages)} messages")
    try:
        resp = await request.app.state.client.post("/completion", json=payload)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"llama-server unavailable: {e}")
    data = resp.json()
    summary = data.get("content", "").strip()
    logger.info(f"Summary generated: {len(summary)} chars")
    return {"summary": summary}


@app.get("/inference", response_class=HTMLResponse)
def inference_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BitNet Inference</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: { accent: '#7cb3ff', surface: '#1a1a1a', bg: '#0f0f0f' }
    }
  }
}
</script>
<style>
  #log { font-family: 'SF Mono', 'Fira Code', monospace; }
  .user { color: #7cb3ff; }
  .assistant { color: #a8e6a1; }
  .assistant p { margin: 0.3em 0; }
  .assistant pre { background: #111; padding: 0.5em; border-radius: 4px; overflow-x: auto; margin: 0.4em 0; }
  .assistant code { background: #111; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }
  .assistant pre code { background: none; padding: 0; }
  .assistant ul, .assistant ol { margin: 0.3em 0 0.3em 1.5em; }
  .assistant h1, .assistant h2, .assistant h3 { margin: 0.5em 0 0.2em; color: #7cb3ff; }
  .assistant blockquote { border-left: 3px solid #555; padding-left: 0.75em; color: #aaa; margin: 0.3em 0; }
  .assistant table { border-collapse: collapse; margin: 0.4em 0; }
  .assistant th, .assistant td { border: 1px solid #444; padding: 0.3em 0.6em; }
  .meta { color: #888; font-size: 0.75rem; }
  .error { color: #ff6b6b; }
  .compact-notice { color: #d4a843; font-size: 0.75rem; font-style: italic; border-top: 1px dashed #444; border-bottom: 1px dashed #444; padding: 0.3em 0; margin: 0.3em 0; }
  .continue-btn { display: inline-block; margin: 0.3em 0; padding: 0.3rem 0.8rem; border-radius: 6px; border: 1px solid #d4a843; background: transparent; color: #d4a843; font-size: 0.75rem; cursor: pointer; }
  .continue-btn:hover { background: #2a2200; }
  #ctx-bar-wrap:hover #ctx-bar-bg { border-color: #7cb3ff; }
  #ctx-bar-wrap:hover #ctx-bar-label::after { content: ' (click to compact)'; color: #888; }
  details summary { list-style: none; }
  details summary::-webkit-details-marker { display: none; }
  details[open] .chevron { transform: rotate(180deg); }
</style>
</head>
<body class="bg-bg text-gray-200 h-screen flex flex-col items-center p-6 font-sans">

<h1 class="text-2xl font-bold text-accent mb-3">BitNet b1.58 2B-4T</h1>
<a class="inline-block mb-4 px-3 py-1.5 rounded-md bg-gray-800 text-green-300 text-sm hover:bg-gray-700 transition" href="/download">Download</a>

<details class="w-full max-w-[720px] mb-4">
  <summary class="flex items-center justify-between cursor-pointer select-none rounded-lg bg-surface border border-gray-700 px-4 py-2.5 hover:border-accent transition">
    <span class="text-sm font-medium text-gray-400">Configuration</span>
    <svg class="chevron w-4 h-4 text-gray-500 transition-transform duration-200" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
  </summary>
  <div class="mt-2 p-4 rounded-lg bg-surface border border-gray-700 space-y-4">
    <div>
      <label class="block text-xs text-gray-500 uppercase tracking-wide mb-1">System Prompt <span class="normal-case text-gray-600">(pinned — survives compaction)</span></label>
      <textarea id="system-prompt" rows="2" placeholder="e.g. You are a helpful assistant that responds concisely." class="w-full px-3 py-2 rounded-md bg-bg border border-gray-700 text-gray-300 text-sm resize-y min-h-[2rem] max-h-24 focus:border-accent focus:text-gray-200 outline-none transition"></textarea>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <div class="flex flex-col gap-1">
        <label class="text-xs text-gray-500 uppercase tracking-wide">Temperature</label>
        <input type="number" id="cfg-temp" value="0.7" min="0" max="2" step="0.1" class="px-2 py-1.5 rounded-md bg-bg border border-gray-700 text-gray-200 text-sm outline-none focus:border-accent transition" />
        <span class="text-[10px] text-gray-600">Randomness</span>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-gray-500 uppercase tracking-wide">Top P</label>
        <input type="number" id="cfg-top-p" value="0.9" min="0" max="1" step="0.05" class="px-2 py-1.5 rounded-md bg-bg border border-gray-700 text-gray-200 text-sm outline-none focus:border-accent transition" />
        <span class="text-[10px] text-gray-600">Lower = more focused</span>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-gray-500 uppercase tracking-wide">Max Tokens</label>
        <input type="number" id="cfg-max-tokens" value="1024" min="1" max="4096" step="1" class="px-2 py-1.5 rounded-md bg-bg border border-gray-700 text-gray-200 text-sm outline-none focus:border-accent transition" />
        <span class="text-[10px] text-gray-600">Max response length</span>
      </div>
      <div class="flex flex-col gap-1">
        <label class="text-xs text-gray-500 uppercase tracking-wide">Stop Sequences</label>
        <input type="text" id="cfg-stop" value="" placeholder="e.g. \\n, END" class="px-2 py-1.5 rounded-md bg-bg border border-gray-700 text-gray-200 text-sm outline-none focus:border-accent transition" />
        <span class="text-[10px] text-gray-600">Comma-separated</span>
      </div>
    </div>
  </div>
</details>

<div id="log" class="w-full max-w-[720px] flex-1 bg-surface border border-gray-700 rounded-lg p-4 overflow-y-auto text-sm leading-relaxed mb-3 whitespace-pre-wrap"></div>

<div id="ctx-bar-wrap" title="Click to compact context" class="w-full max-w-[720px] mb-2 cursor-pointer relative">
  <div id="ctx-bar-bg" class="w-full h-5 bg-surface border border-gray-700 rounded-full overflow-hidden">
    <div id="ctx-bar-fill" class="h-full rounded-full transition-all duration-150"></div>
  </div>
  <div id="ctx-bar-label" class="absolute inset-0 h-5 flex items-center justify-center text-[10px] text-gray-400 pointer-events-none" style="text-shadow:0 0 3px #000">0 / 4096 tokens</div>
</div>

<form id="form" class="w-full max-w-[720px] flex gap-2">
  <input type="text" id="input" placeholder="Type a message..." autocomplete="off" autofocus class="flex-1 px-4 py-3 rounded-lg bg-surface border border-gray-700 text-gray-200 outline-none focus:border-accent disabled:opacity-40 disabled:cursor-not-allowed transition" />
  <button type="submit" id="btn" class="px-5 py-3 rounded-lg bg-accent text-bg font-semibold hover:bg-blue-400 disabled:opacity-50 disabled:cursor-not-allowed transition">Send</button>
  <button type="button" id="stop-btn" style="display:none" class="px-4 py-3 rounded-lg bg-red-600 text-white font-semibold hover:bg-red-700 transition">Stop</button>
  <button type="button" id="clear-btn" title="Clear conversation history" class="px-4 py-3 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition">Clear</button>
</form>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
marked.setOptions({ breaks: true, gfm: true });
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const btn = document.getElementById('btn');
const clearBtn = document.getElementById('clear-btn');
const ctxBarWrap = document.getElementById('ctx-bar-wrap');
const ctxBarFill = document.getElementById('ctx-bar-fill');
const ctxBarLabel = document.getElementById('ctx-bar-label');
const CTX_LIMIT = 4096;
const KEEP_RECENT = 8;
let history = [];
let isCompacting = false;
const systemPromptEl = document.getElementById('system-prompt');

function getPinnedMessages() {
  const sp = systemPromptEl.value.trim();
  return sp ? [{ role: 'system', content: sp }] : [];
}

function buildMessages() {
  return [...getPinnedMessages(), ...history];
}
let currentStreamTokens = 0;
let abortController = null;

function estimateTokens(messages) {
  return messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 3.5) + 4, 0);
}

function updateCtxBar(extra) {
  extra = extra || 0;
  const used = estimateTokens(buildMessages()) + extra;
  const pct = Math.min(100, (used / CTX_LIMIT) * 100);
  ctxBarFill.style.width = pct + '%';
  let color;
  if (pct < 50) color = '#4caf50';
  else if (pct < 75) color = '#d4a843';
  else if (pct < 90) color = '#e67e22';
  else color = '#e74c3c';
  ctxBarFill.style.backgroundColor = color;
  ctxBarLabel.textContent = '~' + used + ' / ' + CTX_LIMIT + ' tokens (' + Math.round(pct) + '%)';
}

let metaEl = null;

function appendEl(cls, text) {
  const el = document.createElement('div');
  el.className = cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function updateMeta(text) {
  if (!metaEl) {
    metaEl = document.createElement('div');
    metaEl.className = 'meta';
    log.appendChild(metaEl);
  }
  // Ensure meta is always last in log (before continue btn)
  const contBtn = document.getElementById('continue-btn');
  if (contBtn) log.insertBefore(metaEl, contBtn);
  else log.appendChild(metaEl);
  metaEl.textContent = text;
  log.scrollTop = log.scrollHeight;
}

function renderHistory() {
  log.innerHTML = '';
  metaEl = null;
  for (const m of history) {
    if (m.role === 'user') appendEl('user', '> ' + m.content);
    else if (m.role === 'assistant') { const el = appendEl('assistant', ''); el.innerHTML = marked.parse(m.content); }
    else if (m.role === 'system') appendEl('compact-notice', m.content);
  }
}

async function compactHistory(auto) {
  if (history.length <= KEEP_RECENT + 1) {
    if (!auto) updateMeta('Nothing to compact (' + history.length + ' messages)');
    return false;
  }
  if (isCompacting) return false;
  isCompacting = true;
  const beforeCount = history.length;
  const beforeTokens = estimateTokens(history);
  const toSummarize = history.slice(0, -KEEP_RECENT);
  const toKeep = history.slice(-KEEP_RECENT);
  if (!auto) updateMeta('Summarizing ' + toSummarize.length + ' older messages...');
  try {
    const res = await fetch('/v1/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: toSummarize })
    });
    if (!res.ok) throw new Error('Summarize failed: HTTP ' + res.status);
    const data = await res.json();
    const summaryText = data.summary || '[Summary unavailable]';
    const summaryMsg = { role: 'system', content: 'Context summary: ' + summaryText };
    history = [summaryMsg, ...toKeep];
    const afterTokens = estimateTokens(history);
    renderHistory();
    updateCtxBar();
    updateMeta((auto ? 'Auto-compacted' : 'Compacted') + ': ' + beforeCount + ' -> ' + history.length + ' msgs | ~' + beforeTokens + ' -> ~' + afterTokens + ' tokens');
  } catch (err) {
    appendEl('error', 'Compaction failed: ' + err.message);
  }
  isCompacting = false;
  return true;
}

updateCtxBar();
ctxBarWrap.addEventListener('click', () => { if (!abortController) compactHistory(false); });

clearBtn.addEventListener('click', () => {
  history = [];
  metaEl = null;
  log.innerHTML = '';
  updateCtxBar();
  updateMeta('Conversation cleared');
});

function removeContinueBtn() {
  const old = document.getElementById('continue-btn');
  if (old) old.remove();
}

const stopBtn = document.getElementById('stop-btn');

let busyPollId = null;
function setStreaming(active) {
  btn.style.display = active ? 'none' : '';
  stopBtn.style.display = active ? '' : 'none';
  setBusy(active);
}
function setBusy(busy) {
  input.disabled = busy;
  btn.disabled = busy;
  clearBtn.disabled = busy;
  document.querySelectorAll('details input, details textarea').forEach(i => i.disabled = busy);
  input.placeholder = busy ? 'Server is busy\u2026' : 'Type a message...';
  if (!busy && document.activeElement === document.body) input.focus();
}
async function pollBusy() {
  try {
    const res = await fetch('/v1/status');
    const data = await res.json();
    if (data.busy && !abortController) {
      setBusy(true);
      if (!busyPollId) busyPollId = setInterval(pollBusy, 2000);
    } else if (!abortController) {
      setBusy(false);
      if (busyPollId) { clearInterval(busyPollId); busyPollId = null; }
    }
  } catch {}
}
pollBusy();
setInterval(pollBusy, 5000);

stopBtn.addEventListener('click', () => {
  if (abortController) abortController.abort();
});

async function streamResponse(el, existingText) {
  const maxTokens = parseInt(document.getElementById('cfg-max-tokens').value) || 1024;
  if (estimateTokens(buildMessages()) + maxTokens > CTX_LIMIT && history.length > KEEP_RECENT + 1) {
    await compactHistory(true);
  }
  abortController = new AbortController();
  setStreaming(true);
  const start = performance.now();
  let fullText = existingText || '';
  let tokenCount = 0;
  let lastFinishReason = null;
  let aborted = false;
  currentStreamTokens = 0;
  try {
    const tempVal = parseFloat(document.getElementById('cfg-temp').value);
    const temp = isNaN(tempVal) ? 0.7 : tempVal;
    const topPVal = parseFloat(document.getElementById('cfg-top-p').value);
    const topP = isNaN(topPVal) ? 0.9 : topPVal;
    const stopRaw = document.getElementById('cfg-stop').value.trim();
    const stopSeqs = stopRaw ? stopRaw.split(',').map(s => s.trim()).filter(Boolean) : null;
    const body = { messages: buildMessages(), stream: true, temperature: temp, top_p: topP, max_tokens: maxTokens };
    if (stopSeqs) body.stop = stopSeqs;
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: abortController.signal
    });
    if (res.status === 503) { abortController = null; appendEl('error', 'Server is busy — please wait and try again.'); setStreaming(false); pollBusy(); return; }
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') continue;
        try {
          const data = JSON.parse(raw);
          if (data.error) { appendEl('error', 'Error: ' + data.error); continue; }
          const fr = data.choices?.[0]?.finish_reason;
          if (fr) lastFinishReason = fr;
          const token = data.choices?.[0]?.delta?.content || '';
          if (token) { tokenCount++; currentStreamTokens++; }
          fullText += token;
          el.innerHTML = marked.parse(fullText);
          log.scrollTop = log.scrollHeight;
          updateCtxBar(currentStreamTokens);
        } catch {}
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      aborted = true;
      lastFinishReason = 'aborted';
    } else {
      appendEl('error', 'Error: ' + err.message);
    }
  }
  abortController = null;
  setStreaming(false);
  // Save whatever was generated
  if (fullText) {
    const lastMsg = history[history.length - 1];
    if (lastMsg && lastMsg.role === 'assistant') {
      lastMsg.content = fullText;
    } else {
      history.push({ role: 'assistant', content: fullText });
    }
  }
  updateCtxBar();
  const ms = Math.round(performance.now() - start);
  const tps = tokenCount > 0 ? (tokenCount / (ms / 1000)).toFixed(1) : '0';
  const reasonLabel = lastFinishReason || 'unknown';
  updateMeta(tokenCount + ' tokens | ' + ms + 'ms | ' + tps + ' t/s | ' + history.length + ' msgs | stop: ' + reasonLabel);
  // If truncated or aborted, show continue button
  if (lastFinishReason === 'length' || aborted) {
    const contBtn = document.createElement('button');
    contBtn.id = 'continue-btn';
    contBtn.className = 'continue-btn';
    contBtn.textContent = aborted ? 'Continue (stopped by user)...' : 'Continue generating...';
    contBtn.addEventListener('click', async () => {
      contBtn.remove();
      await streamResponse(el, fullText);
      input.focus();
    });
    log.appendChild(contBtn);
    log.scrollTop = log.scrollHeight;
  }
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (abortController) return;
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  removeContinueBtn();
  appendEl('user', '> ' + msg);
  history.push({ role: 'user', content: msg });
  updateCtxBar();
  const el = appendEl('assistant', '');
  await streamResponse(el, '');
  input.focus();
});
</script>
</body>
</html>"""


@app.get("/download")
async def download():
    return FileResponse("/app/static/download.zip", filename="download.zip", media_type="application/zip")


@app.get("/v1/models")
def list_models(request: Request):
    check_api_key(request)
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 1700000000,
            "owned_by": "microsoft",
        }],
    }


@app.get("/v1/status")
async def status():
    return {"busy": inference_lock.locked()}


@app.get("/health")
async def health(request: Request):
    try:
        resp = await request.app.state.client.get("/health")
        return {"status": "ok", "backend": resp.json()}
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail={"status": "degraded", "backend": "unavailable"})
