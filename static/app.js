/* BitNet inference UI.
   Extracted from a Python string literal in app.py so it can be linted, and so
   a strict CSP (script-src 'self') can apply -- there are no inline scripts. */
'use strict';

marked.setOptions({ breaks: true, gfm: true });

const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('input');
const btn = document.getElementById('btn');
const stopBtn = document.getElementById('stop-btn');
const clearBtn = document.getElementById('clear-btn');
const ctxBarWrap = document.getElementById('ctx-bar-wrap');
const ctxBarFill = document.getElementById('ctx-bar-fill');
const ctxBarLabel = document.getElementById('ctx-bar-label');
const systemPromptEl = document.getElementById('system-prompt');
const maxTokensEl = document.getElementById('cfg-max-tokens');
const authGate = document.getElementById('auth-gate');
const authForm = document.getElementById('auth-form');
const authKey = document.getElementById('auth-key');
const authError = document.getElementById('auth-error');
const ui = document.getElementById('ui');

const KEEP_RECENT = 8;

// Served by /v1/status rather than hardcoded, so changing BITNET_CTX_SIZE can
// never leave the context bar quietly lying about the real limit.
let CTX_LIMIT = 4096;
let DEFAULT_SYSTEM_PROMPT = '';

let history = [];
let isCompacting = false;
let currentStreamTokens = 0;
let abortController = null;
let metaEl = null;
let busyPollId = null;

/* ---------- rendering ---------- */

// Model output is untrusted: it reaches innerHTML, so it goes through
// DOMPurify first. Without this, asking the model to emit an onerror handler
// is enough to run script in the page.
function renderMarkdown(el, text) {
  el.innerHTML = DOMPurify.sanitize(marked.parse(text));
}

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
    else if (m.role === 'assistant') renderMarkdown(appendEl('assistant', ''), m.content);
    else if (m.role === 'system') appendEl('compact-notice', m.content);
  }
}

function removeContinueBtn() {
  const old = document.getElementById('continue-btn');
  if (old) old.remove();
}

/* ---------- context accounting ---------- */

function getPinnedMessages() {
  const sp = systemPromptEl.value.trim();
  return sp ? [{ role: 'system', content: sp }] : [];
}

function buildMessages() {
  return [...getPinnedMessages(), ...history];
}

function estimateTokens(messages) {
  return messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 3.5) + 4, 0);
}

function updateCtxBar(extra) {
  const used = estimateTokens(buildMessages()) + (extra || 0);
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

/* ---------- compaction ---------- */

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
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const summaryText = data.summary || '[Summary unavailable]';
    // The summary rides in a system message, which makes the server treat the
    // request as "caller supplied a system prompt" and skip its default
    // framing. Re-attach that framing here (fetched from /v1/status) unless
    // the user typed their own, so the anti-rambling anchor survives
    // compaction instead of vanishing exactly when conversations get long.
    // default_system_prompt is only served to authorized callers, and
    // bootstrap may have run before the session cookie existed -- refetch
    // once, now that compaction proves we are authenticated.
    if (!DEFAULT_SYSTEM_PROMPT) {
      try {
        const sres = await fetch('/v1/status');
        const sdata = await sres.json();
        if (sdata.default_system_prompt) DEFAULT_SYSTEM_PROMPT = sdata.default_system_prompt;
      } catch { /* framing is an enhancement, not a requirement */ }
    }
    const ownSystem = systemPromptEl.value.trim();
    const framing = (!ownSystem && DEFAULT_SYSTEM_PROMPT) ? DEFAULT_SYSTEM_PROMPT + '\n\n' : '';
    history = [{ role: 'system', content: framing + 'Context summary: ' + summaryText }, ...toKeep];
    renderHistory();
    updateCtxBar();
    updateMeta((auto ? 'Auto-compacted' : 'Compacted') + ': ' + beforeCount + ' -> ' +
      history.length + ' msgs | ~' + beforeTokens + ' -> ~' + estimateTokens(history) + ' tokens');
  } catch (err) {
    appendEl('error', 'Compaction failed: ' + err.message);
  }
  isCompacting = false;
  return true;
}

/* ---------- busy state ---------- */

function setBusy(busy) {
  input.disabled = busy;
  btn.disabled = busy;
  clearBtn.disabled = busy;
  document.querySelectorAll('details input, details textarea').forEach((i) => { i.disabled = busy; });
  input.placeholder = busy ? 'Server is busy…' : 'Type a message...';
  if (!busy && document.activeElement === document.body) input.focus();
}

function setStreaming(active) {
  btn.hidden = active;
  stopBtn.hidden = !active;
  setBusy(active);
}

async function pollBusy() {
  try {
    const res = await fetch('/v1/status');
    if (!res.ok) return;
    const data = await res.json();
    if (data.busy && !abortController) {
      setBusy(true);
      if (!busyPollId) busyPollId = setInterval(pollBusy, 2000);
    } else if (!abortController) {
      setBusy(false);
      if (busyPollId) { clearInterval(busyPollId); busyPollId = null; }
    }
  } catch { /* transient - the next poll will correct it */ }
}

/* ---------- streaming ---------- */

async function streamResponse(el, existingText, isContinuation) {
  const maxTokens = parseInt(maxTokensEl.value, 10) || 256;
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
    const topPVal = parseFloat(document.getElementById('cfg-top-p').value);
    const stopRaw = document.getElementById('cfg-stop').value.trim();
    const stopSeqs = stopRaw ? stopRaw.split(',').map((s) => s.trim()).filter(Boolean).slice(0, 4) : null;
    const body = {
      messages: buildMessages(),
      stream: true,
      temperature: isNaN(tempVal) ? 0.3 : tempVal,
      max_tokens: maxTokens
    };
    // top_p only when the user set one: always sending it overlays the
    // server's min_p-first sampling with a second truncation.
    if (!isNaN(topPVal)) body.top_p = topPVal;
    if (stopSeqs && stopSeqs.length) body.stop = stopSeqs;
    // Resume the trailing assistant turn rather than opening a new one.
    if (isContinuation) body.continuation = true;

    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: abortController.signal
    });
    if (res.status === 503) {
      abortController = null;
      appendEl('error', 'Server is busy — please wait and try again.');
      setStreaming(false);
      pollBusy();
      return;
    }
    if (res.status === 401) {
      abortController = null;
      setStreaming(false);
      showAuthGate('Session expired. Enter the API key again.');
      return;
    }
    if (!res.ok) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
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
          renderMarkdown(el, fullText);
          log.scrollTop = log.scrollHeight;
          updateCtxBar(currentStreamTokens);
        } catch { /* partial frame - wait for the rest */ }
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

  if (fullText) {
    const lastMsg = history[history.length - 1];
    if (lastMsg && lastMsg.role === 'assistant') lastMsg.content = fullText;
    else history.push({ role: 'assistant', content: fullText });
  }
  updateCtxBar();
  const ms = Math.round(performance.now() - start);
  const tps = tokenCount > 0 ? (tokenCount / (ms / 1000)).toFixed(1) : '0';
  updateMeta(tokenCount + ' tokens | ' + ms + 'ms | ' + tps + ' t/s | ' +
    history.length + ' msgs | stop: ' + (lastFinishReason || 'unknown'));

  if (lastFinishReason === 'length' || aborted) {
    const contBtn = document.createElement('button');
    contBtn.id = 'continue-btn';
    contBtn.className = 'continue-btn';
    contBtn.textContent = aborted ? 'Continue (stopped by user)...' : 'Continue generating...';
    contBtn.addEventListener('click', async () => {
      contBtn.remove();
      await streamResponse(el, fullText, true);
      input.focus();
    });
    log.appendChild(contBtn);
    log.scrollTop = log.scrollHeight;
  }
}

/* ---------- auth ---------- */

function showAuthGate(message) {
  ui.hidden = true;
  authGate.hidden = false;
  if (message) { authError.textContent = message; authError.hidden = false; }
  authKey.focus();
}

function showUI() {
  authGate.hidden = true;
  ui.hidden = false;
  updateCtxBar();
  pollBusy();
  input.focus();
}

authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  authError.hidden = true;
  try {
    const res = await fetch('/v1/auth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: authKey.value })
    });
    if (!res.ok) {
      authError.textContent = 'That key was not accepted.';
      authError.hidden = false;
      return;
    }
    authKey.value = '';
    showUI();
  } catch (err) {
    authError.textContent = 'Could not reach the server: ' + err.message;
    authError.hidden = false;
  }
});

/* ---------- wiring ---------- */

ctxBarWrap.addEventListener('click', () => { if (!abortController) compactHistory(false); });

clearBtn.addEventListener('click', () => {
  history = [];
  metaEl = null;
  log.innerHTML = '';
  updateCtxBar();
  updateMeta('Conversation cleared');
});

stopBtn.addEventListener('click', () => { if (abortController) abortController.abort(); });

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
  await streamResponse(appendEl('assistant', ''), '', false);
  input.focus();
});

async function bootstrap() {
  try {
    const res = await fetch('/v1/status');
    const data = await res.json();
    if (data.context_size) CTX_LIMIT = data.context_size;
    if (data.max_tokens_cap) maxTokensEl.max = String(data.max_tokens_cap);
    if (data.default_system_prompt) DEFAULT_SYSTEM_PROMPT = data.default_system_prompt;
    if (data.auth_required) {
      // A valid session cookie makes an authenticated route succeed.
      const probe = await fetch('/v1/models');
      if (probe.status === 401) { showAuthGate(); return; }
    }
  } catch { /* fall through to the UI; requests will surface the failure */ }
  showUI();
}

setInterval(pollBusy, 5000);
bootstrap();
