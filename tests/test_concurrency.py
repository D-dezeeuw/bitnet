"""Admission control (RBT-1).

The backend serves one request at a time. The old code checked
`inference_lock.locked()` and acquired it later, which left a window where two
callers both passed the check -- and for streaming the lock was only taken once
the generator started producing, so the loser silently blocked on a response
that had already returned 200.
"""

import asyncio


def body(**kw):
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    payload.update(kw)
    return payload


async def test_second_concurrent_request_gets_503(client, backend):
    backend.delay = 0.5
    first, second = await asyncio.gather(
        client.post("/v1/chat/completions", json=body()),
        _delayed(client.post("/v1/chat/completions", json=body())),
    )
    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 503]


async def _delayed(coro, delay=0.05):
    await asyncio.sleep(delay)
    return await coro


async def test_streaming_loser_gets_a_real_503_not_a_silent_stall(client, backend):
    """The important part is that the rejection arrives as an HTTP status.
    Previously both requests returned 200 and the second simply hung."""
    backend.delay = 0.5

    async def stream_once():
        async with client.stream(
            "POST", "/v1/chat/completions", json=body(stream=True)
        ) as resp:
            await resp.aread()
            return resp.status_code

    async def stream_later():
        await asyncio.sleep(0.05)
        return await stream_once()

    first, second = await asyncio.gather(stream_once(), stream_later())
    assert sorted([first, second]) == [200, 503]


async def test_summarize_takes_the_same_slot(client, backend):
    """Summarize used to bypass the lock entirely and contend with live streams."""
    backend.delay = 0.5
    chat, summary = await asyncio.gather(
        client.post("/v1/chat/completions", json=body()),
        _delayed(
            client.post(
                "/v1/summarize", json={"messages": [{"role": "user", "content": "hi"}]}
            )
        ),
    )
    assert sorted([chat.status_code, summary.status_code]) == [200, 503]


async def test_status_reports_busy_during_summarize(client, backend):
    """/v1/status reported busy=false while summarize was running."""
    backend.delay = 0.4

    async def check_busy():
        await asyncio.sleep(0.1)
        return (await client.get("/v1/status")).json()["busy"]

    _, busy = await asyncio.gather(
        client.post(
            "/v1/summarize", json={"messages": [{"role": "user", "content": "hi"}]}
        ),
        check_busy(),
    )
    assert busy is True


async def test_slot_is_free_again_afterwards(client, backend):
    backend.delay = 0.1
    await client.post("/v1/chat/completions", json=body())
    assert (await client.get("/v1/status")).json()["busy"] is False


async def test_requests_are_serialised_not_dropped_when_queue_allows(client, backend, settings):
    """With a queue window wider than the request, the second caller waits its
    turn rather than being rejected."""
    settings.queue_timeout = 5.0
    backend.delay = 0.1
    results = await asyncio.gather(
        *[client.post("/v1/chat/completions", json=body()) for _ in range(3)]
    )
    assert [r.status_code for r in results] == [200, 200, 200]
