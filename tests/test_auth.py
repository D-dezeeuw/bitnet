"""Authentication coverage (SEC-1, SEC-3, SEC-4).

/v1/summarize used to skip the key check entirely, so it is asserted here
alongside the routes that were already gated.
"""

import pytest

PROTECTED = [
    ("post", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("post", "/v1/summarize", {"messages": [{"role": "user", "content": "hi"}]}),
    ("get", "/v1/models", None),
]


async def _call(client, method, path, body, **kw):
    if method == "post":
        return await client.post(path, json=body, **kw)
    return await client.get(path, **kw)


@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_rejects_when_key_missing(client, settings, method, path, body):
    settings.api_key = "s3cret"
    resp = await _call(client, method, path, body)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_accepts_bearer_token(client, settings, method, path, body):
    settings.api_key = "s3cret"
    resp = await _call(
        client, method, path, body, headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_accepts_x_api_key(client, settings, method, path, body):
    settings.api_key = "s3cret"
    resp = await _call(client, method, path, body, headers={"x-api-key": "s3cret"})
    assert resp.status_code == 200


@pytest.mark.parametrize("method,path,body", PROTECTED)
async def test_open_when_no_key_configured(client, settings, method, path, body):
    settings.api_key = None
    resp = await _call(client, method, path, body)
    assert resp.status_code == 200


async def test_wrong_key_rejected(client, settings):
    settings.api_key = "s3cret"
    resp = await client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


async def test_summarize_is_not_an_unauthenticated_inference_backdoor(client, settings):
    """The specific regression: summarize ran inference with no key at all."""
    settings.api_key = "s3cret"
    resp = await client.post(
        "/v1/summarize", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 401


class TestSessionCookie:
    """The browser UI cannot send a bearer header, so it trades the key for a
    cookie. Without this, enabling auth simply broke /inference."""

    async def test_bad_key_gets_no_session(self, client, settings):
        settings.api_key = "s3cret"
        resp = await client.post("/v1/auth", json={"api_key": "wrong"})
        assert resp.status_code == 401
        assert "bitnet_session" not in resp.cookies

    async def test_good_key_issues_httponly_cookie(self, client, settings):
        settings.api_key = "s3cret"
        resp = await client.post("/v1/auth", json={"api_key": "s3cret"})
        assert resp.status_code == 200
        set_cookie = resp.headers["set-cookie"]
        assert "bitnet_session=" in set_cookie
        assert "HttpOnly" in set_cookie
        assert "SameSite=strict" in set_cookie.replace("samesite", "SameSite")

    async def test_session_cookie_authorizes_subsequent_calls(self, client, settings):
        settings.api_key = "s3cret"
        assert (await client.get("/v1/models")).status_code == 401
        await client.post("/v1/auth", json={"api_key": "s3cret"})
        # httpx keeps the cookie on the client
        assert (await client.get("/v1/models")).status_code == 200

    async def test_forged_cookie_rejected(self, client, settings):
        settings.api_key = "s3cret"
        resp = await client.get(
            "/v1/models", headers={"Cookie": "bitnet_session=9999999999.deadbeef"}
        )
        assert resp.status_code == 401

    async def test_expired_cookie_rejected(self, client, settings):
        import app as app_module

        settings.api_key = "s3cret"
        expired = app_module.issue_session_token(now=0)  # ttl added to epoch 0
        resp = await client.get(
            "/v1/models", headers={"Cookie": f"bitnet_session={expired}"}
        )
        assert resp.status_code == 401
