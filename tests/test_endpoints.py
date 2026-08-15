"""Remaining routes: status, health, download, UI (SEC-5, RBT-3, WEB-1)."""


async def test_status_serves_the_real_context_size(client, settings):
    """The UI reads this instead of hardcoding 4096, so the two can't drift."""
    data = (await client.get("/v1/status")).json()
    assert data["context_size"] == settings.ctx_size
    assert data["max_tokens_cap"] == settings.max_tokens_cap
    assert data["model"] == settings.model_id


async def test_status_reports_whether_auth_is_required(client, settings):
    settings.api_key = "s3cret"
    assert (await client.get("/v1/status")).json()["auth_required"] is True


async def test_status_is_reachable_without_a_key(client, settings):
    """The UI needs it before it can authenticate."""
    settings.api_key = "s3cret"
    assert (await client.get("/v1/status")).status_code == 200


async def test_health_ok_when_backend_healthy(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_health_degraded_when_backend_down(client, backend):
    backend.unavailable = True
    assert (await client.get("/health")).status_code == 503


class TestDownload:
    async def test_missing_file_is_404_not_500(self, client, settings, tmp_path):
        """FileResponse on an absent path raised and returned a traceback."""
        settings.download_path = tmp_path / "nope.zip"
        assert (await client.get("/download")).status_code == 404

    async def test_present_file_is_served(self, client, settings, tmp_path):
        target = tmp_path / "download.zip"
        target.write_bytes(b"PK\x03\x04payload")
        settings.download_path = target
        resp = await client.get("/download")
        assert resp.status_code == 200
        assert resp.content == b"PK\x03\x04payload"


class TestSecurityHeaders:
    async def test_csp_is_set(self, client):
        csp = (await client.get("/v1/status")).headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp

    async def test_framing_and_sniffing_blocked(self, client):
        headers = (await client.get("/v1/status")).headers
        assert headers["x-frame-options"] == "DENY"
        assert headers["x-content-type-options"] == "nosniff"


class TestUI:
    async def test_inference_serves_the_extracted_html(self, client):
        resp = await client.get("/inference")
        assert resp.status_code == 200
        assert "BitNet" in resp.text

    async def test_ui_has_no_inline_script_so_csp_holds(self, client):
        """Inline scripts would have to be allowed by the CSP, which defeats it."""
        html = (await client.get("/inference")).text
        assert "<script>" not in html

    async def test_ui_loads_no_third_party_assets(self, client):
        """Both CDNs are vendored; the UI must render with no outbound network."""
        html = (await client.get("/inference")).text
        assert "cdn.tailwindcss.com" not in html
        assert "cdn.jsdelivr.net" not in html

    async def test_vendored_libraries_are_served_locally(self, client):
        for path in ("/static/vendor/marked.min.js", "/static/vendor/purify.min.js"):
            assert (await client.get(path)).status_code == 200


async def test_root_redirects_to_the_ui(client):
    """The bare domain used to return a JSON 404, which reads like a broken
    deployment when the UI is simply at another path."""
    r = await client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/inference"


async def test_status_serves_the_default_system_prompt(client, settings):
    """The UI needs it to keep the framing prompt alive through compaction:
    the compaction summary is a system message, which would otherwise
    suppress the default exactly when conversations get long."""
    data = (await client.get("/v1/status")).json()
    assert data["default_system_prompt"] == settings.system_prompt


async def test_system_prompt_is_not_served_unauthenticated(client, settings, monkeypatch):
    """/v1/status stays reachable pre-auth, but the operator's custom system
    prompt is configuration: with a key set, anonymous visitors get the
    runtime facts without it."""
    monkeypatch.setattr(settings, "api_key", "sekrit")
    data = (await client.get("/v1/status")).json()
    assert "default_system_prompt" not in data
    assert data["auth_required"] is True


async def test_system_prompt_is_served_to_key_holders(client, settings, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "sekrit")
    data = (
        await client.get("/v1/status", headers={"Authorization": "Bearer sekrit"})
    ).json()
    assert data["default_system_prompt"] == settings.system_prompt
