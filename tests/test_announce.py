"""Announcing new runs: waiting for the deploy to serve them, then signed webhooks to partners."""

from __future__ import annotations

import hashlib
import hmac
import json

import httpx
import pytest

from cumulus import announce
from cumulus.settings import get_settings

from tests.test_partner_api import RUN_ID, client  # noqa: F401 - fixtures
from tests.test_subseasonal import settings  # noqa: F401 - the shared settings fixture

SECRET = "whsec_test"
PARTNERS = {
    "nadmo": {"name": "NADMO", "key_sha256": "0" * 64, "webhook_url": "https://nadmo.example/hook", "webhook_secret": SECRET},
    "vra": {"name": "VRA", "key_sha256": "1" * 64},
    "old": {"name": "Former", "key_sha256": "2" * 64, "webhook_url": "https://old.example/hook", "webhook_secret": "x", "revoked": True},
}


@pytest.fixture()
def partners(client, monkeypatch):  # noqa: F811 - pytest fixture injection
    monkeypatch.setenv("CUMULUS_PARTNERS", json.dumps(PARTNERS))
    get_settings.cache_clear()
    return get_settings()


def _fake(live_after: int, hook_status: list[int]):
    """A transport whose API serves the new run after ``live_after`` polls, and a recording webhook."""
    calls = {"polls": 0, "hooks": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/runs/latest":
            calls["polls"] += 1
            run_id = RUN_ID if calls["polls"] > live_after else "ifs_unet_2026092400"
            return httpx.Response(200, json={"meta": {"run_id": run_id, "model": "ifs_unet", "official": True}})
        calls["hooks"].append(request)
        return httpx.Response(hook_status.pop(0) if hook_status else 204)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_waits_for_the_deploy_then_sends_a_signed_event(partners):
    http, calls = _fake(live_after=2, hook_status=[])
    problems = announce.announce(partners, ["ifs_unet"], api_url="https://api", client=http, sleep=lambda _s: None)
    assert problems == [] and calls["polls"] == 3
    # Only the partner with a webhook and secret, not the revoked one.
    [hook] = calls["hooks"]
    assert str(hook.url) == "https://nadmo.example/hook"
    body = hook.content
    expected = hmac.new(SECRET.encode(), hook.headers["X-Cumulus-Timestamp"].encode() + b"." + body, hashlib.sha256)
    assert hook.headers["X-Cumulus-Signature"] == f"sha256={expected.hexdigest()}"
    assert hook.headers["X-Cumulus-Event-Id"] == RUN_ID
    event = json.loads(body)
    assert event["event"] == "run.published" and event["run"]["run_id"] == RUN_ID
    assert event["links"]["summary"] == f"https://api/v1/summary?model=ifs_unet&run={RUN_ID}"


def test_retries_a_failing_webhook_and_reports_a_lost_one(partners):
    http, calls = _fake(live_after=0, hook_status=[500, 503])
    assert announce.announce(partners, ["ifs_unet"], api_url="https://api", client=http, sleep=lambda _s: None) == []
    assert len(calls["hooks"]) == 3

    http, calls = _fake(live_after=0, hook_status=[500, 500, 500])
    problems = announce.announce(partners, ["ifs_unet"], api_url="https://api", client=http, sleep=lambda _s: None)
    assert problems == ["ifs_unet: webhook to nadmo failed"]


def test_reports_a_run_that_never_goes_live(partners):
    http, calls = _fake(live_after=10**6, hook_status=[])
    problems = announce.announce(
        partners, ["ifs_unet"], api_url="https://api", client=http, timeout=0, sleep=lambda _s: None
    )
    assert len(problems) == 1 and "was not live" in problems[0] and calls["hooks"] == []


def test_reads_the_live_run_from_the_real_api(client):  # noqa: F811 - pytest fixture injection
    meta = announce.wait_until_live(client, "", "ifs_unet", RUN_ID, timeout=0, interval=0)
    assert meta["run_id"] == RUN_ID and meta["valid_from"] == "2026-09-25"
