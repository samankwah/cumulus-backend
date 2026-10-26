"""Tell partners about new runs once they are live: ``python -m cumulus.announce --model ifs_unet ...``.

Run by the ingest workflow after it pushes new runs. For each model it waits until the deployed open
API serves the run named in that model's local ``active.json`` (the push redeploys Vercel), then
POSTs a signed ``run.published`` event to every partner in ``CUMULUS_PARTNERS`` with a
``webhook_url`` and ``webhook_secret``.

Partners verify an event by computing HMAC-SHA256, keyed with their webhook secret, over
``"<X-Cumulus-Timestamp>.<raw body>"`` and comparing it with ``X-Cumulus-Signature``
(``sha256=<hex>``). ``X-Cumulus-Event-Id`` is the run id, so a partner can drop repeats.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import hmac
import json
import logging
import time
from typing import Any

import httpx

from cumulus.settings import PartnerConfig, Settings, get_settings
from cumulus.subseasonal.ingest import ACTIVE_FILE_NAME, for_model, source_dir

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://cumulus-backend.vercel.app"
EVENT = "run.published"
WEBHOOK_ATTEMPTS = 3


def active_run_id(settings: Settings) -> str | None:
    """The run the local ``active.json`` promotes, i.e. the one this deploy should serve."""
    path = source_dir(settings) / ACTIVE_FILE_NAME
    return json.loads(path.read_text(encoding="utf-8")).get("run_id") if path.exists() else None


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """The ``X-Cumulus-Signature`` value for a body sent at ``timestamp``."""
    digest = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256)
    return f"sha256={digest.hexdigest()}"


def wait_until_live(
    client: httpx.Client,
    api_url: str,
    model: str,
    run_id: str,
    *,
    timeout: float,
    interval: float,
    sleep=time.sleep,
) -> dict[str, Any]:
    """Polls ``/v1/runs/latest`` until it names ``run_id``; returns that run's ``meta``."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            # A fresh query string each time, so the CDN's 5-minute copy of "latest" is not served.
            response = client.get(
                f"{api_url}/v1/runs/latest", params={"model": model, "fresh": str(time.time_ns())}
            )
            if response.status_code == 200:
                meta = response.json()["meta"]
                if meta.get("run_id") == run_id:
                    return meta
                logger.info("announce_waiting model=%s live=%s want=%s", model, meta.get("run_id"), run_id)
            else:
                logger.info("announce_waiting model=%s status=%s", model, response.status_code)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.info("announce_waiting model=%s error=%s", model, exc)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{model}: {run_id} was not live at {api_url} after {timeout:.0f}s")
        sleep(interval)


def event_for(meta: dict[str, Any], api_url: str) -> dict[str, Any]:
    model, run_id = meta["model"], meta["run_id"]
    return {
        "event": EVENT,
        "id": run_id,
        "sent_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": meta,
        "links": {
            "summary": f"{api_url}/v1/summary?model={model}&run={run_id}",
            "alerts": f"{api_url}/v1/alerts?model={model}&run={run_id}",
            "docs": f"{api_url}/v1/docs",
        },
    }


def send_webhook(
    client: httpx.Client, partner_id: str, partner: PartnerConfig, event: dict[str, Any], *, sleep=time.sleep
) -> bool:
    """POSTs one signed event, retrying errors and non-2xx answers; True when the partner took it."""
    body = json.dumps(event, separators=(",", ":")).encode("utf-8")
    for attempt in range(1, WEBHOOK_ATTEMPTS + 1):
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "cumulus-announce",
            "X-Cumulus-Event": EVENT,
            "X-Cumulus-Event-Id": event["id"],
            "X-Cumulus-Timestamp": timestamp,
            "X-Cumulus-Signature": sign(partner.webhook_secret or "", timestamp, body),
        }
        try:
            response = client.post(partner.webhook_url or "", content=body, headers=headers)
            if response.is_success:
                logger.info("webhook_sent partner=%s run=%s status=%s", partner_id, event["id"], response.status_code)
                return True
            problem = f"status={response.status_code}"
        except httpx.HTTPError as exc:
            problem = f"error={type(exc).__name__}"
        logger.warning("webhook_failed partner=%s run=%s attempt=%s %s", partner_id, event["id"], attempt, problem)
        if attempt < WEBHOOK_ATTEMPTS:
            sleep(2**attempt)
    return False


def subscribers(settings: Settings) -> dict[str, PartnerConfig]:
    """Partners that asked for events: a webhook URL and secret, and not revoked."""
    return {
        partner_id: partner
        for partner_id, partner in settings.partners.items()
        if partner.webhook_url and partner.webhook_secret and not partner.revoked
    }


def announce(
    settings: Settings,
    models: list[str],
    *,
    api_url: str,
    client: httpx.Client,
    timeout: float = 900,
    interval: float = 20,
    sleep=time.sleep,
) -> list[str]:
    """Waits for each model's active run to be live and sends it to every subscriber; returns problems."""
    problems: list[str] = []
    partners = subscribers(settings)
    for model in models:
        run_id = active_run_id(for_model(settings, model))
        if not run_id:
            problems.append(f"{model}: no active run")
            continue
        try:
            meta = wait_until_live(client, api_url, model, run_id, timeout=timeout, interval=interval, sleep=sleep)
        except TimeoutError as exc:
            problems.append(str(exc))
            continue
        print(f"{model}: {run_id} is live")
        event = event_for(meta, api_url)
        for partner_id, partner in partners.items():
            if not send_webhook(client, partner_id, partner, event, sleep=sleep):
                problems.append(f"{model}: webhook to {partner_id} failed")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Announce newly published runs to partners.")
    parser.add_argument("--model", action="append", required=True, help="Model with a new run (repeatable).")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="The deployed API to wait for.")
    parser.add_argument("--timeout", type=float, default=900, help="Seconds to wait for each run to go live.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    print(f"{len(subscribers(settings))} partner webhook(s)")
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        problems = announce(settings, args.model, api_url=args.api_url.rstrip("/"), client=client, timeout=args.timeout)
    for problem in problems:
        print(f"::error::{problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
