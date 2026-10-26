"""Access guards for admin, legacy and partner endpoints."""

from __future__ import annotations

from collections import OrderedDict, deque
import hashlib
import logging
import math
import secrets
import threading
import time

from fastapi import Header, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from cumulus.settings import get_settings

logger = logging.getLogger(__name__)


def require_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Allow the request only with the configured admin key.

    With no key configured the endpoint is hidden (404) rather than open, unless the deployment
    explicitly opts in with ``CUMULUS_ALLOW_UNAUTHENTICATED_ADMIN`` (local dev, tests).
    """
    settings = get_settings()
    expected = settings.admin_api_key
    if not expected:
        if settings.allow_unauthenticated_admin:
            return
        raise HTTPException(status_code=404, detail="Not Found")
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def require_legacy_endpoints() -> None:
    """Hide the retired random-forest/ERA5 endpoints unless CUMULUS_ENABLE_LEGACY_ENDPOINTS is on."""
    if not get_settings().enable_legacy_endpoints:
        raise HTTPException(status_code=404, detail="Not Found")


def partner_key_hash(key: str) -> str:
    """How a partner key is stored: its SHA-256 in hex."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# Declared as a security scheme so the docs offer an "Authorize" box for an optional key.
PARTNER_KEY_HEADER = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="Optional: a partner key from GMet, so your calls are counted as yours."
)

PUBLIC_CALLER = "public"
RATE_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Sliding one-minute windows of call times per caller, in memory.

    The oldest callers are forgotten past ``max_callers``, so a flood of addresses cannot grow memory.
    """

    def __init__(self, max_callers: int = 10_000) -> None:
        self.max_callers = max_callers
        self._calls: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, caller: str, limit: int, now: float | None = None) -> float | None:
        """Counts a call; returns None when allowed, else the seconds until the next call is."""
        if limit <= 0:
            return None
        now = time.monotonic() if now is None else now
        with self._lock:
            calls = self._calls.pop(caller, None) or deque()
            while calls and now - calls[0] >= RATE_WINDOW_SECONDS:
                calls.popleft()
            self._calls[caller] = calls
            while len(self._calls) > self.max_callers:
                self._calls.popitem(last=False)
            if len(calls) >= limit:
                return RATE_WINDOW_SECONDS - (now - calls[0])
            calls.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


open_api_limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """The caller's address: Vercel puts the real one first in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or request.headers.get("x-real-ip", "") or (request.client.host if request.client else "unknown")


def _limit(caller_key: str, limit: int, route: str) -> None:
    wait = open_api_limiter.hit(caller_key, limit)
    if wait is not None:
        logger.warning("api_rate_limited caller=%s route=%s", caller_key, route)
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests: up to {limit} a minute. Wait a moment, or ask GMet for a partner key for more.",
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )


def identify_caller(route: str):
    """A dependency naming who called an open /v1 route: a partner, or "public" without a key.

    The API is free and open, so a key is never required; partners who send one are logged by name,
    get a higher rate limit and can later get webhooks. A key that is wrong or revoked is refused, so
    a partner notices rather than silently being counted as public. Keys are compared in constant time.
    """

    def dependency(request: Request, x_api_key: str | None = Security(PARTNER_KEY_HEADER)) -> str:
        limits = get_settings().open_api
        if not x_api_key:
            _limit(f"ip:{client_ip(request)}", limits.rate_limit_per_minute, route)
            logger.info("api_call caller=%s route=%s", PUBLIC_CALLER, route)
            return PUBLIC_CALLER
        partners = get_settings().partners
        digest = partner_key_hash(x_api_key)
        match = None
        for partner_id, partner in partners.items():
            if secrets.compare_digest(digest, partner.key_sha256.strip().lower()):
                match = partner_id
        if match is None or partners[match].revoked:
            raise HTTPException(status_code=401, detail="Invalid or revoked X-API-Key. Leave it out to use the open API.")
        _limit(f"partner:{match}", limits.partner_rate_limit_per_minute, route)
        logger.info("api_call caller=%s route=%s", match, route)
        return match

    return dependency
