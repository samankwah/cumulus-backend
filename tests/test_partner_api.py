"""The open /v1 API: no key needed, optional partner keys, the per-run summary archive, formats, alerts, caching and CORS."""

from __future__ import annotations

import csv
import gzip
import io
import json

import pytest
from fastapi.testclient import TestClient

from cumulus.api.security import RateLimiter, partner_key_hash
from cumulus.services import partner_service
from cumulus.settings import get_settings
from cumulus.subseasonal import summary
from cumulus.subseasonal.ingest import ingest_run
from cumulus.subseasonal.sources import LocalFolderSource

from tests.test_subseasonal import _make_run, settings  # noqa: F401 - the shared settings fixture

KEY = "cmls_test-key"
PARTNERS = {
    "nadmo": {"name": "NADMO", "key_sha256": partner_key_hash(KEY)},
    "vra": {"name": "VRA", "key_sha256": partner_key_hash("cmls_vra")},
    "old": {"name": "Former partner", "key_sha256": partner_key_hash("cmls_old"), "revoked": True},
}
RUN_ID = "ifs_unet_2026092500"


@pytest.fixture()
def client(settings, monkeypatch, tmp_path):  # noqa: F811 - pytest fixture injection
    monkeypatch.setenv("CUMULUS_PARTNERS", json.dumps(PARTNERS))
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__ARCHIVE_SUMMARIES", "true")
    get_settings.cache_clear()
    scoped = get_settings()
    _make_run(tmp_path / "src", days=14)
    ingest_run(scoped, LocalFolderSource(tmp_path / "src"))
    from cumulus.api.security import open_api_limiter
    from cumulus.main import app

    open_api_limiter.reset()
    return TestClient(app)


def _get(client, path, key=None, **params):
    return client.get(f"/v1{path}", params=params, headers={"X-API-Key": key} if key else {})


def test_ingest_archives_a_summary_for_every_area(client, tmp_path):
    path = tmp_path / "archive" / "ifs_unet" / f"{RUN_ID}.json.gz"
    assert path.exists()
    with gzip.open(path, "rb") as handle:
        stored = json.loads(handle.read())
    assert stored["run_id"] == RUN_ID and stored["valid_from"] == "2026-09-25" and stored["valid_to"] == "2026-10-08"
    assert len(stored["levels"]["region"]) == 16 and len(stored["levels"]["district"]) > 200
    row = next(item for item in stored["levels"]["region"] if item["name"] == "Northern")
    assert row["id"] == "northern" and [week["week"] for week in row["weeks"]] == [1, 2]


def test_summary_matches_the_area_series_the_map_shows(client):
    area = client.get("/subseasonal/area", params={"level": "region", "name": "Northern"}).json()
    rows = _get(client, "/summary", level="region").json()["data"]
    row = next(item for item in rows if item["name"] == "Northern")
    assert row["total"]["rain_mm"] == area["metrics"]["total_mm"]
    assert row["total"]["rainy_days"] == area["metrics"]["rainy_days"]
    assert row["weeks"][0]["rain_mm"] == area["weeks"][0]["value"]


def test_open_without_a_key_and_partner_keys_are_optional(client, caplog):
    caplog.set_level("INFO", logger="cumulus.api.security")
    for path in ("/runs/latest", "/runs", "/areas", "/summary", "/alerts", "/grid", "/map"):
        assert _get(client, path).status_code == 200, path
    assert _get(client, "/summary", key=KEY).status_code == 200
    assert _get(client, "/summary", key="cmls_vra").status_code == 200
    assert "caller=public" in caplog.text and "caller=nadmo" in caplog.text and "caller=vra" in caplog.text
    # A wrong or revoked key is refused, so a partner notices instead of being counted as public.
    assert _get(client, "/runs/latest", key="cmls_wrong").status_code == 401
    assert _get(client, "/runs/latest", key="cmls_old").status_code == 401


def test_any_web_page_may_call_the_open_api(client):
    origin = "https://example.org"
    response = client.get("/v1/runs/latest", headers={"Origin": origin})
    assert response.headers["access-control-allow-origin"] == "*" and "ETag" in response.headers["access-control-expose-headers"]
    preflight = client.options(
        "/v1/summary", headers={"Origin": origin, "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "X-API-Key"}
    )
    assert preflight.status_code == 200 and preflight.headers["access-control-allow-origin"] == "*"
    # The rest of the API keeps the site's own allow-list.
    other = client.get("/subseasonal/models", headers={"Origin": origin})
    assert "access-control-allow-origin" not in other.headers


def test_latest_run_carries_meta_and_an_etag(client):
    response = _get(client, "/runs/latest")
    meta = response.json()["meta"]
    assert meta["run_id"] == RUN_ID and meta["official"] is True and meta["active"] is True
    assert "source_path" not in meta["provenance"]
    assert meta["issued"].startswith("2026-09-25") and "GMet" in meta["attribution"]
    assert response.headers["cache-control"] == "public, max-age=300, s-maxage=300"
    tag = response.headers["etag"]
    again = client.get("/v1/runs/latest", headers={"X-API-Key": KEY, "If-None-Match": tag})
    assert again.status_code == 304


def test_summary_csv_and_geojson(client):
    response = _get(client, "/summary", level="district", format="csv")
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.splitlines()
    assert lines[0].startswith("# ") and RUN_ID in lines[0]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[1:]))))
    assert {row["period"] for row in rows} == {"week 1", "week 2", "whole run"}
    geojson = _get(client, "/summary", level="district", format="geojson").json()
    assert geojson["type"] == "FeatureCollection" and len(geojson["features"]) > 200
    properties = geojson["features"][0]["properties"]
    assert "total_rain_mm" in properties and "week1_rain_mm" in properties
    assert _get(client, "/summary", level="region", format="geojson").status_code == 422


def test_areas_forecast_and_alerts(client):
    areas = _get(client, "/areas", level="region").json()["data"]
    assert {"id": "northern", "name": "Northern", "region": "Northern", "level": "region"} in areas
    forecast = _get(client, "/areas/region/northern/forecast").json()
    assert forecast["data"]["name"] == "Northern" and len(forecast["data"]["days"]) == 14
    assert _get(client, "/areas/region/atlantis/forecast").status_code == 404
    # Days 4-9 are dry everywhere in the synthetic run: a dry spell in week 1 and week 2 of every region.
    found = _get(client, "/alerts", level="region", dry_spell_chance=50).json()
    assert found["meta"]["criteria"]["dry_spell_chance"] == 50
    assert {item["kind"] for item in found["data"]} == {"dry_spell"}
    assert len({item["id"] for item in found["data"]}) == 16


def test_archived_runs_stay_listed_after_the_run_is_pruned(client, tmp_path):
    import shutil

    shutil.rmtree(tmp_path / "artifacts" / "ifs_unet" / RUN_ID)
    (tmp_path / "artifacts" / "ifs_unet" / "active.json").unlink()
    runs = _get(client, "/runs").json()["data"]
    assert runs == [{"run_id": RUN_ID, "issued": "2026-09-25T00:00:00+00:00", "active": False, "live": False, "archived": True}]
    archived = _get(client, "/summary", level="region", run=RUN_ID)
    assert archived.status_code == 200 and archived.headers["cache-control"] == "public, max-age=86400, s-maxage=86400"


def test_official_model_comes_from_the_admin_setting_with_fallbacks(settings, monkeypatch):  # noqa: F811
    assert partner_service.official_model(settings) == "ifs_unet"
    monkeypatch.setattr(partner_service, "_remote_official_model", lambda url: "ifs_diffusion")
    monkeypatch.setenv("CUMULUS_SUBSEASONAL__OFFICIAL_MODEL_URL", "https://example.test/api/forecast-settings")
    get_settings.cache_clear()
    assert partner_service.official_model(get_settings()) == "ifs_diffusion"
    # An unknown model from the admin is ignored rather than breaking the API.
    monkeypatch.setattr(partner_service, "_remote_official_model", lambda url: "not-a-model")
    assert partner_service.official_model(get_settings()) == "ifs_unet"


def test_area_ids_are_stable_slugs():
    assert summary.area_id("Tamale Metropolitan", "Northern") == "northern--tamale-metropolitan"
    assert summary.area_id("Greater Accra") == "greater-accra"
    assert summary.area_id("Ga East Municipal", "Greater Accra") == "greater-accra--ga-east-municipal"


def test_point_forecast_matches_the_map_click(client):
    sample = client.get("/subseasonal/sample", params={"latitude": 9.4, "longitude": -0.85}).json()
    data = _get(client, "/point/forecast", lat=9.4, lon=-0.85).json()["data"]
    assert data["kind"] == "point" and data["metrics"] == sample["metrics"] and len(data["days"]) == 14
    assert _get(client, "/point/forecast", lat=20.0, lon=-0.85).status_code == 422


def test_rainfall_over_any_dates(client):
    area = client.get("/subseasonal/area", params={"level": "region", "name": "Northern"}).json()
    data = _get(client, "/rainfall", start="2026-09-27", end="2026-10-01", level="region", area="northern").json()["data"]
    expected = sum(day["value"] for day in area["days"][2:7])
    assert data["days"] == 5 and data["rain_mm"] == pytest.approx(expected, abs=0.2)
    assert [day["date"] for day in data["daily_mm"]] == ["2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01"]
    point = _get(client, "/rainfall", start="2026-09-25", end="2026-09-25", lat=9.4, lon=-0.85).json()["data"]
    assert point["kind"] == "point" and point["days"] == 1
    assert _get(client, "/rainfall", start="2026-09-01", end="2026-09-05", level="region", area="northern").status_code == 422
    assert _get(client, "/rainfall", start="2026-09-25", end="2026-09-26").status_code == 422


def test_summary_counts_extreme_days(client):
    row = _get(client, "/summary", level="region").json()["data"][0]
    assert row["total"]["extreme_days"] is not None
    assert all(week["extreme_days"] >= 0 for week in row["weeks"])


def test_map_layer_and_grid_downloads(client):
    layer = _get(client, "/map", layer="rainy_days", aggregation="weekly", index=1).json()["data"]
    assert layer["tile_url"].startswith("http://testserver/subseasonal/tiles/") and layer["legend"]["bins"]
    grid = _get(client, "/grid", layer="rainy_days", aggregation="weekly", index=1).json()["data"]
    assert len(grid["values"]) == len(grid["latitudes"]) and len(grid["values"][0]) == len(grid["longitudes"])
    assert any(any(row) for row in grid["in_ghana"])
    period = _get(client, "/grid", start="2026-09-25", end="2026-09-26").json()["data"]
    assert period["layer"] == "rainfall" and (period["start_date"], period["end_date"]) == ("2026-09-25", "2026-09-26")
    rows = list(csv.reader(io.StringIO(_get(client, "/grid", start="2026-09-25", end="2026-09-26", format="csv").text)))
    assert rows[1] == ["latitude", "longitude", "value", "in_ghana"] and len(rows) == 2 + len(period["latitudes"]) * len(period["longitudes"])
    netcdf = _get(client, "/grid", layer="extreme_rain", format="netcdf")
    assert netcdf.headers["content-type"] == "application/x-netcdf" and netcdf.content[:3] == b"CDF"
    assert _get(client, "/grid", start="2026-12-01", end="2026-12-02").status_code == 422


def test_rate_limits_per_address_and_per_partner(client, monkeypatch):
    monkeypatch.setenv("CUMULUS_OPEN_API__RATE_LIMIT_PER_MINUTE", "3")
    monkeypatch.setenv("CUMULUS_OPEN_API__PARTNER_RATE_LIMIT_PER_MINUTE", "5")
    get_settings.cache_clear()
    one = {"X-Forwarded-For": "203.0.113.7"}
    statuses = [client.get("/v1/runs/latest", headers=one).status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    blocked = client.get("/v1/areas", headers=one)
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1
    # Another address, and a partner key, have their own allowance.
    assert client.get("/v1/runs/latest", headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 200
    assert client.get("/v1/runs/latest", headers={**one, "X-API-Key": KEY}).status_code == 200


def test_rate_limiter_window_slides_and_forgets_old_callers():
    limiter = RateLimiter(max_callers=2)
    assert limiter.hit("a", 2, now=0) is None and limiter.hit("a", 2, now=1) is None
    assert limiter.hit("a", 2, now=2) == pytest.approx(58)
    assert limiter.hit("a", 2, now=61) is None
    limiter.hit("b", 2, now=61)
    limiter.hit("c", 2, now=61)
    assert "a" not in limiter._calls and limiter.hit("x", 0) is None
