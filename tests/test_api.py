from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient

import cumulus.services.forecast_service as forecast_service_module
from cumulus.main import app
from cumulus.modeling.predictor import clear_model_bundle_cache
from cumulus.modeling.trainer import train_baseline_model
from cumulus.services.seasonal_map_service import (
    _build_legend,
    _classify_cessation,
    _classify_dry_spell,
    _load_district_catalog,
    clear_seasonal_map_cache,
)
from cumulus.settings import get_settings


def _prepare_public_serving_stack(monkeypatch, tmp_path) -> PathBundle:
    monkeypatch.setenv("CUMULUS_MODEL_ARTIFACT_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CUMULUS_BIAS_ARTIFACT_DIR", str(tmp_path / "bias"))
    monkeypatch.setenv("CUMULUS_EVALUATION_DIR", str(tmp_path / "evaluation"))
    monkeypatch.setenv("CUMULUS_NATIONWIDE__ARTIFACT_DIR", str(tmp_path / "nationwide"))
    monkeypatch.setenv("CUMULUS_SEASONAL_MAP__ARTIFACT_DIR", str(tmp_path / "seasonal-map"))
    clear_model_bundle_cache()
    get_settings.cache_clear()
    settings = get_settings()

    training_frame = pd.DataFrame(
        {
            "location_id": ["accra"] * 40,
            "time": pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC"),
            "latitude": [5.6] * 40,
            "longitude": [-0.18] * 40,
            "precip_mm": [float((index % 6) + 1) for index in range(40)],
            "temp_c": [28.0 + float(index % 3) for index in range(40)],
            "u10": [2.0] * 40,
            "v10": [1.0] * 40,
            "rainfall_mm": [float((index % 6) + 1.5) for index in range(40)],
        }
    )
    train_baseline_model(training_frame, settings)

    forecast_path = tmp_path / "forecast.nc"
    forecast_times = pd.date_range("2026-04-24", periods=12, freq="D")
    dataset = xr.Dataset(
        data_vars={
            "tp": (
                ("time", "latitude", "longitude"),
                [[[0.010]], [[0.012]], [[0.008]], [[0.003]], [[0.000]], [[0.000]], [[0.000]], [[0.000]], [[0.001]], [[0.002]], [[0.011]], [[0.007]]],
            ),
            "t2m": (
                ("time", "latitude", "longitude"),
                [[[301.15]], [[301.15]], [[300.15]], [[299.15]], [[299.15]], [[300.15]], [[301.15]], [[302.15]], [[303.15]], [[304.15]], [[303.15]], [[302.15]]],
            ),
            "u10": (
                ("time", "latitude", "longitude"),
                [[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]],
            ),
            "v10": (
                ("time", "latitude", "longitude"),
                [[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]],
            ),
        },
        coords={"time": forecast_times, "latitude": [5.6], "longitude": [-0.18]},
    )
    dataset["tp"].attrs["units"] = "m"
    dataset["t2m"].attrs["units"] = "K"
    dataset.to_netcdf(forecast_path, engine="scipy")

    monkeypatch.setenv("CUMULUS_UPSTREAM_FORECAST_PATH", str(forecast_path))
    monkeypatch.setenv("CUMULUS_UPSTREAM_FORECAST_ENGINE", "scipy")
    get_settings.cache_clear()
    clear_model_bundle_cache()
    return PathBundle(forecast_path=str(forecast_path))


class PathBundle:
    def __init__(self, forecast_path: str):
        self.forecast_path = forecast_path


def _district_catalog():
    clear_seasonal_map_cache()
    settings = get_settings()
    return _load_district_catalog(
        str(settings.seasonal_map.district_geojson_path),
        settings.seasonal_map.northern_latitude_threshold,
    )


def _mixed_region_name(catalog: list[dict[str, object]]) -> str:
    region_zones: dict[str, set[str]] = {}
    for district in catalog:
        region_name = str(district["region_name"])
        region_zones.setdefault(region_name, set()).add(str(district["regime_zone"]))
    return next(region_name for region_name, zones in region_zones.items() if len(zones) > 1)


def _expected_refresh_count(*, theme: str | None = None, season_profile: str | None = None) -> int:
    settings = get_settings()
    profiles = [season_profile] if season_profile is not None else list(settings.seasonal_map.profiles)
    themes = [theme] if theme is not None else [
        "onset",
        "cessation",
        "early_dry_spell",
        "late_dry_spell",
        "rainfall_amount",
        "rainy_days",
    ]
    total = 0
    for profile_id in profiles:
        profile = settings.seasonal_map.profiles[profile_id]
        for theme_id in themes:
            if theme_id in {"onset", "cessation", "early_dry_spell", "late_dry_spell"}:
                total += 1
            elif theme_id in {"rainfall_amount", "rainy_days"}:
                total += len(profile.calendar_subseasons)
    return total


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "data_sources" in response.json()


def test_predict_endpoint_accepts_valid_coordinates(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "latitude": 5.6037,
            "longitude": -0.1870,
            "location_id": "accra",
            "horizon_days": 7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["location_id"] == "accra"
    assert payload["latitude"] == 5.6037
    assert payload["longitude"] == -0.187
    assert payload["model_version"].startswith("rf_rainfall_")
    assert payload["generated_at"]
    assert payload["horizon_days"] == 7
    assert payload["forecast_source"] == "configured"
    assert payload["data_origin"] == "downloaded_real_source_data"
    assert payload["source_run_id"].startswith("configured-")
    assert payload["spatial_resolution_km"] == 4.0
    assert payload["calibration_version"].startswith("national_backbone_v1")
    assert len(payload["daily_forecast"]) == 7
    assert "rainfall_corrected_mm" in payload["daily_forecast"][0]
    assert payload["agro_characteristics"]["dry_spell_risk"] in {True, False}


def test_public_advisory_endpoint_returns_farmer_blocks(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/advisory",
        json={
            "latitude": 5.6037,
            "longitude": -0.1870,
            "location_id": "accra",
            "horizon_days": 10,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["location_id"] == "accra"
    assert payload["model_version"].startswith("rf_rainfall_")
    assert payload["forecast_source"] == "configured"
    assert payload["data_origin"] == "downloaded_real_source_data"
    assert payload["spatial_resolution_km"] == 4.0
    assert payload["generated_at"]
    assert payload["planting_recommendation"]["headline"]
    assert payload["dry_spell_alert"]["headline"]
    assert payload["irrigation_advice"]["headline"]
    assert payload["agro_characteristics"]["planting_window_signal"]


def test_predict_rejects_out_of_bounds_coordinates(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"latitude": 20.0, "longitude": -0.1870},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_coordinates"


def test_predict_returns_operational_error_when_forecast_source_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CUMULUS_MODEL_ARTIFACT_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CUMULUS_BIAS_ARTIFACT_DIR", str(tmp_path / "bias"))
    monkeypatch.setenv("CUMULUS_EVALUATION_DIR", str(tmp_path / "evaluation"))
    monkeypatch.delenv("CUMULUS_UPSTREAM_FORECAST_PATH", raising=False)
    get_settings.cache_clear()
    clear_model_bundle_cache()

    client = TestClient(app)
    response = client.post("/predict", json={"latitude": 5.6037, "longitude": -0.1870})

    assert response.status_code == 503
    assert response.json()["error_code"] == "forecast_source_not_configured"


def test_predict_returns_operational_error_when_active_model_missing(monkeypatch, tmp_path):
    forecast_path = tmp_path / "forecast.nc"
    dataset = xr.Dataset(
        data_vars={
            "tp": (("time", "latitude", "longitude"), [[[0.010]]]),
            "t2m": (("time", "latitude", "longitude"), [[[301.15]]]),
            "u10": (("time", "latitude", "longitude"), [[[2.0]]]),
            "v10": (("time", "latitude", "longitude"), [[[1.0]]]),
        },
        coords={"time": pd.date_range("2026-04-24", periods=1, freq="D"), "latitude": [5.6], "longitude": [-0.18]},
    )
    dataset["tp"].attrs["units"] = "m"
    dataset["t2m"].attrs["units"] = "K"
    dataset.to_netcdf(forecast_path, engine="scipy")

    monkeypatch.setenv("CUMULUS_MODEL_ARTIFACT_DIR", str(tmp_path / "missing-models"))
    monkeypatch.setenv("CUMULUS_BIAS_ARTIFACT_DIR", str(tmp_path / "missing-bias"))
    monkeypatch.setenv("CUMULUS_EVALUATION_DIR", str(tmp_path / "evaluation"))
    monkeypatch.setenv("CUMULUS_UPSTREAM_FORECAST_PATH", str(forecast_path))
    monkeypatch.setenv("CUMULUS_UPSTREAM_FORECAST_ENGINE", "scipy")
    get_settings.cache_clear()
    clear_model_bundle_cache()

    client = TestClient(app)
    response = client.post("/predict", json={"latitude": 5.6037, "longitude": -0.1870})

    assert response.status_code == 503
    assert response.json()["error_code"] == "model_artifacts_not_available"


def test_forecast_and_legacy_advisory_routes_remain_compatible(monkeypatch, tmp_path):
    paths = _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    forecast_response = client.post(
        "/forecast",
        json={
            "locations": [{"location_id": "accra", "latitude": 5.6037, "longitude": -0.1870}],
            "forecast_source": {"path": paths.forecast_path, "variables": ["tp", "u10", "v10", "t2m"]},
            "horizon_days": 5,
        },
    )
    assert forecast_response.status_code == 200
    assert forecast_response.json()["results"][0]["location_id"] == "accra"
    assert forecast_response.json()["forecast_source"] == "configured"
    assert forecast_response.json()["data_origin"] == "downloaded_real_source_data"
    assert forecast_response.json()["spatial_resolution_km"] == 4.0
    seasonal_refresh = forecast_response.json()["seasonal_refresh"]
    assert seasonal_refresh["forecast_source"] == "configured"
    assert seasonal_refresh["attempted_count"] == _expected_refresh_count()
    assert seasonal_refresh["succeeded_count"] == seasonal_refresh["attempted_count"]
    assert seasonal_refresh["failed_count"] == 0
    onset_active = client.get("/seasonal-map/active?theme=onset&season_profile=northern_single&mode=seasonal")
    assert onset_active.status_code == 200

    legacy_response = client.post(
        "/advisory/legacy",
        json={
            "rainfall_series": [
                {"date": "2024-05-01", "rainfall_mm": 10.0},
                {"date": "2024-05-02", "rainfall_mm": 7.0},
                {"date": "2024-05-03", "rainfall_mm": 6.0},
                {"date": "2024-05-04", "rainfall_mm": 1.0},
            ]
        },
    )
    assert legacy_response.status_code == 200
    assert "cum_rain_7d_mm" in legacy_response.json()


def test_forecast_endpoint_surfaces_distinct_seasonal_refresh_failure(monkeypatch, tmp_path):
    paths = _prepare_public_serving_stack(monkeypatch, tmp_path)

    def _failing_refresh(*args, **kwargs):
        return {
            "forecast_source": "configured",
            "forecast_source_label": "Configured Forecast Feed",
            "requested_theme": None,
            "requested_season_profile": None,
            "requested_mode": None,
            "requested_subseason": None,
            "attempted_count": 1,
            "succeeded_count": 0,
            "failed_count": 1,
            "attempted": [{"theme": "onset", "season_profile": "northern_single", "mode": "seasonal", "subseason": None}],
            "succeeded": [],
            "failed": [
                {
                    "theme": "onset",
                    "season_profile": "northern_single",
                    "mode": "seasonal",
                    "subseason": None,
                    "error": "boom",
                }
            ],
        }

    monkeypatch.setattr(forecast_service_module, "refresh_seasonal_map_products", _failing_refresh)
    client = TestClient(app)
    forecast_response = client.post(
        "/forecast",
        json={
            "locations": [{"location_id": "accra", "latitude": 5.6037, "longitude": -0.1870}],
            "forecast_source": {"path": paths.forecast_path, "variables": ["tp", "u10", "v10", "t2m"]},
            "horizon_days": 5,
        },
    )
    assert forecast_response.status_code == 500
    payload = forecast_response.json()
    assert payload["error_code"] == "seasonal_map_refresh_failed"
    assert "seasonal map refresh failed" in payload["detail"].lower()


def test_forecast_raster_metadata_endpoint_returns_expected_contract(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/raster?variable=rainfall_daily_mm&horizon_day=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["variable"] == "rainfall_daily_mm"
    assert payload["variable_label"] == "Daily Rainfall"
    assert payload["unit"] == "mm"
    assert payload["horizon_day"] == 2
    assert payload["forecast_source"] == "configured"
    assert payload["forecast_source_label"] == "Configured Forecast Feed"
    assert payload["source_run_id"].startswith("configured-")
    assert payload["tile_url"].startswith("/forecast/raster/tiles/{z}/{x}/{y}.png?")
    assert "variable=rainfall_daily_mm" in payload["tile_url"]
    assert "horizon_day=2" in payload["tile_url"]
    assert payload["lower_bound"] <= payload["upper_bound"]
    assert payload["available_horizon_days"] == list(range(1, 13))
    assert len(payload["legend_ticks"]) == 5
    assert payload["color_ramp"][0]["offset"] == 0.0
    assert payload["color_ramp"][-1]["offset"] == 1.0
    assert len(payload["grid"]["latitudes"]) == 1
    assert len(payload["grid"]["longitudes"]) == 1
    assert payload["grid"]["values"][0][0] is not None


def test_forecast_raster_sample_endpoint_returns_nearest_grid_metadata(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/raster/sample?latitude=5.6037&longitude=-0.1870&variable=temperature_c&horizon_day=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["variable"] == "temperature_c"
    assert payload["variable_label"] == "Air Temperature"
    assert payload["unit"] == "C"
    assert payload["horizon_day"] == 1
    assert payload["forecast_source"] == "configured"
    assert payload["forecast_source_label"] == "Configured Forecast Feed"
    assert payload["source_run_id"].startswith("configured-")
    assert payload["nearest_latitude"] == 5.6
    assert payload["nearest_longitude"] == -0.18
    assert payload["value"] is not None


def test_forecast_raster_tile_endpoint_serves_png(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/raster/tiles/6/31/29.png?variable=temperature_c&horizon_day=1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_farmer_advisory_endpoint():
    client = TestClient(app)
    response = client.post(
        "/farmer-advisory",
        json={
            "location_id": "tamale",
            "daily_forecast": [
                {"date": "2024-05-01", "rainfall_mm": 8.0, "temperature_c": 30.0},
                {"date": "2024-05-02", "rainfall_mm": 7.0, "temperature_c": 30.0},
                {"date": "2024-05-03", "rainfall_mm": 6.0, "temperature_c": 30.0},
                {"date": "2024-05-04", "rainfall_mm": 0.0, "temperature_c": 30.0},
                {"date": "2024-05-05", "rainfall_mm": 0.0, "temperature_c": 30.0},
                {"date": "2024-05-06", "rainfall_mm": 0.0, "temperature_c": 30.0},
                {"date": "2024-05-07", "rainfall_mm": 0.0, "temperature_c": 30.0},
                {"date": "2024-05-08", "rainfall_mm": 0.0, "temperature_c": 30.0},
                {"date": "2024-05-09", "rainfall_mm": 0.0, "temperature_c": 30.0},
                {"date": "2024-05-10", "rainfall_mm": 0.0, "temperature_c": 30.0},
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["location_id"] == "tamale"
    assert payload["planting_recommendation"]["level"] == "delay_due_to_dry_spell_risk"
    assert payload["dry_spell_alert"]["level"] == "warning"
    assert payload["irrigation_advice"]["headline"]


def test_train_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("CUMULUS_MODEL_ARTIFACT_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CUMULUS_BIAS_ARTIFACT_DIR", str(tmp_path / "bias"))
    monkeypatch.setenv("CUMULUS_EVALUATION_DIR", str(tmp_path / "evaluation"))
    monkeypatch.setenv("CUMULUS_NATIONWIDE__ARTIFACT_DIR", str(tmp_path / "nationwide"))
    get_settings.cache_clear()

    merged = pd.DataFrame(
        {
            "location_id": ["accra"] * 30,
            "time": pd.date_range("2024-05-01", periods=30, freq="D", tz="UTC"),
            "latitude": [5.6] * 30,
            "longitude": [-0.18] * 30,
            "precip_mm": [float(index % 10) for index in range(30)],
            "temp_c": [28.0] * 30,
            "u10": [2.0] * 30,
            "v10": [1.0] * 30,
            "rainfall_mm": [float((index % 10) + 1) for index in range(30)],
        }
    )
    path = tmp_path / "merged.csv"
    merged.to_csv(path, index=False)

    client = TestClient(app)
    response = client.post("/train", json={"merged_dataset_path": str(path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"].startswith("rf_rainfall_")
    assert payload["bias_method"] in {"mean_bias", "quantile_mapping"}
    assert set(payload["bias_comparison"]["methods"]) == {"raw", "mean_bias", "quantile_mapping"}
    assert "metrics_json" in payload["evaluation_paths"]


def test_nationwide_generation_and_read_endpoints(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    generate = client.post("/nationwide/generate?horizon_days=10")
    assert generate.status_code == 200
    manifest = generate.json()
    assert manifest["run_id"].startswith("nationwide_")
    assert manifest["forecast_source"] == "configured"
    assert manifest["source_run_id"].startswith("configured-")
    assert manifest["spatial_resolution_km"] == 4.0
    assert manifest["horizon_days"] == 10
    assert manifest["available_location_count"] >= 10
    assert manifest["region_count"] >= 10
    assert manifest["district_count"] >= 10

    active = client.get("/nationwide/run/active")
    assert active.status_code == 200
    assert active.json()["run_id"] == manifest["run_id"]

    locations = client.get("/nationwide/locations?page=1&page_size=3&region=Greater%20Accra")
    assert locations.status_code == 200
    location_payload = locations.json()
    assert location_payload["page_size"] == 3
    assert location_payload["total_locations"] >= 1
    assert location_payload["items"][0]["region"] == "Greater Accra"
    assert location_payload["items"][0]["forecast_source"] == "configured"
    assert location_payload["items"][0]["data_origin"] == "nationwide_artifact_cache"
    assert location_payload["items"][0]["spatial_resolution_km"] == 4.0
    assert location_payload["items"][0]["agro_characteristics"]["planting_window_signal"]
    assert location_payload["items"][0]["point_advisory"]["planting_recommendation"]["headline"]

    region = client.get("/nationwide/regions/Greater%20Accra")
    assert region.status_code == 200
    region_payload = region.json()
    assert region_payload["geography_type"] == "region"
    assert region_payload["geography_name"] == "Greater Accra"
    assert region_payload["forecast_source"] == "configured"
    assert region_payload["data_origin"] == "nationwide_artifact_cache"
    assert region_payload["spatial_resolution_km"] == 4.0
    assert region_payload["coverage_count"] >= 1
    assert region_payload["planting_recommendation"]["available_location_count"] >= 1

    district = client.get("/nationwide/districts/Accra%20Metropolitan")
    assert district.status_code == 200
    district_payload = district.json()
    assert district_payload["geography_type"] == "district"
    assert district_payload["geography_name"] == "Accra Metropolitan"
    assert district_payload["forecast_source"] == "configured"
    assert district_payload["data_origin"] == "nationwide_artifact_cache"
    assert district_payload["dry_spell_alert"]["severity_bucket"] in {"low", "moderate", "high"}


def test_nationwide_runs_are_scoped_per_forecast_source(monkeypatch, tmp_path):
    paths = _prepare_public_serving_stack(monkeypatch, tmp_path)
    era5_path = tmp_path / "era5.nc"
    gfs_path = tmp_path / "gfs.nc"

    era5_dataset = xr.open_dataset(paths.forecast_path, engine="scipy")
    era5_dataset.to_netcdf(era5_path, engine="scipy")
    gfs_dataset = era5_dataset.copy(deep=True)
    gfs_dataset["tp"] = gfs_dataset["tp"] * 0.75
    gfs_dataset.to_netcdf(gfs_path, engine="scipy")
    era5_dataset.close()
    gfs_dataset.close()

    monkeypatch.setenv("CUMULUS_ERA5_FORECAST_PATH", str(era5_path))
    monkeypatch.setenv("CUMULUS_ERA5_FORECAST_ENGINE", "scipy")
    monkeypatch.setenv("CUMULUS_GFS_FORECAST_PATH", str(gfs_path))
    monkeypatch.setenv("CUMULUS_GFS_FORECAST_ENGINE", "scipy")
    monkeypatch.setenv("CUMULUS_DEFAULT_FORECAST_SOURCE", "era5")
    get_settings.cache_clear()
    clear_model_bundle_cache()

    client = TestClient(app)
    era5_run = client.post("/nationwide/generate?horizon_days=7&forecast_source=era5")
    gfs_run = client.post("/nationwide/generate?horizon_days=7&forecast_source=gfs")

    assert era5_run.status_code == 200
    assert gfs_run.status_code == 200
    era5_payload = era5_run.json()
    gfs_payload = gfs_run.json()
    assert era5_payload["forecast_source"] == "era5"
    assert gfs_payload["forecast_source"] == "gfs"
    assert era5_payload["run_id"] != gfs_payload["run_id"]

    active_era5 = client.get("/nationwide/run/active?forecast_source=era5")
    active_gfs = client.get("/nationwide/run/active?forecast_source=gfs")
    assert active_era5.status_code == 200
    assert active_gfs.status_code == 200
    assert active_era5.json()["run_id"] == era5_payload["run_id"]
    assert active_gfs.json()["run_id"] == gfs_payload["run_id"]

    era5_locations = client.get("/nationwide/locations?page=1&page_size=1&forecast_source=era5")
    gfs_locations = client.get("/nationwide/locations?page=1&page_size=1&forecast_source=gfs")
    assert era5_locations.status_code == 200
    assert gfs_locations.status_code == 200
    era5_item = era5_locations.json()["items"][0]
    gfs_item = gfs_locations.json()["items"][0]
    assert era5_item["forecast_source"] == "era5"
    assert gfs_item["forecast_source"] == "gfs"
    assert set(era5_item["agro_characteristics"]) == set(gfs_item["agro_characteristics"])
    assert set(era5_item["point_advisory"]) == set(gfs_item["point_advisory"])


def test_predict_and_advisory_can_use_active_nationwide_artifacts(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    generate = client.post("/nationwide/generate?horizon_days=10")
    assert generate.status_code == 200

    monkeypatch.delenv("CUMULUS_UPSTREAM_FORECAST_PATH", raising=False)
    monkeypatch.delenv("CUMULUS_UPSTREAM_FORECAST_ENGINE", raising=False)
    get_settings.cache_clear()
    clear_model_bundle_cache()

    predict = client.post(
        "/predict",
        json={
            "latitude": 5.6037,
            "longitude": -0.1870,
            "location_id": "accra",
            "horizon_days": 7,
        },
    )
    assert predict.status_code == 200
    predict_payload = predict.json()
    assert predict_payload["location_id"] == "accra"
    assert predict_payload["forecast_source"] == "configured"
    assert predict_payload["data_origin"] == "nationwide_artifact_cache"
    assert len(predict_payload["daily_forecast"]) == 7

    advisory = client.post(
        "/advisory",
        json={
            "latitude": 5.6037,
            "longitude": -0.1870,
            "location_id": "accra",
            "horizon_days": 10,
        },
    )
    assert advisory.status_code == 200
    advisory_payload = advisory.json()
    assert advisory_payload["location_id"] == "accra"
    assert advisory_payload["forecast_source"] == "configured"
    assert advisory_payload["data_origin"] == "nationwide_artifact_cache"
    assert advisory_payload["planting_recommendation"]["headline"]


def test_seasonal_map_generation_and_active_endpoint(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)
    artifact_dir = tmp_path / "seasonal-map"
    catalog = _district_catalog()
    north_ids = {district["location_id"] for district in catalog if district["regime_zone"] == "north"}
    north_regions = {str(district["region_name"]) for district in catalog if district["regime_zone"] == "north"}

    generate = client.post("/seasonal-map/generate?theme=onset&season_profile=northern_single&mode=seasonal")
    assert generate.status_code == 200
    manifest = generate.json()
    assert manifest["product_id"].startswith("seasonal_configured_northern_single_onset_seasonal_")
    assert manifest["forecast_source"] == "configured"
    assert manifest["theme"] == "onset"
    assert manifest["season_profile"] == "northern_single"
    assert manifest["mode"] == "seasonal"
    assert manifest["subseason"] is None
    assert manifest["district_count"] == len(north_ids)
    assert manifest["region_count"] == len(north_regions)
    assert (artifact_dir / "active_configured_northern_single_onset_seasonal.json").exists()
    assert (artifact_dir / "active_configured_northern_single_onset.json").exists()

    active = client.get("/seasonal-map/active?theme=onset&season_profile=northern_single&mode=seasonal")
    assert active.status_code == 200
    payload = active.json()
    assert payload["product_id"] == manifest["product_id"]
    assert payload["mode"] == "seasonal"
    assert payload["mode_label"] == "Seasonal"
    assert payload["forecast_cycle"]
    assert payload["forecast_source_label"] == "Configured Forecast Feed"
    assert payload["refresh_status"] in {"fresh", "stale"}
    assert payload["legend"][0]["label"] == "Early"
    district_metric = payload["district_items"][0]["metric"]
    assert district_metric["theme"] == "onset"
    assert district_metric["category_label"] in {"Early", "Normal", "Late"}
    assert district_metric["display_value"].endswith("%")
    assert district_metric["unit"] == "percent"
    assert district_metric["dominant_percentage"] >= 0
    assert district_metric["category_probabilities"]
    assert district_metric["criteria_note"].startswith("Detected from 15 Mar")
    percentages = {item["category_code"]: item["percentage"] for item in district_metric["category_probabilities"]}
    assert len(percentages) == 3
    assert round(sum(percentages.values()), 1) == 100.0
    assert district_metric["dominant_percentage"] == max(percentages.values())
    assert percentages[district_metric["dominant_category_code"]] == district_metric["dominant_percentage"]
    assert payload["region_items"][0]["metric"]["theme"] == "onset"

    deterministic_active = client.get(
        "/seasonal-map/deterministic/active?theme=onset&season_profile=northern_single&mode=seasonal"
    )
    assert deterministic_active.status_code == 200
    deterministic_payload = deterministic_active.json()
    assert deterministic_payload["product_id"] == manifest["product_id"]
    assert deterministic_payload["district_items"][0]["metric"]["display_value"]
    assert deterministic_payload["district_items"][0]["metric"]["unit"] == "days_from_reference"
    assert "legend_label" in deterministic_payload["district_items"][0]["metric"]


@pytest.mark.parametrize("theme", ["early_dry_spell", "late_dry_spell"])
def test_seasonal_map_active_dry_spell_products_serve_whole_day_display_values(monkeypatch, tmp_path, theme):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    generate = client.post(f"/seasonal-map/generate?theme={theme}&season_profile=southern_major&mode=seasonal")
    assert generate.status_code == 200

    active = client.get(f"/seasonal-map/deterministic/active?theme={theme}&season_profile=southern_major&mode=seasonal")
    assert active.status_code == 200
    payload = active.json()
    metrics = [item["metric"] for item in payload["district_items"]] + [item["metric"] for item in payload["region_items"]]

    assert metrics
    assert all(metric["unit"] == "days" for metric in metrics)
    assert all(metric["display_value"] == f"{int(round(metric['value']))} day(s)" for metric in metrics)


def test_seasonal_map_active_early_dry_spell_products_are_not_universally_zero(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    generate = client.post("/seasonal-map/generate?theme=early_dry_spell&season_profile=southern_major&mode=seasonal")
    assert generate.status_code == 200

    active = client.get("/seasonal-map/deterministic/active?theme=early_dry_spell&season_profile=southern_major&mode=seasonal")
    assert active.status_code == 200
    payload = active.json()
    district_values = [item["metric"]["value"] for item in payload["district_items"]]
    region_values = [item["metric"]["value"] for item in payload["region_items"]]

    assert district_values
    assert region_values
    assert any(value > 0 for value in district_values)
    assert any(value > 0 for value in region_values)
    assert not all(value == 0 for value in district_values)


def test_seasonal_map_active_rainy_day_products_serve_whole_day_display_values(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    generate = client.post("/seasonal-map/generate?theme=rainy_days&season_profile=southern_major&mode=calendar&subseason=MAM")
    assert generate.status_code == 200

    active = client.get(
        "/seasonal-map/deterministic/active?theme=rainy_days&season_profile=southern_major&mode=calendar&subseason=MAM"
    )
    assert active.status_code == 200
    payload = active.json()
    metrics = [item["metric"] for item in payload["district_items"]] + [item["metric"] for item in payload["region_items"]]

    assert metrics
    assert all(metric["unit"] == "days" for metric in metrics)
    assert all(metric["display_value"] == f"{int(round(metric['value']))} day(s)" for metric in metrics)


def test_cessation_legend_and_metric_use_late_is_wet_color_semantics():
    get_settings.cache_clear()
    profile = get_settings().seasonal_map.profiles["southern_major"]

    legend = _build_legend("cessation", "seasonal", profile)
    metric_early = _classify_cessation(-10.0, profile, pd.Timestamp("2026-04-26T00:00:00Z").to_pydatetime())
    metric_late = _classify_cessation(10.0, profile, pd.Timestamp("2026-04-26T00:00:00Z").to_pydatetime())

    assert [item["color"] for item in legend] == ["#b47a34", "#b8b9b4", "#2f8f86"]
    assert metric_early["category_code"] == "early"
    assert metric_early["color"] == "#b47a34"
    assert metric_late["category_code"] == "late"
    assert metric_late["color"] == "#2f8f86"


def test_seasonal_map_refresh_endpoint_generates_full_valid_matrix(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)
    artifact_dir = tmp_path / "seasonal-map"

    response = client.post("/seasonal-map/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_source"] == "configured"
    assert payload["attempted_count"] == _expected_refresh_count()
    assert payload["succeeded_count"] == payload["attempted_count"]
    assert payload["failed_count"] == 0

    attempted = {
        (item["theme"], item["season_profile"], item["mode"], item["subseason"])
        for item in payload["attempted"]
    }
    assert ("onset", "northern_single", "seasonal", None) in attempted
    assert ("cessation", "southern_major", "seasonal", None) in attempted
    assert ("late_dry_spell", "southern_minor", "seasonal", None) in attempted
    assert ("rainfall_amount", "southern_major", "calendar", "MAM") in attempted
    assert ("rainy_days", "southern_minor", "calendar", "SON") in attempted
    assert ("onset", "southern_major", "calendar", "MAM") not in attempted
    assert ("rainfall_amount", "southern_minor", "seasonal", None) not in attempted
    assert ("rainy_days", "northern_single", "seasonal", None) not in attempted
    assert ("rainfall_amount", "southern_minor", "calendar", "MAM") not in attempted

    for item in payload["succeeded"]:
        assert Path(item["active_pointer_path"]).exists()
        if item["legacy_active_pointer_path"] is not None:
            assert Path(item["legacy_active_pointer_path"]).exists()

    assert (artifact_dir / "active_configured_northern_single_cessation_seasonal.json").exists()
    assert (artifact_dir / "active_configured_southern_major_early_dry_spell_seasonal.json").exists()
    assert (artifact_dir / "active_configured_southern_minor_late_dry_spell_seasonal.json").exists()
    assert (artifact_dir / "active_configured_northern_single_rainfall_amount_calendar_mjj.json").exists()
    assert (artifact_dir / "active_configured_southern_major_rainy_days_calendar_amj.json").exists()
    assert not (artifact_dir / "active_configured_southern_minor_rainfall_amount_seasonal.json").exists()

    refreshed_product = next(
        item
        for item in payload["succeeded"]
        if item["theme"] == "rainy_days"
        and item["season_profile"] == "southern_minor"
        and item["mode"] == "calendar"
        and item["subseason"] == "SON"
    )
    assert refreshed_product["active_pointer_path"].endswith(
        "active_configured_southern_minor_rainy_days_calendar_son.json"
    )
    active = client.get("/seasonal-map/active?theme=rainy_days&season_profile=southern_minor&mode=calendar&subseason=SON")
    assert active.status_code == 200
    assert active.json()["product_id"] == refreshed_product["product_id"]


def test_seasonal_map_refresh_endpoint_uses_default_source_and_honors_override(monkeypatch, tmp_path):
    paths = _prepare_public_serving_stack(monkeypatch, tmp_path)
    era5_path = tmp_path / "era5.nc"
    gfs_path = tmp_path / "gfs.nc"

    era5_dataset = xr.open_dataset(paths.forecast_path, engine="scipy")
    era5_dataset.to_netcdf(era5_path, engine="scipy")
    gfs_dataset = era5_dataset.copy(deep=True)
    gfs_dataset["tp"] = gfs_dataset["tp"] * 0.75
    gfs_dataset.to_netcdf(gfs_path, engine="scipy")
    era5_dataset.close()
    gfs_dataset.close()

    monkeypatch.setenv("CUMULUS_ERA5_FORECAST_PATH", str(era5_path))
    monkeypatch.setenv("CUMULUS_ERA5_FORECAST_ENGINE", "scipy")
    monkeypatch.setenv("CUMULUS_GFS_FORECAST_PATH", str(gfs_path))
    monkeypatch.setenv("CUMULUS_GFS_FORECAST_ENGINE", "scipy")
    monkeypatch.setenv("CUMULUS_DEFAULT_FORECAST_SOURCE", "era5")
    get_settings.cache_clear()
    clear_model_bundle_cache()

    client = TestClient(app)
    default_refresh = client.post("/seasonal-map/refresh?theme=onset&season_profile=northern_single")
    override_refresh = client.post("/seasonal-map/refresh?theme=onset&season_profile=northern_single&forecast_source=gfs")

    assert default_refresh.status_code == 200
    assert override_refresh.status_code == 200
    default_payload = default_refresh.json()
    override_payload = override_refresh.json()
    assert default_payload["forecast_source"] == "era5"
    assert override_payload["forecast_source"] == "gfs"
    assert default_payload["attempted_count"] == _expected_refresh_count(theme="onset", season_profile="northern_single")
    assert override_payload["attempted_count"] == _expected_refresh_count(theme="onset", season_profile="northern_single")

    active_era5 = client.get("/seasonal-map/active?theme=onset&season_profile=northern_single&mode=seasonal&forecast_source=era5")
    active_gfs = client.get("/seasonal-map/active?theme=onset&season_profile=northern_single&mode=seasonal&forecast_source=gfs")
    assert active_era5.status_code == 200
    assert active_gfs.status_code == 200
    assert active_era5.json()["forecast_source"] == "era5"
    assert active_gfs.json()["forecast_source"] == "gfs"
    assert active_era5.json()["product_id"] == default_payload["succeeded"][0]["product_id"]
    assert active_gfs.json()["product_id"] == override_payload["succeeded"][0]["product_id"]


@pytest.mark.parametrize("theme", ["onset", "cessation", "early_dry_spell", "late_dry_spell"])
def test_regime_bound_themes_only_return_in_profile_footprint_for_northern_profile(monkeypatch, tmp_path, theme):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)
    catalog = _district_catalog()
    north_ids = {district["location_id"] for district in catalog if district["regime_zone"] == "north"}
    south_ids = {district["location_id"] for district in catalog if district["regime_zone"] == "south"}
    region_totals = Counter(str(district["region_name"]) for district in catalog)
    north_region_totals = Counter(
        str(district["region_name"]) for district in catalog if district["regime_zone"] == "north"
    )
    mixed_region_name = next(
        region_name
        for region_name, count in north_region_totals.items()
        if count < region_totals[region_name]
    )

    generate = client.post(f"/seasonal-map/generate?theme={theme}&season_profile=northern_single&mode=seasonal")
    assert generate.status_code == 200
    manifest = generate.json()
    assert manifest["district_count"] == len(north_ids)
    assert manifest["region_count"] == len(north_region_totals)

    active = client.get(f"/seasonal-map/active?theme={theme}&season_profile=northern_single&mode=seasonal")
    assert active.status_code == 200
    payload = active.json()
    returned_ids = {item["location_id"] for item in payload["district_items"]}
    assert returned_ids == north_ids
    assert returned_ids.isdisjoint(south_ids)
    assert all("agro-ecological footprint" in item["coverage_note"] for item in payload["district_items"][:5])

    mixed_region = next(item for item in payload["region_items"] if item["geography_name"] == mixed_region_name)
    assert mixed_region["coverage_count"] == north_region_totals[mixed_region_name]
    assert mixed_region["coverage_count"] < region_totals[mixed_region_name]
    assert "in-footprint district outputs" in mixed_region["coverage_note"]


@pytest.mark.parametrize(
    (
        "theme",
        "value",
        "moderate_days",
        "high_days",
        "expected_code",
        "expected_label",
        "expected_numeric_value",
        "expected_display_value",
        "expected_note",
    ),
    [
        (
            "early_dry_spell",
            4.0,
            5,
            8,
            "low",
            "Short",
            4,
            "4 day(s)",
            "Longest dry run from onset to day 50. Onset criterion uses cumulative rainfall >= 20 mm, dry-day threshold < 1 mm, end-search 15 May, and nbjour: 50. Short below 5 days; Near-Normal from 5 to under 8; Long from 8 days upward.",
        ),
        (
            "early_dry_spell",
            6.0,
            5,
            8,
            "moderate",
            "Near-Normal",
            6,
            "6 day(s)",
            "Longest dry run from onset to day 50. Onset criterion uses cumulative rainfall >= 20 mm, dry-day threshold < 1 mm, end-search 15 May, and nbjour: 50. Short below 5 days; Near-Normal from 5 to under 8; Long from 8 days upward.",
        ),
        (
            "late_dry_spell",
            9.0,
            6,
            9,
            "high",
            "Long",
            9,
            "9 day(s)",
            "Longest dry run from day 51 to cessation. Onset criterion uses cumulative rainfall >= 20 mm, dry-day threshold < 1 mm, end-search 15 May, and nbjour: 50. Short below 6 days; Near-Normal from 6 to under 9; Long from 9 days upward.",
        ),
        (
            "late_dry_spell",
            4.5,
            6,
            9,
            "low",
            "Short",
            5,
            "5 day(s)",
            "Longest dry run from day 51 to cessation. Onset criterion uses cumulative rainfall >= 20 mm, dry-day threshold < 1 mm, end-search 15 May, and nbjour: 50. Short below 6 days; Near-Normal from 6 to under 9; Long from 9 days upward.",
        ),
    ],
)
def test_dry_spell_classification_uses_short_normal_long_labels(
    theme,
    value,
    moderate_days,
    high_days,
    expected_code,
    expected_label,
    expected_numeric_value,
    expected_display_value,
    expected_note,
):
    metric = _classify_dry_spell(
        theme=theme,
        value=value,
        moderate_days=moderate_days,
        high_days=high_days,
    )

    assert metric["category_code"] == expected_code
    assert metric["category_label"] == expected_label
    assert metric["numeric_value"] == expected_numeric_value
    assert metric["display_value"] == expected_display_value
    assert metric["criteria_note"] == expected_note


@pytest.mark.parametrize("theme", ["early_dry_spell", "late_dry_spell"])
def test_dry_spell_legend_uses_short_normal_long_labels(theme):
    get_settings.cache_clear()
    profile = get_settings().seasonal_map.profiles["southern_major"]

    legend = _build_legend(theme, "seasonal", profile)

    assert [item["category_code"] for item in legend] == ["low", "moderate", "high"]
    assert [item["label"] for item in legend] == ["Short", "Near-Normal", "Long"]


@pytest.mark.parametrize("season_profile", ["southern_major", "southern_minor"])
def test_regime_bound_onset_excludes_northern_districts_for_southern_profiles(monkeypatch, tmp_path, season_profile):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)
    catalog = _district_catalog()
    south_ids = {district["location_id"] for district in catalog if district["regime_zone"] == "south"}
    north_ids = {district["location_id"] for district in catalog if district["regime_zone"] == "north"}

    generate = client.post(f"/seasonal-map/generate?theme=onset&season_profile={season_profile}&mode=seasonal")
    assert generate.status_code == 200
    assert generate.json()["district_count"] == len(south_ids)

    active = client.get(f"/seasonal-map/active?theme=onset&season_profile={season_profile}&mode=seasonal")
    assert active.status_code == 200
    returned_ids = {item["location_id"] for item in active.json()["district_items"]}
    assert returned_ids == south_ids
    assert returned_ids.isdisjoint(north_ids)


def test_rainfall_and_rainy_day_products_remain_nationwide_in_seasonal_and_calendar_modes(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)
    catalog = _district_catalog()
    all_ids = {district["location_id"] for district in catalog}
    region_totals = Counter(str(district["region_name"]) for district in catalog)
    mixed_region_name = _mixed_region_name(catalog)

    seasonal_generate = client.post("/seasonal-map/generate?theme=rainfall_amount&season_profile=southern_minor&mode=seasonal")
    assert seasonal_generate.status_code == 200
    seasonal_manifest = seasonal_generate.json()
    assert seasonal_manifest["district_count"] == len(all_ids)
    assert seasonal_manifest["region_count"] == len(region_totals)

    seasonal_active = client.get("/seasonal-map/active?theme=rainfall_amount&season_profile=southern_minor&mode=seasonal")
    assert seasonal_active.status_code == 200
    seasonal_payload = seasonal_active.json()
    assert {item["location_id"] for item in seasonal_payload["district_items"]} == all_ids
    assert "remains nationwide" in seasonal_payload["district_items"][0]["coverage_note"]

    seasonal_mixed_region = next(
        item for item in seasonal_payload["region_items"] if item["geography_name"] == mixed_region_name
    )
    assert seasonal_mixed_region["coverage_count"] == region_totals[mixed_region_name]
    assert "aggregates all" in seasonal_mixed_region["coverage_note"]

    calendar_generate = client.post(
        "/seasonal-map/generate?theme=rainy_days&season_profile=southern_major&mode=calendar&subseason=MAM"
    )
    assert calendar_generate.status_code == 200
    calendar_manifest = calendar_generate.json()
    assert calendar_manifest["district_count"] == len(all_ids)
    assert calendar_manifest["region_count"] == len(region_totals)

    calendar_active = client.get(
        "/seasonal-map/active?theme=rainy_days&season_profile=southern_major&mode=calendar&subseason=MAM"
    )
    assert calendar_active.status_code == 200
    assert {item["location_id"] for item in calendar_active.json()["district_items"]} == all_ids


def test_seasonal_map_calendar_mode_requires_valid_subseason(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    missing = client.get("/seasonal-map/active?theme=rainfall_amount&season_profile=southern_major&mode=calendar")
    assert missing.status_code == 422
    assert missing.json()["error_code"] == "subseason_required"

    invalid = client.get(
        "/seasonal-map/active?theme=rainfall_amount&season_profile=southern_minor&mode=calendar&subseason=MAM"
    )
    assert invalid.status_code == 422
    assert invalid.json()["error_code"] == "invalid_subseason_for_profile"

    disallowed = client.get("/seasonal-map/active?theme=onset&season_profile=southern_major&mode=calendar&subseason=MAM")
    assert disallowed.status_code == 422
    assert disallowed.json()["error_code"] == "invalid_seasonal_mode"


def test_seasonal_map_calendar_generation_and_options_endpoint(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    generate = client.post(
        "/seasonal-map/generate?theme=rainfall_amount&season_profile=southern_major&mode=calendar&subseason=MAM"
    )
    assert generate.status_code == 200
    manifest = generate.json()
    assert manifest["mode"] == "calendar"
    assert manifest["subseason"] == "MAM"
    assert manifest["subseason_label"] == "MAM"

    active = client.get(
        "/seasonal-map/active?theme=rainfall_amount&season_profile=southern_major&mode=calendar&subseason=MAM"
    )
    assert active.status_code == 200
    payload = active.json()
    assert payload["mode"] == "calendar"
    assert payload["subseason"] == "MAM"
    assert payload["district_items"][0]["metric"]["theme"] == "rainfall_amount"

    options = client.get("/seasonal-map/options")
    assert options.status_code == 200
    options_payload = options.json()
    assert options_payload["themes"]["onset"]["modes"] == ["seasonal"]
    assert options_payload["themes"]["rainfall_amount"]["modes"] == ["seasonal", "calendar"]
    assert options_payload["profiles"]["northern_single"]["calendar_subseasons"] == ["MJJ", "JJA", "JAS"]


def test_seasonal_map_profiles_endpoint(monkeypatch, tmp_path):
    _prepare_public_serving_stack(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/seasonal-map/profiles")
    assert response.status_code == 200
    assert response.json() == ["northern_single", "southern_major", "southern_minor"]
