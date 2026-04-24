from __future__ import annotations

import pandas as pd
import xarray as xr
from fastapi.testclient import TestClient

from cumulus.main import app
from cumulus.modeling.predictor import clear_model_bundle_cache
from cumulus.modeling.trainer import train_baseline_model
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

    generate = client.post("/seasonal-map/generate?theme=onset&season_profile=northern_single&mode=seasonal")
    assert generate.status_code == 200
    manifest = generate.json()
    assert manifest["product_id"].startswith("seasonal_configured_northern_single_onset_seasonal_")
    assert manifest["forecast_source"] == "configured"
    assert manifest["theme"] == "onset"
    assert manifest["season_profile"] == "northern_single"
    assert manifest["mode"] == "seasonal"
    assert manifest["subseason"] is None
    assert manifest["district_count"] >= 200
    assert manifest["region_count"] >= 10

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
    assert payload["district_items"][0]["metric"]["theme"] == "onset"
    assert payload["district_items"][0]["metric"]["category_label"] in {"Early", "Normal", "Late"}
    assert payload["district_items"][0]["metric"]["criteria_note"].startswith("Detected from 15 Mar")
    assert payload["region_items"][0]["metric"]["theme"] == "onset"


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
