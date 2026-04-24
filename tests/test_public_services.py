from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import xarray as xr

from cumulus.schemas import PointRequest
from cumulus.services.advisory_service import generate_point_advisory
from cumulus.services.prediction_service import PointPredictionResult, predict_for_point
from cumulus.settings import get_settings


def test_prediction_service_reads_configured_source_and_extracts_one_point(monkeypatch, tmp_path):
    forecast_path = tmp_path / "forecast.nc"
    dataset = xr.Dataset(
        data_vars={
            "tp": (("time", "latitude", "longitude"), [[[0.010]], [[0.012]], [[0.005]]]),
            "t2m": (("time", "latitude", "longitude"), [[[301.15]], [[300.15]], [[299.15]]]),
            "u10": (("time", "latitude", "longitude"), [[[2.0]], [[2.0]], [[2.0]]]),
            "v10": (("time", "latitude", "longitude"), [[[1.0]], [[1.0]], [[1.0]]]),
        },
        coords={"time": pd.date_range("2026-04-24", periods=3, freq="D"), "latitude": [5.6], "longitude": [-0.18]},
    )
    dataset["tp"].attrs["units"] = "m"
    dataset["t2m"].attrs["units"] = "K"
    dataset.to_netcdf(forecast_path, engine="scipy")

    monkeypatch.setenv("CUMULUS_UPSTREAM_FORECAST_PATH", str(forecast_path))
    monkeypatch.setenv("CUMULUS_UPSTREAM_FORECAST_ENGINE", "scipy")
    get_settings.cache_clear()
    settings = get_settings()

    def fake_predict_dataframe(frame: pd.DataFrame, _settings, **_kwargs):
        predicted = frame.copy()
        predicted["rainfall_raw_mm"] = predicted["precip_mm"]
        predicted["rainfall_corrected_mm"] = predicted["precip_mm"] * 0.9
        return predicted, {"model_version": "test-model"}

    monkeypatch.setattr("cumulus.services.prediction_service.predict_dataframe", fake_predict_dataframe)

    result = predict_for_point(
        settings,
        PointRequest(latitude=5.6037, longitude=-0.1870, location_id="accra", horizon_days=2),
    )

    assert result.location_id == "accra"
    assert result.model_version == "test-model"
    assert result.forecast_source == "configured"
    assert result.data_origin == "downloaded_real_source_data"
    assert result.source_run_id.startswith("configured-")
    assert result.calibration_version.startswith("national_backbone_v1")
    assert result.spatial_resolution_km == 4.0
    assert result.horizon_days == 2
    assert len(result.forecast_frame) == 2
    assert result.forecast_frame["temp_c"].iloc[0] > 20.0
    assert result.forecast_frame["rainfall_corrected_mm"].iloc[0] > 0.0


def test_advisory_service_derives_farmer_advice_from_prediction(monkeypatch):
    get_settings.cache_clear()
    forecast_frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-04-24", periods=10, freq="D", tz="UTC"),
            "rainfall_corrected_mm": [8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "temp_c": [30.0] * 10,
        }
    )
    prediction = PointPredictionResult(
        location_id="tamale",
        latitude=9.4075,
        longitude=-0.8533,
        model_version="rf_rainfall_test",
        calibration_version="gfs_national_backbone_v1:quantile_mapping",
        generated_at=datetime.now(UTC),
        horizon_days=10,
        forecast_source="gfs",
        data_origin="downloaded_real_source_data",
        source_run_id="gfs-run-20260424",
        spatial_resolution_km=4.0,
        forecast_frame=forecast_frame,
    )

    monkeypatch.setattr("cumulus.services.advisory_service.predict_for_point", lambda settings, request: prediction)
    settings = get_settings()

    response = generate_point_advisory(
        settings,
        PointRequest(latitude=9.4075, longitude=-0.8533, location_id="tamale", horizon_days=10),
    )

    assert response.location_id == "tamale"
    assert response.model_version == "rf_rainfall_test"
    assert response.forecast_source == "gfs"
    assert response.data_origin == "downloaded_real_source_data"
    assert response.source_run_id == "gfs-run-20260424"
    assert response.spatial_resolution_km == 4.0
    assert response.planting_recommendation.headline
    assert response.dry_spell_alert.headline
    assert response.irrigation_advice.headline
