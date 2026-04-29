from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import xarray as xr

from cumulus.main import app
from cumulus.services import forecast_product_service
from cumulus.services.forecast_product_service import (
    _bilinear_sample_grid,
    _bilinear_sample_probability_grid,
    _district_zone_cell_mask_cached,
    _nearest_sample_grid,
    _nearest_sample_probability_grid,
    _tile_geometry_mask_cached,
)
from cumulus.settings import get_settings


REPO_FORECAST_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "artifacts" / "forecast_products"


def _configure_forecast_artifacts(monkeypatch, tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "forecast-products"
    for theme in ("onset", "early_dry_spell"):
        shutil.copytree(REPO_FORECAST_ARTIFACT_DIR / theme, artifact_dir / theme)
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    get_settings.cache_clear()
    return artifact_dir


def _configure_forecast_product_sources(monkeypatch, tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "forecast-products"
    daily_dir = tmp_path / "daily-corrected"
    final_dir = tmp_path / "final-products"
    daily_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)

    mam_det = final_dir / "Forecast_Det_PRCPMarAprMay_2026.nc"
    mam_prob = final_dir / "Forecast_Prob_PRCPMarAprMay_2026.nc"
    _write_product_pair(mam_det, mam_prob)
    _write_daily_member_files(daily_dir)

    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DAILY_CORRECTED_DIR", str(daily_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DERIVED_MIN_MEMBER_COUNT", "2")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DERIVED_MIN_COVERAGE_FRACTION", "0.8")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_SOURCES", "{}")
    monkeypatch.setenv(
        "CUMULUS_FORECAST_PRODUCTS__RAINFALL_TOTAL_SOURCES",
        json.dumps(
            {
                "MAM": {
                    "deterministic_path": str(mam_det),
                    "probability_path": str(mam_prob),
                    "forecast_year": 2026,
                    "title": "Seasonal Rainfall Total",
                }
            }
        ),
    )
    get_settings.cache_clear()
    return artifact_dir


def _write_product_pair(
    deterministic_path: Path,
    probability_path: Path,
    *,
    deterministic_values: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
) -> None:
    latitudes = np.asarray([5.5, 6.5], dtype=float) if latitudes is None else latitudes
    longitudes = np.asarray([-1.5, -0.5], dtype=float) if longitudes is None else longitudes
    valid_time = np.datetime64("2026-04-01T00:00:00")
    if deterministic_values is None:
        deterministic_values = np.asarray([[[120.0, 150.0], [180.0, 210.0]]], dtype=float)
    deterministic = xr.Dataset(
        {
            "forecast_deterministic": (
                ("T", "Y", "X"),
                deterministic_values,
            )
        },
        coords={"T": [valid_time], "Y": latitudes, "X": longitudes},
    )
    deterministic.to_netcdf(deterministic_path)
    deterministic.close()

    if probabilities is None:
        probabilities = np.asarray(
            [
                [[[0.2, 0.3], [0.1, 0.2]]],
                [[[0.5, 0.4], [0.2, 0.3]]],
                [[[0.3, 0.3], [0.7, 0.5]]],
            ],
            dtype=float,
        )
    probability = xr.Dataset(
        {"forecast_probability": (("probability", "T", "Y", "X"), probabilities)},
        coords={"probability": ["PB", "PN", "PA"], "T": [valid_time], "Y": latitudes, "X": longitudes},
    )
    probability.to_netcdf(probability_path)
    probability.close()


def _write_active_manifest(
    artifact_dir: Path,
    *,
    theme: str,
    view_mode: str,
    data_path: Path,
    selector: str,
    subseason: str | None = None,
    generation_backend: str = "bridge_generated",
) -> Path:
    product_dir = artifact_dir / theme / view_mode / selector
    product_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = product_dir / "active.json"
    manifest = {
        "product_id": f"{theme}_{view_mode}_{selector}_2026_test",
        "theme": theme,
        "view_mode": view_mode,
        "season_profile": None,
        "subseason": subseason,
        "title": "Test Product",
        "forecast_year": 2026,
        "generated_at": "2026-04-28T00:00:00+00:00",
        "source_label": "Cumulus Bridge Product",
        "source_run_id": f"{theme}_{view_mode}_{selector}_2026_test",
        "generation_backend": generation_backend,
        "source_artifact_type": "daily_wass2s_derived" if "daily_wass2s" in generation_backend else "final_netcdf",
        "refresh_interval_seconds": 1800,
        "freshness_threshold_hours": 18,
        "data_path": str(data_path),
        "preview_path": str(product_dir / "preview.png"),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_daily_member_files(
    daily_dir: Path,
    *,
    year: int = 2026,
    wet_start: str = "2026-02-01",
    wet_end: str = "2026-06-30",
    wet_amounts: tuple[float, float] = (5.0, 6.0),
) -> None:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    latitudes = np.asarray([5.5, 6.5, 9.5], dtype=float)
    longitudes = np.asarray([-1.5, -0.5], dtype=float)
    for member_index, wet_amount in enumerate(wet_amounts, start=1):
        values = np.zeros((len(dates), len(latitudes), len(longitudes)), dtype=float)
        wet_mask = (dates >= pd.Timestamp(wet_start)) & (dates <= pd.Timestamp(wet_end))
        values[wet_mask, :, :] = wet_amount
        dataset = xr.Dataset(
            {"corrected": (("T", "Y", "X"), values)},
            coords={"T": dates.to_numpy(), "Y": latitudes, "X": longitudes},
        )
        dataset.to_netcdf(daily_dir / f"forecast_member{member_index}_PRCP_JanIc.nc")
        dataset.close()


def test_forecast_product_options_return_all_target_themes_with_readiness(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/products/options")

    assert response.status_code == 200
    payload = response.json()
    themes = {item["theme"]: item for item in payload}

    assert list(themes) == [
        "onset",
        "early_dry_spell",
        "cessation",
        "late_dry_spell",
        "rainfall_amount",
        "rainy_days",
    ]
    assert themes["onset"]["enabled"] is True
    assert themes["onset"]["requires_season"] is True
    assert themes["onset"]["seasons"] == ["northern_single", "southern_major", "southern_minor"]
    assert themes["early_dry_spell"]["enabled"] is True
    assert themes["cessation"]["enabled"] is False
    assert themes["cessation"]["reason"] == "artifacts_not_generated"
    assert themes["cessation"]["seasons"] == []
    assert themes["rainfall_amount"]["requires_subseason"] is True
    assert themes["rainfall_amount"]["enabled"] is False
    assert themes["rainfall_amount"]["subseasons"] == []
    assert themes["rainy_days"]["enabled"] is False


def test_forecast_product_options_do_not_open_product_datasets(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)

    def fail_dataset_open(*args, **kwargs):
        raise AssertionError("product options should not open NetCDF datasets")

    def fail_dataset_validation(*args, **kwargs):
        raise AssertionError("product options should not run NetCDF validation")

    monkeypatch.setattr(forecast_product_service.xr, "open_dataset", fail_dataset_open)
    monkeypatch.setattr(forecast_product_service, "_validate_product_dataset_for_selection", fail_dataset_validation)
    client = TestClient(app)

    response = client.get("/forecast/products/options")

    assert response.status_code == 200
    themes = {item["theme"]: item for item in response.json()}
    assert themes["onset"]["enabled"] is True
    assert themes["early_dry_spell"]["enabled"] is True


def test_refresh_imports_final_rainfall_amount_products(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_product_sources(monkeypatch, tmp_path)
    client = TestClient(app)

    refresh = client.post("/forecast/products/refresh?theme=rainfall_amount")

    assert refresh.status_code == 200
    refresh_payload = refresh.json()
    assert refresh_payload["succeeded_count"] >= 2
    assert any("rainfall_amount_probability_mam" in item["product_id"] for item in refresh_payload["succeeded"])
    assert any("rainfall_amount_deterministic_mam" in item["product_id"] for item in refresh_payload["succeeded"])

    options = client.get("/forecast/products/options")
    assert options.status_code == 200
    rainfall_option = next(item for item in options.json() if item["theme"] == "rainfall_amount")
    assert rainfall_option["enabled"] is True
    assert "MAM" in rainfall_option["subseasons"]

    active_probability = client.get("/forecast/probability/active?theme=rainfall_amount&subseason=MAM")
    active_deterministic = client.get("/forecast/deterministic/active?theme=rainfall_amount&subseason=MAM")
    sample = client.get("/forecast/probability/sample?theme=rainfall_amount&subseason=MAM&latitude=5.6&longitude=-1.4")
    tile = client.get("/forecast/probability/tiles/6/31/29.png?theme=rainfall_amount&subseason=MAM")
    preview = client.get("/forecast/probability/preview.png?theme=rainfall_amount&subseason=MAM")

    assert active_probability.status_code == 200
    assert active_deterministic.status_code == 200
    assert sample.status_code == 200
    assert tile.status_code == 200
    assert tile.content.startswith(b"\x89PNG")
    assert preview.status_code == 200
    assert preview.content.startswith(b"\x89PNG")
    assert (artifact_dir / "rainfall_amount" / "probability" / "mam" / "active.json").exists()


def test_refresh_discovers_final_rainfall_products_from_configured_directories(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    daily_dir = tmp_path / "daily-corrected"
    final_dir = tmp_path / "final-products"
    daily_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    amj_det = final_dir / "Forecast_Det_PRCPAprMayJun_2026.nc"
    amj_prob = final_dir / "Forecast_Prob_PRCPAprMayJun_2026.nc"
    _write_product_pair(amj_det, amj_prob)
    _write_daily_member_files(daily_dir)

    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DAILY_CORRECTED_DIR", str(daily_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_SOURCES", "{}")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__RAINFALL_TOTAL_SOURCES", "{}")
    get_settings.cache_clear()
    client = TestClient(app)

    refresh = client.post("/forecast/products/refresh?theme=rainfall_amount")

    assert refresh.status_code == 200
    refresh_payload = refresh.json()
    assert any("rainfall_amount_probability_amj" in item["product_id"] for item in refresh_payload["succeeded"])
    assert any("rainfall_amount_deterministic_amj" in item["product_id"] for item in refresh_payload["succeeded"])
    options = client.get("/forecast/products/options")
    rainfall_option = next(item for item in options.json() if item["theme"] == "rainfall_amount")
    assert rainfall_option["subseasons"] == ["AMJ"]
    assert (artifact_dir / "rainfall_amount" / "probability" / "amj" / "active.json").exists()


def test_rainfall_amount_daily_derived_artifacts_are_not_reported_ready_without_final_pair(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    final_dir = tmp_path / "empty-final-products"
    final_dir.mkdir(parents=True)
    stale_dir = tmp_path / "stale-daily"
    stale_dir.mkdir(parents=True)
    stale_det = stale_dir / "Forecast_Det_rainfall_amount_amj_2026.nc"
    stale_prob = stale_dir / "Forecast_Prob_rainfall_amount_amj_2026.nc"
    _write_product_pair(stale_det, stale_prob)
    _write_active_manifest(
        artifact_dir,
        theme="rainfall_amount",
        view_mode="deterministic",
        data_path=stale_det,
        selector="amj",
        subseason="AMJ",
        generation_backend="bridge_generated_daily_wass2s",
    )
    _write_active_manifest(
        artifact_dir,
        theme="rainfall_amount",
        view_mode="probability",
        data_path=stale_prob,
        selector="amj",
        subseason="AMJ",
        generation_backend="bridge_generated_daily_wass2s",
    )
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    get_settings.cache_clear()
    client = TestClient(app)

    options = client.get("/forecast/products/options")
    active = client.get("/forecast/probability/active?theme=rainfall_amount&subseason=AMJ")

    assert options.status_code == 200
    rainfall_option = next(item for item in options.json() if item["theme"] == "rainfall_amount")
    assert rainfall_option["enabled"] is False
    assert rainfall_option["subseasons"] == []
    assert active.status_code == 503


def test_final_rainfall_products_override_stale_daily_derived_artifacts(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    final_dir = tmp_path / "final-products"
    stale_dir = tmp_path / "stale-daily"
    final_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)

    stale_det = stale_dir / "Forecast_Det_rainfall_amount_amj_2026.nc"
    stale_prob = stale_dir / "Forecast_Prob_rainfall_amount_amj_2026.nc"
    _write_product_pair(
        stale_det,
        stale_prob,
        deterministic_values=np.asarray([[[1.0, 1.0], [1.0, 1.0]]], dtype=float),
    )
    _write_active_manifest(
        artifact_dir,
        theme="rainfall_amount",
        view_mode="deterministic",
        data_path=stale_det,
        selector="amj",
        subseason="AMJ",
        generation_backend="bridge_generated_daily_wass2s",
    )
    _write_active_manifest(
        artifact_dir,
        theme="rainfall_amount",
        view_mode="probability",
        data_path=stale_prob,
        selector="amj",
        subseason="AMJ",
        generation_backend="bridge_generated_daily_wass2s",
    )

    final_det = final_dir / "Forecast_Det_PRCPAprMayJun_2026.nc"
    final_prob = final_dir / "Forecast_Prob_PRCPAprMayJun_2026.nc"
    _write_product_pair(
        final_det,
        final_prob,
        deterministic_values=np.asarray([[[300.0, 320.0], [340.0, 360.0]]], dtype=float),
    )
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    get_settings.cache_clear()
    client = TestClient(app)

    refresh = client.post("/forecast/products/refresh?theme=rainfall_amount")
    active = client.get("/forecast/deterministic/active?theme=rainfall_amount&subseason=AMJ")

    assert refresh.status_code == 200
    assert active.status_code == 200
    active_payload = active.json()
    assert active_payload["generation_backend"] == "bridge_generated_final_netcdf"
    assert active_payload["lower_bound"] == 300.0
    manifest = json.loads(
        (artifact_dir / "rainfall_amount" / "deterministic" / "amj" / "active.json").read_text(encoding="utf-8")
    )
    assert Path(manifest["data_path"]).name == "Forecast_Det_PRCPAprMayJun_2026.nc"


def test_final_season_products_are_materializable_and_override_daily_fallbacks(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    final_dir = tmp_path / "final-products"
    stale_dir = tmp_path / "stale-daily"
    final_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)

    final_pairs = {
        "cessation": (
            final_dir / "Forecast_Det_PRCPCessation_2026.nc",
            final_dir / "Forecast_Prob_PRCPCessation_2026.nc",
            300.0,
        ),
        "late_dry_spell": (
            final_dir / "Forecast_Det_PRCPdryspellcessation_2026.nc",
            final_dir / "Forecast_Prob_PRCPdryspellcessation_2026.nc",
            12.0,
        ),
    }
    for theme, (final_det, final_prob, base_value) in final_pairs.items():
        _write_product_pair(
            final_det,
            final_prob,
            deterministic_values=np.asarray(
                [[[base_value, base_value + 1.0], [base_value + 2.0, base_value + 3.0]]],
                dtype=float,
            ),
        )
        stale_det = stale_dir / f"Forecast_Det_{theme}_southern_major_2026.nc"
        stale_prob = stale_dir / f"Forecast_Prob_{theme}_southern_major_2026.nc"
        _write_product_pair(
            stale_det,
            stale_prob,
            deterministic_values=np.asarray([[[1.0, 1.0], [1.0, 1.0]]], dtype=float),
        )
        _write_active_manifest(
            artifact_dir,
            theme=theme,
            view_mode="deterministic",
            data_path=stale_det,
            selector="southern_major",
            generation_backend="bridge_generated_daily_wass2s",
        )
        _write_active_manifest(
            artifact_dir,
            theme=theme,
            view_mode="probability",
            data_path=stale_prob,
            selector="southern_major",
            generation_backend="bridge_generated_daily_wass2s",
        )

    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_SOURCES", "{}")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__RAINFALL_TOTAL_SOURCES", "{}")
    get_settings.cache_clear()
    client = TestClient(app)

    options = client.get("/forecast/products/options")

    assert options.status_code == 200
    option_lookup = {item["theme"]: item for item in options.json()}
    assert option_lookup["cessation"]["enabled"] is True
    assert option_lookup["cessation"]["seasons"] == ["northern_single", "southern_major", "southern_minor"]
    assert option_lookup["late_dry_spell"]["enabled"] is True
    assert option_lookup["late_dry_spell"]["seasons"] == ["northern_single", "southern_major", "southern_minor"]

    for theme, (_, _, base_value) in final_pairs.items():
        active = client.get(f"/forecast/deterministic/active?theme={theme}&season_profile=southern_major")
        assert active.status_code == 200
        payload = active.json()
        assert payload["generation_backend"] == "bridge_generated_final_netcdf"
        assert payload["source_artifact_type"] == "final_netcdf"
        assert payload["grid_shape"] == {"y": 2, "x": 2}
        assert payload["grid_resolution_degrees"] == {"latitude": 1.0, "longitude": 1.0}
        assert payload["is_low_resolution_fallback"] is False
        assert payload["lower_bound"] == base_value
        manifest = json.loads(
            (artifact_dir / theme / "deterministic" / "southern_major" / "active.json").read_text(encoding="utf-8")
        )
        assert Path(manifest["data_path"]).parent == artifact_dir / theme / "deterministic" / "southern_major"
        assert Path(manifest["source_path"]).parent == final_dir


def test_final_rainy_days_products_are_discovered_by_subseason(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    final_dir = tmp_path / "final-products"
    final_dir.mkdir(parents=True)
    det_path = final_dir / "Forecast_Det_rainy_days_mam_2026.nc"
    prob_path = final_dir / "Forecast_Prob_rainy_days_mam_2026.nc"
    _write_product_pair(det_path, prob_path)

    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_SOURCES", "{}")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__RAINFALL_TOTAL_SOURCES", "{}")
    get_settings.cache_clear()
    client = TestClient(app)

    options = client.get("/forecast/products/options")
    active = client.get("/forecast/probability/active?theme=rainy_days&subseason=MAM")

    assert options.status_code == 200
    rainy_days = next(item for item in options.json() if item["theme"] == "rainy_days")
    assert rainy_days["enabled"] is True
    assert rainy_days["subseasons"] == ["MAM"]
    assert active.status_code == 200
    payload = active.json()
    assert payload["source_artifact_type"] == "final_netcdf"
    assert payload["is_low_resolution_fallback"] is False
    assert (artifact_dir / "rainy_days" / "probability" / "mam" / "active.json").exists()


def test_subseason_final_products_are_masked_to_ghana_footprint(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    final_dir = tmp_path / "final-products"
    daily_dir = tmp_path / "daily-corrected"
    final_dir.mkdir(parents=True)
    daily_dir.mkdir(parents=True)
    det_path = final_dir / "Forecast_Det_PRCPAprMayJun_2026.nc"
    prob_path = final_dir / "Forecast_Prob_PRCPAprMayJun_2026.nc"
    _write_product_pair(
        det_path,
        prob_path,
        latitudes=np.asarray([5.5, 6.5], dtype=float),
        longitudes=np.asarray([-0.5, 10.0], dtype=float),
        deterministic_values=np.asarray([[[100.0, 999.0], [200.0, 999.0]]], dtype=float),
    )

    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DAILY_CORRECTED_DIR", str(daily_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_SOURCES", "{}")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__RAINFALL_TOTAL_SOURCES", "{}")
    get_settings.cache_clear()
    client = TestClient(app)

    refresh = client.post("/forecast/products/refresh?theme=rainfall_amount")
    active = client.get("/forecast/deterministic/active?theme=rainfall_amount&subseason=AMJ")

    assert refresh.status_code == 200
    assert active.status_code == 200
    payload = active.json()
    assert payload["lower_bound"] == 100.0
    assert payload["upper_bound"] == 200.0


def test_refresh_derives_daily_cessation_late_dry_spell_and_rainy_days(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_product_sources(monkeypatch, tmp_path)
    client = TestClient(app)

    for theme in ("cessation", "late_dry_spell", "rainy_days"):
        refresh = client.post(f"/forecast/products/refresh?theme={theme}")
        assert refresh.status_code == 200
        assert refresh.json()["succeeded_count"] >= 2

    checks = [
        ("/forecast/probability/active?theme=cessation&season_profile=southern_major", "cessation", "probability", "southern_major"),
        ("/forecast/deterministic/active?theme=late_dry_spell&season_profile=southern_major", "late_dry_spell", "deterministic", "southern_major"),
        ("/forecast/probability/active?theme=rainy_days&subseason=MAM", "rainy_days", "probability", "mam"),
    ]
    for url, theme, view_mode, selector in checks:
        response = client.get(url)
        assert response.status_code == 200
        manifest = artifact_dir / theme / view_mode / selector / "active.json"
        assert manifest.exists()

    probability_manifest = json.loads(
        (artifact_dir / "rainy_days" / "probability" / "mam" / "active.json").read_text(encoding="utf-8")
    )
    with xr.open_dataset(probability_manifest["data_path"]) as dataset:
        data_var = dataset[list(dataset.data_vars)[0]]
        assert data_var.dims == ("probability", "T", "Y", "X")
        assert data_var.coords["probability"].values.tolist() == ["PB", "PN", "PA"]
        probabilities = np.asarray(data_var.isel(T=0).values, dtype=float)
    finite_sum = np.nansum(probabilities, axis=0)
    valid = np.isfinite(probabilities).any(axis=0)
    assert np.isfinite(probabilities).any()
    assert np.allclose(finite_sum[valid], 1.0)


def test_refresh_derives_southern_minor_onset_from_daily_window(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    daily_dir = tmp_path / "daily-corrected"
    final_dir = tmp_path / "empty-final-products"
    daily_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    _write_daily_member_files(
        daily_dir,
        year=2026,
        wet_start="2026-08-24",
        wet_end="2026-09-30",
        wet_amounts=(8.0, 9.0),
    )

    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DAILY_CORRECTED_DIR", str(daily_dir))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DERIVED_MIN_MEMBER_COUNT", "2")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__DERIVED_MIN_COVERAGE_FRACTION", "0.8")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_DIRS", json.dumps([str(final_dir)]))
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__FINAL_PRODUCT_SOURCES", "{}")
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__RAINFALL_TOTAL_SOURCES", "{}")
    get_settings.cache_clear()
    client = TestClient(app)

    refresh = client.post("/forecast/products/refresh?theme=onset")
    active = client.get("/forecast/deterministic/active?theme=onset&season_profile=southern_minor")
    sample = client.get(
        "/forecast/deterministic/sample?theme=onset&season_profile=southern_minor&latitude=5.6&longitude=-1.4"
    )
    probability = client.get("/forecast/probability/active?theme=onset&season_profile=southern_minor")

    assert refresh.status_code == 200
    assert (artifact_dir / "onset" / "deterministic" / "southern_minor" / "active.json").exists()
    assert (artifact_dir / "onset" / "probability" / "southern_minor" / "active.json").exists()
    assert active.status_code == 200
    active_payload = active.json()
    assert active_payload["generation_backend"] == "bridge_generated_daily_wass2s"
    assert active_payload["source_artifact_type"] == "daily_wass2s_derived"
    assert active_payload["grid_shape"] == {"y": 3, "x": 2}
    assert active_payload["is_low_resolution_fallback"] is True
    assert active_payload["forecast_year"] == 2026
    assert active_payload["lower_bound"] == 236.0
    assert active_payload["upper_bound"] == 236.0
    assert sample.status_code == 200
    assert sample.json()["display_value"] == "24 Aug"
    assert probability.status_code == 200


def test_invalid_probability_artifacts_are_listed_in_options_but_rejected_on_use(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    product_dir = tmp_path / "invalid-product"
    product_dir.mkdir(parents=True)
    det_path = product_dir / "Forecast_Det_rainy_days_mam_2026.nc"
    prob_path = product_dir / "Forecast_Prob_rainy_days_mam_2026.nc"
    invalid_probabilities = np.asarray(
        [
            [[[0.6, 0.6], [0.6, 0.6]]],
            [[[0.6, 0.6], [0.6, 0.6]]],
            [[[0.6, 0.6], [0.6, 0.6]]],
        ],
        dtype=float,
    )
    _write_product_pair(det_path, prob_path, probabilities=invalid_probabilities)
    _write_active_manifest(
        artifact_dir,
        theme="rainy_days",
        view_mode="deterministic",
        data_path=det_path,
        selector="mam",
        subseason="MAM",
    )
    _write_active_manifest(
        artifact_dir,
        theme="rainy_days",
        view_mode="probability",
        data_path=prob_path,
        selector="mam",
        subseason="MAM",
    )
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    get_settings.cache_clear()
    client = TestClient(app)

    options = client.get("/forecast/products/options")
    active = client.get("/forecast/probability/active?theme=rainy_days&subseason=MAM")

    assert options.status_code == 200
    rainy_days = next(item for item in options.json() if item["theme"] == "rainy_days")
    assert rainy_days["enabled"] is True
    assert rainy_days["subseasons"] == ["MAM"]
    assert active.status_code == 503
    assert active.json()["error_code"] == "forecast_product_incomplete"


def test_active_probability_product_returns_absolute_asset_urls(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/probability/active?theme=onset&season_profile=northern_single")

    assert response.status_code == 200
    payload = response.json()
    assert payload["theme"] == "onset"
    assert payload["season_profile"] == "northern_single"
    assert payload["season_label"] == "Northern Uni Modal Seasonal"
    assert payload["tile_url"].startswith("http://testserver/forecast/probability/tiles/{z}/{x}/{y}.png?")
    assert "theme=onset" in payload["tile_url"]
    assert "season_profile=northern_single" in payload["tile_url"]
    assert payload["preview_url"].startswith("http://testserver/forecast/probability/preview.png?")
    assert "season_profile=northern_single" in payload["preview_url"]


def test_scoped_probability_product_requires_published_manifest_even_when_generic_exists(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_artifacts(monkeypatch, tmp_path)
    shutil.rmtree(artifact_dir / "onset" / "probability" / "southern_minor")
    client = TestClient(app)

    response = client.get("/forecast/probability/active?theme=onset&season_profile=southern_minor")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "forecast_product_artifacts_not_available"
    assert "theme=onset" in payload["detail"]
    assert "view_mode=probability" in payload["detail"]
    assert "season_profile=southern_minor" in payload["detail"]
    assert not (artifact_dir / "onset" / "probability" / "southern_minor" / "active.json").exists()


def test_southern_minor_onset_rejects_generic_deterministic_manifest(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "forecast-products"
    generic_dir = tmp_path / "generic-onset"
    generic_dir.mkdir(parents=True)
    det_path = generic_dir / "Forecast_Det_PRCPOnset_2025.nc"
    prob_path = generic_dir / "Forecast_Prob_PRCPOnset_2025.nc"
    _write_product_pair(det_path, prob_path)
    _write_active_manifest(
        artifact_dir,
        theme="onset",
        view_mode="deterministic",
        data_path=det_path,
        selector="southern_minor",
    )
    monkeypatch.setenv("CUMULUS_FORECAST_PRODUCTS__ARTIFACT_DIR", str(artifact_dir))
    get_settings.cache_clear()
    client = TestClient(app)

    response = client.get("/forecast/deterministic/active?theme=onset&season_profile=southern_minor")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "forecast_product_artifacts_not_available"
    assert "theme=onset" in payload["detail"]
    assert "view_mode=deterministic" in payload["detail"]
    assert "season_profile=southern_minor" in payload["detail"]


def test_scoped_probability_sample_tile_and_preview_require_published_manifest(monkeypatch, tmp_path):
    artifact_dir = _configure_forecast_artifacts(monkeypatch, tmp_path)
    shutil.rmtree(artifact_dir / "onset" / "probability" / "southern_minor")
    client = TestClient(app)

    responses = [
        client.get(
            "/forecast/probability/sample?theme=onset&season_profile=southern_minor&latitude=5.6037&longitude=-0.187"
        ),
        client.get("/forecast/probability/tiles/6/31/29.png?theme=onset&season_profile=southern_minor"),
        client.get("/forecast/probability/preview.png?theme=onset&season_profile=southern_minor"),
    ]

    for response in responses:
        assert response.status_code == 503
        payload = response.json()
        assert payload["error_code"] == "forecast_product_artifacts_not_available"
        assert "view_mode=probability" in payload["detail"]
        assert "season_profile=southern_minor" in payload["detail"]
    assert not (artifact_dir / "onset" / "probability" / "southern_minor" / "active.json").exists()


def test_bilinear_deterministic_tile_sampling_interpolates_grid_values():
    values = np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=float)
    sampled, lat_inside, lon_inside = _bilinear_sample_grid(
        values,
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.5], dtype=float),
        np.asarray([0.5], dtype=float),
    )

    assert lat_inside.tolist() == [True]
    assert lon_inside.tolist() == [True]
    assert sampled.shape == (1, 1)
    assert sampled[0, 0] == 15.0


def test_bilinear_probability_tile_sampling_renormalizes_categories():
    probabilities = np.asarray(
        [
            [[0.4, 0.4], [0.4, 0.4]],
            [[0.4, 0.4], [0.4, 0.4]],
            [[0.8, 0.8], [0.8, 0.8]],
        ],
        dtype=float,
    )

    sampled, lat_inside, lon_inside = _bilinear_sample_probability_grid(
        probabilities,
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.5], dtype=float),
        np.asarray([0.5], dtype=float),
    )

    assert lat_inside.tolist() == [True]
    assert lon_inside.tolist() == [True]
    assert sampled.shape == (3, 1, 1)
    assert np.allclose(sampled[:, 0, 0], [0.25, 0.25, 0.5])
    assert np.isclose(float(np.sum(sampled[:, 0, 0])), 1.0)


def test_nearest_deterministic_tile_sampling_preserves_grid_cells():
    values = np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=float)
    sampled, lat_inside, lon_inside = _nearest_sample_grid(
        values,
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.49, 0.51], dtype=float),
        np.asarray([0.49, 0.51], dtype=float),
    )

    assert lat_inside.tolist() == [True, True]
    assert lon_inside.tolist() == [True, True]
    assert sampled.tolist() == [[0.0, 10.0], [20.0, 30.0]]


def test_nearest_probability_tile_sampling_preserves_dominant_cells_and_normalizes():
    probabilities = np.asarray(
        [
            [[4.0, 1.0], [1.0, 1.0]],
            [[1.0, 4.0], [1.0, 1.0]],
            [[1.0, 1.0], [4.0, 1.0]],
        ],
        dtype=float,
    )

    sampled, lat_inside, lon_inside = _nearest_sample_probability_grid(
        probabilities,
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.0, 1.0], dtype=float),
        np.asarray([0.49, 0.51], dtype=float),
        np.asarray([0.49, 0.51], dtype=float),
    )

    assert lat_inside.tolist() == [True, True]
    assert lon_inside.tolist() == [True, True]
    assert sampled.shape == (3, 2, 2)
    assert np.argmax(sampled, axis=0).tolist() == [[0, 1], [2, 0]]
    assert np.allclose(np.sum(sampled, axis=0), np.ones((2, 2)))


def test_zone_mask_keeps_raster_cells_that_intersect_district_boundary(tmp_path):
    geojson_path = tmp_path / "districts.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"display_name": "Boundary district", "region": "Boundary"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0.2, 7.95],
                                    [0.8, 7.95],
                                    [0.8, 8.04],
                                    [0.2, 8.04],
                                    [0.2, 7.95],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    mask = _district_zone_cell_mask_cached(str(geojson_path), 8.0, "south", (7.95, 8.05, 8.15), (0.5,))

    assert mask == ((True,), (True,), (False,))


def test_zone_mask_keeps_transition_regions_in_southern_product_extent(tmp_path):
    geojson_path = tmp_path / "districts.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"display_name": "Bono boundary", "region": "Bono"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-2.6, 8.3],
                                    [-2.2, 8.3],
                                    [-2.2, 8.5],
                                    [-2.6, 8.5],
                                    [-2.6, 8.3],
                                ]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"display_name": "Northern boundary", "region": "Northern"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-1.0, 8.3],
                                    [-0.6, 8.3],
                                    [-0.6, 8.5],
                                    [-1.0, 8.5],
                                    [-1.0, 8.3],
                                ]
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    south_mask = _district_zone_cell_mask_cached(str(geojson_path), 8.0, "south", (8.4,), (-2.4, -0.8))
    north_mask = _district_zone_cell_mask_cached(str(geojson_path), 8.0, "north", (8.4,), (-2.4, -0.8))

    assert south_mask == ((True, False),)
    assert north_mask == ((False, True),)


def test_tile_geometry_mask_clips_to_ghana_feature_extent(tmp_path):
    geojson_path = tmp_path / "districts.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"display_name": "Coastal district", "region": "Boundary"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [-1.0, 5.8],
                                    [-0.2, 5.8],
                                    [-0.2, 6.4],
                                    [-1.0, 6.4],
                                    [-1.0, 5.8],
                                ]
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    mask = _tile_geometry_mask_cached(str(geojson_path), 8.0, "ghana", 6, 31, 30)
    flat_mask = [value for row in mask for value in row]

    assert any(flat_mask)
    assert not all(flat_mask)


def test_missing_generated_product_combination_returns_unavailable_error(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/deterministic/active?theme=cessation&season_profile=southern_major")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error_code"] == "forecast_product_artifacts_not_available"
    assert "season_profile=southern_major" in payload["detail"]


def test_probability_sample_endpoint_honors_season_selector(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get(
        "/forecast/probability/sample?theme=onset&season_profile=southern_major&latitude=5.6037&longitude=-0.187"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["theme"] == "onset"
    assert payload["season_profile"] == "southern_major"
    assert payload["season_label"] == "Southern Major Season"
    assert payload["subseason"] is None
    assert payload["nearest_latitude"] is not None
    assert payload["nearest_longitude"] is not None


def test_subseason_theme_requires_subseason_selector(monkeypatch, tmp_path):
    _configure_forecast_artifacts(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/forecast/probability/active?theme=rainfall_amount")

    assert response.status_code == 422
    assert response.json()["error_code"] == "invalid_forecast_product_selection"
