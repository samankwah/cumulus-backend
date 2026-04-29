"""Application settings."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cumulus.data.source_manifests import (
    classify_data_origin,
    discover_default_station_path,
    discover_source_config,
    ensure_raw_data_layout,
)
from cumulus.utils.io import ensure_directory, load_yaml


BACKEND_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BACKEND_ROOT.parent
ML_ROOT = WORKSPACE_ROOT / "ml"
DEFAULT_CONFIG_DIR = BACKEND_ROOT / "configs"
DEFAULT_DATA_DIR = ML_ROOT / "data"
DEFAULT_NATIONWIDE_ARTIFACT_DIR = BACKEND_ROOT / "data" / "artifacts" / "nationwide"
DEFAULT_SEASONAL_MAP_ARTIFACT_DIR = BACKEND_ROOT / "data" / "artifacts" / "seasonal_map"
DEFAULT_FORECAST_PRODUCT_ARTIFACT_DIR = BACKEND_ROOT / "data" / "artifacts" / "forecast_products"
DEFAULT_DISTRICT_GEOJSON_PATH = WORKSPACE_ROOT / "frontend" / "public" / "data" / "ghana_district_polygons_simplified.geojson"
DEFAULT_WASS2S_2026_ROOT = Path.home() / "Desktop" / "MEST_projects" / "wass2s" / "Agro_PRESAGG_2026_ic_1"
DEFAULT_FORECAST_PRODUCT_SOURCE_DIR = DEFAULT_WASS2S_2026_ROOT / "forecasts"
DEFAULT_FORECAST_PRODUCT_DAILY_CORRECTED_DIR = DEFAULT_WASS2S_2026_ROOT / "daily_model_data" / "corrected"


class BiasCorrectionConfig(BaseModel):
    method: str = "quantile_mapping"
    calibration_min_samples: int = 200
    quantile_count: int = 99


class AdvisoryConfig(BaseModel):
    dry_day_threshold_mm: float = 1.0
    dry_spell_days: int = 7
    onset_window_days: int = 3
    onset_threshold_mm: float = 20.0
    onset_guard_days: int = 10
    onset_guard_dry_days: int = 7
    cessation_window_days: int = 14
    cessation_threshold_mm: float = 10.0
    cumulative_windows_days: list[int] = Field(default_factory=lambda: [7, 14])
    farmer: "FarmerAdvisoryConfig" = Field(default_factory=lambda: FarmerAdvisoryConfig())


class FarmerAdvisoryConfig(BaseModel):
    crop_name: str = "maize"
    planting_window_days: int = 3
    planting_rain_threshold_mm: float = 20.0
    planting_guard_days: int = 10
    planting_guard_dry_days: int = 7
    dry_spell_watch_days: int = 5
    dry_spell_warning_days: int = 7
    irrigation_window_days: int = 5
    irrigation_target_rain_mm: float = 15.0
    irrigation_severe_deficit_mm: float = 8.0
    hot_temperature_c: float = 32.0
    high_stress_temperature_c: float = 35.0


class TrainSplitConfig(BaseModel):
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


class GhanaBoundsConfig(BaseModel):
    latitude_min: float = 4.5
    latitude_max: float = 11.5
    longitude_min: float = -3.5
    longitude_max: float = 1.5


class DataPipelineConfig(BaseModel):
    ghana_bounds: GhanaBoundsConfig = Field(default_factory=GhanaBoundsConfig)
    daily_frequency: str = "D"
    station_target_column: str = "rainfall_mm"
    xarray_chunks: dict[str, int] = Field(default_factory=dict)
    target_resolution_km: float = 4.0


class ForecastSourceConfig(BaseModel):
    path: Path | None = None
    engine: str | None = None
    variables: list[str] = Field(default_factory=lambda: ["tp", "u10", "v10", "t2m"])
    variable_aliases: dict[str, str] = Field(default_factory=dict)
    source_run_id: str | None = None
    national_calibration_version: str = "national_backbone_v1"
    regional_calibration_versions: dict[str, str] = Field(default_factory=dict)
    manifest_path: Path | None = None
    data_origin: str | None = None


class NationwideConfig(BaseModel):
    artifact_dir: Path = DEFAULT_NATIONWIDE_ARTIFACT_DIR
    default_page_size: int = 50
    max_page_size: int = 250
    model_strategy: str = "hybrid_global_backbone"
    regionalization_column: str = "agro_ecological_zone"
    known_location_tolerance_degrees: float = 0.15


class SeasonalProfileConfig(BaseModel):
    label: str
    native_zone: str
    onset_search_start_month: int
    onset_search_start_day: int
    onset_reference_month: int
    onset_reference_day: int
    onset_threshold_mm: float = 20.0
    onset_window_days: int = 3
    onset_requires_consecutive_days: bool = False
    onset_guard_window_days: int = 30
    onset_guard_max_dry_spell_days: int = 10
    onset_normal_band_days: int = 10
    cessation_search_start_month: int
    cessation_search_start_day: int
    cessation_reference_month: int
    cessation_reference_day: int
    cessation_soil_water_mm: float = 70.0
    cessation_et_mm_per_day: float = 4.0
    cessation_normal_band_days: int = 10
    rainfall_normal_mm: float
    rainy_days_normal: float
    calendar_subseasons: list[str] = Field(default_factory=list)
    calendar_rainfall_normals_mm: dict[str, float] = Field(default_factory=dict)
    calendar_rainy_days_normals: dict[str, float] = Field(default_factory=dict)
    rainfall_band_pct: float = 15.0
    rainy_days_band: float = 4.0
    early_dry_spell_moderate_days: int = 5
    early_dry_spell_high_days: int = 8
    late_dry_spell_moderate_days: int = 6
    late_dry_spell_high_days: int = 9
    rainfall_factor: float = 1.0
    rainy_days_factor: float = 1.0


class SeasonalMapConfig(BaseModel):
    artifact_dir: Path = DEFAULT_SEASONAL_MAP_ARTIFACT_DIR
    district_geojson_path: Path = DEFAULT_DISTRICT_GEOJSON_PATH
    refresh_interval_minutes: int = 30
    freshness_threshold_hours: int = 18
    rainy_day_threshold_mm: float = 1.0
    dry_day_threshold_mm: float = 1.0
    northern_latitude_threshold: float = 8.0
    profiles: dict[str, SeasonalProfileConfig] = Field(default_factory=dict)


class ForecastProductSourceConfig(BaseModel):
    theme: str
    deterministic_path: Path | None = None
    probability_path: Path | None = None
    preview_path: Path | None = None
    forecast_year: int
    title: str


class ForecastProductPairSourceConfig(BaseModel):
    deterministic_path: Path
    probability_path: Path
    forecast_year: int = 2026
    title: str | None = None


class ForecastProductConfig(BaseModel):
    artifact_dir: Path = DEFAULT_FORECAST_PRODUCT_ARTIFACT_DIR
    refresh_interval_seconds: int = 1800
    freshness_threshold_hours: int = 18
    source_label: str = "Cumulus Bridge Product"
    generation_backend: str = "bridge_generated"
    daily_corrected_dir: Path = DEFAULT_FORECAST_PRODUCT_DAILY_CORRECTED_DIR
    daily_forecast_glob: str = "forecast_*_PRCP_*Ic.nc"
    derived_min_member_count: int = 2
    derived_min_coverage_fraction: float = 0.85
    final_product_dirs: list[Path] = Field(
        default_factory=lambda: [
            Path("E:/"),
            Path("E:/obj_files"),
            DEFAULT_FORECAST_PRODUCT_SOURCE_DIR,
        ]
    )
    final_product_sources: dict[str, dict[str, ForecastProductPairSourceConfig]] = Field(
        default_factory=lambda: {
            "rainfall_amount": {
                "MAM": ForecastProductPairSourceConfig(
                    deterministic_path=Path("E:/Forecast_Det_PRCPMarAprMay_2026.nc"),
                    probability_path=Path("E:/Forecast_Prob_PRCPMarAprMay_2026.nc"),
                    forecast_year=2026,
                    title="Seasonal Rainfall Total",
                ),
                "AMJ": ForecastProductPairSourceConfig(
                    deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPAprMayJun_2026.nc"),
                    probability_path=Path("E:/obj_files/Forecast_Prob_PRCPAprMayJun_2026.nc"),
                    forecast_year=2026,
                    title="Seasonal Rainfall Total",
                ),
                "MJJ": ForecastProductPairSourceConfig(
                    deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPMayJunJul_2026.nc"),
                    probability_path=Path("E:/obj_files/Forecast_Prob_PRCPMayJunJul_2026.nc"),
                    forecast_year=2026,
                    title="Seasonal Rainfall Total",
                ),
                "JJA": ForecastProductPairSourceConfig(
                    deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPJunJulAug_2026.nc"),
                    probability_path=Path("E:/obj_files/Forecast_Prob_PRCPJunJulAug_2026.nc"),
                    forecast_year=2026,
                    title="Seasonal Rainfall Total",
                ),
                "JAS": ForecastProductPairSourceConfig(
                    deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPJulAugSep_2026.nc"),
                    probability_path=Path("E:/obj_files/Forecast_Prob_PRCPJulAugSep_2026.nc"),
                    forecast_year=2026,
                    title="Seasonal Rainfall Total",
                ),
                "SON": ForecastProductPairSourceConfig(
                    deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPSepOctNov_2026.nc"),
                    probability_path=Path("E:/obj_files/Forecast_Prob_PRCPSepOctNov_2026.nc"),
                    forecast_year=2026,
                    title="Seasonal Rainfall Total",
                ),
            }
        }
    )
    rainfall_total_sources: dict[str, ForecastProductPairSourceConfig] = Field(
        default_factory=lambda: {
            "MAM": ForecastProductPairSourceConfig(
                deterministic_path=Path("E:/Forecast_Det_PRCPMarAprMay_2026.nc"),
                probability_path=Path("E:/Forecast_Prob_PRCPMarAprMay_2026.nc"),
                forecast_year=2026,
                title="Seasonal Rainfall Total",
            ),
            "MJJ": ForecastProductPairSourceConfig(
                deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPMayJunJul_2026.nc"),
                probability_path=Path("E:/obj_files/Forecast_Prob_PRCPMayJunJul_2026.nc"),
                forecast_year=2026,
                title="Seasonal Rainfall Total",
            ),
            "JAS": ForecastProductPairSourceConfig(
                deterministic_path=Path("E:/obj_files/Forecast_Det_PRCPJulAugSep_2026.nc"),
                probability_path=Path("E:/obj_files/Forecast_Prob_PRCPJulAugSep_2026.nc"),
                forecast_year=2026,
                title="Seasonal Rainfall Total",
            ),
        }
    )
    products: dict[str, ForecastProductSourceConfig] = Field(
        default_factory=lambda: {
            "onset": ForecastProductSourceConfig(
                theme="onset",
                deterministic_path=DEFAULT_FORECAST_PRODUCT_SOURCE_DIR / "Forecast_Det_PRCPOnset_2025.nc",
                probability_path=DEFAULT_FORECAST_PRODUCT_SOURCE_DIR / "Forecast_Prob_PRCPOnset_2025.nc",
                preview_path=DEFAULT_FORECAST_PRODUCT_SOURCE_DIR / "Consolidated Forecast Onset-2025.png",
                forecast_year=2025,
                title="Onset Date",
            ),
            "early_dry_spell": ForecastProductSourceConfig(
                theme="early_dry_spell",
                deterministic_path=DEFAULT_FORECAST_PRODUCT_SOURCE_DIR / "Forecast_Det_PRCPdryspellonset_2025.nc",
                probability_path=DEFAULT_FORECAST_PRODUCT_SOURCE_DIR / "Forecast_Prob_PRCPdryspellonset_2025.nc",
                preview_path=DEFAULT_FORECAST_PRODUCT_SOURCE_DIR / "Consolidated Forecast dryspellonset-2025.png",
                forecast_year=2025,
                title="Early-Season Dry Spell",
            ),
            "cessation": ForecastProductSourceConfig(
                theme="cessation",
                forecast_year=2026,
                title="Cessation Date",
            ),
            "late_dry_spell": ForecastProductSourceConfig(
                theme="late_dry_spell",
                forecast_year=2026,
                title="Late-Season Dry Spell",
            ),
            "rainfall_amount": ForecastProductSourceConfig(
                theme="rainfall_amount",
                forecast_year=2026,
                title="Seasonal Rainfall Total",
            ),
            "rainy_days": ForecastProductSourceConfig(
                theme="rainy_days",
                forecast_year=2026,
                title="Number of Rainy Days",
            ),
        }
    )


class ModelParamsConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int | None = 16
    min_samples_leaf: int = 5
    n_jobs: int = 1
    random_state: int = 42


class ModelConfig(BaseModel):
    target_column: str = "rainfall_mm"
    estimator: str = "random_forest"
    params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CUMULUS_",
        env_nested_delimiter="__",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    config_dir: Path = DEFAULT_CONFIG_DIR
    data_dir: Path = DEFAULT_DATA_DIR
    raw_data_dir: Path = Path("data/raw")
    model_artifact_dir: Path = Path("data/artifacts/models")
    bias_artifact_dir: Path = Path("data/artifacts/bias")
    evaluation_dir: Path = Path("data/artifacts/evaluation")
    project_name: str = "cumulus"
    api_version: str = "0.1.0"
    log_level: str = "INFO"
    time_zone: str = "UTC"
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
    )
    default_forecast_source: str | None = "era5"
    forecast_sources: dict[str, ForecastSourceConfig] = Field(default_factory=dict)
    era5_forecast_path: Path | None = None
    era5_forecast_engine: str | None = None
    era5_source_run_id: str | None = None
    gfs_forecast_path: Path | None = None
    gfs_forecast_engine: str | None = None
    gfs_source_run_id: str | None = None
    upstream_forecast_path: Path | None = None
    upstream_forecast_engine: str | None = None
    default_station_path: Path | None = None
    default_forecast_variables: list[str] = Field(default_factory=lambda: ["tp", "u10", "v10", "t2m"])
    forecast_horizon_days: int = 14
    max_forecast_horizon_days: int = 16
    data_pipeline: DataPipelineConfig = Field(default_factory=DataPipelineConfig)
    rainfall_variable_aliases: dict[str, str] = Field(default_factory=dict)
    feature_columns: list[str] = Field(default_factory=list)
    bias_correction: BiasCorrectionConfig = Field(default_factory=BiasCorrectionConfig)
    advisory: AdvisoryConfig = Field(default_factory=AdvisoryConfig)
    train: TrainSplitConfig = Field(default_factory=TrainSplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    nationwide: NationwideConfig = Field(default_factory=NationwideConfig)
    seasonal_map: SeasonalMapConfig = Field(default_factory=SeasonalMapConfig)
    forecast_products: ForecastProductConfig = Field(default_factory=ForecastProductConfig)

    @classmethod
    def from_sources(cls) -> "Settings":
        base = cls()
        payload: dict[str, Any] = base.model_dump()
        config_dir = base.config_dir
        payload.update(load_yaml(config_dir / "base.yaml"))
        model_yaml = load_yaml(config_dir / "model.yaml")
        if model_yaml:
            payload["model"] = model_yaml.get("model", model_yaml)
        advisory_yaml = load_yaml(config_dir / "advisory.yaml")
        if advisory_yaml:
            payload["advisory"] = advisory_yaml.get("advisory", advisory_yaml)
        seasonal_map_yaml = load_yaml(config_dir / "seasonal_map.yaml")
        if seasonal_map_yaml:
            payload["seasonal_map"] = seasonal_map_yaml.get("seasonal_map", seasonal_map_yaml)
        if "CUMULUS_DEFAULT_FORECAST_SOURCE" in os.environ:
            payload["default_forecast_source"] = os.environ["CUMULUS_DEFAULT_FORECAST_SOURCE"]
        payload["raw_data_dir"] = _resolve_raw_data_dir(payload)
        payload["model_artifact_dir"] = _resolve_artifact_dir(payload, "model_artifact_dir", Path("artifacts/models"))
        payload["bias_artifact_dir"] = _resolve_artifact_dir(payload, "bias_artifact_dir", Path("artifacts/bias"))
        payload["evaluation_dir"] = _resolve_artifact_dir(payload, "evaluation_dir", Path("artifacts/evaluation"))
        payload["default_station_path"] = _resolve_default_station_path(payload)
        payload["forecast_sources"] = _merge_forecast_sources(payload)
        if not payload.get("default_forecast_source") and payload["forecast_sources"]:
            payload["default_forecast_source"] = next(iter(payload["forecast_sources"]))
        settings = cls(**payload)
        ensure_raw_data_layout(settings.raw_data_dir)
        ensure_directory(settings.model_artifact_dir)
        ensure_directory(settings.bias_artifact_dir)
        ensure_directory(settings.evaluation_dir)
        ensure_directory(settings.nationwide.artifact_dir)
        ensure_directory(settings.seasonal_map.artifact_dir)
        ensure_directory(settings.forecast_products.artifact_dir)
        ensure_directory(settings.data_dir / "processed")
        return settings


def _merge_forecast_sources(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    configured_sources = payload.get("forecast_sources") or {}
    for source_id, config in configured_sources.items():
        merged[_normalize_source_id(source_id)] = dict(config)

    raw_data_dir = Path(payload.get("raw_data_dir") or Path(payload.get("data_dir") or "data") / "raw")
    for source_id in ("era5", "gfs"):
        discovered = discover_source_config(raw_data_dir, source_id)
        current = {**discovered, **merged.get(source_id, {})}
        path = payload.get(f"{source_id}_forecast_path")
        if path is not None:
            current["path"] = path
            current["engine"] = payload.get(f"{source_id}_forecast_engine") or current.get("engine")
            current["source_run_id"] = payload.get(f"{source_id}_source_run_id") or current.get("source_run_id")
            current["data_origin"] = classify_data_origin(path)
        merged[source_id] = current

    if not merged and payload.get("upstream_forecast_path") is not None:
        merged["configured"] = {
            "path": payload["upstream_forecast_path"],
            "engine": payload.get("upstream_forecast_engine"),
            "variables": payload.get("default_forecast_variables") or ["tp", "u10", "v10", "t2m"],
            "national_calibration_version": "national_backbone_v1",
            "regional_calibration_versions": {},
            "data_origin": classify_data_origin(payload["upstream_forecast_path"]),
        }
        payload["default_forecast_source"] = payload.get("default_forecast_source") or "configured"

    for source_id, config in merged.items():
        config.setdefault("variables", payload.get("default_forecast_variables") or ["tp", "u10", "v10", "t2m"])
        config.setdefault("variable_aliases", {})
        config.setdefault("national_calibration_version", "national_backbone_v1")
        config.setdefault("regional_calibration_versions", {})
        config.setdefault("source_run_id", f"{source_id}-run")
        config.setdefault("manifest_path", None)
        config["data_origin"] = classify_data_origin(config.get("path")) if config.get("path") else "missing"
    return merged


def _normalize_source_id(source_id: str) -> str:
    return str(source_id).strip().lower().replace(" ", "_")


def _resolve_default_station_path(payload: dict[str, Any]) -> Path | None:
    raw_data_dir = _resolve_raw_data_dir(payload)
    project_root = Path(payload.get("config_dir") or DEFAULT_CONFIG_DIR).parent
    configured_path = payload.get("default_station_path")
    return discover_default_station_path(raw_data_dir, project_root, configured_path)


def _resolve_raw_data_dir(payload: dict[str, Any]) -> Path:
    raw_data_dir = payload.get("raw_data_dir")
    data_dir = Path(payload.get("data_dir") or DEFAULT_DATA_DIR)
    if raw_data_dir in (None, "data/raw", Path("data/raw")):
        return data_dir / "raw"
    return Path(raw_data_dir)


def _resolve_artifact_dir(payload: dict[str, Any], field_name: str, suffix: Path) -> Path:
    artifact_dir = payload.get(field_name)
    default_value = Path("data") / suffix
    if artifact_dir in (None, default_value, Path(str(default_value))):
        return Path(payload.get("data_dir") or DEFAULT_DATA_DIR) / suffix
    return Path(artifact_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_sources()
