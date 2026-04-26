"""Pydantic schemas for the API."""

from __future__ import annotations

from typing import Any
from datetime import datetime, date

from pydantic import BaseModel, Field


class SchemaModel(BaseModel):
    model_config = {"protected_namespaces": ()}


class LocationRequest(SchemaModel):
    location_id: str
    lat: float = Field(alias="latitude")
    lon: float = Field(alias="longitude")

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


class ForecastSourceRequest(SchemaModel):
    name: str | None = None
    path: str | None = None
    engine: str | None = None
    source_run_id: str | None = None
    variables: list[str] = Field(default_factory=lambda: ["tp", "u10", "v10", "t2m"])


class ForecastRequest(SchemaModel):
    locations: list[LocationRequest]
    forecast_source: ForecastSourceRequest
    horizon_days: int = 14


class PointRequest(SchemaModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    location_id: str | None = None
    forecast_source: str | None = None
    horizon_days: int | None = Field(default=None, ge=1, le=60)


class TrainRequest(SchemaModel):
    merged_dataset_path: str | None = None
    station_path: str | None = None
    forecast_path: str | None = None
    forecast_source: str | None = None


class DailyForecastResponse(SchemaModel):
    date: date
    rainfall_raw_mm: float
    rainfall_corrected_mm: float
    temperature_c: float | None = None
    horizon_day: int | None = None


class SeasonalAdvisoryResponse(SchemaModel):
    onset_date: date | None = None
    cessation_date: date | None = None
    dry_spell_risk: bool
    dry_spell_length_days: int
    cum_rain_7d_mm: float
    cum_rain_14d_mm: float
    seasonal_cum_rain_mm: float
    reason: str


class AgroCharacteristicsResponse(SchemaModel):
    planting_window_signal: str
    dry_spell_risk: bool
    dry_spell_length_days: int
    irrigation_need_signal: str
    irrigation_deficit_mm: float | None = None
    onset_date: date | None = None
    cessation_date: date | None = None
    cum_rain_7d_mm: float
    cum_rain_14d_mm: float
    seasonal_cum_rain_mm: float


class ForecastResultResponse(SchemaModel):
    location_id: str
    lat: float
    lon: float
    spatial_resolution_km: float
    forecast_source: str
    data_origin: str | None = None
    source_run_id: str
    model_version: str
    calibration_version: str
    generated_at: datetime
    daily_forecast: list[DailyForecastResponse]
    agro_characteristics: AgroCharacteristicsResponse
    advisory: SeasonalAdvisoryResponse


class ForecastResponse(SchemaModel):
    model_version: str
    generated_at: datetime
    spatial_resolution_km: float | None = None
    forecast_source: str | None = None
    data_origin: str | None = None
    source_run_id: str | None = None
    calibration_version: str | None = None
    seasonal_refresh: SeasonalMapRefreshResponse | None = None
    results: list[ForecastResultResponse]


class PredictResponse(SchemaModel):
    location_id: str
    latitude: float
    longitude: float
    model_version: str
    calibration_version: str
    generated_at: datetime
    horizon_days: int
    forecast_source: str
    data_origin: str | None = None
    source_run_id: str
    spatial_resolution_km: float
    daily_forecast: list[DailyForecastResponse]
    agro_characteristics: AgroCharacteristicsResponse | None = None


class AdvisorySeriesItem(SchemaModel):
    date: date
    rainfall_mm: float


class LegacyAdvisoryRequest(SchemaModel):
    location_id: str | None = None
    rainfall_series: list[AdvisorySeriesItem]


class FarmerAdvisorySeriesItem(SchemaModel):
    date: date
    rainfall_mm: float
    temperature_c: float


class FarmerAdvisoryRequest(SchemaModel):
    location_id: str
    daily_forecast: list[FarmerAdvisorySeriesItem] = Field(min_length=1)


class FarmerAdvisoryItem(SchemaModel):
    level: str
    headline: str
    recommendation: str
    reason: str
    window_rainfall_mm: float | None = None
    rainfall_deficit_mm: float | None = None
    dry_spell_length_days: int | None = None
    avg_temperature_c: float | None = None
    temperature_band: str | None = None


class FarmerAdvisoryResponse(SchemaModel):
    location_id: str
    planting_recommendation: FarmerAdvisoryItem
    dry_spell_alert: FarmerAdvisoryItem
    irrigation_advice: FarmerAdvisoryItem


class PointAdvisoryResponse(SchemaModel):
    location_id: str
    latitude: float
    longitude: float
    forecast_source: str
    data_origin: str | None = None
    source_run_id: str
    spatial_resolution_km: float
    model_version: str
    calibration_version: str
    generated_at: datetime
    agro_characteristics: AgroCharacteristicsResponse
    planting_recommendation: FarmerAdvisoryItem
    dry_spell_alert: FarmerAdvisoryItem
    irrigation_advice: FarmerAdvisoryItem


class NationwideRunResponse(SchemaModel):
    run_id: str
    generated_at: datetime
    horizon_days: int
    source_run_id: str
    spatial_resolution_km: float
    model_version: str
    calibration_version: str
    model_strategy: str
    forecast_source: str
    data_origin: str | None = None
    location_count: int
    available_location_count: int
    failed_location_count: int
    region_count: int
    district_count: int


class NationwideLocationResponse(SchemaModel):
    location_id: str
    latitude: float
    longitude: float
    region: str | None = None
    district: str | None = None
    agro_ecological_zone: str | None = None
    is_serving_location: bool = True
    forecast_source: str
    data_origin: str | None = None
    source_run_id: str
    spatial_resolution_km: float
    model_version: str
    calibration_version: str
    generated_at: datetime
    horizon_days: int
    daily_forecast: list[DailyForecastResponse]
    agro_characteristics: AgroCharacteristicsResponse
    point_advisory: PointAdvisoryResponse


class NationwideLocationPageResponse(SchemaModel):
    run_id: str
    generated_at: datetime
    page: int
    page_size: int
    total_locations: int
    items: list[NationwideLocationResponse]


class AggregateAdvisoryItem(FarmerAdvisoryItem):
    severity_bucket: str
    available_location_count: int
    alert_count: int = 0
    watch_count: int = 0


class NationwideGeographySummaryResponse(SchemaModel):
    geography_type: str
    geography_name: str
    generated_at: datetime
    forecast_source: str
    data_origin: str | None = None
    source_run_id: str
    spatial_resolution_km: float
    model_version: str
    calibration_version: str
    model_strategy: str
    horizon_days: int
    location_count: int
    coverage_count: int
    daily_forecast: list[DailyForecastResponse]
    planting_recommendation: AggregateAdvisoryItem
    dry_spell_alert: AggregateAdvisoryItem
    irrigation_advice: AggregateAdvisoryItem
    note: str


class SeasonalLegendItemResponse(SchemaModel):
    category_code: str
    label: str
    hint: str
    color: str


class SeasonalThemeMetricResponse(SchemaModel):
    theme: str
    theme_label: str
    category_code: str
    category_label: str
    numeric_value: float | None = None
    display_value: str
    unit: str | None = None
    criteria_note: str
    interpretation: str
    color: str


class SeasonalMapAreaResponse(SchemaModel):
    location_id: str
    geography_type: str
    geography_name: str
    region_name: str
    coverage_count: int = 1
    coverage_note: str
    metric: SeasonalThemeMetricResponse


class SeasonalMapRunResponse(SchemaModel):
    product_id: str
    theme: str
    season_profile: str
    mode: str
    subseason: str | None = None
    mode_label: str
    subseason_label: str | None = None
    generated_at: datetime
    forecast_cycle: str
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    refresh_interval_seconds: int
    freshness_threshold_hours: int
    district_count: int
    region_count: int


class SeasonalMapProductResponse(SeasonalMapRunResponse):
    refresh_status: str
    is_stale: bool
    legend: list[SeasonalLegendItemResponse]
    district_items: list[SeasonalMapAreaResponse]
    region_items: list[SeasonalMapAreaResponse]


class SeasonalMapRefreshCombinationResponse(SchemaModel):
    theme: str
    season_profile: str
    mode: str
    subseason: str | None = None


class SeasonalMapRefreshSuccessResponse(SeasonalMapRefreshCombinationResponse):
    product_id: str
    generated_at: datetime
    active_pointer_path: str
    legacy_active_pointer_path: str | None = None


class SeasonalMapRefreshFailureResponse(SeasonalMapRefreshCombinationResponse):
    error: str


class SeasonalMapRefreshResponse(SchemaModel):
    forecast_source: str
    forecast_source_label: str
    requested_theme: str | None = None
    requested_season_profile: str | None = None
    requested_mode: str | None = None
    requested_subseason: str | None = None
    attempted_count: int
    succeeded_count: int
    failed_count: int
    attempted: list[SeasonalMapRefreshCombinationResponse]
    succeeded: list[SeasonalMapRefreshSuccessResponse]
    failed: list[SeasonalMapRefreshFailureResponse]


class SeasonalThemeOptionsResponse(SchemaModel):
    modes: list[str]
    subseasons: list[str] = Field(default_factory=list)


class SeasonalProfileOptionsResponse(SchemaModel):
    label: str
    calendar_subseasons: list[str] = Field(default_factory=list)


class SeasonalMapOptionsResponse(SchemaModel):
    themes: dict[str, SeasonalThemeOptionsResponse]
    profiles: dict[str, SeasonalProfileOptionsResponse]


class TrainResponse(SchemaModel):
    model_version: str
    metrics: dict[str, Any]
    bias_method: str
    bias_comparison: dict[str, Any] = Field(default_factory=dict)
    evaluation_paths: dict[str, str] = Field(default_factory=dict)


class HealthResponse(SchemaModel):
    status: str
    project_name: str
    default_forecast_source: str | None = None
    active_forecast_source: str | None = None
    active_forecast_path: str | None = None
    active_data_origin: str | None = None
    point_request_data_origin: str | None = None
    source_resolution_error: str | None = None
    station_path: str | None = None
    station_path_exists: bool | None = None
    data_sources: dict[str, Any] = Field(default_factory=dict)
