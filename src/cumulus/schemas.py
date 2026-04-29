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
    family_label: str
    display_order: int
    reverse_probability_scale: bool


class SeasonalProbabilityCategoryResponse(SchemaModel):
    category_code: str
    label: str
    hint: str
    color: str
    percentage: float


class SeasonalProbabilityMetricResponse(SchemaModel):
    theme: str
    theme_label: str
    category_code: str
    category_label: str
    dominant_category_code: str
    dominant_category_label: str
    dominant_percentage: float
    display_value: str
    unit: str | None = None
    criteria_note: str
    interpretation: str
    color: str
    category_probabilities: list[SeasonalProbabilityCategoryResponse]


class SeasonalDeterministicMetricResponse(SchemaModel):
    theme: str
    theme_label: str
    value: float | None = None
    display_value: str
    unit: str | None = None
    criteria_note: str
    interpretation: str
    legend_label: str
    color: str


class SeasonalProbabilityMapAreaResponse(SchemaModel):
    location_id: str
    geography_type: str
    geography_name: str
    region_name: str
    coverage_count: int = 1
    coverage_note: str
    metric: SeasonalProbabilityMetricResponse


class SeasonalDeterministicMapAreaResponse(SchemaModel):
    location_id: str
    geography_type: str
    geography_name: str
    region_name: str
    coverage_count: int = 1
    coverage_note: str
    metric: SeasonalDeterministicMetricResponse


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


class SeasonalProbabilityMapProductResponse(SeasonalMapRunResponse):
    refresh_status: str
    is_stale: bool
    legend: list[SeasonalLegendItemResponse]
    district_items: list[SeasonalProbabilityMapAreaResponse]
    region_items: list[SeasonalProbabilityMapAreaResponse]


class SeasonalDeterministicMapProductResponse(SeasonalMapRunResponse):
    refresh_status: str
    is_stale: bool
    legend: list[SeasonalLegendItemResponse]
    district_items: list[SeasonalDeterministicMapAreaResponse]
    region_items: list[SeasonalDeterministicMapAreaResponse]


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


class ForecastRasterLegendStopResponse(SchemaModel):
    offset: float
    color: str


class ForecastRasterBoundsResponse(SchemaModel):
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


class ForecastRasterGridResponse(SchemaModel):
    latitudes: list[float]
    longitudes: list[float]
    values: list[list[float | None]]


class ForecastRasterMetadataResponse(SchemaModel):
    layer_id: str
    tile_url: str
    variable: str
    variable_label: str
    unit: str
    horizon_day: int
    valid_time: datetime
    generated_at: datetime
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    data_origin: str | None = None
    lower_bound: float
    upper_bound: float
    available_horizon_days: list[int]
    legend_ticks: list[float]
    color_ramp: list[ForecastRasterLegendStopResponse]
    bounds: ForecastRasterBoundsResponse
    grid: ForecastRasterGridResponse


class ForecastRasterSampleResponse(SchemaModel):
    latitude: float
    longitude: float
    nearest_latitude: float | None = None
    nearest_longitude: float | None = None
    value: float | None = None
    variable: str
    variable_label: str
    unit: str
    horizon_day: int
    valid_time: datetime
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    data_origin: str | None = None


class ForecastProductBoundsResponse(SchemaModel):
    latitude_min: float
    latitude_max: float
    longitude_min: float
    longitude_max: float


class ForecastProductGridShapeResponse(SchemaModel):
    y: int
    x: int


class ForecastProductGridResolutionResponse(SchemaModel):
    latitude: float | None = None
    longitude: float | None = None


class ForecastProductLegendItemResponse(SchemaModel):
    category_code: str
    label: str
    hint: str
    color: str
    display_order: int


class ForecastProductColorRampStopResponse(SchemaModel):
    offset: float
    color: str


class ForecastProbabilityProductResponse(SchemaModel):
    product_id: str
    theme: str
    theme_label: str
    season_profile: str | None = None
    season_label: str | None = None
    subseason: str | None = None
    subseason_label: str | None = None
    forecast_year: int
    valid_time: datetime
    generated_at: datetime
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    generation_backend: str
    source_artifact_type: str
    grid_shape: ForecastProductGridShapeResponse
    grid_resolution_degrees: ForecastProductGridResolutionResponse
    is_low_resolution_fallback: bool
    refresh_interval_seconds: int
    freshness_threshold_hours: int
    tile_url: str
    preview_url: str | None = None
    bounds: ForecastProductBoundsResponse
    legend: list[ForecastProductLegendItemResponse]


class ForecastDeterministicProductResponse(SchemaModel):
    product_id: str
    theme: str
    theme_label: str
    season_profile: str | None = None
    season_label: str | None = None
    subseason: str | None = None
    subseason_label: str | None = None
    forecast_year: int
    valid_time: datetime
    generated_at: datetime
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    generation_backend: str
    source_artifact_type: str
    grid_shape: ForecastProductGridShapeResponse
    grid_resolution_degrees: ForecastProductGridResolutionResponse
    is_low_resolution_fallback: bool
    refresh_interval_seconds: int
    freshness_threshold_hours: int
    tile_url: str
    preview_url: str | None = None
    bounds: ForecastProductBoundsResponse
    unit: str
    lower_bound: float
    upper_bound: float
    legend_ticks: list[float]
    color_ramp: list[ForecastProductColorRampStopResponse]


class ForecastProbabilitySampleCategoryResponse(SchemaModel):
    category_code: str
    label: str
    hint: str
    color: str
    percentage: float


class ForecastProbabilitySampleResponse(SchemaModel):
    theme: str
    theme_label: str
    season_profile: str | None = None
    season_label: str | None = None
    subseason: str | None = None
    subseason_label: str | None = None
    latitude: float
    longitude: float
    nearest_latitude: float
    nearest_longitude: float
    dominant_category_code: str
    dominant_category_label: str
    dominant_percentage: float
    display_value: str
    interpretation: str
    criteria_note: str
    category_probabilities: list[ForecastProbabilitySampleCategoryResponse]
    valid_time: datetime
    forecast_year: int
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    generation_backend: str


class ForecastDeterministicSampleResponse(SchemaModel):
    theme: str
    theme_label: str
    season_profile: str | None = None
    season_label: str | None = None
    subseason: str | None = None
    subseason_label: str | None = None
    latitude: float
    longitude: float
    nearest_latitude: float
    nearest_longitude: float
    value: float
    display_value: str
    unit: str
    interpretation: str
    criteria_note: str
    valid_time: datetime
    forecast_year: int
    forecast_source: str
    forecast_source_label: str
    source_run_id: str
    generation_backend: str


class ForecastProductRefreshAttemptResponse(SchemaModel):
    theme: str
    view_mode: str


class ForecastProductRefreshSuccessResponse(ForecastProductRefreshAttemptResponse):
    product_id: str
    generated_at: datetime
    manifest_path: str


class ForecastProductRefreshFailureResponse(ForecastProductRefreshAttemptResponse):
    error: str


class ForecastProductRefreshResponse(SchemaModel):
    attempted_count: int
    succeeded_count: int
    failed_count: int
    attempted: list[ForecastProductRefreshAttemptResponse]
    succeeded: list[ForecastProductRefreshSuccessResponse]
    failed: list[ForecastProductRefreshFailureResponse]


class ForecastThemeOptionResponse(SchemaModel):
    theme: str
    label: str
    title: str
    requires_season: bool
    requires_subseason: bool
    enabled: bool
    reason: str | None = None
    seasons: list[str] = Field(default_factory=list)
    subseasons: list[str] = Field(default_factory=list)


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
