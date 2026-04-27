"""Centralized API error handling."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class CumulusServiceError(Exception):
    status_code = 500
    error_code = "service_error"

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class InvalidCoordinatesError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_coordinates"


class InvalidHorizonError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_horizon"


class InvalidForecastRasterVariableError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_forecast_raster_variable"


class ForecastSourceNotConfiguredError(CumulusServiceError):
    status_code = 503
    error_code = "forecast_source_not_configured"


class ForecastSourceReadError(CumulusServiceError):
    status_code = 503
    error_code = "forecast_source_read_error"


class ModelArtifactsNotAvailableError(CumulusServiceError):
    status_code = 503
    error_code = "model_artifacts_not_available"


class InferenceExecutionError(CumulusServiceError):
    status_code = 500
    error_code = "inference_execution_error"


class NationwideArtifactsNotAvailableError(CumulusServiceError):
    status_code = 503
    error_code = "nationwide_artifacts_not_available"


class SeasonalMapArtifactsNotAvailableError(CumulusServiceError):
    status_code = 503
    error_code = "seasonal_map_artifacts_not_available"


class SeasonalProbabilityProductIncompleteError(CumulusServiceError):
    status_code = 503
    error_code = "seasonal_probability_product_incomplete"


class SeasonalMapRefreshFailedError(CumulusServiceError):
    status_code = 500
    error_code = "seasonal_map_refresh_failed"


class ForecastProductArtifactsNotAvailableError(CumulusServiceError):
    status_code = 503
    error_code = "forecast_product_artifacts_not_available"


class ForecastProductIncompleteError(CumulusServiceError):
    status_code = 503
    error_code = "forecast_product_incomplete"


class InvalidForecastProductThemeError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_forecast_product_theme"


class InvalidForecastProductSelectionError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_forecast_product_selection"


class InvalidSeasonalModeError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_seasonal_mode"


class SubseasonRequiredError(CumulusServiceError):
    status_code = 422
    error_code = "subseason_required"


class SubseasonNotAllowedError(CumulusServiceError):
    status_code = 422
    error_code = "subseason_not_allowed"


class InvalidSubseasonForProfileError(CumulusServiceError):
    status_code = 422
    error_code = "invalid_subseason_for_profile"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(CumulusServiceError)
    async def _service_error_handler(_: Request, exc: CumulusServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_code": exc.error_code},
        )
