"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cumulus.api.errors import install_exception_handlers
from cumulus.api.routes_advisory import router as advisory_router
from cumulus.api.routes_farmer_advisory import router as farmer_advisory_router
from cumulus.api.routes_forecast import router as forecast_router
from cumulus.api.routes_health import router as health_router
from cumulus.api.routes_nationwide import router as nationwide_router
from cumulus.api.routes_seasonal_map import router as seasonal_map_router
from cumulus.api.routes_training import router as training_router
from cumulus.logging import configure_logging
from cumulus.settings import get_settings

settings = get_settings()
configure_logging(settings.log_level)
app = FastAPI(title=settings.project_name, version=settings.api_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
install_exception_handlers(app)
app.include_router(health_router)
app.include_router(training_router)
app.include_router(forecast_router)
app.include_router(nationwide_router)
app.include_router(seasonal_map_router)
app.include_router(advisory_router)
app.include_router(farmer_advisory_router)
