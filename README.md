# Cumulus Backend

FastAPI service and reusable Python package (`cumulus`) for the Ghana seasonal
advisory platform. The Next.js frontend lives in a separate repository,
[`seasonal-fcst-frontend`](https://github.com/samankwah/seasonal-fcst-frontend).

This repository is self-contained: runtime config lives in `configs/`, and the
published forecast products, the district geometry and a trained baseline model
are committed under `data/`. No sibling `ml/` checkout is required.

## Requirements

- Python >= 3.11

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Optional extras: `.[grib]` (read GRIB forecast sources via `cfgrib`), `.[download]`
(ERA5 downloads via `cdsapi`).

## Run locally

```powershell
copy .env.example .env
powershell -ExecutionPolicy Bypass -File .\scripts\start-backend-local.ps1
```

Or directly:

```powershell
python -m uvicorn cumulus.main:app --app-dir src --host 0.0.0.0 --port 8000
```

The helper script picks a forecast source in this order: `CUMULUS_ERA5_FORECAST_PATH`
/ `CUMULUS_GFS_FORECAST_PATH` / `CUMULUS_UPSTREAM_FORECAST_PATH` if set, then any
manifest under `data/raw/{era5,gfs}/manifest.json`, otherwise the bundled
`data/sample_forecast_smoke.nc` (read with the `scipy` engine).

Check it is up:

```powershell
curl http://127.0.0.1:8000/health
```

## Configuration

Settings are read from environment variables (prefix `CUMULUS_`, nested delimiter
`__`) layered over `configs/*.yaml`. See `.env.example` for the common ones. Key
paths default to locations inside this repo:

| Setting | Default | Purpose |
| --- | --- | --- |
| `CUMULUS_CONFIG_DIR` | `configs` | runtime YAML config |
| `CUMULUS_DATA_DIR` | `data` | ML data root (`data/raw`, `data/processed`) and artifact root |
| `CUMULUS_DEFAULT_STATION_PATH` | `data/raw/stations/Rainfall_data.xlsx` | station workbook, only used by `POST /train` |
| `CUMULUS_CORS_ALLOWED_ORIGINS` | localhost:3000 + deployed frontend | extra allowed browser origins (comma-separated) |

## Tests

```powershell
python -m pytest
```

## Layout

```
main.py                 serverless entrypoint (re-exports cumulus.main:app)
configs/                base.yaml, model.yaml, advisory.yaml, seasonal_map.yaml, locations.yaml
data/                   committed forecast products, district geojson, sample forecast, baseline model
src/cumulus/
  main.py               FastAPI app factory
  settings.py           pydantic-settings configuration
  api/                  route modules (health, forecast, nationwide, seasonal_map, advisory, training)
  services/             business logic
  data/ modeling/ ...   loaders, trainer/predictor, advisory rule engines
scripts/                start-backend-local.ps1, local-dev.ps1, batch/refresh utilities
tests/
```
