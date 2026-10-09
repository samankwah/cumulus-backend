# Cumulus Backend

FastAPI service and Python package (`cumulus`) behind the Cumulus forecast map for Ghana. It serves:
- the 46-day IFS-UNet rainfall outlook, with map tiles, area means and point/area series;
- the seasonal (wass2s) products;
- the advisory and model endpoints.

**Live API:** https://cumulus-backend.vercel.app (try [`/health`](https://cumulus-backend.vercel.app/health)) · **Frontend:** [`cumulus-frontend`](https://github.com/samankwah/cumulus-frontend)

The repo is self-contained:
- runtime config is in `configs/`;
- the forecast artifacts, district geometry and a baseline model are committed under `data/`.

## Quick start

Requires Python 3.11 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env
python -m uvicorn cumulus.main:app --app-dir src --port 8000   # http://127.0.0.1:8000/docs
```

`scripts\start-backend-local.ps1` does the same on `0.0.0.0:8000`. It refuses to start if port 8000 is already in use (pass `-ForceRestart` to replace it). It also chooses an upstream forecast source for the seasonal endpoints, in this order:
1. `CUMULUS_ERA5_FORECAST_PATH`, `CUMULUS_GFS_FORECAST_PATH` or `CUMULUS_UPSTREAM_FORECAST_PATH`;
2. otherwise `data/raw/{era5,gfs}/manifest.json`;
3. otherwise the bundled `data/sample_forecast_smoke.nc`.

Optional extras:
- `.[ingest]`: needed for the 46-day ingest.
- `.[grib]`: GRIB forecast sources.
- `.[download]`: ERA5 downloads.

## Configuration

Environment variables use the `CUMULUS_` prefix (with `__` for nested settings) and override `configs/*.yaml`. Relative paths resolve from the repo root. See `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUMULUS_CONFIG_DIR` | `configs` | Runtime YAML config. |
| `CUMULUS_DATA_DIR` | `data` | Data and artifact root. |
| `CUMULUS_CORS_ALLOWED_ORIGINS` | — | Extra browser origins, comma-separated. These are always allowed: `localhost:3000`, `127.0.0.1:3000`, `https://cumulus-gh.vercel.app` and that project's Vercel preview URLs. |
| `CUMULUS_DEFAULT_STATION_PATH` | `data/raw/stations/Rainfall_data.xlsx` | Station workbook, used only by `POST /train`. |
| `CUMULUS_SUBSEASONAL__AZURE_SAS_URL` | — | Container SAS URL (Read + List) for the 46-day ingest. Never commit it. |

## API

Interactive docs are at `/docs`.

| Area | Endpoints |
| --- | --- |
| Health | `GET /health` |
| 46-day outlook | `GET /subseasonal/runs`, `/layer`, `/tiles/{z}/{x}/{y}.png`, `/area-values`, `/sample`, `/area` |
| Seasonal products | `GET /forecast/products/options`; `GET /forecast/{probability,deterministic}/{active,sample,preview.png,tiles/…}`; `POST /forecast/products/refresh` |
| Advice and models | `POST /advisory`, `/farmer-advisory`, `/predict`, `/forecast`, `/train` |
| Batch products | `/nationwide/*`, `/seasonal-map/*` |

**46-day layers:**

| Layer | Periods |
| --- | --- |
| `rainfall`, `rainy_days`, `wet_spell_days`, `dry_spell_days` | daily, weekly, total |
| `onset` | daily (status by day), total (onset date) |

Area means are area-weighted, using district polygons rasterised at 0.01°. Tiles are cached in memory and sent with immutable `Cache-Control` when `run_id` is given. Responses of 1 KB or more are gzip-compressed.

Definitions are in `configs/base.yaml` under `subseasonal:`:
- **Wet day:** at least 1 mm of rain.
- **Dry spell:** 5 or more days in a row below 1 mm.
- **Wet spell:** 3 or more wet days in a row.
- **Onset:** at least 20 mm within 3 days, with no dry spell longer than 10 days in the next 30.

## 46-day data (IFS-UNet ingest)

Upstream delivers one Africa-wide NetCDF4 file per lead day (`Unet/YYYY-MM-DD/…`, about 53 MB per run). The API never reads these files. Instead, an offline ingest:
1. validates the run's init time, grid and lead days (at least 7, in a row);
2. clips small negative values;
3. writes a compact Ghana artifact (NetCDF3, about 0.45 MB) to `data/artifacts/subseasonal/ifs_unet/<run_id>/`;
4. points `active.json` at the newest run, and keeps the newest 10 runs.

```powershell
python -m pip install -e ".[ingest]"
python -m cumulus.subseasonal.ingest --source local --path D:\data\Unet --latest
python -m cumulus.subseasonal.ingest --source azure --latest     # needs CUMULUS_SUBSEASONAL__AZURE_SAS_URL
```

The **Ingest IFS-UNet run** GitHub workflow runs this daily at 06:30 UTC and commits the new artifact. It needs the `IFS_UNET_SAS_URL` repository secret. A run counts as stale after 3 days.

## Tests

```powershell
python -m pytest
```

## Deployment

The `main` branch deploys to Vercel. The root `main.py` adds `src/` to the import path and exposes `cumulus.main:app`. The committed `data/` artifacts are served as they are.

## Project structure

```
main.py              Vercel entrypoint
configs/             base, model, advisory, seasonal_map and locations YAML
data/                committed artifacts, district GeoJSON, sample forecast, baseline model
src/cumulus/
  main.py            FastAPI app (CORS, gzip, routers)
  settings.py        pydantic-settings configuration
  api/               routes: health, subseasonal, forecast, advisory, farmer_advisory, nationwide, seasonal_map, training
  subseasonal/       IFS-UNet ingest, district masks, spell/onset metrics, legends
  services/          business logic
  advisory/ data/ modeling/ preprocessing/ evaluation/
scripts/             start-backend-local.ps1, local-dev.ps1, batch and refresh tools
tests/
```
