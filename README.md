# Cumulus Backend

FastAPI service and reusable Python package (`cumulus`) for the Ghana seasonal
advisory platform. The Next.js frontend lives in a separate repository,
[`cumulus-frontend`](https://github.com/samankwah/cumulus-frontend).

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

## Sub-seasonal rainfall (IFS-UNet, 46 days)

The frontend's default "Next 46 days" view is served from ECMWF IFS extended-range runs
downscaled to 0.1° with a UNet. Upstream delivers one NetCDF4/HDF5 file per lead day
(`Unet/YYYY-MM-DD/YYYY-MM-DD-00-LLLL.nc`, variable `precip_24h`, Africa-wide, ~53 MB per run).
The API never reads those directly: an offline **ingest** turns each run into a compact
Ghana artifact (NetCDF3, int16 at 0.1 mm, ~0.45 MB) that is committed under
`data/artifacts/subseasonal/ifs_unet/<run_id>/`, with `active.json` pointing at the newest run.

```powershell
pip install -e ".[ingest]"     # adds h5netcdf + h5py; not needed to serve the API

# from a local folder holding YYYY-MM-DD run folders
python -m cumulus.subseasonal.ingest --source local --path D:\data\Unet --latest

# from Azure Blob Storage (container SAS URL with Read + List)
$env:CUMULUS_SUBSEASONAL__AZURE_SAS_URL = "https://<account>.blob.core.windows.net/<container>?<sas>"
python -m cumulus.subseasonal.ingest --source azure --list
python -m cumulus.subseasonal.ingest --source azure --latest
```

Ingest clips the small negative values the UNet produces, keeps Ghana + 0.5°, validates the
init time, grid and contiguous lead days (at least 7), and keeps the newest 3 runs. The
`Ingest IFS-UNet run` GitHub workflow does this daily once the `IFS_UNET_SAS_URL` repository
secret is set.

Lead day *d* holds rain accumulated from init + (d-1)·24 h to init + d·24 h, so for a 00 UTC
run day 1 is the init date itself (Ghana is on UTC). Outlook layers use the thresholds in
`configs/base.yaml` (`subseasonal:`): a wet day is ≥ 1 mm, a dry spell is ≥ 5 consecutive
days below that, a wet spell is ≥ 3 consecutive wet days; "spell days" count the days that
fall inside such spells.

| Endpoint | Purpose |
| --- | --- |
| `GET /subseasonal/runs` | runs, init time, staleness, national daily/weekly means |
| `GET /subseasonal/layer` | legend, title, stats and tile URL for `layer` × `aggregation` × `index` |
| `GET /subseasonal/tiles/{z}/{x}/{y}.png` | map tiles (immutable when `run_id` is given) |
| `GET /subseasonal/area-values` | area mean of the current layer for every region or district |
| `GET /subseasonal/sample` / `GET /subseasonal/area` | 46-day series, weekly totals, spells for a point or area |

Layers: `rainfall` (`daily` / `weekly` / `total`), `rainy_days`, `dry_spell_days`,
`wet_spell_days`. Area means are area-weighted using district polygons rasterised at 0.01°.

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
  api/                  route modules (health, forecast, subseasonal, nationwide, seasonal_map, advisory, training)
  subseasonal/          IFS-UNet ingest (Azure/local sources), district rasteriser, spell metrics, legends
  services/             business logic
  data/ modeling/ ...   loaders, trainer/predictor, advisory rule engines
scripts/                start-backend-local.ps1, local-dev.ps1, batch/refresh utilities
tests/
```
