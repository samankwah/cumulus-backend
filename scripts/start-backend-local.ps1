$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $backendRoot
$defaultForecastPath = Join-Path $backendRoot "data\sample_forecast_smoke.nc"
$era5ManifestPath = Join-Path $workspaceRoot "ml\data\raw\era5\manifest.json"
$gfsManifestPath = Join-Path $workspaceRoot "ml\data\raw\gfs\manifest.json"
$hasDownloadedSource = (Test-Path $era5ManifestPath) -or (Test-Path $gfsManifestPath)

if (
  -not $env:CUMULUS_UPSTREAM_FORECAST_PATH -and
  -not $env:CUMULUS_ERA5_FORECAST_PATH -and
  -not $env:CUMULUS_GFS_FORECAST_PATH -and
  -not $hasDownloadedSource
) {
  $env:CUMULUS_UPSTREAM_FORECAST_PATH = $defaultForecastPath
}

if ($env:CUMULUS_UPSTREAM_FORECAST_PATH -and -not $env:CUMULUS_UPSTREAM_FORECAST_ENGINE) {
  $env:CUMULUS_UPSTREAM_FORECAST_ENGINE = "scipy"
}

Write-Host "Starting Cumulus backend on http://127.0.0.1:8000"
if ($hasDownloadedSource -and -not $env:CUMULUS_UPSTREAM_FORECAST_PATH) {
  Write-Host "Forecast source: auto-discovered from ml/data/raw manifests"
} else {
  Write-Host "Forecast source: $env:CUMULUS_UPSTREAM_FORECAST_PATH"
}

Set-Location $backendRoot
python -m uvicorn cumulus.main:app --app-dir src --host 0.0.0.0 --port 8000
