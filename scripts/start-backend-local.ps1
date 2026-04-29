param(
  [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$backendRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $backendRoot
$localDevScript = Join-Path $workspaceRoot "scripts\local-dev.ps1"

. $localDevScript

Assert-CumulusPortAvailable -Port 8000 -Role backend -ForceRestart:$ForceRestart

$defaultForecastPath = Join-Path $backendRoot "data\sample_forecast_smoke.nc"
$era5ManifestPath = Join-Path $workspaceRoot "ml\data\raw\era5\manifest.json"
$gfsManifestPath = Join-Path $workspaceRoot "ml\data\raw\gfs\manifest.json"
$hasDownloadedSource = (Test-Path $era5ManifestPath) -or (Test-Path $gfsManifestPath)
$pythonPath = Get-CumulusCommandPath -Name "python"

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

$forecastSource = $env:CUMULUS_UPSTREAM_FORECAST_PATH
if ($env:CUMULUS_ERA5_FORECAST_PATH) {
  $forecastSource = "ERA5: $env:CUMULUS_ERA5_FORECAST_PATH"
} elseif ($env:CUMULUS_GFS_FORECAST_PATH) {
  $forecastSource = "GFS: $env:CUMULUS_GFS_FORECAST_PATH"
} elseif ($hasDownloadedSource -and -not $env:CUMULUS_UPSTREAM_FORECAST_PATH) {
  $forecastSource = "auto-discovered from ml/data/raw manifests"
}

Write-Host "Starting Cumulus backend on http://127.0.0.1:8000"
Write-Host "Working directory: $backendRoot"
Write-Host "Python executable: $pythonPath"
Write-Host "Forecast source: $forecastSource"
Write-Host "PID/log file: foreground process; use Ctrl+C to stop"

Set-Location $backendRoot
& $pythonPath -m uvicorn cumulus.main:app --app-dir src --host 0.0.0.0 --port 8000
exit $LASTEXITCODE
