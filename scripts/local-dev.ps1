# Shared helpers for the Cumulus backend local startup scripts.
#
# This file used to live at the monorepo root (../../scripts/local-dev.ps1). The
# backend is now a standalone repository, so the helpers are vendored here and
# dot-sourced by start-backend-local.ps1.

function Assert-CumulusPortAvailable {
  param(
    [Parameter(Mandatory = $true)][int]$Port,
    [Parameter(Mandatory = $true)][string]$Role,
    [switch]$ForceRestart
  )

  $owningPids = @()
  try {
    $owningPids = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop |
      Select-Object -ExpandProperty OwningProcess -Unique
  } catch {
    # Get-NetTCPConnection is unavailable or matched nothing; fall back to netstat.
    $matches = netstat -ano | Select-String -Pattern (":{0}\s+\S+\s+LISTENING" -f $Port)
    foreach ($match in $matches) {
      $owningPids += [int](($match.ToString() -split '\s+')[-1])
    }
    $owningPids = $owningPids | Select-Object -Unique
  }

  if (-not $owningPids) {
    return
  }

  foreach ($owningPid in $owningPids) {
    $proc = Get-Process -Id $owningPid -ErrorAction SilentlyContinue
    $name = if ($proc) { $proc.ProcessName } else { "unknown" }
    if ($ForceRestart) {
      Write-Host "Port $Port ($Role) in use by PID $owningPid ($name); stopping it because -ForceRestart was passed."
      Stop-Process -Id $owningPid -Force -ErrorAction SilentlyContinue
    } else {
      throw "Port $Port ($Role) is already in use by PID $owningPid ($name). Stop that process or re-run with -ForceRestart."
    }
  }

  if ($ForceRestart) {
    Start-Sleep -Seconds 1
  }
}

function Get-CumulusCommandPath {
  param(
    [Parameter(Mandatory = $true)][string]$Name
  )

  $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $command) {
    throw "Required executable '$Name' was not found on PATH."
  }
  return $command.Source
}
