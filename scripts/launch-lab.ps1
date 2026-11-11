param([int]$Port = 8050)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & '.\.venv\Scripts\python.exe' -m bioreactor_twin.web.app --port $Port
} finally {
    Pop-Location
}
