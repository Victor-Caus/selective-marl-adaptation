$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$logDirectory = Join-Path $projectRoot "results\runs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "pipeline-quick-console.log"
$env:PYTHONUNBUFFERED = "1"

.\.venv\Scripts\python.exe -m selective_marl pipeline --profile quick --device auto 2>&1 |
    Tee-Object -FilePath $logPath
if ($LASTEXITCODE -ne 0) { throw "Quick pipeline failed with exit code $LASTEXITCODE" }
Write-Host "Console log: $logPath"
