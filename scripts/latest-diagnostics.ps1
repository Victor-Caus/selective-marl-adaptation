$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$bundles = Get-ChildItem -Path (Join-Path $projectRoot "results\runs") `
    -Filter "diagnostic-bundle.zip" -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending

if (-not $bundles) {
    throw "No diagnostic bundle found. Run a pipeline first."
}

$bundles[0].FullName

