$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot "reproductions\references.lock.json"
$externalRoot = Join-Path $projectRoot ".external\reference-sources"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $externalRoot | Out-Null

foreach ($source in $manifest.sources) {
    $destination = Join-Path $externalRoot $source.id
    if (-not (Test-Path -LiteralPath $destination -PathType Container)) {
        git clone --filter=blob:none --no-checkout $source.repository $destination
        if ($LASTEXITCODE -ne 0) { throw "Clone failed for $($source.id)" }
    }
    git -C $destination fetch origin $source.commit --depth 1
    if ($LASTEXITCODE -ne 0) { throw "Fetch failed for $($source.id)" }
    git -C $destination checkout --detach $source.commit
    if ($LASTEXITCODE -ne 0) { throw "Checkout failed for $($source.id)" }
    $actual = git -C $destination rev-parse HEAD
    if ($actual.Trim() -ne $source.commit) {
        throw "Commit mismatch for $($source.id): expected $($source.commit), got $actual"
    }
    Write-Host "$($source.id): $actual"
}

Write-Host "Reference sources are available at $externalRoot"
