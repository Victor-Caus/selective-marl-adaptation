$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot "reproductions\references.lock.json"
$externalRoot = Join-Path $projectRoot ".external\reference-sources"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

function Command-Version([string]$Name, [string[]]$Arguments) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) { return $null }
    return (& $Name @Arguments 2>&1 | Select-Object -First 1).ToString()
}

$sources = foreach ($source in $manifest.sources) {
    $path = Join-Path $externalRoot $source.id
    $actual = $null
    if (Test-Path -LiteralPath $path -PathType Container) {
        $actual = (git -C $path rev-parse HEAD).Trim()
    }
    [ordered]@{
        id = $source.id
        expected_commit = $source.commit
        actual_commit = $actual
        ready = ($actual -eq $source.commit)
    }
}

$report = [ordered]@{
    python = Command-Version "python" @("--version")
    git = Command-Version "git" @("--version")
    docker = Command-Version "docker" @("--version")
    conda = Command-Version "conda" @("--version")
    wsl = [bool](Get-Command "wsl" -ErrorAction SilentlyContinue)
    exact_historical_maddpg_ready = [bool](Get-Command "docker" -ErrorAction SilentlyContinue) -or [bool](Get-Command "conda" -ErrorAction SilentlyContinue)
    sources = $sources
}

$report | ConvertTo-Json -Depth 5
