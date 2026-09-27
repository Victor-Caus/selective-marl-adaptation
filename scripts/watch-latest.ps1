param(
    [ValidateSet("maddpg", "mappo_no_comm", "mappo", "channel_randomized", "rmappo", "selective")]
    [string]$Algorithm = "mappo",
    [ValidateSet("none", "semantic", "behavioral", "both")]
    [string]$Condition = "semantic",
    [int]$Episodes = 2,
    [int]$ChangeEpisode = 1,
    [double]$Fps = 12,
    [switch]$OracleGate
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$runsRoot = Join-Path $projectRoot "results\runs"
$run = Get-ChildItem -LiteralPath $runsRoot -Directory |
    Where-Object { $_.Name -like "*-train-$Algorithm-*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $run) {
    throw "No training run found for algorithm '$Algorithm'."
}

$checkpoint = Join-Path $run.FullName "checkpoint.pt"
if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
    throw "Checkpoint not found: $checkpoint"
}

$arguments = @(
    "-m", "selective_marl", "watch", $checkpoint,
    "--condition", $Condition,
    "--episodes", $Episodes,
    "--change-episode", $ChangeEpisode,
    "--fps", $Fps
)
if ($OracleGate) {
    $arguments += "--oracle-gate"
}

Write-Host "Watching checkpoint: $checkpoint"
& ".\.venv\Scripts\python.exe" @arguments
if ($LASTEXITCODE -ne 0) { throw "Viewer failed with exit code $LASTEXITCODE" }
