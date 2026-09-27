param(
    [ValidateSet("smoke", "pilot", "full")]
    [string]$Profile = "smoke",
    [int]$Seed = 1,
    [int]$RolloutThreads = 0,
    [long]$NumEnvSteps = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$externalRoot = Join-Path $projectRoot ".external"
$sourceRoot = Join-Path $externalRoot "reference-sources\marlbenchmark-on-policy"
$micromamba = Join-Path $externalRoot "tools\micromamba\micromamba.exe"
$environment = Join-Path $externalRoot "envs\mappo-reference"
$resultRoot = Join-Path $projectRoot "results\reference-runs"

if (-not (Test-Path -LiteralPath (Join-Path $environment "python.exe") -PathType Leaf)) {
    throw "Reference environment missing. Run .\scripts\setup-reference-mappo.ps1 first."
}

$defaults = @{
    smoke = @{ steps = 500L; threads = 1 }
    pilot = @{ steps = 20000L; threads = 16 }
    full = @{ steps = 3000000L; threads = 16 }
}
if ($NumEnvSteps -le 0) { $NumEnvSteps = $defaults[$Profile].steps }
if ($RolloutThreads -le 0) { $RolloutThreads = $defaults[$Profile].threads }

# The official source creates output paths relative to __file__. A short same-drive
# junction avoids the Windows 260-character path limit without copying or editing it.
$shortSource = Join-Path $env:LOCALAPPDATA "selective-marl-mappo-source"
if (Test-Path -LiteralPath $shortSource) {
    $existing = Get-Item -LiteralPath $shortSource -Force
    $resolvedTarget = [System.IO.Path]::GetFullPath([string]$existing.Target)
    $expectedTarget = [System.IO.Path]::GetFullPath($sourceRoot)
    if ($existing.LinkType -ne "Junction" -or $resolvedTarget -ne $expectedTarget) {
        throw "$shortSource already exists and is not the expected junction."
    }
} else {
    New-Item -ItemType Junction -Path $shortSource -Target $sourceRoot | Out-Null
}

$timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$experiment = "reference_${Profile}_seed${Seed}_win${RolloutThreads}_${timestamp}"
$runRoot = Join-Path $resultRoot $experiment
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$consoleLog = Join-Path $runRoot "console.log"
$sourceCommit = (git -C $sourceRoot rev-parse HEAD).Trim()

$arguments = @(
    "train\train_mpe.py",
    "--env_name", "MPE",
    "--algorithm_name", "rmappo",
    "--experiment_name", $experiment,
    "--scenario_name", "simple_reference",
    "--num_agents", "2",
    "--num_landmarks", "3",
    "--seed", [string]$Seed,
    "--n_training_threads", "1",
    "--n_rollout_threads", [string]$RolloutThreads,
    "--num_mini_batch", "1",
    "--episode_length", "25",
    "--num_env_steps", [string]$NumEnvSteps,
    "--ppo_epoch", "15",
    "--gain", "0.01",
    "--lr", "0.0007",
    "--critic_lr", "0.0007",
    "--use_wandb",
    "--cuda"
)

$metadata = [ordered]@{
    status = "running"
    profile = $Profile
    started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_repository = "https://github.com/marlbenchmark/on-policy"
    source_commit = $sourceCommit
    algorithm = "rmappo"
    scenario = "simple_reference"
    seed = $Seed
    num_env_steps = $NumEnvSteps
    n_rollout_threads = $RolloutThreads
    official_n_rollout_threads = 128
    host_deviation = if ($RolloutThreads -eq 128) { $null } else { "Reduced rollout processes for Windows memory limits." }
    command_arguments = $arguments
}
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot "metadata.json") -Encoding utf8
& $micromamba run -p $environment python -m pip freeze |
    Set-Content -LiteralPath (Join-Path $runRoot "environment.txt") -Encoding utf8

$scratch = Join-Path $env:TEMP "selective-marl-reference"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$env:TEMP = $scratch
$env:TMP = $scratch
$scriptsRoot = Join-Path $shortSource "onpolicy\scripts"
Push-Location $scriptsRoot
try {
    & $micromamba run -p $environment python @arguments 2>&1 |
        Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

$officialRunRoot = Join-Path $scriptsRoot "results\MPE\simple_reference\rmappo\$experiment"
if (Test-Path -LiteralPath $officialRunRoot -PathType Container) {
    Copy-Item -LiteralPath $officialRunRoot -Destination (Join-Path $runRoot "official-output") -Recurse
}

$metadata.status = if ($exitCode -eq 0) { "complete" } else { "failed" }
$metadata["completed_at_utc"] = (Get-Date).ToUniversalTime().ToString("o")
$metadata["exit_code"] = $exitCode
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot "metadata.json") -Encoding utf8

if ($exitCode -ne 0) { throw "Reference run failed with exit code $exitCode. See $consoleLog" }

& $micromamba run -p $environment python (Join-Path $PSScriptRoot "analyze-reference-mappo.py") --run-dir $runRoot
if ($LASTEXITCODE -ne 0) { throw "Training completed, but analysis failed for $runRoot" }
Write-Host "Reference run and analysis available at $runRoot"
