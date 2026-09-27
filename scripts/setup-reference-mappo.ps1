param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$externalRoot = Join-Path $projectRoot ".external"
$sourceRoot = Join-Path $externalRoot "reference-sources\marlbenchmark-on-policy"
$toolsRoot = Join-Path $externalRoot "tools\micromamba"
$micromamba = Join-Path $toolsRoot "micromamba.exe"
$mambaRoot = Join-Path $externalRoot "micromamba-root"
$environment = Join-Path $externalRoot "envs\mappo-reference"
$micromambaVersion = "2.8.1"
$micromambaSha256 = "8a51f88ec02600488ea20c3acd93fbd4da6c0f03fc499aa53fd234c6749b94b0"

if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    & (Join-Path $PSScriptRoot "fetch-reference-sources.ps1")
}

if (-not (Test-Path -LiteralPath $micromamba -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null
    $archive = Join-Path $toolsRoot "micromamba.tar.bz2"
    $extractRoot = Join-Path $toolsRoot "extract"
    Invoke-WebRequest `
        -Uri "https://micro.mamba.pm/api/micromamba/win-64/$micromambaVersion" `
        -OutFile $archive
    New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
    tar -xjf $archive -C $extractRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not extract micromamba." }
    $extracted = Get-ChildItem -LiteralPath $extractRoot -Recurse -Filter "micromamba.exe" |
        Select-Object -First 1
    if (-not $extracted) { throw "micromamba.exe was not found in the downloaded archive." }
    Copy-Item -LiteralPath $extracted.FullName -Destination $micromamba
}

$actualHash = (Get-FileHash -LiteralPath $micromamba -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $micromambaSha256) {
    throw "micromamba checksum mismatch: expected $micromambaSha256, got $actualHash"
}

if ($Force -and (Test-Path -LiteralPath $environment)) {
    throw "For safety, -Force does not delete environments. Remove only $environment manually after reviewing it."
}

if (-not (Test-Path -LiteralPath (Join-Path $environment "python.exe") -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $environment) | Out-Null
    & $micromamba --root-prefix $mambaRoot create -y -p $environment -c conda-forge `
        python=3.8 pip=23.0 setuptools=65.6.3 wheel=0.38.4
    if ($LASTEXITCODE -ne 0) { throw "Could not create the reference environment." }
}

$requirements = @(
    "numpy==1.23.5",
    "scipy==1.10.1",
    "torch==1.13.1",
    "gym==0.17.2",
    "tensorboardX==2.6.2.2",
    "protobuf==3.20.3",
    "setproctitle==1.3.3",
    "imageio==2.9.0",
    "wandb==0.15.12",
    "absl-py==0.9.0",
    "seaborn==0.10.1",
    "matplotlib==3.5.3"
)

& $micromamba run -p $environment python -m pip install @requirements
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned reference dependencies." }
& $micromamba run -p $environment python -m pip install --no-deps -e $sourceRoot
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned MAPPO source." }

& $micromamba run -p $environment python -c `
    "import gym,numpy,onpolicy,scipy,tensorboardX,torch; print('python environment ready'); print('gym',gym.__version__); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('torch',torch.__version__); print('tensorboardX',tensorboardX.__version__); print('onpolicy',onpolicy.__file__)"
if ($LASTEXITCODE -ne 0) { throw "Reference environment validation failed." }

Write-Host "Official MAPPO reference environment ready at $environment"
