param(
    [string]$EnvName = "puregpu3d",
    [switch]$SkipUpdateIfExists
)

$ErrorActionPreference = "Stop"

function Resolve-CondaCommand {
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        return "conda"
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Conda command not found. Install Miniconda/Anaconda and reopen the shell."
}

function Invoke-CondaChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CondaCommand,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    & $CondaCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed: $CondaCommand $($Arguments -join ' ')"
    }
}

function Invoke-CondaRunChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CondaCommand,
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentName,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    & $CondaCommand run -n $EnvironmentName @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Conda run failed: $CondaCommand run -n $EnvironmentName $($Arguments -join ' ')"
    }
}

function Test-CondaEnvExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CondaCommand,
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentName
    )
    $output = & $CondaCommand env list
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list conda environments."
    }
    $pattern = "^\s*" + [regex]::Escape($EnvironmentName) + "\s"
    return [bool]($output | Select-String -Pattern $pattern)
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repoRoot "environment/environment.yml"
if (-not (Test-Path $envFile)) {
    throw "environment.yml not found: $envFile"
}

$condaCmd = Resolve-CondaCommand
Write-Host "[bootstrap] Using conda command: $condaCmd"
Write-Host "[bootstrap] Using env file: $envFile"

$envExists = Test-CondaEnvExists -CondaCommand $condaCmd -EnvironmentName $EnvName

if ($envExists -and $SkipUpdateIfExists.IsPresent) {
    Write-Host "[bootstrap] Environment '$EnvName' already exists. Skipping update."
    return
}

if ($envExists) {
    Write-Host "[bootstrap] Updating existing environment: $EnvName"
    Invoke-CondaChecked -CondaCommand $condaCmd -Arguments @("env", "update", "-n", $EnvName, "-f", $envFile, "--prune")
} else {
    Write-Host "[bootstrap] Creating new environment: $EnvName"
    Invoke-CondaChecked -CondaCommand $condaCmd -Arguments @("env", "create", "-f", $envFile)
}

Write-Host "[bootstrap] Upgrading pip"
Invoke-CondaRunChecked -CondaCommand $condaCmd -EnvironmentName $EnvName -Arguments @("python", "-m", "pip", "install", "--upgrade", "pip")

Write-Host "[bootstrap] Installing project in editable mode"
Invoke-CondaRunChecked -CondaCommand $condaCmd -EnvironmentName $EnvName -Arguments @("python", "-m", "pip", "install", "-e", $repoRoot)

Write-Host "[bootstrap] Verifying CuPy runtime (non-fatal)"
& $condaCmd run -n $EnvName python -c "import cupy as cp; cp.arange(1, dtype=cp.float32).sum().item(); print('cupy-ok')"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CuPy runtime check failed. The app will use CPU fallback automatically."
}

Write-Host "[bootstrap] Setup complete."
