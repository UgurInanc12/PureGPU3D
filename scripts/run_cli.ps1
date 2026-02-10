[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$EnvName = "puregpu3d",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
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

    throw "Conda command not found. Install Miniconda/Anaconda first."
}

$condaCmd = Resolve-CondaCommand
& $condaCmd run --no-capture-output -n $EnvName python -m puregpu3d.cli @Args
