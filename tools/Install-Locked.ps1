param([string]$Python = 'python', [string]$Environment = '.venv-locked')
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
Push-Location $Repo
try {
    & $Python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12'"
    if ($LASTEXITCODE) { throw 'Python 3.12 is required' }
    if (Test-Path -LiteralPath $Environment) { throw 'Choose a new environment directory' }
    & $Python -m venv $Environment
    if ($LASTEXITCODE) { throw 'venv failed' }
    $Runtime = Join-Path $Environment 'Scripts/python.exe'
    & $Runtime -m pip install --no-deps -r requirements.lock
    if ($LASTEXITCODE) { throw 'Locked dependency installation failed' }
    & $Runtime -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE) { throw 'Project installation failed' }
    & $Runtime -m pip check
    if ($LASTEXITCODE) { throw 'Dependency consistency check failed' }
} finally { Pop-Location }
