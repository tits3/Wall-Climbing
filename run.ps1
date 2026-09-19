# ASCII-only launcher for Windows PowerShell 5.1 and PowerShell 7.
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskVenvConfig = Join-Path $taskRoot '.venv\pyvenv.cfg'
$taskVenvValid = $false
if ((Test-Path -LiteralPath $taskPython) -and (Test-Path -LiteralPath $taskVenvConfig)) {
    $taskHomeLine = Get-Content -LiteralPath $taskVenvConfig | Where-Object { $_ -match '^home = ' }
    if ($taskHomeLine) {
        $taskOriginalPython = Join-Path ($taskHomeLine -replace '^home = ', '') 'python.exe'
        if (Test-Path -LiteralPath $taskOriginalPython) {
            & $taskPython -c 'import sys; sys.exit(0)' 2>$null
            $taskVenvValid = $LASTEXITCODE -eq 0
        }
    }
}
if (-not $taskVenvValid) {
    $taskPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $taskPython)) {
        throw 'Python not found. Rebuild .venv with Python 3.12 and wall_climb/requirements.txt.'
    }
}
# A Python file avoids legacy PowerShell -c quoting and here-string issues.
$taskLauncher = Join-Path $taskRoot 'launch_2d.py'
& $taskPython -B -X utf8 $taskLauncher @args
exit $LASTEXITCODE
