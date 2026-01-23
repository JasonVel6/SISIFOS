$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$BlenderDir = Join-Path $ScriptDir "Blender_4.5"
$BlenderPythonDir = Join-Path $BlenderDir "4.5\python\bin"
$BlenderPython = Join-Path $BlenderPythonDir "python.exe"
$BlenderExe = Join-Path $BlenderDir "blender.exe"

# Helper function to overwrite the current line
function Update-Progress ($Message) {
    # `r moves cursor to start of line. 
    # {0,-60} pads the string to 60 chars to overwrite previous text.
    Write-Host -NoNewline ("`r{0,-60}" -f "[SISIFOS] $Message")
}

if (-not (Test-Path $BlenderPython)) {
    Write-Host "[SISIFOS] Error: Blender Python not found." -ForegroundColor Red
    exit 1
}

try {
    Update-Progress "Bootstrapping pip..."
    & $BlenderPython -m ensurepip --upgrade *>$null
    
    Update-Progress "Upgrading build tools (pip, uv, wheel)..."
    & $BlenderPython -m pip install --upgrade pip setuptools wheel uv -q *>$null
    
    $UvReqFile = Join-Path $env:TEMP "sisifos-uv-req.txt"
    if (Test-Path $UvReqFile) { Remove-Item $UvReqFile -Force }
    
    Update-Progress "Resolving dependencies..."
    Push-Location $ScriptDir
    & $BlenderPython -m uv export --format requirements.txt --locked --no-emit-project --output-file $UvReqFile -q *>$null
    if ($LASTEXITCODE -ne 0) { throw "uv export failed" }
    Pop-Location
    
    Update-Progress "Installing dependencies..."
    & $BlenderPython -m uv pip install --require-hashes --requirements $UvReqFile -q *>$null
    if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }
    
    Update-Progress "Installing project in editable mode..."
    Push-Location $ScriptDir
    & $BlenderPython -m uv pip install --no-deps --editable . -q *>$null
    if ($LASTEXITCODE -ne 0) { throw "editable install failed" }
    Pop-Location
    
    Update-Progress "Configuring environment variables..."
    $env:SISIFOS_OLD_PATH = $env:PATH
    
    if (-not (Test-Path variable:SISIFOS_OLD_PROMPT)) {
        if (Test-Path variable:Global:__VSCodeState) {
            Set-Variable -Name SISIFOS_OLD_PROMPT -Value $Global:__VSCodeState.OriginalPrompt -Scope Global
        } else {
            Set-Variable -Name SISIFOS_OLD_PROMPT -Value { "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) " } -Scope Global
        }
    }
    
    $env:PATH = "$BlenderPythonDir;$BlenderDir;$env:PATH"
    $env:BLENDER = $BlenderExe
    $env:PYTHON = $BlenderPython
    
    if (Test-Path variable:Global:__VSCodeState) {
        Set-Variable -Name SISIFOS_OLD_PROMPT -Value $Global:__VSCodeState.OriginalPrompt -Scope Global -Force
        $Global:__VSCodeState.OriginalPrompt = {
            "[SISIFOS] $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
        }
    } else {
        Set-Variable -Name SISIFOS_OLD_PROMPT -Value { "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) " } -Scope Global
        function prompt {
            "[SISIFOS] $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
        }
    }
    
    function global:deactivate {
        if ($env:SISIFOS_OLD_PATH) {
            $env:PATH = $env:SISIFOS_OLD_PATH
            
            if (Test-Path variable:Global:__VSCodeState) {
                $Global:__VSCodeState.OriginalPrompt = $Global:SISIFOS_OLD_PROMPT
            } else {
                Remove-Item function:prompt -ErrorAction SilentlyContinue
            }
            
            Remove-Item variable:\SISIFOS_OLD_PROMPT -ErrorAction SilentlyContinue
            Remove-Item env:\SISIFOS_OLD_PATH -ErrorAction SilentlyContinue
            Remove-Item env:\BLENDER -ErrorAction SilentlyContinue
            Remove-Item env:\PYTHON -ErrorAction SilentlyContinue
            Remove-Item function:deactivate -ErrorAction SilentlyContinue
            Write-Host "[SISIFOS] Environment deactivated." -ForegroundColor Yellow
        } else {
            Write-Host "[SISIFOS] No active environment to deactivate." -ForegroundColor Yellow
        }
    }
    
    # Clear the progress line before the final message
    Write-Host -NoNewline ("`r" + (" " * 60) + "`r")
    Write-Host "[SISIFOS] Environment activated successfully." -ForegroundColor Green
    exit 0
} catch {
    # Clear the progress line so the error shows cleanly
    Write-Host -NoNewline ("`r" + (" " * 60) + "`r")
    Write-Host "[SISIFOS] Setup failed: $_" -ForegroundColor Red
    exit 1
}