# SISIFOS Environment Activation Script
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

$BlenderDir = Join-Path $ScriptDir "Blender_4.5"
$BlenderPythonDir = Join-Path $BlenderDir "4.5\python\bin"
$BlenderPython = Join-Path $BlenderPythonDir "python.exe"
$BlenderExe = Join-Path $BlenderDir "blender.exe"

# If already activated, deactivate first to avoid duplicate prompts
if (Test-Path function:deactivate) {
    deactivate *>$null
}

# Validation
if (-not (Test-Path $BlenderPython)) {
    Write-Host "[SISIFOS] Error: Environment not found." -ForegroundColor Red
    Write-Host "Please run '.\env\Setup.ps1' first."
    return
}

# Auto-sync dependencies if lock file or pyproject.toml changed
$LockFile = Join-Path $ProjectRoot "uv.lock"
$PyprojectFile = Join-Path $ProjectRoot "pyproject.toml"

Write-Host -NoNewline "[SISIFOS] Syncing dependencies... "

if ((Test-Path $LockFile) -and (Test-Path $PyprojectFile)) {
    & $BlenderPython -m uv sync --no-editable -q *>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`r[SISIFOS] Warning: Failed to sync. Run '.\env\Setup.ps1' to fix." -ForegroundColor Yellow
        return
    }
}

# Overwrite the progress line with final message
Write-Host "`r[SISIFOS] Environment activated.`t`t`t`t`t`t" -ForegroundColor Green

# 1. Environment Variables
$env:SISIFOS_OLD_PATH = $env:PATH

# Save Prompt
if (-not (Test-Path variable:SISIFOS_OLD_PROMPT)) {
    if (Test-Path variable:Global:__VSCodeState) {
        Set-Variable -Name SISIFOS_OLD_PROMPT -Value $Global:__VSCodeState.OriginalPrompt -Scope Global
    }
    else {
        Set-Variable -Name SISIFOS_OLD_PROMPT -Value { "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) " } -Scope Global
    }
}

# Set Paths
$env:PATH = "$BlenderPythonDir;$BlenderDir;$env:PATH"
$env:BLENDER = $BlenderExe
$env:PYTHON = $BlenderPython
$env:UV_PROJECT_ENVIRONMENT = Split-Path -Path $BlenderPythonDir -Parent

# 2. Modify Prompt
if (Test-Path variable:Global:__VSCodeState) {
    Set-Variable -Name SISIFOS_OLD_PROMPT -Value $Global:__VSCodeState.OriginalPrompt -Scope Global -Force
    $Global:__VSCodeState.OriginalPrompt = {
        "[SISIFOS] $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
    }
}
else {
    Set-Variable -Name SISIFOS_OLD_PROMPT -Value { "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) " } -Scope Global
    function prompt {
        "[SISIFOS] $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
    }
}

# 3. Deactivate Function
function global:deactivate {
    if ($env:SISIFOS_OLD_PATH) {
        $env:PATH = $env:SISIFOS_OLD_PATH
        
        if (Test-Path variable:Global:__VSCodeState) {
            $Global:__VSCodeState.OriginalPrompt = $Global:SISIFOS_OLD_PROMPT
        }
        else {
            Remove-Item function:prompt -ErrorAction SilentlyContinue
        }
        
        Remove-Item variable:\SISIFOS_OLD_PROMPT -ErrorAction SilentlyContinue
        Remove-Item env:\SISIFOS_OLD_PATH -ErrorAction SilentlyContinue
        Remove-Item env:\BLENDER -ErrorAction SilentlyContinue
        Remove-Item env:\PYTHON -ErrorAction SilentlyContinue
        Remove-Item env:\UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
        Remove-Item function:deactivate -ErrorAction SilentlyContinue
        Write-Host "[SISIFOS] Environment deactivated." -ForegroundColor Yellow
    }
    else {
        Write-Host "[SISIFOS] No active environment to deactivate." -ForegroundColor Yellow
    }
}

# Final message is printed above in the activation block