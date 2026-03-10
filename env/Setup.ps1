# Everything is relative to this script's location (env/)
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }

# Define Paths
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BlenderDir = Join-Path $ScriptDir "Blender_4.5"
$BlenderPythonDir = Join-Path $BlenderDir "4.5\python\bin"
$BlenderPython = Join-Path $BlenderPythonDir "python.exe"

# Asset Paths
$AssetsDir = Join-Path $ProjectRoot "assets"
$StarMapPath = Join-Path $AssetsDir "starmap_2020_16k.exr"
$StarMapUrl = "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_16k.exr"

function Update-Progress ($Message) {
    Write-Host -NoNewline ("`r{0,-60}" -f "[SISIFOS] $Message")
}

try {
    # 1. Asset Check: Starmap

    if (-not (Test-Path $AssetsDir)) {
        New-Item -ItemType Directory -Path $AssetsDir -Force | Out-Null
    }

    if (-not (Test-Path $StarMapPath)) {
        Update-Progress "Asset missing. Downloading starmap_2020_16k.exr..."
        # This is a large file, so it might take a moment
        Invoke-WebRequest -Uri $StarMapUrl -OutFile $StarMapPath
    }

    # 2. Blender Installation (Inside env/)
    if (-not (Test-Path $BlenderDir)) {
        $BlenderUrl = "https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.6-windows-x64.zip"
        $ZipPath = Join-Path $ScriptDir "blender.zip"
        $ExtractedFolderName = "blender-4.5.6-windows-x64"
        $ExtractedPath = Join-Path $ScriptDir $ExtractedFolderName

        Update-Progress "Blender not found. Downloading..."
        Invoke-WebRequest -Uri $BlenderUrl -OutFile $ZipPath

        Update-Progress "Extracting Blender..."
        Expand-Archive -Path $ZipPath -DestinationPath $ScriptDir -Force

        # Rename to "Blender_4.5"
        Rename-Item -Path $ExtractedPath -NewName "Blender_4.5"

        # Cleanup
        if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    }

    # 3. Check for Python Executable
    if (-not (Test-Path $BlenderPython)) {
        throw "Blender Python not found at $BlenderPython"
    }

    # 4. Prepare Blender Python as a virtual environment for uv
    $BlenderPythonParent = Split-Path -Path $BlenderPythonDir -Parent

    Update-Progress "Preparing Blender Python environment..."

    $PythonVersion = & $BlenderPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    $PyvenvCfg = Join-Path $BlenderPythonParent "pyvenv.cfg"
    @"
home = $BlenderPythonDir
implementation = CPython
version_info = $PythonVersion
include-system-site-packages = false
"@ | Set-Content -Path $PyvenvCfg

    # 5. Install uv
    Update-Progress "Installing uv..."
    & $BlenderPython -m ensurepip --upgrade *>$null
    if ($LASTEXITCODE -ne 0) { throw "Failed to bootstrap pip" }

    & $BlenderPython -m pip install --upgrade uv *>$null
    if ($LASTEXITCODE -ne 0) { throw "Failed to install uv" }

    # 6. Sync dependencies using uv
    Update-Progress "Syncing dependencies..."
    Push-Location $ProjectRoot
    $env:UV_PROJECT_ENVIRONMENT = $BlenderPythonParent

    & $BlenderPython -m uv sync
    if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

    Remove-Item env:\UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    Pop-Location

    # Clear progress
    Write-Host -NoNewline ("`r" + (" " * 60) + "`r")
    Write-Host "SISIFOS Setup complete." -ForegroundColor Green
    Write-Host "Run '. .\env\activate.ps1' to start." -ForegroundColor Cyan

} catch {
    Write-Host -NoNewline ("`r" + (" " * 60) + "`r")
    Write-Host "SISIFOS Setup failed: $_" -ForegroundColor Red
    exit 1
}