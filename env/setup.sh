#!/bin/bash

# SISIFOS Environment Setup Script (Mac/Linux)
# Run this script once to bootstrap the environment

# Helper Functions
update_progress() {
    printf "\r[SISIFOS] %-60s" "$1"
}

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BLENDER_DIR="$SCRIPT_DIR/Blender_4.5"
OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

# Define URLs and paths based on OS
if [[ "$OS_TYPE" == "Linux" ]]; then
    BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.6-linux-x64.tar.xz"
    DOWNLOAD_FILE="blender.tar.xz"
    EXTRACTED_FOLDER_NAME="blender-4.5.6-linux-x64"
    
    BLENDER_EXE="$BLENDER_DIR/blender"
    BLENDER_PYTHON_BIN_DIR="$BLENDER_DIR/4.5/python/bin"

elif [[ "$OS_TYPE" == "Darwin" ]]; then
    if [[ "$ARCH_TYPE" == "arm64" ]]; then
        BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.6-macos-arm64.dmg"
    else
        BLENDER_URL="https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.5/blender-4.5.6-macos-x64.dmg"
    fi
    DOWNLOAD_FILE="blender.dmg"
    
    BLENDER_EXE="$BLENDER_DIR/Blender.app/Contents/MacOS/Blender"
    BLENDER_PYTHON_BIN_DIR="$BLENDER_DIR/Blender.app/Contents/Resources/4.5/python/bin"
else
    echo "Unsupported OS: $OS_TYPE"
    return 1
fi

# Helper function to handle errors
cleanup_and_exit() {
    printf "\r%-60s\r" " "
    echo "SISIFOS Setup failed: $1" >&2
    return 1
}

# Asset paths
ASSETS_DIR="$PROJECT_ROOT/assets"
STARMAP_PATH="$ASSETS_DIR/starmap_2020_16k.exr"
STARMAP_URL="https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_16k.exr"

# 0. Download assets if needed
if [ ! -d "$ASSETS_DIR" ]; then
    mkdir -p "$ASSETS_DIR" || cleanup_and_exit "Failed to create assets directory"
fi

if [ ! -f "$STARMAP_PATH" ]; then
    update_progress "Downloading starmap asset (large file)..."
    if ! curl -L -o "$STARMAP_PATH" "$STARMAP_URL" --show-error; then
        cleanup_and_exit "Failed to download starmap"
    fi
fi

# 1. Download and extract Blender if needed
if [ ! -d "$BLENDER_DIR" ]; then
    update_progress "Downloading Blender..."
    
    cd "$SCRIPT_DIR" || cleanup_and_exit "Failed to change to $SCRIPT_DIR"
    
    if ! curl -L -o "$DOWNLOAD_FILE" "$BLENDER_URL" --show-error; then
        cleanup_and_exit "Failed to download Blender"
    fi

    update_progress "Extracting Blender..."
    
    if [[ "$OS_TYPE" == "Linux" ]]; then
        if ! tar -xf "$DOWNLOAD_FILE"; then
            cleanup_and_exit "Failed to extract Blender tarball"
        fi
        if ! mv "$EXTRACTED_FOLDER_NAME" "Blender_4.5"; then
            cleanup_and_exit "Failed to rename Blender directory"
        fi
        rm "$DOWNLOAD_FILE"
        
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        MOUNT_POINT=$(hdiutil attach "$DOWNLOAD_FILE" -nobrowse -readonly 2>/dev/null | grep -o '/Volumes/.*')
        if [ -z "$MOUNT_POINT" ]; then
            cleanup_and_exit "Could not mount DMG"
        fi
        
        mkdir -p "Blender_4.5"
        if ! cp -R "$MOUNT_POINT/Blender.app" "Blender_4.5/"; then
            hdiutil detach "$MOUNT_POINT" -quiet 2>/dev/null
            cleanup_and_exit "Failed to copy Blender.app"
        fi
        
        hdiutil detach "$MOUNT_POINT" -quiet 2>/dev/null
        rm "$DOWNLOAD_FILE"
    fi
fi

# 2. Validate Python executable
BLENDER_PYTHON="$BLENDER_PYTHON_BIN_DIR/python3.11"
if [ ! -f "$BLENDER_PYTHON" ]; then
    cleanup_and_exit "Blender Python not found at $BLENDER_PYTHON_BIN_DIR"
fi

# 3. Bootstrap Python environment
update_progress "Bootstrapping pip..."
if ! "$BLENDER_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    cleanup_and_exit "Failed to bootstrap pip"
fi

update_progress "Upgrading build tools..."
if ! "$BLENDER_PYTHON" -m pip install --upgrade pip setuptools wheel uv -q >/dev/null 2>&1; then
    cleanup_and_exit "Failed to install build tools"
fi

# 4. Generate lock file
update_progress "Generating lock file..."
cd "$PROJECT_ROOT" || cleanup_and_exit "Failed to change to $PROJECT_ROOT"
if ! "$BLENDER_PYTHON" -m uv lock -q >/dev/null 2>&1; then
    cleanup_and_exit "uv lock failed"
fi

# 5. Export dependencies and install
UV_REQ_FILE="/tmp/sisifos-uv-req.txt"
[ -f "$UV_REQ_FILE" ] && rm "$UV_REQ_FILE"

update_progress "Exporting dependencies..."
if ! "$BLENDER_PYTHON" -m uv export --format requirements.txt --locked --no-emit-project --output-file "$UV_REQ_FILE" -q >/dev/null 2>&1; then
    cleanup_and_exit "uv export failed"
fi

update_progress "Installing dependencies..."
if ! "$BLENDER_PYTHON" -m uv pip install --require-hashes --requirements "$UV_REQ_FILE" -q >/dev/null 2>&1; then
    cleanup_and_exit "uv pip install failed"
fi

update_progress "Installing project in editable mode..."
if ! "$BLENDER_PYTHON" -m uv pip install --no-deps --editable . -q >/dev/null 2>&1; then
    cleanup_and_exit "editable install failed"
fi

# Success
printf "\r%-60s\r" " "
echo "SISIFOS Setup complete."
echo "Run 'source .\env/activate.sh' to start."
