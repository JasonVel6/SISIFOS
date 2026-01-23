#!/bin/bash

# SISIFOS Environment Setup Script (Mac/Linux)

# Helper Functions & Setup

# Function to overwrite the current line
update_progress() {
    # \r moves cursor to start, %-60s pads with spaces
    printf "\r[SISIFOS] %-60s" "$1"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced to activate the environment."
    echo "Usage: source ${0}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_DIR="$SCRIPT_DIR/Blender_4.5"
OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

# Define URLs and Paths based on OS
if [[ "$OS_TYPE" == "Linux" ]]; then
    BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.6-linux-x64.tar.xz"
    DOWNLOAD_FILE="blender.tar.xz"
    # Linux extracts to a folder name usually matching the tarball content
    EXTRACTED_FOLDER_NAME="blender-4.5.6-linux-x64"
    
    # Path definitions for Linux
    BLENDER_EXE="$BLENDER_DIR/blender"
    # Find python binary
    BLENDER_PYTHON_BIN_DIR="$BLENDER_DIR/4.5/python/bin"

elif [[ "$OS_TYPE" == "Darwin" ]]; then
    if [[ "$ARCH_TYPE" == "arm64" ]]; then
        BLENDER_URL="https://download.blender.org/release/Blender4.5/blender-4.5.6-macos-arm64.dmg"
    else
        BLENDER_URL="https://ftp.nluug.nl/pub/graphics/blender/release/Blender4.5/blender-4.5.6-macos-x64.dmg"
    fi
    DOWNLOAD_FILE="blender.dmg"
    
    # Path definitions for macOS (.app bundle structure)
    BLENDER_EXE="$BLENDER_DIR/Blender.app/Contents/MacOS/Blender"
    BLENDER_PYTHON_BIN_DIR="$BLENDER_DIR/Blender.app/Contents/Resources/4.5/python/bin"
else
    echo "Unsupported OS: $OS_TYPE"
    return 1 2>/dev/null || exit 1
fi

# Installation Logic

if [ ! -d "$BLENDER_DIR" ]; then
    update_progress "Blender not found. Downloading..."
    
    cd "$SCRIPT_DIR" || return
    curl -L -o "$DOWNLOAD_FILE" "$BLENDER_URL" --silent --show-error

    update_progress "Extracting Blender..."
    
    if [[ "$OS_TYPE" == "Linux" ]]; then
        tar -xf "$DOWNLOAD_FILE"
        mv "$EXTRACTED_FOLDER_NAME" "Blender_4.5"
        rm "$DOWNLOAD_FILE"
        
    elif [[ "$OS_TYPE" == "Darwin" ]]; then
        # Mount DMG, Copy App, Detach
        MOUNT_POINT=$(hdiutil attach "$DOWNLOAD_FILE" -nobrowse -readonly | grep -o '/Volumes/.*')
        if [ -z "$MOUNT_POINT" ]; then
            echo "Error: Could not mount DMG."
            return 1
        fi
        
        # Create dir and copy
        mkdir -p "Blender_4.5"
        cp -R "$MOUNT_POINT/Blender.app" "Blender_4.5/"
        
        # Detach and clean
        hdiutil detach "$MOUNT_POINT" -quiet
        rm "$DOWNLOAD_FILE"
    fi
fi

# Python Setup

# Assign Blender Python executable
BLENDER_PYTHON="$BLENDER_PYTHON_BIN_DIR/python"

if [ ! -f "$BLENDER_PYTHON" ]; then
    echo ""
    echo "Error: Blender Python not found at $BLENDER_PYTHON_BIN_DIR"
    return 1 2>/dev/null || exit 1
fi

update_progress "Bootstrapping pip..."
"$BLENDER_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1

update_progress "Upgrading build tools..."
"$BLENDER_PYTHON" -m pip install --upgrade pip setuptools wheel uv -q >/dev/null 2>&1

UV_REQ_FILE="/tmp/sisifos-uv-req.txt"
[ -f "$UV_REQ_FILE" ] && rm "$UV_REQ_FILE"

update_progress "Resolving dependencies..."
cd "$SCRIPT_DIR" || return
"$BLENDER_PYTHON" -m uv export --format requirements.txt --locked --no-emit-project --output-file "$UV_REQ_FILE" -q >/dev/null 2>&1
if [ $? -ne 0 ]; then echo -e "\nuv export failed"; return 1; fi

update_progress "Installing dependencies..."
"$BLENDER_PYTHON" -m uv pip install --require-hashes --requirements "$UV_REQ_FILE" -q >/dev/null 2>&1
if [ $? -ne 0 ]; then echo -e "\nuv pip install failed"; return 1; fi

update_progress "Installing project in editable mode..."
"$BLENDER_PYTHON" -m uv pip install --no-deps --editable . -q >/dev/null 2>&1
if [ $? -ne 0 ]; then echo -e "\neditable install failed"; return 1; fi


# Environment Activation

update_progress "Configuring environment variables..."

# Save old state
export SISIFOS_OLD_PATH="$PATH"
export SISIFOS_OLD_PS1="$PS1"

# Set new variables
export PATH="$BLENDER_PYTHON_BIN_DIR:$BLENDER_DIR:$PATH"
export BLENDER="$BLENDER_EXE"
export PYTHON="$BLENDER_PYTHON"

# Update Prompt
PS1="[SISIFOS] $PS1"

# Define Deactivate Function
deactivate() {
    if [ -n "$SISIFOS_OLD_PATH" ]; then
        export PATH="$SISIFOS_OLD_PATH"
        export PS1="$SISIFOS_OLD_PS1"
        
        unset SISIFOS_OLD_PATH
        unset SISIFOS_OLD_PS1
        unset BLENDER
        unset PYTHON
        unset -f deactivate
        
        echo -e "\033[0;33m[SISIFOS] Environment deactivated.\033[0m"
    else
        echo -e "\033[0;33m[SISIFOS] No active environment to deactivate.\033[0m"
    fi
}

# Final cleanup of progress bar
printf "\r%-60s\r" " "
echo -e "\033[0;32mSISIFOS Environment activated successfully.\033[0m"