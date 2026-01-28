#!/bin/bash

# SISIFOS Environment Activation Script (Mac/Linux)
# Activates the Blender Python environment with auto-sync dependencies

# Check if being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced to activate the environment."
    echo "Usage: source ${0}"
    return 1
fi

# Detect the path of this script, handling sourcing and execution
if [ -n "$BASH_SOURCE" ]; then
    # Bash (Linux/Mac) uses BASH_SOURCE
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [ -n "$ZSH_VERSION" ]; then
    # Zsh (Mac default) uses %x prompt expansion
    SCRIPT_PATH="${(%):-%x}"
else
    # Fallback for standard execution (not sourced)
    SCRIPT_PATH="$0"
fi

# Get the absolute directory path
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BLENDER_DIR="$SCRIPT_DIR/Blender_4.5"
OS_TYPE="$(uname -s)"
ARCH_TYPE="$(uname -m)"

# Define paths based on OS
if [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Detected OS: Linux"
    BLENDER_EXE="$BLENDER_DIR/blender"
    BLENDER_PYTHON_BIN_DIR="$BLENDER_DIR/4.5/python/bin"
    BLENDER_PYTHON="$BLENDER_PYTHON_BIN_DIR/python3.11"

elif [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Detected OS: macOS"
    BLENDER_EXE="$BLENDER_DIR/Blender.app/Contents/MacOS/Blender"
    BLENDER_PYTHON_BIN_DIR="$BLENDER_DIR/Blender.app/Contents/Resources/4.5/python/bin"
    BLENDER_PYTHON="$BLENDER_PYTHON_BIN_DIR/python3.11"
else
    echo "Error: Unsupported OS: $OS_TYPE"
    return 1 2>/dev/null || return 1
fi

# If already activated, deactivate first to avoid duplicate prompts
if type deactivate &>/dev/null; then
    deactivate >/dev/null 2>&1
fi

# Validation
if [ ! -f "$BLENDER_PYTHON" ]; then
    echo -e "\033[0;31m[SISIFOS] Error: Environment not found.\033[0m"
    echo "Please run 'source ./env/setup.sh' first."
    return 1 2>/dev/null || return 1
fi

# Auto-sync dependencies if lock file or pyproject.toml changed
LOCK_FILE="$PROJECT_ROOT/uv.lock"
PYPROJECT_FILE="$PROJECT_ROOT/pyproject.toml"

echo -ne "\033[0;36m[SISIFOS] Syncing dependencies...\033[0m"

if [[ -f "$LOCK_FILE" && -f "$PYPROJECT_FILE" ]]; then
    if "$BLENDER_PYTHON" -m uv sync --no-editable -q >/dev/null 2>&1; then
        echo -ne "\r\033[0;32m[SISIFOS] Environment activated.\t\t\t\t\t\t\033[0m\n"
    else
        echo -ne "\r\033[0;33m[SISIFOS] Warning: Failed to sync. Run 'source ./env/setup.sh' to fix.\t\t\033[0m\n"
        return 1 2>/dev/null || return 1
    fi
else
    echo -ne "\r\033[0;32m[SISIFOS] Environment activated.\t\t\t\t\t\t\033[0m\n"
fi

# Save old state
export SISIFOS_OLD_PATH="$PATH"
export SISIFOS_OLD_PS1="$PS1"

# Set environment variables
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