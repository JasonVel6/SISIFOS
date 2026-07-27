#!/bin/bash
# Generate render-free SISIFOS datasets for every probe-c*.json config in this
# directory and symlink each result into SatSLAM/datasets/<name>.
#
# Idempotent: if a symlink already exists in datasets/ and resolves to a real
# Agent_0 directory with gtValues.txt, the corresponding generation is skipped.
# So this script is safe to re-run if interrupted.
#
# Spacing: SISIFOS' output dir is minute-stamped (renders/YYYY-MM-DD_HHMM), so
# back-to-back invocations within the same minute would collide. We sleep 65 s
# between launches to advance the minute boundary.
#
# Usage:
#   bash configs/synthetic-exp/_generate-datasets.sh           # run all
#   bash configs/synthetic-exp/_generate-datasets.sh c5        # only cells matching glob 'c5*'
set -u
cd "$(dirname "$0")/../.."  # cd to SISIFOS/

SISIFOS_DIR="$(pwd)"
DATASETS_DIR=/home/jdflo/satslam/SatSLAM/datasets

filter="${1:-}"
configs=(configs/synthetic-exp/probe-${filter}*.json)
if [ ${#configs[@]} -eq 0 ] || [ ! -e "${configs[0]}" ]; then
    echo "No configs match: configs/synthetic-exp/probe-${filter}*.json"
    exit 1
fi

total=${#configs[@]}
echo "Found $total configs to consider."
ok=0; skipped=0; failed=0

for cfg in "${configs[@]}"; do
    name=$(basename "$cfg" .json)
    sym="$DATASETS_DIR/$name"

    if [ -L "$sym" ] && [ -e "$sym/gtValues.txt" ]; then
        echo "[SKIP] $name (symlink + gtValues.txt already present)"
        skipped=$((skipped+1))
        continue
    fi

    sleep 65  # advance minute boundary to avoid render-dir collision

    echo "[RUN]  $name"
    out=$(timeout 180 blender -b -P main.py -- --sweep_config_path "$cfg" 2>&1 \
        | tee /tmp/_synth_gen_${name}.log \
        | grep "Output written to" | awk '{print $NF}')

    if [ -z "$out" ]; then
        echo "[FAIL] $name: no 'Output written to' line. See /tmp/_synth_gen_${name}.log"
        failed=$((failed+1))
        continue
    fi

    agent="$SISIFOS_DIR/$out/Config_1_RF_Hubble/Agent_0"
    if [ ! -d "$agent" ]; then
        echo "[FAIL] $name: $agent missing. See /tmp/_synth_gen_${name}.log"
        failed=$((failed+1))
        continue
    fi

    ln -sfn "$agent" "$sym"
    echo "[OK]   $name -> $out/Config_1_RF_Hubble/Agent_0"
    ok=$((ok+1))
done

echo
echo "Done. ok=$ok  skipped=$skipped  failed=$failed  (of $total)"
