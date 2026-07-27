#!/usr/bin/env bash
#SBATCH --job-name=irolsd_bepi
#SBATCH --account=gts-pt43
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# IRoLSD BepiColombo fixed-pose/swept-Sun render on PACE.
#
# HARD RULE: qos=embers ONLY (free). NEVER inferno. rtx6000 max 6 CPU:GPU.
# embers is preemptible + 8h max -> the render driver is RESUMABLE, so if this
# job is preempted just resubmit and it continues from disk. Chain with a
# self-resubmitting loop or an array if needed.
#
# Assumes SISIFOS on scratch and the camera_traj.csv already generated (run
# gen_fixedpose_sweptsun.py on the login node first -- it is CPU-only + fast).
#
# Submit:
#   sbatch --export=ALL,AGENT=$AGENT,SIS=$SIS,BLENDER=$BLENDER \
#          scripts/irolsd/embers_render_irolsd.sbatch
# where AGENT = the Agent_0 dir holding camera_traj.csv.
# ---------------------------------------------------------------------------
set -euo pipefail

: "${SIS:=$HOME/scratch/sisifos/SISIFOS}"
: "${BLENDER:=$SIS/env/Blender_4.5/blender}"
: "${CFG:=$SIS/configs/irolsd_bepi.json}"
: "${AGENT:?set AGENT=<Agent_0 dir with camera_traj.csv>}"
: "${BANK:=500}"

echo "host=$(hostname)  SIS=$SIS  AGENT=$AGENT"
nvidia-smi -L || true

cd "$SIS"
SIS="$SIS" BLENDER="$BLENDER" bash scripts/irolsd/render_irolsd.sh "$CFG" "$AGENT" "$BANK"

echo "=== render step done for $AGENT ==="
