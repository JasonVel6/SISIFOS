#!/usr/bin/env bash
# Pilot: render the SAME 150 poses at 3 Sun-perturbation widths (+/-3, 8, 15 deg) to find
# where IRoLSD illumination robustness peaks before a full 1000-pose commit.
# Each width -> its own dataset dir. embers ONLY.
set -euo pipefail
SIS="${SIS:-$HOME/scratch/sisifos/SISIFOS}"
NPOSES="${NPOSES:-150}"; NVARIANTS="${NVARIANTS:-50}"; RANGE="${RANGE:-60}"; SEED="${SEED:-12345}"
DELTAS=(3 8 15)
cd "$SIS"
echo "=== Sun-delta pilot: ${NPOSES} poses x {${DELTAS[*]}} deg x ${NVARIANTS} variants, R=${RANGE}m ==="

for D in "${DELTAS[@]}"; do
  BASE="$SIS/renders/irolsd/bepi_sd${D}"
  OUT="$BASE/Config_1_RF_Bepi-mcs/Agent_0"
  mkdir -p "$BASE/logs"
  # single-shard (150 poses is small) camera_traj with the WIDER perturbation
  python3 scripts/irolsd/gen_fixedpose_sweptsun.py \
    --out "$OUT" --nposes "$NPOSES" --nvariants "$NVARIANTS" --range "$RANGE" \
    --sun-delta-deg "$D" --seed "$SEED"
  echo "  [gen] sd${D}: $(ls "$OUT"/*.csv 2>/dev/null | head -1)"
done

# one array job, one task per width
ARR="$SIS/renders/irolsd/sundelta_pilot.sbatch"
cat > "$ARR" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=irolsd_sdpilot
#SBATCH --account=gts-pt43
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=$SIS/renders/irolsd/sdpilot_%a.%A.out
set -euo pipefail
SIS="$SIS"
DELTAS=(3 8 15)
D=\${DELTAS[\$SLURM_ARRAY_TASK_ID]}
AGENT="$SIS/renders/irolsd/bepi_sd\${D}/Config_1_RF_Bepi-mcs/Agent_0"
echo "host=\$(hostname) sd=\${D} agent=\$AGENT"; nvidia-smi -L || true
cd "\$SIS"
SIS="\$SIS" bash scripts/irolsd/render_irolsd.sh configs/irolsd_bepi.json "\$AGENT" 500
echo "=== SD \${D} DONE ==="
SB

echo "=== submitting pilot array (embers, 3 tasks) ==="
sbatch "$ARR"
squeue -u "$(whoami)" -o "%.12i %.16j %.9T %R" | head
