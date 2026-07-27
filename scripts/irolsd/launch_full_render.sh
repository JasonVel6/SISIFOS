#!/usr/bin/env bash
# Launch the FULL IRoLSD Bepi render as N parallel embers shards.
#
#   [SIS=...] scripts/irolsd/launch_full_render.sh [TOTAL_POSES] [NSHARDS] [NVARIANTS] [RANGE]
#     defaults: TOTAL_POSES=1000  NSHARDS=10  NVARIANTS=50  RANGE=60
#
# Splits the global TOTAL_POSES Fibonacci sphere into NSHARDS contiguous slices,
# generates a self-contained agent dir per shard (login node, fast), then submits
# ONE Slurm ARRAY job (embers) with NSHARDS tasks that render concurrently. Each
# shard is resumable; re-running this script re-submits only unfinished shards'
# work (the driver skips frames already on disk). Pack with pack_all_shards.sh.
#
# HARD RULE: qos=embers ONLY. Never inferno.
set -euo pipefail
SIS="${SIS:-$HOME/scratch/sisifos/SISIFOS}"
TOTAL_POSES="${1:-1000}"
NSHARDS="${2:-10}"
NVARIANTS="${3:-50}"
RANGE="${4:-60}"
SEED="${SEED:-12345}"
BASE="$SIS/renders/irolsd/bepi_train"
cd "$SIS"

echo "=== IRoLSD full render: $TOTAL_POSES poses / $NSHARDS shards / $NVARIANTS variants / R=${RANGE}m ==="

# --- generate each shard's camera_traj.csv (contiguous slice of global sphere) ---
per=$(( (TOTAL_POSES + NSHARDS - 1) / NSHARDS ))   # ceil
for k in $(seq 0 $((NSHARDS - 1))); do
  start=$(( k * per ))
  [ "$start" -ge "$TOTAL_POSES" ] && break
  count=$per; [ $((start + count)) -gt "$TOTAL_POSES" ] && count=$((TOTAL_POSES - start))
  OUT="$BASE/shard_$(printf '%02d' "$k")/Config_1_RF_Bepi-mcs/Agent_0"
  python3 scripts/irolsd/gen_fixedpose_sweptsun.py \
    --out "$OUT" --total-poses "$TOTAL_POSES" --pose-start "$start" --pose-count "$count" \
    --nvariants "$NVARIANTS" --range "$RANGE" --seed "$SEED"
done

NREAL=$(ls -d "$BASE"/shard_*/ 2>/dev/null | wc -l)
echo "=== generated $NREAL shard agents under $BASE ==="

# --- write the array sbatch (one task per shard) ---
ARR="$BASE/render_array.sbatch"
cat > "$ARR" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=irolsd_bepi
#SBATCH --account=gts-pt43
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-$((NREAL - 1))%${NSHARDS}
#SBATCH --output=$BASE/logs/shard_%a.%A.out
set -euo pipefail
SIS="$SIS"
K=\$(printf '%02d' "\$SLURM_ARRAY_TASK_ID")
AGENT="$BASE/shard_\$K/Config_1_RF_Bepi-mcs/Agent_0"
echo "host=\$(hostname) shard=\$K agent=\$AGENT"; nvidia-smi -L || true
cd "\$SIS"
SIS="\$SIS" bash scripts/irolsd/render_irolsd.sh configs/irolsd_bepi.json "\$AGENT" 500
echo "=== shard \$K DONE ==="
SB
mkdir -p "$BASE/logs"

echo "=== submitting array job (embers) ==="
sbatch "$ARR"
squeue -u "$(whoami)" -o "%.12i %.14j %.9T %.6D %R" | head -20
echo
echo "Monitor:  squeue -u \$(whoami) | grep irolsd"
echo "Pack when done:  bash scripts/irolsd/pack_all_shards.sh"
