#!/usr/bin/env bash
#SBATCH --job-name=irolsd_framingprobe
#SBATCH --account=gts-pt43
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# IRoLSD framing probe: render ONLY the canonical (i=0) frames of a few poses
# at each candidate range (80/160/320 m), so we can pick the range where Bepi
# fills ~60-70% of the 1024 frame BEFORE committing the full render. Renders
# ~5 frames/range instead of all 500 -> minutes, not hours.
#
# HARD RULE: qos=embers ONLY. Never inferno.
#
# Submit (from the login node, in $SIS):
#   sbatch --export=ALL,SIS=$HOME/scratch/sisifos/SISIFOS scripts/irolsd/embers_framing_probe.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
: "${SIS:=$HOME/scratch/sisifos/SISIFOS}"
: "${BLENDER:=$SIS/env/Blender_4.5/blender}"
: "${CFG:=$SIS/configs/irolsd_bepi.json}"
: "${NVARIANTS:=50}"
: "${PROBE_POSES:=0 3 6 9}"   # which pose groups to sample the canonical frame of

echo "host=$(hostname)"; nvidia-smi -L || true
cd "$SIS"

for R in 80 160 320; do
  AGENT="$SIS/renders/irolsd/pilot_R${R}/Config_1_RF_Bepi-mcs/Agent_0"
  [ -f "$AGENT/camera_traj.csv" ] || { echo "!!! missing CSV for R=$R ($AGENT)"; continue; }

  # canonical frame index of each probe pose = g*NVARIANTS + 0
  FRAMES=""
  for g in $PROBE_POSES; do FRAMES="${FRAMES:+$FRAMES,}$((g * NVARIANTS))"; done
  echo "=== R=$R : rendering canonical frames [$FRAMES] ==="

  # patch config: filepath -> this agent, render_frames on, only these frame_ids
  TMP="$AGENT/.probe_config.json"
  python3 - "$CFG" "$AGENT" "$FRAMES" "$TMP" <<'PY'
import json, sys
cfg, agent, frames, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = json.load(open(cfg)); b = c.get("base_config", c)
b["trajectory_type"] = "filepath"; b["trajectory_filepath"] = agent
b.setdefault("setup", {})["render_frames"] = True; b["setup"]["generate_video"] = False
b["frame_ids"] = [int(x) for x in frames.split(",")]
json.dump(c, open(out, "w"), indent=2)
PY

  FIXED="$AGENT/_render_out"; mkdir -p "$FIXED"
  # $CFG is a SWEEP file (base_config + sweep_parameters) so it must go through
  # --sweep_config_path, NOT --config_path (which would double-wrap it in another
  # base_config and silently fall back to template defaults -> wrong model).
  SISIFOS_OUTPUT_DIR="$FIXED" "$BLENDER" -b -P main.py -- --sweep_config_path "$TMP" \
    >>"$FIXED/probe.log" 2>&1 || { echo "!!! blender failed R=$R (see $FIXED/probe.log)"; continue; }
  echo "    rendered R=$R -> $FIXED"
done

echo "=== framing probe done; measuring fill fraction ==="
python3 - "$SIS" "$NVARIANTS" <<'PY'
import glob, os, sys, numpy as np
from PIL import Image
sis, nv = sys.argv[1], int(sys.argv[2])
for R in (80, 160, 320):
    agent = f"{sis}/renders/irolsd/pilot_R{R}/Config_1_RF_Bepi-mcs/Agent_0"
    pngs = sorted(glob.glob(os.path.join(agent, "_render_out", "**", "frame_*.png"), recursive=True))
    if not pngs:
        print(f"R={R:>3}: NO FRAMES"); continue
    fracs = []
    for p in pngs:
        im = np.asarray(Image.open(p).convert("L"))
        # spacecraft = non-black pixels; bbox extent / frame width
        ys, xs = np.where(im > 8)
        if xs.size == 0:
            fracs.append(0.0); continue
        w = im.shape[1]
        ext = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        fracs.append(ext / w)
    fracs = np.array(fracs)
    print(f"R={R:>3}m: {len(pngs)} frames, fill-frac (bbox/width) min/med/max = "
          f"{fracs.min():.2f}/{np.median(fracs):.2f}/{fracs.max():.2f}  "
          f"[target ~0.60-0.70]")
PY
echo "=== JOB DONE ==="
