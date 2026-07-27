#!/usr/bin/env bash
# Resumable IRoLSD fixed-pose/swept-Sun render driver.
#
#   [SIS=<sisifos_root>] [BLENDER=<path>] \
#     scripts/irolsd/render_irolsd.sh <config.json> <agent_dir> [bank_size]
#
# Renders EVERY row of <agent_dir>/camera_traj.csv (produced by
# gen_fixedpose_sweptsun.py) into <agent_dir>/images_raw/... , in banks of
# <bank_size> frames (default 500). IDEMPOTENT + RESUMABLE: re-run and it renders
# only the frames not yet on disk. Unlike scripts/render_phased.sh it does NOT
# derive the frame count from tend/tstep (there is no dynamical trajectory here);
# it counts CSV rows instead. The config's trajectory_filepath must point at
# <agent_dir> (or this script patches it for you).
#
# Env (all optional):
#   SIS      SISIFOS repo root (default: ../../ from this script).
#   BLENDER  Blender 4.5 binary (default: $SIS/env/Blender_4.5/blender).
set -u
SIS="${SIS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BLENDER="${BLENDER:-$SIS/env/Blender_4.5/blender}"

CFG="$(readlink -f "${1:?usage: render_irolsd.sh <config.json> <agent_dir> [bank]}")"
AGENT="$(readlink -f "${2:?usage: render_irolsd.sh <config.json> <agent_dir> [bank]}")"
BANK="${3:-500}"
cd "$SIS"

[ -f "$AGENT/camera_traj.csv" ] || { echo "!!! no camera_traj.csv in $AGENT (run the generator first)"; exit 2; }
N=$(($(wc -l < "$AGENT/camera_traj.csv") - 1))   # minus header
echo "=== IRoLSD render: agent=$AGENT frames=$N bank=$BANK ==="

# Patch the config so trajectory_filepath points at $AGENT and render_frames on.
PATCHED="$AGENT/.render_config.json"
python3 - "$CFG" "$AGENT" "$PATCHED" <<'PY'
import json, sys
cfg, agent, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = json.load(open(cfg))
b = c.get("base_config", c)
b["trajectory_type"] = "filepath"
b["trajectory_filepath"] = agent
b.setdefault("setup", {})["render_frames"] = True
b["setup"]["generate_video"] = False
json.dump(c, open(out, "w"), indent=2)
print(f"[cfg] patched -> {out}  (trajectory_filepath={agent})")
PY

# The 'filepath' trajectory mode symlinks the source Agent dir contents into
# <output_dir>/Config_1_<model>/. We render into a fixed output dir so banks
# accumulate. images_raw leaf depends on earth/stars modes (both off here ->
# images_raw/Earth_Stars_OFF/Stars_OFF).
FIXED="$AGENT/_render_out"
mkdir -p "$FIXED"

while true; do
  MISSING=$(python3 - "$AGENT" "$N" "$BANK" <<'PY'
import sys, glob, os, re
agent, N, bank = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
done = set()
for p in glob.glob(os.path.join(agent, '_render_out', '**', 'frame_*.png'), recursive=True):
    m = re.search(r'frame_(\d+)', os.path.basename(p))
    if m:
        done.add(int(m.group(1)))
miss = [i for i in range(N) if i not in done]
sys.stderr.write(f"{len(done)}/{N} done, {len(miss)} remaining\n")
print(','.join(str(i) for i in miss[:bank]))
PY
)
  [ -z "$MISSING" ] && { echo ">>> all $N frames present."; break; }
  echo ">>> rendering bank of $(echo "$MISSING" | tr ',' '\n' | wc -l) frames  [$(date +%H:%M:%S)]"

  # Split the bank into CANONICAL (n % NVARIANTS == 0 -> full GT incl. depth+seg,
  # one per pose) and VARIANT (RGB-only). IRoLSD consumes exactly ONE depth per
  # pose (from the canonical i=0 frame) + the RGB of every variant, so depth/seg
  # on the 49 variants is wasted compute. NVARIANTS comes from sweep_meta.json.
  NVARIANTS=$(python3 -c "import json;print(json.load(open('$AGENT/sweep_meta.json'))['nvariants'])" 2>/dev/null || echo 1)
  for PASS in canonical variant; do
    FIDS=$(python3 - "$MISSING" "$NVARIANTS" "$PASS" <<'PY'
import sys
miss=[int(x) for x in sys.argv[1].split(',') if x!='']
nv=int(sys.argv[2]); pas=sys.argv[3]
sel=[n for n in miss if (n % nv == 0) == (pas=='canonical')]
print(','.join(map(str,sel)))
PY
)
    [ -z "$FIDS" ] && continue
    TMP="$AGENT/.bank_${PASS}.json"
    # canonical: keep config's save_depth/save_segmentation (ON). variant: force OFF.
    python3 - "$PATCHED" "$FIDS" "$PASS" "$TMP" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); b = c.get("base_config", c)
b["frame_ids"] = [int(x) for x in sys.argv[2].split(',')]
if sys.argv[3] == "variant":
    b["save_depth"] = False; b["save_segmentation"] = False
json.dump(c, open(sys.argv[4], "w"), indent=2)
PY
    echo "    [$PASS] $(echo "$FIDS" | tr ',' '\n' | wc -l) frames"
    SISIFOS_OUTPUT_DIR="$FIXED" "$BLENDER" -b -P main.py -- --sweep_config_path "$TMP" >>"$FIXED/render.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { echo "!!! blender exited $rc on $PASS pass (see $FIXED/render.log)"; exit $rc; }
  done
done

echo "=== DONE: $N frames -> $FIXED ==="
echo "    next: python3 scripts/irolsd/pack_contract.py --agent <the Config_1 agent under $FIXED> --out <dataset>/bepi_train"
