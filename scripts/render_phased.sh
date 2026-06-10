#!/usr/bin/env bash
# Phased / resumable SISIFOS render.
#
#   [SIS=<sisifos_root>] [DATASETS=<dir>] [BLENDER=<path>] \
#     scripts/render_phased.sh <config.json> <tag> [bank_size]
#
# Renders the full trajectory in <config.json> into ONE fixed folder
# ($SIS/renders/_phased/<tag>/), in banks of <bank_size> frames (default 500),
# and symlinks the result to $DATASETS/<tag>. IDEMPOTENT + RESUMABLE: re-run the
# same command and it renders only the frames not yet on disk. Kill / shut the
# machine down at any time; nothing is lost (a frame that was mid-render is just
# re-rendered). Requires the SISIFOS_OUTPUT_DIR hook in main.py:run_sweep.
#
# Env vars (all optional):
#   SIS       SISIFOS repo root. Defaults to this script's repo (../ from here).
#   DATASETS  where to symlink the finished dataset (default: $SIS/datasets).
#   BLENDER   Blender 4.5 binary (default: $SIS/env/Blender_4.5/blender).
set -u
SIS="${SIS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASETS="${DATASETS:-$SIS/datasets}"
BLENDER="${BLENDER:-$SIS/env/Blender_4.5/blender}"

CFG="$(readlink -f "${1:?usage: render_phased.sh <config.json> <tag> [bank]}")"
TAG="${2:?usage: render_phased.sh <config.json> <tag> [bank]}"
BANK="${3:-500}"
FIXED="$SIS/renders/_phased/$TAG"
mkdir -p "$FIXED" "$DATASETS"
cd "$SIS"

read -r MODEL N < <(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); t=c["trajectory"]
print(c["selected_model"], int(round(float(t["tend"])/float(t["tstep"]))))
PY
)
AGENT="$FIXED/Config_1_${MODEL}/Agent_0"
echo "=== phased render: tag=$TAG model=$MODEL frames=$N bank=$BANK -> $AGENT ==="

while true; do
  MISSING=$(python3 - "$AGENT" "$N" "$BANK" <<'PY'
import sys,glob,os,re
agent,N,bank=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
done=set()
for p in glob.glob(os.path.join(agent,'images','frame_*.png')):
    mo=re.search(r'(\d+)',os.path.basename(p))
    if mo: done.add(int(mo.group(1)))
miss=[i for i in range(N) if i not in done]
sys.stderr.write(f"{len(done)}/{N} done, {len(miss)} remaining\n")
print(','.join(str(i) for i in miss[:bank]))
PY
)
  [ -z "$MISSING" ] && { echo ">>> all $N frames present."; break; }
  echo ">>> rendering bank of $(echo "$MISSING" | tr ',' '\n' | wc -l) frames  [$(date +%H:%M:%S)]"
  TMP="$FIXED/.bank_config.json"
  python3 - "$CFG" "$MISSING" "$TMP" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
c["frame_ids"]=[int(x) for x in sys.argv[2].split(',')]
c.setdefault("setup",{})["render_frames"]=True
c["setup"]["generate_video"]=False
json.dump(c,open(sys.argv[3],"w"),indent=2)
PY
  SISIFOS_OUTPUT_DIR="$FIXED" "$BLENDER" -b -P main.py -- --config_path "$TMP" >>"$FIXED/render.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "!!! blender exited $rc (see $FIXED/render.log) -- re-run to retry"; exit $rc; }
done

python3 - "$AGENT" "$N" <<'PY'
import sys,os,csv
agent,N=sys.argv[1],int(sys.argv[2])
ts=[r[0] for r in csv.reader(open(os.path.join(agent,'camera_traj.csv')))][1:]
with open(os.path.join(agent,'imgList.txt'),'w') as f:
    for i in range(N): f.write(f"{float(ts[i]):.6f} images/frame_{i:04d}.png\n")
print(f"rebuilt imgList.txt ({N})")
PY
ln -sfn "$AGENT" "$DATASETS/$TAG"
echo "=== DONE: $TAG  images=$(ls "$AGENT"/images/frame_*.png|wc -l)  -> $DATASETS/$TAG ==="
