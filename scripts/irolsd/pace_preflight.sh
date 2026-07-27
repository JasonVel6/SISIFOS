#!/usr/bin/env bash
# PACE pre-flight for the IRoLSD BepiColombo render.
#
# Run this IN YOUR INTERACTIVE PACE SESSION (login node is fine; it only checks
# files + a tiny Blender probe -- no GPU compute, no credits). It verifies that
# ~/scratch/sisifos/SISIFOS is current: the 0_51 blend (Bepi), starmap, Earth/
# scene blends, the irolsd scripts, and that Blender launches.
#
#   PACE_SIS=$HOME/scratch/sisifos/SISIFOS bash pace_preflight.sh
#
# If anything is MISSING, the printed rsync commands (run FROM THE LOCAL box,
# pushing to phoenix) bring PACE up to date. This script does not itself rsync
# (PACE can't pull from your laptop); it tells you exactly what to push.
set -u
SIS="${PACE_SIS:-$HOME/scratch/sisifos/SISIFOS}"
BLENDER="${BLENDER:-$SIS/env/Blender_4.5/blender}"
echo "=== PACE pre-flight: SIS=$SIS ==="

need=()
check() {  # check <path> <label>
  if [ -e "$1" ]; then
    printf "  [ok]  %-28s %s\n" "$2" "$(du -h --apparent-size "$1" 2>/dev/null | cut -f1)"
  else
    printf "  [MISSING] %-24s %s\n" "$2" "$1"
    need+=("$1")
  fi
}

check "$SIS/main.py"                                "repo (main.py)"
check "$SIS/assets/spacecraft_models_0_51.blend"   "Bepi blend (0_51)"
check "$SIS/assets/scene.blend"                    "scene.blend"
check "$SIS/assets/Earth.blend"                    "Earth.blend"
check "$SIS/assets/starmap_2020_16k.exr"           "starmap EXR"
check "$SIS/scripts/irolsd/gen_fixedpose_sweptsun.py" "gen script"
check "$SIS/scripts/irolsd/render_irolsd.sh"       "render driver"
check "$SIS/configs/irolsd_bepi.json"              "render config"
check "$BLENDER"                                   "Blender binary"

echo
if [ "${#need[@]}" -ne 0 ]; then
  echo ">>> ${#need[@]} item(s) missing on PACE. Push them FROM THE LOCAL box:"
  echo "    # (run these locally, not on PACE)"
  echo "    LOCAL=\$HOME/satslam/sisifos2/SISIFOS"
  echo "    rsync -avz --progress \"\$LOCAL/assets/spacecraft_models_0_51.blend\" phoenix:$SIS/assets/"
  echo "    rsync -avz --progress \"\$LOCAL/assets/scene.blend\" \"\$LOCAL/assets/Earth.blend\" \"\$LOCAL/assets/starmap_2020_16k.exr\" phoenix:$SIS/assets/"
  echo "    rsync -avz \"\$LOCAL/scripts/irolsd/\" phoenix:$SIS/scripts/irolsd/"
  echo "    rsync -avz \"\$LOCAL/configs/irolsd_bepi.json\" phoenix:$SIS/configs/"
  echo "    rsync -avz \"\$LOCAL/main.py\" \"\$LOCAL/modules/\" phoenix:$SIS/   # if repo stale"
else
  echo ">>> all render inputs present on PACE."
fi

echo
echo "=== Blender launch probe (no GPU, no credits) ==="
if [ -x "$BLENDER" ]; then
  "$BLENDER" -b --version 2>/dev/null | head -1 || echo "  !!! Blender failed to launch"
  echo "  (For a GPU/OptiX confirm, run a 1-frame render under an embers salloc -- see runbook.)"
else
  echo "  !!! Blender not found at $BLENDER"
fi

echo
echo "=== free space on scratch ==="
df -h "$SIS" 2>/dev/null | tail -1
echo "  (full 1000-pose x 20-var 1024^2 render ~ 20k frames. Budget ~1-2 KB/PNG + depth NPZ; check headroom.)"
