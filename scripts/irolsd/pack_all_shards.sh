#!/usr/bin/env bash
# Pack ALL rendered shards into ONE flat IRoLSD contract dataset.
#
#   [SIS=...] [OUT=...] scripts/irolsd/pack_all_shards.sh
#     OUT default: $SIS/datasets/irolsd/bepi_train
#
# Each shard's pack_contract.py reads pose_start from its sweep_meta.json and
# offsets frame/pose indices to GLOBAL, so shards merge without collision. Images
# are symlinked (fast, no copy); depth is written as depth_{g}.npy. Idempotent.
set -euo pipefail
SIS="${SIS:-$HOME/scratch/sisifos/SISIFOS}"
BASE="$SIS/renders/irolsd/bepi_train"
OUT="${OUT:-$SIS/datasets/irolsd/bepi_train}"
cd "$SIS"

shards=$(ls -d "$BASE"/shard_*/ 2>/dev/null | sort)
[ -z "$shards" ] && { echo "!!! no shards under $BASE"; exit 2; }
mkdir -p "$OUT"

for s in $shards; do
  AGENT="$s/Config_1_RF_Bepi-mcs/Agent_0"
  echo "=== packing $(basename "$s") ==="
  python3 scripts/irolsd/pack_contract.py --agent "$AGENT" --out "$OUT" || echo "  (shard incomplete? continuing)"
done

echo "=== merged dataset: $OUT ==="
echo "  images: $(ls "$OUT"/img_*.png 2>/dev/null | wc -l)   depth: $(ls "$OUT"/depth_*.npy 2>/dev/null | wc -l)"
echo "  -> hand this path to the label-gen track:"
echo "     python label_gen/batch_label_gen.py --images $OUT --out <labels> --n-variants 50"
