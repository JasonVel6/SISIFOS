# IRoLSD training-data render (BepiColombo, fixed-pose / swept-Sun + depth)

Produces the render half of the IRoLSD retraining set: for each of ~1000 fixed
camera poses around **RF_Bepi-mcs**, render N illumination variants (same pose,
Sun perturbed) + one spacecraft-only depth map. Output is laid out in the exact
flat contract the label-gen (`irolsd/label_gen/irolsd_label_gen.py`) consumes.

```
<out>/bepi_train/
  img_{g*NVARIANTS+i}.png   # pose g, illumination variant i (i=0 = canonical)
  depth_{g}.npy             # pose g depth, float32 HxW, spacecraft only (Earth excluded)
```

## Pieces

| file | runs where | what |
|------|-----------|------|
| `gen_fixedpose_sweptsun.py` | login node (CPU, fast) | writes `camera_traj.csv` (pose fixed per group, Sun swept) + `sweep_meta.json` |
| `configs/irolsd_bepi.json`  | — | render config: Bepi, earth/stars off, save_depth, 1024² crop |
| `render_irolsd.sh`          | compute (GPU) | resumable Blender render of every CSV row |
| `embers_render_irolsd.sbatch` | PACE embers | sbatch wrapper for the render driver |
| `pack_contract.py`          | login node (CPU) | remaps SISIFOS output → flat `img_/depth_` contract |
| `pace_preflight.sh`         | PACE login | verifies assets/scripts/Blender present on PACE |

**NVARIANTS must match the label-gen's grouping** (`number // NVARIANTS = pose`).
Default 20 (the paper's number; label-gen `n_illum` default is 20). Set the same
value on both tracks and confirm before the full render.

## Geometry / knobs

- Target at origin, camera at `--range` m on a Fibonacci-sphere viewpoint (+ jitter),
  looking at the target. Range/view-direction are preserved through SISIFOS's
  earth-distance scaling. **Range must be tuned to Bepi's true mesh extent** — do
  this from the pilot images (the f=25 mm lens is tight; see pilot step 2).
- Sun: canonical az/el jittered per group (`--sun-group-jitter-deg`, default 40°)
  so groups see varied lighting; variants perturb ±`--sun-delta-deg` (default 1°,
  the paper's δ∈[−1,1]°) around that base.
- Depth: `earth_mode:off` ⇒ Earth not rendered ⇒ depth is spacecraft-only (Eq. 24).
  Background (`-1`) is replaced by the far spacecraft depth in `pack_contract.py`
  so the label-gen's `(d-min)/ptp` normalization stays clean.

## Pilot first (10 poses × 20 Sun, ~200 frames) — validate the contract end-to-end

```bash
# 0. On PACE: verify state (run in your interactive session)
PACE_SIS=$HOME/scratch/sisifos/SISIFOS bash scripts/irolsd/pace_preflight.sh
#    -> push anything it flags MISSING from the local box (rsync lines printed).

# 1. Generate a small multi-range pilot to pick framing. Three ranges, 10 poses each:
SIS=$HOME/scratch/sisifos/SISIFOS
for R in 80 160 320; do
  python3 scripts/irolsd/gen_fixedpose_sweptsun.py \
    --out $SIS/renders/irolsd/pilot_R$R/Config_1_RF_Bepi-mcs/Agent_0 \
    --nposes 10 --nvariants 20 --range $R --seed 12345
done

# 2. Render each pilot on embers (resumable). Inspect a few img_*.png per range:
for R in 80 160 320; do
  sbatch --export=ALL,SIS=$SIS,AGENT=$SIS/renders/irolsd/pilot_R$R/Config_1_RF_Bepi-mcs/Agent_0 \
    scripts/irolsd/embers_render_irolsd.sbatch
done
#    -> pick the range where Bepi fills ~60-70% of the 1024 frame, centered, not clipped.

# 3. Pack the chosen pilot + hand the path to the label-gen track:
python3 scripts/irolsd/pack_contract.py \
  --agent $SIS/renders/irolsd/pilot_R160/Config_1_RF_Bepi-mcs/Agent_0 \
  --out   $DATASETS/irolsd/bepi_pilot
#    label-gen runs on $DATASETS/irolsd/bepi_pilot with NVARIANTS=20. Confirm labels look right.
```

## Full render (~1000 poses × 20 Sun ≈ 20k frames)

```bash
SIS=$HOME/scratch/sisifos/SISIFOS
AGENT=$SIS/renders/irolsd/bepi_train/Config_1_RF_Bepi-mcs/Agent_0
python3 scripts/irolsd/gen_fixedpose_sweptsun.py \
  --out $AGENT --nposes 1000 --nvariants 20 --range <PILOT_WINNER> --seed 12345

# embers is preemptible + 8h cap. render_irolsd.sh is RESUMABLE, so just resubmit
# until it prints "all N frames present" (chain, or an array of resubmits):
sbatch --export=ALL,SIS=$SIS,AGENT=$AGENT scripts/irolsd/embers_render_irolsd.sbatch
# ... resubmit on preemption/timeout; it continues from disk ...

# Pack once rendering completes:
python3 scripts/irolsd/pack_contract.py --agent $AGENT --out $DATASETS/irolsd/bepi_train
```

Report `$DATASETS/irolsd/bepi_train` to the label-gen track. That's the training set.

## Notes / gotchas

- `discard frames where >80% of the spacecraft is near-black` (paper §5.1): with
  Sun jittered per group most frames are lit, but the label-gen can drop dark
  canonicals; if we want to enforce it at render time, add a brightness gate to
  `pack_contract.py` (not on by default — confirm with label-gen first).
- The `filepath` trajectory mode renders into `<AGENT>/_render_out/Config_1_<model>/…`;
  `pack_contract.py --agent` can point at either `<AGENT>` or that inner dir
  (it searches the tree).
- HARD RULE: **embers QOS only.** Never inferno. The sbatch pins `--qos=embers`.
