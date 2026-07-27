# Generating SatSLAM datasets with SISIFOS

End-to-end runbook for rendering a **real-frontend SatSLAM dataset** (image
sequence + ground truth of a tumbling RSO) and wiring it into SatSLAM. This is the
workflow behind the JGCD ORB-vs-ALIKED experiment (Hubble / Integral / Bepi / Juice).
For the generic simulator quick-start see [../README.md](../README.md); for config
fields see [../modules/ConfigInfo.md](../modules/ConfigInfo.md).

## TL;DR pipeline
1. Define the target's **inertia** in `modules/spacecraft_defaults.yaml`.
2. Pick the **camera focal + range (R0_const)** so the target fills the frame
   without clipping (dial via a 5-frame inspect render).
3. Write a **SceneConfig JSON** (start from a prior render's `Config_1.json`).
4. Render: `env/Blender_4.5/blender -b -P main.py -- --config_path <cfg.json>`.
5. **Symlink** the output `Agent_0/` dir into `SatSLAM/datasets/<name>`.
6. Run SatSLAM (`pipelineSweep <mode3.yaml>`).

---

## 0. Prerequisites
- Blender bundled at `env/Blender_4.5/blender` (GPU/OptiX). Run from the SISIFOS root.
- Spacecraft meshes live in **`assets/spacecraft_models_0_51.blend`** (NOT the older
  `_0_5`). Models are `RF_*` roots (e.g. `RF_Hubble`, `RF_Integral`, `RF_Bepi-mcs`,
  `RF_IBEX_final`, `RF_Juice`). List them:
  ```bash
  env/Blender_4.5/blender -b --python-expr "import bpy; \
    f='assets/spacecraft_models_0_51.blend'; \
    print([n for n in bpy.data.libraries.load(f)[0].objects if n.startswith('RF_')])"
  ```
- Each model's mesh **must carry non-zero material/object pass indices** so
  vision_blender produces a foreground segmentation. ⚠️ `RF_IBEX_final` currently has
  all-zero indices → its masked images come out empty (renders fine in raw). Fix the
  indices in the blend before using IBEX.

## 1. Inertia — `modules/spacecraft_defaults.yaml`
Convention (`InertiaConfig`, `modules/config.py`): **X=highest, Y=intermediate,
Z=lowest** principal axis. Types: `box {m,l,w,h}`, `cylinder {m,r,h}`, `custom
{Jx,Jy,Jz}`, `sphere {m,r}`. Each entry also needs `model_rotation_euler {x,y,z}`
to align the mesh to the body frame (verify visually; tumble looks natural at 0,0,0).
Only inertia **ratios** matter (torque-free tumble), so absolute scale is irrelevant.
For mesh-derived values: export the model OBJ and compute a thin-shell (uniform areal
density) inertia — published spacecraft tensors are generally not available.

## 2. Imaging — focal length + range (the part you must tune)
Camera (fixed for this suite): `resolution [1024,1024]`, `pixel_size_um 6.469`.
Apparent size is set by **`camera.focal_length`** (mm) and **`trajectory.R0_const`**.
There is **no reliable closed-form** for fill (SISIFOS decouples R0 from true range
via the CRO trajectory) — and **OBJ/mesh extents do NOT predict render scale** (blend
models are at real, very different scales). **Always dial empirically** with an inspect
render, targeting **~70% max fill** across the tumble (leaves margin so other seeds
don't clip).

Locked values for the 4 JGCD models (camera as above):

| model | focal | R0_const |
|-------|-------|----------|
| RF_Hubble    | 80 mm | 100  |
| RF_Integral  | 40 mm | 38.5 |
| RF_Bepi-mcs  | 80 mm | 80   |
| RF_Juice     | 80 mm | 80   |
| RF_IBEX_final| 80 mm | ~80 (pending seg fix) |

**Inspect render** (5 views across the tumble, fast — renders only those frames):
set `frame_ids: [0,200,400,600,800]` and `generate_video:false`, render, then measure
fill from the masked `images/` bounding box. Lower R0 = bigger; raise to remove clipping.

## 3. The render recipe (clean, natural-lit, foreground-extracted)
Non-trajectory overrides that make a SatSLAM-ready dataset (see also the
`_RENDER_OVERRIDES` in `configs/synthetic-exp/_generate.py`):
- `setup.render_frames: true`, `setup.earth_mode: "off"`, `setup.stars_mode: "off"`
- `camera.resolution [1024,1024]`, `camera.pixel_size_um 6.469`, `camera.focal_length <per-model>`
- `render.crop_to_border_px: null` (full frame; a crop clips the target)
- `trajectory.EARTH_BACKGROUND_ENABLE: true` (sun in orbital-radial dir = natural lighting)
- `trajectory.SUN_ALIGN_ENABLE: false` (sun not camera-pinned)
- `save_segmentation: true` (REQUIRED — without it masking copies raw → bg noise floor)
- View transform is forced to **Standard** (linear, sensor-faithful) in `modules/renderer.py`
  (not AgX). Foreground masking uses `seg != 0` (`modules/io_utils.py`).

## 4. Build the SceneConfig JSON
Easiest: copy a prior render's expanded config (`renders/<ts>/Config_1.json`) and edit.
It's a flat `SceneConfig` you feed straight back. Key fields:
- `selected_model` = the `RF_*` name
- `objects.Target.blend_path = "assets/spacecraft_models_0_51.blend"`
- `camera.focal_length`, `trajectory.R0_const`  (per-model, §2)
- `trajectory.inertia_config` = the model's inertia (or let it auto-resolve from
  spacecraft_defaults when `inertia_type` is null)
- `trajectory.seed` (+ `illumination_seed`) — see §6
- `trajectory.tstep: 0.1`, `trajectory.tend = 0.1 * N_frames`  (e.g. 250.0 → 2500 frames)
- `trajectory.num_agents: 1`, `num_mc: 1` (single realization)
- `frame_ids: null` to render all frames (or a list for an inspect)
- `setup.generate_video: true/false`

## 5. Render + symlink
```bash
cd /home/jdflo/satslam/sisifos2/SISIFOS
env/Blender_4.5/blender -b -P main.py -- --config_path /tmp/<cfg>.json
# output: renders/<ts>/Config_1_<MODEL>/Agent_0/  (images/, images_raw/, gtValues.txt,
#         camera_traj.csv, GTAnnotations/, imgList.txt, sensormeasurements.txt, frames.mp4)
ln -sfn "$PWD/renders/<ts>/Config_1_<MODEL>/Agent_0" \
        /home/jdflo/satslam/SatSLAM/datasets/<name>
```
Throughput ≈ 2.7 s/frame on GPU (~1h50m per 2500-frame model).
Dataset naming used here: `{model}-{frames}-s{seed_index}` (e.g. `hubble-5000-s1`).

## 6. Multi-model, common-seed generation
For a controlled cross-model comparison: **same seed + same ω/COM/span, per-model
inertia/focal/R0**, so the inertia *shape* is the only varying physics. Seeds used:
s1=`1462526110`, s2=`2024051501` (from `configs/synthetic-exp/_generate.py` SEEDS).
- Hold `illumination_seed` **fixed** across seeds for a clean comparison (a new
  `illumination_seed` moves the sun → different lighting; that confounded s1 vs s2).
- Render is **deterministic**: same seed + tstep → identical poses regardless of `tend`
  (verified md5-identical). So you can render a longer arc and the prefix matches.

## 7. Resume / merge tooling (long renders)
Renders are written per-frame, so a render is resumable. Bundle in
`/home/jdflo/satslam/s1back_resume/` (persistent — `/tmp` is wiped on reboot):
- `resume_s1back.sh` — re-renders only missing frames. Detection filter:
  `dir-date >= 2026-06-06 AND seed==1462526110 AND tend==500` (REQUIRED — seed
  1462526110 is **reused** by old experiments, so seed alone is ambiguous).
- `merge_s1.sh <model|all>` — COPIES front half `[0,2499]` + back half `[2500,4999]`
  into a combined `[0,5000]` dataset under `renders/s1_combined_5000/`, symlinks
  `{model}-5000-s1`. Gathers back-half frames across all (possibly resumed) dirs.
  Uses the back half's authoritative full-5000 `camera_traj`/`gtValues`/`sensormeasurements`.

## 8. Gotchas
- **OBJ extents ≠ render scale** — never set R0 from mesh extents; inspect-render.
- **Reused seeds** — `1462526110` appears in many old renders; filter by date+tend too.
- **IBEX segmentation** — zero indices → empty masks (blend fix needed).
- **Clipping margin** — tune to ~70% max fill, not ~85% (other seeds tumble wider).
- **Config truth** — dataset mesh/scale truth is `renders/<ts>/Config_1.json`
  (`selected_model`, `model_scale_xyz`), not the dataset's `Config.yaml`.

## 9. Running SatSLAM on the dataset (real-frontend Mode 3)
```bash
cd /home/jdflo/satslam/SatSLAM
# pipeline.yaml: FrontEnd.featureType = ORB (default) or ALIKED (needs ALIKED-on build)
./build/bin/pipelineSweep scripts/sweeps/<mode3>.yaml   # datasets: [<name>...]; grid.useSynthetic:["0"]
```
Results: `output/pipeline_sweep/single/<run>/sweep_results.csv` (totalFrames, totalKFs,
totalLandmarks, normATE, omegaErrNorm, inertiaErrNorm, comErrNorm, success, failReason).
`featureType` is NOT a sweep grid axis — set it in `config/pipeline.yaml` and run two
batches (ORB then ALIKED). See SatSLAM `scripts/sweeps/README.md` (Mode 3).

## 10. Current state (2026-06-07)
Per model in `SatSLAM/datasets/`: `{model}-2500-s1`, `{model}-5000-s1` (merged full arc),
`{model}-5000-s2`. IBEX deferred (segmentation). Combined datasets at
`renders/s1_combined_5000/`. s2 has slight clipping + dimmer lighting (different
illumination_seed) — usable but s1 is the cleaner controlled set.
