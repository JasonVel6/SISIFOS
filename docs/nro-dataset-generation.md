# Generating NRO datasets with SISIFOS

End-to-end runbook for rendering a **long-range ("NRO") SatSLAM dataset** — a small
resident space object (12U-class) imaged from **kilometres away** through a realistic
space optic, with an **onboard-style windowed crop** around the target. This is the
unresolved / few-tens-of-pixels regime, distinct from the close-proximity JGCD set
(for that, see [satslam-dataset-generation.md](satslam-dataset-generation.md)). The
imaging math behind these numbers is in [imaging_math.md](imaging_math.md).

## TL;DR pipeline
1. Pick the **range** (3–15 km) and confirm the target's apparent size from the
   imaging math (a 1.0 m axis is ~30 px at 9 km).
2. Use the realistic **space optic** (650 mm, f/7.6, 5472×3648) — fixed for this regime.
3. Turn the **crop on**: `render.crop_to_border_px: [256, 256]` (onboard ROI window).
4. Render: `env/Blender_4.5/blender -b -P main.py -- --config_path <cfg.json>`
   (or `--sweep_config_path` for a range sweep).
5. **Symlink** the output `Agent_0/` dir into `SatSLAM/datasets/<name>`.

---

## 0. Why this is a separate regime
At km range a 12U target subtends only **tens of pixels**, so:
- the camera is a **real long-focal optic** (650 mm f/7.6), not the 25–80 mm prox-ops lens;
- the renderer crops to a small **ROI window** centred on the target (mimics onboard
  processing of a tracked window), so frames are `256×256`, not full-frame;
- **foreground masking is skipped** (see §4) — the cropped frame *is* the dataset image.

This is the opposite of the close-prox recipe, which renders full-frame at 1024² with
`crop_to_border_px: null` and extracts a `seg!=0` foreground.

## 1. Target — a 12U envelope (mesh proxy)
The NRO example configs use `selected_model: RF_Hubble` **compressed into a 12U
envelope** via `model_scale_xyz` (e.g. `[0.08778, 0.0101, 0.01395]` → a
1.0 × 0.2 × 0.2 m world-space box). `model_scale_xyz` multiplies every vertex, so the
rendered geometry is provably bounded by that envelope.

> The *internal shape* is still a (compressed) Hubble mesh, not a true CubeSat bus. For
> pixel-footprint and lighting work that is irrelevant; for shape-feature studies you'd
> want a real CubeSat asset. NRO example configs reference the older
> `assets/spacecraft_models_0_5.blend`.

## 2. Imaging — the long-range optic (fixed for this regime)
From [imaging_math.md](imaging_math.md). Camera block:
```
focal_length 650.0 mm   pixel_size_um 2.4   resolution [5472, 3648]
f_number 7.6   psf_fwhm_px 0.68   wavelength_nm 550   sensor_fit HORIZONTAL
```
This gives `f_px = 650 / 0.0024 = 270,833 px` and `IFOV = 3.7 µrad/px`. Apparent size of
an object of linear extent `L` at range `R` is `pixels = L · f_px / R`:

| range | 1.0 m long axis | 0.2 m short axis |
|---|---|---|
| 3 km | ~90 px | ~18 px |
| 6 km | ~45 px | ~9 px |
| 9 km | ~30 px | ~6 px |
| 12 km | ~23 px | ~4.5 px |
| 15 km | ~18 px | ~3.6 px |

(Measured bbox is ~60–80 % of geometric, from threshold + foreshortening — see
imaging_math §5.) **A 256-px crop comfortably contains the target at all of these
ranges** (unlike close-prox targets, which are ~500 px and would clip at 256).

## 3. The crop — onboard ROI window (the NRO-specific feature)
Set in the `render` block:
```
"render": { "crop_to_border_px": [256, 256], "engine": "CYCLES", "samples": 16 }
```
Behaviour (`modules/renderer.py`):
- Cycles renders **only** the `(crop_w, crop_h)` window centred on the principal point
  (`scene.render.use_border` + `border_min/max`); the output PNG is exactly that size.
- **Camera intrinsics stay full-frame** — the crop is a window into the full model, not a
  new camera. The crop origin/size are exposed via `render_crop_info()`.
- `main.py` writes **`crop_info.json`** once per agent folder:
  `full_frame_resolution_px`, `crop_origin_px`, `crop_size_px`,
  `crop_principal_point_in_crop_px`, and full-frame `focal_length_mm/px`,
  `sensor_width_mm`, `pixel_size_um`. The SLAM frontend uses this to map cropped pixel
  coordinates back onto the full-frame camera model.
- `crop_to_border_px: null` disables it (→ full-frame, the close-prox behaviour).

## 4. Foreground masking is SKIPPED when cropped
`vision_blender`'s depth/segmentation come back at **full-frame** resolution, which can't
index the small cropped image — so `modules/io_utils.py` **copies the raw render through
unmasked** for cropped frames (and onboard-style frames don't need masking anyway).
Consequence: for cropped NRO datasets, `images/` ≈ `images_raw/` (no `seg!=0` extraction).
If you need masked NRO frames, render full-frame (`crop_to_border_px: null`) and mask, then
crop in post — but the standard NRO recipe is crop-on / mask-skipped.

## 5. Trajectory — two config styles
Both are in `configs/nro/`:

- **Tumbling at a fixed range** (SLAM datasets) — `cubesat_12u_9km_tumbling_1000.json`
  (and 3 km / 6 km variants). `trajectory_type: trajectory_generator`,
  `trajectory.R0_const` = the range (e.g. `9000.0`), `path_mode: tumbling`, EARTH-
  background natural lighting (`EARTH_BACKGROUND_ENABLE: true`, `SUN_ALIGN_ENABLE: true`),
  `earth_mode: off` (target on black). `tend/tstep` sets frame count (100 s / 0.1 = 1000).
- **Distance sweep** (apparent-size characterization) — `cubesat_12u_distance_sweep.json`.
  `trajectory_type: sampling_trajectory`, swept over range via
  `sweep_parameters: {"trajectory_sampling.R_RPO": [500, 3000, 6000, 9000, 12000, 15000]}`
  (Fibonacci-sphere viewpoints). One render per range; for the imaging study, not SLAM.

## 6. Render + symlink
```bash
cd /home/jdflo/satslam/sisifos2/SISIFOS
# single config (fixed-range tumbling):
env/Blender_4.5/blender -b -P main.py -- --config_path configs/nro/cubesat_12u_9km_tumbling_1000.json
# range sweep (one render per R_RPO):
env/Blender_4.5/blender -b -P main.py -- --sweep_config_path configs/nro/cubesat_12u_distance_sweep.json

# output: renders/<ts>/Config_1_<MODEL>/Agent_0/  (images/, images_raw/, gtValues.txt,
#         camera_traj.csv, crop_info.json, GTAnnotations/, imgList.txt, sensormeasurements.txt)
ln -sfn "$PWD/renders/<ts>/Config_1_<MODEL>/Agent_0" /home/jdflo/satslam/SatSLAM/datasets/<name>
```
For long runs, the phased/resumable renderer applies here too (see
`/home/jdflo/satslam/phased_render/`): it banks frames and resumes after a shutdown.

## 7. Verify
- `crop_info.json` present, `crop_size_px == [256,256]`, `full_frame_resolution_px ==
  [5472,3648]`.
- A frame is `256×256` and the target is a small, lit blob (tens of px) near centre — not
  empty, not clipped at the crop edge.
- `gtValues.txt` `nSamples` == intended frame count; `omega_GI_G` matches the config tumble.
- **Mesh truth** is `renders/<ts>/Config_1.json` (`selected_model` + `model_scale_xyz`),
  NOT the dataset `Config.yaml` (whose `ShapeModel.Filename` is a stale template).

## 8. Gotchas
- **Crop is regime-specific** — ON (256) for NRO, OFF (null) for close-prox (256 clips
  close-prox targets). Don't copy a crop value across regimes.
- **No masking when cropped** (§4) — `images/` is the raw cropped frame.
- **Range vs apparent size** — set range from the §2 table, but verify with a 1–5 frame
  render; `model_scale_xyz` and foreshortening shift the measured footprint.
- **Older blend** — NRO examples use `spacecraft_models_0_5.blend` (12U envelope proxy),
  not the `_0_51` library used by the close-prox real-spacecraft set.

## 9. Downstream consumption
NRO datasets (`datasets/nro-*km`) feed the **SatSLAM RelDyn backend** for long-range
inertia/ω recovery. At km range, recovery is dt-gated (not range-gated): the sweet spot is
~1 s keyframe spacing. See the SatSLAM long-range write-ups; note the RelDynFactor 1 km
magnitude heuristic must be relaxed (threshold bumped to 1e6 m) so realistic km-scale
geometries aren't misclassified as absolute.
