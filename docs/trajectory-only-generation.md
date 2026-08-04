# Generating trajectories without Blender

Runbook for producing SISIFOS trajectories + ground truth when you never need
the rendered images. No Blender install, no GPU, no spacecraft `.blend` assets.

For the rendering workflow see [satslam-dataset-generation.md](satslam-dataset-generation.md);
for config fields see [../modules/ConfigInfo.md](../modules/ConfigInfo.md).

## 1. Install (once)

```bash
git clone <sisifos repo> && cd SISIFOS
python3 -m venv ~/.venvs/sisifos-traj
~/.venvs/sisifos-traj/bin/pip install -r requirements-trajectory.txt
```

Six packages, ~30 s. `mathutils` is Blender's math library, published standalone
on PyPI; it builds from source, so a C/C++ toolchain is required (`build-essential`
on Debian/Ubuntu). The pin matters: `mathutils` 5.x needs Python 3.13+, so on 3.10 /
3.11 the requirements file selects 3.3.0 automatically.

You do **not** need `assets/` — no `.blend`, no starmap. Nothing in this path opens
them, and the asset paths in a config may point at files that do not exist.

## 2. Generate

```bash
~/.venvs/sisifos-traj/bin/python generate_trajectories.py \
    --sweep_config_path configs/probe_hybrid.json
```

Same flags as `main.py`: `--config_path` for a flat `SceneConfig`,
`--sweep_config_path` for a `{base_config, sweep_parameters}` file. Sweep
expansion, `$SISIFOS_OUTPUT_DIR`, and the `renders/<timestamp>` default all
behave as they do under Blender. `--output_dir` overrides the destination.

Runs from any working directory. `setup.render_frames` is ignored (a `true`
value logs a warning); everything else in the config is honoured.

### Output

```text
<out>/Config_<i>.json               expanded SceneConfig (reproducibility record)
<out>/Config_<i>_trajectory.json    resolved TrajectoryConfig, incl. auto-generated seeds
<out>/sweep_configs.json            the source config, verbatim
<out>/run.log
<out>/Config_<i>_<MODEL>/MC_000_traj.png
<out>/Config_<i>_<MODEL>/Agent_<k>/
    camera_traj.csv          timestamp, p_G_I, q_I_G, p_C_I, q_I_C, sun_az, sun_el
    gtValues.txt             inertia, COM offset, attitudes, omegas, relative states
    sensormeasurements.txt   noisy star-tracker / gyro / accelerometer streams
    Config.yaml              OpenCV-YAML intrinsics + fps for SatSLAM
    imgList.txt              timestamps paired with the filenames a render would produce
```

`Agent_<k>/` is the unit SatSLAM consumes — symlink it into `SatSLAM/datasets/<name>`.

Byte-for-byte identical to what `blender -b -P main.py` writes for the same
config; the two share the generator and differ only in whether images follow.

## 3. Also accepts a bare TrajectoryConfig

`Config_<i>_trajectory.json` can be fed straight back in. That shape carries no
camera block, and **camera intrinsics feed the initial-condition sampler**
(`focal_length_px` gates the frame-to-frame pixel-motion floor `MIN_F2F_PX_MED`),
so `CameraConfig` defaults would silently produce a different trajectory for the
same seed. Borrow a real camera:

```bash
python generate_trajectories.py --config_path out/Config_1_trajectory.json \
    --camera_from configs/probe_hybrid.json
```

## 4. Reproducibility

`seed` alone is not enough. `illumination_seed: null` resolves to a fresh value on
every invocation, which moves the sun and — with `SUN_ALIGN_ENABLE` — the camera
geometry with it. Pin **both** to reproduce a run exactly. Both resolved values are
recorded in `Config_<i>_trajectory.json`.

Configs that pin both (e.g. `configs/probe_hybrid.json`) reproduce byte-for-byte;
`configs/ci_test.json` and `configs/examples/example_trajectory_generation.json`
pin only `seed`.

## 5. Tests

```bash
~/.venvs/sisifos-traj/bin/pip install pytest
~/.venvs/sisifos-traj/bin/python -m pytest tests -q
```

The two Blender-only modules skip automatically, and no assets are needed —
verified on a checkout with no spacecraft models, no starmap and no CI assets.

Under Blender the full suite runs via `blender -b -P tests/run_in_blender.py`, but
first generate the CI assets once:

```bash
blender -b -P tests/create_ci_assets.py    # writes assets/minimal_{scene,cube,sphere}.blend
```

`configs/ci_test.json` references those three files and they are gitignored, so
`test_e2e_pipeline` fails with `Cannot read file ".../minimal_scene.blend"` on a
fresh checkout until you generate them.

## 6. Gotchas

- **`configs/_TEMPLATE.json` is a skeleton to copy and edit, not a runnable config.**
  As shipped its `MIN_F2F_PX_MED: 3.0` is unreachable at its optics/range, so running
  it directly raises `RuntimeError: Tumbling IC unrecoverable after phase search`
  (the sampler tops out at 2.97 px). Set the fields for your scenario after copying;
  for a config that runs as-is, start from `configs/probe_hybrid.json`.
- **`RuntimeError: ... IC unrecoverable after phase search`** in general means the
  requested `MIN_F2F_PX_MED` is unreachable for that camera/range geometry — lower
  the pixel-motion floor or `R0_const`.
- **`omega_min_deg`/`omega_max_deg` left `null`** do not fall back to `init_tumbling`'s
  documented 0.5-2 deg/s — the unified generator overrides to **3-5 deg/s**.
- **No `bpy` in this path.** If an import error mentions `bpy`, something imported a
  render-side module; the Blender-free helpers live in `modules/path_utils.py`.
