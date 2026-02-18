# SISIFOS

**S**pecialized **I**llumination **SI**mulator **F**or **O**rbiting **S**pacecraft (**SISIFOS**) is a Blender-based spacecraft image simulator for generating photorealistic RGB images and annotations for spacecraft Rendezvous and Proximity Operations (RPO) scenarios.

SISIFOS can be used with spacecraft models from sources such as **ESA SciFleet** and **NASA 3D resources**, and with textures (e.g., Earth imagery) from sources such as **NASA Earth Observatory**. **Each asset retains its own license/terms**—you must comply with the specific terms attached to every model/texture you download and use.

This project is an independent research tool and is **not affiliated with, endorsed by, or sponsored by** NASA, ESA, or any other organization.

## Quick start

### 1) Setup & Activate Environment

The project uses a two-step process: **Setup** (one-time installation) and **Activation** (per session).

#### Windows

**First-time Setup:**
Downloads Blender, the Starmap asset, and installs dependencies.
```powershell
.\env\Setup.ps1
```

**Activate Environment:**
Run this whenever you start a new session. It will automatically sync dependencies if the lockfile changes.

```powershell
.\env\Activate.ps1
```

#### MacOS & Linux

**First-time Setup:**
Downloads Blender, the Starmap asset, and installs dependencies.

```bash
source ./env/setup.sh
```

**Activate Environment:**
Run this whenever you start a new session.

```bash
source ./env/activate.sh
```

---

**What happens under the hood?**

1. **Setup:** Downloads **Blender 4.5.6**, the **NASA Starmap (16k)**, bootstraps the Python environment, and installs dependencies via `uv`.
2. **Activation:** Adds the bundled Blender and Python to your `PATH` and sets required environment variables.

**To deactivate:**
Simply run:

```bash
deactivate
```

### 2) Prepare your assets

The **Setup** script automatically downloads the required starmap to `assets/starmap_2020_16k.exr`.

However, this repo does **not** ship with spacecraft models (e.g., ESA/NASA models). You must provide your own spacecraft model as a `.blend` file:

1. Place your model in the `assets/` folder (e.g., `assets/spacecraft_models.blend`).
2. Ensure the file structure looks like this:
```text
assets/
├── starmap_2020_16k.exr  (Downloaded by Setup)
└── spacecraft_models.blend (Provided by you)

```



### 3) Configure your scene

Edit the config file `configs/examples/config_example_basic.json` to match your filenames:

* `scene_blend_path`
* `hdri_path` (should point to `assets/starmap_2020_16k.exr` or your own starmap)
* Each object's `blend_path`

### 4) Generate a reference trajectory file

SISIFOS expects a reference trajectory text file containing the target pose, camera pose, and sun angles per frame.

File format (CSV):

```
timestamp, p_G_I_x, p_G_I_y, p_G_I_z, q_I_G_w, q_I_G_x, q_I_G_y, q_I_G_z, p_C_I_x,p_C_I_y, p_C_I_z, q_I_C_w, q_I_C_x, q_I_C_y, q_I_C_z, sun_az, sun_el
```

Where:

* p_G_I = position of target (G) in inertial frame (I) at radius R_LEO
* q_I_G = orientation of target frame (G) relative to inertial (I), quaternion wxyz
* p_C_I = position of camera (C) in inertial frame (I) at radius (R_LEO + R_RPO)
* q_I_C = orientation of camera frame (C) relative to inertial (I), quaternion wxyz
* sun_az, sun_el = sun azimuth/elevation (recommended: degrees, document your choice and keep it consistent)
* Earth / clouds / atmosphere stay at inertial origin (0,0,0) with fixed orientation.

### 5) Run the simulator

After activating the environment, run the simulator from the root of the repo:

```powershell
$BLENDER -b -P main.py -- --sweep_config_path configs/examples/example_trajectory_generation.json
```

Outputs, unless modified in the code, go to:

```
renders/<timestamp>/
```

### 6) Enable GT exports (optional)

If you have the [Vision Add-On](https://github.com/Cartucho/vision_blender)
, enable it and run:

Then postprocess NPZ -> PNG:

```bash
python -m sisifos.postprocess.export_gt \
  --npz-dir renders/<timestamp>/RF_<MODEL>/GTAnnotations/NPZ \
  --out-dir renders/<timestamp>/RF_<MODEL>/GTAnnotations \
  --r-rpo 70
```

## Third-party assets & outputs

If you generate datasets or renders using third-party assets, your outputs may be subject to the licenses/terms of those assets (including attribution requirements and redistribution limits). This repository does **not** grant you rights to redistribute third-party assets. Before publishing generated datasets, verify that the underlying assets permit redistribution of derivatives (and that you meet attribution requirements).

If you believe any content in this repository violates your rights or should not be referenced/distributed, please open an issue or contact the maintainers. We will investigate and, when appropriate, remove or replace the referenced content.

## Citation

If you use SISIFOS in academic work, please cite:

```bibtex
@inbook{sisifos,
  author    = {Iason Georgios Velentzas and Juan-Diego Florez Castillo and Noah Bruckner and Mehregan Dor and Panagiotis Tsiotras},
  title     = {SISIFOS: Specialized Illumination SImulator for Orbiting Spacecraft},
  booktitle = {AIAA SCITECH 2026 Forum},
  doi       = {10.2514/6.2026-2597},
}
```
