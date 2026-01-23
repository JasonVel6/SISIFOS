# SISIFOS

**S**pecialized **I**llumination **SI**mulator **F**or **O**rbiting **S**pacecraft (**SISIFOS**) is a Blender-based spacecraft image simulator for generating photorealistic RGB images and annotations for spacecraft Rendezvous and Proximity Operations (RPO) scenarios.

SISIFOS can be used with spacecraft models from sources such as **ESA SciFleet** and **NASA 3D resources**, and with textures (e.g., Earth imagery) from sources such as **NASA Earth Observatory**. **Each asset retains its own license/terms**—you must comply with the specific terms attached to every model/texture you download and use.

This project is an independent research tool and is **not affiliated with, endorsed by, or sponsored by** NASA, ESA, or any other organization.

## Quick start

### 1) Setup Environment 

#### Windows

Run the setup script from the `SISIFOS` directory:

```powershell
.\env\activate.ps1
```

#### MacOS & Linux
```bash
source env/activate.sh
```

This script will:
- Automatically download and extract **Blender 4.5.6** (if not already present)
- Bootstrap and configure Blender's Python environment
- Install all required dependencies using `uv`
- Set up environment variables (`blender`, `python`)
- Provide a `deactivate` command to clean up the environment

**Notes:**
- This script requires PowerShell 5.0+
- If you already have Blender 4.5 installed elsewhere, you can skip the auto-download by placing it in `env/Blender_4.5/`
- If an NVIDIA GPU is available, enable **OptiX** in Blender preferences for faster Cycles rendering

To deactivate the environment later:

```powershell
deactivate
```

### 2) Download your assets

As of this moment, this repo does **not** ship ThirdParty Software, meaning ESA/NASA models. The user needs to bring their own spacecraft model as a blend file:

```
assets/
  spacecraft_models.blend
```

### 3) Configure your scene

Edit the paths in the config file accordingly:

- `scene_blend_path`
- `hdri_path`
- each object's `blend_path`

in `configs/examples/config_example_basic.json`.

For the moment, the blender file and hdri assets are under this [folder](https://gtvault-my.sharepoint.com/:f:/g/personal/ivelentzas3_gatech_edu/IgDCVzfY6FrUR6kv23ktq4BrAWC0mL0DFF9N7xTztAOlTUo?e=Op9Eki)

### 4) Generate a reference trajectory file

SISIFOS expects a reference trajectory text file containing the target pose, camera pose, and sun angles per frame.

File format (one row per frame):

```
p_G_I(x) p_G_I(y) p_G_I(z)   q_I_G(w) q_I_G(x) q_I_G(y) q_I_G(z)   p_C_I(x) p_C_I(y) p_C_I(z)   q_I_C(w) q_I_C(x) q_I_C(y) q_I_C(z)   sun_az  sun_el
```

Where:

- p_G_I = position of target (G) in inertial frame (I) at radius R_LEO

- q_I_G = orientation of target frame (G) relative to inertial (I), quaternion wxyz

- p_C_I = position of camera (C) in inertial frame (I) at radius (R_LEO + R_RPO)

- q_I_C = orientation of camera frame (C) relative to inertial (I), quaternion wxyz

- sun_az, sun_el = sun azimuth/elevation (recommended: degrees, document your choice and keep it consistent)

- Earth / clouds / atmosphere stay at inertial origin (0,0,0) with fixed orientation.

### 5) Run the simulator

After activating the environment, run the simulator from the root of the repo:

```powershell
blender -b -P main.py -- configs/examples/config_example_basic.json
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
