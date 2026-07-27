#!/usr/bin/env python3
"""Generate the 35 synthetic-exp SISIFOS configs (7 cells x 5 seeds).

Cells, seeds, and the template are defined in this file. Run with:

    python3 configs/synthetic-exp/_generate.py             # write all 35 JSONs
    python3 configs/synthetic-exp/_generate.py --dry-run   # print what would be written

The plan this serves lives in
~/.claude/projects/-home-jdflo-satslam-SatSLAM/memory/synthetic-experiment-plan.md.

After generating configs, generate datasets with Blender via:

    bash configs/synthetic-exp/_generate-datasets.sh

(That script is intentionally separate so we can review JSONs before burning
~12 min of Blender time on 35 dataset generations.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Resolve relative to this file so it works from any CWD.
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "probe_aniso_fast.json"  # sisifos2/SISIFOS/configs/probe_aniso_fast.json

# Master seeds. Mix of timestamp + mathematical constants so they're
# obviously reproducible but independent of each other.
SEEDS = [
    1462526110,   # s1 - the original probe seed; lets s1 reproduce earlier runs
    2024051501,   # s2 - YYYYMMDDNN timestamp
    314159265,    # s3 - pi
    271828182,    # s4 - e
    1618033988,   # s5 - phi (golden ratio)
]

# Cell definitions. Each entry maps to overrides applied to the trajectory
# block of the template. Field names must match SISIFOS TrajectoryConfig.
#
# Inertia conventions (see modules/spacecraft_defaults.yaml):
#   custom:   {"inertia_type": "custom", "Jx": ..., "Jy": ..., "Jz": ...}
#   cylinder: {"inertia_type": "cylinder", "m": ..., "r": ..., "h": ...}
#   box:      {"inertia_type": "box", "m": ..., "l": ..., "w": ..., "h": ...}
#
# Hubble: m=12200 kg, r=4.2 m, h=13 m (cylinder, axisymmetric Jxx=Jyy)
# INTEGRAL: m=4000 kg, l=5.0, w=3.2, h=2.8 m (box, full 3-axis)
# Non-trajectory render params for the image-render cells (c2/c3). Applied via
# the `_overrides` path-applier in make_config(). Synthetic cells (c1, c4-c7)
# omit this and stay render-free. See docs/real-frontend-render-manifest.md.
_RENDER_OVERRIDES = {
    "setup.render_frames": True,
    "setup.earth_mode": "off",
    "camera.resolution": [1024, 1024],
    "camera.pixel_size_um": 6.469,            # preserves the 25 mm lens's ~15 deg FOV
    "render.crop_to_border_px": None,         # full frame (256 crop clipped the targets)
    "trajectory.EARTH_BACKGROUND_ENABLE": True,   # natural orbital-radial sun (nro recipe)
    "trajectory.SUN_ALIGN_ENABLE": False,         # not camera-pinned
    "camera.focal_length": 40.0,              # larger target (~78% broadside) for ORB track persistence
    "save_segmentation": True,                # clean foreground extraction (bg exactly 0);
                                              # without it the masking copies raw -> bg noise floor
}

CELLS: dict[str, dict] = {
    "c1-anchor": {
        "_role": "headline convergence",
        "inertia_config": {"inertia_type": "custom", "Jx": 0.273, "Jy": 0.26, "Jz": 0.16},
        "omega_min_deg": 3.0,
        "omega_max_deg": 5.0,
        "r_AG_G": [0.1, 0.05, 0.15],
        "path_mode": "tumbling",
    },
    "c2-hubble": {
        "_role": "real spacecraft 1 (cylinder, near-axisymmetric)",
        "inertia_config": {"inertia_type": "cylinder", "m": 12200.0, "r": 4.2, "h": 13.0},
        "omega_min_deg": 3.0,
        "omega_max_deg": 5.0,
        "r_AG_G": [0.1, 0.05, 0.15],
        "path_mode": "tumbling",
        "_overrides": {**_RENDER_OVERRIDES},  # RF_Hubble @ 100 m (template default)
    },
    "c3-integral": {
        "_role": "real spacecraft 2 (box, full 3-axis)",
        "inertia_config": {"inertia_type": "box", "m": 4000.0, "l": 5.0, "w": 3.2, "h": 2.8},
        "omega_min_deg": 3.0,
        "omega_max_deg": 5.0,
        "r_AG_G": [0.1, 0.05, 0.15],
        "path_mode": "tumbling",
        "_overrides": {
            **_RENDER_OVERRIDES,
            "selected_model": "RF_Integral",     # INTEGRAL box mesh
            "trajectory.R0_const": 38.5,         # matched footprint (13:5 vs Hubble@100)
        },
    },
    "c4-nearsphere": {
        "_role": "inertia-degeneracy failure mode",
        "inertia_config": {"inertia_type": "custom", "Jx": 0.20, "Jy": 0.20, "Jz": 0.20},
        "omega_min_deg": 3.0,
        "omega_max_deg": 5.0,
        "r_AG_G": [0.1, 0.05, 0.15],
        "path_mode": "tumbling",
    },
    "c5-slowspin": {
        "_role": "observability lower bound",
        "inertia_config": {"inertia_type": "custom", "Jx": 0.273, "Jy": 0.26, "Jz": 0.16},
        "omega_min_deg": 0.5,
        "omega_max_deg": 1.5,
        "r_AG_G": [0.1, 0.05, 0.15],
        "path_mode": "tumbling",
    },
    "c6-largecom": {
        "_role": "COM observability",
        "inertia_config": {"inertia_type": "custom", "Jx": 0.273, "Jy": 0.26, "Jz": 0.16},
        "omega_min_deg": 3.0,
        "omega_max_deg": 5.0,
        "r_AG_G": [0.35, 0.05, 0.05],  # X-dominant, magnitude 0.357 m
        "path_mode": "tumbling",
    },
    "c7-stronganiso": {
        "_role": "observability upper bound (ratio 2.5)",
        "inertia_config": {"inertia_type": "custom", "Jx": 0.30, "Jy": 0.22, "Jz": 0.12},
        "omega_min_deg": 3.0,
        "omega_max_deg": 5.0,
        "r_AG_G": [0.1, 0.05, 0.15],
        "path_mode": "tumbling",
    },
}


def make_config(cell_name: str, cell: dict, seed: int, seed_idx: int) -> dict:
    """Load template and apply cell + seed overrides."""
    with open(TEMPLATE) as f:
        cfg = json.load(f)

    role = cell.get("_role", "")
    cfg["_comment"] = (
        f"Synthetic-exp suite | cell {cell_name} ({role}) | seed s{seed_idx}={seed}. "
        f"Auto-generated by configs/synthetic-exp/_generate.py — DO NOT hand-edit."
    )

    traj = cfg["base_config"]["trajectory"]
    traj["seed"] = seed
    # Derive a deterministic-but-distinct illumination seed from the master
    # seed so render-time draws don't all collapse to the same value if we
    # later switch to render_frames=true. Mask to 31-bit positive int.
    traj["illumination_seed"] = (seed * 7919) & 0x7FFFFFFF

    # MIN_F2F_PX_MED is a real-frontend descriptor-matching heuristic; the
    # synthetic frontend (perfect data associations) doesn't need it. Drop
    # the floor to a near-zero value so the generator's phase search always
    # succeeds across all seeds and ALL cells — in particular the slow-spin
    # cell C5 (omega 0.5-1.5 deg/s) would deterministically fail the default
    # floor 3.0 because apparent feature motion is sub-pixel at that spin.
    # The CRO amplitude is still set deterministically from R0_const and
    # tumbling_span_frac, NOT shrunk by lowering this floor.
    traj["MIN_F2F_PX_MED"] = 0.1

    for k, v in cell.items():
        if k.startswith("_"):
            continue
        traj[k] = v

    # Render-cell overrides applied to base_config by (optionally dotted) path.
    # Image-render cells (c2/c3) use this for non-trajectory params (mesh,
    # resolution, crop, sun mode, render-on). Synthetic cells omit it.
    for path, val in cell.get("_overrides", {}).items():
        node = cfg["base_config"]
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(HERE),
                    help="Output directory for generated JSONs (default: alongside this script).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without writing.")
    args = ap.parse_args()

    if not TEMPLATE.is_file():
        raise SystemExit(f"Template not found: {TEMPLATE}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[tuple[str, str, int]] = []  # (filename, cell_name, seed)
    for cell_name, cell in CELLS.items():
        for i, seed in enumerate(SEEDS, start=1):
            cfg = make_config(cell_name, cell, seed, i)
            name = f"probe-{cell_name}-s{i}.json"
            path = out_dir / name
            if not args.dry_run:
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=2)
                    f.write("\n")
            written.append((name, cell_name, seed))

    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} {len(written)} configs to {out_dir}")
    by_cell: dict[str, list[str]] = {}
    for name, cell, seed in written:
        by_cell.setdefault(cell, []).append(f"{name} (seed={seed})")
    for cell, items in by_cell.items():
        role = CELLS[cell].get("_role", "")
        print(f"  [{cell}] {role}")
        for it in items:
            print(f"    {it}")


if __name__ == "__main__":
    main()
