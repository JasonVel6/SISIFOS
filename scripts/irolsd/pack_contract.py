#!/usr/bin/env python3
"""Pack SISIFOS fixed-pose/swept-Sun output into the IRoLSD label-gen contract.

SISIFOS renders each CSV row ``n`` to ``<agent>/images_raw/.../frame_{n:0Nd}.png``
plus a NPZ ground-truth file carrying the raw float ``depth_map``. The IRoLSD
label-gen (``irolsd/label_gen/irolsd_label_gen.py``) instead expects a FLAT
folder:

    <out>/bepi_train/
        img_000000.png    # pose 0, canonical (i=0)
        img_000001.png    # pose 0, variant 1
        ...
        depth_0.npy       # pose 0 depth (spacecraft only, no Earth), float32 HxW
        depth_1.npy       # pose 1 depth ...

with ``img_{g*NVARIANTS+i}.png`` and one ``depth_{g}.npy`` per pose group
(indexed by the pose group ``g``, taken from the CANONICAL i=0 frame of the
group). This script performs that remap.

DEPTH: the label-gen does ``np.load(depth).astype(float32)`` then normalizes with
``(d - d.min())/d.ptp()`` before running LSD. SISIFOS marks background
(points at infinity) with ``-1``; passing that through would blow up the
normalization and inject a spurious silhouette-of-the-frame edge. So we replace
background with the FAR spacecraft depth (max valid depth in-frame) -> a flat
background yields no LSD edges except the true spacecraft structure/silhouette,
which is exactly the paper's depth-line intent (Eq. 24, spacecraft only). With
``earth_mode: off`` Earth is not rendered, so valid depth is already
spacecraft-only.

Run OUTSIDE Blender (numpy + optional PIL for the near-black discard check):

    python3 scripts/irolsd/pack_contract.py \
        --agent renders/_phased/bepi_train_src/Config_1_RF_Bepi-mcs/Agent_0 \
        --out   <datasets>/irolsd/bepi_train \
        [--nvariants 20] [--copy]   # default: symlink images, write depth .npy
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np


def _load_meta(agent: str) -> dict:
    p = os.path.join(agent, "sweep_meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _find_images_dir(agent: str) -> str:
    """Locate the images_raw leaf dir SISIFOS actually wrote frames into.

    Works whether ``agent`` is the source Agent_0 (frames rendered in place) or
    a parent dir: in 'filepath' trajectory mode SISIFOS renders into
    ``<agent>/_render_out/Config_1_<model>/images_raw/...`` so we search the
    whole tree for frame_*.png (any images_raw leaf).
    """
    candidates = glob.glob(os.path.join(agent, "**", "frame_*.png"), recursive=True)
    candidates = [c for c in candidates if os.sep + "images_raw" + os.sep in c or "/images_raw/" in c]
    if not candidates:
        # fall back to any frame_*.png anywhere under agent
        candidates = glob.glob(os.path.join(agent, "**", "frame_*.png"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"no frame_*.png under {agent} (render not run, or wrong agent dir)"
        )
    # Pick the SHALLOWEST images_raw dir. Resumable render banks re-symlink the
    # source Agent into _render_out each pass, creating nested phantom paths
    # (_render_out/Config.../_render_out/Config.../...) that alphabetically sort
    # first but are broken symlinks. The real frames live at the shallowest path.
    dirs = {os.path.dirname(c) for c in candidates}
    return min(dirs, key=lambda d: (d.count(os.sep), len(d)))


def _find_npz_dir(agent: str) -> str | None:
    """Locate GTAnnotations/NPZ under ``agent`` (shallowest; may be nested in _render_out)."""
    hits = glob.glob(os.path.join(agent, "**", "GTAnnotations", "NPZ"), recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    # shallowest = the real one (see _find_images_dir note on nested phantom paths)
    return min(hits, key=lambda d: (d.count(os.sep), len(d))) if hits else None


def _frame_index(path: str) -> int:
    m = re.search(r"frame_(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot parse frame index from {path}")
    return int(m.group(1))


def _npz_depth_for(npz_dir: str | None, n: int) -> np.ndarray | None:
    """Return the raw float depth map for frame n, or None if absent.

    SISIFOS moves the per-frame NPZ to GTAnnotations/NPZ/. The NPZ basename is
    the zero-padded frame index (handle_gt_from_npz uses ``{i:04d}.npz`` from
    the renderer, so >=4 digits). We search both the 4-digit and N-digit forms.
    """
    if npz_dir is None:
        return None
    for cand in (f"{n:04d}.npz", f"{n:05d}.npz", f"{n:06d}.npz", f"{n}.npz"):
        p = os.path.join(npz_dir, cand)
        if os.path.exists(p):
            data = np.load(p, allow_pickle=True)
            if "depth_map" in data:
                return data["depth_map"].astype(np.float32)
            return None
    # last resort: glob for any npz whose numeric stem == n
    for p in glob.glob(os.path.join(npz_dir, "*.npz")):
        stem = os.path.splitext(os.path.basename(p))[0]
        digits = re.sub(r"\D", "", stem)
        if digits and int(digits) == n:
            data = np.load(p, allow_pickle=True)
            if "depth_map" in data:
                return data["depth_map"].astype(np.float32)
    return None


def _clean_depth(depth: np.ndarray) -> np.ndarray:
    """Replace background (-1 / non-finite) with the max valid spacecraft depth."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        # no spacecraft in frame -> flat, gives no depth lines
        return np.zeros_like(d)
    far = float(d[valid].max())
    out = d.copy()
    out[~valid] = far
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", required=True, help="SISIFOS Agent_0 dir (holds images_raw/, GTAnnotations/).")
    ap.add_argument("--out", required=True, help="Flat contract output dir (e.g. <datasets>/irolsd/bepi_train).")
    ap.add_argument("--nvariants", type=int, default=None,
                    help="Illumination variants per pose. Default: read from sweep_meta.json.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy image PNGs instead of symlinking (portable but uses disk).")
    ap.add_argument("--no-depth", action="store_true", help="Skip depth .npy emission.")
    ap.add_argument("--pose-start", type=int, default=None,
                    help="Global pose offset for a shard (default: read from sweep_meta.json, else 0). "
                         "Local frame -> global via pose_start*nvariants so shards pack into one dataset.")
    args = ap.parse_args()

    meta = _load_meta(args.agent)
    nvariants = args.nvariants or meta.get("nvariants")
    if not nvariants:
        ap.error("--nvariants not given and sweep_meta.json missing/lacks nvariants")

    img_dir = _find_images_dir(args.agent)
    npz_dir = _find_npz_dir(args.agent)
    frame_pngs = sorted(glob.glob(os.path.join(img_dir, "frame_*.png")), key=_frame_index)
    if not frame_pngs:
        ap.error(f"no frames found in {img_dir}")

    # Sharding: SISIFOS names frames by LOCAL CSV row (0-based). Map local ->
    # global pose index by the shard's pose_start so multiple shards pack into
    # ONE dataset without index collision. pose_start default 0 (standalone).
    pose_start = int(args.pose_start if args.pose_start is not None else meta.get("pose_start", 0))
    frame_offset = pose_start * nvariants

    os.makedirs(args.out, exist_ok=True)

    n_img = 0
    n_depth = 0
    missing_depth_groups = []
    depth_done = set()
    for png in frame_pngs:
        n_local = _frame_index(png)
        n = n_local + frame_offset          # GLOBAL frame index
        g, i = divmod(n, nvariants)         # GLOBAL pose group

        # --- image: img_{n}.png (flat, 6-digit) ---
        dst_img = os.path.join(args.out, f"img_{n:06d}.png")
        if os.path.lexists(dst_img):
            os.remove(dst_img)
        if args.copy:
            shutil.copy2(png, dst_img)
        else:
            os.symlink(os.path.abspath(png), dst_img)
        n_img += 1

        # --- depth: one per pose group, from the canonical i==0 frame ---
        # NPZ on disk is named by the LOCAL frame index, so look up n_local.
        if not args.no_depth and i == 0 and g not in depth_done:
            depth = _npz_depth_for(npz_dir, n_local)
            if depth is None:
                missing_depth_groups.append(g)
            else:
                d = _clean_depth(depth)
                # UNPADDED g to match label-gen batch_label_gen.py: f"depth_{g}.npy"
                # (it does divmod(idx, n_variants) and looks up depth_{g}.npy verbatim).
                np.save(os.path.join(args.out, f"depth_{g}.npy"), d)
                n_depth += 1
                depth_done.add(g)

    nposes = (max(_frame_index(p) for p in frame_pngs) // nvariants) + 1
    summary = {
        "agent": os.path.abspath(args.agent),
        "out": os.path.abspath(args.out),
        "nvariants": nvariants,
        "nposes_seen": nposes,
        "images_packed": n_img,
        "depths_packed": n_depth,
        "missing_depth_groups": missing_depth_groups,
        "image_mode": "copy" if args.copy else "symlink",
    }
    with open(os.path.join(args.out, "pack_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[pack] images: {n_img}  depths: {n_depth}  (nvariants={nvariants}, nposes~{nposes})")
    if missing_depth_groups:
        print(f"[pack] WARNING: {len(missing_depth_groups)} groups missing depth NPZ "
              f"(first few: {missing_depth_groups[:5]}) -- was save_depth on?")
    print(f"[pack] -> {args.out}  (summary: pack_summary.json)")


if __name__ == "__main__":
    main()
