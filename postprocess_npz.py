#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from modules.io_utils import ensure_dir, handle_gt_from_npz


def _expand_inputs(inputs: list[str], recursive: bool) -> list[Path]:
    npz_files: list[Path] = []
    seen: set[Path] = set()
    pattern = "*.npz"

    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            matches = sorted(path.rglob(pattern) if recursive else path.glob(pattern))
            for match in matches:
                resolved = match.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    npz_files.append(match)
            continue

        if path.is_file() and path.suffix.lower() == ".npz":
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                npz_files.append(path)
            continue

        raise FileNotFoundError(f"NPZ input not found or unsupported: {raw}")

    if not npz_files:
        raise FileNotFoundError("No NPZ files found in the provided inputs.")

    return npz_files


def _default_gt_root(npz_path: Path) -> Path:
    if npz_path.parent.name == "NPZ" and npz_path.parent.parent.name == "GTAnnotations":
        return npz_path.parent.parent
    return npz_path.parent / "GTAnnotations"


def _default_raw_images_dir(npz_path: Path) -> Path:
    if npz_path.parent.name == "NPZ" and npz_path.parent.parent.name == "GTAnnotations":
        return npz_path.parent.parent.parent / "images_raw"
    return npz_path.parent


def _default_masked_images_dir(raw_images_dir: Path) -> Path:
    if raw_images_dir.name == "images_raw":
        return raw_images_dir.parent / "images"
    return raw_images_dir.parent / f"{raw_images_dir.name}_masked"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SISIFOS NPZ post-processing on one or more NPZ files.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more NPZ files or directories containing NPZ files.",
    )
    parser.add_argument(
        "--target-dist",
        "--r-rpo",
        dest="target_dist",
        type=float,
        required=True,
        help="Target camera distance used by the depth gate heuristic.",
    )
    parser.add_argument(
        "--raw-images-dir",
        type=Path,
        help="Directory containing the rendered RGB images. Defaults from each NPZ location.",
    )
    parser.add_argument(
        "--masked-images-dir",
        type=Path,
        help="Directory where masked RGB images will be written. Defaults from raw image dir.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        help="Root GTAnnotations directory. Defaults from each NPZ location.",
    )
    parser.add_argument(
        "--image-ext",
        default=".png",
        help="Rendered image extension for matching NPZ stems. Default: .png",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for NPZ files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_ext = args.image_ext if args.image_ext.startswith(".") else f".{args.image_ext}"
    npz_files = _expand_inputs(args.inputs, recursive=args.recursive)

    for npz_path in npz_files:
        raw_images_dir = (args.raw_images_dir or _default_raw_images_dir(npz_path)).expanduser()
        masked_images_dir = (args.masked_images_dir or _default_masked_images_dir(raw_images_dir)).expanduser()
        gt_root = (args.gt_root or _default_gt_root(npz_path)).expanduser()

        gt_npz_dir = ensure_dir(gt_root / "NPZ")
        gt_depth_dir = ensure_dir(gt_root / "Depth")
        gt_norm_dir = ensure_dir(gt_root / "Normal")
        gt_flow_dir = ensure_dir(gt_root / "Flow")
        gt_seg_dir = ensure_dir(gt_root / "Seg")

        raw_image_filename = f"{npz_path.stem}{image_ext}"
        raw_image_path = raw_images_dir / raw_image_filename
        if not raw_image_path.exists():
            raise FileNotFoundError(
                f"Expected rendered image for {npz_path.name} at {raw_image_path}"
            )

        print(f"Processing {npz_path}")
        handle_gt_from_npz(
            npz_src=npz_path,
            gt_npz_dir=gt_npz_dir,
            gt_depth_dir=gt_depth_dir,
            gt_norm_dir=gt_norm_dir,
            gt_flow_dir=gt_flow_dir,
            gt_seg_dir=gt_seg_dir,
            target_dist=args.target_dist,
            raw_image_filename=raw_image_filename,
            raw_images_dir=str(raw_images_dir),
            masked_images_dir=str(masked_images_dir),
        )

    print(f"Processed {len(npz_files)} NPZ file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
