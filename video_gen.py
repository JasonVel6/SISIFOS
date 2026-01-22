#!/usr/bin/env python3
"""
Auto-discover image folders under a base directory and generate MP4 videos
for each folder × substring combination (skipping missing matches).

Behavior:
- Recursively scans base for subfolders that contain at least one image file
- For each such folder, for each substring in SUBSTRINGS:
    - selects images whose filename contains substring
    - if none found: skip (ignore)
    - else: generate an MP4 in <base>/videos/<relative_folder>__<substring>.mp4
- Natural-sort filenames (img_2 before img_10)
- Uses ffmpeg concat demuxer (robust to gaps and arbitrary filenames)

Requirements:
- ffmpeg available in PATH (or set env var FFMPEG_BIN)
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

IMAGE_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ---- User-configurable defaults ----
SUBSTRINGS = [
    "RGB",
    "Depth",
    "Normal",
    "Seg",
    "OpticalFlow",
]
DEFAULT_FPS = 30.0
DEFAULT_GLOB = "*"  # e.g. "*.png" if you want to restrict
OUTPUT_DIRNAME = "videos"  # created under base


# ----------------------------
# Sorting helpers
# ----------------------------
_natural_re = re.compile(r"(\d+)")

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in _natural_re.split(s)]


# ----------------------------
# Discovery
# ----------------------------
def folder_has_images(folder: Path, glob_pattern: str) -> bool:
    for p in folder.glob(glob_pattern):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return True
    return False

def discover_image_folders(base: Path, glob_pattern: str) -> List[Path]:
    """
    Returns a list of folders (including nested) that contain at least one image.
    """
    folders: Set[Path] = set()

    # Scan for image files and collect their parent directories.
    for p in base.rglob(glob_pattern):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            folders.add(p.parent)

    # Stable ordering by relative path
    return sorted(folders, key=lambda f: natural_key(str(f.relative_to(base))))


# ----------------------------
# Selection
# ----------------------------
def select_images(folder: Path, substring: str, glob_pattern: str) -> List[Path]:
    imgs = [
        p for p in folder.glob(glob_pattern)
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and substring in p.name
    ]
    imgs.sort(key=lambda p: natural_key(p.name))
    return imgs


# ----------------------------
# FFmpeg concat generation
# ----------------------------
def write_concat_list(paths: Sequence[Path], list_path: Path) -> None:
    def esc(p: Path) -> str:
        # ffmpeg concat wants file '...'; escape single quotes safely
        s = str(p.resolve()).replace("\\", "/")
        return s.replace("'", r"'\''")

    with list_path.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{esc(p)}'\n")

def ensure_ffmpeg() -> None:
    try:
        subprocess.run([FFMPEG_BIN, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError(f"ffmpeg not available ({FFMPEG_BIN}). Install it or set env var FFMPEG_BIN. {e}")

def run_ffmpeg_concat(list_file: Path, output_mp4: Path, fps: float, overwrite: bool) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BIN,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-vf", f"fps={fps}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_mp4),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode}): {shlex.join(cmd)}")


# ----------------------------
# Output naming
# ----------------------------
def sanitize_for_filename(s: str) -> str:
    # keep it filesystem-friendly
    s = s.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

def output_path_for(base: Path, folder: Path, substring: str, output_dirname: str, fps: float) -> Path:
    rel = folder.relative_to(base)
    rel_str = "_".join(rel.parts)  # preserve tree structure in a flat filename

    fps_tag = f"{fps:g}"  # 30.0 -> "30", 29.97 -> "29.97"
    has_substring_component = substring in rel.parts

    if has_substring_component:
        stem = f"{sanitize_for_filename(rel_str)}_fps_{sanitize_for_filename(fps_tag)}"
    else:
        stem = (
            f"{sanitize_for_filename(rel_str)}__{sanitize_for_filename(substring)}"
            f"_fps_{sanitize_for_filename(fps_tag)}"
        )    
    name = f"{stem}.mp4"

    return base / output_dirname / name

# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discover image subfolders under base and generate MP4 videos per substring."
    )
    p.add_argument("--base", type=Path, required=True, help="Base folder to scan recursively.")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"Output FPS (default {DEFAULT_FPS}).")
    p.add_argument("--glob", dest="glob_pattern", default=DEFAULT_GLOB,
                   help=f"Glob pattern for images (default '{DEFAULT_GLOB}'; e.g. '*.png').")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing mp4 outputs.")
    p.add_argument("--output-dir", default=OUTPUT_DIRNAME,
                   help=f"Directory under base to place mp4s (default '{OUTPUT_DIRNAME}').")
    p.add_argument("--substrings", nargs="+", default=SUBSTRINGS,
                   help=f"List of substrings to try (default: {SUBSTRINGS}).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.expanduser().resolve()

    if not base.exists():
        raise SystemExit(f"Base folder does not exist: {base}")

    ensure_ffmpeg()

    folders = discover_image_folders(base, args.glob_pattern)
    if not folders:
        print(f"[INFO] No image folders found under: {base}")
        return

    print(f"[INFO] Found {len(folders)} image folder(s) under: {base}")
    print(f"[INFO] Substrings: {args.substrings}")
    print(f"[INFO] Output dir: {base / args.output_dir}")

    errors = 0
    generated = 0
    skipped_no_match = 0
    skipped_exists = 0

    for folder in folders:
        for substring in args.substrings:
            imgs = select_images(folder, substring, args.glob_pattern)

            # Requested: ignore if no images match substring in that folder
            if not imgs:
                skipped_no_match += 1
                continue

            out_mp4 = output_path_for(base, folder, substring, args.output_dir, args.fps)

            if out_mp4.exists() and not args.overwrite:
                skipped_exists += 1
                continue

            try:
                with tempfile.TemporaryDirectory() as td:
                    list_file = Path(td) / "concat.txt"
                    write_concat_list(imgs, list_file)
                    rel_folder = folder.relative_to(base)
                    print(f"[RUN ] {rel_folder} | '{substring}' -> {out_mp4.relative_to(base)} "
                          f"(frames={len(imgs)}, fps={args.fps})")
                    run_ffmpeg_concat(list_file, out_mp4, args.fps, overwrite=args.overwrite)
                generated += 1
            except Exception as e:
                errors += 1
                print(f"[ERR ] {folder} | '{substring}': {e}", file=sys.stderr)

    print("\n[SUMMARY]")
    print(f"  Generated videos: {generated}")
    print(f"  Skipped (no match): {skipped_no_match}")
    print(f"  Skipped (exists): {skipped_exists}")
    print(f"  Errors: {errors}")

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
