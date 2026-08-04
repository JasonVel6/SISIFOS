"""Filesystem/dataset helpers that do not depend on Blender.

Split out of `io_utils` so the trajectory stack (and any Blender-free entry
point such as `generate_trajectories.py`) can use them without importing `bpy`.
`io_utils` re-exports these names, so existing imports keep working.
"""

import os
from datetime import datetime
from pathlib import Path

from .log_utils import get_logger


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_folder():
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def create_image_list(renders_base_dir: str, timestamps: list, image_paths):
    """
    Create imgList.txt with timestamp-image pairs.
    """
    imglist_path = os.path.join(renders_base_dir, "imgList.txt")
    with open(imglist_path, "w") as f:
        for i in range(len(timestamps)):
            ts = timestamps[i]
            f.write(f"{ts:.6f} {image_paths[i]}\n")
    get_logger().info("  Created: %s", imglist_path)
    return imglist_path
