import numpy as np
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb

def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """HSV visualization for optical flow (u,v in pixels)."""
    u, v = flow[..., 0], flow[..., 1]
    ang = np.arctan2(-v, -u)
    mag = np.sqrt(u * u + v * v)
    hsv = np.zeros((*u.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2.0 * np.pi)  # hue
    p99 = np.percentile(mag, 99) if np.any(np.isfinite(mag)) else 1.0
    hsv[..., 1] = np.clip(mag / (p99 + 1e-9), 0, 1)  # sat
    hsv[..., 2] = 1.0  # val
    return hsv_to_rgb(hsv).astype(np.float32)

def _norm_to_rgb(normals: np.ndarray) -> np.ndarray:
    """Normals assumed in [-1,1]; map to [0,1]."""
    return np.clip(0.5 * (normals[..., :3] + 1.0), 0, 1).astype(np.float32)


def _id_to_color(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.int32)
    r = ((ids * 37) % 255) / 255.0
    g = ((ids * 57) % 255) / 255.0
    b = ((ids * 97) % 255) / 255.0
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _depth_vis_and_mask_from_rrpo(
    depth: np.ndarray,
    R_RPO: float,  
    cmap_name: str = "magma",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rgb: HxWx3 float32 in [0,1]
      mask: HxW bool (True = near object)
    """
    valid = np.isfinite(depth) & (depth > 0)
    mask = valid & (depth <  float(R_RPO)+5)

    dmin = 0.1
    dmax = float(R_RPO)+5
    denom = (dmax - dmin) if (dmax > dmin) else 1.0

    x = (depth - dmin) / denom
    x = np.clip(x, 0.0, 1.0)

    cmap = mpl.colormaps.get_cmap(cmap_name)   # or mpl.colormaps[cmap_name]
    rgb = cmap(x)[..., :3].astype(np.float32)    
    rgb[~mask] = 0.0  # background black
    return rgb, mask