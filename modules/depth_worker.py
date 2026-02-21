"""Persistent subprocess worker for trimesh + embreex depth computation.

Runs in a separate process to avoid Embree symbol conflicts with Blender.
Communicates via length-prefixed binary messages on stdin/stdout.

Protocol:
  1. INIT message: receives K, res_x, res_y → pre-computes camera ray directions
  2. FRAME messages (loop): receives vertices, faces, extrinsic, max_dist →
     builds mesh + BVH, raytraces, sends depth map back
  3. Shutdown: stdin closes → worker exits
"""
import io
import struct
import sys
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector


def _read_msg(stream):
    """Read a length-prefixed message. Returns None on EOF."""
    header = stream.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack(">I", header)[0]
    data = stream.read(length)
    if len(data) < length:
        return None
    return np.load(io.BytesIO(data), allow_pickle=False)


def _write_msg(stream, **arrays):
    """Write a length-prefixed NPZ message."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    payload = buf.getvalue()
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def _make_T_from_extrinsic(extr: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :4] = extr
    return T


def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _camera_rays_in_camera_frame(W: int, H: int, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)
    dirs = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(float)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    return dirs / (norms + 1e-12)


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts.copy()
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=float)], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # --- INIT ---
    init_data = _read_msg(stdin)
    if init_data is None:
        return
    K = init_data["K"]
    res_x = int(init_data["res_x"])
    res_y = int(init_data["res_y"])
    dirs_C = _camera_rays_in_camera_frame(res_x, res_y, K)

    # --- FRAME loop ---
    while True:
        frame_data = _read_msg(stdin)
        if frame_data is None:
            break

        vertices = frame_data["vertices"]
        faces = frame_data["faces"]
        extrinsic_mat = frame_data["extrinsic_mat"]
        max_dist = float(frame_data["max_dist"])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if mesh.is_empty:
            _write_msg(stdout, depth=np.zeros((res_y, res_x), dtype=np.float32))
            continue

        T_CW = _make_T_from_extrinsic(extrinsic_mat)
        T_WC = _invert_T(T_CW)

        origins_W = np.repeat(T_WC[:3, 3].reshape(1, 3), res_x * res_y, axis=0)
        dirs_W = (T_WC[:3, :3] @ dirs_C.T).T

        intersector = RayMeshIntersector(mesh)
        locs_W, ray_idx, _tri_idx = intersector.intersects_location(
            origins_W, dirs_W, multiple_hits=False
        )

        depth_z = np.full((res_x * res_y,), -1.0, dtype=np.float32)
        if locs_W.shape[0] > 0:
            locs_C = _transform_points(T_CW, locs_W)
            z = locs_C[:, 2]
            valid = (z > 0) & (z <= max_dist) & np.isfinite(z)
            depth_z[ray_idx[valid]] = z[valid].astype(np.float32)

        _write_msg(stdout, depth=depth_z.reshape(res_y, res_x))


if __name__ == "__main__":
    main()
