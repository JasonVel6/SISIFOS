"""Persistent subprocess worker for trimesh + embreex depth computation.

Runs in a separate process to avoid Embree symbol conflicts with Blender.
Communicates via length-prefixed binary messages on stdin/stdout.

Protocol:
  1. INIT message: receives K, res_x, res_y, vertices, faces
     → pre-computes camera ray directions, builds mesh + Embree BVH (cached)
  2. FRAME messages (loop): receives T_CW, T_WO, max_dist
     → transforms rays into object-local space, raytraces cached BVH, sends depth
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


def _make_T44(m34):
    """Pad a 3x4 matrix to 4x4."""
    T = np.eye(4, dtype=float)
    T[:3, :4] = m34
    return T


def _invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _camera_rays_in_camera_frame(W, H, K):
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


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # --- INIT: camera intrinsics + mesh ---
    init_data = _read_msg(stdin)
    if init_data is None:
        return
    K = init_data["K"]
    res_x = int(init_data["res_x"])
    res_y = int(init_data["res_y"])
    vertices = init_data["vertices"]
    faces = init_data["faces"]

    n_rays = res_x * res_y
    dirs_C = _camera_rays_in_camera_frame(res_x, res_y, K)

    # Build mesh + Embree BVH once
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    empty = mesh.is_empty
    intersector = None if empty else RayMeshIntersector(mesh)

    # --- FRAME loop ---
    while True:
        frame_data = _read_msg(stdin)
        if frame_data is None:
            break

        if empty:
            _write_msg(stdout, depth=np.zeros((res_y, res_x), dtype=np.float32))
            continue

        T_CW = _make_T44(frame_data["T_CW"])
        T_WO = _make_T44(frame_data["T_WO"])
        max_dist = float(frame_data["max_dist"])

        # Camera → object-local space (for rays)
        T_WC = _invert_T(T_CW)
        T_OW = _invert_T(T_WO)
        T_OC = T_OW @ T_WC

        # Rays in object-local space
        origins_O = np.repeat(T_OC[:3, 3].reshape(1, 3), n_rays, axis=0)
        dirs_O = (T_OC[:3, :3] @ dirs_C.T).T

        locs_O, ray_idx, _ = intersector.intersects_location(
            origins_O, dirs_O, multiple_hits=False
        )

        depth_z = np.full(n_rays, -1.0, dtype=np.float32)
        if locs_O.shape[0] > 0:
            # Object-local hits → camera frame for z-depth
            T_CO = T_CW @ T_WO
            R_CO = T_CO[:3, :3]
            t_CO = T_CO[:3, 3]
            locs_C = (R_CO @ locs_O.T).T + t_CO
            z = locs_C[:, 2]
            valid = (z > 0) & (z <= max_dist) & np.isfinite(z)
            depth_z[ray_idx[valid]] = z[valid].astype(np.float32)

        _write_msg(stdout, depth=depth_z.reshape(res_y, res_x))


if __name__ == "__main__":
    main()
