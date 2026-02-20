import math
import random
from typing import List, Tuple
from mathutils import Vector, Matrix, Quaternion
import numpy as np
from pathlib import Path
import os
import sys
if sys.platform != "win32":
    _old_dl_flags = sys.getdlopenflags()
    sys.setdlopenflags(os.RTLD_DEEPBIND | os.RTLD_NOW)
import trimesh
if sys.platform != "win32":
    sys.setdlopenflags(_old_dl_flags)
    del _old_dl_flags
import bpy

def fibonacci_sphere(n: int, radius: float = 1.0) -> List[Vector]:
    """Fibonacci sampled points along a sphere of specific radius"""
    if n <= 0:
        return []
    pts = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i) / (n - 1) if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append(Vector((x, y, z)) * radius)
    return pts

def _rand_quat_uniform(rng: random.Random) -> Quaternion:
    """Generate uniformly random rotation in quaternion."""
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    
    q = Quaternion((
        math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2),
        math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2),
        math.sqrt(u1) * math.sin(2.0 * math.pi * u3),
        math.sqrt(u1) * math.cos(2.0 * math.pi * u3),
    ))
    # Convert from (x,y,z,w) to (w,x,y,z)
    q = Quaternion((q[3], q[0], q[1], q[2])).normalized()
    return q

def _rand_unit_vec(rng: random.Random) -> Vector:
    # Random point on sphere
    z = rng.uniform(-1.0, 1.0)
    t = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z*z))
    return Vector((r * math.cos(t), r * math.sin(t), z))

def _small_random_rotation(rng: random.Random, max_deg: float) -> Quaternion:
    """Return a small random axis-angle rotation quaternion."""
    if max_deg <= 0.0:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = _rand_unit_vec(rng).normalized()
    ang = math.radians(rng.uniform(-max_deg, max_deg))
    return Quaternion(axis, ang).normalized()

def quat_wxyz_to_quat(q_wxyz) -> Quaternion:
    """Convert (w,x,y,z) tuple to mathutils.Quaternion."""
    w, x, y, z = q_wxyz
    return Quaternion((w, x, y, z)).normalized()

def quat_to_wxyz(q: Quaternion) -> tuple:
    """Convert mathutils.Quaternion to (w,x,y,z) tuple."""
    q = q.normalized()
    return (q.w, q.x, q.y, q.z)

def make_T_from_q_t(q: Quaternion, t: Vector) -> Matrix:
    """Build 4x4 transform T from quaternion and translation."""
    R = q.to_matrix().to_4x4()
    T = R.copy()
    T.translation = t
    return T

def decompose_T(T: Matrix) -> Tuple[Vector, Quaternion]:
    """Extract translation and rotation from 4x4 matrix."""
    t = T.to_translation()
    R = T.to_3x3()
    q = R.to_quaternion()
    q.normalize()
    return t, q


# Trimesh depth pipeline helpers
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


def _mesh_from_scene(scene) -> trimesh.Trimesh:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = []
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        verts = np.array([obj_eval.matrix_world @ v.co for v in mesh.vertices], dtype=float)
        faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=int)
        if verts.size == 0 or faces.size == 0:
            obj_eval.to_mesh_clear()
            continue
        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        obj_eval.to_mesh_clear()

    if not meshes:
        return trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _make_ray_intersector(mesh: trimesh.Trimesh):
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    return RayMeshIntersector(mesh)


def depth_from_trimesh(scene, extrinsic_mat: np.ndarray) -> np.ndarray:
    from modules.addon_ground_truth_generation import get_scene_resolution, get_camera_parameters_intrinsic
    res_x, res_y = get_scene_resolution(scene)
    f_x, f_y, c_x, c_y = get_camera_parameters_intrinsic(scene)
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]], dtype=float)

    T_CW = _make_T_from_extrinsic(extrinsic_mat)
    T_WC = _invert_T(T_CW)

    dirs_C = _camera_rays_in_camera_frame(res_x, res_y, K)
    origins_W = np.repeat(T_WC[:3, 3].reshape(1, 3), res_x * res_y, axis=0)
    dirs_W = (T_WC[:3, :3] @ dirs_C.T).T

    mesh = _mesh_from_scene(scene)
    if mesh.is_empty:
        return np.zeros((res_y, res_x), dtype=np.float32)

    intersector = _make_ray_intersector(mesh)
    locs_W, ray_idx, _tri_idx = intersector.intersects_location(
        origins_W, dirs_W, multiple_hits=False
    )

    depth_z = np.full((res_x * res_y,), -1.0, dtype=np.float32)
    if locs_W.shape[0] == 0:
        return depth_z.reshape(res_y, res_x)

    locs_C = _transform_points(T_CW, locs_W)
    z = locs_C[:, 2]
    earth_obj = scene.objects.get("Earth")
    if earth_obj is not None:
        earth_pos = np.array(earth_obj.matrix_world.translation[:], dtype=float)
        cam_pos = T_WC[:3, 3]
        earth_radius = 0.5 * float(max(earth_obj.dimensions))
        max_dist = max(0.0, float(np.linalg.norm(cam_pos - earth_pos)) - earth_radius)
    else:
        max_dist = float(scene.camera.data.clip_end)
    valid = (z > 0) & (z <= max_dist) & np.isfinite(z)
    ray_idx = ray_idx[valid]
    z = z[valid]
    depth_z[ray_idx] = z.astype(np.float32)
    return depth_z.reshape(res_y, res_x)