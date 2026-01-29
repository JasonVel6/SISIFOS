import math
import random
from typing import List, Tuple
from mathutils import Vector, Matrix, Quaternion

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