import bpy
import math
from typing import List, Tuple
from mathutils import Vector, Euler, Quaternion


def clear_anim(obj):
    if obj.animation_data:
        obj.animation_data_clear()

def keyframe_pose(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    if obj.rotation_mode == 'QUATERNION':
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    else:
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

def scale_object_by_factor(obj, factor):
    old_scale = obj.scale.copy()
    obj.scale = Vector((old_scale.x * factor,
                        old_scale.y * factor,
                        old_scale.z * factor))
    
    print(f"Scaled {obj.name}: {old_scale} -> {obj.scale}")

def set_scale(obj, scale) -> None:
    """Set object scale."""
    if isinstance(scale, (float, int)):
        obj.scale = Vector((scale, scale, scale))
    else:
        obj.scale = Vector(scale)
    bpy.context.view_layer.update()

def set_position(obj, xyz: Vector) -> None:
    """Set object position."""
    obj.location = Vector(xyz)
    bpy.context.view_layer.update()

def set_rotation_euler_deg(obj, xyz_deg: List[float]) -> None:
    """Set object rotation in degrees (XYZ order)."""
    obj.rotation_euler = Euler([math.radians(v) for v in xyz_deg], 'XYZ')
    bpy.context.view_layer.update()

def look_at(obj, target_point: Vector) -> None:
    direction = (target_point - obj.location).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_quaternion = quat
    bpy.context.view_layer.update()

def set_sun_direction(sun_obj, sun_az_deg: float, sun_el_deg: float):
        """Orient a Sun object to match given azimuth/elevation (no location change) for ray-casting day-night."""
        # Convert to radians
        az = math.radians(sun_az_deg)
        el = math.radians(sun_el_deg)

        # Direction vector in world coordinates
        d = Vector((
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ))
        d.normalize()

        # In Blender, Sun lamp emits along its local -Z axis.
        # CSV positions are negated, so d (= u_sun_I in the true frame) already
        # points in the correct light-travel direction in the negated Blender world.
        # Aligning -Z with d makes light go from -scene-side toward +Earth-side,
        # illuminating the scene without Earth shadow.
        sun_obj.rotation_mode = 'QUATERNION'
        sun_obj.rotation_quaternion = d.to_track_quat('-Z', 'Y')

def append_blend_objects(filepath):
    before = set(bpy.data.objects.keys())

    with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects[:]  # append all objects

    new_objs = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            new_objs.append(obj)

    return new_objs
def get_world_bounds(obj) -> Tuple[Vector, Vector]:
    """Return min and max corners of object and children in world coordinates."""
    min_c = Vector((1e10, 1e10, 1e10))
    max_c = Vector((-1e10, -1e10, -1e10))
    
    objects_to_check = [obj] + list(obj.children_recursive)
    
    for o in objects_to_check:
        if o.type == 'MESH':
            for v in o.bound_box:
                wv = o.matrix_world @ Vector(v)
                min_c = Vector((min(min_c[i], wv[i]) for i in range(3)))
                max_c = Vector((max(max_c[i], wv[i]) for i in range(3)))
    
    return min_c, max_c

def compute_center_of_mass(obj) -> Vector:
    """Return center of mass of all mesh vertices."""
    verts = []
    for o in [obj] + list(obj.children_recursive):
        if o.type == "MESH":
            mesh = o.to_mesh()
            for v in mesh.vertices:
                verts.append(o.matrix_world @ v.co)
            o.to_mesh_clear()
    
    if not verts:
        return Vector((0, 0, 0))
    com = sum(verts, Vector()) / len(verts)
    return com