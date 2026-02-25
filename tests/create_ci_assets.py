import bpy
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def add_emission_material(obj, color, name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Emission Color"].default_value = color[:3] + (1.0,)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    obj.data.materials.append(mat)


def create_mesh(name, primitive_type, filepath, radius=1.0, color=(1, 1, 1, 1)):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if primitive_type == "CUBE":
        bpy.ops.mesh.primitive_cube_add(size=radius * 2.0)
    elif primitive_type == "SPHERE":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=32, ring_count=16)

    obj = bpy.context.active_object
    obj.name = name
    add_emission_material(obj, color, f"{name}_Mat")

    if name == "Target":
        obj.pass_index = 1
        rf = bpy.data.objects.new("RF_Target", None)
        bpy.context.scene.collection.objects.link(rf)
        obj.parent = rf
    elif name == "Earth":
        obj.pass_index = 2

    ensure_dir(os.path.dirname(filepath))
    bpy.ops.wm.save_as_mainfile(filepath=filepath)


def create_minimal_scene(filepath):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if not bpy.context.scene.world:
        bpy.context.scene.world = bpy.data.worlds.new(name="SISIFOS_World")

    cam_obj = bpy.data.objects.new("Camera", bpy.data.cameras.new(name="Camera"))
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    sun_obj = bpy.data.objects.new("Sun", bpy.data.lights.new(name="Sun", type="SUN"))
    bpy.context.scene.collection.objects.link(sun_obj)

    for placeholder in ["Clouds", "Atmo"]:
        bpy.context.scene.collection.objects.link(
            bpy.data.objects.new(placeholder, None)
        )

    ensure_dir(os.path.dirname(filepath))
    bpy.ops.wm.save_as_mainfile(filepath=filepath)


if __name__ == "__main__":
    create_mesh(
        "Target",
        "CUBE",
        "assets/minimal_cube.blend",
        radius=2.5,
        color=(1.0, 0.0, 0.0, 1.0),
    )
    create_mesh(
        "Earth",
        "SPHERE",
        "assets/minimal_sphere.blend",
        radius=637.1,
        color=(0.0, 0.0, 1.0, 1.0),
    )
    create_minimal_scene("assets/minimal_scene.blend")
