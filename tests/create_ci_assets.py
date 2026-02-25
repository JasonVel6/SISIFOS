import bpy
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_mesh(name, primitive_type, filepath):
    """Create a simple .blend containing a single primitive (used for Target/Earth).

    For `Target` we also create an `RF_Target` parent object to match renderer expectations.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if primitive_type == 'CUBE':
        bpy.ops.mesh.primitive_cube_add(size=2.0)
    elif primitive_type == 'SPHERE':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)

    obj = bpy.context.active_object
    obj.name = name

    # Wrap in RF_ target nomenclature expected by the renderer
    if name == "Target":
        rf = bpy.data.objects.new("RF_Target", None)
        bpy.context.scene.collection.objects.link(rf)
        obj.parent = rf

    ensure_dir(os.path.dirname(filepath))
    bpy.ops.wm.save_as_mainfile(filepath=filepath)


def create_minimal_scene(filepath):
    """Create a minimal scene .blend suitable for CI runs.

    Ensures a World, a Camera named "Camera", and a Sun light named "Sun" exist.
    Also places placeholder empties for Clouds and Atmo so renderer finds the names.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Ensure a World exists
    if not bpy.context.scene.world:
        w = bpy.data.worlds.new(name="SISIFOS_World")
        bpy.context.scene.world = w

    # Create Camera named "Camera"
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Create Sun light named "Sun"
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    sun_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.location = (10.0, -10.0, 10.0)

    # Create placeholders for Clouds and Atmo so renderer scaling won't KeyError
    clouds = bpy.data.objects.new("Clouds", None)
    atmo = bpy.data.objects.new("Atmo", None)
    bpy.context.scene.collection.objects.link(clouds)
    bpy.context.scene.collection.objects.link(atmo)

    ensure_dir(os.path.dirname(filepath))
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

if __name__ == '__main__':
    create_mesh("Target", 'CUBE', "assets/minimal_cube.blend")
    create_mesh("Earth", 'SPHERE', "assets/minimal_sphere.blend")
    create_minimal_scene("assets/minimal_scene.blend")