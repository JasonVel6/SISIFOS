import bpy
import os

def create_mesh(name, primitive_type, filepath):
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
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=filepath)

create_mesh("Target", 'CUBE', "assets/minimal_cube.blend")
create_mesh("Earth", 'SPHERE', "assets/minimal_sphere.blend")
create_mesh("Scene", 'CUBE', "assets/minimal_scene.blend")