import bpy
import os
import math
from mathutils import Vector

# =========================================================
# USER CONFIG
# =========================================================
FBX_FOLDER = "/home/ivelentzas3/VELE_HUB/MODELS_3D_ESA"
GLB_FOLDER = "/home/ivelentzas3/VELE_HUB/MODELS_3D_NASA"
OUTPUT_BLEND = "/home/ivelentzas3/VELE_HUB/SISIFOS/spacecraft_models_0_25.blend"

# =========================================================
# CLEAN START (new file)
# =========================================================
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene
collection = scene.collection

# =========================================================
# HELPERS
# =========================================================
def get_root_world_dimensions(root_obj):
    """Compute world-space bounding box dimensions for root object incl. child meshes."""
    depsgraph = bpy.context.evaluated_depsgraph_get()

    min_c = Vector((1e10, 1e10, 1e10))
    max_c = Vector((-1e10, -1e10, -1e10))

    for obj in [root_obj] + list(root_obj.children_recursive):
        if obj.type != 'MESH':
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh.transform(obj_eval.matrix_world)

        for v in mesh.vertices:
            min_c.x = min(min_c.x, v.co.x)
            min_c.y = min(min_c.y, v.co.y)
            min_c.z = min(min_c.z, v.co.z)

            max_c.x = max(max_c.x, v.co.x)
            max_c.y = max(max_c.y, v.co.y)
            max_c.z = max(max_c.z, v.co.z)

        obj_eval.to_mesh_clear()

    return (max_c - min_c)

def power_of_10_scale(max_dim, target_min=5.0, target_max=50.0):
    """Return scale factor (power of 10) that brings max_dim into [target_min, target_max]."""
    if max_dim == 0:
        return 1.0
    scale = 1.0
    while max_dim * scale < target_min:
        scale *= 10.0
    while max_dim * scale > target_max:
        scale /= 10.0
    return scale

def compute_center_of_mass(root_obj):
    """Compute center of mass (average of mesh vertices) in WORLD coordinates for root + children."""
    depsgraph = bpy.context.evaluated_depsgraph_get()

    acc = Vector((0.0, 0.0, 0.0))
    count = 0

    for obj in [root_obj] + list(root_obj.children_recursive):
        if obj.type != 'MESH':
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh.transform(obj_eval.matrix_world)

        for v in mesh.vertices:
            acc += v.co
            count += 1

        obj_eval.to_mesh_clear()

    if count == 0:
        return Vector((0.0, 0.0, 0.0))
    return acc / count

def recenter_model_to_com(root_obj, verbose=True):
    """Move all geometry so RF origin coincides with the model COM."""
    com = compute_center_of_mass(root_obj)
    if verbose:
        print(f"[Recenter] {root_obj.name}")
        print(f"  COM (world): {com}")

    # Shift children so COM moves to origin
    for obj in root_obj.children_recursive:
        obj.location -= com

    bpy.context.view_layer.update()

def normalize_model(root_obj, verbose=True):
    """
    1. Recenter geometry so RF origin = COM
    2. Scale (your current code uses 1/max_dim)
    3. Place RF at (0,0,0)
    """
    recenter_model_to_com(root_obj, verbose=verbose)

    dims = get_root_world_dimensions(root_obj)
    max_dim = max(dims.x, dims.y, dims.z)

    # Your original scaling choice:
    scale = 10.0 / max_dim if max_dim != 0 else 1.0
    if scale >1000:
        scale = 1 / max_dim
    if scale <0.001:
        scale = 5 / max_dim
    if verbose:
        print(f"  Max dim before scale: {max_dim:.4f}")
        print(f"  Scale factor:         {scale}")

    root_obj.scale *= scale
    root_obj.location = Vector((0.0, 0.0, 0.0))
    bpy.context.view_layer.update()

    dims_after = get_root_world_dimensions(root_obj)
    if verbose:
        print(f"  Max dim after scale:  {max(dims_after):.4f}")

def assign_segmentation_ids(imported_objects, start_id=1):
    """Assign unique pass_index to each MESH object in imported_objects."""
    idx = start_id
    for obj in imported_objects:
        if obj.type == 'MESH':
            obj.pass_index = idx
            idx += 1

def create_reference_frame(name):
    """Create a hidden Empty used as a reference frame."""
    rf = bpy.data.objects.new(name, None)
    rf.empty_display_type = 'PLAIN_AXES'
    rf.empty_display_size = 0.0
    rf.hide_viewport = True
    rf.hide_render = True
    collection.objects.link(rf)
    return rf

def import_fbx(filepath):
    """Import FBX and return newly imported objects."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=filepath)
    after = set(bpy.data.objects)
    return list(after - before)

def import_glb(filepath):
    """Import GLB/GLTF and return newly imported objects."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)  # works for .glb and .gltf
    after = set(bpy.data.objects)
    return list(after - before)

def file_stem(path):
    """Filename without extension (safe-ish label)."""
    return os.path.splitext(os.path.basename(path))[0]

# =========================================================
# MODEL NAME MAP (for ESA FBX)
# =========================================================
model_names = {
    # -------------------------------------------------
    # Rosetta / Comet missions
    # -------------------------------------------------
    0:  "Rosetta",
    1:  "Philae",
    2:  "Giotto",

    # -------------------------------------------------
    # BepiColombo mission family
    # -------------------------------------------------
    3:  "Bepi-mpo",
    4:  "Bepi-mmo",
    5:  "Bepi-mtm",
    6:  "Bepi-mcs",

    # -------------------------------------------------
    # Proba missions
    # -------------------------------------------------
    7:  "Proba-2",
    8:  "Proba-3",
    9:  "Proba-3-csc",
    10: "Proba-3-osc",

    # -------------------------------------------------
    # Cassini / Saturn system
    # -------------------------------------------------
    11: "Cassini",
    12: "Cassini-huygens-in",

    # -------------------------------------------------
    # ExoMars / Mars exploration
    # -------------------------------------------------
    13: "Mars-express",
    14: "TGO",
    15: "TGO-edm",
    16: "Schiaparelli",

    # -------------------------------------------------
    # Solar & heliophysics missions
    # -------------------------------------------------
    17: "Solar-orbiter",
    18: "Soho",
    19: "Ulysses",

    # -------------------------------------------------
    # Astrophysics / space observatories
    # -------------------------------------------------
    20: "Euclid",
    21: "Gaia",
    22: "Planck",
    23: "Herschel",
    24: "XMM-Newton",

    # -------------------------------------------------
    # Technology / pathfinder missions
    # -------------------------------------------------
    25: "Lisa-pathfinder",
    26: "Smart-1",
    27: "Cheops",

    # -------------------------------------------------
    # Earth / multi-spacecraft missions
    # -------------------------------------------------
    28: "Cluster",
    29: "Double-star",

    # -------------------------------------------------
    # Other ESA science missions
    # -------------------------------------------------
    30: "Integral",
    31: "Venus-express",
    32: "Juice",

    # =================================================
    # NASA / Other missions
    # =================================================

    # -------------------------------------------------
    # Earth observing / climate & atmosphere
    # -------------------------------------------------
    33: "Acrimsat_B",
    34: "Aqua_B_compact",
    35: "Aqua_B",
    36: "Aqua_B_unfurled",
    37: "Aura_B",
    38: "Cloudsat_A",
    39: "Cloudsat_C",
    40: "Icesat_B",
    41: "ISS",
    42: "Jason1",
    43: "Landsat8",
    44: "QuickSCAT",

    # -------------------------------------------------
    # Space telescopes / astrophysics observatories
    # -------------------------------------------------
    45: "Chandra",
    46: "Hubble",

    # -------------------------------------------------
    # Planetary / deep-space exploration & infrastructure
    # -------------------------------------------------
    47: "Cubesat_Mirata",
    48: "Dawn",
    49: "DeepSpace1",
    50: "Gateway_core",
    51: "IBEX",
    52: "Magellan",
    53: "MarsOdyssey",
    54: "MAVEN",
    55: "MRO",
    56: "NEAR",

    # -------------------------------------------------
    # Communications / relay satellites
    # -------------------------------------------------
    57: "TDRS",

    # -------------------------------------------------
    # Heliophysics / space environment
    # -------------------------------------------------
    58: "THEMIS",
}


# =========================================================
# MAIN: FBX (ESA)
# =========================================================
fbx_files = sorted(f for f in os.listdir(FBX_FOLDER) if f.lower().endswith(".fbx"))
assert len(fbx_files) <= len(model_names), "Not enough model names for FBX files"

for idx, fbx_name in enumerate(fbx_files):
    model_label = model_names[idx]
    fbx_path = os.path.join(FBX_FOLDER, fbx_name)

    print(f"\nImporting FBX {fbx_name} → {model_label}")

    rf = create_reference_frame(f"RF_{model_label}")
    imported_objects = import_fbx(fbx_path)

    for obj in imported_objects:
        obj.parent = rf

    normalize_model(rf)
    assign_segmentation_ids(imported_objects)
    print(f"  Attached {len(imported_objects)} objects with segmentation IDs")

# =========================================================
# MAIN: GLB/GLTF (NASA)
# =========================================================
glb_files = sorted(
    f for f in os.listdir(GLB_FOLDER)
    if f.lower().endswith((".glb", ".gltf"))
)

for glb_name in glb_files:
    glb_path = os.path.join(GLB_FOLDER, glb_name)
    model_label = file_stem(glb_path)  # default: use filename

    print(f"\nImporting GLB {glb_name} → {model_label}")

    rf = create_reference_frame(f"RF_{model_label}")
    imported_objects = import_glb(glb_path)

    for obj in imported_objects:
        obj.parent = rf

    normalize_model(rf)
    assign_segmentation_ids(imported_objects)
    print(f"  Attached {len(imported_objects)} objects with segmentation IDs")

# =========================================================
# PACK + SAVE
# =========================================================
print("\nPacking all external resources...")
bpy.ops.file.pack_all()

print(f"\nSaving blend file to: {OUTPUT_BLEND}")
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT_BLEND)

print("\n✔ Done.")
