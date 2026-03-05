# SISIFOS TODO

## Analysis: Previously Suspected Bugs That Are NOT Bugs

The following were initially flagged as bugs but are correct by design of the
inertial-frame rendering strategy:

- **HDRI star background "frozen"**: The Blender world IS the inertial frame.
  Stars are fixed in inertial space, so a fixed HDRI is correct. The camera uses
  a look-at constraint (always pointing at the target satellite), so it rotates
  at only ~0.06 deg/s — star motion is ~0.5 px/frame, which is correct. The
  4.6 deg/s figure was `q_I_G` (target tumble), not camera rotation.

- **Sun direction constant**: `sun_az/el` in `camera_traj.csv` are inertial-frame
  values. The sun is fixed in inertial space over any 500s window (~0.002 deg
  of apparent motion). Constant sun is physically correct.

- **Earth position not animated (`EARTH_BACKGROUND_ENABLE=False`)**: Earth is
  placed at the inertial-frame origin in Blender, which is correct. It does not
  need per-frame position updates.

- **`earth_dist_scale_factor=0.001`**: Intentional. Scales the 6842 km orbit
  down to 6842 m to fit within Blender's clip plane limits. The relative
  camera-to-target geometry (r_GC_I ≈ 23 m) is preserved exactly.

- **Earth not visible in images**: Physically correct. The look-at camera points
  at the target, which sits ~23 m away in roughly the outward radial direction.
  Earth center is ~123 deg away from the camera boresight. Earth's angular radius
  is 68.5 deg, so the closest limb is still ~42 deg outside the 5.6 deg FOV.
  Earth would only appear if the target were between the camera and Earth nadir.

---

## Real Issues

### 1. Quaternion hemisphere flips in `camera_traj.csv` [LOW]
At 10 frames (337, 595, 795, 1714, 1744, 2211, 3135, 3268, 3557, 4956) the
`q_I_G` quaternion sign flips (q → −q). This represents the same rotation but
causes a computed angular jump of 229° when differencing consecutive quaternions.
The actual frame-to-frame rotation is ~0.47°. Rendering is unaffected (Blender
treats q and −q identically), but any downstream code computing angular velocity
from the trajectory CSV will see spurious spikes.

**Fix:** Post-process trajectory output to enforce consistent quaternion hemisphere
(ensure `q[i] · q[i−1] > 0` for all i) before writing `camera_traj.csv`.

---

### 3. Sun-in-FOV not excluded — sensor damage / blown-out frames possible [LOW]
The sun direction `u_sun_I` is sampled randomly in inertial space with no constraint
on its relationship to the camera boresight. If the sun falls within ~10–15° of the
boresight, the rendered image will be dominated by lens flare / solar saturation and
carries no useful information.

**Note:** Back-lit and partially shadowed targets are intentionally allowed and
desirable — the goal is to stress-test dynamics-aided SLAM through low/no visual
information regions (as in SISIFOS1, where co-rotating body-frame camera creates
recurring dark regions as the target tumbles). The only physically unrealistic case
is sun-in-FOV, which no real mission would permit (sensor protection constraint).

**Fix:** At trajectory generation time, reject any MC trial where the sun falls
within a exclusion cone around the camera boresight for more than a small fraction
of frames:
```python
u_LOS_I = (p_G_I_0 - p_C_I_0) / |p_G_I_0 - p_C_I_0|
sun_in_fov = angle(u_sun_I, u_LOS_I) < SUN_EXCLUSION_DEG  # e.g. 15°
```
This is a minimal filter — everything else (side-lit, back-lit, fully dark due to
tumble) is realistic and valuable for dataset diversity.

---

### 4. No eclipse modeling — target always illuminated regardless of orbital position [MEDIUM]
The sun direction is a fixed inertial vector for the entire trajectory. No test is
performed to check whether Earth blocks the sun (i.e. whether the satellite is in
Earth's shadow). For a general LEO orbit, ~35% of each orbital period is in eclipse.
Over a 500s window this can be a significant fraction of frames. The renderer will
fully illuminate the target even during eclipse, producing physically incorrect images.

SISIFOS1 also had no eclipse modeling, but sidestepped the problem by using a
synthetic sun direction (not tied to real orbital mechanics), so eclipse was never
a concern for that dataset.

**Fix:** Add a cylindrical (or conical) shadow test at trajectory generation time:
```python
# Cylindrical approximation
dot = np.dot(p_G_I, u_sun_I)
r_perp = np.linalg.norm(p_G_I - dot * u_sun_I)
in_eclipse = (dot < 0) and (r_perp < R_EARTH_M)
```
If `in_eclipse`, either (a) reject the MC trial, (b) skip rendering those frames,
or (c) set sun lamp intensity to zero and render with earthshine-only ambient.

---

### 2. NPZ `object_poses` records sub-mesh parts, not the animated root [MEDIUM]
The vision_blender addon records poses for all named objects in the scene. The
animated root (`RF_Hubble`) moves and tumbles correctly, but the named entries in
`object_poses` (`01 - Default`, `02 - Default`, etc.) are child mesh parts whose
world-space pose changes only when the root moves — yet they appear static across
all 5000 frames. This suggests either (a) the root object's pose is not included
in the NPZ, or (b) the addon captures poses before Blender propagates the
parent-child transform for the current frame.

The ground-truth object poses in the NPZ are therefore not usable for pose
estimation evaluation.

**Fix:** Ensure `bpy.context.view_layer.update()` (or `bpy.context.evaluated_depsgraph_get()`)
is called before the addon captures poses, and verify that the `RF_Hubble` root object
itself appears in `object_poses` with its correct per-frame world transform.

---

### 5. Stars not visible despite `stars_mode: "on"` [MEDIUM]
In the 2026-02-24_1531 Proba-3 render, the background is pure black even though
`stars_mode: "on"` and the HDRI path (`starmap_2020_16k.exr`) are set correctly.
The Hubble render (2026-02-23_2227) does show stars. Suspected causes:
- `bg_color: [0,0,0,1]` in the render config may be overriding the HDRI for camera
  rays (Cycles world visibility setting)
- HDRI world strength may be zero in the `.blend` scene file for this model/scene
- HDRI may be set to lighting-only (not visible to camera rays) in Cycles world settings

**Fix:** Check renderer.py's `stars_mode` handling — confirm it sets both world
strength > 0 AND enables camera ray visibility for the world shader node. Compare
the Blender world node setup between the Hubble render (works) and Proba-3 (broken).
