import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Blender
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------- utilities ---------------------------

def _set_equal_3d(ax, X, Y, Z, margin=0.05):
    """Equal aspect for 3D axes covering data bounds."""
    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    rx = 0.5 * (xmax - xmin)
    ry = 0.5 * (ymax - ymin)
    rz = 0.5 * (zmax - zmin)
    r = max(rx, ry, rz, 1e-9) * (1.0 + margin)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)


def _draw_axes_triad(ax, origin, scale, label_prefix, linewidth=1.8, alpha=0.9):
    """Draw +X,+Y,+Z axes from origin, labeled with <prefix>_x/y/z."""
    ox, oy, oz = origin
    ax.quiver(ox, oy, oz, scale, 0,     0,     color="r", linewidth=linewidth, alpha=alpha)
    ax.quiver(ox, oy, oz, 0,     scale, 0,     color="g", linewidth=linewidth, alpha=alpha)
    ax.quiver(ox, oy, oz, 0,     0,     scale, color="b", linewidth=linewidth, alpha=alpha)
    # label tips
    ax.text(ox + 1.05*scale, oy,             oz,             f"{label_prefix}_x", fontsize=9, color="r")
    ax.text(ox,             oy + 1.05*scale, oz,             f"{label_prefix}_y", fontsize=9, color="g")
    ax.text(ox,             oy,             oz + 1.05*scale, f"{label_prefix}_z", fontsize=9, color="b")


def _nice_title_for_mode(mc_idx, rotMode_Gframe):
    m = str(rotMode_Gframe).strip()
    if m == "1":
        tag = "Inertial case"
    elif m == "2":
        tag = "Hill case"
    elif m == "3":
        tag = "Tumbling case"
    else:
        tag = f"Mode {m}"
    return f"MC {mc_idx:03d} — {tag}"

def _draw_g_axes(ax, GXs, GYs, GZs, axis_length=None):
    """
    Draw G-frame axes (Gx, Gy, Gz) at the origin in the G-frame subplot.
    Also appends their endpoints into GXs/GYs/GZs so the later bounds include them.
    axis_length: if None, choose ~15% of current data span (with a small floor).
    """
    # Compute a provisional span from what has been plotted so far.
    GX = np.concatenate(GXs) if GXs else np.array([0.0])
    GY = np.concatenate(GYs) if GYs else np.array([0.0])
    GZ = np.concatenate(GZs) if GZs else np.array([0.0])

    span_x = GX.max() - GX.min() if GX.size else 1.0
    span_y = GY.max() - GY.min() if GY.size else 1.0
    span_z = GZ.max() - GZ.min() if GZ.size else 1.0
    span   = max(span_x, span_y, span_z, 1.0)

    L = float(axis_length) if axis_length is not None else max(0.15 * span, 1e-3)

    # Plot triad (no legend labels to avoid clutter)
    ax.plot([0.0, L], [0.0, 0.0], [0.0, 0.0], lw=2)  # Gx
    ax.plot([0.0, 0.0], [0.0, L], [0.0, 0.0], lw=2)  # Gy
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, L], lw=2)  # Gz

    # Axis tags near the tips
    ax.text(L, 0.0, 0.0, "Gx", fontsize="small")
    ax.text(0.0, L, 0.0, "Gy", fontsize="small")
    ax.text(0.0, 0.0, L, "Gz", fontsize="small")

    # Ensure bounds include the triad tips by appending their endpoints
    GXs.append(np.array([0.0, L])); GYs.append(np.array([0.0, 0.0])); GZs.append(np.array([0.0, 0.0]))
    GXs.append(np.array([0.0, 0.0])); GYs.append(np.array([0.0, L])); GZs.append(np.array([0.0, 0.0]))
    GXs.append(np.array([0.0, 0.0])); GYs.append(np.array([0.0, 0.0])); GZs.append(np.array([0.0, L]))

# --------------------------- main entry ---------------------------

def plot_trial_trajectories(
    state_A_I,         # (nbSteps, 6) Target/Frame A state in I (pos[0:3], vel[3:6])
    state_C_I,         # (num_agents, nbSteps, 6) Agent states in I
    r_CG_G,        # (num_agents, nbSteps, 3) Relative positions in G (what SLAM uses)
    R_IG_all=None, # (nbSteps, 3,3) for projecting variants into I
    out_dir=None,
    rotMode_Gframe=None,
    show=True,
    save=True,
    variant_rCGG=None,   # dict like {"_nmc": (num_agents,T,3), "_cro": (num_agents,T,3)} to overlay in G-frame
    mc_idx=None
):
    """
    One figure per MC `i` with THREE synchronized 3D views:
      (1) A-centered Inertial: r_CA^I(t) = r_C^I - r_A^I
      (2) Earth Inertial (absolute): r_A^I(t), r_C^I(t)
      (3) Body/SLAM G-frame: r_CG^G(t)
    """
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    num_agents = state_C_I.shape[0]
    print(f"[PLOTS] Plotting trajectories for {num_agents} agents...")
    colors = cm.tab10(np.linspace(0, 1, max(3, num_agents)))

    # Extract this trial’s trajectories
    rA = state_A_I[:, 0:3]  # (nbSteps,3)
    R_IG_i = R_IG_all  # (T,3,3) or None

    # Build relative (A-centered inertial) and absolute lists
    relI_list = []
    abs_listX, abs_listY, abs_listZ = [rA[:,0]], [rA[:,1]], [rA[:,2]]

    for agent_idx in range(num_agents):
        rS = state_C_I[agent_idx, :, 0:3]
        relI_list.append(rS - rA)
        abs_listX.append(rS[:,0]); abs_listY.append(rS[:,1]); abs_listZ.append(rS[:,2])

    # --- Figure with 3 subplots ---
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.22)

    ax_relI = fig.add_subplot(gs[0, 0], projection="3d")
    ax_inr  = fig.add_subplot(gs[0, 1], projection="3d")
    ax_G    = fig.add_subplot(gs[0, 2], projection="3d")

    # -------- (1) A-centered Inertial --------
    ax_relI.set_title(f"{_nice_title_for_mode(mc_idx, rotMode_Gframe)} — A-Centered Relative Paths")
    ax_relI.set_xlabel("x_A [m]"); ax_relI.set_ylabel("y_A [m]"); ax_relI.set_zlabel("z_A [m]")
    ax_relI.grid(True, alpha=0.3)

    for agent_idx, rel in enumerate(relI_list):
        c = colors[agent_idx]
        ax_relI.plot(rel[:,0], rel[:,1], rel[:,2], lw=1.6, color=c, label=f"Agent {agent_idx}")
        ax_relI.scatter(rel[0,0],  rel[0,1],  rel[0,2],  s=18, color=c, marker="o")
        ax_relI.scatter(rel[-1,0], rel[-1,1], rel[-1,2], s=28, color=c, marker="x")
        ax_relI.text(rel[0,0], rel[0,1], rel[0,2], f"Agent {agent_idx} start", fontsize=8, color=c)
    ax_relI.scatter(0.0, 0.0, 0.0, s=36, color="k", marker="*", label="A (origin)")
    XR = np.concatenate([r[:,0] for r in relI_list]) if relI_list else np.array([0.0])
    YR = np.concatenate([r[:,1] for r in relI_list]) if relI_list else np.array([0.0])
    ZR = np.concatenate([r[:,2] for r in relI_list]) if relI_list else np.array([0.0])
    _set_equal_3d(ax_relI, XR, YR, ZR)
    ax_relI.legend(loc="upper right", fontsize="small")

    # --- Overlay inertial variants on the A-centered plot (relI): r_CA^I = R_IG * r_CG^G
    if str(rotMode_Gframe).strip() == "1" and isinstance(variant_rCGG, dict) and R_IG_i is not None:
        for suf, r_var in variant_rCGG.items():
            if r_var is None: 
                continue
            # normalize shape
            if r_var.ndim == 4 and r_var.shape[0] == 1:
                r_var = r_var[0]
            if r_var.ndim != 3 or r_var.shape[2] != 3:
                continue
            label_base = f"traj{mc_idx}_{'nmc' if suf.endswith('nmc') else 'cro' if suf.endswith('cro') else suf.strip('_')}"
            for agent_idx in range(min(r_var.shape[0], len(relI_list))):
                T = r_var.shape[1]
                relI_var = np.empty_like(r_var[agent_idx])
                for j in range(T):
                    relI_var[j] = R_IG_i[j] @ r_var[agent_idx, j]
                L = label_base if agent_idx == 0 else None
                ax_relI.plot(relI_var[:,0], relI_var[:,1], relI_var[:,2],
                             lw=1.8, linestyle="--", label=L)
        # refresh legend to include overlays
        ax_relI.legend(loc="upper right", fontsize="small")


    # -------- (2) Earth Inertial (absolute) --------
    ax_inr.set_title(f"{_nice_title_for_mode(mc_idx, rotMode_Gframe)} — Inertial Trajectories (I)")
    ax_inr.set_xlabel("x_I [m]"); ax_inr.set_ylabel("y_I [m]"); ax_inr.set_zlabel("z_I [m]")
    ax_inr.grid(True, alpha=0.3)

    # A path
    ax_inr.plot(rA[:,0], rA[:,1], rA[:,2], lw=2.0, color="k", label="A (in I)")
    ax_inr.scatter(rA[0,0], rA[0,1], rA[0,2], s=20, color="k", marker="o")
    ax_inr.scatter(rA[-1,0], rA[-1,1], rA[-1,2], s=30, color="k", marker="x")

    # Agents
    for agent_idx in range(num_agents):
        rS = state_C_I[agent_idx, :, 0:3]
        c  = colors[agent_idx]
        ax_inr.plot(rS[:,0], rS[:,1], rS[:,2], lw=1.6, color=c, label=f"Agent {agent_idx} (in I)")
        ax_inr.scatter(rS[0,0],  rS[0,1],  rS[0,2],  s=18, color=c, marker="o")
        ax_inr.scatter(rS[-1,0], rS[-1,1], rS[-1,2], s=28, color=c, marker="x")

    XI = np.concatenate(abs_listX); YI = np.concatenate(abs_listY); ZI = np.concatenate(abs_listZ)
    _set_equal_3d(ax_inr, XI, YI, ZI, margin=0.08)
    # I-frame triad at origin:
    span = max(np.ptp(XI), np.ptp(YI), np.ptp(ZI), 1.0)
    triad = 0.12 * span
    _draw_axes_triad(ax_inr, origin=(0,0,0), scale=triad, label_prefix="I")
    ax_inr.legend(loc="upper right", fontsize="small")

    # --- Overlay inertial variants on absolute plot: r_CO^I = r_AO^I + R_IG * r_CG^G
    if str(rotMode_Gframe).strip() == "1" and isinstance(variant_rCGG, dict) and R_IG_i is not None:
        for suf, r_var in variant_rCGG.items():
            if r_var is None:
                continue
            if r_var.ndim == 4 and r_var.shape[0] == 1:
                r_var = r_var[0]
            if r_var.ndim != 3 or r_var.shape[2] != 3:
                continue
            label_base = f"Agent 0 ({'NMC' if suf.endswith('nmc') else 'CRO' if suf.endswith('cro') else suf.strip('_').upper()})"
            for agent_idx in range(min(r_var.shape[0], state_C_I.shape[0])):
                T = r_var.shape[1]
                r_CO_I_var = np.empty_like(r_var[agent_idx])
                for j in range(T):
                    r_CO_I_var[j] = rA[j] + (R_IG_i[j] @ r_var[agent_idx, j])
                L = label_base if agent_idx == 0 else None
                ax_inr.plot(r_CO_I_var[:,0], r_CO_I_var[:,1], r_CO_I_var[:,2],
                            lw=1.8, linestyle="--", label=L)
        ax_inr.legend(loc="upper right", fontsize="small")


    # -------- (3) Body/SLAM G-frame (r_CG^G) --------
    ax_G.set_title("Body/SLAM G-frame: r_CG^G")
    ax_G.set_xlabel("x_G [m]"); ax_G.set_ylabel("y_G [m]"); ax_G.set_zlabel("z_G [m]")
    ax_G.grid(True, alpha=0.3)

    GXs, GYs, GZs = [], [], []
    for agent_idx in range(num_agents):
        rG = r_CG_G[agent_idx, :, :]  # (T,3)
        c  = colors[agent_idx]
        ax_G.plot(rG[:,0], rG[:,1], rG[:,2], lw=1.6, color=c, label=f"Agent {agent_idx}")
        ax_G.scatter(rG[0,0],  rG[0,1],  rG[0,2],  s=18, color=c, marker="o")
        ax_G.scatter(rG[-1,0], rG[-1,1], rG[-1,2], s=28, color=c, marker="x")
        GXs.append(rG[:,0]); GYs.append(rG[:,1]); GZs.append(rG[:,2])

    # ----- overlay inertial variants: traj{i}_nmc / traj{i}_cro -----
    if str(rotMode_Gframe).strip() == "1" and isinstance(variant_rCGG, dict) and len(variant_rCGG) > 0:
        # plot each provided variant; use dashed linestyle so overlap is visible
        for suf, r_var in variant_rCGG.items():
            if r_var is None:
                continue
            # Accept either (num_agents,T,3) or (1,num_agents,T,3)
            if r_var.ndim == 4 and r_var.shape[0] == 1:
                r_var = r_var[0]
            if r_var.ndim != 3 or r_var.shape[2] != 3:
                continue  # skip unexpected shapes silently

            label_base = f"traj{mc_idx}_{'nmc' if suf.endswith('nmc') else 'cro' if suf.endswith('cro') else suf.strip('_')}"
            for agent_idx in range(min(r_var.shape[0], num_agents)):
                L = label_base if agent_idx == 0 else None  # only first agent gets legend label
                ax_G.plot(
                    r_var[agent_idx, :, 0],
                    r_var[agent_idx, :, 1],
                    r_var[agent_idx, :, 2],
                    lw=1.6, linestyle="--", label=L
                )
    
    # draw G-frame triad at the origin and include it in bounds
    _draw_g_axes(ax_G, GXs, GYs, GZs)

    GX = np.concatenate(GXs) if GXs else np.array([0.0])
    GY = np.concatenate(GYs) if GYs else np.array([0.0])
    GZ = np.concatenate(GZs) if GZs else np.array([0.0])
    _set_equal_3d(ax_G, GX, GY, GZ)
    ax_G.legend(loc="upper right", fontsize="small")
    

    fig.suptitle(f"MC {mc_idx:03d} — Coordinated Views", y=0.98)

    if save:
        out_png = os.path.join(out_dir, f"MC_{mc_idx:03d}_traj.png")
        fig.savefig(out_png, dpi=170)
        print(f"[PLOTS] Saved: {out_png}")

    if show:
        plt.show(block=True)
    return fig

# ======================================================
# 3D SCENE VISUALIZATION
# ======================================================
def plot_scene_frame(frame_idx, camera_loc, target_loc, sun_dir_I,
                     camera_trajectory, target_trajectory, output_dir,
                     show_trajectory_window=50, sun_az_deg=None, sun_el_deg=None,
                     sun_cam_angle_G=None):
    """
    Generate a 3D plot showing the Blender scene geometry for a single frame.

    Scene setup (INERTIAL FRAME):
    - Earth is at origin (0, 0, 0)
    - Sun direction from sun_az/sun_el (per-frame)
    - Target MOVES around Earth
    - Camera MOVES with target, looking at target

    Parameters:
    -----------
    frame_idx : int
        Current frame number
    camera_loc : array-like (3,)
        Camera position in inertial/world coordinates
    target_loc : array-like (3,)
        Target position in inertial/world coordinates
    sun_dir_I : array-like (3,)
        Sun direction unit vector (from sun_az/sun_el converted to Cartesian)
    camera_trajectory : array-like (N, 3)
        Full camera trajectory in inertial coordinates
    target_trajectory : array-like (N, 3)
        Full target trajectory in inertial coordinates
    output_dir : str
        Directory to save the plot
    show_trajectory_window : int
        Number of frames before/after to show in trajectory
    sun_az_deg : float, optional
        Sun azimuth in degrees (for display)
    sun_el_deg : float, optional
        Sun elevation in degrees (for display)
    sun_cam_angle_G : float, optional
        Pre-computed sun-camera angle in G frame (accurate lighting metric)
    """
    fig = plt.figure(figsize=(16, 5))

    # Convert to numpy arrays
    camera_loc = np.array(camera_loc)
    target_loc = np.array(target_loc)
    sun_dir = np.array(sun_dir_I)
    sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-9)

    # Camera look-at direction (camera looks toward target)
    cam_to_target = target_loc - camera_loc
    cam_to_target_dist = np.linalg.norm(cam_to_target)
    look_at_dir = cam_to_target / (cam_to_target_dist + 1e-9)

    # Earth direction from target
    earth_dir_from_target = -target_loc / (np.linalg.norm(target_loc) + 1e-9)

    # Earth direction from target (for display)
    earth_lookat_angle = np.degrees(np.arccos(np.clip(np.dot(earth_dir_from_target, look_at_dir), -1, 1)))

    # Use pre-computed G-frame angle if available (accurate), otherwise fall back to inertial approx
    if sun_cam_angle_G is not None:
        sun_camera_angle = sun_cam_angle_G
    else:
        # Fallback: compute in inertial frame (less accurate)
        sun_camera_angle = np.degrees(np.arccos(np.clip(np.dot(sun_dir, -look_at_dir), -1, 1)))

    # Refined illumination classification with transition zone
    # - Full front-lit: sun_camera_angle < 80° (definitely lit)
    # - Transition zone: 80° <= angle < 100° (partial/uncertain lighting)
    # - Full back-lit: angle >= 100° (definitely in shadow)
    if sun_camera_angle < 80:
        target_color = 'gold'
        lit_status = "FRONT-LIT"
    elif sun_camera_angle < 100:
        target_color = 'orange'
        lit_status = f"TRANSITION ({sun_camera_angle:.1f}°)"
    else:
        target_color = 'gray'
        lit_status = "BACK-LIT"

    is_front_lit = sun_camera_angle < 90  # Keep for legacy compatibility

    # Trajectory window
    n_frames = len(camera_trajectory)
    start_idx = max(0, frame_idx - show_trajectory_window)
    end_idx = min(n_frames, frame_idx + show_trajectory_window)

    # ===== Plot 1: World frame view (Earth at origin) =====
    ax1 = fig.add_subplot(1, 4, 1, projection='3d')

    # Earth at origin
    ax1.scatter([0], [0], [0], c='green', s=200, marker='o', label='Earth', zorder=5)

    # Target position
    ax1.scatter([target_loc[0]], [target_loc[1]], [target_loc[2]],
                c=target_color, s=100, marker='*', label='Target', zorder=5)

    # Camera position
    ax1.scatter([camera_loc[0]], [camera_loc[1]], [camera_loc[2]],
                c='blue', s=80, marker='^', label='Camera', zorder=5)

    # Camera look-at arrow (from camera toward target)
    arrow_scale = cam_to_target_dist * 0.8
    look_arrow = look_at_dir * arrow_scale
    ax1.quiver(camera_loc[0], camera_loc[1], camera_loc[2],
               look_arrow[0], look_arrow[1], look_arrow[2],
               color='blue', arrow_length_ratio=0.1, lw=2, label='Look-at')

    # Sun direction arrow (from origin, showing light direction)
    sun_arrow = sun_dir * arrow_scale
    ax1.quiver(0, 0, 0, sun_arrow[0], sun_arrow[1], sun_arrow[2],
               color='orange', arrow_length_ratio=0.1, lw=3, label='Sun dir')

    # Target trajectory
    if end_idx > start_idx:
        traj_t = target_trajectory[start_idx:end_idx]
        ax1.plot(traj_t[:, 0], traj_t[:, 1], traj_t[:, 2], 'r-', alpha=0.4, lw=1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    sun_info = f"az={sun_az_deg:.1f}° el={sun_el_deg:.1f}°" if sun_az_deg is not None else f"∠={sun_camera_angle:.1f}°"
    ax1.set_title(f'World Frame (Earth at origin)\n{lit_status} | Sun {sun_info}')
    ax1.legend(loc='upper left', fontsize=6)

    # ===== Plot 2: Target-centered view =====
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')

    # Positions relative to target
    cam_rel = camera_loc - target_loc
    earth_rel = -target_loc

    # Target at center
    ax2.scatter([0], [0], [0], c=target_color, s=150, marker='*', label='Target', zorder=5)

    # Camera position relative to target
    ax2.scatter([cam_rel[0]], [cam_rel[1]], [cam_rel[2]],
                c='blue', s=100, marker='^', label='Camera', zorder=5)

    # Camera look-at arrow (from camera toward target = toward origin)
    look_arrow_rel = -cam_rel / (np.linalg.norm(cam_rel) + 1e-9) * cam_to_target_dist * 0.5
    ax2.quiver(cam_rel[0], cam_rel[1], cam_rel[2],
               look_arrow_rel[0], look_arrow_rel[1], look_arrow_rel[2],
               color='blue', arrow_length_ratio=0.15, lw=2, label='Look-at')

    # Sun direction arrow
    arrow_scale2 = cam_to_target_dist * 0.6
    sun_arrow2 = sun_dir * arrow_scale2
    ax2.quiver(0, 0, 0, sun_arrow2[0], sun_arrow2[1], sun_arrow2[2],
               color='orange', arrow_length_ratio=0.15, lw=3, label='Sun dir')

    # Earth direction arrow (from target toward Earth)
    earth_arrow = earth_dir_from_target * arrow_scale2
    ax2.quiver(0, 0, 0, earth_arrow[0], earth_arrow[1], earth_arrow[2],
               color='green', arrow_length_ratio=0.15, lw=3, label='Earth dir')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'Frame {frame_idx} | Range: {cam_to_target_dist:.1f}m')
    ax2.legend(loc='upper left', fontsize=6)

    max_range = cam_to_target_dist * 1.5
    ax2.set_xlim([-max_range, max_range])
    ax2.set_ylim([-max_range, max_range])
    ax2.set_zlim([-max_range, max_range])

    # ===== Plot 3: Top-down view (X-Y plane) =====
    ax3 = fig.add_subplot(1, 4, 3)

    # Draw lit hemisphere indicator
    r_indicator = max_range * 0.9
    sun_2d = sun_dir[:2]
    sun_2d_norm = np.linalg.norm(sun_2d)
    if sun_2d_norm > 0.1:
        sun_2d_unit = sun_2d / sun_2d_norm
        sun_angle = np.arctan2(sun_2d_unit[1], sun_2d_unit[0])
        theta_lit = np.linspace(sun_angle - np.pi/2, sun_angle + np.pi/2, 50)
        x_lit = r_indicator * np.cos(theta_lit)
        y_lit = r_indicator * np.sin(theta_lit)
        ax3.fill(np.append(x_lit, 0), np.append(y_lit, 0), color='yellow', alpha=0.15, label='Lit side')

    # Target at center
    ax3.scatter([0], [0], c=target_color, s=150, marker='*', label='Target', zorder=5)

    # Camera
    ax3.scatter([cam_rel[0]], [cam_rel[1]], c='blue', s=100, marker='^', label='Camera', zorder=5)

    # Camera look-at arrow
    ax3.arrow(cam_rel[0], cam_rel[1], look_arrow_rel[0], look_arrow_rel[1],
              head_width=2, head_length=1.5, fc='blue', ec='blue', lw=1.5)

    # Sun direction
    ax3.arrow(0, 0, sun_arrow2[0], sun_arrow2[1], head_width=3, head_length=2,
              fc='orange', ec='orange', lw=2, label='Sun')

    # Earth direction
    ax3.arrow(0, 0, earth_arrow[0], earth_arrow[1], head_width=3, head_length=2,
              fc='green', ec='green', lw=2, label='Earth')

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Top-Down (X-Y)\nSun-Cam angle: {sun_camera_angle:.1f}°')
    ax3.set_aspect('equal')
    ax3.legend(loc='upper left', fontsize=6)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-max_range, max_range])
    ax3.set_ylim([-max_range, max_range])

    # ===== Plot 4: Alignment diagram (conceptual) =====
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_xlim(-1.5, 1.5)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_aspect('equal')
    ax4.axis('off')

    # Draw alignment: Sun -> Camera -> Target -> Earth
    positions = {'Sun': -1.2, 'Camera': -0.4, 'Target': 0.4, 'Earth': 1.2}
    colors = {'Sun': 'orange', 'Camera': 'blue', 'Target': target_color, 'Earth': 'green'}
    markers = {'Sun': 'o', 'Camera': '^', 'Target': '*', 'Earth': 'o'}
    sizes = {'Sun': 200, 'Camera': 150, 'Target': 200, 'Earth': 200}

    for name, x in positions.items():
        ax4.scatter([x], [0], c=colors[name], s=sizes[name], marker=markers[name], zorder=5)
        ax4.text(x, -0.25, name, ha='center', fontsize=9)

    # Draw arrows showing ideal alignment
    ax4.annotate('', xy=(-0.5, 0), xytext=(-1.0, 0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax4.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.annotate('', xy=(1.0, 0), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

    ax4.set_title(f'{lit_status}\nSun-Cam: {sun_camera_angle:.1f}° | Earth-LookAt: {earth_lookat_angle:.1f}°')

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"scene_{frame_idx:04d}.png")
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

    return output_path

def generate_scene_plots(output_dir: str,
                            p_C_I: np.ndarray,
                            p_G_I: np.ndarray,
                            sun_az_I: np.ndarray,
                            sun_el_I: np.ndarray,
                            r_CG_arr: np.ndarray,
                            q_IG_arr: np.ndarray,
                            every_n_frames=1,
                            max_frames=None):
    """
    Generate 3D scene plots for multiple frames in the INERTIAL FRAME.

    Scene setup:
    - Earth is at origin (0, 0, 0) in inertial frame
    - Sun is STATIC in inertial frame (set from frame 0 sun_az/sun_el)
    - Target MOVES around Earth AND TUMBLES (via q_IG)
    - Camera MOVES with target, looking at target
    - Lighting is computed by transforming sun direction into body frame

    Parameters:
    -----------
    output_dir : str
        Directory to save plots
    p_C_I : array-like (N, 3)
        Camera positions in inertial coordinates for each frame
    p_G_I : array-like (N, 3)
        Target positions in inertial coordinates for each frame
    sun_az_I : array-like (N,)
        Sun azimuth angles (degrees) per frame - frame 0 used for static sun
    sun_el_I : array-like (N,)
        Sun elevation angles (degrees) per frame - frame 0 used for static sun
    r_CG_arr : array-like (N, 3)
        Camera position in G frame (body frame) per frame
    q_IG_arr : array-like (N, 4)
        Quaternion [w, x, y, z] from Inertial to Body (G) frame per frame
    every_n_frames : int
        Generate plot every N frames (1 = every frame)
    max_frames : int or None
        Maximum number of frames to process
    """
    n_frames = min(
        len(p_C_I),
        len(p_G_I),
        len(sun_az_I),
        len(sun_el_I),
        len(r_CG_arr),
        len(q_IG_arr),
    )

    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    if n_frames <= 0:
        raise ValueError("No frames available to plot scene figures.")
    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be >= 1.")

    # Compute STATIC sun direction in inertial frame from frame 0
    az0_rad = np.radians(sun_az_I[0])
    el0_rad = np.radians(sun_el_I[0])
    sun_dir_I = np.array([
        np.cos(el0_rad) * np.cos(az0_rad),
        np.cos(el0_rad) * np.sin(az0_rad),
        np.sin(el0_rad)
    ])

    scene_dir = os.path.join(output_dir, "ScenePlots")
    os.makedirs(scene_dir, exist_ok=True)

    print(f"\n=== Generating scene plots to {scene_dir} ===")
    print(f"    Earth: at origin (0,0,0)")
    print(f"    Sun: STATIC in inertial frame (from frame 0: az={sun_az_I[0]:.1f}° el={sun_el_I[0]:.1f}°)")
    print(f"    Target: tumbling via q_IG")
    print(f"    Lighting: sun transformed to body frame, compared with camera direction")

    for i in range(0, n_frames, every_n_frames):
        # Get rotation from Inertial to Body frame at this timestep
        qw, qx, qy, qz = q_IG_arr[i]
        rot_IG = Rotation.from_quat([qx, qy, qz, qw])

        # Transform static sun direction from Inertial to Body frame
        sun_dir_body = rot_IG.apply(sun_dir_I)

        # Camera direction toward target in body frame: -r_CG / |r_CG|
        r_CG_i = r_CG_arr[i]
        cam_dir_body = -r_CG_i / (np.linalg.norm(r_CG_i) + 1e-9)

        # Compute sun-camera angle in body frame (accurate lighting metric for tumbling target)
        sun_cam_angle_body = np.degrees(np.arccos(np.clip(np.dot(sun_dir_body, cam_dir_body), -1, 1)))

        plot_scene_frame(
            frame_idx=i,
            camera_loc=p_C_I[i],
            target_loc=p_G_I[i],
            sun_dir_I=sun_dir_I,  # Static sun in inertial frame for visualization
            camera_trajectory=p_C_I,
            target_trajectory=p_G_I,
            output_dir=scene_dir,
            sun_az_deg=sun_az_I[0],  # Frame 0 values (static)
            sun_el_deg=sun_el_I[0],
            sun_cam_angle_G=sun_cam_angle_body  # Accurate angle accounting for tumbling
        )

        if i % 50 == 0:
            print(f"  Generated scene plot for frame {i}/{n_frames}")

    print(f"  Scene plots saved to: {scene_dir}")
    return scene_dir
