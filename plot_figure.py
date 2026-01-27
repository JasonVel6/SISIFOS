import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
    i,             # MC index (int)
    s_A_I,         # (num_mc, nbSteps, 6) Target/Frame A state in I (pos[0:3], vel[3:6])
    s_c_I,         # (num_mc, num_agents, nbSteps, 6) Agent states in I
    r_CG_G,        # (num_mc, num_agents, nbSteps, 3) Relative positions in G (what SLAM uses)
    R_IG_all=None, # (num_mc, nbSteps, 3,3) for projecting variants into I
    out_dir=None,
    rotMode_Gframe=None,
    show=True,
    save=True,
    variant_rCGG=None,   # dict like {"_nmc": (num_agents,T,3), "_cro": (num_agents,T,3)} to overlay in G-frame
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

    num_agents = s_c_I.shape[1]
    colors = cm.tab10(np.linspace(0, 1, max(3, num_agents)))

    # Extract this trial’s trajectories
    rA = s_A_I[i, :, 0:3]  # (nbSteps,3)
    R_IG_i = None if R_IG_all is None else R_IG_all[i]  # (T,3,3) or None

    # Build relative (A-centered inertial) and absolute lists
    relI_list = []
    abs_listX, abs_listY, abs_listZ = [rA[:,0]], [rA[:,1]], [rA[:,2]]

    for agent_idx in range(num_agents):
        rS = s_c_I[i, agent_idx, :, 0:3]
        relI_list.append(rS - rA)
        abs_listX.append(rS[:,0]); abs_listY.append(rS[:,1]); abs_listZ.append(rS[:,2])

    # --- Figure with 3 subplots ---
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.22)

    ax_relI = fig.add_subplot(gs[0, 0], projection="3d")
    ax_inr  = fig.add_subplot(gs[0, 1], projection="3d")
    ax_G    = fig.add_subplot(gs[0, 2], projection="3d")

    # -------- (1) A-centered Inertial --------
    ax_relI.set_title(f"{_nice_title_for_mode(i, rotMode_Gframe)} — A-Centered Relative Paths")
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
            label_base = f"traj{i}_{'nmc' if suf.endswith('nmc') else 'cro' if suf.endswith('cro') else suf.strip('_')}"
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
    ax_inr.set_title(f"{_nice_title_for_mode(i, rotMode_Gframe)} — Inertial Trajectories (I)")
    ax_inr.set_xlabel("x_I [m]"); ax_inr.set_ylabel("y_I [m]"); ax_inr.set_zlabel("z_I [m]")
    ax_inr.grid(True, alpha=0.3)

    # A path
    ax_inr.plot(rA[:,0], rA[:,1], rA[:,2], lw=2.0, color="k", label="A (in I)")
    ax_inr.scatter(rA[0,0], rA[0,1], rA[0,2], s=20, color="k", marker="o")
    ax_inr.scatter(rA[-1,0], rA[-1,1], rA[-1,2], s=30, color="k", marker="x")

    # Agents
    for agent_idx in range(num_agents):
        rS = s_c_I[i, agent_idx, :, 0:3]
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
            for agent_idx in range(min(r_var.shape[0], s_c_I.shape[1])):
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
        rG = r_CG_G[i, agent_idx, :, :]  # (T,3)
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

            label_base = f"traj{i}_{'nmc' if suf.endswith('nmc') else 'cro' if suf.endswith('cro') else suf.strip('_')}"
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
    

    fig.suptitle(f"MC {i:03d} — Coordinated Views", y=0.98)

    if save:
        out_png = os.path.join(out_dir, f"MC_{i:03d}_traj.png")
        fig.savefig(out_png, dpi=170)
        print(f"[PLOTS] Saved: {out_png}")

    if show:
        plt.show(block=True)
    return fig
