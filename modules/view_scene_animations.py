#!/usr/bin/env python3
"""
Interactive scene/pose viewer for a single agent output folder.

Usage:
    python modules/view_scene_animations.py /path/to/.../Agent_0
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import matplotlib

# Prefer an interactive backend if the environment default is non-interactive.
if "agg" in matplotlib.get_backend().lower():
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
from scipy.spatial.transform import Rotation

NON_INTERACTIVE_BACKENDS = {
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
}


def _set_equal_3d(ax, X, Y, Z, margin=0.05):
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


@dataclass
class AgentTraj:
    timestamps: np.ndarray
    p_g_i: np.ndarray
    q_i_g: np.ndarray
    p_c_i: np.ndarray
    q_i_c: np.ndarray
    sun_az: np.ndarray
    sun_el: np.ndarray


def _load_camera_traj(agent_dir: str) -> AgentTraj:
    csv_path = os.path.join(agent_dir, "camera_traj.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing camera trajectory file: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding=None)
    if getattr(data, "shape", ()) == ():
        data = np.array([data], dtype=data.dtype)

    def cols(*names: str) -> np.ndarray:
        return np.column_stack([data[n] for n in names]).astype(float)

    return AgentTraj(
        timestamps=np.asarray(data["timestamp"], dtype=float),
        p_g_i=cols("p_G_I_x", "p_G_I_y", "p_G_I_z"),
        q_i_g=cols("q_I_G_w", "q_I_G_x", "q_I_G_y", "q_I_G_z"),
        p_c_i=cols("p_C_I_x", "p_C_I_y", "p_C_I_z"),
        q_i_c=cols("q_I_C_w", "q_I_C_x", "q_I_C_y", "q_I_C_z"),
        sun_az=np.asarray(data["sun_az"], dtype=float),
        sun_el=np.asarray(data["sun_el"], dtype=float),
    )


def _rot_from_wxyz(q_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


class SceneAnimationViewer:
    def __init__(self, agent_dir: str):
        self.agent_dir = os.path.abspath(agent_dir)
        self.traj = _load_camera_traj(self.agent_dir)
        self.n_frames = len(self.traj.timestamps)
        if self.n_frames == 0:
            raise ValueError("No frames found in camera_traj.csv")

        az0 = np.radians(float(self.traj.sun_az[0]))
        el0 = np.radians(float(self.traj.sun_el[0]))
        self.sun_dir_i = np.array(
            [np.cos(el0) * np.cos(az0), np.cos(el0) * np.sin(az0), np.sin(el0)],
            dtype=float,
        )
        self.sun_dir_i /= np.linalg.norm(self.sun_dir_i) + 1e-9

        self.p_cg_i_all = self.traj.p_c_i - self.traj.p_g_i
        self.r_cg_g_all = np.zeros_like(self.p_cg_i_all)
        for i in range(self.n_frames):
            self.r_cg_g_all[i] = _rot_from_wxyz(self.traj.q_i_g[i]).apply(self.p_cg_i_all[i])

        max_pcg_dist = float(np.max(np.linalg.norm(self.p_cg_i_all, axis=1)))
        self.pose_axis_length = max(0.15 * max_pcg_dist, 1.0)
        self.pose_plot_lim = max(max_pcg_dist * 0.85 + self.pose_axis_length * 0.8, self.pose_axis_length * 2.0)

        self.current_idx = 0
        self._ignore_slider_callback = False

        self._build_figures()
        self._update_all(0)

    def _build_figures(self):
        self.scene_fig = plt.figure(figsize=(17, 7))
        self.scene_fig.subplots_adjust(left=0.04, right=0.985, top=0.90, bottom=0.14, wspace=0.32)

        self.ax_world = self.scene_fig.add_subplot(1, 4, 1, projection="3d")
        self.ax_target = self.scene_fig.add_subplot(1, 4, 2, projection="3d")
        self.ax_top = self.scene_fig.add_subplot(1, 4, 3)
        self.ax_align = self.scene_fig.add_subplot(1, 4, 4)

        slider_ax = self.scene_fig.add_axes([0.12, 0.06, 0.58, 0.035])
        self.slider = Slider(
            ax=slider_ax,
            label="Frame",
            valmin=0,
            valmax=max(self.n_frames - 1, 0),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_changed)

        self.btn_prev10 = Button(self.scene_fig.add_axes([0.73, 0.055, 0.05, 0.045]), "-10")
        self.btn_prev = Button(self.scene_fig.add_axes([0.785, 0.055, 0.04, 0.045]), "-1")
        self.btn_next = Button(self.scene_fig.add_axes([0.83, 0.055, 0.04, 0.045]), "+1")
        self.btn_next10 = Button(self.scene_fig.add_axes([0.875, 0.055, 0.05, 0.045]), "+10")
        self.btn_prev10.on_clicked(lambda _evt: self.step(-10))
        self.btn_prev.on_clicked(lambda _evt: self.step(-1))
        self.btn_next.on_clicked(lambda _evt: self.step(1))
        self.btn_next10.on_clicked(lambda _evt: self.step(10))

        self.pose_fig = plt.figure(figsize=(9, 8))
        self.pose_fig.subplots_adjust(top=0.90)
        self.ax_pose = self.pose_fig.add_subplot(111, projection="3d")

        for fig in (self.scene_fig, self.pose_fig):
            fig.canvas.mpl_connect("key_press_event", self._on_key_press)
            fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _on_slider_changed(self, value):
        if self._ignore_slider_callback:
            return
        self._update_all(int(round(value)))

    def _on_key_press(self, event):
        if event.key in ("right", "d"):
            self.step(1)
        elif event.key in ("left", "a"):
            self.step(-1)
        elif event.key in ("up", "pageup"):
            self.step(10)
        elif event.key in ("down", "pagedown"):
            self.step(-10)
        elif event.key == "home":
            self._set_frame(0)
        elif event.key == "end":
            self._set_frame(self.n_frames - 1)

    def _on_scroll(self, event):
        if event.button == "up":
            self.step(1)
        elif event.button == "down":
            self.step(-1)

    def _set_frame(self, idx: int):
        idx = max(0, min(self.n_frames - 1, int(idx)))
        if idx == self.current_idx:
            return
        self._ignore_slider_callback = True
        self.slider.set_val(idx)
        self._ignore_slider_callback = False
        self._update_all(idx)

    def step(self, delta: int):
        self._set_frame(self.current_idx + delta)

    def _update_all(self, idx: int):
        self.current_idx = idx
        self._draw_scene_frame(idx)
        self._draw_relative_pose(idx)
        self.scene_fig.canvas.draw_idle()
        self.pose_fig.canvas.draw_idle()

    def _draw_scene_frame(self, idx: int):
        p_c_i = self.traj.p_c_i[idx]
        p_g_i = self.traj.p_g_i[idx]
        r_cg_g = self.r_cg_g_all[idx]

        # Preserve user camera angle when stepping frames.
        world_view = (self.ax_world.elev, self.ax_world.azim)
        target_view = (self.ax_target.elev, self.ax_target.azim)

        self.ax_world.cla()
        self.ax_target.cla()
        self.ax_top.cla()
        self.ax_align.cla()

        cam_to_target = p_g_i - p_c_i
        cam_to_target_dist = np.linalg.norm(cam_to_target)
        look_at_dir = cam_to_target / (cam_to_target_dist + 1e-9)
        earth_dir_from_target = -p_g_i / (np.linalg.norm(p_g_i) + 1e-9)
        earth_lookat_angle = np.degrees(np.arccos(np.clip(np.dot(earth_dir_from_target, look_at_dir), -1, 1)))

        rot_ig = _rot_from_wxyz(self.traj.q_i_g[idx])
        sun_dir_body = rot_ig.apply(self.sun_dir_i)
        cam_dir_body = -r_cg_g / (np.linalg.norm(r_cg_g) + 1e-9)
        sun_camera_angle = np.degrees(np.arccos(np.clip(np.dot(sun_dir_body, cam_dir_body), -1, 1)))

        if sun_camera_angle < 80:
            target_color = "gold"
            lit_status = "FRONT-LIT"
        elif sun_camera_angle < 100:
            target_color = "orange"
            lit_status = f"TRANSITION ({sun_camera_angle:.1f}°)"
        else:
            target_color = "gray"
            lit_status = "BACK-LIT"

        # World frame
        ax1 = self.ax_world
        ax1.scatter([0], [0], [0], c="green", s=200, marker="o", label="Earth", zorder=5)
        ax1.scatter([p_g_i[0]], [p_g_i[1]], [p_g_i[2]], c=target_color, s=100, marker="*", label="Target", zorder=5)
        ax1.scatter([p_c_i[0]], [p_c_i[1]], [p_c_i[2]], c="blue", s=80, marker="^", label="Camera", zorder=5)

        arrow_scale = cam_to_target_dist * 0.8
        look_arrow = look_at_dir * arrow_scale
        sun_arrow = self.sun_dir_i * arrow_scale
        ax1.quiver(*p_c_i, *look_arrow, color="blue", arrow_length_ratio=0.1, lw=2, label="Look-at")
        ax1.quiver(0, 0, 0, *sun_arrow, color="orange", arrow_length_ratio=0.1, lw=3, label="Sun dir")

        start_idx = max(0, idx - 50)
        end_idx = min(self.n_frames, idx + 50)
        traj_t = self.traj.p_g_i[start_idx:end_idx]
        if len(traj_t) > 1:
            ax1.plot(traj_t[:, 0], traj_t[:, 1], traj_t[:, 2], "r-", alpha=0.4, lw=1)

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(
            f"World Frame (Earth at origin)\n{lit_status} | Sun az={self.traj.sun_az[0]:.2f}° el={self.traj.sun_el[0]:.2f}°"
        )
        ax1.legend(loc="upper left", fontsize=6)
        world_X = np.array([0.0, p_g_i[0], p_c_i[0]])
        world_Y = np.array([0.0, p_g_i[1], p_c_i[1]])
        world_Z = np.array([0.0, p_g_i[2], p_c_i[2]])
        _set_equal_3d(ax1, world_X, world_Y, world_Z, margin=0.15)
        ax1.view_init(*world_view)

        # Target-centered view
        ax2 = self.ax_target
        cam_rel = p_c_i - p_g_i
        ax2.scatter([0], [0], [0], c=target_color, s=150, marker="*", label="Target", zorder=5)
        ax2.scatter([cam_rel[0]], [cam_rel[1]], [cam_rel[2]], c="blue", s=100, marker="^", label="Camera", zorder=5)
        look_arrow_rel = -cam_rel / (np.linalg.norm(cam_rel) + 1e-9) * cam_to_target_dist * 0.5
        arrow_scale2 = cam_to_target_dist * 0.6
        sun_arrow2 = self.sun_dir_i * arrow_scale2
        earth_arrow = earth_dir_from_target * arrow_scale2
        ax2.quiver(*cam_rel, *look_arrow_rel, color="blue", arrow_length_ratio=0.15, lw=2, label="Look-at")
        ax2.quiver(0, 0, 0, *sun_arrow2, color="orange", arrow_length_ratio=0.15, lw=3, label="Sun dir")
        ax2.quiver(0, 0, 0, *earth_arrow, color="green", arrow_length_ratio=0.15, lw=3, label="Earth dir")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.set_title(f"Frame {idx} | t={self.traj.timestamps[idx]:.3f}s | Range: {cam_to_target_dist:.2f}m")
        ax2.legend(loc="upper left", fontsize=6)
        max_range = max(cam_to_target_dist * 1.5, 1.0)
        ax2.set_xlim([-max_range, max_range])
        ax2.set_ylim([-max_range, max_range])
        ax2.set_zlim([-max_range, max_range])
        ax2.view_init(*target_view)

        # Top-down view
        ax3 = self.ax_top
        r_indicator = max_range * 0.9
        sun_2d = self.sun_dir_i[:2]
        sun_2d_norm = np.linalg.norm(sun_2d)
        if sun_2d_norm > 0.1:
            sun_2d_unit = sun_2d / sun_2d_norm
            sun_angle = np.arctan2(sun_2d_unit[1], sun_2d_unit[0])
            theta_lit = np.linspace(sun_angle - np.pi / 2, sun_angle + np.pi / 2, 50)
            x_lit = r_indicator * np.cos(theta_lit)
            y_lit = r_indicator * np.sin(theta_lit)
            ax3.fill(np.append(x_lit, 0), np.append(y_lit, 0), color="yellow", alpha=0.15, label="Lit side")

        ax3.scatter([0], [0], c=target_color, s=150, marker="*", label="Target", zorder=5)
        ax3.scatter([cam_rel[0]], [cam_rel[1]], c="blue", s=100, marker="^", label="Camera", zorder=5)
        ax3.arrow(cam_rel[0], cam_rel[1], look_arrow_rel[0], look_arrow_rel[1], head_width=2, head_length=1.5, fc="blue", ec="blue", lw=1.5)
        ax3.arrow(0, 0, sun_arrow2[0], sun_arrow2[1], head_width=3, head_length=2, fc="orange", ec="orange", lw=2, label="Sun")
        ax3.arrow(0, 0, earth_arrow[0], earth_arrow[1], head_width=3, head_length=2, fc="green", ec="green", lw=2, label="Earth")
        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_title(f"Top-Down (X-Y)\nSun-Cam angle: {sun_camera_angle:.1f}°")
        ax3.set_aspect("equal")
        ax3.legend(loc="upper left", fontsize=6)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([-max_range, max_range])
        ax3.set_ylim([-max_range, max_range])

        # Alignment panel
        ax4 = self.ax_align
        ax4.set_xlim(-1.5, 1.5)
        ax4.set_ylim(-0.5, 0.5)
        ax4.set_aspect("equal")
        ax4.axis("off")
        positions = {"Sun": -1.2, "Camera": -0.4, "Target": 0.4, "Earth": 1.2}
        colors = {"Sun": "orange", "Camera": "blue", "Target": target_color, "Earth": "green"}
        markers = {"Sun": "o", "Camera": "^", "Target": "*", "Earth": "o"}
        sizes = {"Sun": 200, "Camera": 150, "Target": 200, "Earth": 200}
        for name, x in positions.items():
            ax4.scatter([x], [0], c=colors[name], s=sizes[name], marker=markers[name], zorder=5)
            ax4.text(x, -0.25, name, ha="center", fontsize=9)
        ax4.annotate("", xy=(-0.5, 0), xytext=(-1.0, 0), arrowprops=dict(arrowstyle="->", color="orange", lw=2))
        ax4.annotate("", xy=(0.3, 0), xytext=(-0.3, 0), arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax4.annotate("", xy=(1.0, 0), xytext=(0.5, 0), arrowprops=dict(arrowstyle="->", color="gray", lw=2, ls="--"))
        ax4.set_title(f"{lit_status}\nSun-Cam: {sun_camera_angle:.1f}° | Earth-LookAt: {earth_lookat_angle:.1f}°")

        self.scene_fig.suptitle(
            f"{os.path.basename(self.agent_dir)}  |  frame {idx}/{self.n_frames - 1}  |  "
            "Controls: mouse rotate/zoom, scroll=step, ←/→, PgUp/PgDn",
            fontsize=11,
        )

    def _draw_relative_pose(self, idx: int):
        pose_view = (self.ax_pose.elev, self.ax_pose.azim)
        ax = self.ax_pose
        ax.cla()

        p_cg_i = self.p_cg_i_all[idx]
        q_ig_wxyz = self.traj.q_i_g[idx]
        q_ic_wxyz = self.traj.q_i_c[idx]

        ax.set_xlim(-self.pose_plot_lim, self.pose_plot_lim)
        ax.set_ylim(-self.pose_plot_lim, self.pose_plot_lim)
        ax.set_zlim(-self.pose_plot_lim, self.pose_plot_lim)
        ax.set_xlabel("I_x")
        ax.set_ylabel("I_y")
        ax.set_zlabel("I_z")
        ax.set_title("Relative Pose in I: q_I_G and q_I_C")
        ax.set_box_aspect((1, 1, 1))
        ax.grid(True, alpha=0.2)

        ax.plot([0.0], [0.0], [0.0], "ko", markersize=5)
        ax.plot([p_cg_i[0]], [p_cg_i[1]], [p_cg_i[2]], "mo", markersize=5)
        ax.plot([0.0, p_cg_i[0]], [0.0, p_cg_i[1]], [0.0, p_cg_i[2]], "k--", linewidth=1.4)

        r_ig = _rot_from_wxyz(q_ig_wxyz)
        r_ic = _rot_from_wxyz(q_ic_wxyz)
        g_basis = r_ig.apply(np.eye(3)) * self.pose_axis_length
        c_basis = r_ic.apply(np.eye(3)) * self.pose_axis_length

        for vec, color in zip(g_basis, ("r", "g", "b")):
            ax.quiver(0.0, 0.0, 0.0, vec[0], vec[1], vec[2], color=color, linewidth=2.2, arrow_length_ratio=0.15)
        for vec, color in zip(c_basis, ("#ff7f0e", "#2ca02c", "#17becf")):
            ax.quiver(
                p_cg_i[0], p_cg_i[1], p_cg_i[2],
                vec[0], vec[1], vec[2],
                color=color, linewidth=2.0, arrow_length_ratio=0.15, alpha=0.9
            )

        handles = [
            Line2D([0], [0], color="r", lw=2.2, label="G x-axis"),
            Line2D([0], [0], color="g", lw=2.2, label="G y-axis"),
            Line2D([0], [0], color="b", lw=2.2, label="G z-axis"),
            Line2D([0], [0], color="#ff7f0e", lw=2.0, label="C x-axis"),
            Line2D([0], [0], color="#2ca02c", lw=2.0, label="C y-axis"),
            Line2D([0], [0], color="#17becf", lw=2.0, label="C z-axis"),
            Line2D([0], [0], marker="o", color="k", lw=0, markersize=6, label="G center"),
            Line2D([0], [0], marker="o", color="m", lw=0, markersize=6, label="C center"),
            Line2D([0], [0], color="k", lw=1.4, linestyle="--", label="G->C"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.text2D(
            0.02,
            0.95,
            f"frame={idx}   t={self.traj.timestamps[idx]:.3f}s   |p_CG_I|={np.linalg.norm(p_cg_i):.3f}",
            transform=ax.transAxes,
        )
        ax.view_init(*pose_view)

        self.pose_fig.suptitle(
            f"{os.path.basename(self.agent_dir)}  |  Relative Pose Viewer  |  frame {idx}/{self.n_frames - 1}",
            fontsize=11,
        )

    def show(self):
        plt.show()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python modules/view_scene_animations.py /path/to/Agent_X", file=sys.stderr)
        return 2

    agent_dir = argv[1]
    if not os.path.isdir(agent_dir):
        print(f"Agent folder not found: {agent_dir}", file=sys.stderr)
        return 2

    backend = str(matplotlib.get_backend())
    if backend.lower() in NON_INTERACTIVE_BACKENDS:
        print(
            f"Current matplotlib backend is non-interactive ({backend}). "
            "Set an interactive backend (e.g. TkAgg/QtAgg) to use the viewer.",
            file=sys.stderr,
        )
        return 1

    viewer = SceneAnimationViewer(agent_dir)
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
